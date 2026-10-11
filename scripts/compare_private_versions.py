"""Pair a baseline checkout with this checkout on fictional fixtures.

Both versions answer the same frozen questions in isolated processes, in BM25
and in hybrid with the local E5 model. Fixtures must be marked synthetic; this
tool never unlocks a private corpus. A fixture whose questions are a holdout is
reserved before the first search and cannot be rerun as fresh.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
CODE_FILES = ('mr_private.py', 'mr_private_index.py', 'mr_private_facts.py', 'mr_private_semantic.py',
              'mr_private_routing.py', 'mr_private_policy.py', 'mr_private_eval.py', 'mr_index.py', 'mr_corpus.py')


def code_fingerprint(root):
    return {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
            for name in CODE_FILES if (Path(root) / name).is_file()}


def worker(code, fixture_path, mode, repetitions):
    sys.path.insert(0, str(Path(code).resolve()))
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', MR_LOG='0')
    from unittest.mock import patch
    from mr_private import PrivateRAG, make_chunks
    from mr_private_eval import evaluate
    fixture = json.loads(Path(fixture_path).read_text(encoding='utf-8'))
    if not fixture.get('synthetic'):
        raise ValueError('Only explicitly fictional fixtures are allowed')
    documents = fixture['documents']
    encoder_factory = None
    if mode == 'hybrid':
        from mr_private_semantic import LocalEncoder
        encoder = LocalEncoder()
        encoder_factory = lambda: encoder
    rag = PrivateRAG({'documents': documents, 'chunks': make_chunks(documents)}, mode=mode,
                     encoder_factory=encoder_factory)
    try:
        with patch('socket.socket.connect', side_effect=AssertionError('Unexpected network access')), \
             patch('socket.create_connection', side_effect=AssertionError('Unexpected network access')):
            started = time.perf_counter()
            if mode == 'hybrid':
                for scope in sorted({d['scope'] for d in documents}):
                    rag.prepare_semantic(scope)
            preparation = time.perf_counter() - started
            report = evaluate(rag, fixture, repetitions=repetitions)
    finally:
        rag.close()
    report['preparation_seconds'] = round(preparation, 3)
    return report


def run_worker(code, fixture_path, mode, repetitions):
    command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--worker',
               '--code', str(code), '--fixture', str(fixture_path), '--mode', mode,
               '--repetitions', str(repetitions)]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', cwd=str(code))
    if result.returncode:
        raise RuntimeError('Worker failed for ' + str(code) + ' (' + mode + '): ' + result.stderr[-2000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def compare(before, after):
    old = {row['id']: row for row in before['questions']}
    new = {row['id']: row for row in after['questions']}
    if set(old) != set(new) or before['summary']['labels_sha256'] != after['summary']['labels_sha256']:
        raise ValueError('Paired comparisons require identical questions and frozen labels')
    positives = sorted(key for key, row in old.items() if row['positive'])
    negatives = sorted(key for key, row in old.items() if not row['positive'])
    def changed(ok):
        return {'gains': [key for key in positives if not ok(old[key]) and ok(new[key])],
                'regressions': [key for key in positives if ok(old[key]) and not ok(new[key])]}
    return {'positive_questions': len(positives),
            'file': changed(lambda row: row['rank'] is not None),
            'top1': changed(lambda row: row['rank'] == 1),
            'evidence': changed(lambda row: row['evidence_complete']),
            'still_missing_file': [key for key in positives if new[key]['rank'] is None],
            'still_missing_evidence': [key for key in positives if not new[key]['evidence_complete']],
            'negatives_with_candidates': {'before': sum(bool(old[key]['candidates']) for key in negatives),
                                          'after': sum(bool(new[key]['candidates']) for key in negatives)}}


def headline(summary):
    return {key: summary[key] for key in ('positive_questions', 'file_hits_at_k', 'evidence_complete_at_k',
                                           'preview_complete_at_k', 'mrr', 'negative_questions',
                                           'negative_related_candidates', 'modes', 'warm_p95_seconds')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-code', type=Path)
    parser.add_argument('--fixture', type=Path, action='append', default=[])
    parser.add_argument('--regression-fixture', type=Path, action='append', default=[],
                        help='Previously consumed fixture, reported only as a known regression')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', choices=('bm25', 'hybrid'), action='append')
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--code', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.code, args.fixture[0], args.mode[0], args.repetitions), ensure_ascii=False))
        return 0
    if not args.baseline_code or not (args.fixture or args.regression_fixture) or not args.output:
        parser.error('--baseline-code, a fixture and --output are required')
    modes = args.mode or ['bm25', 'hybrid']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error('Refusing to overwrite an existing comparison report')
    baseline, candidate = args.baseline_code.resolve(), CODE
    frozen_code = {'baseline': code_fingerprint(baseline), 'candidate': code_fingerprint(candidate)}
    report = {'version': 1, 'boundary': 'Fictional fixtures only; not real private accuracy.',
              'baseline_code': str(baseline), 'code_sha256': frozen_code, 'datasets': {}}
    for path in [*args.fixture, *args.regression_fixture]:
        fixture = json.loads(path.read_text(encoding='utf-8'))
        if not fixture.get('synthetic'):
            raise ValueError('Only explicitly fictional fixtures are allowed')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        holdout = path in args.fixture and any(q.get('split') == 'holdout' for q in fixture['questions'])
        if holdout:
            # Exclusive creation: an interrupted run still counts as consumed.
            marker = args.output.parent / (path.stem + '.consumed.json')
            with marker.open('x', encoding='utf-8') as stream:
                json.dump({'fixture_sha256': digest, 'started_epoch': time.time()}, stream)
        dataset = {'fixture_sha256': digest, 'documents': len(fixture['documents']),
                   'evaluation_role': ('fresh_once_holdout' if holdout else
                                       'known_regression' if path in args.regression_fixture else 'development'),
                   'label_origin': fixture.get('label_origin'), 'modes': {}}
        for mode in modes:
            before = run_worker(baseline, path.resolve(), mode, args.repetitions)
            after = run_worker(candidate, path.resolve(), mode, args.repetitions)
            dataset['modes'][mode] = {'before': headline(before['summary']), 'after': headline(after['summary']),
                                      'changes': compare(before, after),
                                      'questions': {'before': before['questions'], 'after': after['questions']}}
            changes = dataset['modes'][mode]['changes']
            print(json.dumps({'fixture': path.name, 'mode': mode,
                              'before': {k: before['summary'][k] for k in ('file_hits_at_k', 'evidence_complete_at_k', 'mrr')},
                              'after': {k: after['summary'][k] for k in ('file_hits_at_k', 'evidence_complete_at_k', 'mrr')},
                              'positives': changes['positive_questions'],
                              'file_gains': len(changes['file']['gains']), 'file_regressions': len(changes['file']['regressions']),
                              'evidence_gains': len(changes['evidence']['gains']),
                              'evidence_regressions': len(changes['evidence']['regressions'])}, ensure_ascii=False), flush=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Fixture changed during the comparison')
        report['datasets'][path.name] = dataset
    if {'baseline': code_fingerprint(baseline), 'candidate': code_fingerprint(candidate)} != frozen_code:
        raise ValueError('Code changed during the comparison')
    regressions = sum(len(row['changes'][field]['regressions']) for data in report['datasets'].values()
                      for row in data['modes'].values() for field in ('file', 'evidence'))
    report['no_file_or_evidence_regressions'] = regressions == 0
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if regressions == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
