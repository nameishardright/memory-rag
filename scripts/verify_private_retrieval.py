"""Compare old/new retrieval on explicitly synthetic fixtures with real E5.

This script never unlocks the production corpus. Synthetic results cannot be
reported as the user's private-corpus accuracy.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tempfile

CODE = Path(__file__).resolve().parents[1]
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MR_LOG='0',
                  HF_HUB_DISABLE_TELEMETRY='1')


def reviewed_fixture(path):
    original = json.loads(path.read_text(encoding='utf-8'))
    questions = []
    for identifier, query, expected, facts, *source in original['positives']:
        questions.append({'id': identifier, 'query': query, 'expected_documents': [expected],
                          'required_text': facts, 'scope': source[0] if source else 'personal',
                          'case_group': expected, 'split': 'regression'})
    for identifier, query in original['negatives']:
        questions.append({'id': identifier, 'query': query, 'positive': False, 'scope': 'personal',
                          'case_group': 'negative-' + identifier, 'split': 'regression'})
    return {'version': 1, 'synthetic': True, 'documents': original['documents'], 'questions': questions}


def legacy(fixture, module_path):
    sys.path.insert(0, str(module_path))
    sys.path.append(str(CODE))
    from mr_private import PrivateRAG, make_chunks
    from mr_private_facts import compact_search
    rag = PrivateRAG({'documents': fixture['documents'], 'chunks': make_chunks(fixture['documents'])})
    rows = []
    for question in fixture['questions']:
        scope = question.get('scope', 'personal')
        if scope == 'auto':
            scope = 'personal'  # The deployed old UI defaults to personal.
        result = compact_search(rag, question['query'], topk=5, scope=scope, preview_chars=220)
        ui_result = compact_search(rag, question['query'], topk=5, scope=scope, preview_chars=700)
        expected = question.get('expected_documents', [])
        rank = next((i+1 for i,h in enumerate(result['hits']) if h['document_id'] in expected), None)
        preview = '\n'.join(h['excerpt'] for h in result['hits'] if h['document_id'] in expected)
        ui_preview = '\n'.join(h['excerpt'] for h in ui_result['hits'] if h['document_id'] in expected)
        full = '\n'.join(rag.get(h['document_id'])['text'] for h in result['hits'] if h['document_id'] in expected)
        facts = question.get('required_text', [])
        rows.append({'id': question['id'], 'positive': question.get('positive', True), 'rank': rank,
                     'preview_complete': bool(rank) and all(f.casefold() in preview.casefold() for f in facts),
                     'ui_preview_complete': bool(rank) and all(f.casefold() in ui_preview.casefold() for f in facts),
                     'full_document_complete': bool(rank) and all(f.casefold() in full.casefold() for f in facts)})
    positive = [r for r in rows if r['positive']]
    return {'summary': {'positive_questions': len(positive),
                        'file_hits_at_5': sum(bool(r['rank']) for r in positive),
                        'preview_complete_at_5': sum(r['preview_complete'] for r in positive),
                        'ui_preview_complete_at_5': sum(r['ui_preview_complete'] for r in positive),
                        'full_document_complete_at_5': sum(r['full_document_complete'] for r in positive)},
            'questions': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-fixture', type=Path, required=True)
    parser.add_argument('--old-app', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--holdout', type=Path)
    parser.add_argument('--regression-fixture', type=Path, action='append', default=[],
                        help='Previously consumed synthetic holdout, now used only as regression')
    parser.add_argument('--previous-results', type=Path,
                        help='Preserve hits and evidence from a prior quality run')
    parser.add_argument('--legacy-worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.legacy_worker:
        fixture = json.loads(args.legacy_worker.read_text(encoding='utf-8'))
        if not fixture.get('synthetic'):
            raise ValueError('Synthetic fixtures only')
        print(json.dumps(legacy(fixture, args.old_app), ensure_ascii=False))
        return 0
    sys.path.insert(0, str(CODE))
    from mr_private import PrivateRAG, index_cache_path, make_chunks, save_store, load_store
    from mr_private_eval import evaluate, labels_sha256
    from mr_private_semantic import LocalEncoder
    encoder = LocalEncoder()
    # A real load/inference must work without any socket connection. This is
    # separate from fake-encoder unit tests and never receives private text.
    from unittest.mock import patch
    import numpy as np
    with patch('socket.socket.connect', side_effect=AssertionError('Unexpected network access')), \
         patch('socket.create_connection', side_effect=AssertionError('Unexpected network access')):
        matrix = encoder.encode(['passage: Annual compensation CAD 92000.',
                                 'passage: Travel itinerary Toronto to Montreal.'])
        query_vector = encoder.encode(['How much do I earn?'], query=True)[0]
        long_body = ('中英文混合 source text with overlapping evidence windows. ' * 180).strip()
        windows = encoder.windows({'heading': 'Synthetic long source', 'text': long_body,
                                   'md5': hashlib.md5(long_body.encode()).hexdigest()})
        window_lengths = [len(encoder._get_tokenizer()(w['text'])['input_ids']) for w in windows]
    if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4):
        raise ValueError('Local encoder output is invalid')
    if not windows or windows[-1]['end'] != len(long_body) or max(window_lengths) > 512:
        raise ValueError('Long source windows lose evidence or exceed model capacity')
    runtime_check = {'offline_inference_passed': True, 'backend': encoder.backend,
                     'dimensions': matrix.shape[1], 'long_source_windows': len(windows),
                     'maximum_window_tokens': max(window_lengths), 'long_source_tail_covered': True,
                     'synthetic_salary_similarity': round(float(matrix[0] @ query_vector), 6),
                     'synthetic_travel_similarity': round(float(matrix[1] @ query_vector), 6)}
    fixtures = [('regression', reviewed_fixture(args.review_fixture))]
    for index, path in enumerate(args.regression_fixture):
        fixture = json.loads(path.read_text(encoding='utf-8'))
        if not fixture.get('synthetic'):
            raise ValueError('Synthetic fixtures only')
        fixtures.append(('previous_holdout_regression' + (str(index + 1) if index else ''), fixture))
    if args.holdout:
        holdout = json.loads(args.holdout.read_text(encoding='utf-8'))
        if not holdout.get('synthetic'):
            raise ValueError('Synthetic fixtures only')
        marker = args.output.parent / 'synthetic-holdout-consumed.json'
        marker.parent.mkdir(parents=True, exist_ok=True)
        with marker.open('x', encoding='utf-8') as stream:
            json.dump({'labels_sha256': labels_sha256(holdout), 'started_epoch': time.time()}, stream)
        fixtures.append(('holdout', holdout))
    report = {'boundary': 'Synthetic diagnostics, not real private accuracy.',
              'model_runtime_check': runtime_check, 'datasets': {}}
    previous = json.loads(args.previous_results.read_text(encoding='utf-8')) if args.previous_results else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, fixture in fixtures:
        frozen_path = args.output.parent / (name + '-frozen.json')
        if frozen_path.exists():
            if json.loads(frozen_path.read_text(encoding='utf-8')) != fixture:
                raise ValueError('Frozen fixture changed')
        else:
            frozen_path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding='utf-8')
        command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()),
                   '--review-fixture', str(args.review_fixture), '--old-app', str(args.old_app),
                   '--output', str(args.output), '--legacy-worker', str(frozen_path)]
        old = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=True)
        baseline = json.loads(old.stdout)
        docs = fixture['documents']
        payload = {'documents': docs, 'chunks': make_chunks(docs)}
        checkup = None
        if name == 'regression':
            from mr_private_checkup import run_checkup
            with tempfile.TemporaryDirectory(prefix='rag-synthetic-checkup-') as directory:
                store = Path(directory) / 'corpus.enc'
                copied_docs = [dict(d, text_sha256=hashlib.sha256(d['text'].encode()).hexdigest()) for d in docs]
                save_store(store, 'synthetic-test-password', {'version': 1, 'documents': copied_docs,
                    'chunks': make_chunks(copied_docs), 'updated_at': '2026-10-03',
                    'coverage': {'documents': len(docs), 'visible_files': len(docs), 'content_gaps': {}}})
                save_store(store.parent / 'coverage-review.enc', 'synthetic-test-password',
                           {'version': 1, 'gaps': [], 'not_processed': []})
                (store.parent / 'retrieval-settings.json').write_text(json.dumps({
                    'version': 1, 'mode': 'hybrid', 'encrypted_index_cache': True}), encoding='utf-8')
                checkup = run_checkup(store, 'synthetic-test-password', collect=False, fresh_latency=True)
                if not checkup['passed']:
                    raise ValueError('Actual-model encrypted checkup or native UI failed')
                if not load_store(index_cache_path(store, 'personal', 'semantic'), 'synthetic-test-password')['complete']:
                    raise ValueError('Actual-model encrypted semantic index is incomplete')
        lexical = PrivateRAG(payload)
        lexical_result = evaluate(lexical, fixture)
        rag = PrivateRAG(payload, mode='hybrid', encoder_factory=lambda: encoder)
        preparation = []
        started = time.perf_counter()
        for scope in dict.fromkeys(d['scope'] for d in docs if d.get('purpose') not in {'excluded','evaluation'}):
            preparation.append(rag.prepare_semantic(scope))
        preparation_seconds = time.perf_counter() - started
        hybrid = evaluate(rag, fixture, repetitions=2)
        new_rows = {q['id']: q for q in hybrid['questions']}
        lost_old_hits = [q['id'] for q in baseline['questions']
                         if q['positive'] and q['rank'] and not new_rows[q['id']]['rank']]
        lost_old_evidence = [q['id'] for q in baseline['questions']
                             if q['positive'] and q['ui_preview_complete'] and not new_rows[q['id']]['evidence_complete']]
        lost_bm_hits = [q['id'] for q in lexical_result['questions']
                        if q['positive'] and q['rank'] and not new_rows[q['id']]['rank']]
        lost_bm_evidence = [q['id'] for q in lexical_result['questions']
                            if q['positive'] and q['evidence_complete'] and not new_rows[q['id']]['evidence_complete']]
        prior_rows = next((row['new_hybrid']['questions'] for row in previous['datasets'].values()
                           if row['labels_sha256'] == labels_sha256(fixture)), []) if previous else []
        lost_prior_hits = [q['id'] for q in prior_rows
                           if q['positive'] and q['rank'] and not new_rows[q['id']]['rank']]
        lost_prior_evidence = [q['id'] for q in prior_rows
                               if q['positive'] and q['evidence_complete'] and not new_rows[q['id']]['evidence_complete']]
        acceptance = {'old_positive_hits_preserved': not lost_old_hits,
                      'old_ui_evidence_preserved': not lost_old_evidence,
                      'bm25_positive_hits_preserved': not lost_bm_hits,
                      'bm25_evidence_preserved': not lost_bm_evidence,
                      'prior_hybrid_hits_preserved': not lost_prior_hits,
                      'prior_hybrid_evidence_preserved': not lost_prior_evidence,
                      'actual_hybrid': hybrid['summary']['modes'] == ['hybrid'],
                      'lost_old_hits': lost_old_hits, 'lost_old_evidence': lost_old_evidence,
                      'lost_bm25_hits': lost_bm_hits, 'lost_bm25_evidence': lost_bm_evidence,
                      'lost_prior_hits': lost_prior_hits, 'lost_prior_evidence': lost_prior_evidence}
        acceptance['passed'] = all(acceptance[k] for k in ('old_positive_hits_preserved',
            'old_ui_evidence_preserved', 'bm25_positive_hits_preserved', 'bm25_evidence_preserved',
            'prior_hybrid_hits_preserved', 'prior_hybrid_evidence_preserved', 'actual_hybrid'))
        model_path = encoder.directory / ('onnx/model_qint8_avx512_vnni.onnx' if encoder.backend == 'onnx_int8' else 'model.safetensors')
        model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        report['datasets'][name] = {'labels_sha256': labels_sha256(fixture),
             'evaluation_role': 'fresh_once_holdout' if name == 'holdout' else 'known_regression',
             'baseline_old_app': baseline, 'new_bm25': lexical_result, 'new_hybrid': hybrid,
             'preparation': preparation, 'preparation_seconds': round(preparation_seconds, 4),
             'acceptance': acceptance,
             'model_weights_sha256': model_sha}
        if checkup:
            report['datasets'][name]['actual_model_encrypted_checkup'] = checkup
        lexical.close(); rag.close()
        print(json.dumps({'dataset': name, 'before': baseline['summary'],
                          'after_bm25': lexical_result['summary'], 'after_hybrid': hybrid['summary']}, ensure_ascii=False), flush=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    report['code_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.glob('mr_private*.py')}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if all(row['acceptance']['passed'] for row in report['datasets'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
