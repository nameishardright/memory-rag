"""Evaluate source retrieval separately from evidence coverage and answer accuracy.

Real questions, labels and per-question results use authenticated encrypted
files. Only aggregate counts and timings may be written as a JSON summary.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time


def labels_sha256(labels):
    return hashlib.sha256(json.dumps(labels, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':')).encode()).hexdigest()


def validate_labels(labels):
    if labels.get('version') != 1 or not labels.get('questions'):
        raise ValueError('A versioned, nonempty evaluation set is required')
    ids, groups = set(), defaultdict(set)
    for question in labels['questions']:
        if not question.get('id') or question['id'] in ids or not question.get('query', '').strip():
            raise ValueError('Evaluation ids and queries must be valid')
        ids.add(question['id'])
        split = question.get('split', 'regression')
        if split not in {'regression', 'holdout'} or not question.get('case_group'):
            raise ValueError('Evaluation splits require event groups')
        groups[question['case_group']].add(split)
        if question.get('positive', True) and (not question.get('expected_documents') or not question.get('required_text')):
            raise ValueError('Positive questions need source and evidence labels')
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError('Related events must not cross evaluation splits')


def evaluate(rag, labels, topk=5, repetitions=1):
    from mr_private_facts import compact_search
    validate_labels(labels)
    # Confirm labels against source bodies, never against the system's answer.
    for question in labels['questions']:
        if not question.get('positive', True):
            continue
        originals = [rag.get(identifier) for identifier in question['expected_documents']]
        if not all(doc.get('ok') for doc in originals):
            raise ValueError('An evaluation source reference is missing or stale')
        source_text = '\n'.join(doc['text'] for doc in originals).casefold()
        if not all(value.casefold() in source_text for value in question['required_text']):
            raise ValueError('An evidence label is absent from the original sources')
    rows, warm_timings = [], []
    for question in labels['questions']:
        started = time.perf_counter()
        result = compact_search(rag, question['query'], topk=topk,
                                scope=question.get('scope', 'auto'), preview_chars=220)
        cold_seconds = time.perf_counter() - started
        expected = set(question.get('expected_documents', []))
        ranks = [i + 1 for i, hit in enumerate(result['hits']) if hit['document_id'] in expected]
        context = '\n'.join(hit.get('evidence', {}).get('context', '')
                            for hit in result['hits'] if hit['document_id'] in expected)
        preview = '\n'.join(hit['excerpt'] for hit in result['hits'] if hit['document_id'] in expected)
        required = question.get('required_text', [])
        coverage = bool(ranks) and all(value.casefold() in context.casefold() for value in required)
        preview_coverage = bool(ranks) and all(value.casefold() in preview.casefold() for value in required)
        for _ in range(repetitions):
            started = time.perf_counter()
            compact_search(rag, question['query'], topk=topk, scope=question.get('scope', 'auto'))
            warm_timings.append(time.perf_counter() - started)
        rows.append({'id': question['id'], 'case_group': question['case_group'],
                     'split': question.get('split', 'regression'),
                     'positive': question.get('positive', True),
                     'rank': min(ranks) if ranks else None, 'evidence_complete': coverage,
                     'preview_complete': preview_coverage, 'candidates': len(result['hits']),
                     'answerability': result.get('answerability', 'unverified'),
                     'mode': result['mode'], 'scope': result['scope'],
                     'first_call_seconds': round(cold_seconds, 4),
                     'review_reasons': sorted({reason for hit in result['hits']
                          for reason in hit.get('evidence', {}).get('review_reasons', [])})})
    positives = [row for row in rows if row['positive']]
    negatives = [row for row in rows if not row['positive']]
    def split_summary(split):
        chosen = [r for r in positives if r['split'] == split]
        return {'positive_questions': len(chosen), 'file_hits': sum(r['rank'] is not None for r in chosen),
                'evidence_complete': sum(r['evidence_complete'] for r in chosen)}
    summary = {'labels_sha256': labels_sha256(labels), 'positive_questions': len(positives),
               'negative_questions': len(negatives), 'file_hits_at_k': sum(r['rank'] is not None for r in positives),
               'evidence_complete_at_k': sum(r['evidence_complete'] for r in positives),
               'preview_complete_at_k': sum(r['preview_complete'] for r in positives),
               'mrr': round(sum(1 / r['rank'] for r in positives if r['rank']) / max(1, len(positives)), 6),
               'negative_related_candidates': sum(bool(r['candidates']) for r in negatives),
               'negative_candidates_marked_unverified': sum(r['answerability'] in {'unverified', 'no_evidence'} for r in negatives),
               'regression': split_summary('regression'), 'holdout': split_summary('holdout'),
               'modes': sorted(set(r['mode'] for r in rows)),
               'warm_p95_seconds': round(sorted(warm_timings)[max(0, math.ceil(.95 * len(warm_timings)) - 1)], 4) if warm_timings else None,
               'answer_accuracy': None,
               'boundary': 'Retrieval and source-evidence evaluation. No answer-generation or refusal-accuracy claim.'}
    return {'version': 1, 'summary': summary, 'questions': rows}


def main(argv=None):
    import getpass
    from mr_private import default_store, load_private_rag, load_store, save_store, atomic_write
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=Path, default=default_store())
    parser.add_argument('--labels', type=Path, required=True, help='Encrypted, source-checked question set')
    parser.add_argument('--output', type=Path, required=True, help='Encrypted per-question report')
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--mode', choices=['bm25', 'hybrid'], default='hybrid')
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parent
    if any(path.resolve().is_relative_to(repo) for path in (args.store, args.labels, args.output)):
        parser.error('Private data and results must stay outside the checkout')
    password = getpass.getpass('Archive password: ')
    labels = load_store(args.labels, password)
    validate_labels(labels)
    # A consumed result holdout cannot silently be rerun and called fresh.
    if any(q.get('split') == 'holdout' for q in labels['questions']):
        marker = args.labels.with_name(args.labels.name + '.consumed.enc')
        if marker.exists():
            raise ValueError('Holdout already consumed; reuse it as regression explicitly')
        save_store(marker, password, {'version': 1, 'labels_sha256': labels_sha256(labels)})
    rag = load_private_rag(args.store, password, mode=args.mode)
    try:
        if args.prepare:
            rag.prepare_semantic('personal')
        report = evaluate(rag, labels, repetitions=3)
        save_store(args.output, password, report)
        if args.summary:
            atomic_write(args.summary, json.dumps(report['summary'], indent=2).encode())
        print(json.dumps(report['summary']))
        positives = report['summary']['positive_questions']
        return 0 if report['summary']['evidence_complete_at_k'] == positives else 1
    finally:
        rag.close()
        password = None


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        # OS exceptions and tracebacks can include personal source filenames.
        print('Private evaluation failed: ' + type(error).__name__)
        raise SystemExit(2)
