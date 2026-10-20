"""Reproduce known fictional BM25/hybrid regressions with the local E5 model.

This tool uses public fictional fixtures only; it never unlocks a private corpus.
Previously consumed fixtures are regressions, not fresh holdouts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1', MR_LOG='0')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from mr_private import PrivateRAG, make_chunks
    from mr_private_eval import evaluate
    from mr_private_semantic import LocalEncoder
    fixtures = [CODE / 'tests' / 'fixtures' / name for name in
                ('private_retrieval_holdout.json', 'private_ranking_holdout.json', 'private_natural_dev.json',
                 'private_natural_holdout.json')]
    frozen = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in fixtures}
    encoder = LocalEncoder()
    results = {}
    for path in fixtures:
        fixture = json.loads(path.read_text(encoding='utf-8'))
        if not fixture.get('synthetic'):
            raise ValueError('Only explicitly fictional fixtures are allowed')
        results[path.name] = {'evaluation_role': 'known_regression', 'fixture_sha256': frozen[path.name]}
        for mode in ('bm25', 'hybrid'):
            rag = PrivateRAG({'documents': fixture['documents'], 'chunks': make_chunks(fixture['documents'])},
                             mode=mode, encoder_factory=lambda: encoder)
            try:
                with patch('socket.socket.connect', side_effect=AssertionError('Unexpected network access')), \
                     patch('socket.create_connection', side_effect=AssertionError('Unexpected network access')):
                    if mode == 'hybrid':
                        for scope in sorted({d['scope'] for d in fixture['documents']}):
                            rag.prepare_semantic(scope)
                    report = evaluate(rag, fixture, repetitions=0)
                results[path.name][mode] = report['summary']
            finally:
                rag.close()
    # A long, named document requires the dense passage inside that document,
    # not a plausible neighbouring file or only the opening paragraph.
    long_body = ('# Background\nThis fictional record explains ordinary workplace procedures.\n' +
                 'Background prose without a compensation amount. ' * 45 + '\n\n') * 12
    target = {'id': 'named-long', 'file': 'fictional-long-offer.txt', 'scope': 'personal',
              'text': long_body + '# Compensation agreement\nAnnual compensation: CAD 81234.'}
    neighbour = {'id': 'named-neighbour', 'file': 'fictional-neighbour.txt', 'scope': 'personal',
                 'text': '# Yearly pay\nThe fixed yearly pay in this unrelated fictional example is CAD 91234.'}
    docs = [target, neighbour]
    labels = {'version': 1, 'questions': [
        {'id': 'named-long-source', 'query': 'In the record named "fictional-long-offer", how much is the fixed yearly pay?',
         'case_group': 'named-long', 'split': 'regression', 'scope': 'personal',
         'expected_documents': ['named-long'], 'required_text': ['Annual compensation', 'CAD 81234']},
        {'id': 'missing-named-source', 'query': '档案《fictional-source-not-present》里，年薪是多少？',
         'case_group': 'missing-named-source', 'split': 'regression', 'scope': 'personal', 'positive': False}]}
    rag = PrivateRAG({'documents': docs, 'chunks': make_chunks(docs)}, mode='hybrid', encoder_factory=lambda: encoder)
    try:
        with patch('socket.socket.connect', side_effect=AssertionError('Unexpected network access')), \
             patch('socket.create_connection', side_effect=AssertionError('Unexpected network access')):
            rag.prepare_semantic('personal')
            named = evaluate(rag, labels, repetitions=0)['summary']
    finally:
        rag.close()
    unchanged = all(hashlib.sha256(path.read_bytes()).hexdigest() == frozen[path.name] for path in fixtures)
    passed = unchanged and all(row['hybrid']['evidence_complete_at_k'] == row['hybrid']['positive_questions']
                               and row['hybrid']['modes'] == ['hybrid'] for row in results.values())
    passed = passed and named['evidence_complete_at_k'] == 1 and named['negative_related_candidates'] == 0
    report = {'passed': passed, 'offline_model_inference': True, 'fixtures_unchanged': unchanged,
              'datasets': results, 'named_source_long_evidence': named,
              'boundary': 'Fictional known regressions, not real private or final-answer accuracy.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'passed': passed, 'fictional_positives': sum(r['hybrid']['positive_questions'] for r in results.values()) + 1,
                      'named_long_evidence_complete': named['evidence_complete_at_k'], 'fixtures_unchanged': unchanged}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
