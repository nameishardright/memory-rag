import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from cryptography.exceptions import InvalidTag

from mr_private import PrivateRAG, load_private_rag, make_chunks, save_store, load_store
from mr_private_eval import validate_labels
from mr_private_facts import compact_search
from mr_private_index import PrivateBM25
from mr_private_policy import classify
from mr_private_routing import route_query
from mr_private_semantic import PrivateVector
from mr_index import rrf_files


def document(identifier, text, scope='personal', **metadata):
    return {'id': identifier, 'file': identifier + '.txt', 'scope': scope, 'text': text,
            'text_sha256': hashlib.sha256(text.encode()).hexdigest(), **metadata}


class TestEncoder:
    """Only checks plumbing. Actual ranking is tested separately with real E5."""
    signature = 'synthetic-encoder-v1'

    def __init__(self):
        self.calls = 0

    def windows(self, chunk):
        text = chunk['text']
        return [{'key': hashlib.sha256((chunk['md5'] + text).encode()).hexdigest(),
                 'start': 0, 'end': len(text), 'text': text}]

    def encode(self, texts, query=False):
        import numpy as np
        self.calls += len(texts)
        vectors = []
        for text in texts:
            money = any(t in text.lower() for t in ('earn', 'compensation', '年薪'))
            vectors.append([1.0, 0.0] if money else [0.0, 1.0])
        return np.asarray(vectors, dtype='<f4')


class PrivateRetrievalTests(unittest.TestCase):
    def test_encrypted_lexical_cache_restores_exact_ranking(self):
        docs = [document('a', '# Family\nPassport address 42'), document('b', 'passport trip passport')]
        chunks = make_chunks(docs)
        original = PrivateBM25(chunks)
        restored = PrivateBM25.from_state(chunks, original.to_state())
        for query in ('passport', 'address', '42', 'absent'):
            self.assertEqual(original.search(query), restored.search(query))
        state = original.to_state(); state['lengths'] = ''
        with self.assertRaises(ValueError):
            PrivateBM25.from_state(chunks, state)

    def test_scope_routing_is_bounded_and_explicit_scope_wins(self):
        for query, scope in [('我的年薪是多少', 'personal'), ('my earnings', 'personal'),
                             ('GitHub Java project', 'technical'), ('stock share price', 'stock'),
                             ('historical backup', 'history'), ('my passport', 'personal')]:
            self.assertEqual(route_query(query)['scope'], scope)
        docs = [document('p', 'Java passport'), document('t', 'Java scheduler', 'technical')]
        rag = PrivateRAG({'documents': docs, 'chunks': make_chunks(docs)})
        self.assertEqual(rag.search('Java', scope='auto')['scope'], 'technical')
        self.assertEqual([h['document_id'] for h in rag.search('Java', scope='personal')['hits']], ['p'])

    def test_complete_evidence_contains_separate_contract_dates(self):
        docs = [document('contract', '# Start\nContract starts 2024-02-01.\n\n# End\nContract ends 2025-08-31.',
                         warnings=['synthetic_ocr_warning'], method='image_ocr')]
        rag = PrivateRAG({'documents': docs, 'chunks': make_chunks(docs)})
        result = compact_search(rag, 'contract start end', preview_chars=30)
        hit = result['hits'][0]
        self.assertLessEqual(len(hit['excerpt']), 30)
        for value in ('2024-02-01', '2025-08-31'):
            self.assertIn(value, hit['evidence']['context'])
        self.assertEqual(hit['evidence']['warnings'], ['synthetic_ocr_warning'])
        self.assertTrue(hit['evidence']['context_complete'])
        self.assertFalse(rag.evidence('contract', expected_md5='stale')['ok'])

    def test_missing_year_and_current_status_are_not_certified_answers(self):
        docs = [document('tax', 'Tax balance for 2024: CAD 480.')]
        rag = PrivateRAG({'documents': docs, 'chunks': make_chunks(docs)})
        result = compact_search(rag, 'current tax balance for 2026')
        self.assertEqual(result['answerability'], 'unverified')
        reasons = result['hits'][0]['evidence']['review_reasons']
        self.assertIn('requested_year_not_in_evidence', reasons)
        self.assertIn('current_status_requires_confirmation', reasons)
        self.assertEqual(rag.search('AbsentXY912')['answerability'], 'no_evidence')

    def test_wrong_person_and_passport_appointment_require_review(self):
        docs = [document('passport', '# Passport\nHolder: Jordan Sample.\nDate of birth: 1992-03-04.\nExpiry: 2031-06-30.')]
        rag = PrivateRAG({'documents': docs, 'chunks': make_chunks(docs)})
        result = rag.evidence('passport', "What is my mother's date of birth?")
        self.assertIn('requested_person_role_not_in_evidence', result['review_reasons'])
        appointment = rag.evidence('passport', 'What is my passport renewal appointment date?')
        self.assertIn('requested_appointment_not_in_evidence', appointment['review_reasons'])
        self.assertEqual(appointment['answerability'], 'unverified')

    def test_shared_file_fusion_retrieves_semantic_only_document(self):
        docs = [document('pay', 'Annual compensation CAD 92000.'), document('trip', 'Travel itinerary Toronto.')]
        encoder = TestEncoder()
        rag = PrivateRAG({'documents': docs, 'chunks': make_chunks(docs)}, mode='hybrid', encoder_factory=lambda: encoder)
        before = rag.search('How much do I earn?')
        self.assertEqual(before['mode'], 'bm25(fallback)')
        rag.prepare_semantic()
        after = rag.search('How much do I earn?')
        self.assertEqual(after['mode'], 'hybrid')
        self.assertEqual(after['hits'][0]['document_id'], 'pay')
        self.assertEqual(after['answerability'], 'unverified')
        rag.close()
        with self.assertRaises(InterruptedError):
            rag.search('earn')

    def test_literal_source_survives_many_double_voted_neighbors(self):
        query = 'The complete fictional invoice refers to exact source phrase ABCDE-9137.'
        docs = [document('target', query)] + [document('near-' + str(i),
                'A fictional invoice refers to a source phrase, with general payment guidance. '
                'Brochure number ' + str(i)) for i in range(60)]
        rag = PrivateRAG({'documents': docs, 'chunks': make_chunks(docs)}, mode='hybrid')
        chunks, index = rag._index('personal')
        bm = index.search(query, topk=50, group_by_document=True)
        self.assertEqual(chunks[bm[0][0]]['document_id'], 'target')
        dense = [(i, 0.9) for i, chunk in enumerate(chunks)
                 if chunk['document_id'] != 'target'][:50]
        old = rrf_files(chunks, [bm, dense], topk=5)
        self.assertNotIn('target', [chunks[i]['document_id'] for i, _, _ in old])
        class Neighbors:
            ready = True
            def search(self, *_args, **_kwargs):
                return dense
        with patch.object(rag, '_vector', return_value=Neighbors()):
            result = rag.search(query.upper())
        self.assertEqual(result['hits'][0]['document_id'], 'target')
        self.assertTrue(result['hits'][0]['literal_query_match'])
        self.assertEqual(result['answerability'], 'unverified')

    def test_vector_cache_encryption_incremental_reuse_and_lock(self):
        docs = [document('secret', 'Annual compensation SyntheticSecret-9137'), document('trip', 'Travel Toronto')]
        chunks = make_chunks(docs)
        encoder = TestEncoder()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'semantic.enc'
            loader = lambda path: load_store(path, 'synthetic-password')
            saver = lambda path, data: save_store(path, 'synthetic-password', data)
            vector = PrivateVector(chunks, encoder, cache, loader, saver)
            first = vector.prepare()
            self.assertEqual(first['encoded'], 2)
            self.assertNotIn(b'SyntheticSecret-9137', cache.read_bytes())
            self.assertNotIn(b'secret.txt', cache.read_bytes())
            restored = PrivateVector(chunks, encoder, cache, loader, saver)
            self.assertTrue(restored.ready)
            self.assertEqual(vector.search('earn'), restored.search('earn'))
            docs[1]['text'] += ' changed'
            updated = PrivateVector(make_chunks(docs), encoder, cache, loader, saver)
            self.assertFalse(updated.ready)
            changed = updated.prepare()
            self.assertEqual(changed['encoded'], 1)
            self.assertEqual(changed['reused'], 1)
            with self.assertRaises(InvalidTag):
                load_store(cache, 'wrong')
            array = updated.emb
            updated.close()
            self.assertFalse(array.any())
            self.assertEqual([p.suffix for p in Path(directory).iterdir()], ['.enc'])

    def test_model_runtime_failure_retains_lexical_search_and_reports_fallback(self):
        docs = [document('pay', 'Annual compensation CAD 92000.')]
        encoder = TestEncoder()
        rag = PrivateRAG({'documents': docs, 'chunks': make_chunks(docs)},
                         mode='hybrid', encoder_factory=lambda: encoder)
        rag.prepare_semantic()
        def unavailable(*args, **kwargs):
            raise RuntimeError('Synthetic model runtime failure')
        encoder.encode = unavailable
        result = rag.search('compensation')
        self.assertEqual(result['mode'], 'bm25(fallback)')
        self.assertEqual(result['hits'][0]['document_id'], 'pay')
        self.assertEqual(result['semantic_state'], 'unavailable:RuntimeError')
        rag.close()

    def test_cancelled_preparation_does_not_publish_an_index(self):
        docs = [document('a', 'Annual compensation')]
        cancelled = threading.Event(); cancelled.set()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'semantic.enc'
            vector = PrivateVector(make_chunks(docs), TestEncoder(), cache=cache,
                    saver=lambda p, d: save_store(p, 'password', d), cancelled=cancelled)
            with self.assertRaises(InterruptedError):
                vector.prepare()
            self.assertFalse(cache.exists())

    def test_labels_cannot_leak_between_related_events(self):
        q = {'id': 'a', 'query': 'salary', 'expected_documents': ['pay'], 'required_text': ['92000'],
             'case_group': 'same-event', 'split': 'regression'}
        labels = {'version': 1, 'questions': [q, {**q, 'id': 'b', 'split': 'holdout'}]}
        with self.assertRaises(ValueError):
            validate_labels(labels)
        self.assertEqual(classify('tests/fixtures/private_retrieval_holdout.json')['purpose'], 'evaluation')
        self.assertEqual(classify('tests/fixtures/private_ranking_holdout.json')['purpose'], 'evaluation')


if __name__ == '__main__':
    unittest.main()
