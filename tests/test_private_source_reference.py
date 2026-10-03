from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mr_private import PrivateRAG, make_chunks
from mr_private_routing import named_source
from mr_private_semantic import PrivateVector


class PrivateSourceReferenceTests(unittest.TestCase):
    def test_single_named_source_is_parsed_but_multiple_sources_are_not_guessed(self):
        self.assertEqual(named_source('档案《Example-Offer》里，每年拿多少钱？'),
                         {'name': 'example-offer', 'query': '每年拿多少钱？'})
        self.assertEqual(named_source('In the record named "Example-Offer", How much is yearly pay?'),
                         {'name': 'example-offer', 'query': 'How much is yearly pay?'})
        self.assertIsNone(named_source('比较《first》与《second》'))
        self.assertIsNone(named_source('how much is yearly pay'))
        self.assertIsNone(named_source('《算法导论》讨论了什么概念？'))

    def test_source_name_is_resolved_before_global_candidate_cutoff(self):
        target = {'id': 'target', 'file': 'fictional-offer.txt', 'scope': 'personal',
                  'text': '# Terms\nAnnual compensation CAD 81000.'}
        noise = [{'id': 'noise-' + str(i), 'file': 'noise-' + str(i) + '.txt', 'scope': 'personal',
                  'text': '# Much yearly pay\nHow much yearly pay is commonly asked.'} for i in range(70)]
        rag = PrivateRAG({'documents': [target, *noise], 'chunks': make_chunks([target, *noise])})
        try:
            result = rag.search('In the record named "fictional-offer", How much is yearly pay?')
            self.assertEqual([hit['document_id'] for hit in result['hits']], ['target'])
            self.assertEqual(result['route']['source_reference'], 'unique')
            self.assertEqual(result['answerability'], 'unverified')
        finally:
            rag.close()

    def test_missing_ambiguous_and_wrong_scope_sources_do_not_return_neighbours(self):
        docs = [{'id': 'one', 'file': 'a/same.txt', 'scope': 'personal', 'text': '# Salary\nSalary CAD 70000.'},
                {'id': 'two', 'file': 'b/same.txt', 'scope': 'personal', 'text': '# Salary\nSalary CAD 80000.'},
                {'id': 'technical', 'file': 'technical.txt', 'scope': 'technical', 'text': '# Salary\nSalary example'}]
        rag = PrivateRAG({'documents': docs, 'chunks': make_chunks(docs)})
        try:
            for name, status in [('unknown', 'missing'), ('same', 'ambiguous'), ('technical', 'missing')]:
                result = rag.search('档案《' + name + '》里，工资是多少？', scope='personal')
                self.assertEqual(result['hits'], [])
                self.assertEqual(result['route']['source_reference'], status)
                self.assertEqual(result['answerability'], 'no_evidence')
            self.assertEqual(rag.search('档案《a/same.txt》里，工资是多少？')['hits'][0]['document_id'], 'one')
        finally:
            rag.close()

    def test_dense_document_filter_keeps_original_chunk_positions(self):
        chunks = [{'document_id': 'one'}, {'document_id': 'two'}]
        class Encoder:
            def encode(self, *args, **kwargs):
                return np.array([[1.0, 0.0]], dtype=np.float32)
        vector = PrivateVector(chunks, Encoder())
        vector.emb = np.array([[0.8, 0.6], [1.0, 0.0]], dtype=np.float32)
        vector.rows = [{'chunk': 0}, {'chunk': 1}]; vector.ready = True
        try:
            self.assertEqual(vector.search('fictional query', document_ids={'one'}), [(0, float(np.float32(.8)))])
            self.assertEqual(vector.search('fictional query', document_ids={'missing'}), [])
            self.assertEqual(vector.search('fictional query')[0][0], 1)
        finally:
            vector.close()

    def test_named_hybrid_prefers_semantic_passage_inside_the_requested_source(self):
        doc = {'id': 'target', 'file': 'fictional-offer.txt', 'scope': 'personal',
               'text': '# Pay questions\nYearly pay is discussed.\n\n# Compensation\nAnnual compensation CAD 81000.'}
        chunks = make_chunks([doc]); rag = PrivateRAG({'documents': [doc], 'chunks': chunks}, mode='hybrid')
        position = next(i for i, chunk in enumerate(chunks) if 'CAD 81000' in chunk['text'])
        class Vector:
            ready = True
            def search(self, query, **kwargs):
                assert kwargs['document_ids'] == {'target'}
                assert 'fictional-offer' not in query
                return [(position, .9)]
        try:
            with patch.object(rag, '_vector', return_value=Vector()):
                result = rag.search('In the record named "fictional-offer", How much is yearly pay?')
            self.assertEqual(result['mode'], 'hybrid')
            self.assertIn('CAD 81000', result['hits'][0]['text'])
        finally:
            rag.close()
