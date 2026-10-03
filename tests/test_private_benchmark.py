import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mr_private import PrivateRAG, make_chunks, load_store, save_store
from mr_private_benchmark import compare_reports, consume_holdout, paired_evaluate, isolated_baseline
from mr_private_eval import validate_labels, labels_sha256
from mr_private_natural import author_sets, authoring_counts, source_candidates


def documents(count=24):
    return [{'id': 'fictional-' + str(i), 'file': 'fictional-offer-' + str(i) + '.txt',
             'scope': 'personal', 'method': 'plain_text',
             'text': '# Hypothetical offer ' + str(i) + '\nAnnual compensation: CAD ' +
                     str(80000 + i * 100) + '.\nThis is fictional test data and represents no real person or employer.'}
            for i in range(count)]


class PrivateBenchmarkTests(unittest.TestCase):
    def test_natural_authoring_never_calls_retrieval_and_groups_languages(self):
        with patch.object(PrivateRAG, 'search', side_effect=AssertionError('Authoring searched')):
            sets = author_sets(documents())
        for labels in sets.values():
            validate_labels(labels)
        self.assertEqual(authoring_counts(sets['holdout'])['positive_questions'], 24)
        self.assertEqual(authoring_counts(sets['holdout'])['negative_questions'], 4)
        self.assertEqual(authoring_counts(sets['holdout'])['event_groups'], 12)
        dev_groups = {q['case_group'] for q in sets['development']['questions']}
        held_groups = {q['case_group'] for q in sets['holdout']['questions']}
        self.assertFalse(dev_groups & held_groups)
        dev_queries = {q['query'] for q in sets['development']['questions']}
        self.assertFalse(dev_queries & {q['query'] for q in sets['holdout']['questions']})
        self.assertEqual(sets, author_sets(documents()))

    def test_source_authoring_excludes_previous_probes_duplicates_and_curated_notes(self):
        docs = documents()
        duplicate = {**docs[0], 'id': 'duplicate', 'file': 'duplicate.txt'}
        curated = {**docs[1], 'id': 'curated', 'file': 'curated.txt', 'method': 'curated_memory'}
        excluded = {'fictional-0'}
        rows = source_candidates(docs + [duplicate, curated], excluded)
        self.assertNotIn('fictional-0', {row['document_id'] for row in rows})
        self.assertNotIn('duplicate', {row['document_id'] for row in rows})
        self.assertNotIn('curated', {row['document_id'] for row in rows})

    def test_authoring_rejects_ambiguous_source_names_and_unverifiable_fields(self):
        first = documents(1)[0]
        same_name = {**documents(2)[1], 'file': 'other/' + first['file']}
        no_fact = {**first, 'id': 'missing', 'file': 'no-fact.txt',
                   'text': '# Hypothetical\nAnnual compensation: not recorded.\n' + 'Fictional prose. ' * 10}
        self.assertEqual(source_candidates([first, same_name, no_fact]), [])
        with self.assertRaises(ValueError):
            author_sets(documents(2))

    def test_holdout_marker_is_encrypted_exclusive_and_survives_a_failed_run(self):
        labels = author_sets(documents())['holdout']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.enc'
            marker = consume_holdout(path, 'fictional-test-password', labels)
            self.assertEqual(load_store(marker, 'fictional-test-password')['labels_sha256'], labels_sha256(labels))
            self.assertNotIn(b'fictional-offer', marker.read_bytes())
            with self.assertRaises(FileExistsError):
                consume_holdout(path, 'fictional-test-password', labels)

    def test_paired_comparison_records_gains_regressions_and_correlated_events(self):
        def report(rows):
            return {'summary': {'labels_sha256': 'frozen', 'modes': ['hybrid']}, 'questions': rows}
        rows = [{'id': 'one', 'positive': True, 'case_group': 'event', 'rank': None, 'evidence_complete': False},
                {'id': 'two', 'positive': True, 'case_group': 'event', 'rank': 1, 'evidence_complete': True}]
        new = [{**rows[0], 'rank': 1, 'evidence_complete': True},
               {**rows[1], 'rank': None, 'evidence_complete': False}]
        paired = compare_reports(report(rows), report(new))
        self.assertEqual(paired['file'], {'gains': 1, 'regressions': 1})
        self.assertEqual(paired['evidence'], {'gains': 1, 'regressions': 1})
        self.assertEqual(paired['event_groups'], 1)
        self.assertIsNone(paired['answer_accuracy'])
        self.assertIsNone(paired['refusal_accuracy'])
        changed = report(new); changed['summary']['labels_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            compare_reports(report(rows), changed)

    def test_paired_details_are_encrypted_and_fallback_is_not_a_semantic_gain(self):
        docs = documents(1)
        labels = {'version': 1, 'questions': [{'id': 'query', 'query': 'annual compensation',
                  'case_group': 'event', 'scope': 'personal', 'split': 'regression',
                  'expected_documents': [docs[0]['id']], 'required_text': ['CAD 80000']}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); label_path = root / 'labels.enc'; output = root / 'results.enc'
            save_store(label_path, 'fictional-test-password', labels)
            before = label_path.read_bytes()
            def open_rag(*args, **kwargs):
                return PrivateRAG({'documents': docs, 'chunks': make_chunks(docs)}, mode=kwargs['mode'])
            with patch('mr_private_benchmark.load_private_rag', side_effect=open_rag):
                summary = paired_evaluate(root / 'unused.enc', label_path, output,
                                          'fictional-test-password', repetitions=0)
            self.assertEqual(before, label_path.read_bytes())
            self.assertFalse(summary['comparison']['hybrid_actually_used'])
            self.assertFalse(summary['retrieval_improved_in_this_sample'])
            self.assertEqual(load_store(output, 'fictional-test-password')['labels'], labels)
            self.assertNotIn(b'CAD 80000', output.read_bytes())
            self.assertNotIn('fictional-offer', json.dumps(summary))

    def test_source_change_is_rejected_without_rewriting_labels(self):
        docs = documents(1)
        labels = {'version': 1, 'questions': [{'id': 'query', 'query': 'annual compensation',
                  'case_group': 'event', 'scope': 'personal', 'split': 'holdout',
                  'expected_documents': [docs[0]['id']], 'required_text': ['CAD 80000']}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / 'labels.enc'
            save_store(path, 'fictional-test-password', labels)
            before = path.read_bytes()
            def open_rag(*args, **kwargs):
                changed = docs if kwargs['mode'] == 'bm25' else [{**docs[0], 'text': docs[0]['text'] + '\nChanged source'}]
                return PrivateRAG({'documents': changed, 'chunks': make_chunks(changed)}, mode=kwargs['mode'])
            with patch('mr_private_benchmark.load_private_rag', side_effect=open_rag):
                with self.assertRaises(ValueError):
                    paired_evaluate(root / 'unused.enc', path, root / 'results.enc',
                                    'fictional-test-password', repetitions=0)
            self.assertEqual(before, path.read_bytes())
            self.assertTrue(Path(str(path) + '.consumed.enc').exists())
            self.assertFalse((root / 'results.enc').exists())

    def test_baseline_worker_uses_encrypted_files_in_an_isolated_process(self):
        docs = documents(1)
        labels = {'version': 1, 'questions': [{'id': 'query', 'query': 'annual compensation',
                  'case_group': 'event', 'scope': 'personal', 'split': 'regression',
                  'expected_documents': [docs[0]['id']], 'required_text': ['CAD 80000']}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / 'labels.enc'; store = root / 'corpus.enc'
            save_store(path, 'fictional-test-password', labels)
            save_store(store, 'fictional-test-password', {'version': 1, 'documents': docs,
                       'chunks': make_chunks(docs), 'coverage': {'documents': 1}})
            output = root / 'baseline.enc'
            result = isolated_baseline(Path(__file__).resolve().parents[1], store, path, output,
                                       'fictional-test-password', 'bm25')
            self.assertEqual(result['report']['summary']['evidence_complete_at_k'], 1)
            self.assertNotIn(b'CAD 80000', output.read_bytes())

    def test_isolated_worker_rejects_an_unreserved_holdout(self):
        docs = documents(1)
        labels = {'version': 1, 'questions': [{'id': 'query', 'query': 'annual compensation',
                  'case_group': 'event', 'scope': 'personal', 'split': 'holdout',
                  'expected_documents': [docs[0]['id']], 'required_text': ['CAD 80000']}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / 'labels.enc'; store = root / 'corpus.enc'
            save_store(path, 'fictional-test-password', labels)
            save_store(store, 'fictional-test-password', {'version': 1, 'documents': docs,
                       'chunks': make_chunks(docs), 'coverage': {'documents': 1}})
            output = root / 'baseline.enc'
            with self.assertRaises(RuntimeError):
                isolated_baseline(Path(__file__).resolve().parents[1], store, path, output,
                                  'fictional-test-password', 'bm25')
            self.assertFalse(output.exists())
