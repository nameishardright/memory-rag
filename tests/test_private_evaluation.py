import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from mr_private import PrivateRAG, make_chunks, save_store, load_store
from mr_private_checkup import source_probes, run_checkup, acceptance_checks
from mr_private_eval import evaluate
from mr_private_sources import stage_github_snapshots, audit_sources


class PrivateEvaluationTests(unittest.TestCase):
    def test_startup_baseline_miss_does_not_mask_actual_retrieval_regressions(self):
        baseline = {'labels_sha256': 'frozen', 'file_hits_at_k': 20, 'evidence_complete_at_k': 20}
        summary = {'personal_text_hashes_match': True, 'native_ui': {'passed': True},
                   'bm25_source_probes': baseline,
                   'hybrid_source_probes': {**baseline, 'modes': ['hybrid']},
                   'fresh_process_start': {
                       'bm25': {'actual_mode': 'bm25', 'topk': 3, 'file_hit': False, 'evidence_complete': False},
                       'hybrid': {'actual_mode': 'hybrid', 'topk': 3, 'file_hit': True, 'evidence_complete': True}}}
        self.assertTrue(all(acceptance_checks(summary).values()))
        summary['hybrid_source_probes']['file_hits_at_k'] = 19
        self.assertFalse(all(acceptance_checks(summary).values()))
        summary['hybrid_source_probes']['file_hits_at_k'] = 20
        summary['fresh_process_start']['hybrid']['actual_mode'] = 'bm25(fallback)'
        self.assertFalse(all(acceptance_checks(summary).values()))

    def test_evidence_metrics_are_separate_from_preview_and_answers(self):
        docs = [{'id': 'a', 'file': 'a.txt', 'scope': 'personal',
                 'text': '# Contract start\nContract begins 2024-02-01.\n\n# Contract end\nContract ends 2025-08-31.'}]
        rag = PrivateRAG({'documents': docs, 'chunks': make_chunks(docs)})
        labels = {'version': 1, 'questions': [{'id': 'q', 'query': 'contract start end dates',
                  'case_group': 'contract', 'expected_documents': ['a'],
                  'required_text': ['2024-02-01', '2025-08-31']},
                  {'id': 'n', 'query': 'current contract for 2030', 'case_group': 'contract', 'positive': False}]}
        summary = evaluate(rag, labels, repetitions=0)['summary']
        self.assertEqual(summary['file_hits_at_k'], 1)
        self.assertEqual(summary['evidence_complete_at_k'], 1)
        self.assertEqual(summary['preview_complete_at_k'], 0)
        self.assertIsNone(summary['answer_accuracy'])
        self.assertEqual(summary['negative_candidates_marked_unverified'], 1)
        labels['questions'][0]['required_text'] = ['unrecorded fact']
        with self.assertRaises(ValueError):
            evaluate(rag, labels)

    def test_public_snapshots_are_verified_and_originals_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); original = root / 'archives'; original.mkdir()
            path = original / 'memory-rag-abcdef123456.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('memory-rag/README.md', 'Synthetic public project description')
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            first = stage_github_snapshots(original, root / 'OneDrive')
            second = stage_github_snapshots(original, root / 'OneDrive')
            self.assertEqual(first, second)
            self.assertEqual(first['archives'], 1)
            self.assertFalse(first['indexed'])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            copied = Path(first['destination']) / path.name
            copied.write_bytes(b'different')
            with self.assertRaises(ValueError):
                stage_github_snapshots(original, root / 'OneDrive')

    def test_source_audit_requires_body_documents_not_just_archive_presence(self):
        docs = [{'id': 'github', 'file': '06_代码项目/GitHub资料快照/memory-rag.zip!/README.md',
                 'scope': 'technical', 'text': 'Synthetic project source'},
                {'id': 'ai', 'file': '导入/AI备份/projects/readme.md', 'scope': 'technical', 'text': 'Synthetic AI project'}]
        rag = PrivateRAG({'documents': docs, 'chunks': make_chunks(docs)})
        audit = audit_sources(rag)
        self.assertEqual(audit['github_snapshot_documents'], 1)
        self.assertTrue(audit['github_snapshot_content_confirmed'])
        self.assertTrue(audit['onedrive_ai_backup_content_confirmed'])
        manifest = [{'id': 'manifest', 'file': '06_代码项目/GitHub资料快照/github-source-manifest.json',
                     'scope': 'technical', 'text': 'Snapshot metadata'}]
        other = PrivateRAG({'documents': manifest, 'chunks': make_chunks(manifest)})
        self.assertFalse(audit_sources(other)['github_snapshot_content_confirmed'])

    def test_source_probes_are_frozen_from_bodies_and_labelled_as_smoke_checks(self):
        docs = [{'id': 'a', 'file': 'a.txt', 'scope': 'personal',
                 'text': 'Synthetic source line with enough detail for a retrieval probe.'}]
        rag = PrivateRAG({'documents': docs, 'chunks': make_chunks(docs)})
        labels = source_probes(rag)
        self.assertEqual(labels['questions'][0]['expected_documents'], ['a'])
        self.assertEqual(labels['questions'][0]['required_text'], [docs[0]['text']])
        self.assertIn('not independently', labels['boundary'])

    def test_checkup_persists_encrypted_indexes_and_tests_native_ui(self):
        # This validates the real encrypted-worker/UI path, not model quality.
        from test_private_retrieval import TestEncoder
        body = '# 护照\nSynthetic passport source for a local encrypted retrieval check.\nSYNTHETIC-PRIVATE-MARKER-Checkup-3194'
        docs = [{'id': 'personal:passport', 'file': 'synthetic-passport.txt',
                 'scope': 'personal', 'text': body,
                 'text_sha256': hashlib.sha256(body.encode()).hexdigest()}]
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory) / 'corpus.enc'
            save_store(store, 'synthetic-password', {'version': 1, 'documents': docs,
                 'chunks': make_chunks(docs), 'updated_at': '2026-01-01',
                 'coverage': {'documents': 1, 'visible_files': 1, 'content_gaps': {}}})
            save_store(store.parent / 'coverage-review.enc', 'synthetic-password',
                       {'version': 1, 'gaps': [], 'not_processed': []})
            (store.parent / 'retrieval-settings.json').write_text(json.dumps({
                'version': 1, 'mode': 'hybrid', 'encrypted_index_cache': True}), encoding='utf-8')
            with patch('mr_private_semantic.LocalEncoder', side_effect=lambda directory=None: TestEncoder()):
                summary = run_checkup(store, 'synthetic-password', collect=False)
            self.assertTrue(summary['passed'])
            self.assertTrue(summary['native_ui']['lock_clears_session'])
            self.assertEqual(summary['hybrid_source_probes']['modes'], ['hybrid'])
            self.assertIsNone(summary['natural_question_accuracy'])
            self.assertTrue((store.parent / 'corpus.semantic.personal.enc').is_file())
            report = load_store(store.parent / 'personal-checkup' / 'source-probe-results.enc', 'synthetic-password')
            self.assertIn('source_probes', report)
            frozen_path = store.parent / 'personal-checkup' / 'source-probes.enc'
            frozen_bytes = frozen_path.read_bytes()
            def failed_hybrid(rag, labels, **kwargs):
                result = evaluate(rag, labels, **kwargs)
                if rag.mode == 'hybrid':
                    result['summary']['file_hits_at_k'] = 0
                    result['summary']['evidence_complete_at_k'] = 0
                return result
            with patch('mr_private_semantic.LocalEncoder', side_effect=lambda directory=None: TestEncoder()), \
                 patch('mr_private_eval.evaluate', side_effect=failed_hybrid):
                failed = run_checkup(store, 'synthetic-password', collect=False)
            self.assertFalse(failed['passed'])
            state = json.loads((store.parent / 'optimization.status.json').read_text(encoding='utf-8'))
            self.assertEqual(state['state'], 'failed_checks')
            self.assertEqual(frozen_path.read_bytes(), frozen_bytes)
            for path in store.parent.rglob('*'):
                if path.is_file():
                    self.assertNotIn(b'SYNTHETIC-PRIVATE-MARKER-Checkup-3194', path.read_bytes())


if __name__ == '__main__':
    unittest.main()
