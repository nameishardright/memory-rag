import tempfile
from pathlib import Path
import time
import tkinter as tk
import unittest
from unittest.mock import patch

from mr_private import make_chunks, save_store
from mr_private_ui import PrivateMemoryApp


class PrivateUITests(unittest.TestCase):
    def test_copy_includes_contract_evidence_and_keeps_card_copy_bounded(self):
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest('No native display available')
        root.withdraw()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'corpus.enc'
            docs = [{'id': 'personal:contract', 'file': 'synthetic-contract.txt', 'scope': 'personal',
                     'text': '# Start\nContract starts 2024-02-01.\n\n# End\nContract ends 2025-08-31.',
                     'method': 'image_ocr', 'warnings': ['synthetic_ocr_warning']}]
            save_store(path, 'synthetic-password', {'version': 1, 'documents': docs,
                'chunks': make_chunks(docs), 'updated_at': '2026-01-01',
                'coverage': {'documents': 1, 'visible_files': 1}})
            app = PrivateMemoryApp(root, path)
            def until(check):
                deadline = time.monotonic() + 10
                while not check() and time.monotonic() < deadline:
                    root.update(); time.sleep(.01)
                self.assertTrue(check())
            copied = []
            try:
                app.password_entry.insert(0, 'synthetic-password'); app.unlock()
                until(lambda: app.rag is not None)
                app.search_for('contract start end', '个人资料')
                until(lambda: bool(app.rows))
                app.tree.selection_set('0')
                # Preserve the user's actual clipboard; test the submitted text.
                with patch.object(root, 'clipboard_clear'), patch.object(root, 'clipboard_append', side_effect=copied.append):
                    app.copy_selected()
                    self.assertIn('2024-02-01', copied[-1])
                    self.assertIn('2025-08-31', copied[-1])
                    self.assertIn('synthetic-contract.txt', copied[-1])
                    self.assertIn('synthetic_ocr_warning', copied[-1])
                    app.rows['card'] = {'kind': 'card', 'number': '4242424242424242',
                                        'excerpt': 'Synthetic adjacent authentication field'}
                    app.tree.insert('', 'end', iid='card')
                    app.tree.selection_set('card')
                    app.copy_selected()
                    self.assertEqual(copied[-1], '4242424242424242')
            finally:
                app.close()

    def test_scope_selector_and_coverage_review_on_native_ui(self):
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest('No native display available')
        root.withdraw()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'corpus.enc'
            docs = [{'id': 'personal', 'file': 'synthetic.txt', 'scope': 'personal', 'text': 'amex hilton 378282246310005 Synthetic family passport'},
                    {'id': 'stock', 'file': 'synthetic-stock.md', 'scope': 'stock', 'text': 'Synthetic earnings revenue credit card annual fees'}]
            save_store(path, 'password', {'version': 1, 'documents': docs, 'chunks': make_chunks(docs),
                       'updated_at': '2026-01-01', 'coverage': {'documents': 2, 'visible_files': 2}})
            save_store(path.parent / 'coverage-review.enc', 'password', {'version': 1, 'updated_at': '2026-01-01',
                       'gaps': [{'path': 'locked.pdf', 'method': 'password_locked_pdf', 'warnings': []}], 'not_processed': ['waiting.md']})
            app = PrivateMemoryApp(root, path)
            def until(check):
                end = time.monotonic() + 10
                while not check() and time.monotonic() < end:
                    root.update(); time.sleep(.01)
                self.assertTrue(check())
            try:
                app.password_entry.insert(0, 'password'); app.unlock(); until(lambda: app.rag is not None)
                app.scope_choice.set('股票资料'); app.search_for('earnings'); until(lambda: bool(app.rows))
                self.assertEqual(app.rows['0']['document_id'], 'stock')
                app.search_for('credit card'); until(lambda: str(app.search_button['state']) == 'normal')
                self.assertEqual(app.rows['0']['kind'], 'passage')
                self.assertEqual(app.rows['0']['document_id'], 'stock')
                app.coverage_review(); until(lambda: app.rows.get('0', {}).get('kind') == 'cleanup')
                self.assertEqual(len(app.rows), 2)
                self.assertIn('原文件密码', app.rows['0']['excerpt'])
                app.lock(); self.assertFalse(app.rows); self.assertIsNone(app.password)
            finally:
                app.close()

    def test_automatic_scope_reports_every_source_it_searched(self):
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest('No native display available')
        root.withdraw()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'corpus.enc'
            docs = [{'id': 'resume', 'file': 'fictional-resume.txt', 'scope': 'personal',
                     'text': 'Led the Harbor Scheduler project, cutting latency by 37%.'},
                    {'id': 'readme', 'file': 'fictional-readme.md', 'scope': 'technical',
                     'text': 'Harbor Scheduler project. Storage layer: SQLite.'}]
            save_store(path, 'password', {'version': 1, 'documents': docs, 'chunks': make_chunks(docs),
                       'updated_at': '2026-01-01', 'coverage': {'documents': 2, 'visible_files': 2}})
            app = PrivateMemoryApp(root, path)
            def until(check):
                end = time.monotonic() + 10
                while not check() and time.monotonic() < end:
                    root.update(); time.sleep(.01)
                self.assertTrue(check())
            try:
                app.password_entry.insert(0, 'password'); app.unlock(); until(lambda: app.rag is not None)
                app.search_for('scheduler 项目 latency'); until(lambda: len(app.rows) == 2)
                self.assertEqual({row['document_id'] for row in app.rows.values()}, {'resume', 'readme'})
                self.assertIn('代码项目＋个人资料', app.status.get())
                deferred = {'ok': True, 'mode': 'bm25', 'scope': 'personal', 'semantic_state': 'disabled',
                            'answerability': 'no_evidence', 'hits': [],
                            'route': {'scope': 'personal', 'deferred_scope': 'stock'}}
                with patch('mr_private_ui.compact_search', return_value=deferred):
                    app.search_for('stock options'); until(lambda: '未自动加载' in app.status.get())
                self.assertIn('股票资料', app.status.get())
            finally:
                app.close()

    def test_unlock_card_lookup_and_stale_results_after_lock(self):
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest('No native display available')
        root.withdraw()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'corpus.enc'
            docs = [{'id': 'test', 'file': 'fictional.txt', 'scope': 'personal',
                     'text': 'amex hilton 378282246310005'}]
            save_store(path, 'synthetic-password', {'version': 1, 'documents': docs,
                       'chunks': make_chunks(docs), 'updated_at': '2026-01-01',
                       'coverage': {'read_success': 1, 'visible_files': 1, 'documents': 1}})
            app = PrivateMemoryApp(root, path)
            try:
                app.password_entry.insert(0, 'synthetic-password')
                app.unlock()
                end = time.monotonic() + 10
                while app.rag is None and time.monotonic() < end:
                    root.update()
                    time.sleep(.02)
                self.assertIsNotNone(app.rag)
                self.assertEqual(app.password_entry.get(), '')
                app.search_for('信用卡')
                while not app.rows and time.monotonic() < end:
                    root.update()
                    time.sleep(.02)
                self.assertEqual(len(app.rows), 1)
                self.assertEqual(app.rows['0']['number'], '378282246310005')
                # A background import must be visible on the next query.
                docs[0]['text'] = 'Rogers 5555555555554444'
                save_store(path, 'synthetic-password', {'version': 1, 'documents': docs,
                           'chunks': make_chunks(docs), 'updated_at': '2026-01-02',
                           'coverage': {'read_success': 1, 'visible_files': 1, 'documents': 1}})
                app.search_for('信用卡')
                while app.rows.get('0', {}).get('number') != '5555555555554444' and time.monotonic() < end:
                    root.update()
                    time.sleep(.02)
                self.assertEqual(app.rows['0']['number'], '5555555555554444')
                # Cleanup evidence uses the same unlock and must clear on lock.
                save_store(path.parent / 'cleanup-review.enc', 'synthetic-password', {
                    'version': 1, 'created_at': '2026-01-02', 'cache_candidates': [],
                    'exact_duplicates': [{'contains_cache_files': False, 'potential_redundant_bytes': 20,
                                          'files': [{'path': 'a.txt'}, {'path': 'copy/a.txt'}],
                                          'sha256': 'synthetic-hash', 'copies': 2, 'keep_reference': 'a.txt'}],
                    'same_name_different': [], 'largest_files': [],
                    'summary': {'hash_verified_files': 2, 'visible_files': 3, 'exact_duplicate_groups': 1}})
                app.cleanup_review()
                end = time.monotonic() + 10
                while app.rows.get('0', {}).get('kind') != 'cleanup' and time.monotonic() < end:
                    root.update()
                    time.sleep(.02)
                self.assertEqual(app.rows['0']['kind'], 'cleanup')
                self.assertIn('copy/a.txt', app.rows['0']['excerpt'])
                app.tree.selection_set('0')
                app.full_source()
                self.assertIn('synthetic-hash', app.detail.get('1.0', 'end'))
                old_generation = app.generation
                app.last_activity = time.monotonic() - 601
                app.tick()
                self.assertIsNone(app.password)
                self.assertIsNone(app.rag)
                self.assertFalse(app.rows)
                app.messages.put((old_generation, 'cards', [{'number': 'stale'}], None))
                app.poll()
                self.assertFalse(app.rows)
            finally:
                app.close()


if __name__ == '__main__':
    unittest.main()
