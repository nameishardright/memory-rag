from collections import Counter
import hashlib
from pathlib import Path
import random
import tempfile
import time
import json
import os
import io
import zipfile
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cryptography.exceptions import InvalidTag
from mr_index import BM25, tokenize as shared_tokenize
from mr_private import (PrivateRAG, append_journal, import_journals,
                        load_private_rag, load_store, make_chunks, read_journal, save_store,
                        prune_generated_partitions)
from mr_private_index import PrivateBM25
from mr_private_policy import classify
from mr_private_media import transcribe_media
from mr_private_onenote import page_text, matching_notebooks
from mr_private_maintenance import source_fingerprint, cycle, main as maintenance_main
from mr_private_updates import stop_updates
from mr_private_handoff import handoff_if_requested


def document(path, text, scope='personal', purpose='knowledge'):
    return {'id': path, 'file': path, 'text': text, 'scope': scope, 'purpose': purpose,
            'text_sha256': hashlib.sha256(text.encode()).hexdigest()}


class PrivateOptimizationTests(unittest.TestCase):
    def test_old_project_configuration_is_not_personal_and_historical_tax_is_kept(self):
        for filename in ('settings.json', 'README.md', 'project.txt', 'page.html'):
            self.assertEqual(classify('06_代码项目/old/' + filename)['scope'], 'technical')
        self.assertEqual(classify('04_财税与房产/2013/tax.pdf')['scope'], 'personal')
        self.assertEqual(classify('01_身份与移民/old/passport.pdf')['scope'], 'personal')
        self.assertEqual(classify('导入/user/AI备份/股票分析/ABC/analysis.md')['scope'], 'stock')
        # Built-in routing still sends raw filings to stock; the shipped policy file excludes them.
        self.assertEqual(classify('导入/user/AI备份/股票分析/ABC/originals/10-k.htm',
                                  policy={'version': 1, 'rules': []})['scope'], 'stock')
        self.assertEqual(classify('导入/user/AI备份/股票分析/ABC/originals/10-k.htm')['scope'], 'exclude')
        self.assertEqual(classify('90_历史归档/tax.pdf')['scope'], 'history')

    def test_dependencies_and_evaluations_are_excluded(self):
        for filename in ('node_modules/foo/README.md', 'project/dist/index.js', '.venv/data.txt',
                         'project/app.log', 'project/.env', 'project/.env.production',
                         '.claude/projects/work/session.jsonl', 'golden/answers.json',
                         'library/runtime.class', 'library/driver.jar', '01_identity/driver.jar!/config.properties'):
            self.assertEqual(classify(filename)['scope'], 'exclude')
        self.assertEqual(classify('notes.md', text='---\nrag_purpose: evaluation\n---\nExample')['scope'], 'exclude')
        self.assertNotEqual(classify('notes.md', text='# How to mark an evaluation\n```yaml\nrag_purpose: evaluation\n```')['scope'], 'exclude')
        self.assertEqual(classify('导入/user/海马云/design/README.md')['scope'], 'technical')
        self.assertEqual(classify('database.sqlite')['scope'], 'exclude')

    def test_password_handoff_rejects_untrusted_controller_and_wrong_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir(); drive = root / 'drive'; drive.mkdir()
            request = {'parent_pid': 99123, 'controller': str(root / 'untrusted.py')}
            marker = source / 'rag-restart.request.json'; marker.write_text(json.dumps(request))
            with patch('mr_private_handoff.os.getppid', return_value=123), \
                 patch('mr_private_handoff.subprocess.Popen') as child:
                with self.assertRaises(ValueError): handoff_if_requested('synthetic-password', source, drive)
                child.assert_not_called()
            with patch('mr_private_handoff.os.getppid', return_value=99123), \
                 patch('mr_private_handoff.subprocess.Popen') as child:
                with self.assertRaises(ValueError): handoff_if_requested('synthetic-password', source, drive)
                child.assert_not_called()

    def test_unchanged_sources_skip_rebuild_and_output_changes_do_not_reingest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            drive = root / 'drive'; drive.mkdir()
            note = drive / 'note.txt'; note.write_text('synthetic initial')
            before = source_fingerprint(source, drive, notebook_probe=lambda: 'notebooks:0')
            output = drive / '_Private-Memory-RAG'; output.mkdir(); (output / 'corpus.enc').write_bytes(b'generated output')
            self.assertEqual(before, source_fingerprint(source, drive, notebook_probe=lambda: 'notebooks:0'))
            note.write_text('synthetic modified longer')
            self.assertNotEqual(before, source_fingerprint(source, drive, notebook_probe=lambda: 'notebooks:0'))
            self.assertNotEqual(before, source_fingerprint(source, drive, notebook_probe=lambda: 'notebooks:1'))
            store = output / 'corpus.enc'
            (output / 'maintenance-checkpoint.json').write_text(json.dumps({'source_fingerprint': 'unchanged', 'checked_at_epoch': time.time()}))
            with patch('mr_private_maintenance.sys.stdin', io.StringIO('synthetic-password\n')), \
                 patch('mr_private_maintenance.read_journal', return_value=iter([{}])), \
                 patch('mr_private_maintenance.stop_legacy_watcher'), \
                 patch('mr_private_maintenance.source_fingerprint', return_value='unchanged'), \
                 patch('mr_private_maintenance.cycle') as run_cycle, \
                 patch('mr_private_maintenance.time.sleep', side_effect=lambda _: (source / 'stop.request').touch()):
                result = maintenance_main(['--source', str(source), '--store', str(store), '--stdin-password', '--interval', '30'])
            self.assertEqual(result, 0)
            run_cycle.assert_not_called()

    def test_collection_is_independent_of_archive_packaging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            output = root / 'private'; output.mkdir(); run_dir = root / 'run'; run_dir.mkdir()
            coverage = {'content_gaps': {}, 'documents': 1}
            with patch('mr_private_maintenance.run_script') as command, \
                 patch('mr_private_onenote.harvest', return_value={'state': 'no_linked_notebooks'}), \
                 patch('mr_private_maintenance.import_journals', return_value={'coverage': coverage}), \
                 patch('mr_private_maintenance.verify_live', return_value={'passed': True}):
                cycle(source, output / 'corpus.enc', 'synthetic-password', run_dir)
            self.assertEqual([call.args[1:3] for call in command.call_args_list],
                             [('build_memory.py', ['collect']), ('build_memory.py', ['archives'])])

    def test_stop_control_targets_only_matched_collectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            with patch('mr_private_updates.running_processes', side_effect=lambda script, source: [73123] if script == 'mr_private_maintenance.py' else []), \
                 patch('mr_private_updates.subprocess.run') as command:
                stop_updates(root / 'corpus.enc', source)
            self.assertTrue((source / 'stop.request').exists())
            self.assertTrue((root / 'stop.request').exists())
            self.assertIn('ProcessId=73123', command.call_args.args[0][-1])
            self.assertIn("Contains('mr_private_maintenance.py')", command.call_args.args[0][-1])

    def test_deleted_archive_members_do_not_survive_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            owner = {'path': '01_identity/source.zip', 'size': 10, 'mtime_ns': 20}
            append_journal(source / 'extraction.enc', 'password', [
                {'record_type': 'inventory', 'time': '2026-01-01', 'files': [owner]},
                {**owner, 'record_type': 'file', 'read_ok': True, 'sha256': 'new-content', 'text': '', 'method': 'archive_contents_indexed'},
                {'record_type': 'archive_member', 'archive': owner['path'], 'path': owner['path'] + '!/deleted.txt',
                 'read_ok': True, 'text': 'Removed synthetic evidence', 'method': 'text'},
                {'record_type': 'archive_done', 'archive': owner['path'], 'source_sha256': 'new-content', 'members': []}])
            payload = import_journals(source, root / 'private' / 'corpus.enc', 'password')
            self.assertEqual(payload['documents'], [])

    def test_real_collector_pipeline_handles_zip_and_html_without_plaintext_exports(self):
        import importlib.util
        collector_path = Path(os.environ.get('MR_PRIVATE_SOURCE_TOOLS') or
                              Path.home() / 'Documents' / 'Private-Memory-RAG' / 'tools') / 'build_memory.py'
        if not collector_path.is_file(): self.skipTest('Installed collector unavailable')
        specification = importlib.util.spec_from_file_location('private_pipeline_fixture', collector_path)
        collector = importlib.util.module_from_spec(specification); specification.loader.exec_module(collector)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            drive = root / 'drive'; drive.mkdir()
            (drive / '01_identity').mkdir()
            (drive / '01_identity' / 'record.html').write_text('<p>Synthetic passport evidence</p><script>hidden</script>')
            with zipfile.ZipFile(drive / '01_identity' / 'source.zip', 'w') as archive:
                archive.writestr('travel.txt', 'Synthetic Montreal departure record')
                archive.writestr('node_modules/dependency.txt', 'Should be excluded')
                inner = io.BytesIO()
                with zipfile.ZipFile(inner, 'w') as nested:
                    nested.writestr('extra.txt', 'Synthetic nested boarding evidence')
                    nested.writestr('library.class', b'Compiled should be excluded')
                archive.writestr('nested.zip', inner.getvalue())
            collector.ROOT, collector.BASE, collector.JOURNAL = drive, source, source / 'extraction.enc'
            with redirect_stdout(io.StringIO()):
                collector.collect('synthetic-password')
                collector.archive_members('synthetic-password')
            payload = import_journals(source, root / 'private' / 'corpus.enc', 'synthetic-password', partitioned=True)
            rag = PrivateRAG(payload)
            self.assertTrue(rag.search('passport')['hits'])
            self.assertTrue(rag.search('Montreal')['hits'])
            self.assertTrue(rag.search('boarding')['hits'])
            self.assertEqual(rag.search('excluded')['hits'], [])
            for path in source.glob('*.enc'):
                self.assertNotIn(b'Synthetic passport', path.read_bytes())

    def test_inverted_scores_and_document_topk_match_full_scan(self):
        random.seed(815)
        docs = [document('d%03d.md' % i, ' '.join(random.choices(
            ['passport', 'Montreal', 'family', 'project', 'tax', 'contract', 'flight'], k=random.randint(10, 100))))
                for i in range(120)]
        chunks = make_chunks(docs)
        chunks.extend({**c, 'md5': c['md5'] + 'copy', 'text': c['text'] + ' flight'} for c in chunks[:10])
        baseline = BM25(chunks)
        # Scoring and exact document top-k are checked with the full scan's own
        # tokens; the private tokenizer is covered in test_private_natural_queries.
        index = PrivateBM25(chunks, tokenizer=shared_tokenize)
        for query in ('family passport', 'flight flight tax', 'Montreal contract', 'absentunique', '家庭'):
            old = baseline.search(query, topk=len(chunks))
            seen, expected = set(), []
            for i, score in old:
                doc_id = chunks[i]['document_id']
                if doc_id not in seen:
                    seen.add(doc_id)
                    expected.append((i, score))
                if len(expected) == 5:
                    break
            actual = index.search(query, 5, group_by_document=True)
            self.assertEqual([i for i, _ in actual], [i for i, _ in expected])
            for (_, actual_score), (_, old_score) in zip(actual, expected):
                self.assertAlmostEqual(actual_score, old_score, places=11)

    def test_partitions_deduplicate_evidence_and_never_load_stock_for_personal_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'journals'
            source.mkdir()
            store = root / 'private' / 'corpus.enc'
            files = [{'path': path, 'size': 100, 'mtime_ns': 20} for path in (
                'family.txt', 'copy/family.txt', '06_代码项目/example/settings.json',
                '导入/user/AI备份/股票分析/ABC/analysis.md', 'node_modules/foo/README.md', 'evaluation.md')]
            rows = [{'record_type': 'inventory', 'time': '2026-01-01', 'files': files}]
            for item in files:
                text = 'passport family SyntheticSecret-local-only'
                if 'settings' in item['path']: text = 'project setting'
                if 'analysis' in item['path']: text = 'stock earnings revenue'
                if 'evaluation' in item['path']: text = '---\nrag_purpose: evaluation\n---\npassport family'
                rows.append({**item, 'record_type': 'file', 'read_ok': True, 'text': text, 'method': 'text'})
            append_journal(source / 'extraction.enc', 'synthetic-password', rows)
            payload = import_journals(source, store, 'synthetic-password', partitioned=True)
            self.assertEqual(payload['coverage']['excluded_files'], 2)
            self.assertEqual(payload['coverage']['duplicate_documents_collapsed'], 1)
            self.assertEqual(payload['coverage']['stock_documents'] if 'stock_documents' in payload['coverage'] else payload['coverage']['scope_documents']['stock'], 1)
            self.assertNotIn('stock', {d['scope'] for d in payload['documents']})
            rag = load_private_rag(store, 'synthetic-password')
            hit = rag.search('passport family')['hits'][0]
            self.assertEqual(len(hit['duplicate_sources']), 1)
            alias = hit['duplicate_sources'][0]
            self.assertEqual(rag.get(alias['id'])['file'], alias['file'])
            self.assertEqual(rag.get(alias['id'])['text'], rag.get(hit['document_id'])['text'])
            self.assertNotIn('stock', rag.loaded_scopes)
            stock_hit = rag.search('earnings', scope='stock')['hits'][0]
            self.assertEqual(rag.get(stock_hit['document_id'])['scope'], 'stock')
            self.assertEqual(rag.search('earnings')['hits'], [])
            for path in store.parent.iterdir():
                self.assertNotIn(b'SyntheticSecret-local-only', path.read_bytes())
                self.assertNotIn(b'family.txt', path.read_bytes())
            with self.assertRaises(InvalidTag):
                load_private_rag(store, 'incorrect')

    def test_unchanged_partitions_are_reused_after_personal_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            store = root / 'private' / 'corpus.enc'
            files = [{'path': 'family.txt', 'size': 20, 'mtime_ns': 1},
                     {'path': '06_代码项目/example/README.md', 'size': 20, 'mtime_ns': 1}]
            append_journal(source / 'extraction.enc', 'password', [
                {'record_type': 'inventory', 'time': '2026-01-01', 'files': files},
                {**files[0], 'record_type': 'file', 'read_ok': True, 'text': 'Synthetic family passport', 'method': 'text'},
                {**files[1], 'record_type': 'file', 'read_ok': True, 'text': 'Synthetic architecture design', 'method': 'text'}])
            first = import_journals(source, store, 'password', partitioned=True)
            first_name = first['shards']['technical']['file']
            first_bytes = store.with_name(first_name).read_bytes()
            append_journal(source / 'extraction.enc', 'password', [
                {**files[0], 'record_type': 'file', 'read_ok': True, 'text': 'Synthetic updated family tax passport', 'method': 'text'}])
            second = import_journals(source, store, 'password', partitioned=True)
            self.assertNotEqual(first['generation'], second['generation'])
            self.assertEqual(first_name, second['shards']['technical']['file'])
            self.assertEqual(first_bytes, store.with_name(first_name).read_bytes())
            rag = load_private_rag(store, 'password')
            self.assertTrue(rag.search('architecture', scope='technical')['hits'])
            self.assertTrue(rag.search('tax')['hits'])

    def test_retention_removes_only_authenticated_expired_generated_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); store = root / 'corpus.enc'
            candidates = []
            for generation, scope in (('a'*16, 'technical'), ('b'*16, 'stock')):
                path = root / ('corpus.' + generation + '.' + scope + '.enc')
                save_store(path, 'password', {'version': 1, 'generation': generation, 'scope': scope, 'partitioned': True})
                os.utime(path, (time.time()-172800, time.time()-172800)); candidates.append(path)
            original = root / 'source-archive.enc'; original.write_bytes(b'Original retained evidence')
            invalid = root / ('corpus.' + 'c'*16 + '.history.enc'); invalid.write_bytes(b'Not a generated partition')
            os.utime(invalid, (time.time()-172800, time.time()-172800))
            count = prune_generated_partitions(store, 'password', {candidates[1].name})
            self.assertEqual(count, 1)
            self.assertFalse(candidates[0].exists()); self.assertTrue(candidates[1].exists())
            self.assertEqual(original.read_bytes(), b'Original retained evidence')
            self.assertTrue(invalid.exists())

    def test_local_transcript_keeps_timestamps_and_marks_uncertain_speech(self):
        segment = SimpleNamespace(text='A synthetic family travel record.', start=2, end=8, avg_logprob=-1.2)
        fake = SimpleNamespace(transcribe=lambda *args, **kwargs: (iter([segment]),
                                SimpleNamespace(language='en', language_probability=.9)))
        with patch('mr_private_media.audio_info', return_value={'has_audio': True, 'duration_seconds': 10}), \
             patch('mr_private_media._load_model', return_value=fake):
            result = transcribe_media('synthetic.wav')
        self.assertIn('00:00:02 --> 00:00:08', result['text'])
        self.assertIn('low_confidence_segments:1', result['warnings'])
        self.assertEqual(result['method'], 'local_speech_transcript')

    def test_onenote_rich_text_and_ocr_are_searchable_without_markup(self):
        xml = '<Page xmlns="http://schemas.microsoft.com/office/onenote/2013/onenote"><Title><T><![CDATA[<span>Trip notes</span>]]></T></Title><OE><T><![CDATA[<b>Montreal</b> departure 2026-01-02]]></T></OE><Image><OCRText>passport evidence</OCRText></Image></Page>'
        text = page_text(xml)
        for value in ('Trip notes', 'Montreal', '2026-01-02', 'passport evidence'):
            self.assertIn(value, text)
        self.assertNotIn('<span>', text)

    def test_onenote_account_id_does_not_select_unrelated_notebooks(self):
        from xml.etree import ElementTree as ET
        unrelated = ET.fromstring('<Notebook name="Other" path="https://d.docs.live.net/123abc/Documents/Other"/>')
        linked = ET.fromstring('<Notebook name="Travel" path="https://d.docs.live.net/123abc/Documents/Travel"/>')
        target = 'https://onedrive.live.com/redir?resid=123abc!789'
        self.assertEqual(matching_notebooks([unrelated], Path('Travel.url'), target), [])
        self.assertEqual(matching_notebooks([linked, unrelated], Path('Travel.url'), target), [linked])
        self.assertEqual(matching_notebooks([linked, linked], Path('Travel.url'), target), [])

    def test_deleted_onenote_page_is_not_retained_from_an_older_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            store = root / 'private' / 'corpus.enc'
            owner = {'path': '05_学习与笔记/notebook.url', 'size': 10, 'mtime_ns': 20}
            page = {'record_type': 'archive_member', 'archive': owner['path'],
                    'path': owner['path'] + '!/OneNote/page.md', 'read_ok': True,
                    'text': 'Deleted historical synthetic page', 'method': 'onenote_com'}
            append_journal(source / 'extraction.enc', 'password', [
                {'record_type': 'inventory', 'time': '2026-01-01', 'files': [owner]},
                {**owner, 'record_type': 'file', 'read_ok': True, 'text': '', 'method': 'web_notebook_link_only'}])
            append_journal(source / 'onenote.enc', 'password', [page,
                {'record_type': 'external_inventory', 'time': '2026-01-02', 'owner': owner['path'], 'paths': []}])
            payload = import_journals(source, store, 'password')
            self.assertEqual(payload['documents'], [])

    def test_collector_excludes_dependencies_and_extracts_visible_html(self):
        import importlib.util
        path = Path(os.environ.get('MR_PRIVATE_SOURCE_TOOLS') or
                    Path.home() / 'Documents' / 'Private-Memory-RAG' / 'tools') / 'build_memory.py'
        if not path.is_file(): self.skipTest('This test covers the installed source collector')
        specification = importlib.util.spec_from_file_location('private_collector_fixture', path)
        collector = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(collector)
        excluded = collector.process({'path': 'node_modules/foo/README.md', 'dependency': True, 'size': 1, 'mtime_ns': 1})
        self.assertEqual(excluded['method'], 'policy_excluded')
        self.assertNotIn('sha256', excluded)
        html = collector.extract(b'<html><style>hidden_style</style><script>hidden_script</script><p>Contract evidence 2026</p></html>', '.html', 'synthetic.html')
        self.assertIn('Contract evidence 2026', html['text'])
        self.assertNotIn('hidden_style', html['text'])
        self.assertNotIn('hidden_script', html['text'])
        link = collector.extract(b'[InternetShortcut]\nURL=https://onedrive.live.com/redir?resid=synthetic', '.url', 'notebook.url')
        self.assertEqual(link['text'], '')
        self.assertEqual(link['method'], 'web_notebook_link_only')

    def test_journal_recovery_authenticates_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'notes.enc'
            append_journal(path, 'password', [{'test': 1}])
            original = path.read_bytes()
            with self.assertRaises(InvalidTag):
                append_journal(path, 'wrong', [{'test': 2}])
            self.assertEqual(path.read_bytes(), original)
            with path.open('ab') as stream:
                stream.write(b'\x01\x00')
            append_journal(path, 'password', [{'test': 2}])
            self.assertEqual(list(read_journal(path, 'password')), [{'test': 1}, {'test': 2}])


if __name__ == '__main__':
    unittest.main()
