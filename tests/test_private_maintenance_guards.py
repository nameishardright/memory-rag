import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

import mr_private
import mr_private_maintenance
from mr_private import append_journal, import_journals, load_private_rag, load_store
from mr_private_maintenance import cycle, deployment_complete, main as maintenance_main

PASSWORD = 'synthetic-password'
FILES = [{'path': 'family.txt', 'size': 20, 'mtime_ns': 1},
         {'path': '06_代码项目/example/README.md', 'size': 20, 'mtime_ns': 1}]


def journal(source, family_text, stamp, project_text='Synthetic architecture design'):
    append_journal(source / 'extraction.enc', PASSWORD, [
        {'record_type': 'inventory', 'time': stamp, 'files': FILES},
        {**FILES[0], 'record_type': 'file', 'read_ok': True, 'text': family_text, 'method': 'text'},
        {**FILES[1], 'record_type': 'file', 'read_ok': True, 'text': project_text, 'method': 'text'}])


class Clock:
    def __init__(self, marker, run_seconds):
        self.now = self.start = 1_800_000_000.0
        self.marker, self.run_seconds = marker, run_seconds

    def time(self):
        return self.now

    monotonic = perf_counter = time

    def sleep(self, seconds):
        self.now += seconds
        if self.now - self.start >= self.run_seconds:
            self.marker.touch()


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.source = self.root / 'source'; self.source.mkdir()
        self.run_dir = self.root / 'run'; self.run_dir.mkdir()
        self.store = self.root / 'private' / 'corpus.enc'
        self.sidecars = [self.store, self.store.with_suffix('.status.json'), self.store.with_name('coverage-review.enc')]

    def snapshot(self):
        return [path.read_bytes() for path in self.sidecars]

    def test_reimport_without_changes_does_not_rewrite_the_store(self):
        journal(self.source, 'Synthetic family passport', '2026-01-01')
        first = import_journals(self.source, self.store, PASSWORD, partitioned=True)
        published = self.snapshot()
        journal(self.source, 'Synthetic family passport', '2026-01-02')
        second = import_journals(self.source, self.store, PASSWORD, partitioned=True)
        self.assertTrue(second['unchanged'])
        self.assertEqual((second['updated_at'], second['generation']), (first['updated_at'], first['generation']))
        self.assertEqual(self.snapshot(), published)
        journal(self.source, 'Synthetic updated family tax passport', '2026-01-03')
        third = import_journals(self.source, self.store, PASSWORD, partitioned=True)
        self.assertNotIn('unchanged', third)
        self.assertNotEqual(self.snapshot()[0], published[0])

    def run_cycle(self, passed):
        seen = []

        def verify(store, password):
            seen.append(Path(store).name)
            return {'passed': passed, 'snapshot_updated_at': load_store(store, password)['updated_at']}

        with patch('mr_private_maintenance.run_script'), \
             patch('mr_private_onenote.harvest', return_value={'state': 'no_linked_notebooks'}), \
             patch('mr_private_maintenance.verify_live', side_effect=verify):
            try:
                cycle(self.source, self.store, PASSWORD, self.run_dir)
                return seen, None
            except Exception as error:
                return seen, error

    def leftovers(self):
        return sorted(p.name for p in self.store.parent.iterdir() if 'candidate' in p.name)

    def partitions(self):
        return sorted(p.name for p in self.store.parent.iterdir() if re.fullmatch(r'corpus\.[0-9a-f]{16}\.\w+\.enc', p.name))

    def root_stamp(self):
        return load_store(self.store, PASSWORD)['updated_at']

    def status_stamp(self):
        return json.loads(self.store.with_suffix('.status.json').read_text(encoding='utf-8'))['updated_at']

    def blocked_switch(self, final):
        real = os.replace

        def replace(origin, target):
            if Path(target) == final and 'candidate' in Path(origin).name:
                raise PermissionError('synthetic')
            return real(origin, target)

        return patch('mr_private.os.replace', side_effect=replace)

    def test_rejected_candidate_takes_its_new_partitions_with_it(self):
        journal(self.source, 'Synthetic family passport', '2026-01-01')
        self.run_cycle(True)
        published = self.partitions()
        self.assertEqual(len(published), 1)
        journal(self.source, 'Synthetic family passport', '2026-01-02', 'Synthetic architecture design, second revision')
        seen, error = self.run_cycle(False)
        self.assertEqual(seen, ['corpus.candidate.enc'])
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(self.partitions(), published)
        self.assertEqual(self.leftovers(), [])

    def test_a_failed_switch_leaves_the_previous_generation_whole(self):
        journal(self.source, 'Synthetic family passport', '2026-01-01')
        self.run_cycle(True)
        accepted, published = self.snapshot(), self.partitions()
        verified = (self.store.parent / 'verification.json').read_bytes()
        journal(self.source, 'Synthetic updated family tax passport', '2026-01-02', 'Synthetic architecture design, second revision')
        with self.blocked_switch(self.store), patch('mr_private.REPLACE_WAIT', 0, create=True):
            seen, error = self.run_cycle(True)
        self.assertEqual(seen, ['corpus.candidate.enc'])
        self.assertIsInstance(error, PermissionError)
        self.assertEqual(self.snapshot(), accepted)
        self.assertEqual((self.store.parent / 'verification.json').read_bytes(), verified)
        self.assertEqual(self.partitions(), published)
        self.assertEqual(self.leftovers(), [])

        self.assertEqual(self.run_cycle(True), (['corpus.candidate.enc'], None))
        self.assertTrue(load_private_rag(self.store, PASSWORD).search('tax')['hits'])
        self.assertEqual(self.status_stamp(), self.root_stamp())

    def test_a_stale_sidecar_fails_the_run_and_is_repaired_on_the_next_one(self):
        journal(self.source, 'Synthetic family passport', '2026-01-01')
        self.run_cycle(True)
        journal(self.source, 'Synthetic updated family tax passport', '2026-01-02')
        with self.blocked_switch(self.store.with_suffix('.status.json')), patch('mr_private.REPLACE_WAIT', 0, create=True):
            seen, error = self.run_cycle(True)
        self.assertEqual(seen, ['corpus.candidate.enc'])
        self.assertIsInstance(error, RuntimeError)
        self.assertTrue(load_private_rag(self.store, PASSWORD).search('tax')['hits'])
        report = json.loads((self.store.parent / 'verification.json').read_text(encoding='utf-8'))
        self.assertEqual(report['snapshot_updated_at'], self.root_stamp())
        self.assertNotEqual(self.status_stamp(), self.root_stamp())

        self.assertEqual(self.run_cycle(True), (['corpus.candidate.enc'], None))
        self.assertEqual(self.status_stamp(), self.root_stamp())
        self.assertEqual(self.leftovers(), [])
        self.assertEqual(self.run_cycle(False), ([], None))

    def test_verification_is_not_skipped_when_the_accepted_report_is_for_another_root(self):
        journal(self.source, 'Synthetic family passport', '2026-01-01')
        self.assertEqual(self.run_cycle(True), (['corpus.enc'], None))
        self.assertEqual(self.run_cycle(False), ([], None))
        status_path, report_path = self.store.with_suffix('.status.json'), self.store.parent / 'verification.json'
        for stamp in ('2020-01-01T00:00:00+00:00', None):
            status = json.loads(status_path.read_text(encoding='utf-8'))
            status['updated_at'] = stamp
            status_path.write_text(json.dumps(status), encoding='utf-8')
            report_path.write_text(json.dumps({'passed': True, 'snapshot_updated_at': stamp}), encoding='utf-8')
            seen, error = self.run_cycle(True)
            self.assertIsNone(error)
            self.assertEqual(len(seen), 1, stamp)
            self.assertEqual(self.status_stamp(), self.root_stamp())
            report = json.loads(report_path.read_text(encoding='utf-8'))
            self.assertEqual(report['snapshot_updated_at'], self.root_stamp())

    def test_a_different_gap_list_or_a_moved_source_is_a_new_generation(self):
        def with_gap(source, reason, stamp):
            files = [FILES[0], {'path': 'scan.pdf', 'size': 9, 'mtime_ns': 1}]
            append_journal(source / 'extraction.enc', PASSWORD, [
                {'record_type': 'inventory', 'time': stamp, 'files': files},
                {**files[0], 'record_type': 'file', 'read_ok': True, 'text': 'Synthetic family passport', 'method': 'text'},
                {**files[1], 'record_type': 'file', 'read_ok': False, 'text': '', 'method': 'pdf_text', 'warnings': [reason]}])

        def listed():
            review = load_store(self.store.with_name('coverage-review.enc'), PASSWORD)
            return [(row['path'], row['warnings']) for row in review['gaps']]

        with_gap(self.source, 'first reason', '2026-01-01')
        first = import_journals(self.source, self.store, PASSWORD, partitioned=True)
        with_gap(self.source, 'second reason', '2026-01-02')
        second = import_journals(self.source, self.store, PASSWORD, partitioned=True)
        self.assertEqual(second['coverage'], first['coverage'])
        self.assertNotIn('unchanged', second)
        self.assertEqual(listed(), [('scan.pdf', ['second reason'])])
        self.assertTrue(import_journals(self.source, self.store, PASSWORD, partitioned=True)['unchanged'])

        moved = self.root / 'moved'
        shutil.copytree(self.source, moved)
        third = import_journals(moved, self.store, PASSWORD, partitioned=True)
        self.assertNotIn('unchanged', third)
        self.assertEqual(load_store(self.store, PASSWORD)['source'], str(moved.resolve()))

    def test_a_candidate_root_is_checked_against_its_own_review(self):
        self.assertEqual(mr_private.review_path(Path('private/corpus.enc')).name, 'coverage-review.enc')
        self.assertEqual(mr_private.review_path(Path('private/corpus.candidate.enc')).name, 'coverage-review.candidate.enc')
        pairs = [(candidate.name, final.name) for candidate, final in mr_private.candidate_files(Path('private/corpus.enc'))]
        self.assertEqual(pairs[1], ('coverage-review.candidate.enc', 'coverage-review.enc'))

    def test_new_generation_goes_live_only_after_it_passes_verification(self):
        journal(self.source, 'Synthetic family passport', '2026-01-01')
        self.assertEqual(self.run_cycle(True), (['corpus.enc'], None))
        accepted = self.snapshot()
        verified = (self.store.parent / 'verification.json').read_bytes()

        journal(self.source, 'Synthetic updated family tax passport', '2026-01-02')
        seen, error = self.run_cycle(False)
        self.assertEqual(seen, ['corpus.candidate.enc'])
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(self.snapshot(), accepted)
        self.assertEqual((self.store.parent / 'verification.json').read_bytes(), verified)
        self.assertFalse(json.loads((self.store.parent / 'verification-rejected.json').read_text(encoding='utf-8'))['passed'])
        self.assertEqual(self.leftovers(), [])
        self.assertFalse(load_private_rag(self.store, PASSWORD).search('tax')['hits'])

        self.assertEqual(self.run_cycle(True), (['corpus.candidate.enc'], None))
        self.assertNotEqual(self.snapshot()[0], accepted[0])
        self.assertEqual(self.leftovers(), [])
        rag = load_private_rag(self.store, PASSWORD)
        self.assertTrue(rag.search('tax')['hits'])
        self.assertTrue(rag.search('architecture', scope='technical')['hits'])
        status = json.loads(self.store.with_suffix('.status.json').read_text(encoding='utf-8'))
        self.assertEqual(status['updated_at'], rag.payload['updated_at'])

        live = self.snapshot()
        journal(self.source, 'Synthetic updated family tax passport', '2026-01-03')
        self.assertEqual(self.run_cycle(False), ([], None))
        self.assertEqual(self.snapshot(), live)

    def test_verification_crash_discards_the_candidate(self):
        journal(self.source, 'Synthetic family passport', '2026-01-01')
        self.run_cycle(True)
        accepted = self.snapshot()
        journal(self.source, 'Synthetic updated family tax passport', '2026-01-02')
        with patch('mr_private_maintenance.run_script'), \
             patch('mr_private_onenote.harvest', return_value={'state': 'no_linked_notebooks'}), \
             patch('mr_private_maintenance.verify_live', side_effect=OSError('synthetic')):
            with self.assertRaises(OSError):
                cycle(self.source, self.store, PASSWORD, self.run_dir)
        self.assertEqual(self.snapshot(), accepted)
        self.assertEqual(self.leftovers(), [])


class OnceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.source = self.root / 'source'; self.source.mkdir()
        self.output = self.root / 'private'; self.output.mkdir()
        self.store = self.output / 'corpus.enc'
        self.sidecars = [self.store, self.store.with_suffix('.status.json'), self.store.with_name('coverage-review.enc')]

    def run_once(self, passed):
        seen = []

        def verify(store, password):
            seen.append(Path(store).name)
            return {'passed': passed, 'snapshot_updated_at': load_store(store, password)['updated_at']}

        with patch('mr_private_maintenance.sys.stdin', io.StringIO(PASSWORD + '\n')), \
             patch('mr_private_maintenance.read_journal', return_value=iter([{}])), \
             patch('mr_private_maintenance.stop_legacy_watcher'), \
             patch('mr_private_maintenance.source_fingerprint', return_value='v1'), \
             patch('mr_private_maintenance.run_script'), \
             patch('mr_private_onenote.harvest', return_value={'state': 'no_linked_notebooks'}), \
             patch('mr_private_maintenance.verify_live', side_effect=verify):
            code = maintenance_main(['--source', str(self.source), '--store', str(self.store),
                                     '--stdin-password', '--once', '--worker'])
        return code, seen

    def test_one_shot_run_keeps_the_live_store_until_the_new_generation_is_verified(self):
        journal(self.source, 'Synthetic family passport', '2026-01-01')
        self.assertEqual(self.run_once(True), (0, ['corpus.enc']))
        accepted = [path.read_bytes() for path in self.sidecars]

        journal(self.source, 'Synthetic updated family tax passport', '2026-01-02')
        self.assertEqual(self.run_once(False), (2, ['corpus.candidate.enc']))
        self.assertEqual([path.read_bytes() for path in self.sidecars], accepted)
        self.assertEqual(sorted(p.name for p in self.output.iterdir() if 'candidate' in p.name), [])
        self.assertFalse(load_private_rag(self.store, PASSWORD).search('tax')['hits'])

        self.assertEqual(self.run_once(True), (0, ['corpus.candidate.enc']))
        self.assertTrue(load_private_rag(self.store, PASSWORD).search('tax')['hits'])


class LockTests(unittest.TestCase):
    def test_only_one_cycle_runs_for_a_source_at_a_time(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            store = source / 'private' / 'corpus.enc'
            with mr_private_maintenance.single_writer(source):
                with self.assertRaises(RuntimeError):
                    with mr_private_maintenance.single_writer(source):
                        pass
                with patch('mr_private_maintenance.run_script') as collect:
                    with self.assertRaises(RuntimeError):
                        cycle(source, store, PASSWORD, source)
                    collect.assert_not_called()
            with mr_private_maintenance.single_writer(source):
                pass


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.source = self.root / 'source'; self.source.mkdir()
        self.output = self.root / 'private'; self.output.mkdir()
        self.store = self.output / 'corpus.enc'
        self.fingerprint = 'v1'

    def run_loop(self, run_seconds, cycle_effect, interval=30, **extra):
        clock = Clock(self.source / 'stop.request', run_seconds)
        self.clock = clock
        patches = {'source_fingerprint': lambda *_a, **_k: self.fingerprint() if callable(self.fingerprint) else self.fingerprint,
                   'implementation_signature': lambda: 'same', **extra}
        with patch('mr_private_maintenance.sys.stdin', io.StringIO(PASSWORD + '\n')), \
             patch('mr_private_maintenance.read_journal', return_value=iter([{}])), \
             patch('mr_private_maintenance.stop_legacy_watcher'), \
             patch('mr_private_maintenance.time', clock), \
             patch.multiple('mr_private_maintenance', **patches), \
             patch('mr_private_maintenance.cycle', side_effect=cycle_effect) as run_cycle:
            result = maintenance_main(['--source', str(self.source), '--store', str(self.store),
                                       '--stdin-password', '--interval', str(interval)])
        self.assertEqual(result, 0)
        return run_cycle

    def state(self, name):
        return json.loads((self.output / name).read_text(encoding='utf-8'))

    def test_repeated_failures_back_off_instead_of_rerunning_every_interval(self):
        run_cycle = self.run_loop(4 * 3600, RuntimeError('synthetic'))
        self.assertTrue(5 <= run_cycle.call_count <= 10, run_cycle.call_count)
        self.assertEqual(self.state('maintenance-failure.json')['failures'], run_cycle.call_count)
        self.assertFalse((self.output / 'maintenance-checkpoint.json').exists())

    def test_changed_source_is_retried_at_once_and_success_clears_the_backoff(self):
        attempts = []

        def flaky(*_args):
            attempts.append((self.clock.now - self.clock.start, self.fingerprint()))
            if self.fingerprint() == 'v1':
                raise RuntimeError('synthetic')
            return {'coverage': {'content_gaps': {}}}

        self.fingerprint = lambda: 'v1' if self.clock.now - self.clock.start < 3000 else 'v2'
        self.run_loop(2 * 3600, flaky)
        fixed = [second for second, fingerprint in attempts if fingerprint == 'v2']
        self.assertEqual(len(fixed), 1)
        self.assertLess(fixed[0], 3000 + 60)
        self.assertFalse((self.output / 'maintenance-failure.json').exists())
        self.assertEqual(self.state('maintenance-checkpoint.json')['source_fingerprint'], 'v2')

    def test_the_same_permanent_gaps_are_retried_less_and_less_often(self):
        summary = {'coverage': {'content_gaps': {'read_failed': 12}}}
        run_cycle = self.run_loop(48 * 3600, lambda *_args: summary, interval=300)
        self.assertTrue(6 <= run_cycle.call_count <= 7, run_cycle.call_count)
        self.assertGreaterEqual(self.state('maintenance-checkpoint.json')['gap_retries'], 4)

    def test_unreadable_checkpoint_does_not_end_the_loop(self):
        (self.output / 'maintenance-checkpoint.json').write_text('{not json', encoding='utf-8')
        run_cycle = self.run_loop(60, lambda *_args: {'coverage': {'content_gaps': {}}})
        self.assertEqual(run_cycle.call_count, 1)
        self.assertEqual(self.state('maintenance-checkpoint.json')['retryable_gaps'], 0)

    def test_restart_waits_for_a_complete_deployment(self):
        signatures = iter(['same'] + ['changed'] * 1000)
        with patch('mr_private_maintenance.subprocess.Popen') as child:
            run_cycle = self.run_loop(120, lambda *_args: {'coverage': {'content_gaps': {}}},
                                      implementation_signature=lambda: next(signatures),
                                      deployment_complete=lambda: False)
            child.assert_not_called()
        self.assertEqual(run_cycle.call_count, 0)
        self.assertEqual(self.state('watcher.status.json')['state'], 'waiting_for_deployment')


    def test_unreadable_code_files_wait_instead_of_ending_the_loop(self):
        answers = iter(['same'])

        def signature():
            for answer in answers:
                return answer
            raise PermissionError('synthetic')

        run_cycle = self.run_loop(120, lambda *_args: {'coverage': {'content_gaps': {}}}, implementation_signature=signature)
        self.assertEqual(run_cycle.call_count, 0)
        self.assertEqual(self.state('watcher.status.json')['state'], 'waiting_for_deployment')


class ManifestTests(unittest.TestCase):
    def test_deployment_is_complete_only_when_every_listed_file_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory) / 'app'; (app / 'scripts').mkdir(parents=True)
            (app / 'mr_private.py').write_text('new', encoding='utf-8')
            (app / 'scripts' / 'worker.py').write_text('worker', encoding='utf-8')
            self.assertTrue(deployment_complete(app))
            digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
            manifest = Path(directory) / 'app-manifest.json'
            manifest.write_text(json.dumps({'sha256': {'mr_private.py': digest('new'), 'scripts/worker.py': digest('worker')}}))
            self.assertTrue(deployment_complete(app))
            (app / 'mr_private.py').write_text('half synced', encoding='utf-8')
            self.assertFalse(deployment_complete(app))
            (app / 'mr_private.py').write_text('new', encoding='utf-8')
            (app / 'scripts' / 'worker.py').unlink()
            self.assertFalse(deployment_complete(app))
            manifest.write_text('{truncated', encoding='utf-8')
            self.assertFalse(deployment_complete(app))
            (app / 'scripts' / 'worker.py').write_text('worker', encoding='utf-8')
            manifest.write_text(json.dumps({'sha256': {'mr_private.py': digest('new'), 'scripts/worker.py': digest('worker')},
                                            'vendored': {'bundled': {'version': '1', 'files': {'bundled/main.py': digest('bundled')}}}}))
            self.assertFalse(deployment_complete(app))
            (app / 'bundled').mkdir()
            (app / 'bundled' / 'main.py').write_text('bundled', encoding='utf-8')
            self.assertTrue(deployment_complete(app))


if __name__ == '__main__':
    unittest.main()
