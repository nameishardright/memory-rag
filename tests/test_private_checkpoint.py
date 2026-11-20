import base64
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import mr_private_semantic
from mr_private_semantic import PrivateVector, VECTOR_VERSION


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now


class Encoder:
    signature = 'synthetic-checkpoint-encoder'

    def __init__(self, clock, seconds_per_batch):
        self.clock, self.seconds_per_batch = clock, seconds_per_batch

    def windows(self, chunk):
        return [{'key': chunk['md5'], 'start': 0, 'end': 1, 'text': chunk['md5']}]

    def encode(self, texts, query=False):
        self.clock.now += self.seconds_per_batch
        return np.ones((len(texts), 2), dtype='<f4')


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.enterContext(patch.object(mr_private_semantic, 'time', self.clock))
        self.cache = Path(self.enterContext(tempfile.TemporaryDirectory())) / 'semantic.enc'
        self.saves = []

    def run_prepare(self, total, seconds_per_batch, save_seconds_per_row=0.0, reused=0):
        chunks = [{'document_id': 'd', 'md5': 'k%d' % i, 'text': ''} for i in range(total)]
        encoder = Encoder(self.clock, seconds_per_batch)
        loader = None
        if reused:
            self.cache.touch()
            old = {'vector_version': VECTOR_VERSION, 'encoder_signature': encoder.signature, 'complete': False,
                   'rows': [{'key': 'k%d' % i, 'chunk': i, 'start': 0, 'end': 1} for i in range(reused)],
                   'dimensions': 2, 'vectors': base64.b64encode(np.ones((reused, 2), dtype='<f4').tobytes()).decode('ascii')}
            loader = lambda path: old

        def saver(path, payload):
            started = self.clock.now
            self.clock.now += len(payload['rows']) * save_seconds_per_row
            self.saves.append({'rows': len(payload['rows']), 'complete': payload['complete'],
                               'started': started, 'ended': self.clock.now})

        vector = PrivateVector(chunks, encoder, self.cache, loader, saver)
        result = vector.prepare()
        self.assertTrue(vector.ready)
        self.assertEqual((result['windows'], result['reused']), (total, reused))
        self.assertEqual([s['complete'] for s in self.saves], [False] * (len(self.saves) - 1) + [True])
        self.assertEqual(self.saves[-1]['rows'], total)
        return self.saves[:-1]

    def test_reused_prefix_no_longer_decides_whether_progress_is_saved(self):
        for reused in (0, 16, 1003):
            self.saves.clear()
            self.clock.now = 0.0
            if self.cache.exists():
                self.cache.unlink()
            middle = self.run_prepare(24000, seconds_per_batch=0.2, reused=reused)
            self.assertGreaterEqual(len(middle), 40, reused)
            marks = [reused] + [s['rows'] for s in middle] + [24000]
            self.assertLessEqual(max(b - a for a, b in zip(marks, marks[1:])), 512 + 16, reused)

    def test_small_job_that_never_aligned_now_saves(self):
        middle = self.run_prepare(1536, seconds_per_batch=0.2, reused=1003)
        self.assertEqual([s['rows'] for s in middle], [1003 + 512])

    def test_slow_saves_are_spaced_so_they_stay_a_small_share_of_the_run(self):
        middle = self.run_prepare(200000, seconds_per_batch=0.2, save_seconds_per_row=0.0001)
        self.assertLess(len(middle), 80)
        self.assertLess(sum(s['rows'] for s in self.saves), 4_000_000)
        save_seconds = sum(s['ended'] - s['started'] for s in self.saves)
        self.assertLess(save_seconds, 0.15 * self.clock.now)
        for previous, following in zip(middle, middle[1:]):
            unsaved = following['started'] - previous['ended']
            allowed = max(mr_private_semantic.CHECKPOINT_SECONDS,
                          mr_private_semantic.CHECKPOINT_COST_RATIO * (previous['ended'] - previous['started']))
            self.assertLessEqual(unsaved, allowed + 0.2 + 1e-6)

    def test_slow_encoder_is_saved_by_elapsed_time(self):
        middle = self.run_prepare(2000, seconds_per_batch=5.0)
        self.assertGreaterEqual(len(middle), 9)
        marks = [0.0] + [s['started'] for s in middle]
        self.assertLessEqual(max(b - a for a, b in zip(marks, marks[1:])), 60 + 5)


if __name__ == '__main__':
    unittest.main()
