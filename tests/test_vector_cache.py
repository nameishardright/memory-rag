import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import mr_index


def unit(text):
    seed = int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:8], "big")
    row = np.random.default_rng(seed).normal(size=8).astype("float32")
    return row / np.linalg.norm(row)


class FakeModel:
    def __init__(self):
        self.texts = []

    def encode(self, texts, **kwargs):
        self.texts.extend(texts)
        return np.stack([unit(text) for text in texts])


def chunk(name):
    return {"file": "f.md", "heading": name, "text": "body " + name, "md5": "key-" + name, "source": "case"}


class VectorCacheTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.spec = {"key": "fake", "id": "fake/model", "slug": "fake", "prefix": "",
                     "local": str(self.root), "weights": "model.safetensors"}
        self.enterContext(patch.object(mr_index, "CACHE", str(self.root)))
        self.enterContext(patch.object(mr_index, "has_weights", return_value=True))
        self.enterContext(patch.dict(sys.modules, {"sentence_transformers": types.SimpleNamespace(
            SentenceTransformer=object, CrossEncoder=object)}))
        self.path = Path(mr_index.cache_file("fake"))

    def build(self, names, model=None):
        model = model or FakeModel()
        index = mr_index.VecIndex([chunk(n) for n in names], model=model, spec=self.spec)
        self.assertTrue(index.available, index.err)
        return index, model

    def top(self, index, name):
        return index.chunks[index.search(mr_index._index_text(chunk(name)), topk=1)[0][0]]["heading"]

    def stored(self):
        return mr_index._read_vectors(str(self.path), "fake/model")

    def legacy(self, keys, rows):
        (self.root/"emb_meta-fake.json").write_text(json.dumps(
            {"model": "fake/model", "ver": mr_index.CACHE_VER, "md5s": ["key-" + k for k in keys]}), encoding="utf-8")
        np.save(self.root/"emb-fake.npy", np.stack([unit(mr_index._index_text(chunk(r))) for r in rows]))

    def test_keys_and_rows_are_published_as_one_file(self):
        index, model = self.build("abc")
        self.assertEqual(len(model.texts), 3)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["emb-fake.npz"])
        for name in "abc":
            self.assertEqual(self.top(index, name), name)
            np.testing.assert_allclose(self.stored()["key-" + name], unit(mr_index._index_text(chunk(name))))

    def test_unchanged_corpus_reuses_rows_and_does_not_rewrite(self):
        self.build("abc")
        before = self.path.stat().st_mtime_ns
        with patch.object(mr_index, "_publish_vectors", wraps=mr_index._publish_vectors) as publish:
            index, model = self.build("cab")
        self.assertEqual(model.texts, [])
        publish.assert_not_called()
        self.assertEqual(self.path.stat().st_mtime_ns, before)
        self.assertEqual(self.top(index, "c"), "c")

    def test_same_length_legacy_pair_from_two_writers_is_rejected(self):
        self.legacy("abc", "bac")
        index, model = self.build("abc")
        for name in "abc":
            self.assertEqual(self.top(index, name), name)
            np.testing.assert_allclose(self.stored()["key-" + name], unit(mr_index._index_text(chunk(name))))

    def test_consistent_legacy_pair_is_migrated_without_a_full_recompute(self):
        names = [str(i) for i in range(200)]
        self.legacy(names, names)
        index, model = self.build(names)
        self.assertLessEqual(len(model.texts), mr_index.LEGACY_SAMPLE)
        self.assertEqual(len(self.stored()), 200)
        self.assertEqual(self.top(index, "137"), "137")
        self.assertTrue((self.root/"emb-fake.npy").exists())

    def test_tail_of_a_long_legacy_pair_is_checked(self):
        names = [str(i) for i in range(200)]
        self.legacy(names, names[:-2] + [names[-1], names[-2]])
        index, model = self.build(names)
        self.assertEqual(self.top(index, "199"), "199")
        self.assertEqual(self.top(index, "198"), "198")

    def assert_rows_belong_to_their_keys(self, names):
        stored = self.stored()
        self.assertEqual(set(stored), {"key-" + n for n in names})
        for name in names:
            np.testing.assert_allclose(stored["key-" + name], unit(mr_index._index_text(chunk(name))))

    def test_writer_merges_rows_another_process_published_while_it_was_encoding(self):
        self.build("ab")
        test = self

        class Racing(FakeModel):
            def encode(self, texts, **kwargs):
                if not self.texts:
                    test.build("abyz")
                return super().encode(texts, **kwargs)

        index, _ = self.build("abxw", Racing())
        self.assert_rows_belong_to_their_keys("abxwyz")
        self.assertEqual(self.top(index, "x"), "x")

    def test_last_writer_wins_whole_when_both_publish_at_once(self):
        self.build("ab")
        real = mr_index._publish_vectors
        raced = []

        def publish(path, model_id, vectors):
            if not raced:
                raced.append(True)
                self.build("abyz")
            return real(path, model_id, vectors)

        with patch.object(mr_index, "_publish_vectors", side_effect=publish):
            self.build("abxw")
        self.assert_rows_belong_to_their_keys("abxw")
        index, model = self.build("abyz")
        self.assertEqual(len(model.texts), 2)
        self.assertEqual(self.top(index, "y"), "y")

    def test_blocked_replace_keeps_the_previous_version_and_the_index_usable(self):
        self.build("ab")
        with patch.object(mr_index.os, "replace", side_effect=PermissionError("in use")), \
             patch.object(mr_index.time, "sleep"):
            index, _ = self.build("abc")
        self.assertEqual(self.top(index, "c"), "c")
        self.assertEqual(set(self.stored()), {"key-a", "key-b"})
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["emb-fake.npz"])

    def test_another_model_or_version_in_the_file_is_ignored(self):
        self.build("ab")
        other = dict(self.spec, id="other/model")
        model = FakeModel()
        index = mr_index.VecIndex([chunk(n) for n in "ab"], model=model, spec=other)
        self.assertTrue(index.available, index.err)
        self.assertEqual(len(model.texts), 2)


if __name__ == "__main__":
    unittest.main()
