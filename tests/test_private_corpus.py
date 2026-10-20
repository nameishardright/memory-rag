import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import mr_private as private


class PrivateCorpusTests(unittest.TestCase):
    PASSWORD = "synthetic-test-password"
    SECRET = "SyntheticSecret-Only-For-Tests-7183"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "sources"
        self.source.mkdir()
        self.store = self.root / "private" / "corpus.enc"

    def tearDown(self):
        self.temp.cleanup()

    def journal(self, name, rows, tail=b""):
        salt = os.urandom(16)
        aes = AESGCM(private.key_for(self.PASSWORD, salt))
        data = private.JOURNAL_MAGIC + salt
        for row in rows:
            nonce = os.urandom(12)
            blob = nonce + aes.encrypt(nonce, zlib.compress(json.dumps(row).encode()), private.JOURNAL_MAGIC)
            data += struct.pack("<I", len(blob)) + blob
        path = self.source / name
        path.write_bytes(data + tail)
        return path

    def rows(self):
        files = [{"path": "family.txt", "size": 100, "mtime_ns": 20},
                 {"path": "work/example.js", "size": 200, "mtime_ns": 30},
                 {"path": "cloud.pdf", "size": 300, "mtime_ns": 40}]
        inventory = {"record_type": "inventory", "time": "2026-01-01T00:00:00+00:00", "files": files}
        family = {**files[0], "record_type": "file", "read_ok": True, "method": "text",
                  "text": "# Family\n\n家庭旅行 Montreal password: " + self.SECRET,
                  "sha256": "synthetic-original-hash", "extractor_version": 1}
        technical = {**files[1], "record_type": "file", "read_ok": True, "method": "text",
                     "text": "Montreal technical fixture", "dependency": False}
        return inventory, family, technical

    def test_encryption_roundtrip_wrong_password_and_tampering(self):
        payload = {"version": 1, "text": self.SECRET}
        private.save_store(self.store, self.PASSWORD, payload)
        self.assertEqual(private.load_store(self.store, self.PASSWORD), payload)
        self.assertNotIn(self.SECRET.encode(), self.store.read_bytes())
        with self.assertRaises(InvalidTag):
            private.load_store(self.store, "wrong-password")
        damaged = bytearray(self.store.read_bytes())
        damaged[-1] ^= 1
        self.store.write_bytes(damaged)
        with self.assertRaises(InvalidTag):
            private.load_store(self.store, self.PASSWORD)

    def test_concurrent_partial_tail_and_corrupt_frame(self):
        path = self.journal("extraction.enc", [{"ok": True}], b"\x80\x00")
        self.assertEqual(list(private.read_journal(path, self.PASSWORD)), [{"ok": True}])
        path = self.journal("extraction.enc", [{"ok": True}], struct.pack("<I", 100) + b"partial")
        self.assertEqual(len(list(private.read_journal(path, self.PASSWORD))), 1)
        path = self.journal("extraction.enc", [{"ok": True}])
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with self.assertRaises(InvalidTag):
            list(private.read_journal(path, self.PASSWORD))

    def test_import_preserves_values_and_latest_refinement(self):
        inv, family, technical = self.rows()
        self.journal("extraction.enc", [inv, family, technical])
        refined = {**family, "text": family["text"] + "\nVerified refinement.", "extractor_version": 2}
        self.journal("refined.enc", [refined])
        self.journal("core.enc", [family])  # older cached core must not override refinement
        payload = private.import_journals(self.source, self.store, self.PASSWORD)
        restored = private.load_store(self.store, self.PASSWORD)
        doc = next(d for d in restored["documents"] if d["file"] == "family.txt")
        self.assertEqual(doc["text"], refined["text"])
        self.assertEqual(doc["text_sha256"], hashlib.sha256(refined["text"].encode()).hexdigest())
        self.assertEqual(payload["coverage"]["not_processed"], 1)
        self.assertEqual(payload["coverage"]["personal_documents"], 1)
        self.assertEqual(payload["coverage"]["technical_documents"], 1)
        self.assertEqual(payload["coverage"]["source_status"], "incomplete")
        for path in self.store.parent.iterdir():
            self.assertNotIn(self.SECRET.encode(), path.read_bytes())
            self.assertNotIn(b"family.txt", path.read_bytes())

    def test_deleted_and_stale_sources_are_not_indexed(self):
        inv, family, technical = self.rows()
        self.journal("bulk.enc", [inv, family, technical])
        new = {**inv, "time": "2026-01-02T00:00:00+00:00", "files": [{**inv["files"][0], "mtime_ns": 999}]}
        self.journal("extraction.enc", [new])
        payload = private.import_journals(self.source, self.store, self.PASSWORD)
        self.assertEqual(payload["documents"], [])
        self.assertEqual(payload["coverage"]["stale_records_excluded"], 1)

    def test_retrieval_full_source_and_scope_do_not_write_files(self):
        inv, family, technical = self.rows()
        self.journal("extraction.enc", [inv, family, technical])
        payload = private.import_journals(self.source, self.store, self.PASSWORD)
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        rag = private.PrivateRAG(payload)
        result = rag.search("Montreal")
        self.assertEqual(len(result["hits"]), 1)
        hit = result["hits"][0]
        self.assertIn(self.SECRET, hit["text"])
        self.assertEqual(rag.get(hit["document_id"])["text"], family["text"])
        self.assertEqual(len(rag.search("Montreal", scope="all")["hits"]), 2)
        self.assertEqual(rag.search("AbsentUniqueTerm991912")["hits"], [])
        self.assertFalse(rag.get("../../arbitrary-file")["ok"])
        after = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_moved_archive_keeps_only_current_member_sources(self):
        original = {'path': 'old/archive.zip', 'size': 50, 'mtime_ns': 42,
                    'record_type': 'file', 'read_ok': True, 'text': ''}
        moved = {**original, 'path': 'organized/archive.zip'}
        member = {'record_type': 'archive_member', 'path': 'old/archive.zip!/note.txt',
                  'archive': original['path'], 'read_ok': True, 'text': 'Synthetic archived family note'}
        relocated = {**member, 'path': 'organized/archive.zip!/note.txt', 'archive': moved['path']}
        inventory = {'record_type': 'inventory', 'time': '2026-01-02', 'files': [moved]}
        self.journal('extraction.enc', [inventory, original, member, moved, relocated])
        payload = private.import_journals(self.source, self.store, self.PASSWORD)
        self.assertEqual([doc['file'] for doc in payload['documents']], [relocated['path']])

    def test_store_cannot_be_written_into_git_checkout(self):
        with self.assertRaises(ValueError):
            private.import_journals(self.source, Path(private.__file__).parent / "private.enc", self.PASSWORD)

    def test_failed_reimport_keeps_previous_ciphertext(self):
        inv, family, technical = self.rows()
        self.journal("extraction.enc", [inv, family, technical])
        private.import_journals(self.source, self.store, self.PASSWORD)
        original = self.store.read_bytes()
        with self.assertRaises(InvalidTag):
            private.import_journals(self.source, self.store, "incorrect")
        self.assertEqual(self.store.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
