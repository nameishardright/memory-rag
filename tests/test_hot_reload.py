import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mr_corpus
import mr_index
import mr_tools


class HotReloadTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.tick = 0
        for d in ("mem", "work", "pdf"):
            (self.root/d).mkdir()
        for i in range(4):
            self.write("mem/note-%d.md" % i, "## 复盘 %d\n直合之前先核对止损 marker%d。\n" % (i, i))
        self.write("work/old/readme.md", "## 方案\n供应商超时重试 shared。\n")
        self.write("work/new/readme.md", "## 方案\n供应商超时重试 shared。\n")
        self.policy = self.root/"policy.json"
        self.enterContext(patch.object(mr_corpus, "SOURCES", [
            {"key": "mem", "dir": str(self.root/"mem"), "redact": True},
            {"key": "work", "dir": str(self.root/"work"), "redact": True, "recursive": True}]))
        self.enterContext(patch.object(mr_corpus, "LAZY_SOURCES", frozenset()))
        self.enterContext(patch.object(mr_corpus, "pdf_enabled", return_value=False))
        self.enterContext(patch.object(mr_corpus, "CORPUS_POLICY", str(self.policy)))
        self.enterContext(patch.object(mr_tools, "_STATE", {}))
        self.enterContext(patch.object(mr_tools, "LOG_ON", False))
        self.enterContext(patch.object(mr_index.BM25, "_TOK_CACHE", {}))

    def write(self, name, text):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.tick += 1
        stamp = 1_700_000_000 + self.tick
        os.utime(path, (stamp, stamp))
        return path

    def poll(self):
        mr_tools._STATE["sig_at"] = 0
        return mr_tools._ensure("bm25")

    def found(self, word):
        return [h["file"] for h in mr_tools.memory_search(word, mode="bm25", rerank=False)["hits"]]

    def assert_same_as_a_full_load(self):
        self.assertEqual(self.poll()["chunks"], mr_corpus.load_chunks())

    def test_edit_that_lands_while_the_corpus_is_being_read_is_picked_up_by_the_next_poll(self):
        real = mr_corpus.load_chunks

        def racing(**kwargs):
            chunks = real(**kwargs)
            self.write("mem/note-0.md", "## 复盘 0\n直合之前先核对入场区 lateword。\n")
            return chunks

        with patch.object(mr_corpus, "load_chunks", side_effect=racing):
            mr_tools._ensure("bm25")
        self.assertEqual(self.found("lateword"), [])
        self.poll()
        self.assertEqual(self.found("lateword"), ["note-0.md"])
        self.assertEqual(self.found("marker0"), [])

    def test_hot_reload_reads_only_the_changed_file(self):
        mr_tools._ensure("bm25")
        self.write("mem/note-2.md", "## 复盘 2\n改过的内容 freshword。\n")
        with patch.object(mr_corpus, "_file_blocks", wraps=mr_corpus._file_blocks) as read, \
             patch.object(mr_corpus, "redact", wraps=mr_corpus.redact) as redact:
            self.poll()
        self.assertEqual([call.args[1] for call in read.call_args_list], ["note-2.md"])
        self.assertEqual(redact.call_count, 2)
        self.assertEqual(self.found("freshword"), ["note-2.md"])

    def test_unchanged_corpus_is_not_reread_when_only_the_policy_changes(self):
        mr_tools._ensure("bm25")
        self.policy.write_text(json.dumps({"version": 1, "documents": [
            {"source": "mem", "file": "note-1.md", "purpose": "evaluation"}]}), encoding="utf-8")
        with patch.object(mr_corpus, "_file_blocks", wraps=mr_corpus._file_blocks) as read:
            state = self.poll()
        read.assert_not_called()
        self.assertEqual({c["purpose"] for c in state["chunks"] if c["file"] == "note-1.md"}, {"evaluation"})
        self.assertEqual(self.found("marker1"), [])
        self.assert_same_as_a_full_load()

    def test_edits_additions_and_deletions_match_a_full_load(self):
        mr_tools._ensure("bm25")
        self.write("mem/note-1.md", "---\nrag_purpose: evaluation\n---\n## 复盘 1\n评测复盘。\n")
        self.write("mem/added.md", "# 新笔记\n\n新增的一段 addedword。\n\n## 小节\n第二段。\n")
        (self.root/"mem"/"note-3.md").unlink()
        self.assert_same_as_a_full_load()
        self.assertEqual(self.found("addedword"), ["added.md"])
        self.assertEqual(self.found("marker3"), [])
        self.assertNotIn(("mem", "note-3.md"), mr_tools._STATE["files"])

    def test_duplicate_paragraph_follows_the_newest_copy(self):
        state = mr_tools._ensure("bm25")
        self.assertEqual([c["file"] for c in state["chunks"] if c["source"] == "work"], ["new/readme.md"])
        self.write("work/old/readme.md", "## 方案\n供应商超时重试 shared。\n")
        state = self.poll()
        self.assertEqual([c["file"] for c in state["chunks"] if c["source"] == "work"], ["old/readme.md"])
        self.assert_same_as_a_full_load()

    def test_pdf_pages_are_processed_once_per_file_version(self):
        pdf = self.root/"pdf"/"book.pdf"
        pdf.write_bytes(b"one")
        pages = ["教材演示内容。" * 10, "第二页的内容 pageword。" * 8]
        with patch.object(mr_corpus, "pdf_enabled", return_value=True), \
             patch.object(mr_corpus, "pdf_files", return_value=[str(pdf)]), \
             patch.object(mr_corpus, "extract_pdf", return_value=pages) as extract:
            mr_tools._ensure("bm25")
            self.write("mem/note-0.md", "## 复盘 0\n别的内容。\n")
            state = self.poll()
            self.assertEqual(extract.call_count, 1)
            self.assertEqual([c["heading"] for c in state["chunks"] if c["source"] == "pdf"], ["p1", "p2"])
            pdf.write_bytes(b"second version")
            self.poll()
            self.assertEqual(extract.call_count, 2)
            self.assertEqual(state["chunks"][-2:], mr_corpus.load_pdf_chunks())


if __name__ == "__main__":
    unittest.main()
