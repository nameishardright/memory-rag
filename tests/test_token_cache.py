"""BM25 token cache survives index rebuilds: a hot reload only tokenizes changed chunks."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import mr_corpus
import mr_index
import mr_tools


class TokenCacheTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for d in ("mem", "work"):
            (self.root/d).mkdir()
        for i in range(6):
            (self.root/"mem"/("note-%d.md" % i)).write_text("## 复盘 %d\n直合之前先核对止损。\n" % i, encoding="utf-8")
        # Built after the notes index and much smaller: it used to wipe the shared cache.
        (self.root/"work"/"plan.md").write_text("## 方案\n供应商超时重试。\n", encoding="utf-8")
        self.enterContext(patch.object(mr_corpus, "SOURCES", [
            {"key": "mem", "dir": str(self.root/"mem"), "redact": True},
            {"key": "work", "dir": str(self.root/"work"), "redact": True}]))
        self.enterContext(patch.object(mr_corpus, "pdf_enabled", return_value=False))
        self.enterContext(patch.object(mr_corpus, "CORPUS_POLICY", str(self.root/"policy.json")))
        self.enterContext(patch.object(mr_tools, "_STATE", {}))
        self.enterContext(patch.object(mr_tools, "LOG_ON", False))
        self.enterContext(patch.object(mr_index.BM25, "_TOK_CACHE", {}))

    def test_hot_reload_tokenizes_only_the_changed_note(self):
        mr_tools._ensure("bm25")
        (self.root/"mem"/"note-0.md").write_text("## 复盘 0\n直合之前先核对入场区。\n", encoding="utf-8")
        with patch.object(mr_index, "tokenize", wraps=mr_index.tokenize) as tok:
            mr_tools._rebuild("test")
        indexed = {c.args[0].split(" ", 1)[0] for c in tok.call_args_list
                   if c.args[0].startswith(("note-", "plan.md"))}
        self.assertEqual(indexed, {"note-0.md"})

    def test_cache_drops_chunks_that_left_the_corpus(self):
        mr_tools._ensure("bm25")
        (self.root/"mem"/"note-5.md").unlink()
        mr_tools._rebuild("test")
        self.assertEqual(len(mr_index.BM25._TOK_CACHE), 6)   # 5 notes + 1 work plan


if __name__ == "__main__":
    unittest.main()
