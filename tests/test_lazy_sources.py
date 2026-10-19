"""On-demand sources: not loaded at startup, loaded by the first query that names them."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import mr_corpus
import mr_index
import mr_tools


class LazySourceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for d in ("mem", "stock", "work"):
            (self.root/d).mkdir()
        (self.root/"mem"/"note.md").write_text("## 复盘\n直合之前先核对止损和入场区。\n", encoding="utf-8")
        for i in range(5):
            (self.root/"mem"/("filler-%d.md" % i)).write_text("## 杂记 %d\n周报每条一行带量化。\n" % i, encoding="utf-8")
        # The work sub-index is built after the notes one and is the smallest.
        (self.root/"work"/"plan.md").write_text("## 方案\n供应商超时重试。\n", encoding="utf-8")
        for name, body in (("card_UBER#mid_2026-09-17.md", "入场 68-70，止损 59。"),
                           ("card_NVDA#short_2026-09-18.md", "站上 190 再买，止损 176。")):
            (self.root/"stock"/name).write_text("## 判断\n" + body, encoding="utf-8")
        self.enterContext(patch.object(mr_corpus, "SOURCES", [
            {"key": "mem", "dir": str(self.root/"mem"), "redact": True},
            {"key": "stock", "dir": str(self.root/"stock"), "redact": True},
            {"key": "work", "dir": str(self.root/"work"), "redact": True}]))
        self.enterContext(patch.object(mr_index.BM25, "_TOK_CACHE", {}))
        self.enterContext(patch.object(mr_corpus, "LAZY_SOURCES", frozenset({"stock"})))
        self.enterContext(patch.object(mr_corpus, "pdf_enabled", return_value=False))
        self.enterContext(patch.object(mr_corpus, "CORPUS_POLICY", str(self.root/"policy.json")))
        self.enterContext(patch.object(mr_tools, "_STATE", {}))
        self.enterContext(patch.object(mr_tools, "LOG_ON", False))

    def sources(self):
        return {c["source"] for c in mr_tools._STATE["chunks"]}

    def rows(self, result):
        return [(h["source"], h["file"], h["heading"], h["score"]) for h in result["hits"]]

    def test_stock_waits_for_the_first_stock_query(self):
        mr_tools.memory_search("止损", mode="bm25", rerank=False)
        self.assertEqual(self.sources(), {"mem", "work"})
        result = mr_tools.memory_search("UBER 止损", source="stock", mode="bm25", rerank=False)
        self.assertEqual(result["hits"][0]["file"], "card_UBER#mid_2026-09-17.md")
        self.assertEqual(self.sources(), {"mem", "stock", "work"})

    def test_first_stock_query_in_a_fresh_process_loads_once(self):
        with patch.object(mr_tools, "_rebuild", wraps=mr_tools._rebuild) as rebuild:
            mr_tools.memory_search("UBER 止损", source="stock", mode="bm25", rerank=False)
            mr_tools.memory_search("NVDA 站上", source="stock", mode="bm25", rerank=False)
        self.assertEqual(rebuild.call_count, 1)

    def test_unloaded_stock_rewrites_do_not_change_the_live_fingerprint(self):
        before_live, before_all = mr_corpus.signature(skip={"stock"}), mr_corpus.signature()
        (self.root/"stock"/"card_UBER#mid_2026-09-17.md").write_text("## 判断\n入场 60-62，止损 55。\n", encoding="utf-8")
        self.assertEqual(before_live, mr_corpus.signature(skip={"stock"}))
        self.assertNotEqual(before_all, mr_corpus.signature())

    def test_loading_stock_does_not_retokenize_the_notes(self):
        mr_tools._ensure("bm25")
        with patch.object(mr_index, "tokenize", wraps=mr_index.tokenize) as tok:
            mr_tools.memory_search("UBER 止损", source="stock", mode="bm25", rerank=False)
        texts = [call.args[0] for call in tok.call_args_list]
        self.assertTrue(any(t.startswith("card_") for t in texts))
        self.assertFalse(any(t.startswith(("note.md", "filler-", "plan.md")) for t in texts))

    def test_results_match_loading_everything_at_startup(self):
        queries = [("止损", None), ("UBER 止损", "stock"), ("站上 再买", "stock")]
        lazy = [self.rows(mr_tools.memory_search(q, source=s, mode="bm25", rerank=False)) for q, s in queries]
        mr_tools._STATE.clear()
        with patch.object(mr_corpus, "LAZY_SOURCES", frozenset()):
            eager = [self.rows(mr_tools.memory_search(q, source=s, mode="bm25", rerank=False)) for q, s in queries]
        self.assertEqual(lazy, eager)
        self.assertTrue(all(lazy))


if __name__ == "__main__":
    unittest.main()
