import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mr_corpus
import mr_index
import mr_tools


class ToolGuardTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for d in ("mem", "work"):
            (self.root/d).mkdir()
        self.tick = 0
        self.write("mem/good.md", "# 标题\n\n正文 检索 测试 内容 goodword\n")
        self.write("work/方案/plan.md", "## 方案\n供应商超时重试 planword。\n")
        self.write("work/报告/plan.md", "## 报告\n另一份同名文件 reportword。\n")
        self.log = self.root/"query-log.jsonl"
        self.enterContext(patch.object(mr_corpus, "SOURCES", [
            {"key": "mem", "dir": str(self.root/"mem"), "redact": True},
            {"key": "work", "dir": str(self.root/"work"), "redact": True, "recursive": True}]))
        self.enterContext(patch.object(mr_corpus, "LAZY_SOURCES", frozenset()))
        self.enterContext(patch.object(mr_corpus, "pdf_enabled", return_value=False))
        self.enterContext(patch.object(mr_corpus, "CORPUS_POLICY", str(self.root/"policy.json")))
        self.enterContext(patch.object(mr_tools, "_STATE", {}))
        self.enterContext(patch.object(mr_tools, "LOG_ON", True))
        self.enterContext(patch.object(mr_tools, "LOG_PATH", str(self.log)))
        self.enterContext(patch.object(mr_index, "CACHE", str(self.root)))
        self.enterContext(patch.object(mr_index.BM25, "_TOK_CACHE", {}))

    def write(self, name, text):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.tick += 1
        stamp = 1_700_000_000 + self.tick
        os.utime(path, (stamp, stamp))
        return path

    def search(self, query, **kwargs):
        return mr_tools.call("memory_search", {"query": query, "mode": "bm25", "rerank": False, **kwargs})

    def poll(self):
        mr_tools._STATE["sig_at"] = 0

    def records(self):
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]

    def test_misspelled_purpose_skips_that_file_and_keeps_search_working(self):
        self.assertTrue(self.search("goodword")["ok"])
        self.write("mem/bad.md", "---\nrag_purpose: evalution\n---\n\n# 另一篇\n\n别的内容 badword\n")
        self.poll()
        result = self.search("goodword")
        self.assertTrue(result["ok"], result)
        self.assertEqual([h["file"] for h in result["hits"]], ["good.md"])
        self.assertIn("bad.md", result["note"])
        self.assertEqual(self.search("badword")["hits"], [])
        self.assertEqual(self.search("badword", include_evaluation=True)["hits"], [])
        self.write("mem/bad.md", "---\nrag_purpose: evaluation\n---\n\n# 另一篇\n\n别的内容 badword\n")
        self.poll()
        fixed = self.search("badword", include_evaluation=True)
        self.assertEqual([h["file"] for h in fixed["hits"]], ["bad.md"])
        self.assertNotIn("bad.md", fixed["note"] or "")

    def test_misspelled_purpose_fails_closed_for_a_direct_load_and_for_get(self):
        self.write("mem/bad.md", "---\nrag_purpose: evalution\n---\n\n# 另一篇\n\n别的内容\n")
        with self.assertRaisesRegex(ValueError, "Unknown rag_purpose"):
            mr_corpus.load_chunks()
        got = mr_tools.call("memory_get", {"file": "bad.md", "source": "mem"})
        self.assertFalse(got["ok"])

    def test_broken_policy_keeps_the_previous_index_in_a_running_process(self):
        self.assertTrue(self.search("goodword")["ok"])
        (self.root/"policy.json").write_text('{"version": 2}', encoding="utf-8")
        self.poll()
        result = self.search("goodword")
        self.assertTrue(result["ok"], result)
        self.assertEqual([h["file"] for h in result["hits"]], ["good.md"])
        self.assertIn("重建失败", result["note"])
        (self.root/"policy.json").write_text('{"version": 1, "documents": []}', encoding="utf-8")
        self.poll()
        self.assertNotIn("重建失败", self.search("goodword")["note"] or "")

    def test_broken_policy_fails_closed_on_a_cold_start(self):
        (self.root/"policy.json").write_text('{"version": 2}', encoding="utf-8")
        self.assertFalse(self.search("goodword")["ok"])

    def test_long_full_text_is_capped_with_an_outline(self):
        sections = "".join("## 小节 %d\n%s\n\n" % (i, ("第 %d 节的内容。" % i) * 300) for i in range(12))
        self.write("mem/long.md", "# 长文\n\n开头。\n\n" + sections)
        with patch.object(mr_tools, "GET_MAX_CHARS", 5000):
            got = mr_tools.memory_get("long.md", source="mem")
            self.assertTrue(got["ok"])
            self.assertTrue(got["truncated"])
            self.assertEqual(len(got["text"]), 5000)
            self.assertGreater(got["total_chars"], 5000)
            self.assertIn("小节 7", got["headings"])
            part = mr_tools.memory_get("long.md", source="mem", heading="小节 7 (1)")
            self.assertTrue(part["ok"], part)
            self.assertNotIn("truncated", part)
            short = mr_tools.memory_get("good.md", source="mem")
            self.assertNotIn("truncated", short)

    def test_answer_check_still_reads_the_whole_source(self):
        self.write("mem/long.md", "# 长文\n\n" + "填充内容。" * 2000 + "\n\n结尾事实 tailfact 42。\n")
        with patch.object(mr_tools, "GET_MAX_CHARS", 5000):
            internal = mr_tools._memory_get("long.md", source="mem")
        self.assertIn("tailfact", internal["text"])
        self.assertNotIn("truncated", internal)

    def test_get_log_keeps_the_resolved_file_id(self):
        mr_tools.memory_get("方案/plan.md", source="work")
        mr_tools.memory_get("missing.md", source="work")
        rows = [r for r in self.records() if r["tool"] == "get"]
        self.assertEqual([(r["file"], r["ok"]) for r in rows], [("方案/plan.md", True), ("missing.md", False)])

    def test_search_log_does_not_keep_credentials_typed_into_a_query(self):
        self.search("为什么 password=Hunter2Example 登不上")
        row = [r for r in self.records() if r["tool"] == "search"][-1]
        self.assertNotIn("Hunter2Example", json.dumps(row, ensure_ascii=False))
        self.assertIn("为什么", row["q"])


if __name__ == "__main__":
    unittest.main()
