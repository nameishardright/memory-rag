"""Small, offline regressions at the search/tool boundary."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mr_corpus
import mr_index
import mr_tools


def chunk(source, file, heading, text, purpose="knowledge"):
    return {"id": hashlib.md5((source + file + heading).encode()).hexdigest()[:12],
            "source": source, "file": file, "heading": heading, "text": text,
            "md5": mr_corpus.chunk_key(file, heading, text), "purpose": purpose}


class SearchContractTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(mr_tools, "_STATE", {}))
        self.enterContext(patch.object(mr_tools, "LOG_ON", False))
        self.enterContext(patch.object(mr_tools, "GATE_MODE", "note"))
        self.enterContext(patch.object(mr_corpus, "signature", return_value="fixture"))

    def search(self, chunks, **kwargs):
        mr_tools._STATE.clear()
        with patch.object(mr_corpus, "load_chunks", return_value=chunks):
            return mr_tools.memory_search("cherry-pick 直合", mode="bm25", rerank=False, **kwargs)

    def test_stock_and_pdf_do_not_change_default_note_scores(self):
        notes = [chunk("mem", "workflow.md", "直接合入", "cherry-pick 直合 步骤"),
                 chunk("book", "unrelated.md", "分词", "文本 切分")]
        stocks = [chunk("stock", "card-%d.md" % i, "历史判断", "直合 复盘 盈利" * 20)
                  for i in range(20)]
        before = self.search(notes)
        foreign = stocks + [chunk("pdf", "textbook.pdf", "p%d" % i, "cherry-pick 直合" * 20) for i in range(10)]
        after = self.search(notes + foreign)
        self.assertEqual(before["hits"], after["hits"], "Stock/PDF statistics leaked into default note scores")

    def test_stock_keeps_its_existing_non_pdf_statistics(self):
        chunks = [chunk("mem", "workflow.md", "直接合入", "cherry-pick 直合 步骤"),
                  chunk("stock", "card.md", "历史判断", "cherry-pick 直合 复盘"),
                  chunk("pdf", "textbook.pdf", "p2", "直合" * 100)]
        legacy = mr_tools._SubBM25(chunks, [0, 1]).search("cherry-pick 直合", mask={1})
        result = self.search(chunks, source="stock")
        self.assertEqual(result["hits"][0]["score"], round(legacy[0][1], 4))

    def test_adding_pdf_does_not_opt_into_stock(self):
        chunks = [chunk(s, s + ".md", "直合", "cherry-pick 直合") for s in ("mem", "stock", "pdf")]
        result = self.search(chunks, include_pdf=True)
        self.assertEqual({h["source"] for h in result["hits"]}, {"mem", "pdf"})

    def test_knowledge_mentioning_tests_is_not_automatically_excluded(self):
        chunks = [chunk("book", "lesson.md", "直合测试", "cherry-pick 直合可以做集成测试")]
        self.assertEqual(self.search(chunks)["hits"][0]["file"], "lesson.md")

    def test_evaluation_documents_are_opt_in(self):
        chunks = [chunk("case", "evaluation.md", "阴性问题误命中", "cherry-pick 直合 只是测试问题", "evaluation")]
        result = self.search(chunks)
        self.assertEqual(result["hits"], [], "An evaluation discussion was returned as ordinary evidence")
        self.assertEqual(result["confidence"], "low")
        result = self.search(chunks, include_evaluation=True)
        self.assertEqual(result["hits"][0]["file"], "evaluation.md")
        self.assertEqual(result["hits"][0]["purpose"], "evaluation")

    def test_low_scoring_representative_checks_the_same_files_lexical_answer(self):
        chunks = [chunk("case", "memory.md", "记忆纠错回路", "记忆越攒越脏时，写入、更新、删错三个动作都要有。"),
                  chunk("case", "memory.md", "记忆的成本", "记忆索引的大小与正文加载成本。")]
        vector = SimpleNamespace(available=True, search=lambda query, **kwargs: [(1, .8)])
        reranker = SimpleNamespace(available=True, spec={"key": "fixture", "hi": .5, "lo": .1},
                                  score=lambda query, passages: [.9 if "删错" in p else .01 for p in passages])
        with patch.object(mr_corpus, "load_chunks", return_value=chunks), patch.object(mr_tools, "_get_reranker", return_value=reranker), patch.object(mr_tools, "RERANK_MODE", "gate"):
            state = mr_tools._ensure("bm25")
            state["vecs"] = {mr_index.model_spec()["slug"]: vector}
            result = mr_tools.memory_search("记忆越攒越脏", rerank=True)
        self.assertEqual(result["confidence"], "high")
        self.assertIn("删错", result["hits"][0]["snippet"])
        self.assertEqual(result["hits"][0]["rerank"], .9)

    def test_relevant_answer_is_not_replaced_by_a_higher_scoring_overview(self):
        chunks = [chunk("book", "mcp.md", "MCP 方法概览", "MCP 方法 MCP 方法 MCP 方法，本章概览。"),
                  chunk("book", "mcp.md", "实现步骤", "MCP 方法是 initialize、tools/list、tools/call。")]
        vector = SimpleNamespace(available=True, search=lambda query, **kwargs: [(1, .8)])
        reranker = SimpleNamespace(available=True, spec={"key": "fixture", "hi": .5, "lo": .1},
                                  score=lambda query, passages: [.2 if "tools/call" in p else .9 for p in passages])
        with patch.object(mr_corpus, "load_chunks", return_value=chunks), patch.object(mr_tools, "_get_reranker", return_value=reranker), patch.object(mr_tools, "RERANK_MODE", "gate"):
            state = mr_tools._ensure("bm25")
            state["vecs"] = {mr_index.model_spec()["slug"]: vector}
            result = mr_tools.memory_search("MCP 方法", rerank=True)
        self.assertIn("tools/call", result["hits"][0]["snippet"])
        self.assertEqual(result["confidence"], "medium")

    def _budget_result(self, answer_index, topk=5):
        chunks = [chunk("case", "file-%d.md" % i,"Body","correct-marker" if i==answer_index else "background")
                  for i in range(6)]
        order = [(i,10-i) for i in range(6)]
        vector = SimpleNamespace(available=True,search=lambda *a,**kw:[(i,.8-i*.02) for i in range(6)])
        calls=[]
        def score(query,passages):
            calls.extend(passages)
            return [.9 if "correct-marker" in p else .01 for p in passages]
        rk=SimpleNamespace(available=True,spec={"key":"fixture","hi":.5,"lo":.1,"gate_candidates":3,
                           "fallback_candidates":5,"max_windows":2,"max_length":256},score=score)
        with patch.object(mr_corpus,"load_chunks",return_value=chunks), \
                patch.object(mr_tools,"_get_reranker",return_value=rk), patch.object(mr_tools,"RERANK_MODE","gate"):
            state=mr_tools._ensure("bm25")
            state["bm25"]=SimpleNamespace(search=lambda *a,**kw:order)
            state["vecs"]={mr_index.model_spec()["slug"]:vector}
            result=mr_tools.memory_search("fixture query",topk=topk,rerank=True)
        return result,calls

    def test_weak_first_three_expand_to_the_already_returned_fifth(self):
        result,calls=self._budget_result(4)
        self.assertEqual(result["confidence"],"high")
        self.assertEqual(result["rerank"]["expanded_candidates"],2)
        self.assertEqual(result["rerank"]["n"],5)
        self.assertEqual(len(calls),5)
        self.assertEqual([h["file"] for h in result["hits"]],["file-%d.md"%i for i in range(5)])

    def test_strong_initial_candidate_does_not_pay_for_two_more(self):
        result,calls=self._budget_result(0)
        self.assertEqual(result["confidence"],"high")
        self.assertEqual(result["rerank"]["expanded_candidates"],0)
        self.assertEqual(len(calls),3)

    def test_unreturned_candidate_cannot_raise_confidence(self):
        result,calls=self._budget_result(4,topk=2)
        self.assertEqual(result["confidence"],"low")
        self.assertEqual(result["rerank"]["n"],2)
        self.assertEqual(len(calls),2)


class FusionContractTests(unittest.TestCase):
    def test_body_replaces_navigation_without_changing_file_ranks(self):
        chunks = [chunk("mem", "workflow.md", "索引行归档(旧索引)", "cherry-pick 直合 看正文"),
                  chunk("mem", "workflow.md", "直合步骤", "在最新 release 上 cherry-pick，保留 Change-Id，再 push %submit。"),
                  chunk("book", "other.md", "其他资料", "相关背景")]
        # BM25 finds the index line first; dense retrieval finds the actual answer.
        legs = [[(0, 12), (2, 5), (1, 4)], [(1, .8), (2, .6)]]
        ranked = mr_index.rrf_files(chunks, legs)
        self.assertEqual([chunks[r[0]]["file"] for r in ranked], ["workflow.md", "other.md"])
        self.assertEqual(ranked[0][0], 1, "The correct file still represents itself with an index line")
        self.assertAlmostEqual(ranked[0][1], 2 / 61)
        self.assertEqual(ranked[0][2], [0, 0])

    def test_navigation_is_still_available_when_it_is_the_only_candidate(self):
        chunks = [chunk("mem", "workflow.md", "索引行归档", "唯一的定位信息")]
        self.assertEqual(mr_index.rrf_files(chunks, [[(0, 9)], []])[0][0], 0)


class ToolDeclarationTests(unittest.TestCase):
    def test_stock_is_declared_by_both_tools(self):
        for name in ("memory_search", "memory_get"):
            with self.subTest(tool=name):
                self.assertIn("stock", mr_tools.REGISTRY[name]["schema"]["properties"]["source"]["enum"])


class CorpusPolicyTests(unittest.TestCase):
    def test_pdf_labels_apply_to_one_page_without_matching_page_twenty(self):
        policy = {"version": 1, "documents": [], "pdf_pages": [
            {"source": "pdf", "file": "guide.pdf", "heading_prefix": "p2", "purpose": "evaluation"}]}
        self.policy.write_text(json.dumps(policy), encoding="utf-8")
        pages = ["这是教材正文，保留足够字符以被抽取。" * 4 for _ in range(20)]
        pdf = self.root / "guide.pdf"
        pdf.touch()
        with patch.object(mr_corpus, "pdf_files", return_value=[str(pdf)]), patch.object(mr_corpus, "extract_pdf", return_value=pages), patch.object(mr_corpus, "PDF_DIR", str(self.root)):
            chunks = mr_corpus.load_pdf_chunks()
            by_heading = {c["heading"]: c["purpose"] for c in chunks}
            self.assertEqual(by_heading["p2"], "evaluation")
            self.assertEqual(by_heading["p20"], "knowledge")
            self.assertEqual(mr_tools.memory_get("guide.pdf#p2", source="pdf")["purpose"], "evaluation")

    def test_unique_heading_prefix_returns_the_resolved_heading(self):
        (self.root / "lesson.md").write_text("# 操作步骤  (1)\n完整的步骤。", encoding="utf-8")
        result = mr_tools.memory_get("lesson.md", source="book", heading="操作步骤")
        self.assertTrue(result["ok"])
        self.assertEqual(result["heading"], "操作步骤  (1)")
        self.assertEqual(result["text"], "完整的步骤。")

    def test_ambiguous_heading_prefix_lists_candidates_instead_of_truncating(self):
        (self.root / "lesson.md").write_text("# 操作步骤 (1)\n第一段。\n# 操作步骤 (2)\n第二段。", encoding="utf-8")
        result = mr_tools.memory_get("lesson.md", source="book", heading="操作步骤")
        self.assertFalse(result["ok"])
        self.assertEqual(result["candidates"], ["操作步骤 (1)", "操作步骤 (2)"])
        self.assertNotIn("text", result)

    def test_exact_heading_wins_over_longer_prefix_matches(self):
        (self.root / "lesson.md").write_text("# 操作步骤\n准确段落。\n# 操作步骤补充\n另一个段落。", encoding="utf-8")
        result = mr_tools.memory_get("lesson.md", source="book", heading="操作步骤")
        self.assertEqual(result["text"], "准确段落。")

    def test_missing_heading_offers_available_headings(self):
        (self.root / "lesson.md").write_text("# 操作步骤\n正文。", encoding="utf-8")
        result = mr_tools.memory_get("lesson.md", source="book", heading="不存在")
        self.assertFalse(result["ok"])
        self.assertEqual(result["candidates"], ["操作步骤"])

    def test_frontmatter_followed_by_body_and_a_rule_is_still_body(self):
        c = chunk("mem", "note.md", "(开头)", "---\ndescription: 维护规则\n---\n写入、更新、删错。\n---")
        self.assertEqual(mr_corpus.evidence_kind(c), "body")
        c["text"] = "---\ndescription: 维护规则\n---"
        self.assertEqual(mr_corpus.evidence_kind(c), "navigation")

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.policy = self.root / "policy.json"
        self.enterContext(patch.object(mr_corpus, "CORPUS_POLICY", str(self.policy)))
        self.enterContext(patch.object(mr_corpus, "SOURCES", [{"key": "book", "dir": str(self.root), "redact": False}]))
        self.enterContext(patch.object(mr_corpus, "pdf_enabled", return_value=False))
        self.enterContext(patch.object(mr_tools, "LOG_ON", False))

    def test_only_labeled_report_sections_are_excluded(self):
        (self.root / "lesson.md").write_text("# 接入步骤\n执行工具协议。\n## 实测报告\n问题命中过一次。", encoding="utf-8")
        policy = {"version": 1, "documents": [], "sections": [
            {"source": "book", "file": "lesson.md", "heading_prefix": "实测报告", "purpose": "evaluation"}]}
        self.policy.write_text(json.dumps(policy), encoding="utf-8")
        chunks = mr_corpus.load_chunks()
        self.assertEqual([(c["heading"], c["purpose"]) for c in chunks], [("接入步骤", "knowledge"), ("实测报告", "evaluation")])
        result = mr_tools.memory_get("lesson.md", source="book", heading="实测报告")
        self.assertEqual(result["purpose"], "evaluation")
        self.assertEqual(result["text"], "问题命中过一次。")
        result = mr_tools.memory_get("lesson.md", source="book", heading="接入步骤")
        self.assertEqual(result["purpose"], "knowledge")
        self.assertNotIn("命中过", result["text"])
        self.assertFalse(mr_tools.memory_get("lesson.md", source="book", heading="不存在")["ok"])

    def test_policy_edits_trigger_the_existing_reload_fingerprint(self):
        self.policy.write_text('{"version":1,"documents":[]}', encoding="utf-8")
        before = mr_corpus.signature()
        self.policy.write_text('{"version":1,"documents":[],"sections":[]}', encoding="utf-8")
        self.assertNotEqual(before, mr_corpus.signature())

    def test_new_evaluation_notes_can_label_their_frontmatter(self):
        (self.root / "run.md").write_text("---\nrag_purpose: evaluation\n---\n# 分析\n本轮得分。", encoding="utf-8")
        self.assertEqual({c["purpose"] for c in mr_corpus.load_chunks()}, {"evaluation"})


if __name__ == "__main__":
    unittest.main()
