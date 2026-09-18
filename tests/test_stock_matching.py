"""Stock card identity through the public search/get boundary."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import mr_corpus
import mr_index
import mr_tools


class StockMatchingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.tmp)
        self.enterContext(patch.object(mr_corpus, "SOURCES", [
            {"key": "stock", "dir": str(self.root), "redact": True}]))
        self.enterContext(patch.object(mr_corpus, "pdf_enabled", return_value=False))
        self.enterContext(patch.object(mr_corpus, "CORPUS_POLICY", str(self.root/"policy.json")))
        self.enterContext(patch.object(mr_tools, "_STATE", {}))
        self.enterContext(patch.object(mr_tools, "LOG_ON", False))

    def card(self, name, body="入场 68-70，止损 59。"):
        (self.root/name).write_text("## 判断\n" + body, encoding="utf-8")

    def search(self, query, **kw):
        return mr_tools.memory_search(query, source="stock", mode="bm25", rerank=False, **kw)

    def test_explicit_card_identity_does_not_return_other_tickers_dates_or_horizons(self):
        wanted = "card_UBER#mid_2026-09-17.md"
        for name in [wanted, "card_UBER#short_2026-09-17.md",
                     "card_UBER#mid_2026-09-16.md", "card_NVDA#mid_2026-09-17.md"]:
            self.card(name)
        result = self.search("UBER 2026-09-17 中线卡的入场区和止损是多少？")
        self.assertEqual([h["file"] for h in result["hits"]], [wanted])
        self.assertEqual(result["hits"][0]["stock"],
                         {"ticker": "UBER", "card_date": "2026-09-17", "horizon": "mid"})

    def test_filename_only_ticker_is_searchable_without_narrowing_analogies(self):
        self.card("card_ZZZZ#short_2026-09-17.md")
        self.card("card_ABCD#short_2026-09-17.md")
        result = self.search("ZZZZ")
        self.assertEqual([h["file"] for h in result["hits"]], ["card_ZZZZ#short_2026-09-17.md"])
        self.assertEqual(result["stock_filter"], {})

    def test_plan_abbreviations_return_the_numeric_passage_not_in_transit(self):
        name = "card_TEST#short_2026-09-17.md"
        (self.root/name).write_text("## 判断\n三个数:入 40-41 / 损 38;当时价 43\n\n## 结果\n在途\n", encoding="utf-8")
        result = self.search("TEST 2026-09-17 短线卡历史参考入场区与止损位")
        self.assertEqual(result['hits'][0]['heading'], '判断')
        self.assertIn('40-41', result['hits'][0]['snippet'])

    def test_legacy_mid_marker_and_hong_kong_ticker_are_identity_fields(self):
        name = "card_0005.HK#MID#mid_2026-09-17.md"
        self.card(name)
        result = self.search("0005.hk 2026年9月17日 中期卡")
        self.assertEqual(result["hits"][0]["stock"]["ticker"], "0005.HK")
        fetched = mr_tools.memory_get(name, source="stock", heading="判断")
        self.assertEqual(fetched["stock"], result["hits"][0]["stock"])

    def test_missing_exact_date_is_not_replaced_by_nearby_card(self):
        self.card("card_UBER#mid_2026-09-16.md")
        result = self.search("UBER 2026-09-17 中线卡 入场 止损")
        self.assertEqual(result["hits"], [])
        self.assertEqual(self.search("MISSING 2026-09-17 中线卡 入场 止损")["hits"], [])

    def test_multiple_dates_or_periods_and_date_ranges_stay_broad(self):
        self.card("card_UBER#mid_2026-09-17.md")
        self.card("card_UBER#short_2026-09-16.md")
        for query in ["UBER 2026-09-16 与 2026-09-17 卡入场比较",
                      "UBER 中线和短线卡入场", "UBER 2026-09-16 以来入场变化",
                      "UBER 从 2026-09-16 到 09-17 的入场变化"]:
            with self.subTest(query=query):
                result = self.search(query)
                self.assertEqual(result["stock_filter"], {})
                self.assertEqual(len({h["file"] for h in result["hits"]}), 2)

    def test_dated_news_question_is_not_silently_limited_to_cards(self):
        self.card("card_TEST#short_2026-09-17.md")
        (self.root/'event-expansion.md').write_text("## 公告\nTEST 2026-09-17 宣布扩产。",encoding='utf-8')
        result=self.search("TEST 2026-09-17 宣布扩产的公告是什么？")
        self.assertEqual(result['stock_filter'],{})
        self.assertIn('event-expansion.md',[h['file'] for h in result['hits']])

    def test_explicit_filter_and_invalid_filter_are_observable(self):
        self.card("card_UBER#mid_2026-09-17.md")
        result = self.search("入场 止损", stock_filter={"ticker": "uber", "horizon": "mid"})
        self.assertEqual(result["stock_filter_origin"], "explicit")
        self.assertEqual(result["hits"][0]["stock"]["ticker"], "UBER")
        self.assertEqual(self.search("入场", stock_filter={"ticker": "MISSING"})["hits"], [])
        for fields in [{"card_date": "2026-02-31"}, {"typo": "mid"}, {}, {"horizon": "daily"}]:
            self.assertFalse(mr_tools.call("memory_search", {"query": "入场", "source": "stock",
                "mode": "bm25", "stock_filter": fields})["ok"])

    def test_both_hybrid_legs_receive_the_same_identity_mask(self):
        self.card("card_UBER#mid_2026-09-17.md")
        self.card("card_UBER#short_2026-09-17.md")
        state = mr_tools._ensure("bm25")
        wanted = {i for i,c in enumerate(state["chunks"]) if "#mid_" in c["file"]}
        masks = []
        def vector_search(query, topk, mask):
            masks.append(mask)
            return [(i, .8) for i in sorted(mask)]
        state["vecs"] = {mr_index.model_spec()["slug"]: SimpleNamespace(available=True, search=vector_search)}
        for mode in ("hybrid", "vec"):
            result = mr_tools.memory_search("UBER 2026-09-17 中线卡的入场", source="stock", mode=mode, rerank=False)
            self.assertEqual(result["hits"][0]["file"], "card_UBER#mid_2026-09-17.md")
        self.assertEqual(masks, [wanted, wanted])

    def test_answer_tool_rereads_sources_and_rejects_a_stale_reference(self):
        file = "card_UBER#mid_2026-09-17.md"
        self.card(file, "当时价 105；站上 110；回踩 95-98。")
        hit = self.search("UBER 2026-09-17 中线卡")["hits"][0]
        ref = {"id":"e1", **{k:hit[k] for k in ("source","file","heading","chunk_md5")}}
        args = {"question":"触发了吗？", "evidence":[ref], "answer": {
            "decision":"answer", "reason":"", "claims":[{"text":"两条腿都尚未到位。", "citations":[
                {"evidence_id":"e1", "quote":"当时价 105；站上 110；回踩 95-98。"}]}]}}
        result = mr_tools.call("memory_check_answer", args)
        self.assertTrue(result["ok"])
        self.assertFalse(result["checks_pass"])
        self.card(file, "内容改变，不能继续引用旧版本。")
        stale = mr_tools.call("memory_check_answer", args)
        self.assertFalse(stale["ok"])
        self.assertEqual(stale["get_errors"][0]["id"], "e1")

    def test_answer_tool_does_not_trust_caller_supplied_source_text(self):
        args = {"question":"触发了吗？", "answer":{"decision":"insufficient","claims":[],"reason":"缺少成交记录"},
                "evidence":[{"id":"e1","source":"stock","file":"made-up.md","text":"fake proof"}]}
        self.assertFalse(mr_tools.call("memory_check_answer", args)["ok"])


if __name__ == "__main__":
    unittest.main()
