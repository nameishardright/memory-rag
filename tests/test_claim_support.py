"""A real quotation does not by itself entail the sentence beside it."""
import unittest

import mr_answer_eval as ae
import mr_support
from scripts import report_answer_evaluation


def evidence(file="card_TEST#short_2026-09-17.md", text="当时价 105；站上 110；回踩 95-98。", id="e1"):
    return {"id": id, "source": "stock", "file": file, "heading": "判断", "text": text}


def answer(text, quote, id="e1"):
    return {"decision": "answer", "claims": [
        {"text": text, "citations": [{"evidence_id": id, "quote": quote}]}], "reason": ""}


class ClaimSupportTests(unittest.TestCase):
    def test_price_snapshot_does_not_prove_neither_leg_triggered(self):
        context = evidence()
        q = {"q": "当时价格和触发情况？", "answer_contract": {"decision": "answer", "facts": [
            {"id": "snapshot", "claim_terms": ["105"], "evidence": [
                {"source": "stock", "file": context["file"], "heading": "判断", "required_text": ["当时价 105"]}]}]}}
        a = answer("当时价 105，两条腿都尚未到位。", context["text"])
        result = ae.grade_answer(q, a, [context])
        self.assertFalse(result["contract_pass"])
        self.assertIn("trade_outcome_unproven", [i["code"] for i in result["support_check"]["issues"]])

    def check(self, text, contexts, citations=None):
        a = answer(text, contexts[0]["text"])
        if citations is not None:
            a["claims"][0]["citations"] = citations
        return mr_support.check_answer(a, contexts)

    def test_price_from_another_date_is_not_support_for_this_date(self):
        c = evidence(text="2026-09-16 收盘 123.5。")
        result = self.check("2026-09-17 收盘为 123.5。", [c])
        self.assertFalse(result["checks_pass"])
        self.assertIn("price_date_not_bound", [i["code"] for i in result["issues"]])

    def test_date_comma_price_still_binds_to_that_date(self):
        c=evidence(text='2026-09-16 收盘 123.5；2026-09-17 收盘 119.2。')
        self.assertFalse(self.check('2026-09-17，收盘为 123.5。',[c])['checks_pass'])
        self.assertTrue(self.check('2026-09-17，收盘为 119.2。',[c])['checks_pass'])

    def test_multi_date_or_multi_metric_quote_cannot_supply_arbitrary_pairs(self):
        cases=[('2026-09-16 和 2026-09-17 收盘分别为 123.5 和 119.2。','2026-09-16 收盘为 119.2。'),
               ('2026-09-17 收盘123.5、开盘119.2。','2026-09-17 收盘119.2。'),
               ('如果 2026-09-17 收盘123.5 就买入。','2026-09-17 收盘123.5。')]
        for quote,claim in cases:
            with self.subTest(quote=quote):
                self.assertFalse(self.check(claim,[evidence(text=quote)])['checks_pass'])

    def test_two_real_numbers_cannot_be_swapped_between_dates(self):
        c = evidence(text="2026-09-16 收盘 123.5；2026-09-17 收盘 119.2。")
        result = self.check("2026-09-16 收盘 119.2；2026-09-17 收盘 123.5。", [c])
        self.assertEqual(len(result["sentences"]), 2)
        self.assertFalse(result["checks_pass"])

    def test_exact_dated_observation_and_plan_prices_remain_allowed(self):
        for text in ["2026-09-16 收盘 123.5。", "卡中参考入场 95-98，止损 90。",
                     "美东正常交易时段收盘低于 95 就退出。"]:
            with self.subTest(text=text):
                self.assertTrue(self.check(text, [evidence(text=text)])["checks_pass"])

    def test_card_date_does_not_supply_missing_observation_date(self):
        c = evidence(text="当时价 123.5。")
        self.assertFalse(self.check("2026-09-17 的收盘价是 123.5。", [c])["checks_pass"])

    def test_other_claim_citation_cannot_rescue_a_fabricated_number(self):
        c = evidence(text="参考入场 95。")
        other = evidence(text="参考入场 105。", id="e2")
        a = answer("参考入场 105。", c["text"])
        a["claims"].append(answer("另一张卡参考入场 105。", other["text"], "e2")["claims"][0])
        result = mr_support.check_answer(a, [c, other])
        self.assertFalse(result["checks_pass"])
        self.assertIn("number_not_quoted", [i["code"] for i in result["sentences"][0]["issues"]])

    def test_mixed_horizons_require_separate_attribution(self):
        a = evidence(text="参考止损 90。")
        b = evidence(file="card_TEST#mid_2026-09-17.md", text="参考止损 80。", id="e2")
        result = self.check("止损为 90 和 80。", [a,b], [
            {"evidence_id":"e1", "quote":a["text"]}, {"evidence_id":"e2", "quote":b["text"]}])
        self.assertFalse(result["checks_pass"])
        self.assertIn("mixed_card_scopes", [i["code"] for i in result["issues"]])

    def test_well_quoted_fact_from_an_explicitly_wrong_card_needs_correction(self):
        c = evidence(file="card_TEST#mid_2026-09-17.md", text="参考止损 80。")
        result = mr_support.check_answer(answer("参考止损 80。",c['text']), [c],
                                         "只看 TEST 2026-09-17 短线卡的止损。")
        self.assertIn("outside_requested_card", [i["code"] for i in result["issues"]])
        result=mr_support.check_answer(answer('参考止损 80。',c['text']),[c],
                                       'TEST 2026-09-17 短线卡的止损是什么？')
        self.assertFalse(result['checks_pass'])

    def test_later_confirmation_cannot_be_added_to_an_earlier_card_quote(self):
        c = evidence(file="card_TEST#mid_2026-09-16.md", text="100 单日击穿，待确认。")
        result = self.check("09-16 中线卡记为 100 单日击穿，因此 09-17 才二次确认。", [c])
        self.assertFalse(result['checks_pass'])
        self.assertIn('date_not_quoted', [i['code'] for i in result['issues']])

    def test_compressed_two_day_reference_keeps_both_dates(self):
        c = evidence(text="09-17 复核：100 减仓线，09-15/16 两收盘在下，二次确认。")
        self.assertTrue(self.check("09-17 复核依据是 09-15 和 09-16 的两个收盘均在 100 减仓线以下。", [c])['checks_pass'])

    def test_short_source_date_cannot_be_moved_to_a_different_year(self):
        c = evidence(text="09-17 收盘 123.5。")
        self.assertFalse(self.check("2025-09-17 收盘 123.5。", [c])['checks_pass'])

    def test_small_plan_price_range_is_not_a_calendar_date(self):
        c = evidence(text="参考入场区间 10-11。")
        self.assertTrue(self.check("参考入场区间 10-11。", [c])['checks_pass'])
        self.assertFalse(self.check("参考入场区间 11-12。", [c])['checks_pass'])

    def test_uncertainty_in_another_clause_does_not_waive_outcome_check(self):
        result = self.check("无法知道成交数量，但两条腿都尚未到位。", [evidence()])
        self.assertFalse(result["checks_pass"])
        self.assertFalse(self.check('如果以后回踩再复核，但整个窗口未触发。',[evidence()])['checks_pass'])

    def test_negated_or_conditional_quote_does_not_prove_execution(self):
        for text in ["截至复核时未成交。", "如果触发，那么已经成交才记账。"]:
            self.assertFalse(self.check("已经成交。", [evidence(text=text)])["checks_pass"])

    def test_explicit_outcome_record_can_be_quoted_but_snapshot_warning_is_allowed(self):
        record = "结果记录：全程逐笔回放，两条腿均未触发。"
        self.assertTrue(self.check(record, [evidence(text=record)])["checks_pass"])
        self.assertTrue(self.check("当时价 105，不能据此证明整个窗口未触发。", [evidence()])["checks_pass"])

    def test_cropped_conditional_quote_is_checked_in_its_source_sentence(self):
        c = evidence(text="如果已经成交，才登记执行结果。")
        result = self.check("已经成交。", [c], [{"evidence_id":"e1", "quote":"已经成交"}])
        self.assertFalse(result["checks_pass"])
        c=evidence(text='如果已经成交，才登记执行结果。注意后续复核。')
        result=self.check('已经成交。',[c],[{'evidence_id':'e1','quote':'已经成交，才登记执行结果。注意后续复核。'}])
        self.assertFalse(result['checks_pass'])

    def test_later_factual_occurrence_is_not_hidden_by_an_earlier_hypothesis(self):
        c=evidence(text='如果已经成交，才登记执行结果。实际核对：已经成交。')
        result=self.check('已经成交。',[c],[{'evidence_id':'e1','quote':'已经成交'}])
        self.assertTrue(result['checks_pass'])

    def test_a_required_fact_cannot_borrow_an_unrelated_claims_citation(self):
        from test_answer_evaluation import fixture
        q,c,a = fixture()
        other = dict(c, id="e2", file="different.md", text="另一个方案也提到60秒TTL。")
        a["claims"][0]["citations"] = [{"evidence_id":"e2", "quote":other["text"]}]
        a["claims"].append({"text":"还有背景材料。", "citations":[{"evidence_id":"e1", "quote":c["text"]}]})
        self.assertFalse(ae.grade_answer(q,a,[c,other])["contract_pass"])

    def test_same_passage_identity_cannot_hide_a_wrong_quote_in_another_claim(self):
        from test_answer_evaluation import fixture
        q,c,a=fixture()
        c['text']='推断结果只配60秒TTL。60 只是另一条日志的编号。'
        a['claims'][0]['citations']=[{'evidence_id':'e1','quote':'60 只是另一条日志的编号。'}]
        a['claims'].append({'text':'资料规定推断结果只配60秒TTL。',
                            'citations':[{'evidence_id':'e1','quote':'推断结果只配60秒TTL。'}]})
        self.assertFalse(ae.grade_answer(q,a,[c])['contract_pass'])

    def test_review_must_cover_every_sentence_and_cannot_hide_a_bad_one(self):
        a = answer("参考价 105。两条腿都尚未到位。", evidence()["text"])
        row = {"id":"one", "case_group":"event", "answer":a, "grade":{
            "support_check":mr_support.check_answer(a,[evidence()])}}
        review = {"method":"source review", "rows":[{
            "id":"one", "policy":"passage", "answer_correct":True, "citations_support":True,
            "scope_correct":True, "refusal_correct":True, "rationale":"review"}]}
        run = {"answers":{"passage":[row]}}
        with self.assertRaisesRegex(ValueError,"every sentence"):
            report_answer_evaluation.review_summary(run,review)
        review['rows'][0]['sentence_reviews'] = [
            {"claim":0,"sentence":0,"supported":True,"scope_correct":True,"rationale":"Snapshot is quoted"},
            {"claim":0,"sentence":1,"supported":False,"scope_correct":False,"rationale":"No full-window observation"}]
        with self.assertRaisesRegex(ValueError,"failed sentence"):
            report_answer_evaluation.review_summary(run,review)
        review['rows'][0].update(answer_correct=False,citations_support=False,scope_correct=False)
        summary = report_answer_evaluation.review_summary(run,review)['passage']
        self.assertEqual(summary['semantic_pass'],0)
        self.assertEqual((summary['sentences_supported'],summary['sentences_reviewed']),(1,2))


if __name__ == "__main__":
    unittest.main()
