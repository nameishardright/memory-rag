import copy
import json
from pathlib import Path
import tempfile
import unittest

import mr_answer_eval as ae
from scripts import eval_answers
from scripts import report_answer_evaluation


def fixture():
    label = {"source": "case", "file": "cache.md", "heading": "缓存",
             "required_text": ["推断结果只配60秒TTL"]}
    q = {"id": "q1", "case_group": "cache-event", "split": "development", "q": "推断状态应缓存多久？",
         "expect": ["cache.md"], "evidence": [label],
         "source_units": [{"source": "case", "file": "cache.md"}],
         "answer_contract": {"decision": "answer", "facts": [
             {"id": "ttl", "claim_terms": ["60", "TTL"], "evidence": [label]}],
             "forbidden_patterns": ["永久缓存", "不需要.*TTL"]}}
    c = {"id": "e1", "source": "case", "file": "cache.md", "heading": "缓存",
         "text": "推断结果只配60秒TTL，不能永久缓存。"}
    a = {"decision": "answer", "claims": [{"text": "推断结果只配60秒TTL。", "citations": [
        {"evidence_id": "e1", "quote": "推断结果只配60秒TTL"}]}], "reason": ""}
    return q, c, a


class AnswerEvaluationTests(unittest.TestCase):
    def test_generator_never_receives_labels_expected_decision_or_confidence(self):
        q, c, _ = fixture()
        q["answer_contract"]["secret_expected_answer"] = "gold-secret"
        q["confidence"] = "high"
        payload = ae.answer_payload(q, [c])
        self.assertEqual(set(payload), {"question", "evidence"})
        self.assertNotIn("gold-secret", json.dumps(payload))
        self.assertNotIn("answer_contract", json.dumps(payload))
        self.assertNotIn("confidence", json.dumps(payload))

    def test_correct_fact_with_exact_support_passes(self):
        q, c, a = fixture()
        self.assertTrue(ae.grade_answer(q, a, [c])["contract_pass"])

    def test_correct_keywords_do_not_allow_explicit_wrong_policy(self):
        q, c, a = fixture()
        a["claims"][0]["text"] = "虽然提到60秒TTL，仍应永久缓存。"
        self.assertFalse(ae.grade_answer(q, a, [c])["contract_pass"])

    def test_quote_from_wrong_source_cannot_support_fact(self):
        q, c, a = fixture()
        c["file"] = "similar-but-wrong.md"
        result = ae.grade_answer(q, a, [c])
        self.assertEqual(result["grounded_facts"], [])
        self.assertFalse(result["contract_pass"])

    def test_fabricated_quote_cannot_be_rescued_by_correct_answer(self):
        q, c, a = fixture()
        a["claims"][0]["citations"][0]["quote"] += "还需每秒强制刷新"
        result = ae.grade_answer(q, a, [c])
        self.assertEqual(len(result["invalid_quotes"]), 1)
        self.assertFalse(result["contract_pass"])

    def test_unknown_citation_id_fails(self):
        q, c, a = fixture()
        a["claims"][0]["citations"][0]["evidence_id"] = "e999"
        self.assertFalse(ae.grade_answer(q, a, [c])["contract_pass"])

    def test_missing_second_fact_is_not_full_answer(self):
        q, c, a = fixture()
        q["answer_contract"]["facts"].append({"id": "observed", "claim_terms": ["观测"],
                                               "evidence": q["evidence"]})
        self.assertFalse(ae.grade_answer(q, a, [c])["contract_pass"])

    def test_extra_claim_is_flagged_even_with_a_real_quote(self):
        q, c, a = fixture()
        a["claims"].append({"text": "客户已经收到文件。", "citations": a["claims"][0]["citations"]})
        result = ae.grade_answer(q, a, [c])
        self.assertEqual(result["unmapped_claims"], [1])
        self.assertTrue(result["semantic_review_required"])
        self.assertNotIn('semantic_correct', result)

    def test_negative_requires_actual_refusal(self):
        q, c, a = fixture()
        q["expect"] = []
        q["answer_contract"] = {"decision": "insufficient", "absence_reason": "No current measurement"}
        self.assertFalse(ae.grade_answer(q, a, [c])["contract_pass"])
        refusal = {"decision": "insufficient", "claims": [], "reason": "缺少实时测量。"}
        self.assertTrue(ae.grade_answer(q, refusal, [c])["contract_pass"])

    def test_refusal_cannot_hide_an_answer_or_add_unscored_prose(self):
        _, _, a = fixture()
        a["decision"] = "insufficient"
        a["reason"] = "仍给个答案"
        with self.assertRaises(ValueError):
            ae.parse_answer(json.dumps(a))
        _, _, a = fixture()
        a["extra_answer"] = "an ungrounded extra answer"
        with self.assertRaises(ValueError):
            ae.parse_answer(json.dumps(a))

    def test_pdf_neighbour_quote_keeps_real_page_identity(self):
        result = {"hits": [{"source": "pdf", "file": "guide.pdf", "heading": "p10", "snippet": "preview"}]}
        def get(**kwargs):
            self.assertEqual(kwargs, {"source": "pdf", "file": "guide.pdf#p10"})
            return {"ok": True, "source": "pdf", "file": "guide.pdf", "page": 10,
                    "text": "── p9 ──\nolder fact\n\n── p10 ──\ncentre\n\n── p11 ──\nnext fact"}
        contexts = ae.assemble_context(result, get)["contexts"]
        self.assertEqual([c["heading"] for c in contexts], ["p10", "p9", "p11"])
        self.assertEqual(contexts[1]["text"].strip(), "older fact")

    def test_get_failure_is_visible_and_not_filled_with_gold(self):
        r = {"hits": [{"source": "case", "file": "missing.md", "heading": "H", "snippet": "x"}]}
        c = ae.assemble_context(r, lambda **kw: {"ok": False, "error": "missing"})
        self.assertEqual(c["contexts"], [])
        self.assertEqual(len(c["get_errors"]), 1)

    def test_context_budget_is_shared_and_exposes_truncation(self):
        r = {"hits": [{"source": "case", "file": str(i), "heading": "H", "snippet": "abcdef"}
                      for i in range(3)]}
        c = ae.assemble_context(r, policy="snippet", max_chars=10)
        self.assertEqual(c["characters"], 10)
        self.assertEqual(len(c["contexts"]), 2)
        self.assertTrue(c["contexts"][-1]["truncated"])

    def test_get_identity_mismatch_fails_closed(self):
        r = {"hits": [{"source": "case", "file": "one.md", "heading": "H", "snippet": "x"}]}
        with self.assertRaisesRegex(ValueError, "identity"):
            ae.assemble_context(r, lambda **kw: {"ok": True, "source": "case", "file": "other.md", "text": "x"})

    def test_paraphrase_source_cannot_cross_development_blind(self):
        q, _, _ = fixture()
        other = copy.deepcopy(q)
        other.update(id="q2", q="换一种问法", split="blind", case_group="disguised-new-group")
        with self.assertRaisesRegex(ValueError, "Source unit crosses"):
            ae.validate_suites([{"queries": [q, other]}])

    def test_pdf_interval_overlap_is_rejected_but_disjoint_chapters_allowed(self):
        self.assertTrue(ae.units_overlap({"source": "pdf", "file": "x", "pages": [10, 20]},
                                         {"source": "pdf", "file": "x", "pages": [20, 30]}))
        self.assertFalse(ae.units_overlap({"source": "pdf", "file": "x", "pages": [10, 20]},
                                          {"source": "pdf", "file": "x", "pages": [21, 30]}))

    def test_prior_consumed_holdout_is_a_reference_for_next_blind(self):
        q, _, _ = fixture()
        reference = copy.deepcopy(q)
        q.update(split="blind", case_group="renamed-event", q="new-looking wording")
        with self.assertRaisesRegex(ValueError, "prior evaluation"):
            ae.validate_suites([{"queries": [q]}], [{"queries": [reference]}])

    def test_empty_fact_alternative_is_rejected(self):
        q, _, _ = fixture()
        q["answer_contract"]["facts"][0]["claim_terms"] = [[""]]
        with self.assertRaisesRegex(ValueError, "empty"):
            ae.validate_suites([{"queries": [q]}])

    def test_source_review_rejects_fabricated_required_fact(self):
        q, c, _ = fixture()
        c["text"] = "topic words but no actual fact"
        with self.assertRaisesRegex(ValueError, "not supported"):
            ae.validate_source_evidence({"queries": [q]}, [c])

    def test_consumed_trial_cannot_run_twice_even_after_failed_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"seal.json"
            inputs = {"dataset_hash": "123", "prompt": "fixed"}
            eval_answers.seal_trial(path, inputs)
            eval_answers.consume_trial(path, inputs)
            with self.assertRaises(FileExistsError):
                eval_answers.consume_trial(path, inputs)

    def test_changed_prompt_rejected_before_consuming_trial(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"seal.json"
            eval_answers.seal_trial(path, {"prompt": "original"})
            with self.assertRaisesRegex(ValueError, "changed"):
                eval_answers.consume_trial(path, {"prompt": "tuned after seal"})
            self.assertFalse(path.with_suffix('.consumed.json').exists())

    def test_frozen_source_mutation_fails(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d)/"source.md"
            source.write_text("first", encoding='utf-8')
            manifest = Path(d)/"snapshot.json"
            manifest.write_text(json.dumps({"source.md":eval_answers.sha(source)}), encoding='utf-8')
            eval_answers.verify_snapshot(manifest)
            source.write_text("changed", encoding='utf-8')
            with self.assertRaisesRegex(ValueError, "Frozen source changed"):
                eval_answers.verify_snapshot(manifest)

    def test_low_confidence_warning_does_not_drop_a_negative_from_evaluation(self):
        data = {"ok": True, "mode": "hybrid", "confidence": "low", "note": "证据不足，别硬凑",
                "rerank": {"model": "base"}, "corpus_sha256": "frozen"}
        eval_answers.validate_search_response(data, {"mode": "hybrid", "rerank": True})

    def test_actual_vector_fallback_is_not_accepted_as_hybrid(self):
        with self.assertRaisesRegex(RuntimeError, "degraded"):
            eval_answers.validate_search_response({"ok": True, "mode": "bm25"}, {"mode": "hybrid"})

    def test_missing_gate_cannot_be_reported_as_gate_evaluation(self):
        with self.assertRaisesRegex(RuntimeError, "gate did not run"):
            eval_answers.validate_search_response({"ok": True, "mode": "hybrid", "rerank": None},
                                                   {"mode": "hybrid", "rerank": True})

    def test_mcp_get_error_is_recorded_as_missing_context_not_dropped_question(self):
        result={'isError':True,'content':[{'type':'text','text':json.dumps(
            {'ok':False,'error':'ambiguous heading','candidates':['same title']})}]}
        self.assertFalse(eval_answers.decode_tool_result('memory_get',result)['ok'])
        with self.assertRaisesRegex(RuntimeError,'MCP tool error'):
            eval_answers.decode_tool_result('memory_search',result)

    def test_fact_terms_and_support_can_span_multiple_claims_without_rewriting_quotes(self):
        q,c,a=fixture()
        q['answer_contract']['facts'][0]['claim_terms']=['推断','TTL']
        a['claims']=[{'text':'这是推断结果。','citations':[{'evidence_id':'e1','quote':'推断结果只配60秒TTL'}]},
                     {'text':'需要TTL。','citations':[{'evidence_id':'e1','quote':'推断结果只配60秒TTL'}]}]
        self.assertEqual(ae.grade_answer(q,a,[c])['grounded_facts'],['ttl'])

    def test_context_passes_content_reference_and_keeps_repeated_titles_separate(self):
        h={'source':'case','file':'x.md','heading':'重复标题','snippet':'preview'}
        hits=[dict(h,chunk_md5='a'*32),dict(h,chunk_md5='b'*32)]
        def get(**kwargs):
            text='first body' if kwargs['chunk_md5']=='a'*32 else 'second body'
            return dict(ok=True,source='case',file='x.md',heading='重复标题',text=text,chunk_md5=kwargs['chunk_md5'])
        result=ae.assemble_context({'hits':hits},get)
        self.assertEqual([c['text'] for c in result['contexts']],['first body','second body'])

    def test_model_format_failure_stays_in_accuracy_denominator(self):
        rows = [{"id":"a", "expected_decision":"answer", "error":"invalid JSON"},
                {"id":"b", "expected_decision":"insufficient", "grade":{"contract_pass":True}}]
        summary = ae.summarize_answers(rows)
        self.assertEqual(summary['questions'], 2)
        self.assertEqual(summary['contract_pass'], 1)
        self.assertEqual(summary['generation_or_format_errors'], 1)

    def test_semantic_review_cannot_omit_a_failed_answer(self):
        run = {'answers':{'snippet':[{'id':'one','case_group':'a'}],
                          'passage':[{'id':'one','case_group':'a'}]}}
        review = {'method':'source audit','rows':[{'id':'one','policy':'snippet'}]}
        with self.assertRaisesRegex(ValueError, 'every answer'):
            report_answer_evaluation.review_summary(run, review)

    def test_semantic_review_does_not_count_two_paraphrases_as_two_groups(self):
        run={'answers':{'passage':[{'id':'one','case_group':'a'},{'id':'two','case_group':'a'}]}}
        rows=[dict(id=id,policy='passage',answer_correct=True,citations_support=True,
                   scope_correct=True,refusal_correct=True,rationale='source supported') for id in ['one','two']]
        rows[1]['answer_correct']=False
        summary=report_answer_evaluation.review_summary(run,{'method':'source audit','rows':rows})['passage']
        self.assertEqual(summary['reviewed'],2)
        self.assertEqual(summary['semantic_pass'],1)
        self.assertEqual(summary['groups'],1)
        self.assertEqual(summary['all_questions_in_group_pass'],0)


if __name__ == "__main__":
    unittest.main()
