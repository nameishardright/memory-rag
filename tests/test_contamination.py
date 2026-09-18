import unittest
from unittest.mock import Mock

import mr_eval


GOLDEN = {"source": "stock", "queries": [{"q": "今天天气怎么样", "expect": []}]}


def passage(**changes):
    return {"source": "case", "file": "new-review.md", "heading": "本次复盘", "id": "new-case",
            "text": "我们用「今天天气怎么样」测试拒答。", **changes}


class ContaminationTests(unittest.TestCase):
    def test_new_unlisted_review_fails_even_outside_the_selected_source(self):
        with self.assertRaises(mr_eval.CorpusContaminationError) as caught:
            mr_eval.require_clean_corpus([passage()], [GOLDEN])
        finding = caught.exception.audit["findings"][0]
        self.assertEqual(finding["file"], "new-review.md")
        self.assertEqual(finding["source"], "case")
        self.assertNotIn("text", finding)

    def test_explicit_evaluation_material_is_allowed_for_intentional_lookup(self):
        result = mr_eval.require_clean_corpus([passage(purpose="evaluation")], [GOLDEN])
        self.assertTrue(result["ok"])
        self.assertEqual(result["evaluation_chunks"], 1)

    def test_a_question_in_the_heading_is_also_indexed_content(self):
        with self.assertRaises(mr_eval.CorpusContaminationError):
            mr_eval.require_clean_corpus([passage(text="说明", heading="今天天气怎么样")], [GOLDEN])

    def test_quality_evaluation_stops_before_any_query_is_run(self):
        search = Mock()
        with self.assertRaises(mr_eval.CorpusContaminationError):
            mr_eval.evaluate(GOLDEN, search, chunks=[passage()])
        search.assert_not_called()

    def test_evaluation_cannot_silently_omit_the_corpus_audit(self):
        with self.assertRaisesRegex(ValueError, "corpus"):
            mr_eval.evaluate(GOLDEN, Mock())

    def test_literal_check_does_not_claim_to_detect_paraphrases(self):
        result = mr_eval.require_clean_corpus([passage(text="我们用今日的天气如何测试拒答。")], [GOLDEN])
        self.assertTrue(result["ok"])


if __name__ == "__main__":
    unittest.main()
