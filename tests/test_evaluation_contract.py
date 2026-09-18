import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mr
import mr_eval
import mr_tools


def hit(file, heading="正文", snippet="", source="mem"):
    return dict(source=source, file=file, heading=heading, snippet=snippet)


class EvaluationContractTests(unittest.TestCase):
    def test_cli_reports_actual_top_five_separately_from_deduplicated_files(self):
        hits = [hit("other.md")] * 5 + [hit("right.md")]
        response = {"ok": True, "mode": "bm25", "confidence": "high", "hits": hits, "note": ""}
        with tempfile.TemporaryDirectory() as directory:
            golden = Path(directory) / "golden.json"
            golden.write_text(json.dumps({"queries": [{"q": "query", "expect": ["right.md"]}]}), encoding="utf-8")
            output = io.StringIO()
            with patch.object(mr_tools, "memory_search", return_value=response), patch.object(mr_tools, "_ensure", return_value={"chunks": []}), patch.object(mr_tools, "LOG_ON", False), contextlib.redirect_stdout(output):
                mr.cmd_eval("bm25", False, str(golden), rerank=False)
            self.assertIn("returned@5 0/1", output.getvalue())
            self.assertIn("document@5 1/1", output.getvalue())

    def test_duplicate_chunks_do_not_inflate_actual_top_five(self):
        hits = [hit("other.md")] * 5 + [hit("right.md")]
        result = mr_eval.grade_hits({"expect": ["right.md"]}, hits)
        self.assertEqual(result["returned_rank"], 6)
        self.assertEqual(result["document_rank"], 2)

    def test_correct_file_is_not_sufficient_passage_evidence(self):
        query = {"expect": ["workflow.md"], "evidence": [
            {"source": "mem", "file": "workflow.md", "heading_contains": "直合步骤",
             "required_text": ["cherry-pick", "Change-Id", "%submit"]}]}
        for candidate in (hit("workflow.md", "索引行归档", "cherry-pick Change-Id %submit"),
                          hit("workflow.md", "直合步骤", "cherry-pick")):
            grade = mr_eval.grade_hits(query, [candidate])
            self.assertEqual(grade["returned_rank"], 1)
            self.assertIsNone(grade["evidence_rank"])
        grade = mr_eval.grade_hits(query, [hit("workflow.md", "直合步骤", "cherry-pick 保留 Change-Id，再 %submit")])
        self.assertEqual(grade["evidence_rank"], 1)

    def test_source_and_page_are_part_of_document_identity(self):
        query = {"expect": ["guide.pdf"], "source": "pdf", "expect_pages": [80, 82]}
        hits = [hit("guide.pdf", "p81", source="book"), hit("guide.pdf", "p70", source="pdf"),
                hit("guide.pdf", "p81 (2)", source="pdf")]
        self.assertEqual(mr_eval.grade_hits(query, hits)["returned_rank"], 3)

    def test_related_but_wrong_passage_is_a_hard_negative(self):
        query = {"expect": [], "hard_negatives": [{"source": "case", "file": "eval.md", "heading_contains": "复盘"}]}
        grade = mr_eval.grade_hits(query, [hit("eval.md", "测试复盘", "问题在这里出现过", "case")])
        self.assertIsNone(grade["evidence_rank"])
        self.assertEqual(grade["hard_negative_ranks"], [1])

    def test_paraphrases_of_one_case_cannot_cross_splits(self):
        with self.assertRaisesRegex(ValueError, "crosses"):
            mr_eval.validate_labels({"queries": [
                {"expect": [], "case_group": "same-case", "split": "regression"},
                {"expect": [], "case_group": "same-case", "split": "holdout"}]})

    def test_empty_fact_does_not_make_every_snippet_pass(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            mr_eval.validate_labels({"queries": [{"expect": ["x.md"], "evidence": [
                {"source": "mem", "file": "x.md", "heading": "步骤", "required_text": [[""]]}]}]})


if __name__ == "__main__":
    unittest.main()
