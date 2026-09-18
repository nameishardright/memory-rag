import contextlib
import io
import unittest
import json
from pathlib import Path
import tempfile
import random
import os
from unittest.mock import patch

import mr_corpus
import mr_eval
import mr_tools
import mr_mine
import mr_index
from scripts.sealed_eval import seal, evaluate_once
from scripts.verify_optimized import calibrate


class OptimizationContractTests(unittest.TestCase):
    def test_source_budget_variants_share_weights_and_reset_token_limit(self):
        class Encoder:
            max_seq_length=0
            seen=[]
            def predict(self,pairs,**kwargs):
                self.seen.append(self.max_seq_length)
                return [.9 for _ in pairs]
        def init(reranker,spec):
            reranker.spec=spec
            reranker.available=True
            reranker._model=None
        with patch.dict(os.environ,{},clear=True),patch.dict(mr_tools._STATE,{},clear=True), \
                patch.object(mr_index.Reranker,"__init__",init):
            notes=mr_tools._get_reranker("bge-reranker-base")
            notes._model=Encoder()
            pdf=mr_tools._get_reranker("bge-reranker-base",legacy=True)
            self.assertIs(notes._model,pdf._model)
            self.assertEqual(pdf.spec["gate_candidates"],5)
            self.assertEqual(pdf.spec["max_windows"],3)
            notes.score("q",["a"])
            pdf.score("q",["a"])
            notes.score("q",["a"])
            self.assertEqual(notes._model.seen,[256,512,256])
            with tempfile.TemporaryDirectory() as tmp:
                custom=Path(tmp)/"bge-reranker-base"
                custom.mkdir()
                separate=mr_tools._get_reranker(str(custom),legacy=True)
                self.assertIsNot(separate,pdf)
                self.assertIsNone(separate._model,"Same directory basename must not share different model weights")

    def test_calibration_rejects_fast_profile_that_needs_an_overfit_floor(self):
        negative=[0,0,.001,.002,.015,.02,.024,.042,.052,.085,.295,.362]
        with self.assertRaisesRegex(RuntimeError,"No conservative"):
            calibrate([.045,.405,.8],negative)

    def test_calibration_may_keep_old_cutoffs_when_new_scores_support_them(self):
        negative=[0,0,.001,.002,.015,.02,.024,.052,.085,.295,.35,.412]
        result=calibrate([.405,.7,.8],negative)
        self.assertEqual((result["selected"]["hi"],result["selected"]["lo"]),(.5,.1))
        self.assertEqual(result["selected"]["negative_high"],0)
        with self.assertRaises(ValueError):calibrate([float("nan")],negative)

    def test_new_file_is_warned_at_real_reload_and_label_clears_warning(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(mr_tools._STATE,{},clear=True), \
                patch.object(mr_corpus,"SOURCES",[{"key":"case","dir":tmp,"redact":False}]), \
                patch.object(mr_corpus,"CORPUS_POLICY",str(Path(tmp)/"absent-policy.json")), \
                patch.object(mr_corpus,"pdf_enabled",return_value=False), \
                patch.object(mr_tools,"RELOAD_SEC",0), patch.object(mr_tools,"LOG_ON",False), \
                patch.object(mr_eval,"load_goldens",return_value=[{"queries":[{"q":"literal question","expect":[]}]}]), \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            Path(tmp,"knowledge.md").write_text("# Knowledge\nUseful facts.",encoding="utf-8")
            first=mr_tools.memory_search("facts",mode="bm25",rerank=False)
            self.assertEqual(first["corpus_audit"]["status"],"clean")
            review=Path(tmp,"new-review.md")
            review.write_text("# Review\nTested literal question yesterday.",encoding="utf-8")
            second=mr_tools.memory_search("literal question",mode="bm25",rerank=False)
            self.assertEqual(second["corpus_audit"]["status"],"warning")
            self.assertNotEqual(first["corpus_sha256"],second["corpus_sha256"])
            self.assertTrue(any(h["file"]=="new-review.md" for h in second["hits"]))
            self.assertIn("[corpus-audit]",errors.getvalue())
            self.assertNotIn("literal question",errors.getvalue())
            review.write_text("---\nrag_purpose: evaluation\n---\n# Review\nTested literal question yesterday.",encoding="utf-8")
            third=mr_tools.memory_search("literal question",mode="bm25",rerank=False)
            self.assertEqual(third["corpus_audit"]["status"],"clean")
            self.assertFalse(any(h["file"]=="new-review.md" for h in third["hits"]))

    def test_paragraph_spans_are_bounded_and_reconstruct_the_original(self):
        rng = random.Random(20260918)
        for _ in range(30):
            paragraphs = [(str(n) + " abc。" * rng.randrange(1, 850)) for n in range(7)]
            original = "\n\n".join(paragraphs)
            spans = mr_corpus._paragraph_spans(original)
            covered = 0
            rebuilt = ""
            for start,end in spans:
                self.assertLessEqual(start,covered)
                self.assertGreater(end,covered)
                self.assertLessEqual(end-start,mr_corpus.MAX_CHUNK)
                rebuilt += original[covered:end]
                covered = end
            self.assertEqual(rebuilt,original)

    def test_stock_and_pdf_keep_their_existing_chunk_boundaries(self):
        body = "a" * 3300
        for source in ("stock","pdf"):
            self.assertEqual(mr_corpus.chunk_strategy(source),"character")
            parts = mr_corpus._hard_wrap("Body",body,strategy=mr_corpus.chunk_strategy(source))
            self.assertEqual(parts[1][1],body[1350:2850])

    def test_a_consumed_blind_set_cannot_be_retried_as_blind(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            dataset,manifest,output=directory/"blind.json",directory/"seal.json",directory/"result.json"
            dataset.write_text(json.dumps({"split":"blind","queries":[
                {"q":"not-in-corpus","expect":[],"case_group":"new-event"}]}),encoding="utf-8")
            seal(dataset,manifest,[])
            from unittest.mock import Mock
            search=Mock(return_value={"ok":True,"mode":"hybrid","hits":[],"confidence":"low"})
            evaluate_once(dataset,manifest,output,search,chunks=[])
            with self.assertRaises(FileExistsError):
                evaluate_once(dataset,manifest,output,search,chunks=[])
            self.assertEqual(search.call_count,1)

    def test_blind_seal_rejects_development_source_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            reference,dataset=directory/"regression.json",directory/"blind.json"
            reference.write_text(json.dumps({"queries":[{"q":"old question","expect":["same.md"]}]}),encoding="utf-8")
            dataset.write_text(json.dumps({"split":"blind","queries":[
                {"q":"different words","expect":["same.md"],"case_group":"renamed-group"}]}),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"overlaps"):
                seal(dataset,directory/"seal.json",[reference])

    def test_all_required_evidence_groups_must_be_present(self):
        def passage(fact):
            return {"source": "case", "file": "answer.md", "heading": fact, "snippet": fact}
        def label(fact):
            return {"source": "case", "file": "answer.md", "heading": fact, "required_text": [fact]}
        query = {"expect": ["answer.md"], "evidence_groups": [[label("cause")], [label("repair")]]}
        self.assertIsNone(mr_eval.grade_hits(query, [passage("cause")])["evidence_rank"])
        self.assertEqual(mr_eval.grade_hits(query, [passage("cause"), passage("repair")])["evidence_rank"], 2)

    def test_unranked_mining_never_calls_retrieval(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(mr_mine, "scan", return_value=([], {"sessions": 0, "turns": 0, "rag_calls": 0, "direct_reads": 0})), \
                patch.object(mr_mine, "candidates", return_value=[{"q": "question", "expect": ["answer.md"]}]), \
                patch.object(mr_mine, "rank") as rank, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp) / "candidates.json"
            mr_mine.main(out=str(path), unranked=True)
            rank.assert_not_called()
            self.assertFalse(json.loads(path.read_text(encoding="utf-8"))["ranked"])

    def test_mrr_at10_excludes_rank11_for_all_rank_types(self):
        golden = {"queries": [{"q": "needle", "expect": ["answer.md"], "evidence": [
            {"source": "case", "file": "answer.md", "heading": "Answer", "required_text": ["fact"]}]}]}
        hits = [{"source": "case", "file": f"other-{n}.md", "heading": "Body", "snippet": "other"}
                for n in range(10)]
        hits.append({"source": "case", "file": "answer.md", "heading": "Answer", "snippet": "fact"})
        result = mr_eval.evaluate(golden, lambda *a, **kw: {"ok": True, "mode": "hybrid", "hits": hits}, chunks=[])
        self.assertEqual(result["rows"][0]["returned_rank"], 11)
        for metric in ("returned_mrr_at10", "document_mrr_at10", "evidence_mrr_at10"):
            self.assertEqual(result["summary"][metric], 0.0, metric)

    def test_whole_short_paragraphs_survive_long_section_wrapping(self):
        paragraphs = [str(i) + "x" * 398 + "." for i in range(5)]
        body = "\n\n".join(paragraphs)
        parts = mr_corpus._hard_wrap("Section", body)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(text) <= mr_corpus.MAX_CHUNK for _, text in parts))
        for paragraph in paragraphs:
            self.assertTrue(any(paragraph in text for _, text in parts), paragraph[:8])

    def test_reload_warns_about_new_unmarked_evaluation_without_blocking(self):
        query = "new evaluation query"
        chunk = {"source": "case", "file": "new-review.md", "heading": "Review", "id": "test",
                 "text": query, "md5": "test", "purpose": "knowledge"}
        output = io.StringIO()
        with patch.dict(mr_tools._STATE, {}, clear=True), \
                patch.object(mr_corpus, "load_chunks", return_value=[chunk]), \
                patch.object(mr_corpus, "signature", return_value="test"), \
                patch.object(mr_eval, "load_goldens", return_value=[{"queries": [{"q": query, "expect": []}]}]), \
                contextlib.redirect_stderr(output):
            state = mr_tools._ensure("bm25")
            self.assertEqual(len(state["chunks"]), 1)
            self.assertIn("[corpus-audit]", output.getvalue())
            self.assertNotIn(query, output.getvalue())


if __name__ == "__main__":
    unittest.main()
