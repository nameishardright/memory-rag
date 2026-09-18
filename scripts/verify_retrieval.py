"""Paired live-corpus checks against a preserved pre-change code directory.

Runs both versions on identical in-memory chunks and local model weights. The
JSON report contains locations, scores and snippet hashes, never corpus text.
Model instances and identical query responses are reused: this is a quality
regression check, not a performance benchmark.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import mr_eval


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_stack(directory):
    names = ("mr_corpus", "mr_index", "mr_tools")
    previous = {name: sys.modules.get(name) for name in names}
    modules = {}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(name, directory / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[name] = module
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return SimpleNamespace(corpus=modules["mr_corpus"], index=modules["mr_index"], tools=modules["mr_tools"])


def safe_report(result):
    rows = []
    for row in result["rows"]:
        hits = []
        for hit in row["hits"]:
            safe = {k: v for k, v in hit.items() if k != "snippet"}
            safe["snippet_sha256"] = hashlib.sha256(hit["snippet"].encode()).hexdigest()
            hits.append(safe)
        rows.append(dict(row, hits=hits))
    return dict(result, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="Skip cross-encoder and PDF checks during development")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(MR_LOG="0", MR_PDF="0", MR_RERANK="", MR_RERANK_MODE="gate", MR_GATE="note",
                      MR_EMBED_MODEL="bge-small-zh-v1.5", MR_EMBED_MODEL_PDF="bge-m3",
                      MR_MODELS_DIR=str(CODE / ".cache"), MR_CACHE_DIR=str(args.cache),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    os.environ.pop("MR_EMBED_ALLOW_DOWNLOAD", None)
    for pattern in ("emb-*.npy", "emb_meta-*.json"):
        for original in (CODE / ".cache").glob(pattern):
            if not (args.cache / original.name).exists():
                shutil.copy2(original, args.cache / original.name)
    before, after = load_stack(args.baseline), load_stack(CODE)
    for stack in (before, after):
        stack.index.CACHE = str(args.cache)
        stack.corpus.PDF_CACHE = str(CODE / ".cache" / "pdf-text")
        stack.tools.LOG_ON = False
    protected = sorted((CODE / "golden").glob("*.json")) + sorted(CODE.glob("*.py")) + [CODE / "corpus-policy.json"]
    hashes = {str(p.relative_to(CODE)): digest(p) for p in protected}
    report = {"run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "quick": args.quick,
              "quality_only": True, "models": {"notes": "bge-small-zh-v1.5", "pdf": "bge-m3", "gate": "bge-reranker-base"},
              "baseline_code_sha256": {p.name: digest(p) for p in args.baseline.glob("*.py")},
              "current_sha256": hashes, "runs": {"before": {}, "after": {}}, "checks": {}}

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    original_vec, original_reranker = after.index.VecIndex, after.index.Reranker
    shared_vectors, shared_rerankers = {}, {}

    def vector_factory(chunks, model=None, spec=None):
        key = (spec["slug"], tuple(c["md5"] for c in chunks))
        if key not in shared_vectors:
            shared_vectors[key] = original_vec(chunks, model=model, spec=spec)
        return shared_vectors[key]

    def reranker_factory(spec):
        if spec["key"] not in shared_rerankers:
            shared_rerankers[spec["key"]] = original_reranker(spec)
        return shared_rerankers[spec["key"]]

    for stack in (before, after):
        stack.index.VecIndex = vector_factory
        stack.index.Reranker = reranker_factory

    chunks = after.corpus.load_chunks()
    all_goldens = mr_eval.load_goldens(CODE / "golden")
    report["corpus_audit"] = mr_eval.require_clean_corpus(chunks, all_goldens)
    live_signature = after.corpus.signature()

    def freeze(rows):
        fingerprint = hashlib.sha256("\n".join(c["source"] + "/" + c["md5"] for c in rows).encode()).hexdigest()
        for stack in (before, after):
            stack.tools._STATE.clear()
            stack.corpus.load_chunks = lambda: rows
            stack.corpus.signature = lambda: fingerprint
        return fingerprint

    report["corpus_sha256"] = freeze(chunks)
    report["corpora"] = {source: {"files": len({c["file"] for c in chunks if c["source"] == source}),
                                  "chunks": sum(c["source"] == source for c in chunks)}
                          for source in sorted({c["source"] for c in chunks})}
    report["purposes"] = dict(collections.Counter(c.get("purpose", "knowledge") for c in chunks))
    print(json.dumps({"corpora": report["corpora"], "sha256": report["corpus_sha256"], "purposes": report["purposes"]}, ensure_ascii=False), flush=True)
    cached = {"before": {}, "after": {}}

    def run(version, label, golden_file, mode="hybrid", rerank=False, model=None):
        stack = before if version == "before" else after
        golden = json.loads((CODE / "golden" / golden_file).read_text(encoding="utf-8"))

        def search(q, **kwargs):
            if version == "before":
                kwargs.pop("include_evaluation", None)  # The old version always includes evaluation documents.
            key = json.dumps([q, kwargs], sort_keys=True, ensure_ascii=False)
            if key not in cached[version]:
                cached[version][key] = stack.tools.memory_search(q, **kwargs)
            return cached[version][key]

        result = mr_eval.evaluate(golden, search, mode, model, rerank, chunks=stack.tools._STATE.get("chunks", active_chunks))
        report["runs"][version][label] = safe_report(result)
        print(json.dumps({"version": version, "suite": label, **result["summary"]}, ensure_ascii=False), flush=True)
        for row in result["rows"]:
            if row["passage_labeled"]:
                print(json.dumps({"id": row["id"], "version": version, "evidence_rank": row["evidence_rank"],
                                  "top": [{"source": h["source"], "file": h["file"], "heading": h["heading"]}
                                          for h in row["hits"][:3]]}, ensure_ascii=False), flush=True)
        save()

    active_chunks = chunks
    for version in ("before", "after"):
        for prefix, file in (("notes", "golden_queries.json"), ("stock", "golden_stock.json")):
            for mode in ("bm25", "vec", "hybrid"):
                run(version, prefix + "_" + mode, file, mode)
        run(version, "negative_hybrid", "golden_negative.json")
        run(version, "evidence_hybrid", "golden_evidence.json")
    # One-variable comparison: old code, same candidates/vectors/labels; replace only ordinary-note IDF.
    old_bm25 = before.tools._STATE["bm25"]
    note_idx = [i for i, c in enumerate(chunks) if c["source"] in {"case", "book", "mem"}]
    before.tools._STATE["bm25"] = before.tools._SubBM25(chunks, note_idx)
    golden = json.loads((CODE / "golden" / "golden_queries.json").read_text(encoding="utf-8"))
    report["isolated_statistics_only"] = safe_report(mr_eval.evaluate(golden, before.tools.memory_search, rerank=False, chunks=chunks))
    before.tools._STATE["bm25"] = old_bm25
    print(json.dumps({"isolated_statistics_only": report["isolated_statistics_only"]["summary"]}, ensure_ascii=False), flush=True)
    if not args.quick:
        for version in ("before", "after"):
            run(version, "notes_gate", "golden_queries.json", rerank="bge-reranker-base")
            run(version, "negative_gate", "golden_negative.json", rerank="bge-reranker-base")
            run(version, "evidence_gate", "golden_evidence.json", rerank="bge-reranker-base")
        # Load only the PDF population: its BM25 and vector routes are already source-masked.
        for path in after.corpus.pdf_files():
            pdf = Path(path)
            entry = json.loads((Path(after.corpus.PDF_CACHE) / (pdf.stem + ".json")).read_text(encoding="utf-8"))
            stat = pdf.stat()
            if entry.get("size") != stat.st_size or int(entry.get("mtime", -1)) != int(stat.st_mtime):
                raise RuntimeError("PDF extraction cache is stale")
        pdf_chunks = after.corpus.load_pdf_chunks()
        report["pdf_corpus_audit"] = mr_eval.require_clean_corpus(pdf_chunks, all_goldens)
        report["pdf_corpus_sha256"] = freeze(pdf_chunks)
        active_chunks = pdf_chunks
        report["pdf_chunks"] = len(pdf_chunks)
        cached = {"before": {}, "after": {}}
        for version in ("before", "after"):
            run(version, "pdf_m3", "golden_pdf.json", model="bge-m3")

    def summary(version, label):
        return report["runs"][version][label]["summary"]

    checks = report["checks"]
    checks["corpus_audit_passed"] = report["corpus_audit"]["ok"] and (args.quick or report["pdf_corpus_audit"]["ok"])
    checks["no_fallbacks"] = all(not r["summary"]["fallbacks"] for runs in report["runs"].values() for r in runs.values())
    checks["notes_hit5_preserved"] = summary("after", "notes_hybrid")["returned_hit_at5"] >= summary("before", "notes_hybrid")["returned_hit_at5"]
    checks["notes_mrr_improved"] = summary("after", "notes_hybrid")["returned_mrr_at10"] > summary("before", "notes_hybrid")["returned_mrr_at10"]
    checks["stock_rankings_preserved"] = all(
        [(r["returned_rank"], [(h["source"], h["file"], h["score"]) for h in r["hits"]]) for r in report["runs"]["before"]["stock_" + mode]["rows"]] ==
        [(r["returned_rank"], [(h["source"], h["file"], h["score"]) for h in r["hits"]]) for r in report["runs"]["after"]["stock_" + mode]["rows"]]
        for mode in ("bm25", "vec", "hybrid"))
    evidence = summary("after", "evidence_hybrid")
    checks["passage_hit5_complete"] = evidence["evidence_hit_at5"] == evidence["evidence_labeled"]
    checks["evaluation_notes_still_searchable"] = all(r["returned_rank"] and r["returned_rank"] <= 5
        for r in report["runs"]["after"]["evidence_hybrid"]["rows"] if (r["id"] or "").startswith("evaluation-"))
    if not args.quick:
        gate_evidence = summary("after", "evidence_gate")
        checks["passage_gate_hit5_complete"] = gate_evidence["evidence_hit_at5"] == gate_evidence["evidence_labeled"]
        checks["negative_high_reduced"] = summary("after", "negative_gate")["negative_high"] < summary("before", "negative_gate")["negative_high"]
        checks["weather_recipe_low"] = all(r["confidence"] == "low" for r in report["runs"]["after"]["evidence_gate"]["rows"] if not r["positive"])
        checks["positive_high_preserved"] = summary("after", "notes_gate")["positive_confidence"].get("high", 0) >= summary("before", "notes_gate")["positive_confidence"].get("high", 0)
        checks["pdf_hit5_preserved"] = summary("after", "pdf_m3")["returned_hit_at5"] >= summary("before", "pdf_m3")["returned_hit_at5"]
        checks["pdf_mrr_preserved"] = summary("after", "pdf_m3")["returned_mrr_at10"] >= summary("before", "pdf_m3")["returned_mrr_at10"]
    checks["code_and_labels_unchanged_during_run"] = all(digest(p) == hashes[str(p.relative_to(CODE))] for p in protected)
    # The live loader is independent of the frozen stacks used above.
    # Other live sessions may write notes while this test runs. That does not
    # invalidate a paired test: both versions used the SAME frozen rows above.
    # Surface the drift separately; the real MCP smoke test checks the live view.
    report["live_corpus_unchanged_during_run"] = load_stack(CODE).corpus.signature() == live_signature
    report["passed"] = all(checks.values())
    save()
    print(json.dumps({"passed": report["passed"], "checks": checks, "output": str(args.output)}, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
