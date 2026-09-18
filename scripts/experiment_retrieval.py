"""Frozen-corpus chunk and gate ablations. Outputs contain private source labels.

Pair-score caching is for quality comparison only, never a latency benchmark.
Run latency independently with uncached real searches after choosing a profile.
"""
import argparse
import collections
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from scripts.verify_retrieval import load_stack
import mr_eval


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("chunks", "gates"), required=True)
    parser.add_argument("--chunker", choices=("character", "paragraph"), default="paragraph")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(MR_LOG="0", MR_PDF="0", MR_RERANK="", MR_GATE="note", MR_RERANK_MODE="gate",
                      MR_MODELS_DIR=str(args.models), MR_CACHE_DIR=str(args.cache),
                      MR_CORPUS_POLICY=str(args.frozen / "corpus-policy.json"), MR_CHUNK_STRATEGY=args.chunker,
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    for pattern in ("emb-*.npy", "emb_meta-*.json"):
        for path in args.models.glob(pattern):
            if not (args.cache / path.name).exists():
                shutil.copy2(path, args.cache / path.name)
    original, current = load_stack(args.baseline), load_stack(CODE)
    for stack in (original, current):
        for source in stack.corpus.SOURCES:
            source["dir"] = str(args.frozen / source["key"])
        stack.corpus.PDF_CACHE = str(args.models / "pdf-text")
        stack.tools.LOG_ON = False
    goldens = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in {
        "notes": CODE / "golden/golden_queries.json", "stock": CODE / "golden/golden_stock.json",
        "negative": CODE / "golden/golden_negative.json", "evidence": args.evidence,
        "original_evidence": CODE / "golden/golden_evidence.json"}.items()}
    report = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stage": args.stage,
              "quality_only": True, "evidence_sha256": sha(args.evidence),
              "code_sha256": {p.name: sha(p) for p in CODE.glob("*.py")},
              "frozen_policy_sha256": sha(args.frozen / "corpus-policy.json"), "runs": {}}
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    models, vectors = {}, {}
    real_vec = current.index.VecIndex
    def vec_factory(chunks, model=None, spec=None):
        key = (spec["slug"], tuple(c["md5"] for c in chunks))
        if key not in vectors:
            v = real_vec(chunks, model=model or models.get(spec["slug"]), spec=spec)
            if not v.available:
                raise RuntimeError(v.err)
            vectors[key] = v
            models[spec["slug"]] = v._get_model()
        return vectors[key]
    for stack in (original, current):
        stack.index.VecIndex = vec_factory

    def prepare(stack):
        chunks = stack.corpus.load_chunks()
        mr_eval.require_clean_corpus(chunks, list(goldens.values()))
        corpus_sha = hashlib.sha256(json.dumps([(c["source"],c["md5"],c.get("purpose")) for c in chunks]).encode()).hexdigest()
        stack.corpus.load_chunks = lambda: chunks
        stack.corpus.signature = lambda: corpus_sha
        stack.tools._STATE.clear()
        return chunks, corpus_sha

    def run_suite(stack, chunks, version, suite, mode="hybrid", rerank=False):
        result = mr_eval.evaluate(goldens[suite], stack.tools.memory_search, mode=mode, rerank=rerank, chunks=chunks)
        if result["summary"]["fallbacks"]:
            raise RuntimeError("An ablation silently fell back")
        report["runs"].setdefault(version, {})[suite + "_" + mode] = result
        print(json.dumps({"version":version,"suite":suite+"_"+mode,**result["summary"]},ensure_ascii=False),flush=True)
        save()
        return result

    if args.stage == "chunks":
        for name, stack in (("character", original), ("paragraph", current)):
            chunks, fingerprint = prepare(stack)
            report.setdefault("corpora", {})[name] = {"sha256":fingerprint,"chunks":len(chunks),
                "sources":dict(collections.Counter(c["source"] for c in chunks))}
            for suite in ("notes", "stock"):
                for mode in ("bm25", "vec", "hybrid"):
                    run_suite(stack, chunks, name, suite, mode)
            for suite in ("evidence", "negative", "original_evidence"):
                run_suite(stack, chunks, name, suite)
        # A notes-boundary change also triggers stock-IDF review. This is an
        # experiment, not an automatic mode switch on a draft 12-question set.
        saved = current.tools._STATE.get("bm25_nonpdf")
        current.tools._STATE["bm25_nonpdf"] = current.tools._SubBM25(chunks,
            [i for i,c in enumerate(chunks) if c["source"] == "stock"])
        for mode in ("bm25", "hybrid"):
            run_suite(current, chunks, "stock_only_idf", "stock", mode)
        current.tools._STATE["bm25_nonpdf"] = saved
    else:
        chunks, fingerprint = prepare(current)
        report["corpus_sha256"] = fingerprint
        real_reranker = current.index.Reranker
        live = real_reranker(current.index.reranker_spec("bge-reranker-base"))
        if not live.available:
            raise RuntimeError(live.err)
        cache_file = args.cache / "pair-scores.private.json"
        cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
        weights_sha = sha(args.models / "bge-reranker-base/model.safetensors")
        class CachedReranker:
            available = True
            err = ""
            def __init__(self, spec):
                self.spec = dict(spec)
                self.spec.setdefault("max_length",512)
                # Deliberately provisional while comparing parameter tuples.
                self.spec.update(hi=0.5,lo=0.1)
            def score(self, query, passages):
                length = self.spec["max_length"]
                keys = [hashlib.sha256(json.dumps([weights_sha,length,query,p],ensure_ascii=False).encode()).hexdigest()
                        for p in passages]
                missing = list(dict.fromkeys(i for i,k in enumerate(keys) if k not in cache))
                if missing:
                    encoder = live._get()
                    live.spec["max_length"] = length
                    encoder.max_seq_length = length
                    if encoder.max_seq_length != length:
                        raise RuntimeError("CrossEncoder truncation did not change")
                    scores = live.score(query,[passages[i] for i in missing])
                    for i,score in zip(missing,scores):
                        cache[keys[i]] = score
                    cache_file.write_text(json.dumps(cache),encoding="utf-8")
                return [cache[k] for k in keys]
        current.index.Reranker = CachedReranker
        original.index.Reranker = CachedReranker
        baseline_chunks, baseline_fingerprint = prepare(original)
        report["baseline_corpus_sha256"] = baseline_fingerprint
        for suite in ("evidence","negative","original_evidence"):
            run_suite(original,baseline_chunks,"baseline-character",suite,rerank="bge-reranker-base")
        # Full factorial, changing only these three knobs. Thresholds are the
        # OLD provisional comparison thresholds; calibrate only the final choice.
        for length, candidates, windows in itertools.product((512,256),(5,3),(3,2)):
            os.environ["MR_RERANK_MAX_LENGTH"] = str(length)
            os.environ["MR_RERANK_G"] = str(candidates)
            os.environ["MR_RERANK_MAXWIN"] = str(windows)
            os.environ["MR_RERANK_FALLBACK_G"] = str(candidates)  # Static factorial: adaptive rescue is a separate experiment.
            current.tools.RERANK_G, current.tools.RERANK_MAXWIN = candidates, windows
            current.tools._STATE["rerankers"] = {}
            name = f"l{length}-g{candidates}-w{windows}"
            for suite in ("evidence","negative","original_evidence"):
                run_suite(current,chunks,name,suite,rerank="bge-reranker-base")
            report.setdefault("profiles",{})[name] = {"max_length":length,"candidates":candidates,"max_windows":windows,
                "thresholds":"provisional old 0.50/0.10; not final calibration"}
            save()
    report["completed"] = True
    save()


if __name__ == "__main__":
    main()
