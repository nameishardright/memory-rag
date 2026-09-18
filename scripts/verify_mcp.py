"""Exercise the real stdio server against local data and isolated writable caches."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import mr_eval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=CODE / ".cache")
    parser.add_argument("--frozen", type=Path, help="Optional Markdown source snapshot; the child patches sources before starting the real server")
    parser.add_argument("--first-search",choices=("stock-bm25","hybrid"),default="stock-bm25",
                        help="Measure either the legacy first stock BM25 call or a full default hybrid plus gate call")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    for pattern in ("emb-*.npy", "emb_meta-*.json"):
        for path in args.models.glob(pattern):
            if not (args.cache / path.name).exists():
                shutil.copy2(path, args.cache / path.name)
    env = dict(os.environ)
    env.update(MR_LOG="0", MR_PDF="1", MR_RERANK="bge-reranker-base", MR_RERANK_MODE="gate", MR_GATE="note",
               MR_EMBED_MODEL_PDF="bge-m3",
               MR_EMBED_MODEL="bge-small-zh-v1.5", MR_CACHE_DIR=str(args.cache), MR_MODELS_DIR=str(args.models.resolve()),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    env.pop("MR_EMBED_ALLOW_DOWNLOAD", None)
    env.pop("MR_WARM_HEAVY", None)
    env["MR_VERIFY_PDF_CACHE"] = str((args.models / "pdf-text").resolve())
    if args.frozen:
        env["MR_VERIFY_FROZEN"] = str(args.frozen.resolve())
        env["MR_CORPUS_POLICY"] = str((args.frozen / "corpus-policy.json").resolve())
    evidence = json.loads((CODE / "golden" / "golden_evidence.json").read_text(encoding="utf-8"))
    stock = json.loads((CODE / "golden" / "golden_stock.json").read_text(encoding="utf-8"))
    pdf = json.loads((CODE / "golden" / "golden_pdf.json").read_text(encoding="utf-8"))
    notes = json.loads((CODE / "golden/golden_queries.json").read_text(encoding="utf-8"))
    negatives = json.loads((CODE / "golden/golden_negative.json").read_text(encoding="utf-8"))
    report = {"run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "transport": "real stdio JSON-RPC",
              "checks": {}, "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.glob("*.py")}}
    checks = report["checks"]

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with (args.output.parent / "mcp-stderr.log").open("w", encoding="utf-8") as stderr:
        process_started = time.perf_counter()
        bootstrap = ("import os,runpy,mr_corpus; "
                     "mr_corpus.PDF_CACHE=os.environ['MR_VERIFY_PDF_CACHE']; "
                     "snapshot=os.environ.get('MR_VERIFY_FROZEN'); "
                     "[s.update(dir=os.path.join(snapshot,s['key'])) for s in mr_corpus.SOURCES] if snapshot else None; "
                     "runpy.run_path('mr_mcp_server.py',run_name='__main__')")
        process = subprocess.Popen([sys.executable, "-X", "utf8", "-u", "-c", bootstrap],
                                   cwd=CODE, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=stderr, text=True, encoding="utf-8")
        received = queue.Queue()

        def read_lines():
            for line in process.stdout:
                received.put(line)
            received.put(None)

        reader = threading.Thread(target=read_lines, daemon=True)
        reader.start()
        request_id = 0
        corpus_observations = []

        def request(method, params):
            nonlocal request_id
            request_id += 1
            process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}, ensure_ascii=False) + "\n")
            process.stdin.flush()
            line = received.get(timeout=240)
            if line is None:
                raise RuntimeError("MCP server closed stdout")
            reply = json.loads(line)
            assert reply.get("jsonrpc") == "2.0" and reply.get("id") == request_id and "error" not in reply
            return reply["result"]

        def call(name, arguments):
            result = request("tools/call", {"name": name, "arguments": arguments})
            assert not result.get("isError"), "MCP tool returned isError"
            data = json.loads(result["content"][0]["text"])
            assert data["ok"]
            if name == "memory_search":
                assert data["mode"] == arguments.get("mode", "hybrid"), "MCP silently fell back"
                corpus_observations.append({"request_id":request_id,"sha256":data.get("corpus_sha256"),
                                            "audit":data.get("corpus_audit")})
            return data

        try:
            initialized = request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "rag-regression", "version": "1"}})
            checks["initialize"] = initialized["serverInfo"]["name"] == "rh-memory"
            process.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
            process.stdin.flush()
            listed = request("tools/list", {})
            schemas = {t["name"]: t["inputSchema"] for t in listed["tools"]}
            checks["stock_in_both_schemas"] = all("stock" in schemas[name]["properties"]["source"]["enum"] for name in ("memory_search", "memory_get"))
            checks["evaluation_option_declared"] = "include_evaluation" in schemas["memory_search"]["properties"]
            checks["passage_get_declared"] = "heading" in schemas["memory_get"]["properties"]
            report["schemas"] = schemas
            save()
            print("MCP initialize/tools-list: stock, evaluation scope and paragraph lookup declared", flush=True)

            if args.first_search=="hybrid":
                first=call("memory_search",{"query":notes["queries"][0]["q"],"topk":5,"rerank":True})
                report["latency"]={"first_search_ms":round((time.perf_counter()-process_started)*1000,2),
                                   "first_search_kind":"new stdio process + default hybrid + base gate; PDF loaded",
                                   "first_search_rerank":first.get("rerank")}
            found = call("memory_search", {"query": stock["queries"][0]["q"], "source": "stock", "mode": "bm25", "topk": 1, "rerank": False})
            if args.first_search=="stock-bm25":
                report["latency"] = {"first_search_ms": round((time.perf_counter() - process_started) * 1000, 2),
                                     "first_search_kind":"new stdio process + stock BM25 rerank=false; includes server warmup"}
            selected = found["hits"][0]
            checks["stock_search"] = selected["source"] == "stock"
            fetched = call("memory_get", {"file": selected["file"], "source": "stock"})
            checks["stock_get"] = fetched["source"] == "stock" and len(fetched["text"]) > 100
            report["stock_result"] = {"source": selected["source"], "file": selected["file"], "text_sha256": hashlib.sha256(fetched["text"].encode()).hexdigest()}
            print("MCP stock search/get: passed", flush=True)

            found = call("memory_search", {"query": pdf["queries"][0]["q"], "source": "pdf", "mode": "bm25", "topk": 1, "rerank": False})
            selected = found["hits"][0]
            page = int(re.match(r"p(\d+)", selected["heading"])[1])
            fetched = call("memory_get", {"file": selected["file"] + "#p%d" % page, "source": "pdf"})
            checks["pdf_search_get_route"] = selected["source"] == fetched["source"] == "pdf" and fetched["page"] == page and bool(fetched["text"])
            report["pdf_result"] = {"mode": "bm25", "file": selected["file"], "page": page,
                                    "text_sha256": hashlib.sha256(fetched["text"].encode()).hexdigest()}
            print("MCP PDF search/get: page route verified (bge-m3 quality checked in paired suite)", flush=True)
            example = call("memory_get", {"file": "从零开始构建智能体-Hello-Agents-V1.0.3.pdf#p423", "source": "pdf"})
            checks["pdf_example_remains_accessible_and_labeled"] = example.get("purpose") == "evaluation"

            direct = next(q for q in evidence["queries"] if q["id"] == "passage-direct-submit")
            found = call("memory_search", {"query": direct["q"], "topk": 5, "rerank": False})
            checks["direct_submit_evidence_at1"] = mr_eval.grade_hits(direct, found["hits"])["evidence_rank"] == 1
            selected = found["hits"][0]
            fetched = call("memory_get", {"source": selected["source"], "file": selected["file"], "heading": selected["heading"]})
            checks["get_exact_answer_paragraph"] = (fetched["heading"] == selected["heading"] and
                any(mr_eval.matches_passage(dict(fetched, snippet=fetched["text"]), label) for label in direct["evidence"]))
            prefixed = call("memory_get", {"source": selected["source"], "file": selected["file"], "heading": selected["heading"][:-1]})
            checks["get_unique_prefix"] = prefixed["heading"] == selected["heading"] and prefixed["text"] == fetched["text"]
            report["direct_submit"] = {"source": selected["source"], "file": selected["file"], "heading": selected["heading"],
                                        "snippet_sha256": hashlib.sha256(selected["snippet"].encode()).hexdigest()}
            print("MCP direct-submit search/get: answer paragraph verified", flush=True)

            report["negative_confidence"] = {}
            suffix_hit = None
            for q in evidence["queries"]:
                if q["expect"]:
                    continue
                found = call("memory_search", {"query": q["q"], "topk": 5, "rerank": True})
                checks[q["id"] + "_low"] = found["confidence"] == "low"
                checks[q["id"] + "_no_evaluation_hits"] = all(h["purpose"] != "evaluation" for h in found["hits"])
                report["negative_confidence"][q["id"]] = {"confidence": found["confidence"], "gate": found["gate"]}
                suffix_hit = suffix_hit or next((h for h in found["hits"] if h["source"] == "book" and re.search(r" \(\d+\)$", h["heading"])), None)
                print("MCP %s: confidence=%s" % (q["id"], found["confidence"]), flush=True)

            for q in evidence["queries"]:
                if not q.get("include_evaluation"):
                    continue
                found = call("memory_search", {"query": q["q"], "source": q["source"], "include_evaluation": True,
                                               "topk": 5, "rerank": False})
                checks[q["id"]] = any(h["file"] in q["expect"] and h["purpose"] == "evaluation" for h in found["hits"])
            if suffix_hit is None:
                found = call("memory_search", {"query": "向量 混合检索", "source": "book", "mode": "bm25", "topk": 10, "rerank": False})
                suffix_hit = next((h for h in found["hits"] if re.search(r" \(\d+\)$", h["heading"])), None)
            checks["get_suffix_sample_available"] = suffix_hit is not None
            if suffix_hit:
                prefix = re.sub(r" \(\d+\)$", "", suffix_hit["heading"]).rstrip()
                reply = request("tools/call", {"name": "memory_get", "arguments": {
                    "source": suffix_hit["source"], "file": suffix_hit["file"], "heading": prefix}})
                data = json.loads(reply["content"][0]["text"])
                checks["get_omitted_suffix_resolves_or_lists_candidates"] = (
                    data.get("ok") and data.get("heading") == suffix_hit["heading"] or
                    bool(reply.get("isError")) and suffix_hit["heading"] in data.get("candidates", []))
                exact = call("memory_get", {"source": suffix_hit["source"], "file": suffix_hit["file"], "heading": suffix_hit["heading"]})
                checks["get_suffix_exact_title_still_works"] = exact["heading"] == suffix_hit["heading"]

            # Sequential warm observations with no result cache in this client.
            # Run this verifier separately from the paired quality benchmark.
            latency_rows = []
            timed_cases = [dict(notes["queries"][i],id="p%02d" % (i+1)) for i in (0,1,3,7,8,13,18,22)]
            timed_cases += [dict(negatives["queries"][i],id="n%02d" % (i+1)) for i in (0,3,5,9)]
            for q in timed_cases:
                call("memory_search", {"query": q["q"], "topk": 5, "rerank": True})
            for repeat in range(3):
                for q in timed_cases:
                    start = time.perf_counter()
                    found = call("memory_search", {"query": q["q"], "topk": 5, "rerank": True})
                    rr = found.get("rerank") or {}
                    latency_rows.append({"id": q["id"], "repeat": repeat + 1,
                                         "total_ms": round((time.perf_counter() - start) * 1000, 2),
                                         "rerank_ms": rr.get("ms"), "pairs": rr.get("pairs"),
                                         "max_length":rr.get("max_length"),"initial_candidates":rr.get("initial_candidates"),
                                         "max_windows":rr.get("max_windows"),"scored_candidates":rr.get("n"),
                                         "expanded_candidates":rr.get("expanded_candidates",0),
                                         "extra_passages": rr.get("extra_passages", 0), "confidence": found["confidence"],
                                         "corpus_sha256":found.get("corpus_sha256")})
            elapsed = sorted(row["total_ms"] for row in latency_rows)
            p50, p95 = statistics.median(elapsed), elapsed[math.ceil(.95 * len(elapsed)) - 1]
            budget_ok = p50 <= 2000 and p95 <= 4000 and report["latency"]["first_search_ms"] <= 60000
            report["latency"].update({"warm_samples": len(elapsed), "warm_p50_ms": round(p50, 2),
                                      "warm_p95_ms": round(p95, 2), "samples": latency_rows,
                                      "distinct_queries":len(timed_cases), "quantile_method":"nearest-rank; repeated queries are not independent population samples",
                                      "budget": {"warm_p50_ms": 2000, "warm_p95_ms": 4000, "first_search_ms": 60000},
                                      "budget_status": "达标" if budget_ok else "未达标，性能优化待办"})
            print(json.dumps({"latency": {k: v for k, v in report["latency"].items() if k != "samples"}}, ensure_ascii=False), flush=True)
            checks["code_unchanged_during_mcp_test"] = all(hashlib.sha256((CODE / name).read_bytes()).hexdigest() == digest for name, digest in report["code_sha256"].items())
            report["corpus_observations"] = corpus_observations
            fingerprints = {r["sha256"] for r in corpus_observations}
            report["corpus_stable_during_mcp"] = bool(corpus_observations) and None not in fingerprints and len(fingerprints) == 1
            checks["each_search_reports_corpus_identity"] = None not in fingerprints
            checks["ordinary_note_budget_visible_in_real_mcp"] = all(
                row["max_length"]==256 and row["initial_candidates"]==3 and row["max_windows"]==2
                and row["scored_candidates"]<=5 for row in latency_rows)
            if args.frozen:
                checks["frozen_mcp_corpus_stable"] = report["corpus_stable_during_mcp"]
            report["passed"] = all(checks.values())
            save()
            print(json.dumps({"passed": report["passed"], "checks": checks, "output": str(args.output)}, ensure_ascii=False), flush=True)
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()  # Only the test-owned child; no existing user sessions.
                process.wait(timeout=15)
            process.stdout.close()
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
