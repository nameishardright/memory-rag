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
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    for pattern in ("emb-*.npy", "emb_meta-*.json"):
        for path in (CODE / ".cache").glob(pattern):
            if not (args.cache / path.name).exists():
                shutil.copy2(path, args.cache / path.name)
    env = dict(os.environ)
    env.update(MR_LOG="0", MR_PDF="1", MR_RERANK="bge-reranker-base", MR_RERANK_MODE="gate", MR_GATE="note",
               MR_EMBED_MODEL_PDF="bge-m3",
               MR_EMBED_MODEL="bge-small-zh-v1.5", MR_CACHE_DIR=str(args.cache), MR_MODELS_DIR=str(CODE / ".cache"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    env.pop("MR_EMBED_ALLOW_DOWNLOAD", None)
    env.pop("MR_WARM_HEAVY", None)
    evidence = json.loads((CODE / "golden" / "golden_evidence.json").read_text(encoding="utf-8"))
    stock = json.loads((CODE / "golden" / "golden_stock.json").read_text(encoding="utf-8"))
    pdf = json.loads((CODE / "golden" / "golden_pdf.json").read_text(encoding="utf-8"))
    report = {"run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "transport": "real stdio JSON-RPC",
              "checks": {}, "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.glob("*.py")}}
    checks = report["checks"]

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with (args.output.parent / "mcp-stderr.log").open("w", encoding="utf-8") as stderr:
        process_started = time.perf_counter()
        process = subprocess.Popen([sys.executable, "-X", "utf8", "-u", str(CODE / "mr_mcp_server.py")],
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

            found = call("memory_search", {"query": stock["queries"][0]["q"], "source": "stock", "mode": "bm25", "topk": 1, "rerank": False})
            report["latency"] = {"first_search_ms": round((time.perf_counter() - process_started) * 1000, 2)}
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
            timed_cases = [direct] + [q for q in evidence["queries"] if not q["expect"]]
            for q in timed_cases:
                call("memory_search", {"query": q["q"], "topk": 5, "rerank": True})
            for repeat in range(2):
                for q in timed_cases:
                    start = time.perf_counter()
                    found = call("memory_search", {"query": q["q"], "topk": 5, "rerank": True})
                    rr = found.get("rerank") or {}
                    latency_rows.append({"id": q["id"], "repeat": repeat + 1,
                                         "total_ms": round((time.perf_counter() - start) * 1000, 2),
                                         "rerank_ms": rr.get("ms"), "pairs": rr.get("pairs"),
                                         "extra_passages": rr.get("extra_passages", 0), "confidence": found["confidence"]})
            elapsed = sorted(row["total_ms"] for row in latency_rows)
            p50, p95 = statistics.median(elapsed), elapsed[math.ceil(.95 * len(elapsed)) - 1]
            budget_ok = p50 <= 2000 and p95 <= 4000 and report["latency"]["first_search_ms"] <= 60000
            report["latency"].update({"warm_samples": len(elapsed), "warm_p50_ms": round(p50, 2),
                                      "warm_p95_ms": round(p95, 2), "samples": latency_rows,
                                      "budget": {"warm_p50_ms": 2000, "warm_p95_ms": 4000, "first_search_ms": 60000},
                                      "budget_status": "达标" if budget_ok else "未达标，性能优化待办"})
            print(json.dumps({"latency": {k: v for k, v in report["latency"].items() if k != "samples"}}, ensure_ascii=False), flush=True)
            checks["code_unchanged_during_mcp_test"] = all(hashlib.sha256((CODE / name).read_bytes()).hexdigest() == digest for name, digest in report["code_sha256"].items())
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
