# -*- coding: utf-8 -*-
"""CLI 入口。
  py -3.12 mr.py stats                          # 语料/索引状态
  py -3.12 mr.py search "查询" [--mode hybrid|bm25|vec] [--topk 5] [--source case|book|mem]
  py -3.12 mr.py eval [--mode all|bm25|vec|hybrid] [-v]   # 金标评测:recall@5 / MRR@10 / 延迟
  py -3.12 mr.py setup-vec                      # 装向量依赖(清华镜像)+预拉 bge 模型
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import mr_tools  # noqa: E402
import mr_index  # noqa: E402


def cmd_stats():
    st = mr_tools._ensure("hybrid")
    per = {}
    for c in st["chunks"]:
        per.setdefault(c["source"], set()).add(c["file"])
    print("语料: %d chunks" % len(st["chunks"]))
    for k, files in sorted(per.items()):
        n = sum(1 for c in st["chunks"] if c["source"] == k)
        print("  %-5s %3d 文件 %4d chunks" % (k, len(files), n))
    print("分词器: %s" % mr_index.TOKENIZER)
    v = st.get("vec")
    print("向量层: %s" % ("可用(%s)" % mr_index.MODEL if v and v.available else "不可用 — %s" % getattr(v, "err", "")))


def cmd_search(q, mode, topk, source):
    t0 = time.time()
    r = mr_tools.memory_search(q, mode=mode, topk=topk, source=source)
    print("[%s] tokenizer=%s %.0fms  %s" % (r["mode"], r["tokenizer"], (time.time() - t0) * 1000, r["note"]))
    for h in r["hits"]:
        print("%2d. %.4f [%s] %s :: %s" % (h["rank"], h["score"], h["source"], h["file"], h["heading"]))
        print("      %s" % h["snippet"][:120].replace("\n", " "))


def cmd_eval(mode, verbose):
    with open(os.path.join(BASE, "golden", "golden_queries.json"), encoding="utf-8") as f:
        golden = json.load(f)
    qs = golden["queries"]
    modes = ["bm25", "vec", "hybrid"] if mode == "all" else [mode]
    print("金标 %d 条(draft=%s) | recall@5=前5含期望文件 | MRR@10=首个命中名次倒数均值" % (len(qs), golden.get("draft")))
    print("%-8s %-9s %-7s %s" % ("mode", "recall@5", "MRR@10", "avg延迟"))
    for m in modes:
        probe = mr_tools.memory_search("探针", mode=m, topk=1)
        if probe["mode"].endswith("(fallback)") and m != "bm25":
            print("%-8s 跳过 — %s" % (m, probe["note"]))
            continue
        hit5, rr, t0, misses = 0, 0.0, time.time(), []
        for g in qs:
            r = mr_tools.memory_search(g["q"], mode=m, topk=10)
            files, seen = [], set()
            for h in r["hits"]:
                if h["file"] not in seen:
                    seen.add(h["file"])
                    files.append(h["file"])
            rank = next((i + 1 for i, f in enumerate(files) if f in g["expect"]), None)
            if rank and rank <= 5:
                hit5 += 1
            if rank:
                rr += 1.0 / rank
            else:
                misses.append((g["q"], files[:3]))
        n = len(qs)
        print("%-8s %2d/%-6d %.3f   %4.0fms" % (m, hit5, n, rr / n, (time.time() - t0) * 1000 / n))
        if verbose and misses:
            for q, top in misses:
                print("    miss: %s -> 实际top3=%s" % (q, top))


def cmd_ranks():
    """逐题×逐模式:期望文件的名次矩阵(归因用——miss 只是名次>10 的特例)。"""
    with open(os.path.join(BASE, "golden", "golden_queries.json"), encoding="utf-8") as f:
        golden = json.load(f)
    qs = golden["queries"]
    modes = ["bm25", "vec", "hybrid"]
    print("tokenizer=%s | 名次=期望文件首次出现的去重文件位;'-'=前10没有" % mr_index.TOKENIZER)
    print("q#  bm25  vec  hyb | query")
    for gi, g in enumerate(qs, 1):
        cells, miss_tops = [], []
        for m in modes:
            r = mr_tools.memory_search(g["q"], mode=m, topk=10)
            if r["mode"].endswith("(fallback)") and m != "bm25":
                cells.append("×")
                continue
            files = []
            for h in r["hits"]:
                if h["file"] not in files:
                    files.append(h["file"])
            rank = next((i + 1 for i, f in enumerate(files) if f in g["expect"]), None)
            cells.append(str(rank) if rank else "-")
            if not rank and files:
                miss_tops.append("%s@%s" % (m, files[0]))
        line = "%2d  %4s %4s %4s | %s" % (gi, cells[0], cells[1], cells[2], g["q"][:26])
        if miss_tops:
            line += "  ← " + " ".join(dict.fromkeys(miss_tops))
        print(line)


def cmd_setup_vec():
    print("装依赖(清华镜像,含 torch CPU,几百 MB,耐心)…")
    rc = subprocess.call([sys.executable, "-m", "pip", "install", "-q",
                          "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
                          "sentence-transformers", "jieba"])
    if rc != 0:
        print("pip 失败 rc=%d,检查网络/镜像" % rc)
        return
    print("预拉模型 %s(走 hf-mirror.com)…" % mr_index.MODEL)
    st = mr_tools._ensure("vec")
    v = st["vec"]
    print("向量层: %s" % ("OK,已建 %d 条 embedding 缓存" % len(st["chunks"]) if v.available else "失败 — %s" % v.err))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--mode", default="hybrid", choices=["bm25", "vec", "hybrid"])
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--source", default=None, choices=["case", "book", "mem"])
    p = sub.add_parser("eval")
    p.add_argument("--mode", default="all", choices=["all", "bm25", "vec", "hybrid"])
    p.add_argument("-v", action="store_true")
    sub.add_parser("ranks")
    p = sub.add_parser("get")           # 二段式第二步:search 定位→get 全文(别只吃 snippet,截断缝合教训)
    p.add_argument("file")
    sub.add_parser("setup-vec")
    a = ap.parse_args()
    if a.cmd == "stats":
        cmd_stats()
    elif a.cmd == "search":
        cmd_search(a.query, a.mode, a.topk, a.source)
    elif a.cmd == "eval":
        cmd_eval(a.mode, a.v)
    elif a.cmd == "ranks":
        cmd_ranks()
    elif a.cmd == "get":
        r = mr_tools.memory_get(a.file)
        print(r["text"] if r.get("ok") else r.get("error"))
    elif a.cmd == "setup-vec":
        cmd_setup_vec()
