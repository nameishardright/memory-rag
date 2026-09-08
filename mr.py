# -*- coding: utf-8 -*-
"""CLI 入口。
  py -3.12 mr.py stats                          # 语料/索引状态
  py -3.12 mr.py search "查询" [--mode hybrid|bm25|vec] [--topk 5] [--source case|book|mem] [--model bge-m3]
  py -3.12 mr.py eval [--mode all|bm25|vec|hybrid] [-v] [--model bge-m3]   # 金标评测:recall@5 / MRR@10 / 延迟
  py -3.12 mr.py setup-vec                      # 装向量依赖(清华镜像)+预拉默认模型
  py -3.12 mr.py pull-model bge-m3              # 从 hf-mirror 拉某个预设模型的权重到 .cache/<名>/(可断点续传)
  embedding 模型一键切换:--model(单次)/ 环境变量 MR_EMBED_MODEL(进程默认,MCP 用这个)/ memory_search(model=)(单次查询)
"""
import argparse
import json
import re
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import mr_corpus
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
    cur = mr_index.model_spec()
    print("向量层: %s" % ("可用(%s)" % cur["key"] if v and v.available else "不可用 — %s" % getattr(v, "err", "")))
    pdf_m = os.environ.get("MR_EMBED_MODEL_PDF")
    print("embedding 模型(默认=%s,MR_EMBED_MODEL 可改;source=pdf 走 %s;* = 当前默认):" % (
        cur["key"], mr_index.model_spec(pdf_m)["key"] if pdf_m else "默认(MR_EMBED_MODEL_PDF 可改,推荐 bge-m3)"))
    for name in mr_index.MODELS:
        sp = mr_index.model_spec(name)
        emb = os.path.join(mr_index.CACHE, "emb-%s.npy" % sp["slug"])
        print("  %s%-22s 权重:%s  缓存:%s  %s" % ("*" if name == cur["key"] else " ", name,
              "有" if mr_index.has_weights(sp) else "无(pull-model %s)" % name,
              "有" if os.path.exists(emb) else "无", sp["note"]))


def cmd_search(q, mode, topk, source, model=None):
    t0 = time.time()
    r = mr_tools.memory_search(q, mode=mode, topk=topk, source=source, model=model)
    print("[%s] tokenizer=%s model=%s %.0fms  %s" % (r["mode"], r["tokenizer"], r.get("model") or "-", (time.time() - t0) * 1000, r["note"]))
    for h in r["hits"]:
        tag = ("  [%s]" % h["legs"]) if h.get("legs") else ""
        print("%2d. %.4f [%s] %s :: %s%s" % (h["rank"], h["score"], h["source"], h["file"], h["heading"], tag))
        print("      %s" % h["snippet"][:120].replace("\n", " "))


def cmd_eval(mode, verbose, golden_path=None, model=None):
    with open(golden_path or os.path.join(BASE, "golden", "golden_queries.json"), encoding="utf-8") as f:
        golden = json.load(f)
    qs = golden["queries"]
    modes = ["bm25", "vec", "hybrid"] if mode == "all" else [mode]
    print("金标 %d 条(draft=%s) | 模型 %s | recall@5=前5含期望文件 | MRR@10=首个命中名次倒数均值" % (
        len(qs), golden.get("draft"), mr_index.model_spec(model)["key"]))
    print("%-8s %-9s %-7s %s" % ("mode", "recall@5", "MRR@10", "avg延迟"))
    for m in modes:
        probe = mr_tools.memory_search("探针", mode=m, topk=1, model=model)
        if probe["mode"].endswith("(fallback)") and m != "bm25":
            print("%-8s 跳过 — %s" % (m, probe["note"]))
            continue
        hit5, rr, t0, misses = 0, 0.0, time.time(), []
        for g in qs:
            r = mr_tools.memory_search(g["q"], mode=m, topk=10, source=golden.get("source") or g.get("source"), model=model)
            files, seen = [], set()
            for h in r["hits"]:
                # 有 expect_pages 时按「文件#页」去重(同一本 PDF 不同页各算一条),否则按文件
                key = h["file"] + ("#" + re.sub(r" \(\d+\)$", "", h["heading"] or "") if g.get("expect_pages") else "")
                if key not in seen:
                    seen.add(key)
                    files.append((h["file"], h["heading"]))
            # expect_pages=[lo,hi]:PDF 源用,命中文件还得页码落在区间(heading 形如 p81 / p81 (2))
            def ok(f, head):
                if f not in g["expect"]:
                    return False
                if g.get("expect_pages"):
                    mm = re.match(r"p(\d+)", head or "")
                    return bool(mm) and g["expect_pages"][0] <= int(mm.group(1)) <= g["expect_pages"][1]
                return True
            rank = next((i + 1 for i, (f, hd) in enumerate(files) if ok(f, hd)), None)
            if rank and rank <= 5:
                hit5 += 1
            if rank:
                rr += 1.0 / rank
            else:
                misses.append((g["q"], [f + "#" + (hd or "") for f, hd in files[:3]]))
        n = len(qs)
        print("%-8s %2d/%-6d %.3f   %4.0fms" % (m, hit5, n, rr / n, (time.time() - t0) * 1000 / n))
        if verbose and misses:
            for q, top in misses:
                print("    miss: %s -> 实际top3=%s" % (q, top))


def cmd_ranks(model=None):
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
            r = mr_tools.memory_search(g["q"], mode=m, topk=10, model=model)
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
    cmd_pull_model(mr_index.model_spec()["key"])
    st = mr_tools._ensure("vec")
    v = st["vec"]
    print("向量层: %s" % ("OK,已建 %d 条 embedding 缓存" % len(st["chunks"]) if v.available else "失败 — %s" % v.err))


def cmd_pull_model(name):
    """从 HF 镜像把预设模型的权重和配置拉到 .cache/<名>/。断点续传(curl -C -),已完整的文件跳过。
    🔴 镜像对不存在的文件回 15 字节的 "Entry not found" 而不是 404 报错——不查的话会留下一个假权重文件,
       之后 sentence-transformers 报的错跟"没权重"完全不像(2026-09-08 bge-m3 踩过:它仓里只有 pytorch_model.bin)。"""
    sp = mr_index.model_spec(name)
    base = "%s/%s/resolve/main/" % (os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/"), sp["id"])
    os.makedirs(os.path.join(sp["local"], "1_Pooling"), exist_ok=True)
    print("拉 %s → %s" % (sp["id"], sp["local"]))
    for f in mr_index.COMMON_FILES + list(sp.get("extra", [])) + [sp["weights"]]:
        dst = os.path.join(sp["local"], f)
        if os.path.isfile(dst) and os.path.getsize(dst) > 32 and f != sp["weights"]:
            continue
        rc = subprocess.call(["curl", "-sL", "-C", "-", "--retry", "3", "--retry-delay", "5", "-o", dst, base + f])
        size = os.path.getsize(dst) if os.path.isfile(dst) else -1
        if 0 <= size < 32 and b"not found" in open(dst, "rb").read():
            os.remove(dst)
            size = -1
        if size < 0:
            if f == sp["weights"]:
                print("  ✗ %s 拉不到(rc=%d),权重缺失,模型不可用" % (f, rc))
                return False
            print("  - %s 仓里没有,跳过" % f)
        else:
            print("  ✓ %-32s %8.1f MB" % (f, size / 1048576))
    ok = mr_index.has_weights(sp)
    print("权重 %s;首次 search/eval 会建 embedding 缓存(.cache/emb-%s.npy)" % ("就绪" if ok else "缺失", sp["slug"]))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--mode", default="hybrid", choices=["bm25", "vec", "hybrid"])
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--source", default=None, choices=["case", "book", "mem", "pdf"])
    p.add_argument("--include-pdf", action="store_true")
    p.add_argument("--model", default=None, help="embedding 模型预设名(bge-small-zh-v1.5 / bge-m3 / Qwen3-Embedding-0.6B 或短名 small/m3/qwen3)")
    p = sub.add_parser("eval")
    p.add_argument("--mode", default="all", choices=["all", "bm25", "vec", "hybrid"])
    p.add_argument("-v", action="store_true")
    p.add_argument("--golden", default=None, help="金标文件,默认 golden/golden_queries.json;PDF 源用 golden/golden_pdf.json")
    p.add_argument("--model", default=None, help="同 search --model;只影响 vec/hybrid 两行")
    p = sub.add_parser("pull-model")
    p.add_argument("name", help="预设名或短名:bge-small-zh-v1.5|bge-m3|Qwen3-Embedding-0.6B|small|m3|qwen3")
    sub.add_parser("pdf-extract")   # 把教材 PDF 按页抽文本进 .cache/pdf-text(一次 10 分钟),之后 MR_PDF=1 秒开
    p = sub.add_parser("ranks")
    p.add_argument("--model", default=None)
    p = sub.add_parser("get")           # 二段式第二步:search 定位→get 全文(别只吃 snippet,截断缝合教训)
    p.add_argument("file")
    sub.add_parser("setup-vec")
    a = ap.parse_args()
    if a.cmd == "stats":
        cmd_stats()
    elif a.cmd == "search":
        cmd_search(a.query, a.mode, a.topk, a.source, a.model)
    elif a.cmd == "eval":
        cmd_eval(a.mode, a.v, a.golden, a.model)
    elif a.cmd == "pull-model":
        sys.exit(0 if cmd_pull_model(a.name) else 1)
    elif a.cmd == "pdf-extract":
        for path in mr_corpus.pdf_files():
            t0 = time.time()
            pages = mr_corpus.extract_pdf(path, progress=lambda i, n, _p=path: print("[pdf] %s %d/%d" % (os.path.basename(_p), i, n), file=sys.stderr))
            print("%s: %d 页, 非空 %d 页, %.0fs" % (os.path.basename(path), len(pages), sum(1 for p in pages if len(p) >= mr_corpus.MIN_PAGE_CHARS), time.time() - t0))
    elif a.cmd == "ranks":
        cmd_ranks(a.model)
    elif a.cmd == "get":
        r = mr_tools.memory_get(a.file)
        print(r["text"] if r.get("ok") else r.get("error"))
    elif a.cmd == "setup-vec":
        cmd_setup_vec()
