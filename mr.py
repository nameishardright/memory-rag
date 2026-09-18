# -*- coding: utf-8 -*-
"""CLI 入口。
  py -3.12 mr.py stats                          # 语料/索引状态
  py -3.12 mr.py search "查询" [--mode hybrid|bm25|vec] [--topk 5] [--source case|book|mem] [--model bge-m3]
  py -3.12 mr.py eval [--mode all|bm25|vec|hybrid] [-v] [--model bge-m3]   # 金标评测:recall@5 / MRR@10 / 延迟
  py -3.12 mr.py eval --golden golden/golden_negative.json -v   # 阴性对照:无答案问题的拦截率(拒答门 MR_GATE=note|drop|off)
  py -3.12 mr.py setup-vec                      # 装向量依赖(清华镜像)+预拉默认模型
  py -3.12 mr.py pull-model bge-m3              # 从 hf-mirror 拉某个预设模型的权重到 .cache/<名>/(可断点续传);重排模型同理 pull-model bge-reranker-base
  py -3.12 mr.py eval --rerank bge-reranker-base   # 第三级重排(cross-encoder)对照;进程默认用 MR_RERANK=<预设>
  py -3.12 mr.py log [--days 7] [--src mcp]     # 查询日志汇总:0 命中的问题 + search→get 隐式正标签挖成金标候选
  py -3.12 mr.py mine-transcripts [--days 0] [--multi] [--show-easy]   # 从 CC 会话 transcript 挖「问了 X→真读了文件 Y」,按现在 RAG 的名次排难度,只给答不好的
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
import mr_eval  # noqa: E402


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
    cur_rr = os.environ.get("MR_RERANK")
    print("重排模型(MR_RERANK=%s,MR_RERANK_MODE=%s):" % (cur_rr or "关", mr_tools.RERANK_MODE))
    for name in mr_index.RERANKERS:
        sp = mr_index.reranker_spec(name)
        print("  %s%-22s 权重:%s  %s" % ("*" if cur_rr and mr_index.reranker_spec(cur_rr)["key"] == name else " ", name,
              "有" if mr_index.has_weights(sp) else "无(pull-model %s)" % name, sp["note"]))
    for name in mr_index.MODELS:
        sp = mr_index.model_spec(name)
        emb = os.path.join(mr_index.CACHE, "emb-%s.npy" % sp["slug"])
        print("  %s%-22s 权重:%s  缓存:%s  %s" % ("*" if name == cur["key"] else " ", name,
              "有" if mr_index.has_weights(sp) else "无(pull-model %s)" % name,
              "有" if os.path.exists(emb) else "无", sp["note"]))


def cmd_search(q, mode, topk, source, model=None, rerank=None, include_pdf=False, include_evaluation=False):
    t0 = time.time()
    r = mr_tools.memory_search(q, mode=mode, topk=topk, source=source, model=model, rerank=rerank,
                               include_pdf=include_pdf, include_evaluation=include_evaluation)
    print("[%s] tokenizer=%s model=%s rerank=%s conf=%s %.0fms  %s" % (r["mode"], r["tokenizer"], r.get("model") or "-",
          (r.get("rerank") or {}).get("model", "-"), r.get("confidence"), (time.time() - t0) * 1000, r["note"]))
    for h in r["hits"]:
        tag = ("  [%s]" % h["legs"]) if h.get("legs") else ""
        print("%2d. %.4f [%s] %s :: %s%s" % (h["rank"], h["score"], h["source"], h["file"], h["heading"], tag))
        print("      %s" % h["snippet"][:120].replace("\n", " "))


def cmd_eval(mode, verbose, golden_path=None, model=None, rerank=None):
    mr_tools.LOG_ON = False   # 金标几百条别淹掉真实查询日志
    rr_name = (mr_index.reranker_spec(rerank if isinstance(rerank, str) else None) or {}).get("key") if (rerank or os.environ.get("MR_RERANK")) else None
    with open(golden_path or os.path.join(BASE, "golden", "golden_queries.json"), encoding="utf-8") as f:
        golden = json.load(f)
    modes = ["bm25", "vec", "hybrid"] if mode == "all" else [mode]
    mr_eval.validate_labels(golden)
    # Audit every active source and every local golden before warming models or
    # emitting retrieval scores, even when this run queries only one source.
    chunks = mr_tools._ensure("bm25")["chunks"]
    audit = mr_eval.require_clean_corpus(chunks, mr_eval.load_goldens(os.path.join(BASE, "golden"), golden_path))
    print("CORPUS_AUDIT_OK: %d 块 / %d 个原问句" % (audit["scanned_chunks"], audit["question_count"]))
    print("金标 %d 条(draft=%s) | 模型 %s | 重排 %s | 检索评测,不是最终答案正确率" % (
        len(golden["queries"]), golden.get("draft"), mr_index.model_spec(model or (os.environ.get("MR_EMBED_MODEL_PDF") if golden.get("source") == "pdf" else None))["key"],
        ("%s(%s)" % (rr_name, mr_tools.RERANK_MODE)) if rr_name else "-"))
    print("returned@5/MRR:实际返回项;document@5/MRR:去重文件(历史口径);evidence:标题和预览内必需事实均匹配")
    reports = {}
    for m in modes:
        probe = mr_tools.memory_search("探针", mode=m, topk=1, source=golden.get("source"), model=model, rerank=False)
        if probe["mode"].endswith("(fallback)") and m != "bm25":
            print("%-8s 跳过 — %s" % (m, probe["note"]))
            continue
        report = mr_eval.evaluate(golden, mr_tools.memory_search, m, model, rerank, chunks=chunks)
        reports[m] = report
        s = report["summary"]
        if s["positives"]:
            print("%-8s returned@5 %d/%d MRR %.3f | document@5 %d/%d MRR %.3f | %.0fms" % (
                m, s["returned_hit_at5"], s["positives"], s["returned_mrr_at10"],
                s["document_hit_at5"], s["positives"], s["document_mrr_at10"], s["mean_ms"]))
        if s["evidence_labeled"]:
            print("         evidence@1 %d/%d | evidence@5 %d/%d" % (
                s["evidence_hit_at1"], s["evidence_labeled"], s["evidence_hit_at5"], s["evidence_labeled"]))
        if s["negatives"]:
            dist = s["negative_confidence"]
            print("%-8s negative low %d / medium %d / high %d (共 %d)" % (
                m, dist.get("low", 0), dist.get("medium", 0), dist.get("high", 0), s["negatives"]))
        if s["fallbacks"]:
            print("    ⚠️ %d 次回退,不能当作 %s 模式验收" % (s["fallbacks"], m))
        if verbose:
            for row in report["rows"]:
                if ((row["positive"] and (not row["returned_rank"] or row["returned_rank"] > 5 or row["confidence"] != "high")) or
                        (row["passage_labeled"] and (not row["evidence_rank"] or row["evidence_rank"] > 5)) or
                        (not row["positive"] and row["confidence"] == "high")):
                    print("    %s: returned=%s document=%s evidence=%s confidence=%s" % (
                        row["q"], row["returned_rank"], row["document_rank"], row["evidence_rank"], row["confidence"]))
    return reports


def cmd_audit_corpus(golden_path=None):
    audit = mr_eval.require_clean_corpus(mr_corpus.load_chunks(), mr_eval.load_goldens(os.path.join(BASE, "golden"), golden_path))
    print("CORPUS_AUDIT_OK: %d 块 / %d 个原问句 / %d 个已标评测块" % (
        audit["scanned_chunks"], audit["question_count"], audit["evaluation_chunks"]))
    return audit


def cmd_ranks(model=None):
    """逐题×逐模式:期望文件的名次矩阵(归因用——miss 只是名次>10 的特例)。"""
    mr_tools.LOG_ON = False
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


def cmd_log(days, src):
    """查询日志汇总 + 金标候选:search 后 120s 内 get 了命中文件 → {"q","expect"};0 命中的问题 → 语料缺口 / 阴性题候选。"""
    import datetime
    p = mr_tools.LOG_PATH
    if not os.path.exists(p):
        print("还没有日志:%s(MCP/CLI search 会自动写;MR_LOG=0 关)" % p)
        return
    since = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    recs = []
    with open(p, encoding="utf-8") as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("ts", "") >= since and (src == "all" or r.get("src") == src):
                recs.append(r)
    searches = [r for r in recs if r.get("tool") == "search"]
    gets = [r for r in recs if r.get("tool") == "get"]
    print("近 %d 天(src=%s):search %d 次,get %d 次" % (days, src, len(searches), len(gets)))
    zero = [r for r in searches if r.get("n") == 0]
    if zero:
        print("\n0 命中(%d)——语料缺口或阴性题候选:" % len(zero))
        for r in zero[-20:]:
            print("  %s  %s  [%s%s]" % (r["ts"][5:16], r["q"][:50], r.get("mode"), (" " + r["source"]) if r.get("source") else ""))
    cands = []
    for sr in searches:
        if not sr.get("top"):
            continue
        t0 = sr["ts"]
        hit_files = [t.split("#", 1)[0] for t in sr["top"]]
        for g in gets:
            if t0 <= g["ts"] <= _plus(t0, 120) and g.get("file") in hit_files:
                cands.append({"q": sr["q"], "expect": [g["file"]], **({"source": sr["source"]} if sr.get("source") else {})})
                break
    seen, uniq = set(), []
    for c in cands:
        k = (c["q"], c["expect"][0])
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    if uniq:
        print("\n金标候选(%d,search→get 隐式正标签;核对后挑进 golden/*.json):" % len(uniq))
        for c in uniq[-30:]:
            print("  " + json.dumps(c, ensure_ascii=False))
    top = {}
    for sr in searches:
        for t in sr.get("top", [])[:1]:
            top[t.split("#", 1)[0]] = top.get(t.split("#", 1)[0], 0) + 1
    if top:
        print("\n最常排第一的文件:")
        for f_, n in sorted(top.items(), key=lambda kv: -kv[1])[:10]:
            print("  %3d  %s" % (n, f_))


def _plus(ts, sec):
    import datetime
    return (datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S") + datetime.timedelta(seconds=sec)).strftime("%Y-%m-%dT%H:%M:%S")


def cmd_pull_model(name):
    """从 HF 镜像把预设模型的权重和配置拉到 .cache/<名>/。断点续传(curl -C -),已完整的文件跳过。
    🔴 镜像对不存在的文件回 15 字节的 "Entry not found" 而不是 404 报错——不查的话会留下一个假权重文件,
       之后 sentence-transformers 报的错跟"没权重"完全不像(2026-09-08 bge-m3 踩过:它仓里只有 pytorch_model.bin)。"""
    rr = mr_index.reranker_spec(name) if (name.lower() in {k.lower() for k in mr_index.RERANKERS} or name.lower() in mr_index._RR_ALIASES) else None
    sp = rr or mr_index.model_spec(name)
    base = "%s/%s/resolve/main/" % (os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/"), sp["id"])
    os.makedirs(os.path.join(sp["local"], "1_Pooling"), exist_ok=True)
    print("拉 %s → %s" % (sp["id"], sp["local"]))
    files = sp["files"] if rr else (mr_index.COMMON_FILES + list(sp.get("extra", [])) + [sp["weights"]])
    for f in files:
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
    if rr:
        print("重排权重 %s;用法:MR_RERANK=%s 或 --rerank %s" % ("就绪" if ok else "缺失", sp["key"], sp["key"]))
    else:
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
    p.add_argument("--source", default=None, choices=["case", "book", "mem", "pdf", "stock"])
    p.add_argument("--include-pdf", action="store_true")
    p.add_argument("--include-evaluation", action="store_true", help="包含标为 evaluation 的评测复盘,用于检索评测方法和运行记录")
    p.add_argument("--model", default=None, help="embedding 模型预设名(bge-small-zh-v1.5 / bge-m3 / Qwen3-Embedding-0.6B 或短名 small/m3/qwen3)")
    p.add_argument("--rerank", default=None, help="重排预设名(bge-reranker-base / bge-reranker-v2-m3 或短名 base/v2-m3);不传跟 MR_RERANK")
    p = sub.add_parser("eval")
    p.add_argument("--mode", default="all", choices=["all", "bm25", "vec", "hybrid"])
    p.add_argument("-v", action="store_true")
    p.add_argument("--golden", default=None, help="金标文件,默认 golden/golden_queries.json;PDF 源用 golden/golden_pdf.json")
    p.add_argument("--model", default=None, help="同 search --model;只影响 vec/hybrid 两行")
    p.add_argument("--rerank", default=None, help="同 search --rerank")
    p = sub.add_parser("audit-corpus", help="扫描全部已装载语料中的未标记金标问句,发现即失败")
    p.add_argument("--golden", default=None, help="额外题集;始终同时检查本仓 golden/*.json")
    p = sub.add_parser("log")            # 查询日志汇总 + 金标候选(search→get 隐式正标签)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--src", default="mcp", choices=["mcp", "cli", "all"])
    p = sub.add_parser("mine-transcripts")   # query-log 只记经过 RAG 的流量(11 次);transcript 才是真实日志(直接读语料 477 次)
    p.add_argument("--days", type=int, default=0, help="只挖近 N 天的用户轮;0=全部")
    p.add_argument("--multi", action="store_true", help="也收一轮读了多个语料文件的轮(默认只收单文件轮,多文件轮里大半是干活的工具性阅读)")
    p.add_argument("--show-easy", action="store_true", help="连现在已经排 ≤3 名的也打印")
    p.add_argument("--unranked", action="store_true", help="只挖候选，不调用检索或按成绩筛题；封存留出集时使用")
    p.add_argument("--out", default=None, help="全部候选(含名次)落盘路径,默认 .cache/mined-transcripts.json")
    p = sub.add_parser("pull-model")
    p.add_argument("name", help="预设名或短名:bge-small-zh-v1.5|bge-m3|Qwen3-Embedding-0.6B|small|m3|qwen3")
    sub.add_parser("pdf-extract")   # 把教材 PDF 按页抽文本进 .cache/pdf-text(一次 10 分钟),之后 MR_PDF=1 秒开
    p = sub.add_parser("ranks")
    p.add_argument("--model", default=None)
    p = sub.add_parser("get")           # 二段式第二步:search 定位→get 全文(别只吃 snippet,截断缝合教训)
    p.add_argument("file")
    p.add_argument("--source", choices=["case", "book", "mem", "pdf", "stock"])
    p.add_argument("--heading", help="精确取 search 命中的完整 Markdown 段落")
    sub.add_parser("setup-vec")
    a = ap.parse_args()
    if a.cmd == "stats":
        cmd_stats()
    elif a.cmd == "search":
        cmd_search(a.query, a.mode, a.topk, a.source, a.model, a.rerank, a.include_pdf, a.include_evaluation)
    elif a.cmd == "eval":
        try:
            cmd_eval(a.mode, a.v, a.golden, a.model, a.rerank)
        except mr_eval.CorpusContaminationError as error:
            print(error, file=sys.stderr)
            sys.exit(2)
    elif a.cmd == "audit-corpus":
        try:
            cmd_audit_corpus(a.golden)
        except mr_eval.CorpusContaminationError as error:
            print(error, file=sys.stderr)
            sys.exit(2)
    elif a.cmd == "log":
        cmd_log(a.days, a.src)
    elif a.cmd == "mine-transcripts":
        import mr_mine
        mr_mine.main(a.days, a.multi, a.show_easy, a.out, a.unranked)
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
        r = mr_tools.memory_get(a.file, a.source, a.heading)
        print(r["text"] if r.get("ok") else r.get("error"))
    elif a.cmd == "setup-vec":
        cmd_setup_vec()
