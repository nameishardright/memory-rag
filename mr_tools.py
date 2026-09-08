# -*- coding: utf-8 -*-
"""工具真源:REGISTRY 一处注册,CLI(mr.py)/MCP(mr_mcp_server.py) 两处消费——照 taskid ta_tools 模式。
工具永不抛异常:查不到/降级都是正常返回值,让调用方(人或模型)自己决策。
"""
import os
import re
import sys
import threading
import time

import mr_corpus
import mr_index

_STATE = {}
_LOCK = threading.Lock()   # MCP server 启动预热线程和首个查询会并发进来,不锁会撞半成品状态
RELOAD_SEC = float(os.environ.get("MR_RELOAD_SEC", "5"))   # 最多每 5s 看一次语料指纹(stat ~150 个文件,毫秒级)


class _SubBM25:
    """在一个子集上建 BM25,对外返回全局下标。为什么要分开建:BM25 的 IDF / 平均长度是语料级统计,
    1296 页 PDF 混进来会把笔记的分数拉偏(2026-09-08 实测 MR_PDF=1 即便检索时剔掉 PDF,主金标 MRR 仍 .835→.806)。"""

    def __init__(self, chunks, idx):
        self.idx = idx
        self.bm = mr_index.BM25([chunks[i] for i in idx])

    def search(self, query, topk=50, mask=None):
        local_mask = None if mask is None else {l for l, g in enumerate(self.idx) if g in mask}
        return [(self.idx[l], s) for l, s in self.bm.search(query, topk=topk, mask=local_mask)]


def _rebuild(reason):
    old_n = len(_STATE.get("chunks", []))
    chunks = mr_corpus.load_chunks()
    _STATE["chunks"] = chunks
    notes_idx = [i for i, c in enumerate(chunks) if c["source"] != "pdf"]
    pdf_idx = [i for i, c in enumerate(chunks) if c["source"] == "pdf"]
    _STATE["idx"] = {"notes": notes_idx, "pdf": pdf_idx}
    _STATE["bm25"] = _SubBM25(chunks, notes_idx)                          # 默认三源
    _STATE["bm25_pdf"] = _SubBM25(chunks, pdf_idx) if pdf_idx else None   # source=pdf
    _STATE.pop("bm25_all", None)                                          # include_pdf=True 时懒建
    for slug, v in list(_STATE.get("vecs", {}).items()):   # 每个已装的模型都增量重建:md5 缓存只算新块,模型实例复用
        _STATE["vecs"][slug] = mr_index.VecIndex(chunks, model=v._model, spec=v.spec)
    _STATE["sig"] = mr_corpus.signature()
    print("[reload] %s:语料 %d→%d 块" % (reason, old_n, len(chunks)), file=sys.stderr)


def _ensure(mode, model=None):
    """🔴 热更(2026-09-08):原来进程启动装一次索引就不看文件了,会话里新写的笔记查不到,要重启会话——
    现在每次查询前(最多每 RELOAD_SEC 秒一次)比对语料指纹,变了就重建;向量层按 md5 增量,只多算新块。"""
    with _LOCK:
        now = time.time()
        if "chunks" not in _STATE:
            _rebuild("首次加载")
            _STATE["sig_at"] = now
        elif now - _STATE.get("sig_at", 0) >= RELOAD_SEC:
            _STATE["sig_at"] = now
            if mr_corpus.signature() != _STATE.get("sig"):
                _rebuild("检测到文件变化")
        if mode in ("vec", "hybrid"):
            # 一键切换:每个模型各建各的向量层,按 slug 存在 _STATE["vecs"],切回来不重算;没权重的模型 available=False,查询时回退 bm25
            spec = mr_index.model_spec(model)
            vecs = _STATE.setdefault("vecs", {})
            if spec["slug"] not in vecs:
                vecs[spec["slug"]] = mr_index.VecIndex(_STATE["chunks"], spec=spec)
            _STATE["vec"] = vecs[spec["slug"]]   # 「最近一次用的」;stats / 老脚本还读这个名字
        return _STATE


def _snippet(text, query, width=300):
    """片段以首个命中词为中心截窗,不再固定取块头——命中词在窗外的预览等于没预览(2026-08-21 用户实踩)。
    纯语义命中(块里没有查询词面)退回块头。"""
    toks = sorted({t for t in mr_index.tokenize(query) if len(t) >= 2}, key=len, reverse=True)
    low = text.lower()
    pos = min((p for p in (low.find(t) for t in toks) if p >= 0), default=-1)
    if pos <= 40:
        return text[:width]
    start = pos - 40
    return "…" + text[start:start + width]


def memory_search(query, mode="hybrid", topk=5, source=None, include_pdf=False, model=None):
    """model=None 时向量腿用进程默认模型(MR_EMBED_MODEL,默认 bge-small-zh-v1.5);传预设名可单次切换,见 mr_index.MODELS。
    🔴 按源选模型(2026-09-08 用户 GPU 对照,同一 3116 块):笔记金标 small 18/18 .835 vs bge-m3 17/18 .822(一题之差,噪声级),
       PDF 金标 small 11/12 .694 vs bge-m3 12/12 .917——截尾的代价取决于块长和答案位置:笔记按标题分段主题在段首,PDF 一页一块答案在页中任何位置。
       所以 source="pdf" 且没显式传 model 时走 MR_EMBED_MODEL_PDF(没设=默认模型);笔记留 small(快 4 倍),原文页走 bge-m3。
    source=None 时默认只查 case/book/mem 三源;PDF 原文(要 MR_PDF=1 装载)只有 source="pdf" 或 include_pdf=True 才参与。
    🔴 为什么按查询开关而不是全混:2026-09-08 实测把 1296 页原文混进默认检索,主金标 hybrid 18/18→17/18、MRR .835→.750——
    原文页把带 ⭐ 页码的笔记挤下去;而 source=pdf 单独查,原文页 9/12 能找到。两层开关:环境变量管装不装,参数管查不查。"""
    if model is None and source == "pdf":
        model = os.environ.get("MR_EMBED_MODEL_PDF") or None
    st = _ensure(mode, model)
    spec = mr_index.model_spec(model)
    chunks, note, used_model = st["chunks"], "", None
    if source:
        allowed = {source}
    elif include_pdf:
        allowed = None
    else:
        allowed = {"case", "book", "mem"}
    keep = lambda i: allowed is None or chunks[i]["source"] in allowed   # noqa: E731
    # 掩码 = 允许的全局下标;BM25 用对应子集索引(统计不串),向量层用掩码(先掩后切)
    if allowed is None:
        if "bm25_all" not in st:
            st["bm25_all"] = _SubBM25(chunks, list(range(len(chunks))))
        bm_index, mask = st["bm25_all"], None
    elif allowed == {"pdf"}:
        bm_index, mask = st.get("bm25_pdf"), set(st["idx"]["pdf"])
    else:
        mask = {i for i in st["idx"]["notes"] if chunks[i]["source"] in allowed}
        bm_index = st["bm25"]
    K = 50
    bm = bm_index.search(query, topk=K, mask=mask) if bm_index else []
    leg_names = None
    if mode == "bm25":
        ranked = bm
    else:
        v = st.get("vecs", {}).get(spec["slug"])
        if v is None or not v.available:
            ranked, note = bm, "向量层不可用(%s),已回退 bm25" % getattr(v, "err", "")
            mode = "bm25(fallback)"
        elif mode == "vec":
            ranked, used_model = v.search(query, topk=K, mask=mask), spec["key"]
        else:
            used_model = spec["key"]
            ranked = mr_index.rrf_files(chunks, [bm, v.search(query, topk=K, mask=mask)],
                                        unit=("page" if allowed == {"pdf"} else "file"))
            leg_names = ("bm25", "vec")
    if source == "pdf" and not any(c["source"] == "pdf" for c in chunks):
        note = "PDF 源没装载:服务端要 MR_PDF=1 起(先 py -3.12 mr.py pdf-extract 抽一次文本)"
    hits = []
    for row in ranked:
        i, score, legs = (row if len(row) == 3 else (row[0], row[1], None))
        c = chunks[i]
        if not keep(i):
            continue
        h = {"rank": len(hits) + 1, "score": round(float(score), 4),
             "source": c["source"], "file": c["file"], "heading": c["heading"],
             "snippet": _snippet(c["text"], query)}
        if legs is not None and leg_names:
            # 只披露机制(每条腿排第几),不下"低置信"判断——weak 旗试过一版当场撤:
            # "无腿前3=凑数"把双腿共识型正确答案(bm25#6+vec#7 靠合力赢)标成了低置信,
            # RRF 的价值恰恰是两票中庸胜过一票高分。置信判断要有自己的金标才配自动化,先留给读者。
            h["legs"] = "+".join(f"{n}#{r + 1}" for n, r in zip(leg_names, legs) if r is not None)
        hits.append(h)
        if len(hits) >= topk:
            break
    return {"ok": True, "mode": mode, "tokenizer": mr_index.TOKENIZER, "model": used_model, "note": note, "hits": hits}


def memory_get(file, source=None):
    # PDF 原文:file 形如「Agentic-AI指南.pdf#p81」→ 回那一页 ±1 页(页码 Acrobat 1-indexed);缓存没建就现抽(首次几分钟)
    m = re.match(r"^(.+?\.pdf)(?:#p(\d+))?$", os.path.basename(file), re.I)
    if m:
        path = os.path.join(mr_corpus.PDF_DIR, m.group(1))
        if not os.path.isfile(path):
            return {"ok": False, "error": "没有这个 PDF: %s" % m.group(1)}
        if not m.group(2):
            return {"ok": False, "error": "PDF 要指定页:写成 %s#p<页码>(页码来自 memory_search 的 heading,如 p81)" % m.group(1)}
        pages = mr_corpus.extract_pdf(path)
        p = int(m.group(2))
        if p < 1 or p > len(pages):
            return {"ok": False, "error": "页码越界:%s 共 %d 页" % (m.group(1), len(pages))}
        lo, hi = max(1, p - 1), min(len(pages), p + 1)
        text = "\n\n".join("── p%d ──\n%s" % (i, pages[i - 1]) for i in range(lo, hi + 1))
        return {"ok": True, "source": "pdf", "file": m.group(1), "page": p, "pages": [lo, hi], "text": text}
    for src in mr_corpus.SOURCES:
        p = os.path.join(src["dir"], os.path.basename(file))   # basename 防目录穿越
        if os.path.isfile(p) and (not source or src["key"] == source):
            with open(p, encoding="utf-8", errors="replace") as f:
                text = f.read()
            if src["redact"]:
                text = mr_corpus.redact(text)
            return {"ok": True, "source": src["key"], "file": os.path.basename(file), "text": text}
    return {"ok": False, "error": "三个语料目录都没有 %s(把 memory_search 结果里的 file 字段原样传)" % file}


REGISTRY = {
    "memory_search": {
        "fn": memory_search,
        "desc": "查 zihao 以前踩过的坑/定过的原则/记过的事实。排查问题、写方案、做设计决策前先调它:"
                "输入自然语言问题,返回三层语料里最相关的片段——case(实战学习笔记)/book(两本 agent 教材章节笔记)/"
                "mem(个人长期记忆);要教材原文页时 source=pdf(按页返回,笔记答「哪一章」更准,原文答「书上原话怎么说」)。查不到返回空 hits,也是正常结果。hybrid 模式每条带 legs(词法/语义两腿"
                "各排第几,双腿都靠前=共识强);snippet 已对准命中词。引用前用 memory_get 看全文。"
                "查询技巧:①语料以中文为主,查询用中文(英文短查询两条腿都弱——向量模型是中文特化的);"
                "②用意图里的罕见词问(如 直合/cherry-pick),别堆高频实体词(gerrit/合入 满语料都是,零区分度);"
                "③已知目标文件名就直接 memory_get,别绕检索。",
        "schema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "自然语言问题或关键词,中文为主"},
            "mode": {"type": "string", "enum": ["bm25", "vec", "hybrid"],
                     "description": "默认 hybrid;向量层没装会自动回退 bm25 并在 note 里说明"},
            "topk": {"type": "integer", "description": "返回条数,默认 5,上限 10"},
            "source": {"type": "string", "enum": ["case", "book", "mem", "pdf"], "description": "可选,只查某一源。默认只查 case/book/mem;要教材**原文那一页**用 source=pdf(命中的 heading 是页码,再 memory_get 「xxx.pdf#p81」看整页)"},
            "include_pdf": {"type": "boolean", "description": "默认 false;true 时默认检索也混入 PDF 原文页(会挤掉笔记,一般别开)"},
            "model": {"type": "string", "enum": list(mr_index.MODELS),
                      "description": "可选,向量腿用哪个 embedding 模型,单次切换;不传=服务端默认(MR_EMBED_MODEL,默认 bge-small-zh-v1.5;"
                                     "source=pdf 时默认走 MR_EMBED_MODEL_PDF,原文页上 bge-m3 明显更准)。"
                                     "本地没权重会回退 bm25 并在 note 里说明(先 py -3.12 mr.py pull-model <名>)"},
        }, "required": ["query"]},
    },
    "memory_get": {
        "fn": memory_get,
        "desc": "取某个笔记/记忆文件全文。memory_search 命中的片段不够看时,把 hits 里的 file 字段原样传入。",
        "schema": {"type": "object", "properties": {
            "file": {"type": "string", "description": "文件名,来自 memory_search 结果的 file 字段;教材原文写成「xxx.pdf#p81」取那一页 ±1"},
            "source": {"type": "string", "enum": ["case", "book", "mem", "pdf"]},
        }, "required": ["file"]},
    },
}


def call(name, args):
    if name not in REGISTRY:
        return {"ok": False, "error": "未知工具 %s" % name}
    try:
        kw = dict(args or {})
        if "topk" in kw:
            kw["topk"] = max(1, min(10, int(kw["topk"])))
        return REGISTRY[name]["fn"](**kw)
    except Exception as e:
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}


if __name__ == "__main__":
    for n, t in REGISTRY.items():
        print("%-14s %s" % (n, t["desc"][:60]))
