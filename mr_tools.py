# -*- coding: utf-8 -*-
"""工具真源:REGISTRY 一处注册,CLI(mr.py)/MCP(mr_mcp_server.py) 两处消费——照 taskid ta_tools 模式。
工具永不抛异常:查不到/降级都是正常返回值,让调用方(人或模型)自己决策。
"""
import os
import threading

import mr_corpus
import mr_index

_STATE = {}
_LOCK = threading.Lock()   # MCP server 启动预热线程和首个查询会并发进来,不锁会撞半成品状态


def _ensure(mode):
    with _LOCK:
        if "chunks" not in _STATE:
            _STATE["chunks"] = mr_corpus.load_chunks()
            _STATE["bm25"] = mr_index.BM25(_STATE["chunks"])
        if mode in ("vec", "hybrid") and "vec" not in _STATE:
            _STATE["vec"] = mr_index.VecIndex(_STATE["chunks"])
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


def memory_search(query, mode="hybrid", topk=5, source=None):
    st = _ensure(mode)
    chunks, note = st["chunks"], ""
    bm = st["bm25"].search(query, topk=50)
    leg_names = None
    if mode == "bm25":
        ranked = bm
    else:
        v = st.get("vec")
        if v is None or not v.available:
            ranked, note = bm, "向量层不可用(%s),已回退 bm25" % getattr(v, "err", "")
            mode = "bm25(fallback)"
        elif mode == "vec":
            ranked = v.search(query, topk=50)
        else:
            ranked = mr_index.rrf_files(chunks, [bm, v.search(query, topk=50)])
            leg_names = ("bm25", "vec")
    hits = []
    for row in ranked:
        i, score, legs = (row if len(row) == 3 else (row[0], row[1], None))
        c = chunks[i]
        if source and c["source"] != source:
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
    return {"ok": True, "mode": mode, "tokenizer": mr_index.TOKENIZER, "note": note, "hits": hits}


def memory_get(file, source=None):
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
                "mem(个人长期记忆)。查不到返回空 hits,也是正常结果。hybrid 模式每条带 legs(词法/语义两腿"
                "各排第几,双腿都靠前=共识强);snippet 已对准命中词。引用前用 memory_get 看全文。"
                "查询技巧:①语料以中文为主,查询用中文(英文短查询两条腿都弱——向量模型是中文特化的);"
                "②用意图里的罕见词问(如 直合/cherry-pick),别堆高频实体词(gerrit/合入 满语料都是,零区分度);"
                "③已知目标文件名就直接 memory_get,别绕检索。",
        "schema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "自然语言问题或关键词,中文为主"},
            "mode": {"type": "string", "enum": ["bm25", "vec", "hybrid"],
                     "description": "默认 hybrid;向量层没装会自动回退 bm25 并在 note 里说明"},
            "topk": {"type": "integer", "description": "返回条数,默认 5,上限 10"},
            "source": {"type": "string", "enum": ["case", "book", "mem"], "description": "可选,只查某一源"},
        }, "required": ["query"]},
    },
    "memory_get": {
        "fn": memory_get,
        "desc": "取某个笔记/记忆文件全文。memory_search 命中的片段不够看时,把 hits 里的 file 字段原样传入。",
        "schema": {"type": "object", "properties": {
            "file": {"type": "string", "description": "文件名,来自 memory_search 结果的 file 字段"},
            "source": {"type": "string", "enum": ["case", "book", "mem"]},
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
