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


def memory_search(query, mode="hybrid", topk=5, source=None):
    st = _ensure(mode)
    chunks, note = st["chunks"], ""
    bm = st["bm25"].search(query, topk=50)
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
    hits = []
    for i, score in ranked:
        c = chunks[i]
        if source and c["source"] != source:
            continue
        hits.append({"rank": len(hits) + 1, "score": round(float(score), 4),
                     "source": c["source"], "file": c["file"], "heading": c["heading"],
                     "snippet": c["text"][:300]})
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
                "mem(个人长期记忆)。查不到返回空 hits,也是正常结果。",
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
