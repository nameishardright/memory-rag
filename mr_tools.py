# -*- coding: utf-8 -*-
"""工具真源:REGISTRY 一处注册,CLI(mr.py)/MCP(mr_mcp_server.py) 两处消费——照 taskid ta_tools 模式。
工具永不抛异常:查不到/降级都是正常返回值,让调用方(人或模型)自己决策。
"""
import os
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time

import mr_corpus
import mr_index
import mr_stock

_STATE = {}
_LOCK = threading.Lock()   # MCP server 启动预热线程和首个查询会并发进来,不锁会撞半成品状态
RELOAD_SEC = float(os.environ.get("MR_RELOAD_SEC", "5"))   # 最多每 5s 看一次语料指纹(stat ~150 个文件,毫秒级)

# ── 观测留痕(2026-09-08,《指南》p70-71「可溯源 + 观测」;HA ch8 笔记「对照」第 4 条):每次 search / get 追加一行 jsonl ──
# 用途:①挖金标——一次 search 后 2 分钟内 get 了某个命中文件 = 隐式正标签;②0 命中的真实问题 = 语料缺口或阴性题候选。
# 日志在 .cache/(gitignore),MR_LOG=0 关;CLI 的 eval/ranks 不记(几百条金标会淹掉真实查询),search 记为 src=cli,MCP 记为 src=mcp。
# ── 拒答门(2026-09-08,《指南》p70-71「证据不足要有保守策略」;观澜 kb.js 同款思路,但这里是**查询级三级置信**不是逐条删):
# 两条腿各取最高分和各自门槛比:两腿都过 = high,一条过 = medium,都没过 = low(语料里很可能没有)。
# 为什么不能像 kb.js 那样一刀切拒答:个人语料 1800 块题材杂,无答案的问题靠「今天/怎么/为什么」BM25 也能到 19、靠语义近邻余弦也能到 .60,
# 分差断崖(top1/top5、cos 领先幅度)也分不开(2026-09-08 标定 3)。能量出来的只有:正例 30/30 都是 high;阴性 12 条里 low 2–3、medium 6–7、high 3
# (high 的三条「量子计算 / agent 发音 / 英语时态」本来就贴着语料题材)。所以它是**证据强度标签**,拒答判断留给读结果的模型;真要硬拒答得上重排模型。
# ⚠️ 不能逐条过滤:正确答案常常只有一条腿看见(bm25#1 + 向量未进榜),逐条 AND 会把它删掉;「cherry-pick 直合」这类罕见词问法就是 medium。
# 默认只标不删;MR_GATE=drop 时 low 清空 hits;MR_GATE=off 关。阈值:MR_MIN_BM25 / MR_MIN_BM25_PDF,余弦门在 mr_index.MODELS[*]["min_cos"](换模型要重标)。
GATE_MODE = os.environ.get("MR_GATE", "note").lower()
# ── 重排(项 7):MR_RERANK=<预设> 进程默认开;memory_search(rerank=) 单次 True/False/<预设>;只重排融合榜前 RERANK_N 个候选(消费单元级)。
RERANK_MODE = os.environ.get("MR_RERANK_MODE", "gate").lower()   # gate=只定置信不动顺序(默认,标定过) | rerank=按 cross-encoder 分重排(对照用)
RERANK_N = int(os.environ.get("MR_RERANK_N", "15"))               # rerank 模式打分的候选数
RERANK_G = int(os.environ.get("MR_RERANK_G", "3"))                # base 初评前三个;弱证据时最多补到原前五个
RERANK_WIN = int(os.environ.get("MR_RERANK_WIN", "450"))          # 字符窗口;token 截断预算独立,实际值随 rerank 结果返回
RERANK_MAXWIN = int(os.environ.get("MR_RERANK_MAXWIN", "2"))      # base 每块两窗;其他模型保留各自预设
RERANK_FALLBACK_G = int(os.environ.get("MR_RERANK_FALLBACK_G", "5"))
MIN_BM25 = {"notes": float(os.environ.get("MR_MIN_BM25", "8")), "pdf": float(os.environ.get("MR_MIN_BM25_PDF", "12"))}

LOG_PATH = os.path.join(mr_index.CACHE, "query-log.jsonl")
LOG_ON = os.environ.get("MR_LOG", "1").lower() not in ("0", "false", "no")
LOG_SRC = "cli"


def _log(rec):
    if not LOG_ON:
        return
    try:
        import json
        rec = dict(rec, ts=time.strftime("%Y-%m-%dT%H:%M:%S"), src=LOG_SRC)
        os.makedirs(mr_index.CACHE, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass   # 日志永远不能让查询失败


class _SubBM25:
    """在一个子集上建 BM25,对外返回全局下标。为什么要分开建:BM25 的 IDF / 平均长度是语料级统计,
    1296 页 PDF 混进来会把笔记的分数拉偏(2026-09-08 实测 MR_PDF=1 即便检索时剔掉 PDF,主金标 MRR 仍 .835→.806)。"""

    def __init__(self, chunks, idx):
        self.idx = idx
        self.bm = mr_index.BM25([chunks[i] for i in idx])

    def search(self, query, topk=50, mask=None):
        local_mask = None if mask is None else {l for l, g in enumerate(self.idx) if g in mask}
        return [(self.idx[l], s) for l, s in self.bm.search(query, topk=topk, mask=local_mask)]


class _SubVec:
    """Source-only embeddings with global indices at the retrieval boundary.

PDF-only search already masks every non-PDF row. Building only those permitted
rows keeps the same cosine scores without embedding unused stocks and notes.
The actual model/cache remains shared with the ordinary VecIndex implementation.
"""
    scope = 'pdf'

    def __init__(self, chunks, idx, model=None, spec=None):
        self.idx = list(idx)
        self.vector = mr_index.VecIndex([chunks[i] for i in self.idx], model=model, spec=spec)
        self.spec = self.vector.spec
        self.available, self.err = self.vector.available, self.vector.err

    @property
    def _model(self):
        return self.vector._model

    def search(self, query, topk=50, mask=None):
        local_mask = None if mask is None else {local for local, global_index in enumerate(self.idx) if global_index in mask}
        return [(self.idx[local], score) for local, score in self.vector.search(query, topk=topk, mask=local_mask)]


def _rebuild(reason):
    old_n = len(_STATE.get("chunks", []))
    chunks = mr_corpus.load_chunks()
    _STATE["chunks"] = chunks
    # Observability only: never use gold labels to filter, score or classify a hit.
    # Warn again after each changed corpus, including startup; failures do not
    # make everyday retrieval unavailable. Evaluation retains its strict gate.
    import mr_eval
    started = time.perf_counter()
    try:
        audit = mr_eval.require_clean_corpus(chunks, mr_eval.load_goldens(Path(mr_corpus.BASE) / "golden"))
        status, count = "clean", 0
    except mr_eval.CorpusContaminationError as exc:
        status, count = "warning", len(exc.audit["findings"])
        print("[corpus-audit] %d unmarked chunks contain a known evaluation question; run mr.py audit-corpus" % count,
              file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError):
        status, count = "unavailable", None
        print("[corpus-audit] audit unavailable; retrieval continues without a clean-corpus claim", file=sys.stderr)
    _STATE["corpus_audit"] = {"status": status, "findings": count,
                              "ms": round((time.perf_counter() - started) * 1000, 2)}
    _STATE["corpus_sha256"] = hashlib.sha256(json.dumps(
        [(c["source"], c["md5"], c.get("purpose", "knowledge")) for c in chunks],
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    notes_idx = [i for i, c in enumerate(chunks) if c["source"] in mr_corpus.NOTE_SOURCES]
    ordinary_idx = [i for i in notes_idx if chunks[i].get("purpose") != "evaluation"]
    nonpdf_idx = [i for i, c in enumerate(chunks) if c["source"] != "pdf"]
    pdf_idx = [i for i, c in enumerate(chunks) if c["source"] == "pdf"]
    _STATE["idx"] = {"notes": notes_idx, "pdf": pdf_idx, "nonpdf": nonpdf_idx}
    _STATE["bm25"] = _SubBM25(chunks, ordinary_idx)                      # 默认三源,统计不含 stock/评测复盘
    pdf_knowledge = [i for i in pdf_idx if chunks[i].get("purpose") != "evaluation"]
    _STATE["bm25_pdf"] = _SubBM25(chunks, pdf_knowledge) if pdf_idx else None
    for key in ("bm25_all", "bm25_nonpdf", "bm25_evaluation", "bm25_pdf_evaluation"):
        _STATE.pop(key, None)   # 混合/PDF、股票原统计、含评测笔记的统计均按需重建
    for slug, v in list(_STATE.get("vecs", {}).items()):   # 每个已装的模型都增量重建:md5 缓存只算新块,模型实例复用
        _STATE["vecs"][slug] = (_SubVec(chunks, pdf_idx, model=v._model, spec=v.spec)
                                 if getattr(v, 'scope', None) == 'pdf'
                                 else mr_index.VecIndex(chunks, model=v._model, spec=v.spec))
    _STATE["sig"] = mr_corpus.signature()
    print("[reload] %s:语料 %d→%d 块" % (reason, old_n, len(chunks)), file=sys.stderr)


def _ensure(mode, model=None, vec_scope=None):
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
            key = spec['slug'] + ('|pdf' if vec_scope == 'pdf' else '')
            if key not in vecs:
                shared = next((v._model for v in vecs.values()
                               if getattr(v,'spec',{}).get('slug') == spec['slug'] and getattr(v,'_model',None) is not None), None)
                vecs[key] = (_SubVec(_STATE['chunks'], _STATE['idx']['pdf'], model=shared, spec=spec)
                             if vec_scope == 'pdf' else mr_index.VecIndex(_STATE['chunks'], model=shared, spec=spec))
            _STATE["vec"] = vecs[key]   # 「最近一次用的」;stats / 老脚本还读这个名字
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


def _get_reranker(key, legacy=False):
    """按模型与预算缓存实例;同一模型的预算变体共享权重,首次评分时才加载。"""
    spec = mr_index.reranker_spec(key if isinstance(key, str) else None,legacy=legacy)
    if spec is None:
        return None
    with _LOCK:
        rs = _STATE.setdefault("rerankers", {})
        model_path = os.path.normcase(os.path.realpath(spec["local"]))
        cache_key = (model_path,spec["key"],spec.get("max_length",512),spec.get("gate_candidates",5),
                     spec.get("max_windows",3),spec.get("fallback_candidates",5))
        if cache_key not in rs:
            instance = mr_index.Reranker(spec)
            instance._model = next((other._model for other in rs.values()
                                    if os.path.normcase(os.path.realpath(other.spec["local"]))==model_path
                                    and other._model is not None),None)
            rs[cache_key] = instance
        return rs[cache_key]


def _windows(c, max_windows=None):
    """一块切成最多 RERANK_MAXWIN 个 450 字窗口,每窗都带文件名+标题;分数取窗口最大值——答案在块中后段时不至于被截掉。"""
    head = "%s\n%s\n" % (c["file"], c["heading"])
    body = c["text"] or ""
    out = [head + body[s0:s0 + RERANK_WIN] for s0 in range(0, max(1, len(body)), RERANK_WIN)][:max_windows or RERANK_MAXWIN]
    return out or [head]


def memory_search(query, mode="hybrid", topk=5, source=None, include_pdf=False, model=None, rerank=None,
                  include_evaluation=False, stock_filter=None):
    """model=None 时向量腿用进程默认模型(MR_EMBED_MODEL,默认 bge-small-zh-v1.5);传预设名可单次切换,见 mr_index.MODELS。
    🔴 按源选模型(2026-09-08 用户 GPU 对照,同一 3116 块):笔记金标 small 18/18 .835 vs bge-m3 17/18 .822(一题之差,噪声级),
       PDF 金标 small 11/12 .694 vs bge-m3 12/12 .917——截尾的代价取决于块长和答案位置:笔记按标题分段主题在段首,PDF 一页一块答案在页中任何位置。
       所以 source="pdf" 且没显式传 model 时走 MR_EMBED_MODEL_PDF(没设=默认模型);笔记留 small(快 4 倍),原文页走 bge-m3。
    source=None 时默认只查 case/book/mem 三源;PDF 原文(要 MR_PDF=1 装载)只有 source="pdf" 或 include_pdf=True 才参与。
    🔴 为什么按查询开关而不是全混:2026-09-08 实测把 1296 页原文混进默认检索,主金标 hybrid 18/18→17/18、MRR .835→.750——
    原文页把带 ⭐ 页码的笔记挤下去;而 source=pdf 单独查,原文页 9/12 能找到。两层开关:环境变量管装不装,参数管查不查。"""
    if stock_filter is not None and source != "stock":
        raise ValueError("stock_filter requires source=stock")
    if model is None and source == "pdf":
        model = os.environ.get("MR_EMBED_MODEL_PDF") or None
    st = _ensure(mode, model, vec_scope='pdf') if source == 'pdf' else _ensure(mode, model)
    spec = mr_index.model_spec(model)
    chunks, note, used_model = st["chunks"], "", None
    if source:
        allowed = {source}
    elif include_pdf:
        allowed = mr_corpus.NOTE_SOURCES | {"pdf"}
    else:
        allowed = {"case", "book", "mem"}
    stock_scope = (mr_stock.query_filter(query, {c["file"] for c in chunks if c["source"] == "stock"}, stock_filter)
                   if source == "stock" else {})
    keep = lambda i: ((allowed is None or chunks[i]["source"] in allowed) and   # noqa: E731
                      (include_evaluation or chunks[i].get("purpose") != "evaluation") and
                      (not stock_scope or mr_stock.matches(chunks[i]["file"], stock_scope)))
    mask = {i for i in range(len(chunks)) if keep(i)}
    # 掩码 = 允许的全局下标;BM25 用对应子集索引(统计不串),向量层用掩码(先掩后切)
    if "pdf" in allowed and len(allowed) > 1:
        if "bm25_all" not in st:
            st["bm25_all"] = _SubBM25(chunks, list(range(len(chunks))))
        bm_index = st["bm25_all"]
    elif allowed == {"pdf"}:
        bm_index = st.get("bm25_pdf")
        if include_evaluation:
            if "bm25_pdf_evaluation" not in st:
                st["bm25_pdf_evaluation"] = _SubBM25(chunks, st["idx"]["pdf"])
            bm_index = st["bm25_pdf_evaluation"]
    elif allowed <= mr_corpus.NOTE_SOURCES:
        bm_index = st["bm25"]
        if include_evaluation:
            if "bm25_evaluation" not in st:
                st["bm25_evaluation"] = _SubBM25(chunks, st["idx"]["notes"])
            bm_index = st["bm25_evaluation"]
    else:
        # stock 保持原来的全体非 PDF 统计;不能套用曾使股票金标退步的独立股票 IDF。
        if "bm25_nonpdf" not in st:
            st["bm25_nonpdf"] = _SubBM25(chunks, st["idx"]["nonpdf"])
        bm_index = st["bm25_nonpdf"]
    K = 50
    bm = bm_index.search(query, topk=K, mask=mask) if bm_index else []
    leg_names, vl = None, None
    if mode == "bm25":
        ranked = bm
    else:
        vector_key = spec['slug'] + ('|pdf' if source == 'pdf' else '')
        v = st.get("vecs", {}).get(vector_key)
        if v is None or not v.available:
            ranked, note = bm, "向量层不可用(%s),已回退 bm25" % getattr(v, "err", "")
            mode = "bm25(fallback)"
        elif mode == "vec":
            vl = v.search(query, topk=K, mask=mask)
            ranked, used_model = vl, spec["key"]
        else:
            used_model = spec["key"]
            vl = v.search(query, topk=K, mask=mask)
            ranked = mr_index.rrf_files(chunks, [bm, vl], unit=("page" if allowed == {"pdf"} else "file"))
            leg_names = ("bm25", "vec")
    if source == "pdf" and not any(c["source"] == "pdf" for c in chunks):
        note = "PDF 源没装载:服务端要 MR_PDF=1 起(先 py -3.12 mr.py pdf-extract 抽一次文本)"
    # 查询级拒答门:两条腿的最高分都低于门槛 → 语料里很可能没有
    gate = None
    if GATE_MODE != "off":
        bm_max = max((s_ for _, s_ in bm), default=0.0) if mode != "vec" else None
        cos_max = max((s_ for _, s_ in vl), default=0.0) if vl is not None else None
        min_bm, min_cos = MIN_BM25["pdf" if allowed == {"pdf"} else "notes"], spec.get("min_cos")
        checks = [c for c in ((bm_max >= min_bm) if bm_max is not None else None,
                              (cos_max >= min_cos) if (cos_max is not None and min_cos is not None) else None) if c is not None]
        level = "high" if checks and all(checks) else ("low" if checks and not any(checks) else ("medium" if checks else "high"))
        gate = {"bm25_max": round(bm_max, 2) if bm_max is not None else None, "cos_max": round(cos_max, 3) if cos_max is not None else None,
                "min_bm25": min_bm, "min_cos": min_cos, "level": level}
        if level == "low":
            note = (note + ";" if note else "") + "⚠️ 两条腿都低于门槛(bm25 最高 %s / 余弦最高 %s),语料里很可能没有,别硬凑" % (
                gate["bm25_max"], gate["cos_max"])
        elif level == "medium":
            note = (note + ";" if note else "") + "ℹ️ 只有一条腿过线(bm25 最高 %s / 余弦最高 %s),命中可信度中等" % (gate["bm25_max"], gate["cos_max"])
    # 重排:融合榜前 RERANK_N 个候选(已过来源过滤)交给 cross-encoder 逐对打分,按它的分重排;它的分同时改写置信等级(标定过阈值的模型才改)
    rr = None
    want_rr = (rerank if rerank is not None else bool(os.environ.get("MR_RERANK")))
    if want_rr and ranked:
        rk = _get_reranker(rerank if isinstance(rerank, str) else None,
                           legacy=not allowed <= mr_corpus.NOTE_SOURCES)
        if rk is None:
            note = (note + ";" if note else "") + "重排未配置(MR_RERANK 或 rerank=<预设>)"
        elif not rk.available:
            note = (note + ";" if note else "") + "重排不可用(%s),按融合榜返回" % rk.err
        else:
            budget = rk.spec.get("gate_candidates",RERANK_G)
            max_windows = rk.spec.get("max_windows",RERANK_MAXWIN)
            fallback_budget = rk.spec.get("fallback_candidates",RERANK_FALLBACK_G)
            eligible = [row for row in ranked if keep(row[0])]
            cand = eligible[:(RERANK_N if RERANK_MODE == "rerank" else min(budget,topk))]
            t0 = time.time()
            pairs, owner = [], []
            for j, row in enumerate(cand):
                for w in _windows(chunks[row[0]],max_windows):
                    pairs.append(w)
                    owner.append(j)
            ws = rk.score(query, pairs)
            scores = [max(sw for sw, o in zip(ws, owner) if o == j) for j in range(len(cand))]
            hi, lo = rk.spec.get("hi"), rk.spec.get("lo")
            expanded_candidates = 0
            if RERANK_MODE == "gate" and lo is not None and scores and max(scores) < lo:
                # A smaller initial budget must not call a question unanswerable
                # while an already returned fourth/fifth candidate has evidence.
                tail = eligible[len(cand):min(fallback_budget,topk)]
                extra_pairs, extra_owner = [], []
                for j,row in enumerate(tail):
                    for window in _windows(chunks[row[0]],max_windows):
                        extra_pairs.append(window)
                        extra_owner.append(j)
                if extra_pairs:
                    extra_scores = rk.score(query,extra_pairs)
                    scores.extend(max(s for s,o in zip(extra_scores,extra_owner) if o==j) for j in range(len(tail)))
                    cand.extend(tail)
                    pairs.extend(extra_pairs)
                    expanded_candidates = len(tail)
            extra_passages = 0
            if (RERANK_MODE == "gate" and leg_names and hi is not None and lo is not None and
                    scores and max(scores) < hi):
                # A file's fused representative may cover a different aspect.
                # Before labeling the query weak, score at most ONE lexical body
                # from each already returned file/page. No new documents or
                # excluded evaluation material can enter through this path.
                def unit_key(i):
                    c = chunks[i]
                    return (c["source"], c["file"], c["heading"].split(" (")[0] if allowed == {"pdf"} else "")

                lexical = {}
                for i, _ in bm:
                    if keep(i) and mr_corpus.evidence_kind(chunks[i]) == "body":
                        lexical.setdefault(unit_key(i), i)
                alternatives = [(j, lexical[unit_key(row[0])]) for j, row in enumerate(cand)
                                if scores[j] < lo and unit_key(row[0]) in lexical and lexical[unit_key(row[0])] != row[0]]
                extra, owners = [], []
                for j, i in alternatives:
                    for window in _windows(chunks[i],max_windows):
                        extra.append(window)
                        owners.append(j)
                if extra:
                    extra_scores = rk.score(query, extra)
                    pairs.extend(extra)
                    extra_passages = len(alternatives)
                    replacements = {}
                    for j, i in alternatives:
                        score = max(s for s, owner_j in zip(extra_scores, owners) if owner_j == j)
                        # Only rescue a below-floor passage. An already relevant
                        # answer must not lose to a higher-scoring topic overview.
                        if score >= lo:
                            old = cand[j]
                            cand[j] = (i,) + old[1:]
                            scores[j] = score
                            replacements[old[0]] = cand[j]
                    ranked = [replacements.get(row[0], row) for row in ranked]
            if RERANK_MODE == "rerank":
                order = sorted(range(len(cand)), key=lambda j: -scores[j])
                ranked = [(cand[j][0], scores[j], (cand[j][2] if len(cand[j]) == 3 else None)) for j in order]
            else:   # gate:顺序不动,把分挂到对应块上
                _rr_scores = {cand[j][0]: scores[j] for j in range(len(cand))}
            rr = {"model": rk.spec["key"], "mode": RERANK_MODE, "n": len(cand), "pairs": len(pairs), "top": round(max(scores), 3),
                  "extra_passages": extra_passages, "expanded_candidates":expanded_candidates,
                  "max_length":rk.spec.get("max_length",512),"initial_candidates":budget,
                  "max_windows":max_windows,"window_chars":RERANK_WIN,
                  "ms": int((time.time() - t0) * 1000)}
            hi, lo = rk.spec.get("hi"), rk.spec.get("lo")
            if gate is not None and hi is not None and lo is not None:
                lvl = "high" if rr["top"] >= hi else ("medium" if rr["top"] >= lo else "low")
                gate.update({"rerank_top": rr["top"], "rerank_hi": hi, "rerank_lo": lo, "level_legs": gate["level"], "level": lvl})
                note = re.sub(r"(;)?(⚠️|ℹ️)[^;]*", "", note or "").strip(";")   # 两腿的置信说明作废,以重排为准
                if lvl == "low":
                    note = (note + ";" if note else "") + "⚠️ 重排最高分 %.2f 低于 %.2f,语料里很可能没有,别硬凑" % (rr["top"], lo)
                elif lvl == "medium":
                    note = (note + ";" if note else "") + "ℹ️ 重排最高分 %.2f(%.2f–%.2f),命中可信度中等" % (rr["top"], lo, hi)
    hits = []
    for row in ranked:
        i, score, legs = (row if len(row) == 3 else (row[0], row[1], None))
        c = chunks[i]
        if not keep(i):
            continue
        h = {"rank": len(hits) + 1, "score": round(float(score), 4),
             "source": c["source"], "file": c["file"], "heading": c["heading"],
             "snippet": mr_corpus.redact(_snippet(c["text"], query)), "chunk_id": c["id"],
             "purpose": c.get("purpose", "knowledge"), "evidence_kind": mr_corpus.evidence_kind(c)}
        if c["source"] != "pdf":
            h["chunk_md5"] = c["md5"]  # heading alone is ambiguous when a document repeats headings
        if c["source"] == "stock" and mr_stock.card_identity(c["file"]):
            h["stock"] = dict(mr_stock.card_identity(c["file"]))
        if rr is not None:
            if rr["mode"] == "rerank":
                h["rerank"] = round(float(score), 3)
            elif i in _rr_scores:
                h["rerank"] = round(float(_rr_scores[i]), 3)
        if legs is not None and leg_names:
            # 只披露机制(每条腿排第几),不下"低置信"判断——weak 旗试过一版当场撤:
            # "无腿前3=凑数"把双腿共识型正确答案(bm25#6+vec#7 靠合力赢)标成了低置信,
            # RRF 的价值恰恰是两票中庸胜过一票高分。置信判断要有自己的金标才配自动化,先留给读者。
            h["legs"] = "+".join(f"{n}#{r + 1}" for n, r in zip(leg_names, legs) if r is not None)
        hits.append(h)
        if len(hits) >= topk:
            break
    confidence = gate["level"] if gate else "high"
    if confidence == "low" and GATE_MODE == "drop":
        hits = []
    if any(h["purpose"] == "evaluation" for h in hits):
        note = (note + ";" if note else "") + "包含评测复盘:其中的样例问题不代表事实答案,请核对正文用途"
    _log({"tool": "search", "q": query, "mode": mode, "source": source, "model": used_model, "topk": topk, "n": len(hits),
          "top": ["%s#%s" % (h["file"], h["heading"]) for h in hits[:3]], "legs": [h.get("legs") for h in hits[:3]],
          "confidence": confidence, "gate": gate, "rerank": rr, "note": note or None,
          "include_evaluation": include_evaluation})
    return {"ok": True, "mode": mode, "tokenizer": mr_index.TOKENIZER, "model": used_model, "confidence": confidence,
            "gate": gate, "rerank": rr, "note": note, "hits": hits,
            **({"stock_filter": stock_scope, "stock_filter_origin": "explicit" if stock_filter is not None else "query"}
               if source == "stock" else {}),
            "corpus_sha256": st.get("corpus_sha256"), "corpus_audit": st.get("corpus_audit")}


def memory_get(file, source=None, heading=None, chunk_md5=None):
    r = _memory_get(file, source, heading, chunk_md5)
    _log({"tool": "get", "file": os.path.basename(file), "source": r.get("source") or source, "ok": r.get("ok")})
    return r


def _memory_get(file, source=None, heading=None, chunk_md5=None):
    if chunk_md5 is not None and (not isinstance(chunk_md5, str) or not re.fullmatch(r"[0-9a-f]{32}", chunk_md5)):
        return {"ok": False, "error": "chunk_md5 必须是 memory_search 返回的 32 位段落指纹"}
    # PDF 原文:file 形如「Agentic-AI指南.pdf#p81」→ 回那一页 ±1 页(页码 Acrobat 1-indexed);缓存没建就现抽(首次几分钟)
    m = re.match(r"^(.+?\.pdf)(?:#p(\d+))?$", os.path.basename(file), re.I)
    if m:
        if chunk_md5 is not None:
            return {"ok": False, "error": "PDF 请用 file 的 #p 页码读取；chunk_md5 仅用于 Markdown 段落"}
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
        text = "\n\n".join("── p%d ──\n%s" % (i, mr_corpus.redact(pages[i - 1])) for i in range(lo, hi + 1))
        purposes = mr_corpus.load_purposes()
        purpose = mr_corpus.chunk_purpose("pdf", m.group(1), "p%d" % p,
                                         purposes.get(("pdf", m.group(1)), "knowledge"), purposes)
        return {"ok": True, "source": "pdf", "file": m.group(1), "page": p, "pages": [lo, hi], "text": text, "purpose": purpose}
    for src in mr_corpus.SOURCES:
        p = os.path.join(src["dir"], os.path.basename(file))   # basename 防目录穿越
        if os.path.isfile(p) and (not source or src["key"] == source):
            with open(p, encoding="utf-8", errors="replace") as f:
                text = f.read()
            text = mr_corpus.redact(text)
            purposes = mr_corpus.load_purposes()
            purpose = mr_corpus.document_purpose(src["key"], os.path.basename(file), text, purposes)
            if heading is not None or chunk_md5 is not None:
                blocks = [(head, mr_corpus.redact(body)) for h in mr_corpus._split_heading(text)
                          for head, body in mr_corpus._hard_wrap(*h, strategy=mr_corpus.chunk_strategy(src["key"]))]
                requested = heading.strip() if heading is not None else None
                if chunk_md5 is not None:
                    matches = list(dict.fromkeys(block for block in blocks
                        if mr_corpus.chunk_key(os.path.basename(file), *block) == chunk_md5))
                    if len(matches) != 1:
                        return {"ok": False, "error": "段落内容已变化或指纹不存在；请重新 memory_search 后再读取",
                                "stale_reference": True}
                    if requested is not None and not matches[0][0].startswith(requested):
                        return {"ok": False, "error": "heading 与 chunk_md5 指向不同段落"}
                else:
                    matches = [block for block in blocks if block[0] == requested]
                    if not matches and requested:
                        matches = [block for block in blocks if block[0].startswith(requested)]
                if len(matches) != 1:
                    candidates = list(dict.fromkeys(head for head, _ in (matches or blocks)))
                    return {"ok": False, "error": "段落标题不存在或不唯一；请使用候选完整标题，或省略 heading 读取全文",
                            "candidates": candidates[:10], "candidate_count": len(candidates)}
                heading, text = matches[0]
                purpose = mr_corpus.chunk_purpose(src["key"], os.path.basename(file), heading, purpose, purposes)
            result = {"ok": True, "source": src["key"], "file": os.path.basename(file), "text": text, "purpose": purpose}
            if src["key"] == "stock" and mr_stock.card_identity(result["file"]):
                result["stock"] = dict(mr_stock.card_identity(result["file"]))
            if heading is not None:
                result["heading"] = heading
                result["chunk_md5"] = mr_corpus.chunk_key(os.path.basename(file), heading, text)
            return result
    return {"ok": False, "error": "语料目录都没有 %s(把 memory_search 结果里的 file 字段原样传)" % file}


def memory_check_answer(question, answer, evidence):
    """Read cited sources ourselves; caller-provided source text is not proof."""
    import mr_answer_eval
    import mr_support
    answer = mr_answer_eval.parse_answer(json.dumps(mr_corpus.redact_values(answer), ensure_ascii=False))
    if not isinstance(evidence, list) or len(evidence) > 20:
        raise ValueError("evidence must contain at most 20 source references")
    contexts, errors = [], []
    for reference in evidence:
        if (not isinstance(reference, dict)
                or set(reference) - {"id", "source", "file", "heading", "chunk_md5"}
                or not all(isinstance(reference.get(k), str) and reference[k] for k in ("id", "source", "file"))
                or reference["source"] not in mr_corpus.NOTE_SOURCES | {"pdf", "stock"}):
            raise ValueError("Evidence needs id/source/file and optional heading/chunk_md5; never supply text")
        fetched = _memory_get(**{k: v for k, v in reference.items() if k != "id"})
        if not fetched.get("ok") or fetched.get("source") != reference["source"]:
            errors.append({"id": reference["id"], "error": fetched.get("error", "Source identity mismatch")})
            continue
        text, heading = fetched["text"], fetched.get("heading", "(全文)")
        if fetched["source"] == "pdf":
            heading = "p" + str(fetched["page"])
            pages = re.split(r"(?:^|\n)── p(\d+) ──\n", text)
            text = next((pages[i + 1] for i in range(1, len(pages), 2) if pages[i] == str(fetched["page"])), "")
        contexts.append({"id": reference["id"], "source": fetched["source"], "file": fetched["file"],
                         "heading": heading, "text": text})
    if errors:
        return {"ok": False, "error": "引用资料读取失败；请重新搜索并获取有效段落。", "get_errors": errors}
    result = mr_support.check_answer(answer, contexts, question)
    _log({"tool": "check_answer", "sentences": len(result["sentences"]), "issues": len(result["issues"])})
    return {"ok": True, **result}


REGISTRY = {
    "memory_search": {
        "fn": memory_search,
        "desc": "查 zihao 以前踩过的坑/定过的原则/记过的事实。排查问题、写方案、做设计决策前先调它:"
                "输入自然语言问题,返回三层语料里最相关的片段——case(实战学习笔记)/book(两本 agent 教材章节笔记)/"
                "mem(个人长期记忆);股票历史判断用 source=stock;教材原文页用 source=pdf。默认不查 stock/PDF。查不到返回空 hits,也是正常结果;"
                "股票单标的+单日期/周期问题自动按卡片字段匹配；比较多日期/周期时保持宽搜，可用 stock_filter 明确范围。stock.card_date 是卡片日期，不保证正文每个价格都发生在该日。"
                "confidence 是检索相关性标签,不是答案正确率:未启用 cross-encoder 时由词法/语义分数门确定,启用时由 gate 分数确定。"
                "low 表示证据不足,别硬凑。hybrid 每条带 legs(词法/语义名次);优先返回同文件正文,仅有索引块时标 evidence_kind=navigation。"
                "snippet 是预览,引用前用 memory_get(file,source,heading,chunk_md5) 看完整段落；chunk_md5 可精确定位同名标题并拒绝过期片段。或省略 heading/指纹看全文。"
                "写好引用后可用 memory_check_answer 逐句查数字、日期和周期；检查通过仍需阅读原文确认结论含义。"
                "评测复盘默认排除;检索评测方法/运行复盘时用 include_evaluation=true,结果会标 purpose=evaluation。"
                "查询技巧:①语料以中文为主,查询用中文(英文短查询两条腿都弱——向量模型是中文特化的);"
                "②用意图里的罕见词问(如 直合/cherry-pick),别堆高频实体词(gerrit/合入 满语料都是,零区分度);"
                "③已知目标文件名就直接 memory_get,别绕检索。",
        "schema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "自然语言问题或关键词,中文为主"},
            "mode": {"type": "string", "enum": ["bm25", "vec", "hybrid"],
                     "description": "默认 hybrid;向量层没装会自动回退 bm25 并在 note 里说明"},
            "topk": {"type": "integer", "minimum": 1, "maximum": 10, "description": "返回条数,默认 5,上限 10"},
            "source": {"type": "string", "enum": ["case", "book", "mem", "pdf", "stock"], "description": "可选,只查某一源。默认只查 case/book/mem;股票历史判断用 stock;教材原文用 pdf(heading 是页码,再 memory_get 「xxx.pdf#p81」看整页)"},
            "include_pdf": {"type": "boolean", "description": "默认 false;true 时默认检索也混入 PDF 原文页(会挤掉笔记,一般别开)"},
            "include_evaluation": {"type": "boolean", "description": "默认 false,排除标记为 evaluation 的评测复盘;查评测方法/复盘时设 true。样例问题不等于事实答案"},
            "stock_filter": {"type": "object", "additionalProperties": False, "minProperties": 1,
                             "description": "仅 source=stock。显式限定卡片身份；不存在时返回空，不拿相邻日期/其他周期代替。",
                             "properties": {"ticker": {"type": "string"},
                                            "card_date": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
                                            "horizon": {"type": "string", "enum": list(mr_stock.HORIZONS)}}},
            "rerank": {"anyOf": [{"type": "boolean"}, {"type": "string", "enum": list(mr_index.RERANKERS)}],
                       "description": "可选,开关或 cross-encoder 预设名;默认跟 MR_RERANK。普通笔记 base gate 先评前 3 条,低于 lo 时最多补到原前 5 条（不超过 topk）;PDF/stock 保留原五候选预算。每个文件最多补评一段词法正文。文件顺序和 score 不变,另给 rerank 分与 confidence;"
                                      "仅服务端 MR_RERANK_MODE=rerank 时评分前 15 条并按该分重排(候选数可由环境变量调整)"},
            "model": {"type": "string", "enum": list(mr_index.MODELS),
                      "description": "可选,向量腿用哪个 embedding 模型,单次切换;不传=服务端默认(MR_EMBED_MODEL,默认 bge-small-zh-v1.5;"
                                     "source=pdf 时默认走 MR_EMBED_MODEL_PDF,原文页上 bge-m3 明显更准)。"
                                     "本地没权重会回退 bm25 并在 note 里说明(先 py -3.12 mr.py pull-model <名>)"},
        }, "required": ["query"]},
    },
    "memory_get": {
        "fn": memory_get,
        "desc": "取笔记/记忆/股票历史判断全文。把 memory_search 的 file/source 原样传入;heading 优先精确匹配,唯一前缀也可;歧义时返回候选标题。",
        "schema": {"type": "object", "properties": {
            "file": {"type": "string", "description": "文件名,来自 memory_search 结果的 file 字段;教材原文写成「xxx.pdf#p81」取那一页 ±1"},
            "source": {"type": "string", "enum": ["case", "book", "mem", "pdf", "stock"]},
            "heading": {"type": "string", "description": "可选,Markdown 标题优先精确匹配;唯一前缀可省略 (1) 等后缀,歧义或未找到时返回 candidates。省略则取全文;PDF 用 file 的 #p 页码"},
            "chunk_md5": {"type": "string", "pattern": "^[0-9a-f]{32}$", "description": "可选,原样传入 memory_search 返回的 Markdown 段落指纹。同名标题用它精确定位；内容改变时返回 stale_reference，须重新搜索。PDF 用 #p 页码，不传此项。"},
        }, "required": ["file"]},
    },
    "memory_check_answer": {
        "fn": memory_check_answer,
        "desc": "回答前逐句核对引用。服务端重读实际资料，检查引文、数字、价格日期、卡片周期及把价格快照推成窗口结果的错误。"
                "answer 用 decision=answer/insufficient、claims=[{text,citations:[{evidence_id,quote}]}]、reason；每条 claim 尽量一个事实。"
                "evidence 是 id 和 memory_get 的 source/file/heading/chunk_md5，不能传自编 text。PDF 一页一个 id，file 带 #p 页码。"
                "checks_pass=false 时必须修正/删去问题句或明确证据不足。true 仅代表未发现这些反例，不代表语义正确；仍逐句阅读引文，核对对象、日期、条件和否定。"
                "历史卡片日期不能替代每个价格的观测日期；计划与单点价格不能证明整个窗口是否触发或实际成交。本工具不调用模型。",
        "schema": {"type": "object", "additionalProperties": False, "properties": {
            "question": {"type": "string"},
            "answer": {"type": "object", "additionalProperties": False, "required": ["decision", "claims", "reason"],
                       "properties": {"decision": {"type": "string", "enum": ["answer", "insufficient"]},
                                      "reason": {"type": "string"},
                                      "claims": {"type": "array", "maxItems": 12, "items": {
                                          "type": "object", "additionalProperties": False, "required": ["text", "citations"],
                                          "properties": {"text": {"type": "string"}, "citations": {
                                              "type": "array", "minItems": 1, "maxItems": 10, "items": {
                                                  "type": "object", "additionalProperties": False, "required": ["evidence_id", "quote"],
                                                  "properties": {"evidence_id": {"type": "string"}, "quote": {"type": "string"}}}}}}}}},
            "evidence": {"type": "array", "maxItems": 20, "items": {
                "type": "object", "additionalProperties": False, "required": ["id", "source", "file"],
                "properties": {"id": {"type": "string"}, "source": {"type": "string", "enum": ["case", "book", "mem", "pdf", "stock"]},
                               "file": {"type": "string"}, "heading": {"type": "string"},
                               "chunk_md5": {"type": "string", "pattern": "^[0-9a-f]{32}$"}}}},
        }, "required": ["question", "answer", "evidence"]},
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
