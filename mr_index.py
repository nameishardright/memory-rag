# -*- coding: utf-8 -*-
"""检索层三形态:BM25(零依赖手写) / 本地向量(bge-small-zh,懒加载+缓存) / RRF 混合。
向量依赖缺失或模型拉不下来时 available=False,上层自动回退 BM25,不炸。
"""
import math
import os
import re

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # GFW:HuggingFace 走镜像

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, ".cache")
MODEL = "BAAI/bge-small-zh-v1.5"          # 24M 参数中文 embedding,CPU 秒级
# 本地模型优先:hf 下载会被 Windows 注册表系统代理(v2rayN)劫持,curl 预拉到 .cache 则永久离线可用
MODEL_LOCAL = os.path.join(BASE, ".cache", "bge-small-zh-v1.5")


def _model_path():
    return MODEL_LOCAL if os.path.isfile(os.path.join(MODEL_LOCAL, "model.safetensors")) else MODEL
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："  # bge-zh 官方建议的查询指令

_ASCII = re.compile(r"[A-Za-z0-9_./:\-]+")
_CJK = re.compile(r"[一-鿿]+")

try:
    if os.environ.get("MR_TOKENIZER") == "bigram":   # 实验开关:量化"分词器换掉损失多少"
        raise ImportError("MR_TOKENIZER=bigram 强制退化")
    import jieba                            # 可选:纯 python 分词;没装退化到 bigram
    jieba.setLogLevel(60)

    def _cjk_cut(s):
        return [w for w in jieba.cut_for_search(s) if w.strip()]
    TOKENIZER = "jieba"
except ImportError:
    def _cjk_cut(s):
        return [s[i:i + 2] for i in range(len(s) - 1)] or [s]
    TOKENIZER = "bigram"


def tokenize(text):
    """中文没有空格,分词是检索第一关:ASCII 串按词,中文按 jieba/字符 bigram。"""
    toks = [t.lower() for t in _ASCII.findall(text)]
    for run in _CJK.findall(text):
        toks.extend(_cjk_cut(run))
    return toks


def _index_text(c):
    # 文件名+标题也进索引:短文件的主题词经常只在文件名里
    return c["file"] + " " + c["heading"] + " " + c["text"]


class BM25:
    K1, B = 1.5, 0.75

    _TOK_CACHE = {}   # md5 → tokens:热更重建时只切新块(2026-09-08 实测 1816 块全切要 ~10s,占重建时间大头)

    def __init__(self, chunks):
        self.chunks = chunks
        cache = BM25._TOK_CACHE
        self.docs = []
        for c in chunks:
            key = c["md5"] + "|" + c["file"] + "|" + c["heading"]   # _index_text 含文件名+标题,同正文不同标题分词不同
            if key not in cache:
                cache[key] = tokenize(_index_text(c))
            self.docs.append(cache[key])
        if len(cache) > 4 * max(1, len(chunks)):   # 语料反复增删时别无限涨
            cache.clear()
        self.N = len(self.docs)
        self.avgdl = sum(map(len, self.docs)) / max(1, self.N)
        self.tf, self.df = [], {}
        for toks in self.docs:
            d = {}
            for t in toks:
                d[t] = d.get(t, 0) + 1
            self.tf.append(d)
            for t in d:
                self.df[t] = self.df.get(t, 0) + 1

    def search(self, query, topk=50, mask=None):
        scores = [0.0] * self.N
        for t in tokenize(query):
            df = self.df.get(t)
            if not df:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for i, d in enumerate(self.tf):
                f = d.get(t)
                if f:
                    dl = len(self.docs[i])
                    scores[i] += idf * f * (self.K1 + 1) / (
                        f + self.K1 * (1 - self.B + self.B * dl / self.avgdl))
        cand = range(self.N) if mask is None else [i for i in range(self.N) if i in mask]
        order = sorted(cand, key=lambda i: -scores[i])[:topk]
        return [(i, scores[i]) for i in order if scores[i] > 0]


class VecIndex:
    """embedding 按 chunk md5 缓存(.cache/):语料改一块只重算一块。"""

    def __init__(self, chunks, model=None):
        self.chunks, self.available, self.err = chunks, False, ""
        self._model = model     # 热更重建时把旧实例的模型传进来,省掉再加载
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError as e:
            self.err = "依赖未装(%s),先跑: py -3.12 mr.py setup-vec" % e.name
            return
        try:
            import json
            self.np = np
            os.makedirs(CACHE, exist_ok=True)
            meta_p = os.path.join(CACHE, "emb_meta.json")
            emb_p = os.path.join(CACHE, "emb.npy")
            old = {}
            if os.path.exists(meta_p) and os.path.exists(emb_p):
                try:
                    with open(meta_p, encoding="utf-8") as f:
                        m = json.load(f)
                    arr = np.load(emb_p)
                    if m.get("model") == MODEL and len(m.get("md5s", [])) == len(arr):
                        old = {h: arr[i] for i, h in enumerate(m["md5s"])}
                except Exception:
                    old = {}
            todo = [c for c in chunks if c["md5"] not in old]
            if todo:
                vecs = self._get_model().encode(
                    [_index_text(c) for c in todo],
                    normalize_embeddings=True, show_progress_bar=False)
                for c, v in zip(todo, vecs):
                    old[c["md5"]] = v
            self.emb = np.stack([old[c["md5"]] for c in chunks])
            # 缓存存「见过的全部」而不是「当前这批」:MR_PDF 开关来回切、或某源临时缺目录时,不用把 1100 页重算一遍
            keys = list(old.keys())
            with open(meta_p, "w", encoding="utf-8") as f:
                json.dump({"model": MODEL, "md5s": keys}, f)
            np.save(emb_p, np.stack([old[k] for k in keys]))
            self.available = True
        except Exception as e:                      # 模型下载失败/磁盘问题等,都降级不炸
            self.err = "%s: %s" % (type(e).__name__, e)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(_model_path())
        return self._model

    def search(self, query, topk=50, mask=None):
        qv = self._get_model().encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        sims = self.emb @ qv
        if mask is not None:   # 先掩后切:否则 PDF 页先占掉前 K 个名次,剔完只剩空榜
            idx = self.np.fromiter(mask, dtype=int)
            if len(idx) == 0:
                return []
            order = idx[sims[idx].argsort()[::-1][:topk]]
        else:
            order = sims.argsort()[::-1][:topk]
        return [(int(i), float(sims[i])) for i in order]


def rrf(rank_lists, k=60, topk=50):
    """Reciprocal Rank Fusion:只看名次不看分值,免掉两套分数(BM25 无界/余弦 0-1)的归一化难题。
    ⚠️ chunk 粒度融合有坑:同文件不同 chunk 的票会被切碎(实验 2b 实锤,q7 两路都靠前仍 miss),
    检索的消费单元是文件时用 rrf_files。保留本函数当对照教材。"""
    agg = {}
    for lst in rank_lists:
        for r, (i, _s) in enumerate(lst):
            agg[i] = agg.get(i, 0.0) + 1.0 / (k + r + 1)
    return sorted(agg.items(), key=lambda kv: -kv[1])[:topk]


def rrf_files(chunks, rank_lists, k=60, topk=50, unit="file"):
    """文件级 RRF:先把每路 chunk 榜折成文件榜(取该文件最好名次),再融合——
    索引单元(chunk)和消费单元(文件)不一致时,融合必须发生在消费单元上。
    返回 [(代表chunk_idx, score, per_leg_ranks)];per_leg_ranks 与 rank_lists 对齐(0起,None=该腿没投)。
    腿名次同时是置信信号:没有任何一条腿排进前3的文件多半是凑数——
    「分差断崖」的机制版,比分数百分比阈值诚实(RRF 分数天然贴平,光看分看不出谁在凑)。"""
    # unit="page":PDF 源用,消费单元是「文件#页」而不是文件——一本书 700 页,按文件折叠就永远只出一页
    key_of = (lambda c: c["file"] + "#" + c["heading"].split(" (")[0]) if unit == "page" else (lambda c: c["file"])
    file_rank_per_list, best_chunk = [], {}
    for lst in rank_lists:
        fr = {}
        for r, (i, _s) in enumerate(lst):
            f = key_of(chunks[i])
            if f not in fr:
                fr[f] = r
                if f not in best_chunk:
                    best_chunk[f] = i
        file_rank_per_list.append(fr)
    agg = {}
    for li, fr in enumerate(file_rank_per_list):
        for f, r in fr.items():
            if f not in agg:
                agg[f] = [0.0, [None] * len(rank_lists)]
            agg[f][0] += 1.0 / (k + r + 1)
            agg[f][1][li] = r
    ranked = sorted(agg.items(), key=lambda kv: -kv[1][0])[:topk]
    return [(best_chunk[f], s, legs) for f, (s, legs) in ranked]
