# -*- coding: utf-8 -*-
"""检索层三形态:BM25(零依赖手写) / 本地向量(bge-small-zh 默认,可一键切 bge-m3 / Qwen3,懒加载+按模型分缓存) / RRF 混合。
向量依赖缺失或模型拉不下来时 available=False,上层自动回退 BM25,不炸。
"""
import math
import os
import re

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # GFW:HuggingFace 走镜像

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, ".cache")                                   # embedding 缓存 / PDF 文本
MODELS_DIR = os.environ.get("MR_MODELS_DIR") or CACHE                    # 模型权重目录:<MODELS_DIR>/<模型名>/

# ── embedding 模型预设:一键切换(2026-09-08)──
# 选谁:memory_search(model=) > 环境变量 MR_EMBED_MODEL > 默认 bge-small-zh-v1.5;键不分大小写,也认短名 small/m3/qwen3。
# 🔴 每个预设自带 query 前缀:bge-zh v1.5 要那句中文指令,bge-m3 不要,Qwen3 要 Instruct 格式——
#    换模型不换前缀会无声掉分,金标只会告诉你"变差了"不会告诉你为什么。
# 权重放 .cache/<名>/(gitignore),`py -3.12 mr.py pull-model <名>` 从 hf-mirror 拉;embedding 缓存按模型分文件,切回来不重算。
# min_cos = 拒答门的余弦阈值(查询级,看向量腿最高分;和 BM25 阈值一起用,见 mr_tools.memory_search),按 golden_negative.json 标定,换模型要重标。
MODELS = {
    "bge-small-zh-v1.5": {"id": "BAAI/bge-small-zh-v1.5", "prefix": "为这个句子生成表示以用于检索相关文章：",
                          "weights": "model.safetensors", "extra": ["vocab.txt", "sentence_bert_config.json"], "min_cos": 0.48,
                          "note": "24M 参数,512 token,92MB,CPU 毫秒级;语料 40% 块超限被截尾"},
    "bge-m3": {"id": "BAAI/bge-m3", "prefix": "", "weights": "pytorch_model.bin",   # 仓里没有 safetensors
               "extra": ["sentencepiece.bpe.model", "sentence_bert_config.json"], "min_cos": 0.52,
               "note": "568M 参数,8192 token,2.27GB;查询不加指令"},
    "Qwen3-Embedding-0.6B": {"id": "Qwen/Qwen3-Embedding-0.6B", "weights": "model.safetensors",
                             "prefix": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
                             "extra": ["merges.txt", "vocab.json"], "min_cos": None,   # 没标定过,余弦门不启用
                             "note": "0.6B 参数,32k token,1.19GB;查询要 Instruct 前缀"},
}
_ALIASES = {"small": "bge-small-zh-v1.5", "m3": "bge-m3", "qwen3": "Qwen3-Embedding-0.6B", "qwen3-0.6b": "Qwen3-Embedding-0.6B"}
DEFAULT_MODEL = "bge-small-zh-v1.5"
COMMON_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "modules.json", "config_sentence_transformers.json", "1_Pooling/config.json"]


# ── 重排模型(cross-encoder,2026-09-08 项 7):query 和候选段落拼在一起过一遍模型直接出相关分,比「两条腿各自的分」分得开。
# 只对融合榜前 RERANK_N 个候选打分(每对一次前向,CPU 上 base 约 0.1-0.2s/对),所以它是第三级,不替代召回。
# hi / lo = 置信阈值(sigmoid 后 0-1),按 golden_negative 标定(2026-09-08:两个模型正例中位 ≥ .8、阴性中位 ≤ .05,.50/.10 两边都有余量);没标定的模型只打分不改置信。
# 🔴 2026-09-08 标定结论:当**重排**用在两套金标上都掉分(笔记 hybrid .835→.692(base)/.764(v2-m3),PDF .917→.748/.817)——
#    RRF 前几名已经是对的,cross-encoder 只看一个块的前 ~480 字、又偏爱「像答案」的段落,把对的文件往下压;当**置信信号**用却远比两腿分数门强。
#    所以默认 MR_RERANK_MODE=gate(只打分定置信不动顺序),rerank 模式留作对照。
RERANKERS = {
    "bge-reranker-base": {"id": "BAAI/bge-reranker-base", "weights": "model.safetensors", "extra": ["sentencepiece.bpe.model"],
                          "hi": 0.50, "lo": 0.10, "note": "278M(xlm-roberta-base),中英,1.1GB"},
    "bge-reranker-v2-m3": {"id": "BAAI/bge-reranker-v2-m3", "weights": "model.safetensors", "extra": ["sentencepiece.bpe.model"],
                           "hi": 0.50, "lo": 0.10, "note": "568M(bge-m3 底座),多语,2.27GB,比 base 慢约 2 倍"},
}
_RR_ALIASES = {"rerank-base": "bge-reranker-base", "base": "bge-reranker-base", "rerank-m3": "bge-reranker-v2-m3", "v2-m3": "bge-reranker-v2-m3"}
RERANKER_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


def reranker_spec(key=None):
    key = (key or os.environ.get("MR_RERANK") or "").strip()
    if not key:
        return None
    k = _RR_ALIASES.get(key.lower(), key)
    hit = next((n for n in RERANKERS if n.lower() == k.lower()), None)
    if hit:
        spec = dict(RERANKERS[hit], key=hit, local=os.path.join(MODELS_DIR, hit))
    elif os.path.isdir(key):
        name = os.path.basename(os.path.normpath(key))
        spec = {"key": name, "id": name, "local": key, "weights": "model.safetensors", "extra": [], "hi": None, "lo": None, "note": "本地目录"}
    else:
        name = key.split("/")[-1]
        spec = {"key": name, "id": key, "local": os.path.join(MODELS_DIR, name), "weights": "model.safetensors", "extra": [], "hi": None, "lo": None, "note": "自定义"}
    spec["files"] = RERANKER_FILES + list(spec.get("extra", [])) + [spec["weights"]]
    return spec


class Reranker:
    """cross-encoder 打分器:score(query, passages) → [0-1]。权重缺失/依赖缺失 available=False,上层跳过重排并在 note 里说明。"""

    def __init__(self, spec):
        self.spec, self.available, self.err, self._model = spec, False, "", None
        if not has_weights(spec) and not os.environ.get("MR_EMBED_ALLOW_DOWNLOAD"):
            self.err = "重排模型 %s 没有本地权重(%s),先跑: py -3.12 mr.py pull-model %s" % (spec["key"], spec["local"], spec["key"])
            return
        try:
            from sentence_transformers import CrossEncoder  # noqa: F401
            self.available = True
        except ImportError as e:
            self.err = "依赖未装(%s),先跑: py -3.12 mr.py setup-vec" % e.name

    def _get(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_path(self.spec), max_length=512)
        return self._model

    def score(self, query, passages, batch_size=16):
        import numpy as np
        out = self._get().predict([(query, p) for p in passages], batch_size=batch_size, show_progress_bar=False)
        out = np.asarray(out, dtype=float).reshape(-1)
        if out.size and (out.min() < 0 or out.max() > 1):   # 有的版本回 logit,统一成 0-1
            out = 1.0 / (1.0 + np.exp(-out))
        return [float(x) for x in out]


def model_spec(key=None):
    """预设名 / 短名 / 任意 HF id / 本地目录 → {key, id, local, prefix, weights, extra, slug}。
    没权重不在这里报错,留给 VecIndex 降级(hybrid 退 bm25 并在 note 里说明)。"""
    key = (key or os.environ.get("MR_EMBED_MODEL") or DEFAULT_MODEL).strip()
    k = _ALIASES.get(key.lower(), key)
    hit = next((n for n in MODELS if n.lower() == k.lower()), None)
    if hit:
        spec = dict(MODELS[hit], key=hit, local=os.path.join(MODELS_DIR, hit))
    elif os.path.isdir(key):                                   # 自己 scp 来的权重目录
        name = os.path.basename(os.path.normpath(key))
        spec = {"key": name, "id": name, "local": key, "prefix": os.environ.get("MR_EMBED_PREFIX", ""),
                "weights": "model.safetensors", "extra": [], "note": "本地目录"}
    else:                                                      # 任意 HF id:前缀自己用 MR_EMBED_PREFIX 给
        name = key.split("/")[-1]
        spec = {"key": name, "id": key, "local": os.path.join(MODELS_DIR, name),
                "prefix": os.environ.get("MR_EMBED_PREFIX", ""), "weights": "model.safetensors", "extra": [], "note": "自定义"}
    spec["slug"] = re.sub(r"[^A-Za-z0-9._-]+", "_", spec["key"])
    return spec


def has_weights(spec):
    return any(os.path.isfile(os.path.join(spec["local"], w))
               for w in dict.fromkeys([spec["weights"], "model.safetensors", "pytorch_model.bin"]))


def model_path(spec):
    # 本地权重优先:hf 下载会被 Windows 注册表系统代理(v2rayN)劫持,curl 预拉到 .cache 则永久离线可用
    return spec["local"] if has_weights(spec) else spec["id"]


# 向后兼容的模块级常量 = 进程默认模型(旧代码 / 实验脚本还读这几个名字)
_DEF = model_spec()
MODEL, MODEL_LOCAL, QUERY_PREFIX = _DEF["id"], _DEF["local"], _DEF["prefix"]


def _model_path():
    return model_path(_DEF)

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
            key = c["md5"]   # md5 已含文件名+标题+正文(mr_corpus.chunk_key),就是 _index_text 的指纹
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


CACHE_VER = 2   # 改 _index_text / chunk_key / 分块规则时 +1:老缓存整体作废重算,而不是新旧键混在一个文件里越攒越大(v2 = 键含文件名+标题)


class VecIndex:
    """embedding 按 chunk md5 缓存(.cache/emb-<模型>.npy):语料改一块只重算一块;缓存按模型分文件,切换模型不互相冲掉。"""

    def __init__(self, chunks, model=None, spec=None):
        self.spec = spec or model_spec()
        self.chunks, self.available, self.err = chunks, False, ""
        self._model = model     # 热更重建时把旧实例的模型传进来,省掉再加载
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError as e:
            self.err = "依赖未装(%s),先跑: py -3.12 mr.py setup-vec" % e.name
            return
        if not has_weights(self.spec) and not os.environ.get("MR_EMBED_ALLOW_DOWNLOAD"):
            # 宁可明说也不偷偷去 HF 拉:内网拉不动、代理劫持、拉一半的权重更难查。要联网拉设 MR_EMBED_ALLOW_DOWNLOAD=1
            self.err = "模型 %s 没有本地权重(%s),先跑: py -3.12 mr.py pull-model %s" % (
                self.spec["key"], self.spec["local"], self.spec["key"])
            return
        try:
            import json
            self.np = np
            os.makedirs(CACHE, exist_ok=True)
            meta_p = os.path.join(CACHE, "emb_meta-%s.json" % self.spec["slug"])
            emb_p = os.path.join(CACHE, "emb-%s.npy" % self.spec["slug"])
            self._migrate_legacy(meta_p, emb_p)
            old = {}
            if os.path.exists(meta_p) and os.path.exists(emb_p):
                try:
                    with open(meta_p, encoding="utf-8") as f:
                        m = json.load(f)
                    arr = np.load(emb_p)
                    if m.get("model") == self.spec["id"] and m.get("ver") == CACHE_VER and len(m.get("md5s", [])) == len(arr):
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
                json.dump({"model": self.spec["id"], "ver": CACHE_VER, "md5s": keys}, f)
            np.save(emb_p, np.stack([old[k] for k in keys]))
            self.available = True
        except Exception as e:                      # 模型加载失败/磁盘问题等,都降级不炸
            self.err = "%s: %s" % (type(e).__name__, e)

    def _migrate_legacy(self, meta_p, emb_p):
        """老缓存文件名 emb.npy / emb_meta.json(单模型时代)→ 按模型分文件;只在 meta 里的模型对得上时改名,不重算。"""
        import json
        lm, le = os.path.join(CACHE, "emb_meta.json"), os.path.join(CACHE, "emb.npy")
        if os.path.exists(meta_p) or not (os.path.exists(lm) and os.path.exists(le)):
            return
        try:
            with open(lm, encoding="utf-8") as f:
                same = json.load(f).get("model") == self.spec["id"]
            if same:   # 🔴 改名必须在 with 块外:Windows 上文件还开着 os.replace 报 PermissionError,吞掉就成了静默全量重算(2026-09-08 踩)
                os.replace(lm, meta_p)
                os.replace(le, emb_p)
        except Exception:
            pass

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_path(self.spec))
        return self._model

    def search(self, query, topk=50, mask=None):
        qv = self._get_model().encode([self.spec["prefix"] + query], normalize_embeddings=True)[0]
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
    # 文档身份 = 源/文件名:三个语料目录各自独立,同名文件是两份文档,只按文件名折叠会把它们的票合并(2026-09-08 评审项④;当时跨源同名 0 个,潜伏)
    doc = lambda c: c["source"] + "/" + c["file"]   # noqa: E731
    key_of = (lambda c: doc(c) + "#" + c["heading"].split(" (")[0]) if unit == "page" else doc
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
