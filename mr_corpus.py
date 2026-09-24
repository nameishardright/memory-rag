# -*- coding: utf-8 -*-
"""语料层:三源加载 → 按标题切块 → 凭据脱敏(只对 CC memory 源)。
设计决策与概念对应见 Desktop\\Agent学习计划\\notes\\第5-6周-毕业项目2.0-记忆RAG.md
"""
import hashlib
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")

SOURCES = [
    {"key": "case", "name": "rh学习记忆(实战原则)",
     "dir": os.path.join(HOME, r"Desktop\Agent学习计划\rh学习记忆"), "redact": False},
    {"key": "book", "name": "章节笔记(两本教材)",
     "dir": os.path.join(HOME, r"Desktop\Agent学习计划\notes"), "redact": False},
    {"key": "mem", "name": "CC长期记忆",
     "dir": os.path.join(HOME, r".claude\projects\C--Users-zihao\memory"), "redact": True},
    # 第五源(2026-09-16):股票判断案例库——stock-analysis 的 case_lib.py 把每张历史判断卡(三个数/依据/自我否决/结果)写成一份 md,
    # 目录不存在就静默跳过;私有数据,永不进公开仓(只有路径在代码里)。决策时 memory_search(source="stock") 回放同类旧账。
    {"key": "stock", "name": "股票判断案例库",
     "dir": os.environ.get("MR_STOCK_DIR") or os.path.join(HOME, r"Desktop\股票分析\cases\md"), "redact": True},
    # 第六源(2026-09-24):海马云工作笔记,唯一递归的源——file 是相对路径(方案/xxx.md),子目录里同名 README 很多。
    # 私有数据,永不进公开仓。默认检索要不要带上它由 MR_WORK_DEFAULT 决定,见 DEFAULT_SOURCES。
    {"key": "work", "name": "海马云工作笔记",
     "dir": os.environ.get("MR_WORK_DIR") or os.path.join(HOME, r"Desktop\海马云"), "redact": True, "recursive": True},
]
DENY = {"MEMORY.md"}          # 纯索引,内容都在单文件里,入库只会造重复命中
NOTE_SOURCES = frozenset({"case", "book", "mem"})
# work 源递归时剪掉的目录:依赖/构建产物/本仓自己(本仓 docs 引用金标问句,入库会触发污染审计)
WORK_SKIP_DIRS = frozenset({"node_modules", ".git", ".venv", "venv", "site-packages", ".pytest_cache", "__pycache__",
                            "dist", "build", ".cache", "memory_rag", "openapi-metrics-preview"})
# 凭据台账整份不入库(keys-台账.md / <vendor>_keys.md):正则脱敏兜不住表格里的裸 key
RE_SECRET_FILE = re.compile(r"(?i)(^|[-_])keys?([-_.]|$)")


def work_default():
    return os.environ.get("MR_WORK_DEFAULT", "").strip().lower() in ("1", "true", "on", "yes")


# source=None 时查哪些源;work 是否并入默认检索按金标对照定(开关两态都跑)
DEFAULT_SOURCES = NOTE_SOURCES | ({"work"} if work_default() else frozenset())
PARAGRAPH_SOURCES = NOTE_SOURCES | {"work"}


def _scan(src):
    """[(file_id, path, stat)],按 file_id 排序。平铺源 file_id=文件名;递归源 file_id=相对路径(/ 分隔),
    剪 WORK_SKIP_DIRS、跳凭据台账。用 scandir 的 DirEntry.stat():Windows 上 mtime/size 随目录枚举一起回来,
    不再逐文件发系统调用——指纹每 RELOAD_SEC 在查询路径上算一次,2000 个文件逐个 os.stat 要 300~450ms(2026-09-24 实测)。"""
    d = src["dir"]
    if not os.path.isdir(d):
        return []
    out, stack = [], [d]
    while stack:
        root = stack.pop()
        try:
            with os.scandir(root) as it:
                entries = list(it)
        except OSError:
            continue
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    if src.get("recursive") and e.name not in WORK_SKIP_DIRS:
                        stack.append(e.path)
                elif (e.name.endswith(".md") and e.name not in DENY
                      and not (src.get("recursive") and RE_SECRET_FILE.search(e.name))):
                    out.append((os.path.relpath(e.path, d).replace(os.sep, "/"), e.path, e.stat()))
            except OSError:
                pass
    return sorted(out, key=lambda x: x[0])


def source_files(src):
    """[(file_id, path)],按 file_id 排序;规则见 _scan。"""
    return [(fid, path) for fid, path, _st in _scan(src)]


def resolve_file(src, file):
    """memory_get 的文件定位。平铺源取 basename(防目录穿越);递归源只认清单里的 file_id,
    也接受唯一的裸文件名。返回 (file_id, path) 或 None。"""
    if not src.get("recursive"):
        fn = os.path.basename(file)
        p = os.path.join(src["dir"], fn)
        return (fn, p) if os.path.isfile(p) else None
    files = dict(source_files(src))
    fid = file.replace("\\", "/")
    if fid in files:
        return fid, files[fid]
    same = [k for k in files if k.rsplit("/", 1)[-1] == fid]
    return (same[0], files[same[0]]) if len(same) == 1 else None
PURPOSES = frozenset({"knowledge", "evaluation"})
CORPUS_POLICY = os.environ.get("MR_CORPUS_POLICY") or os.path.join(BASE, "corpus-policy.json")


def load_purposes():
    """Explicit document-use labels; never infer them from a query or golden answers."""
    if not os.path.isfile(CORPUS_POLICY):
        return {}
    with open(CORPUS_POLICY, encoding="utf-8") as f:
        policy = json.load(f)
    if policy.get("version") != 1:
        raise ValueError("corpus-policy.json: unsupported version")
    result = {}
    for entry in policy["documents"] + policy.get("sections", []) + policy.get("pdf_pages", []):
        key = (entry["source"], entry["file"])
        if "heading_prefix" in entry:
            if not isinstance(entry["heading_prefix"], str) or not entry["heading_prefix"].strip():
                raise ValueError("corpus-policy.json: empty section prefix")
            key += (entry["heading_prefix"],)
        # 平铺源只收裸文件名;递归源(work)收 / 分隔的相对路径,不许 .. / 反斜杠 / 绝对路径
        recursive = any(s["key"] == key[0] and s.get("recursive") for s in SOURCES)
        valid_id = (os.path.basename(key[1]) == key[1] or
                    (recursive and "\\" not in key[1] and not key[1].startswith("/") and ".." not in key[1].split("/")))
        if (key[0] not in ({s["key"] for s in SOURCES} | {"pdf"}) or
                not valid_id or entry["purpose"] not in PURPOSES or key in result):
            raise ValueError("corpus-policy.json: invalid or duplicate document label")
        result[key] = entry["purpose"]
    return result


def document_purpose(source, file, text, purposes):
    """A sidecar label overrides the optional top-level `rag_purpose` frontmatter key."""
    purpose = "knowledge"
    frontmatter = re.match(r"\A\ufeff?---\s*\n(.*?)\n---(?:\s*\n|\Z)", text, re.S)
    if frontmatter:
        label = re.search(r"^rag_purpose:\s*([\w-]+)\s*$", frontmatter[1], re.M)
        if label:
            purpose = label[1]
            if purpose not in PURPOSES:
                raise ValueError("Unknown rag_purpose in %s" % file)
    return purposes.get((source, file), purpose)


def chunk_purpose(source, file, heading, default, purposes):
    """Mixed teaching documents can label only their evaluation/report sections."""
    def matches_prefix(prefix):
        if source == "pdf":
            return heading == prefix or heading.startswith(prefix + " (")
        return heading.startswith(prefix)
    matches = [(len(key[2]), purpose) for key, purpose in purposes.items()
               if len(key) == 3 and key[:2] == (source, file) and matches_prefix(key[2])]
    return max(matches)[1] if matches else default


def evidence_kind(chunk):
    """Keep navigation searchable, but prefer an answer-bearing body in the same file."""
    if chunk["heading"].startswith("索引行归档"):
        return "navigation"
    if chunk["heading"] == "(开头)":
        lines = chunk["text"].strip().splitlines()
        if lines and lines[0].lstrip("\ufeff").rstrip() == "---":
            closing = next((i for i, line in enumerate(lines[1:], 1) if line.rstrip() == "---"), None)
            if closing is not None and not any(line.strip() for line in lines[closing + 1:]):
                return "navigation"
    return "body"

# ── 第四源:教材 PDF 原文,按页切块,开关 MR_PDF=1(2026-09-08 用户定「PDF 源做成开关跑金标」)──
# 笔记层(book)是蒸馏过的、带 ⭐ 页码,答「对应哪一章」更准;原文层是给「全面」用的。
# 块数会翻倍,所以默认关:主金标 18 条是尺子,开关两态各跑一遍,数字说了算。
PDF_DIR = os.path.join(HOME, r"Desktop\Agent学习计划")
PDF_SOURCE = {"key": "pdf", "name": "教材原文(PDF 按页)", "dir": PDF_DIR, "redact": False}
PDF_CACHE = os.path.join(BASE, ".cache", "pdf-text")   # .cache/ 已 gitignore:书的原文绝不能进公开仓
MIN_PAGE_CHARS = 40                                     # 空白页/纯图页跳过


def pdf_enabled():
    return os.environ.get("MR_PDF", "").strip().lower() in ("1", "true", "on", "yes")


def pdf_files():
    if not os.path.isdir(PDF_DIR):
        return []
    return sorted(os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))


def extract_pdf(path, progress=None):
    """按页抽文本(pypdf + NFKC 归一化,康熙部首 ⼀ 类归成常用字),按 (size, mtime) 缓存到 PDF_CACHE。
    全书一次 4~6 分钟,只在缓存缺失/PDF 变了时跑;页码 = Acrobat 1-indexed(和笔记里的 ⭐ 页一致)。"""
    import json
    import unicodedata
    st = os.stat(path)
    os.makedirs(PDF_CACHE, exist_ok=True)
    cp = os.path.join(PDF_CACHE, os.path.splitext(os.path.basename(path))[0] + ".json")
    if os.path.isfile(cp):
        try:
            with open(cp, encoding="utf-8") as f:
                c = json.load(f)
            if c.get("size") == st.st_size and int(c.get("mtime", -1)) == int(st.st_mtime):
                return c["pages"]
        except Exception:
            pass
    import pypdf
    r = pypdf.PdfReader(path)
    pages = []
    for i, p in enumerate(r.pages):
        try:
            t = p.extract_text() or ""
        except Exception as e:                       # 个别页解析炸了不影响全书
            t = ""
            print("[pdf] %s p%d 抽取失败: %s" % (os.path.basename(path), i + 1, e), file=sys.stderr)
        t = unicodedata.normalize("NFKC", t)
        t = re.sub(r"[ \t]+\n", "\n", t)
        pages.append(t.strip())
        if progress and (i + 1) % 50 == 0:
            progress(i + 1, len(r.pages))
    with open(cp, "w", encoding="utf-8") as f:
        json.dump({"size": st.st_size, "mtime": int(st.st_mtime), "pages": pages}, f, ensure_ascii=False)
    return pages


def load_pdf_chunks():
    chunks = []
    purposes = load_purposes()
    for path in pdf_files():
        fn = os.path.basename(path)
        for i, text in enumerate(extract_pdf(path, progress=lambda a, b, _f=fn: print("[pdf] %s %d/%d" % (_f, a, b), file=sys.stderr))):
            text = redact(text)
            if len(text) < MIN_PAGE_CHARS:
                continue
            head = "p%d" % (i + 1)
            for h, body in _hard_wrap(head, text, strategy=chunk_strategy("pdf")):
                body = redact(body)  # a character boundary can cut a redaction placeholder
                chunks.append({
                    "id": hashlib.md5(("pdf|%s|%s" % (fn, h)).encode()).hexdigest()[:12],
                    "source": "pdf", "file": fn, "heading": h, "text": body,
                    "md5": chunk_key(fn, h, body),
                    "purpose": chunk_purpose("pdf", fn, h, purposes.get(("pdf", fn), "knowledge"), purposes),
                })
    return chunks


def signature():
    """语料指纹:所有源目录里 md 的 (名, mtime, size)+ 开着的 PDF 源 —— 变了就该重建索引(热更用)。"""
    parts = ["chunker|%s|%s|%d|%d" % (CHUNKER_VERSION, CHUNK_STRATEGY, MAX_CHUNK, OVERLAP)]
    if os.path.isfile(CORPUS_POLICY):
        st = os.stat(CORPUS_POLICY)
        parts.append("policy|%s|%d|%d" % (CORPUS_POLICY, st.st_mtime_ns, st.st_size))
    for src in SOURCES:
        for fn, _path, st in _scan(src):
            parts.append("%s|%s|%d|%d" % (src["key"], fn, st.st_mtime_ns, st.st_size))
    if pdf_enabled():
        for p in pdf_files():
            st = os.stat(p)
            parts.append("pdf|%s|%d|%d" % (os.path.basename(p), int(st.st_mtime), st.st_size))
    return hashlib.md5("\n".join(parts).encode()).hexdigest()
MAX_CHUNK = 1500              # 字符;标题块超长再硬切
CHUNKER_VERSION = "paragraph-v1"
CHUNK_STRATEGY = os.environ.get("MR_CHUNK_STRATEGY", "paragraph")


def chunk_key(fn, heading, body):
    """chunk 的缓存键(分词缓存 / embedding 缓存都用它)。
    🔴 必须哈希「真正被索引的文本」= 文件名+标题+正文(见 mr_index._index_text),不能只哈希正文:
       只哈希正文时,改标题/改文件名不会触发重算,向量里编码的还是旧标题(2026-09-08 外部评审指出,当时暴露面 1 组:两篇笔记里同一段文字)。"""
    return hashlib.md5(("%s\n%s\n%s" % (fn, heading, body)).encode()).hexdigest()
OVERLAP = 150

# 凭据脱敏:值抹掉、键留着——搜"哪里有key"仍可命中,但值不进索引/不出工具
RE_CRED = re.compile(r"(?i)(password|passwd|pwd|secret|token|api[_-]?key|accesskey|密码|口令|密钥)([\"']?\s*[:=：]\s*)([^\s,;}`]+)")
RE_QUOTED_CRED = re.compile(r"(?i)(password|passwd|pwd|secret|token|api[_-]?key|accesskey|密码|口令|密钥)([\"']?\s*[:=：]\s*)([\"'`])(.*?)(\3)")
RE_INLINE_LOGIN = re.compile(r"`([A-Za-z0-9_.@-]{1,64}):([^`\s:<>]{6,})`")
RE_URL_LOGIN = re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://[^\s/:@]+:)([^\s/@]+)(@)")
RE_AUTH_HEADER = re.compile(r"(?i)(\b(?:Authorization|Proxy-Authorization)[\"']?\s*[:=]\s*[\"']?(?:Basic|Bearer)\s+)([^\s\"'`,;}]+)")
# 裸 key(表格/清单里没有 key= 前缀):sk- 供应商 key、腾讯云 AKID / AWS AKIA / 阿里 LTAI;2026-09-24 work 源扫出 7 个漏网
RE_BARE_KEY = re.compile(r"\b(sk-|AKID|AKIA|LTAI)[A-Za-z0-9_-]{12,}")


def redact(text):
    text = RE_BARE_KEY.sub(lambda m: m[1] + '<REDACTED>', text)
    text = RE_URL_LOGIN.sub(lambda m: m[1] + '<REDACTED>' + m[3], text)
    text = RE_AUTH_HEADER.sub(lambda m: m[1] + '<REDACTED>', text)
    def inline(match):
        value = match[2]
        nearby = text[max(0, match.start()-60):match.start()]
        credential_context = re.search(r"(?i)账号|账户|凭据|账密|认证|password|credential|auth", nearby)
        mixed_value = re.search(r'[A-Za-z]', value) and re.search(r'[0-9!$%^&*+=]', value)
        return '`' + match[1] + ':<REDACTED>`' if credential_context or mixed_value else match[0]
    text = RE_INLINE_LOGIN.sub(inline, text)
    text = RE_QUOTED_CRED.sub(lambda m: m[1] + m[2] + m[3] + '<REDACTED>' + m[3], text)
    def value(match):
        value = match[3]
        if value.startswith(('"', "'")):
            return match[0]  # already handled as a quoted value; retain balanced quotes
        return match[1] + match[2] + '<REDACTED>'
    return RE_CRED.sub(value, text)


def redact_values(value):
    """Preserve JSON structure while scrubbing nested diagnostic/model strings."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [redact_values(item) for item in value]
    if isinstance(value, dict):
        return {key: ('<REDACTED>' if RE_CRED.search(key + '=placeholder')
                     else redact_values(item)) for key, item in value.items()}
    return value


def _split_heading(text):
    """按 #/##/### 切块,标题作为块头;frontmatter/无标题前导归入'(开头)'块。"""
    blocks, cur_head, cur = [], "(开头)", []
    for ln in text.splitlines():
        if re.match(r"^#{1,3}\s", ln):
            if "".join(cur).strip():
                blocks.append((cur_head, "\n".join(cur).strip()))
            cur_head, cur = ln.lstrip("#").strip(), []
        else:
            cur.append(ln)
    if "".join(cur).strip():
        blocks.append((cur_head, "\n".join(cur).strip()))
    return blocks


def chunk_strategy(source):
    # Stock and PDF retain their established chunk boundaries. Ordinary-note
    # chunking is measured independently; changing their IDF still needs a stock check.
    return CHUNK_STRATEGY if source in PARAGRAPH_SOURCES else "character"


def _paragraph_spans(body):
    """Contiguous source spans, whole paragraphs first, then sentences/lines.

    A single overlong sentence is the only character-cut fallback. Offsets keep
    the original text recoverable without inserting or dropping characters.
    """
    ends = [m.end() for m in re.finditer(r"\n[ \t]*\n+", body)] + [len(body)]
    units, start = [], 0
    for end in ends:
        while end - start > MAX_CHUNK:
            segment = body[start:start + MAX_CHUNK]
            boundaries = [m.end() for m in re.finditer(r"[。！？!?；;](?:[\"'”’])?\s*|(?<=\.)\s+|\n", segment)]
            cut = start + (boundaries[-1] if boundaries else MAX_CHUNK)
            units.append((start, cut))
            start = cut
        if end > start:
            units.append((start, end))
        start = end
    spans, i = [], 0
    while i < len(units):
        j = i + 1
        while j < len(units) and units[j][1] - units[i][0] <= MAX_CHUNK:
            j += 1
        spans.append((units[i][0], units[j - 1][1]))
        if j == len(units):
            break
        # Whole trailing paragraph overlap; 150 is the target, at most 1/3 of
        # the chunk, and always enough space for the next unseen unit.
        budget = min(MAX_CHUNK // 3, MAX_CHUNK - (units[j][1] - units[j][0]))
        k = j
        while k > i + 1 and units[j - 1][1] - units[k - 1][0] <= budget:
            k -= 1
            if units[j - 1][1] - units[k][0] >= OVERLAP:
                break
        i = k
    return spans


def _hard_wrap(head, body, strategy=None):
    strategy = strategy or CHUNK_STRATEGY
    if strategy not in {"paragraph", "character"}:
        raise ValueError("Unknown chunk strategy: " + strategy)
    if len(body) <= MAX_CHUNK:
        return [(head, body)]
    if strategy == "paragraph":
        return [("%s (%d)" % (head, n), body[start:end])
                for n, (start, end) in enumerate(_paragraph_spans(body), 1)]
    out, i, n = [], 0, 1
    while i < len(body):
        out.append(("%s (%d)" % (head, n), body[i:i + MAX_CHUNK]))
        i += MAX_CHUNK - OVERLAP
        n += 1
    return out


def load_chunks():
    chunks = []
    purposes = load_purposes()
    for src in SOURCES:
        d = src["dir"]
        if not os.path.isdir(d):
            print("[corpus] 缺目录: %s" % d, file=sys.stderr)
            continue
        files = _scan(src)
        seen = None
        if src.get("recursive"):
            # 工作目录里同一仓拷了 9 份(supplier-health*):同一段落只留最新文件那份,否则前 5 名全是复本
            files.sort(key=lambda f: -f[2].st_mtime_ns)
            seen = set()
        for fn, path, _st in files:
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError as e:
                print("[corpus] 读失败 %s: %s" % (fn, e), file=sys.stderr)
                continue
            # Credential syntax is a content property, not a source property.
            text = redact(text)
            purpose = document_purpose(src["key"], fn, text, purposes)
            for head, body in (hb for h in _split_heading(text)
                               for hb in _hard_wrap(*h, strategy=chunk_strategy(src["key"]))):
                body = redact(body)
                if seen is not None:
                    dup = hashlib.md5(("%s\n%s" % (head, body)).encode()).hexdigest()
                    if dup in seen:
                        continue
                    seen.add(dup)
                chunks.append({
                    "id": hashlib.md5(("%s|%s|%s" % (src["key"], fn, head)).encode()).hexdigest()[:12],
                    "source": src["key"], "file": fn, "heading": head, "text": body,
                    "md5": chunk_key(fn, head, body),
                    "purpose": chunk_purpose(src["key"], fn, head, purpose, purposes),
                })
    if pdf_enabled():
        chunks.extend(load_pdf_chunks())
    return chunks


if __name__ == "__main__":
    cs = load_chunks()
    per = {}
    for c in cs:
        per[c["source"]] = per.get(c["source"], 0) + 1
    files = {(c["source"], c["file"]) for c in cs}
    print("chunks=%d files=%d 按源=%s" % (len(cs), len(files), per))
