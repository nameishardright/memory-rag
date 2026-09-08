# -*- coding: utf-8 -*-
"""语料层:三源加载 → 按标题切块 → 凭据脱敏(只对 CC memory 源)。
设计决策与概念对应见 Desktop\\Agent学习计划\\notes\\第5-6周-毕业项目2.0-记忆RAG.md
"""
import hashlib
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
]
DENY = {"MEMORY.md"}          # 纯索引,内容都在单文件里,入库只会造重复命中

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
    for path in pdf_files():
        fn = os.path.basename(path)
        for i, text in enumerate(extract_pdf(path, progress=lambda a, b, _f=fn: print("[pdf] %s %d/%d" % (_f, a, b), file=sys.stderr))):
            if len(text) < MIN_PAGE_CHARS:
                continue
            head = "p%d" % (i + 1)
            for h, body in _hard_wrap(head, text):
                chunks.append({
                    "id": hashlib.md5(("pdf|%s|%s" % (fn, h)).encode()).hexdigest()[:12],
                    "source": "pdf", "file": fn, "heading": h, "text": body,
                    "md5": hashlib.md5(body.encode()).hexdigest(),
                })
    return chunks


def signature():
    """语料指纹:所有源目录里 md 的 (名, mtime, size)+ 开着的 PDF 源 —— 变了就该重建索引(热更用)。"""
    parts = []
    for src in SOURCES:
        d = src["dir"]
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".md") and fn not in DENY:
                try:
                    st = os.stat(os.path.join(d, fn))
                    parts.append("%s|%s|%d|%d" % (src["key"], fn, int(st.st_mtime), st.st_size))
                except OSError:
                    pass
    if pdf_enabled():
        for p in pdf_files():
            st = os.stat(p)
            parts.append("pdf|%s|%d|%d" % (os.path.basename(p), int(st.st_mtime), st.st_size))
    return hashlib.md5("\n".join(parts).encode()).hexdigest()
MAX_CHUNK = 1500              # 字符;标题块超长再硬切
OVERLAP = 150

# 凭据脱敏:值抹掉、键留着——搜"哪里有key"仍可命中,但值不进索引/不出工具
RE_CRED = re.compile(r"(?i)(password|passwd|pwd|secret|token|api[_-]?key|accesskey)(\s*[:=]\s*)(\S+)")


def redact(text):
    return RE_CRED.sub(lambda m: m.group(1) + m.group(2) + "<REDACTED>", text)


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


def _hard_wrap(head, body):
    if len(body) <= MAX_CHUNK:
        return [(head, body)]
    out, i, n = [], 0, 1
    while i < len(body):
        out.append(("%s (%d)" % (head, n), body[i:i + MAX_CHUNK]))
        i += MAX_CHUNK - OVERLAP
        n += 1
    return out


def load_chunks():
    chunks = []
    for src in SOURCES:
        d = src["dir"]
        if not os.path.isdir(d):
            print("[corpus] 缺目录: %s" % d, file=sys.stderr)
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".md") or fn in DENY:
                continue
            try:
                with open(os.path.join(d, fn), encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError as e:
                print("[corpus] 读失败 %s: %s" % (fn, e), file=sys.stderr)
                continue
            if src["redact"]:
                text = redact(text)
            for head, body in (hb for h in _split_heading(text) for hb in _hard_wrap(*h)):
                chunks.append({
                    "id": hashlib.md5(("%s|%s|%s" % (src["key"], fn, head)).encode()).hexdigest()[:12],
                    "source": src["key"], "file": fn, "heading": head, "text": body,
                    "md5": hashlib.md5(body.encode()).hexdigest(),
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
