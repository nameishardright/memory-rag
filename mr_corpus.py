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
    return chunks


if __name__ == "__main__":
    cs = load_chunks()
    per = {}
    for c in cs:
        per[c["source"]] = per.get(c["source"], 0) + 1
    files = {(c["source"], c["file"]) for c in cs}
    print("chunks=%d files=%d 按源=%s" % (len(cs), len(files), per))
