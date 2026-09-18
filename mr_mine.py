# -*- coding: utf-8 -*-
"""从 Claude Code 会话 transcript 里挖金标候选(2026-09-16)。

为什么不只看 .cache/query-log.jsonl:全库 865 个会话 / 2,366 个用户轮里 memory_search/get 真调用 11 次,
而 agent 直接 Read/cat/grep 语料文件 477 次(按轮去重)——问题天天在问,只是没经过 RAG 的埋点。
自家埋点的沉默,既可能是「没问题」也可能是「没人用」,两者长得一样;transcript 才是那份真实日志。

隐式正标签 = 用户问了 X → 同一轮里 agent 真去读了语料文件 Y(Read/Grep/cat/sed),
和 `mr.py log` 的 search→get 同构。三类系统性假阳性,必须先掐掉:
  1. 流程性阅读 ≠ 答案性阅读:cc-teach-agent-principles.md 每轮都读(规矩要求),跟用户问什么无关 → PROCEDURAL 黑名单
  2. 一轮读 N 个文件 → 一题 N 标签,其中 N-1 个是干活的工具性阅读 → 默认只收单文件轮(--multi 放开)
  3. 挖出来的标签可能比 RAG 还差(「HK 普罗米修斯埋点」标成 config-repo,RAG 排第一的 hk-endpoint-metrics 才对)
     → 只当候选池,**不自动入集**;先用现在的 RAG 反查排难度,答得对的(≤3 名)扔掉,人工只核 miss / 4~10 名。
"""
import collections
import glob
import io
import json
import os
import re

import mr_corpus

ROOT = os.path.join(mr_corpus.HOME, r".claude\projects")
SOURCES = [   # 路径统一换成 / 再匹配,免得跟 Windows 路径的 JSON 转义打架
    (re.compile(r"C--Users-zihao/memory/([^/\"'\s]+\.md)"), "mem"),
    (re.compile(r"rh学习记忆/([^/\"'\s]+\.md)"), "case"),
    (re.compile(r"Agent学习计划/notes/([^/\"'\s]+\.md)"), "book"),
]
DENY = {"MEMORY.md", "MEMORY-backup-20260907.md", "*.md"}
# 因为「规矩要求每轮读」而被读的文件,不是因为它回答了问题
PROCEDURAL = {"cc-teach-agent-principles.md", "weekly-report.md", "claude-md-rules-details.md",
              "memory-rules.md", "interaction-preferences.md", "gerrit-workflow.md"}
SKIP_TOOLS = {"Write", "Edit", "NotebookEdit"}
WRITEY = re.compile(r">\s*[\"']?[^|\s]*\.md|Set-Content|Out-File|sed -i|Remove-Item|\brm\b|git add|git commit")
QLIKE = re.compile(r"吗|怎么|为什么|是不是|哪|什么|如何|多少|咋|\?|？|能不能|有没有")
JUNK = re.compile(r"^\[Image|^http|^[A-Za-z]:\\|^\d+$|^(go|ok|yes|continue|merge it)$", re.I)
QLEN = (6, 60)
RE_ID = re.compile(r"\b\d{15,}\b")        # 任务 id 之类的长数字不该进公开金标,先打码给人看


def _turn_text(rec):
    m = rec.get("message") or {}
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
            return None          # 工具回包不是人说的话
        return "\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return None


def scan(days=0):
    """逐会话走一遍:每个用户轮 → 这一轮 assistant 读了哪些语料文件。返回 (turns, stats)。
    turns = [{ts, q, files:[(src, fn)]}];stats 带 RAG 被用 vs 被绕过的计数。"""
    import datetime
    since = ""
    if days:
        since = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    turns, stats = [], collections.Counter()
    for f in glob.glob(os.path.join(ROOT, "**", "*.jsonl"), recursive=True):
        stats["sessions"] += 1
        cur = None
        try:
            fh = io.open(f, encoding="utf-8")
        except Exception:
            continue
        with fh:
            for ln in fh:
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                t = r.get("type")
                if t == "user":
                    if r.get("isMeta"):
                        continue
                    txt = _turn_text(r)
                    txt = (txt or "").strip()
                    if not txt or txt.startswith("<") or txt.startswith("Caveat:"):
                        continue
                    ts = (r.get("timestamp") or "")[:19]
                    cur = None if (since and ts < since) else {"ts": ts, "q": txt, "files": []}
                    if cur:
                        turns.append(cur)
                        stats["turns"] += 1
                elif t == "assistant" and cur is not None:
                    for b in ((r.get("message") or {}).get("content") or []):
                        if not isinstance(b, dict) or b.get("type") != "tool_use":
                            continue
                        name = b.get("name") or ""
                        if name.startswith("mcp__rh-memory__"):
                            stats["rag_calls"] += 1
                        if name in SKIP_TOOLS:
                            continue
                        blob = json.dumps(b.get("input"), ensure_ascii=False).replace("\\\\", "/").replace("\\", "/")
                        if WRITEY.search(blob):
                            continue
                        for rx, src in SOURCES:
                            for fn in rx.findall(blob):
                                if fn in DENY or (src, fn) in cur["files"]:
                                    continue
                                cur["files"].append((src, fn))
                                stats["direct_reads"] += 1
    return turns, stats


def candidates(turns, multi=False):
    """问句样 + 单文件轮(默认)+ 去流程性文件 → 候选 [{q, expect, src, ts, n_files}]。"""
    seen, out = set(), []
    for t in turns:
        q = t["q"].replace("\n", " ").strip()
        files = [(s, f) for s, f in t["files"] if f not in PROCEDURAL]
        if not files or (len(files) > 1 and not multi):
            continue
        if JUNK.search(q) or not (QLEN[0] <= len(q) <= QLEN[1]) or not QLIKE.search(q):
            continue
        for src, fn in files:
            k = (q, fn)
            if k in seen:
                continue
            seen.add(k)
            out.append({"q": q, "expect": [fn], "src": src, "ts": t["ts"], "n_files": len(files)})
    return out


def rank(cands, topk=10):
    """每条拿现在的 RAG 真查一遍:期望文件排第几。miss=前 topk 没有。"""
    import mr_tools
    mr_tools.LOG_ON = False    # 挖矿不是真实查询,别污染 query-log
    rows = []
    for c in cands:
        r = mr_tools.memory_search(c["q"], mode="hybrid", topk=topk)
        files = []
        for h in r["hits"]:
            if h["file"] not in files:
                files.append(h["file"])
        rk = (files.index(c["expect"][0]) + 1) if c["expect"][0] in files else None
        rows.append({**c, "rank": rk, "confidence": r.get("confidence"), "actual_top2": files[:2]})
    return rows


def _show_q(q):
    return RE_ID.sub(lambda m: m.group(0)[:4] + "…", mr_corpus.redact(q))


def main(days=0, multi=False, show_easy=False, out=None, unranked=False):
    turns, st = scan(days)
    print("扫了 %d 个会话 / %d 个用户轮%s" % (st["sessions"], st["turns"], ("(近 %d 天)" % days) if days else ""))
    print("RAG 被用 vs 被绕过:memory_search/get 调用 %d 次,直接读语料文件 %d 次" % (st["rag_calls"], st["direct_reads"]))
    cands = candidates(turns, multi=multi)
    print("问句样候选 %d 条(%s)" % (len(cands), "含多文件轮" if multi else "只收单文件轮,--multi 放开"))
    if not cands:
        return
    if unranked:
        # A holdout candidate must not be retrieved or selected by its score.
        # Labels remain unreviewed until checked against source documents.
        out = out or os.path.join(mr_corpus.BASE, ".cache", "mined-unranked.json")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with io.open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"ranked": False, "draft": True, "candidates": cands}, f, ensure_ascii=False, indent=2)
        print("未调用检索，未按分数筛题；%d 个待原文核验候选已保存:%s" % (len(cands), out))
        return
    rows = rank(cands)
    easy = [x for x in rows if x["rank"] and x["rank"] <= 3]
    hard = [x for x in rows if x["rank"] and x["rank"] > 3]
    miss = [x for x in rows if x["rank"] is None]
    print("期望文件 ≤3 名 %d(收进金标只会抬分,默认不显示)/ 4~%d 名 %d / 前 10 没有 %d" % (len(easy), 10, len(hard), len(miss)))

    def show(title, xs):
        if not xs:
            return
        print("\n%s —— 人工核对后挑进 golden/golden_queries.json(标签可能比 RAG 还差,别直接收):" % title)
        for x in xs:
            print("  #%-4s %-46s 期望 %-40s 实际top2=%s" % (x["rank"] or "-", _show_q(x["q"])[:46], x["expect"][0][:40],
                                                         [t[:24] for t in x["actual_top2"]]))
            print("        " + json.dumps({"q": _show_q(x["q"]), "expect": x["expect"]}, ensure_ascii=False))
    show("真·miss(最值钱:真实问过、现在答不出)", miss)
    show("排 4~10 名(难题候选)", sorted(hard, key=lambda x: -x["rank"]))
    if show_easy:
        show("≤3 名(已经答对)", easy)
    out = out or os.path.join(mr_corpus.BASE, ".cache", "mined-transcripts.json")   # .cache 已 gitignore:问句是用户原话
    os.makedirs(os.path.dirname(out), exist_ok=True)
    io.open(out, "w", encoding="utf-8", newline="\n").write(json.dumps(rows, ensure_ascii=False, indent=1))
    print("\n全部 %d 条(含名次)落盘:%s" % (len(rows), out))
