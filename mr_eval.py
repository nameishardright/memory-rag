"""Retrieval evaluation over the exact returned hits, with optional passage labels.

This module is consumed by evaluation only. Runtime retrieval never reads goldens.
"""
import collections
import json
from pathlib import Path
import re
import time
import unicodedata


class CorpusContaminationError(ValueError):
    def __init__(self, audit):
        self.audit = audit
        locations = ["  %s/%s :: %s" % (f["source"], f["file"], f["heading"]) for f in audit["findings"][:10]]
        super().__init__("CORPUS_CONTAMINATION_FAIL: %d 个未标 evaluation 的块含金标原问句；先复核用途再评测。\n%s" % (
            len(audit["findings"]), "\n".join(locations)))


def load_goldens(directory, extra=None):
    """Read evaluation-owned files only; retrieval never calls this function."""
    paths = {p.resolve() for p in Path(directory).glob("*.json")}
    if extra is not None:
        paths.add(Path(extra).resolve())
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(paths)]


def require_clean_corpus(chunks, goldens):
    """Fail on literal golden questions in any unmarked active corpus chunk.

    All sources are inspected before query/source filtering. This detects known
    question strings, not paraphrases, and never auto-labels or deletes content.
    """
    if chunks is None:
        raise ValueError("A complete corpus snapshot is required for the contamination audit")
    chunks = list(chunks)
    goldens = [goldens] if isinstance(goldens, dict) else goldens
    queries = set()
    for golden in goldens:
        for query in golden["queries"]:
            q = query["q"]
            if not isinstance(q, str) or not q.strip():
                raise ValueError("Golden questions must be non-empty strings")
            queries.add(q)
    if not queries:
        raise ValueError("No golden questions available for the corpus audit")
    findings, labeled = [], 0
    for chunk in chunks:
        if chunk.get("purpose") == "evaluation":
            labeled += 1
            continue
        indexed_text = "\n".join(chunk.get(key, "") for key in ("file", "heading", "text"))
        matches = sorted(q for q in queries if q in indexed_text)
        if matches:
            findings.append({key: chunk.get(key) for key in ("source", "file", "heading", "id")} | {"queries": matches})
    audit = {"ok": not findings, "scanned_chunks": len(chunks), "question_count": len(queries),
             "evaluation_chunks": labeled, "findings": findings}
    if findings:
        raise CorpusContaminationError(audit)
    return audit


def _normal(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def matches_passage(hit, label):
    """A passage must match its location AND every required fact in its preview."""
    for key in ("source", "file", "heading"):
        if key in label and hit.get(key) != label[key]:
            return False
    if label.get("heading_contains") and label["heading_contains"] not in hit.get("heading", ""):
        return False
    text = _normal(hit.get("snippet", ""))
    for alternatives in label.get("required_text", []):
        alternatives = [alternatives] if isinstance(alternatives, str) else alternatives
        if not any(_normal(term) in text for term in alternatives):
            return False
    return True


def _expected_file(hit, query, source=None):
    if hit["file"] not in query["expect"]:
        return False
    if source and hit["source"] != source:
        return False
    if query.get("expect_pages"):
        match = re.match(r"p(\d+)", hit.get("heading", ""))
        return bool(match) and query["expect_pages"][0] <= int(match[1]) <= query["expect_pages"][1]
    return True


def grade_hits(query, hits, source=None):
    """Keep returned-item rank separate from the historically deduplicated rank."""
    source = query.get("source", source)
    seen, documents = set(), []
    for hit in hits:
        key = (hit["source"], hit["file"])
        if query.get("expect_pages"):
            key += (re.sub(r" \(\d+\)$", "", hit.get("heading", "")),)
        if key not in seen:
            seen.add(key)
            documents.append(hit)
    return {
        "returned_rank": next((i + 1 for i, h in enumerate(hits) if _expected_file(h, query, source)), None),
        "document_rank": next((i + 1 for i, h in enumerate(documents) if _expected_file(h, query, source)), None),
        "evidence_rank": next((i + 1 for i, h in enumerate(hits)
                               if any(matches_passage(h, label) for label in query.get("evidence", []))), None),
        "hard_negative_ranks": [i + 1 for i, h in enumerate(hits)
                                if any(matches_passage(h, label) for label in query.get("hard_negatives", []))],
    }


def validate_labels(golden):
    groups = {}
    for q in golden["queries"]:
        if "evidence" in q and q["expect"]:
            if not q["evidence"]:
                raise ValueError("A positive passage label needs at least one acceptable passage")
            for label in q["evidence"]:
                if (label.get("file") not in q["expect"] or not label.get("source") or
                        not (label.get("heading") or label.get("heading_contains")) or not label.get("required_text")):
                    raise ValueError("Passage labels need source, expected file, heading and required_text")
                for terms in label["required_text"]:
                    terms = [terms] if isinstance(terms, str) else terms
                    if not terms or any(not isinstance(t, str) or not t.strip() for t in terms):
                        raise ValueError("Required facts cannot contain empty alternatives")
        group = q.get("case_group")
        split = q.get("split", golden.get("split", "regression"))
        if group:
            if group in groups and groups[group] != split:
                raise ValueError("Case group crosses evaluation splits: " + group)
            groups[group] = split


def evaluate(golden, search, mode="hybrid", model=None, rerank=None, *, chunks=None):
    validate_labels(golden)
    audit = require_clean_corpus(chunks, [golden])
    rows = []
    for query in golden["queries"]:
        kwargs = dict(mode=mode, topk=10, source=query.get("source", golden.get("source")), model=model, rerank=rerank)
        if query.get("include_evaluation", golden.get("include_evaluation", False)):
            kwargs["include_evaluation"] = True
        start = time.perf_counter()
        result = search(query["q"], **kwargs)
        if not result.get("ok"):
            raise RuntimeError("Search failed during evaluation: " + str(result.get("error")))
        grade = grade_hits(query, result["hits"], kwargs["source"])
        rows.append({"q": query["q"], "id": query.get("id"), "positive": bool(query["expect"]),
                     "passage_labeled": bool(query.get("evidence")), **grade,
                     "confidence": result.get("confidence", "high"), "mode": result["mode"],
                     "gate": result.get("gate"), "rerank": result.get("rerank"),
                     "hits": result["hits"], "ms": (time.perf_counter() - start) * 1000})
    positives = [r for r in rows if r["positive"]]
    negatives = [r for r in rows if not r["positive"]]
    evidence = [r for r in positives if r["passage_labeled"]]

    def count(population, key, k):
        return sum(r[key] is not None and r[key] <= k for r in population)

    def mrr(population, key):
        return round(sum(1 / r[key] for r in population if r[key]) / len(population), 6) if population else None

    summary = {
        "total": len(rows), "positives": len(positives), "negatives": len(negatives),
        "returned_hit_at5": count(positives, "returned_rank", 5),
        "returned_mrr_at10": mrr(positives, "returned_rank"),
        "document_hit_at5": count(positives, "document_rank", 5),
        "document_mrr_at10": mrr(positives, "document_rank"),
        "evidence_labeled": len(evidence), "evidence_hit_at1": count(evidence, "evidence_rank", 1),
        "evidence_hit_at5": count(evidence, "evidence_rank", 5), "evidence_mrr_at10": mrr(evidence, "evidence_rank"),
        "positive_confidence": dict(collections.Counter(r["confidence"] for r in positives)),
        "negative_confidence": dict(collections.Counter(r["confidence"] for r in negatives)),
        "negative_high": sum(r["confidence"] == "high" for r in negatives),
        "hard_negative_at1": sum(1 in r["hard_negative_ranks"] for r in rows),
        "fallbacks": sum(r["mode"] != mode for r in rows),
        "mean_ms": round(sum(r["ms"] for r in rows) / len(rows), 2) if rows else None,
    }
    return {"summary": summary, "rows": rows, "corpus_audit": audit}
