"""Seal a locally reviewed holdout, then consume its first evaluation once.

This enforces result blindness, not independent human authorship. A failed or
interrupted attempted evaluation is consumed too: reruns belong to regression.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import mr_eval


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(dataset, manifest, references):
    dataset, manifest = Path(dataset), Path(manifest)
    golden = json.loads(dataset.read_text(encoding="utf-8"))
    mr_eval.validate_labels(golden)
    questions, groups, files = set(), set(), set()
    for reference in references:
        other = json.loads(Path(reference).read_text(encoding="utf-8"))
        for q in other["queries"]:
            questions.add(q["q"])
            groups.add(q.get("case_group"))
            files.update(q["expect"])
    seen_groups = set()
    for q in golden["queries"]:
        if q.get("split", golden.get("split")) != "blind" or not q.get("case_group"):
            raise ValueError("Every held-out question needs blind split and a reviewed event group")
        if q["q"] in questions or q["case_group"] in groups or set(q["expect"]) & files:
            raise ValueError("Held-out question/group/source overlaps development references")
        if q["expect"] and not (q.get("evidence") or q.get("evidence_groups")):
            raise ValueError("Positive held-out questions need source-reviewed passage facts")
        seen_groups.add(q["case_group"])
    if not golden["queries"]:
        raise ValueError("Empty holdout")
    record = {"sealed_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"dataset_sha256":digest(dataset),
              "questions":len(golden["queries"]),"case_groups":len(seen_groups),
              "references_sha256":{str(p):digest(p) for p in references},
              "authorship":"same agent source review; not an independent human blind audit"}
    with manifest.open("x",encoding="utf-8") as f:
        json.dump(record,f,ensure_ascii=False,indent=2)
    return record


def evaluate_once(dataset, manifest, output, search, *, chunks, rerank="bge-reranker-base",
                  baseline_search=None, baseline_chunks=None):
    dataset, manifest, output = Path(dataset), Path(manifest), Path(output)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if digest(dataset) != record["dataset_sha256"]:
        raise ValueError("Sealed dataset changed")
    for path, expected in record["references_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Development reference changed since seal")
    golden = json.loads(dataset.read_text(encoding="utf-8"))
    mr_eval.require_clean_corpus(chunks, [golden])
    if baseline_search is not None:
        mr_eval.require_clean_corpus(baseline_chunks,[golden])
    # Exclusive creation survives process crashes and prevents concurrent reuse.
    used = manifest.with_suffix(".consumed.json")
    with used.open("x",encoding="utf-8") as f:
        json.dump({"attempt_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "dataset_sha256":record["dataset_sha256"],"subsequent_use":"regression only"},f)
    result = mr_eval.evaluate(golden,search,rerank=rerank,chunks=chunks)
    if baseline_search is not None:
        # Both immutable configurations belong to this single consumed trial;
        # neither result is released to the caller until both have completed.
        result["baseline"] = mr_eval.evaluate(golden,baseline_search,rerank=rerank,chunks=baseline_chunks)
        result["trial_kind"] = "one paired comparison of two fixed configurations"
    result["seal"] = record
    result["corpus_sha256"] = hashlib.sha256(json.dumps(
        [(c["source"],c["md5"],c.get("purpose")) for c in chunks]).encode()).hexdigest()
    result["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",type=Path,required=True)
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--freeze",action="store_true")
    ap.add_argument("--reference",type=Path,action="append",default=[])
    ap.add_argument("--frozen",type=Path)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--baseline",type=Path,help="Optional fixed pre-change code, evaluated in the same single trial")
    args = ap.parse_args()
    if args.freeze:
        print(json.dumps(seal(args.dataset,args.manifest,args.reference),ensure_ascii=False))
        return
    if not args.output or not args.frozen:
        ap.error("An evaluation requires --output and --frozen")
    import mr_corpus, mr_tools
    for source in mr_corpus.SOURCES:
        source["dir"] = str(args.frozen / source["key"])
    mr_corpus.CORPUS_POLICY = str(args.frozen / "corpus-policy.json")
    mr_tools.LOG_ON = False
    chunks = mr_corpus.load_chunks()
    mr_corpus.load_chunks = lambda **_: chunks
    mr_corpus.signature = lambda **_: "sealed-corpus"
    baseline_search,baseline_chunks = None,None
    if args.baseline:
        from scripts.verify_retrieval import load_stack
        before = load_stack(args.baseline)
        for source in before.corpus.SOURCES:
            source["dir"] = str(args.frozen/source["key"])
        before.corpus.CORPUS_POLICY = str(args.frozen/"corpus-policy.json")
        before.tools.LOG_ON = False
        baseline_chunks = before.corpus.load_chunks()
        before.corpus.load_chunks = lambda **_: baseline_chunks
        before.corpus.signature = lambda **_: "sealed-baseline-corpus"
        baseline_search = before.tools.memory_search
    result = evaluate_once(args.dataset,args.manifest,args.output,mr_tools.memory_search,chunks=chunks,
                           baseline_search=baseline_search,baseline_chunks=baseline_chunks)
    result["code_sha256"]={p.name:digest(p) for p in CODE.glob("*.py")}
    if args.baseline:
        result["baseline_code_sha256"]={p.name:digest(p) for p in args.baseline.glob("*.py")}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result["summary"],ensure_ascii=False))
    if result.get("baseline"):
        print(json.dumps({"baseline":result["baseline"]["summary"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
