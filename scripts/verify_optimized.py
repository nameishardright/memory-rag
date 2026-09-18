"""Calibrate the chosen gate once, then verify real retrieval on frozen sources.

Private output contains questions and evidence previews. No score/result cache is
used. The independent held-out set is deliberately never opened by this script.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
import mr_eval


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def calibrate(positive,negative,minimum_negative_low=8):
    """Conservative coarse grid; preserve the old thresholds when still valid.

    Never split two almost-identical scores just to get a perfect development
    score. This is regression calibration, not an independent accuracy estimate.
    """
    if not positive or not negative or not all(math.isfinite(x) and 0<=x<=1 for x in positive+negative):
        raise ValueError("Calibration needs nonempty positive and negative finite scores in [0,1]")
    grid=[]
    for hi in (.35,.4,.45,.5,.55,.6,.7,.8):
        for lo in (.03,.05,.075,.1,.15,.2,.3):
            if lo>=hi:continue
            row={"hi":hi,"lo":lo,"positive_low":sum(x<lo for x in positive),
                 "positive_high":sum(x>=hi for x in positive),"negative_high":sum(x>=hi for x in negative),
                 "negative_low":sum(x<lo for x in negative)}
            row["eligible"]=(row["positive_low"]==0 and row["negative_high"]==0 and row["negative_low"]>=minimum_negative_low)
            grid.append(row)
    feasible=[r for r in grid if r["eligible"]]
    if not feasible:raise RuntimeError("No conservative calibrated threshold pair; do not deploy this profile")
    selected=min(feasible,key=lambda r:(abs(r["hi"]-.5)+abs(r["lo"]-.1),-r["positive_high"],-r["negative_low"]))
    return {"selected":selected,"grid":grid,"positive_scores":positive,"negative_scores":negative,
            "method":"coarse grid; 0 positive low, 0 negative high, >=8 negative low; retain prior thresholds if feasible"}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("frozen","models","cache","evidence","baseline","output","integrity"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--original-golden-dir",type=Path,help="Original working copy, for byte-exact SHA verification across Git newline conversion")
    ap.add_argument("--calibration-from",type=Path,help="Reuse an already fixed calibration when only non-note routing changed")
    args=ap.parse_args()
    os.environ.update(MR_LOG="0",MR_PDF="0",MR_RERANK="bge-reranker-base",MR_GATE="note",MR_RERANK_MODE="gate",
                      MR_EMBED_MODEL="bge-small-zh-v1.5",MR_EMBED_MODEL_PDF="bge-m3",
                      MR_MODELS_DIR=str(args.models),MR_CACHE_DIR=str(args.cache),MR_CORPUS_POLICY=str(args.frozen/"corpus-policy.json"),
                      HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
    import mr_corpus,mr_tools,mr_index
    for src in mr_corpus.SOURCES:src["dir"]=str(args.frozen/src["key"])
    mr_corpus.PDF_CACHE=str(args.models/"pdf-text")
    chunks=mr_corpus.load_chunks()
    mr_corpus.load_chunks=lambda:chunks
    mr_corpus.signature=lambda:"frozen-final"
    golden={name:json.loads(path.read_text(encoding="utf-8")) for name,path in {
        "notes":CODE/"golden/golden_queries.json","evidence":args.evidence,"negative":CODE/"golden/golden_negative.json",
        "original_evidence":CODE/"golden/golden_evidence.json","stock":CODE/"golden/golden_stock.json",
        "pdf":CODE/"golden/golden_pdf.json"}.items()}
    mr_eval.require_clean_corpus(chunks,list(golden.values()))
    report={"at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"code_sha256":{p.name:digest(p) for p in CODE.glob("*.py")},
              "evidence_sha256":digest(args.evidence),"runs":{},"checks":{},"profile":mr_index.reranker_spec("bge-reranker-base")}
    report["evaluated_golden_sha256"]={p.name:digest(p) for p in (CODE/"golden").glob("*.json")}
    def save():args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    def run(name,dataset,corpus=chunks,search=mr_tools.memory_search,model=None,rerank="bge-reranker-base",mode="hybrid"):
        r=mr_eval.evaluate(dataset,search,mode=mode,model=model,rerank=rerank,chunks=corpus)
        report["runs"][name]=r
        if r["summary"]["fallbacks"]:raise RuntimeError("Unexpected fallback: "+name)
        print(json.dumps({"suite":name,**r["summary"]},ensure_ascii=False),flush=True)
        save()
        return r
    # The sole calibration happens after chunk/parameter selection.
    if args.calibration_from:
        prior=json.loads(args.calibration_from.read_text(encoding="utf-8"))
        keys=("max_length","gate_candidates","max_windows","fallback_candidates","hi","lo")
        if (prior["evidence_sha256"]!=report["evidence_sha256"] or
                prior["code_sha256"]["mr_corpus.py"]!=report["code_sha256"]["mr_corpus.py"] or
                any(prior["profile"][k]!=report["profile"][k] for k in keys)):
            raise RuntimeError("Calibration profile, chunker or evidence labels changed")
        report["calibration"]=prior["calibration"]
        report["calibration_artifact_sha256"]=digest(args.calibration_from)
    else:
        cp=run("calibration_positive",golden["evidence"])
        cn=run("calibration_negative",golden["negative"])
        report["calibration"]=calibrate([r["rerank"]["top"] for r in cp["rows"]],
                                       [r["rerank"]["top"] for r in cn["rows"]])
    selected=report["calibration"]["selected"]
    print(json.dumps({"calibrated":selected}),flush=True)
    configured=mr_index.reranker_spec("bge-reranker-base")
    if (selected["hi"],selected["lo"])!=(configured["hi"],configured["lo"]):
        save()
        raise RuntimeError("Configured thresholds differ from calibration; apply the recorded pair before verification")
    for name in ("evidence","negative","original_evidence","stock"):
        run(name+"_gate",golden[name])
    for mode in ("bm25","vec","hybrid"):
        run("notes_"+mode,golden["notes"],mode=mode,rerank=False)
        run("stock_"+mode,golden["stock"],mode=mode,rerank=False)
    # Reuse the model/cache, not query responses; check unchanged page boundaries.
    pdf_chunks=mr_corpus.load_pdf_chunks()
    mr_eval.require_clean_corpus(pdf_chunks,list(golden.values()))
    from scripts.verify_retrieval import load_stack
    before=load_stack(args.baseline)
    before.corpus.PDF_CACHE=mr_corpus.PDF_CACHE
    old_pdf=before.corpus.load_pdf_chunks()
    report["checks"]["pdf_chunk_boundaries_preserved"]=[c["md5"] for c in pdf_chunks]==[c["md5"] for c in old_pdf]
    mr_tools._STATE.clear()
    mr_corpus.load_chunks=lambda:pdf_chunks
    mr_corpus.signature=lambda:"frozen-pdf"
    run("pdf_m3",golden["pdf"],pdf_chunks,model="bge-m3",rerank=False)
    run("pdf_gate",golden["pdf"],pdf_chunks,model="bge-m3")
    before.corpus.load_chunks=lambda:old_pdf
    before.corpus.signature=lambda:"frozen-old-pdf"
    before.tools.LOG_ON=False
    # Share the source-identical PDF vector object, without sharing the gate.
    before.index.VecIndex=lambda *a,**kw:mr_tools._STATE["vec"]
    run("baseline_pdf_gate",golden["pdf"],old_pdf,search=before.tools.memory_search,model="bge-m3")
    summary=lambda name:report["runs"][name]["summary"]
    c=report["checks"]
    c["notes_hit5_preserved"]=summary("notes_hybrid")["returned_hit_at5"]==23
    c["notes_hybrid_mrr_improved"]=summary("notes_hybrid")["returned_mrr_at10"]>.773188
    c["strict_evidence_not_reduced"]=summary("evidence_gate")["evidence_hit_at5"]>=9
    c["old_four_passages_preserved"]=summary("original_evidence_gate")["evidence_hit_at5"]==4
    c["positive_no_new_low"]=summary("evidence_gate")["positive_confidence"].get("low",0)==0
    c["negative_no_high"]=summary("negative_gate")["negative_high"]==0
    c["negative_low_not_reduced"]=summary("negative_gate")["negative_confidence"].get("low",0)>=8
    c["stock_hit5_preserved"]=summary("stock_hybrid")["returned_hit_at5"]==11
    c["stock_mrr_preserved"]=summary("stock_hybrid")["returned_mrr_at10"]>=.756944
    c["pdf_hit5_preserved"]=summary("pdf_m3")["returned_hit_at5"]==12
    c["pdf_gate_hit5_preserved"]=summary("pdf_gate")["returned_hit_at5"]==summary("baseline_pdf_gate")["returned_hit_at5"]
    c["pdf_gate_no_added_low"]=summary("pdf_gate")["positive_confidence"].get("low",0)<=summary("baseline_pdf_gate")["positive_confidence"].get("low",0)
    protected=json.loads(args.integrity.read_text(encoding="utf-8"))["golden_sha256"]
    source=args.original_golden_dir or CODE/"golden"
    report["original_goldens_unchanged"]={name:digest(source/name)==expected for name,expected in protected.items()}
    report["worktree_goldens_same_json"]={name:json.loads((source/name).read_bytes())==json.loads((CODE/"golden"/name).read_bytes())
                                         for name in protected}
    report["worktree_goldens_only_newline_differences"]={name:(source/name).read_bytes().replace(b"\r\n",b"\n") ==
                                                           (CODE/"golden"/name).read_bytes().replace(b"\r\n",b"\n") for name in protected}
    c["original_goldens_unchanged"]=all(report["original_goldens_unchanged"].values()) and all(report["worktree_goldens_same_json"].values())
    c["evaluated_goldens_unchanged_during_run"]=all(digest(CODE/"golden"/name)==expected for name,expected in report["evaluated_golden_sha256"].items())
    c["stock_still_draft"]=golden["stock"].get("draft") is True
    c["code_unchanged_during_run"]=all(digest(CODE/name)==value for name,value in report["code_sha256"].items())
    report["passed"]=all(c.values())
    report["completed_at"]=time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    print(json.dumps({"passed":report["passed"],"checks":c}),flush=True)
    return 0 if report["passed"] else 1


if __name__=="__main__":raise SystemExit(main())
