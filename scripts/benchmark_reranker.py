"""Uncached, interleaved CPU timing of gate candidates on a frozen corpus.

These old thresholds are provisional during configuration selection. Calibrate
the selected tuple, then run the full quality and fresh MCP checks separately.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frozen",type=Path,required=True)
    ap.add_argument("--models",type=Path,required=True)
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--repeats",type=int,default=2)
    args=ap.parse_args()
    os.environ.update(MR_LOG="0",MR_PDF="0",MR_RERANK_MODE="gate",MR_GATE="note",MR_CHUNK_STRATEGY="paragraph",
                      MR_MODELS_DIR=str(args.models),MR_CACHE_DIR=str(args.cache),
                      MR_CORPUS_POLICY=str(args.frozen/"corpus-policy.json"),
                      HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
    import mr_corpus,mr_tools,mr_index
    for src in mr_corpus.SOURCES:
        src["dir"]=str(args.frozen/src["key"])
    chunks=mr_corpus.load_chunks()
    mr_corpus.load_chunks=lambda:chunks
    mr_corpus.signature=lambda:"frozen-latency"
    positives=json.loads((CODE/"golden/golden_queries.json").read_text(encoding="utf-8"))["queries"]
    negatives=json.loads((CODE/"golden/golden_negative.json").read_text(encoding="utf-8"))["queries"]
    cases=[("p%02d"%(i+1),positives[i]["q"]) for i in (0,1,3,7,8,13,18,22)]
    cases += [("n%02d"%(i+1),negatives[i]["q"]) for i in (0,3,5,9)]
    profiles=[(512,3,2),(256,5,2),(256,3,2)]
    mr_tools._ensure("hybrid")
    rk=mr_index.Reranker(mr_index.reranker_spec("bge-reranker-base"))
    if not rk.available:raise RuntimeError(rk.err)
    encoder=rk._get()
    mr_tools._get_reranker=lambda *_,**__:rk
    for length,g,w in profiles:
        encoder.max_seq_length=length
        mr_tools.RERANK_G,mr_tools.RERANK_MAXWIN=g,w
        rk.spec.update(max_length=length,gate_candidates=g,max_windows=w,fallback_candidates=g)
        mr_tools.memory_search(cases[0][1],rerank="bge-reranker-base")
    import torch
    result={"at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"uncached":True,
            "torch":torch.__version__,"threads":torch.get_num_threads(),"device":str(encoder.model.device),
            "distinct_queries":len(cases),"samples":[],"summaries":{}}
    def save():
        for profile in profiles:
            name="l%d-g%d-w%d"%profile
            rows=[x for x in result["samples"] if x["profile"]==name]
            if rows:
                values=sorted(r["total_ms"] for r in rows)
                result["summaries"][name]={"n":len(values),"p50_ms":statistics.median(values),
                     "empirical_p95_ms":values[math.ceil(.95*len(values))-1],
                     "mean_pairs":statistics.mean(r["pairs"] for r in rows)}
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    for repeat in range(args.repeats):
        for n,(ident,query) in enumerate(cases):
            offset=(n+repeat)%len(profiles)
            for length,g,w in profiles[offset:]+profiles[:offset]:
                encoder.max_seq_length=length
                if encoder.max_seq_length!=length:raise RuntimeError("Truncation override failed")
                mr_tools.RERANK_G,mr_tools.RERANK_MAXWIN=g,w
                rk.spec.update(max_length=length,gate_candidates=g,max_windows=w,fallback_candidates=g)
                started=time.perf_counter()
                found=mr_tools.memory_search(query,rerank="bge-reranker-base")
                total=(time.perf_counter()-started)*1000
                if found["mode"]!="hybrid" or not found.get("rerank"):raise RuntimeError("Missing expected model path")
                rr=found["rerank"]
                result["samples"].append({"profile":"l%d-g%d-w%d"%(length,g,w),"id":ident,"repeat":repeat+1,
                    "total_ms":round(total,2),"rerank_ms":rr["ms"],"pairs":rr["pairs"],
                    "confidence":found["confidence"],"corpus_sha256":found.get("corpus_sha256")})
                save()
        print(json.dumps({"round":repeat+1,"summaries":result["summaries"]}),flush=True)
    result["completed"]=True
    save()


if __name__=="__main__":main()
