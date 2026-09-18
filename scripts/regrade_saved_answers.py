"""Correct PDF page coverage on immutable saved outputs, without any model call.

Keep raw blind results unchanged. This is an explicitly post-hoc scorer repair,
not another blind trial or permission to tune labels to observed answers.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import time

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
import mr_eval
import mr_answer_eval as ae
from scripts.eval_answers import retrieval_summary


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def regrade(original, queries):
    if not original.get('harness_complete') or original.get('sanitized_after_pilot'):
        raise ValueError('Only completed, unchanged final trials can be regraded')
    result=copy.deepcopy(original)
    result['raw_retrieval_summary']=copy.deepcopy(original['retrieval_summary'])
    result['raw_answer_summary']=copy.deepcopy(original.get('answer_summary'))
    row_by_id={r['id']:r for r in result['retrieval']}
    if set(queries)!=set(row_by_id):
        raise ValueError('Query set differs from the saved trial')
    for row in result['retrieval']:
        contexts=row['contexts']['passage']['contexts']
        old_payload=ae.digest(ae.answer_payload(queries[row['id']],contexts))
        for c in contexts:
            if c['source']=='pdf' and re.fullmatch(r'p\d+',c['heading']):
                c['evidence_scope']='pdf_page'
        if ae.digest(ae.answer_payload(queries[row['id']],contexts))!=old_payload:
            raise ValueError('Scorer repair changed model input')
        row['contexts']['passage']['context_sha256']=ae.digest(contexts)
        row['get_grade']=mr_eval.grade_hits(queries[row['id']],[dict(c,snippet=c['text']) for c in contexts])
    for policy,answers in result['answers'].items():
        for answer in answers:
            q=queries[answer['id']]
            contexts=row_by_id[answer['id']]['contexts'][policy]['contexts']
            if answer.get('input_sha256') and ae.digest(ae.answer_payload(q,contexts))!=answer['input_sha256']:
                raise ValueError('Saved input hash differs')
            if answer.get('answer'):
                answer['grade']=ae.grade_answer(q,answer['answer'],contexts)
    result['retrieval_summary']=retrieval_summary(result['retrieval'])
    result['retrieval_by_source']={s:retrieval_summary([r for r in result['retrieval'] if r['source']==s])
                                   for s in sorted({r['source'] for r in result['retrieval']})}
    if result.get('answer_summary') is not None:
        result['answer_summary']={p:ae.summarize_answers(rows) for p,rows in result['answers'].items()}
    for policy in result['answers']:
        for before,after in zip(original['answers'][policy],result['answers'][policy]):
            if {k:v for k,v in before.items() if k!='grade'}!={k:v for k,v in after.items() if k!='grade'}:
                raise ValueError('A recorded model output changed')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--dataset',type=Path,action='append',required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists() or args.output.resolve()==args.input.resolve():
        raise FileExistsError('Use a new output file; keep the original immutable')
    original=json.loads(args.input.read_text(encoding='utf-8'))
    questions={}
    for path in args.dataset:
        expected=original['inputs']['files_sha256'].get(str(path.resolve()))
        if not expected or sha(path)!=expected:
            raise ValueError('Original dataset bytes changed or not part of this trial')
        golden=json.loads(path.read_text(encoding='utf-8'))
        for i,q in enumerate(golden['queries'],1):
            q=dict(q);q.setdefault('id',path.stem+'-'+str(i));q.setdefault('source',golden.get('source'))
            if q['id'] in questions:raise ValueError('Duplicate question id')
            questions[q['id']]=q
    result=regrade(original,questions)
    result['posthoc_scoring_correction']={
        'at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'rule':'A retrieved full PDF page covers its labelled wrapped chunks on that same page; previews remain exact-chunk checks.',
        'raw_result_sha256':sha(args.input),'raw_result_path':str(args.input.resolve()),
        'model_inputs_unchanged':True,'model_outputs_unchanged':True,'labels_unchanged':True,
        'additional_searches':0,'additional_generations':0,
        'scorer_sha256':{str(p.relative_to(CODE)):sha(p) for p in [CODE/'mr_eval.py',CODE/'mr_answer_eval.py',Path(__file__).resolve()]}}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'retrieval':result['retrieval_summary'],'answers':result.get('answer_summary')},ensure_ascii=False))


if __name__=='__main__':
    main()
