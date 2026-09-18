"""Publish only aggregate counts; raw questions, labels and answers stay private."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import mr_answer_eval as ae


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def review_summary(run, review):
    expected = {(p,r['id']) for p,rows in run['answers'].items() for r in rows}
    rows = review['rows']
    actual = {(r['policy'],r['id']) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError('Semantic review must cover every answer exactly once')
    for row in rows:
        for key in ('answer_correct','citations_support','scope_correct','refusal_correct'):
            if type(row.get(key)) is not bool:
                raise ValueError('Semantic judgments must be explicit booleans')
        if not row.get('rationale'):
            raise ValueError('Semantic judgments need source-based rationale')
    def passed(row):
        return all(row[k] for k in ('answer_correct','citations_support','scope_correct','refusal_correct'))
    output = {}
    for policy in run['answers']:
        selected = [r for r in rows if r['policy']==policy]
        groups = collections.defaultdict(list)
        for answer in run['answers'][policy]:
            judgment = next(r for r in selected if r['id']==answer['id'])
            groups[answer['case_group']].append(passed(judgment))
        output[policy] = {'reviewed':len(selected),'semantic_pass':sum(passed(r) for r in selected),
                          'citation_support_pass':sum(r['citations_support'] for r in selected),
                          'scope_pass':sum(r['scope_correct'] for r in selected),
                          'groups':len(groups),'all_questions_in_group_pass':sum(all(v) for v in groups.values()),
                          'method':review['method']}
    return output


def git_state(repo):
    def git(*args):
        return subprocess.check_output(['git','-C',str(repo),*args],text=True,encoding='utf-8').strip()
    state={'head':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),
           'worktree_clean':not bool(git('status','--porcelain'))}
    return state


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--development',type=Path,required=True)
    ap.add_argument('--blind',type=Path,required=True)
    ap.add_argument('--regression',type=Path,required=True)
    ap.add_argument('--development-review',type=Path,required=True)
    ap.add_argument('--blind-review',type=Path,required=True)
    ap.add_argument('--unit',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    report={'data':{},'sha256':{},'limitations':[
        'Constructed source-grounded questions, not a random sample of user traffic.',
        'Same supervising agent authored labels and reviewed semantics; no independent human audit.',
        '36 questions share 24 event groups; question counts are not independent sample counts.',
        'Strict contract metrics and semantic review are separate; neither is a universal correctness guarantee.',
        'Fixed search/get/answer workflow, not arbitrary host-agent tool selection.',
        'Existing stock dataset remains draft; source fidelity does not verify the underlying financial facts.',
        'Cold index construction and model-answer generation are outside the old warm retrieval timing comparison.'
    ]}
    for name,path,review_path in [('development',args.development,args.development_review),
                                  ('blind',args.blind,args.blind_review)]:
        data,review=load(path),load(review_path)
        if not data.get('harness_complete') or not data.get('inputs_unchanged'):
            raise ValueError('Evaluation did not finish with fixed inputs: '+name)
        if review.get('result_sha256')!=sha(path):
            raise ValueError('Semantic review belongs to a different result file')
        report['data'][name]={'retrieval':data['retrieval_summary'],'by_source':data['retrieval_by_source'],
                              'strict_answer_contract':data['answer_summary'],
                              'semantic_review':review_summary(data,review),
                              'corpus_sha256':data['corpus_sha256'],
                              'posthoc_scoring_correction':data.get('posthoc_scoring_correction'),
                              'raw_retrieval_summary':data.get('raw_retrieval_summary'),
                              'raw_answer_summary':data.get('raw_answer_summary'),
                              'model_ids':sorted({model for rows in data['answers'].values() for r in rows
                                                   for model in r.get('model_usage',{})})}
        report['sha256'][name]=sha(path)
        report['sha256'][name+'_review']=sha(review_path)
    regression=load(args.regression)
    if not regression.get('harness_complete'):
        raise ValueError('Regression collection incomplete')
    report['data']['regression']={'retrieval':regression['retrieval_summary'],
                                 'by_source':regression['retrieval_by_source'],
                                 'corpus_sha256':regression['corpus_sha256']}
    if len({d['corpus_sha256'] for d in report['data'].values()})!=1:
        raise ValueError('Results do not use the same frozen corpus')
    unit=load(args.unit)
    if not unit.get('passed'):
        raise ValueError('Unit tests failed')
    for name,expected in unit['files_sha256'].items():
        if sha(CODE/name)!=expected:
            raise ValueError('Code changed after unit validation: '+name)
    report['unit_tests']={k:unit[k] for k in ['tests_run','failures','errors','seconds','passed']}
    report['sha256']['regression']=sha(args.regression)
    report['sha256']['unit']=sha(args.unit)
    # Independent data-quality goals, not hard-coded to whichever counts we got.
    blind=report['data']['blind']
    passage=blind['semantic_review']['passage']
    strict=blind['strict_answer_contract']['passage']
    report['acceptance']={
        'file_hit5_at_least_90pct':blind['retrieval']['file_hit_at5']/blind['retrieval']['positives']>=.9,
        'evidence_after_get_at_least_90pct':blind['retrieval']['evidence_after_get_from_top5']/blind['retrieval']['evidence_labeled']>=.9,
        'semantic_at_least_90pct':passage['semantic_pass']/passage['reviewed']>=.9,
        'no_unsupported_answers_to_negatives':strict['unsafe_answers']==0,
        'no_fabricated_quotes':strict['invalid_quote_count']==0,
        'no_generation_or_format_errors':strict['generation_or_format_errors']==0,
    }
    report['quality_acceptance_pass']=all(report['acceptance'].values())
    report['delivery_observation']=git_state(CODE)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'quality_acceptance_pass':report['quality_acceptance_pass'],
                      'acceptance':report['acceptance'],'unit_tests':report['unit_tests']},ensure_ascii=False))


if __name__=='__main__':
    main()
