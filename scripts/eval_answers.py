"""Evaluate fixed snippet/passages flows over frozen, privately labelled data.

Uses the real stdio MCP server. An optional command JSON file supplies a model
test target (argv, output_format=claude_json|text, timeout); the target receives
only a question and retrieved evidence on stdin, never the answer key.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import mr_answer_eval as ae
import mr_eval


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def verify_snapshot(manifest):
    manifest = Path(manifest)
    expected = json.loads(manifest.read_text(encoding='utf-8'))
    for name, value in expected.items():
        if sha(manifest.parent / name) != value:
            raise ValueError('Frozen source changed: ' + name)
    return sha(manifest)


def trial_inputs(args):
    files = sorted(CODE.glob('*.py')) + sorted((CODE/'scripts').glob('*.py')) + [CODE/'corpus-policy.json']
    files += args.dataset + args.reference + [args.snapshot_manifest]
    if args.command:
        files.append(args.command)
    return {'files_sha256': {str(p.resolve()): sha(p) for p in files},
            'snapshot_sha256': verify_snapshot(args.snapshot_manifest),
            'policies': ['snippet', 'passage'], 'topk_search': 10, 'topk_context': 5,
            'max_context_chars': 18000, 'prompt_sha256': ae.digest(ae.ANSWER_SYSTEM),
            'models_path': str(args.models.resolve()),
            'request_timeout': args.request_timeout,
            'answer_workers': args.workers,
            'author_independence': 'same agent authored labels; generator cannot see labels; not human blind audit'}


def seal_trial(manifest, inputs):
    record = {'sealed_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **inputs}
    with Path(manifest).open('x', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return record


def consume_trial(manifest, inputs):
    manifest = Path(manifest)
    record = json.loads(manifest.read_text(encoding='utf-8'))
    if {k: record[k] for k in inputs} != inputs:
        raise ValueError('Sealed inputs/code/policy changed')
    consumed = manifest.with_suffix('.consumed.json')
    # Create before any retrieval or model call, including failed attempts.
    with consumed.open('x', encoding='utf-8') as f:
        json.dump({'attempt_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                   'seal_sha256': sha(manifest), 'subsequent_use': 'regression only'}, f, indent=2)
    return record


def validate_search_response(data, arguments):
    if not data.get('ok') or data.get('mode') != arguments.get('mode', 'hybrid'):
        raise RuntimeError('Search failed or silently degraded')
    if arguments.get('rerank') and not (data.get('rerank') or {}).get('model'):
        raise RuntimeError('Requested confidence gate did not run')
    # note also contains legitimate low/medium-confidence explanations. Check
    # runtime identities, never treat a warning on an unanswerable query as a
    # model failure (or omit that negative from evaluation).
    if not data.get('corpus_sha256'):
        raise RuntimeError('Search omitted corpus identity')


def decode_tool_result(name, result):
    data = json.loads(result['content'][0]['text'])
    if result.get('isError') and name != 'memory_get':
        raise RuntimeError('MCP tool error')
    if result.get('isError') and data.get('ok'):
        raise RuntimeError('MCP error contradicts its success payload')
    return data


class MCP:
    def __init__(self, args):
        args.cache.mkdir(parents=True, exist_ok=True)
        for pattern in ('emb-*.npy', 'emb_meta-*.json'):
            for source in args.models.glob(pattern):
                dest = args.cache/source.name
                if not dest.exists():
                    shutil.copy2(source, dest)
        env = dict(os.environ)
        # Do not let experiment knobs inherited from another terminal leak in.
        for name in list(env):
            if name.startswith('MR_'):
                env.pop(name)
        env.update(MR_PDF='1', MR_LOG='0', MR_GATE='note', MR_RERANK='bge-reranker-base',
                   MR_RERANK_MODE='gate', MR_CHUNK_STRATEGY='paragraph',
                   MR_EMBED_MODEL='bge-small-zh-v1.5', MR_EMBED_MODEL_PDF='bge-m3',
                   MR_CACHE_DIR=str(args.cache.resolve()), MR_MODELS_DIR=str(args.models.resolve()),
                   MR_CORPUS_POLICY=str((args.frozen/'corpus-policy.json').resolve()),
                   MR_EVAL_FROZEN=str(args.frozen.resolve()), MR_EVAL_PDF=str(args.pdf_cache.resolve()),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                   PYTHONIOENCODING='utf-8')
        bootstrap = ("import os,runpy,mr_corpus; "
                     "[s.update(dir=os.path.join(os.environ['MR_EVAL_FROZEN'],s['key'])) for s in mr_corpus.SOURCES]; "
                     "mr_corpus.PDF_CACHE=os.environ['MR_EVAL_PDF']; "
                     "runpy.run_path('mr_mcp_server.py',run_name='__main__')")
        self.stderr = args.output.with_suffix('.mcp-stderr.log').open('w', encoding='utf-8')
        self.trace = args.output.with_suffix('.mcp-trace.private.jsonl').open('w', encoding='utf-8')
        self.process = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', '-c', bootstrap],
                                        cwd=CODE, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, text=True, encoding='utf-8')
        self.queue, self.request_id, self.corpus_hashes = queue.Queue(), 0, []
        self.timeout = args.request_timeout
        threading.Thread(target=self._read, daemon=True).start()
        self.initialize = self.request('initialize', {'protocolVersion':'2024-11-05', 'capabilities':{},
                                                     'clientInfo':{'name':'answer-evaluation','version':'1'}})
        self.process.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
        self.process.stdin.flush()
        self.schemas = self.request('tools/list', {})

    def _read(self):
        try:
            for line in self.process.stdout:
                self.queue.put(line)
        finally:
            self.queue.put(None)

    def request(self, method, params):
        self.request_id += 1
        self.process.stdin.write(json.dumps({'jsonrpc':'2.0','id':self.request_id,
                                            'method':method,'params':params}, ensure_ascii=False)+'\n')
        self.process.stdin.flush()
        line = self.queue.get(timeout=self.timeout)
        if line is None:
            raise RuntimeError('MCP closed stdout')
        result = json.loads(line)
        if result.get('id') != self.request_id or 'error' in result:
            raise RuntimeError('MCP protocol error')
        return result['result']

    def call(self, name, **arguments):
        result = self.request('tools/call', {'name':name,'arguments':arguments})
        self.trace.write(json.dumps({'tool':name,'arguments':arguments,'result':result},ensure_ascii=False)+'\n')
        self.trace.flush()
        data = decode_tool_result(name, result)
        if name == 'memory_search':
            validate_search_response(data, arguments)
            self.corpus_hashes.append(data['corpus_sha256'])
            if len(set(self.corpus_hashes)) != 1:
                raise RuntimeError('Corpus changed during MCP evaluation')
        return data

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.stderr.close()
        self.trace.close()


def generate(command, question, contexts, cwd):
    import mr_corpus
    if any(mr_corpus.redact(c['text']) != c['text'] for c in contexts):
        raise ValueError('Credential-like content reached the model boundary; request blocked')
    payload = ae.answer_payload(question, contexts)
    argv = [ae.ANSWER_SYSTEM if part == '{system_prompt}' else part for part in command['argv']]
    start = time.perf_counter()
    process = subprocess.run(argv, input=json.dumps(payload, ensure_ascii=False), text=True, encoding='utf-8',
                             capture_output=True, cwd=cwd, timeout=command.get('timeout',180))
    try:
        envelope_original=json.loads(process.stdout)
        envelope_safe=mr_corpus.redact_values(envelope_original)
        output_redacted=envelope_original!=envelope_safe
        safe_stdout=json.dumps(envelope_safe,ensure_ascii=False) if output_redacted else process.stdout
    except (ValueError,TypeError):
        safe_stdout=mr_corpus.redact(process.stdout)
        output_redacted=safe_stdout!=process.stdout
    record = {'ms':round((time.perf_counter()-start)*1000,2), 'input_sha256':ae.digest(payload),
              'exit_code':process.returncode, 'stderr':mr_corpus.redact(process.stderr), 'raw_stdout':safe_stdout,
              'output_redacted':output_redacted}
    try:
        if process.returncode:
            raise RuntimeError('Model command failed; exit=%d' % process.returncode)
        text = safe_stdout
        if command.get('output_format') == 'claude_json':
            envelope = json.loads(text)
            if envelope.get('is_error'):
                raise RuntimeError('Model returned is_error')
            record['model_usage'] = envelope.get('modelUsage', {})
            record['reported_cost_usd'] = envelope.get('total_cost_usd')
            record['num_turns'] = envelope.get('num_turns')
            text = envelope['result']
        record['answer'] = ae.parse_answer(text)
    except Exception as exc:
        record['error'] = type(exc).__name__ + ': ' + str(exc)
    return record


def retrieval_summary(rows):
    positives = [r for r in rows if r['positive']]
    negatives = [r for r in rows if not r['positive']]
    labeled = [r for r in positives if r['passage_labeled']]
    return {'questions':len(rows),'positives':len(positives),'negatives':len(negatives),
            'file_hit_at5':sum(r['grade']['returned_rank'] is not None and r['grade']['returned_rank'] <= 5 for r in positives),
            'mrr_at10':round(sum(1/r['grade']['returned_rank'] for r in positives
                              if r['grade']['returned_rank'] is not None and r['grade']['returned_rank'] <= 10)/len(positives),6) if positives else None,
            'evidence_labeled':len(labeled),
            'snippet_evidence_at5':sum(r['grade']['evidence_rank'] is not None and r['grade']['evidence_rank'] <= 5 for r in labeled),
            'evidence_after_get_from_top5':sum(r['get_grade']['evidence_rank'] is not None for r in labeled),
            'negative_high':sum(r['confidence']=='high' for r in negatives),
            'negative_confidence':dict(__import__('collections').Counter(r['confidence'] for r in negatives)),
            'positive_confidence':dict(__import__('collections').Counter(r['confidence'] for r in positives)),
            'get_errors':sum(len(r['contexts']['passage']['get_errors']) for r in rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, action='append', required=True)
    ap.add_argument('--reference', type=Path, action='append', default=[])
    ap.add_argument('--frozen', type=Path, required=True)
    ap.add_argument('--pdf-cache', type=Path, required=True)
    ap.add_argument('--snapshot-manifest', type=Path, required=True)
    ap.add_argument('--models', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--command', type=Path, help='Optional JSON command configuration; model test target has no tools')
    ap.add_argument('--seal', type=Path)
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--workers', type=int, choices=(1,2), default=2)
    ap.add_argument('--request-timeout', type=int, default=300,
                    help='MCP deadline in seconds; first uncached embedding build may need separate prewarm')
    args=ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    goldens=[json.loads(p.read_text(encoding='utf-8')) for p in args.dataset]
    references=[json.loads(p.read_text(encoding='utf-8')) for p in args.reference]
    blind=any(q.get('split',g.get('split'))=='blind' for g in goldens for q in g['queries'])
    if blind and not references:
        ap.error('Blind evaluation requires regression/development references for leakage checks')
    if args.command or blind:
        ae.validate_suites(goldens,references)
    for g in goldens:
        mr_eval.validate_labels(g)
    inputs=trial_inputs(args)
    if args.freeze:
        if not args.seal or not blind:
            ap.error('Freeze requires a blind dataset and --seal')
        seal_trial(args.seal,inputs)
        print('Blind trial sealed; no retrieval or generation performed',flush=True)
        return 0
    if args.output.exists():
        raise FileExistsError('Keep previous trial outputs; choose a fresh output path')
    report={'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'inputs':inputs,
            'transport':'real MCP stdio','retrieval':[],'answers':{'snippet':[],'passage':[]},
            'quality_pass':None,'harness_complete':False}
    if blind:
        if not args.seal:
            ap.error('Blind evaluation requires --seal; do not silently rerun as blind')
        report['seal']=consume_trial(args.seal,inputs)
    save(args.output,report)
    # Read frozen sources for label/audit verification only, before starting MCP.
    os.environ['MR_PDF']='1'
    import mr_corpus
    for s in mr_corpus.SOURCES:
        s['dir']=str(args.frozen/s['key'])
    mr_corpus.CORPUS_POLICY=str(args.frozen/'corpus-policy.json')
    mr_corpus.PDF_CACHE=str(args.pdf_cache)
    chunks=mr_corpus.load_chunks()
    report['corpus_audit']=mr_eval.require_clean_corpus(chunks,goldens+references)
    report['source_label_checks']=[ae.validate_source_evidence(g,chunks) for g in goldens]
    command=json.loads(args.command.read_text(encoding='utf-8')) if args.command else None
    cwd=args.output.parent/'answer-target'
    cwd.mkdir(exist_ok=True)
    questions={}
    client=None
    try:
        client=MCP(args)
        report['mcp_initialize']=client.initialize
        report['mcp_tool_schema_sha256']=ae.digest(client.schemas)
        for golden in goldens:
            for i,q in enumerate(golden['queries'],1):
                q=dict(q)
                q.setdefault('id',args.dataset[goldens.index(golden)].stem+'-'+str(i))
                q.setdefault('source',golden.get('source'))
                questions[q['id']]=q
                kwargs={'query':q['q'],'mode':'hybrid','topk':10,'rerank':True}
                if q['source']:
                    kwargs['source']=q['source']
                if q.get('include_evaluation',golden.get('include_evaluation')):
                    kwargs['include_evaluation']=True
                t=time.perf_counter()
                result=client.call('memory_search',**kwargs)
                retrieval_ms=(time.perf_counter()-t)*1000
                contexts={p:ae.assemble_context(result,lambda **kw:client.call('memory_get',**kw),policy=p)
                          for p in ['snippet','passage']}
                row={'id':q['id'],'case_group':q.get('case_group'),'source':q['source'] or 'notes',
                     'positive':bool(q['expect']),'passage_labeled':bool(q.get('evidence') or q.get('evidence_groups')),
                     'confidence':result['confidence'],'grade':mr_eval.grade_hits(q,result['hits']),
                     'get_grade':mr_eval.grade_hits(q,[dict(c,snippet=c['text']) for c in contexts['passage']['contexts']]),
                     'retrieval_ms':round(retrieval_ms,2),'search':result,'contexts':contexts}
                report['retrieval'].append(row)
                save(args.output,report)
                print('retrieved '+q['id'],flush=True)
        report['corpus_sha256']=client.corpus_hashes[0]
        report['corpus_observations']=len(client.corpus_hashes)
        report['retrieval_summary']=retrieval_summary(report['retrieval'])
        report['retrieval_by_source']={s:retrieval_summary([r for r in report['retrieval'] if r['source']==s])
                                       for s in sorted({r['source'] for r in report['retrieval']})}
    finally:
        if client:
            client.close()
    if command:
        def task(row,policy):
            q=questions[row['id']]
            result={'id':q['id'],'case_group':q['case_group'],'source':row['source'],
                    'expected_decision':q['answer_contract']['decision'],'policy':policy}
            try:
                generated=generate(command,q,row['contexts'][policy]['contexts'],cwd)
                result.update(generated)
                if not generated.get('error'):
                    result['grade']=ae.grade_answer(q,generated['answer'],row['contexts'][policy]['contexts'])
            except Exception as exc:
                result['error']=type(exc).__name__+': '+str(exc)
            return result
        futures=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for row in report['retrieval']:
                # Counterbalance request order without sampling or result-based retries.
                policies=['snippet','passage'] if int(hashlib.sha256(row['id'].encode()).hexdigest()[-1],16)%2 else ['passage','snippet']
                for policy in policies:
                    futures.append(pool.submit(task,row,policy))
            for future in concurrent.futures.as_completed(futures):
                result=future.result()
                report['answers'][result['policy']].append(result)
                save(args.output,report)
                print('answered '+result['policy']+' '+result['id']+(' ERROR' if result.get('error') else ''),flush=True)
        for rows in report['answers'].values():
            rows.sort(key=lambda r:r['id'])
        report['answer_summary']={p:ae.summarize_answers(rows) for p,rows in report['answers'].items()}
    report['inputs_unchanged']=inputs==trial_inputs(args)
    report['completed_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    # A malformed model answer is a measured failure, not a reason to remove
    # the question or rerun a blind trial until its output looks valid.
    report['harness_complete']=report['inputs_unchanged'] and (
        not command or all(len(rows)==len(report['retrieval']) for rows in report['answers'].values()))
    report['all_generations_valid']=not any(r.get('error') for rows in report['answers'].values() for r in rows)
    report['quality_pass']='requires semantic review; collection success is not answer acceptance'
    save(args.output,report)
    print(json.dumps({'harness_complete':report['harness_complete'],
                      'retrieval':report['retrieval_summary'],'answers':report.get('answer_summary')},ensure_ascii=False),flush=True)
    return 0 if report['harness_complete'] else 1


if __name__=='__main__':
    raise SystemExit(main())
