"""Locally unlock, refresh sources, prepare encrypted indexes and test retrieval.

The native password window transfers the password only through a private stdin
pipe. The worker releases it when the task ends. Detailed results stay encrypted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from mr_private import (atomic_write, default_store, import_journals, load_private_rag,
                        load_store, now, save_store)


def status(store, state, **extra):
    data = {'updated_at': now(), 'state': state, **extra}
    atomic_write(Path(store).parent / 'optimization.status.json', json.dumps(data, indent=2).encode())


def source_probes(rag, limit=20):
    """Source-derived integrity probes, deliberately not natural-question accuracy."""
    documents = [d for d in rag.documents.values() if d['scope'] == 'personal' and
                 d.get('purpose') not in {'navigation', 'evaluation', 'excluded'} and
                 d.get('method') != 'curated_memory']
    questions = []
    for document in sorted(documents, key=lambda d: d['id']):
        lines = [line.strip() for line in document['text'].splitlines()
                 if 30 <= len(line.strip()) <= 180 and not line.lstrip().startswith('#')]
        for line in lines[:20]:
            # Freeze expected alternatives from source bodies before searching.
            alternatives = [d['id'] for d in documents if line in d['text']]
            if len(alternatives) > 3:
                continue
            questions.append({'id': 'source-probe-' + str(len(questions) + 1),
                 'case_group': document['id'], 'split': 'regression', 'query': line,
                 'scope': 'personal', 'expected_documents': alternatives, 'required_text': [line]})
            break
        if len(questions) >= limit:
            break
    if not questions:
        raise ValueError('No suitable source-derived probes')
    return {'version': 1, 'label_origin': 'automatically_derived_from_original_source_text',
            'boundary': 'Integrity and retrieval probes, not independently authored user-question accuracy.',
            'questions': questions}


def acceptance_checks(summary):
    """Baseline quality may be worse; startup measurements cannot veto that gain."""
    baseline, candidate = summary['bm25_source_probes'], summary['hybrid_source_probes']
    starts = summary.get('fresh_process_start')
    checks = {
        'original_source_hashes_match': bool(summary['personal_text_hashes_match']),
        'native_ui_passed': bool(summary['native_ui']['passed']),
        'hybrid_was_used': candidate['modes'] == ['hybrid'],
        'labels_unchanged_between_modes': baseline['labels_sha256'] == candidate['labels_sha256'],
        'file_hits_preserved': candidate['file_hits_at_k'] >= baseline['file_hits_at_k'],
        'evidence_preserved': candidate['evidence_complete_at_k'] >= baseline['evidence_complete_at_k'],
    }
    if starts:
        checks.update({
            'fresh_baseline_mode': starts['bm25']['actual_mode'] == 'bm25',
            'fresh_candidate_mode': starts['hybrid']['actual_mode'] == 'hybrid',
            'fresh_candidate_file_hit': bool(starts['hybrid']['file_hit']),
            'fresh_candidate_evidence': bool(starts['hybrid']['evidence_complete']),
            'fresh_start_budgets_match': starts['bm25'].get('topk') == starts['hybrid'].get('topk'),
        })
    return checks


def fresh_start(store, labels_path, password, mode):
    """New Python process; OS file cache is deliberately not flushed."""
    command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()),
               '--store', str(store), '--stdin-password', '--latency-worker', mode,
               '--labels', str(labels_path)]
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', MR_LOG='0')
    started = time.perf_counter()
    process = subprocess.run(command, input=(password + '\n').encode('utf-8'),
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             env=env, timeout=180,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if process.returncode:
        raise RuntimeError('Fresh-process measurement failed')
    result = json.loads(process.stdout.decode('utf-8'))
    result['process_end_to_end_seconds'] = round(time.perf_counter() - started, 4)
    result['boundary'] = ('New process including imports, encrypted benchmark-query loading, '
                          'RAG loading, first query and cleanup. OS file cache was not flushed.')
    return result


def latency_worker(store, labels_path, password, mode):
    from mr_private_facts import compact_search
    labels = load_store(labels_path, password)
    question = next(q for q in labels['questions'] if q.get('positive', True))
    started = time.perf_counter()
    rag = load_private_rag(store, password, mode=mode, use_index_cache=mode == 'hybrid')
    load_seconds = time.perf_counter() - started
    try:
        started = time.perf_counter()
        result = compact_search(rag, question['query'], topk=5, scope=question.get('scope', 'personal'))
        query_seconds = time.perf_counter() - started
        matches = [h for h in result['hits'] if h['document_id'] in question['expected_documents']]
        context = '\n'.join(h['evidence']['context'] for h in matches).casefold()
        return {'requested_mode': mode, 'actual_mode': result['mode'], 'topk': 5,
                'session_load_seconds': round(load_seconds, 4),
                'first_query_seconds': round(query_seconds, 4), 'file_hit': bool(matches),
                'evidence_complete': bool(matches) and all(v.casefold() in context
                                                    for v in question['required_text'])}
    finally:
        rag.close()


def run_checkup(store, password, collect=True, fresh_latency=False):
    from mr_private_eval import evaluate
    from mr_private_sources import audit_sources
    store = Path(store).resolve()
    run_dir = store.parent / 'personal-checkup'
    run_dir.mkdir(parents=True, exist_ok=True)
    status(store, 'authenticating')
    snapshot = load_store(store, password)
    source = Path(snapshot['source']) if snapshot.get('source') else None
    source_root = Path(snapshot.get('source_root') or store.parent.parent)
    del snapshot
    notebook = {'state': 'not_checked'}
    if collect:
        if source is None or not (source / 'tools' / 'build_memory.py').is_file():
            raise ValueError('Source collection tools are unavailable on this computer')
        from mr_private_maintenance import run_script
        from mr_private_updates import running_processes
        if running_processes('mr_private_maintenance.py', str(source)):
            raise RuntimeError('A source collector is already running')
        for script_args, phase in ((['collect'], 'collecting'), (['archives'], 'archive_members')):
            status(store, phase)
            run_script(source, 'build_memory.py', script_args, password, run_dir)
        status(store, 'onenote')
        try:
            from mr_private_onenote import harvest
            notebook = harvest(source_root, source / 'onenote.enc', password)
        except Exception as error:
            notebook = {'state': 'unavailable', 'error_type': type(error).__name__}
        atomic_write(store.parent / 'onenote.status.json', json.dumps({'updated_at': now(), **notebook}, indent=2).encode())
        status(store, 'publishing')
        payload = import_journals(source, store, password, partitioned=True)
        del payload
    status(store, 'preparing_semantics', completed_windows=0)
    rag = load_private_rag(store, password, mode='hybrid')
    try:
        labels_path = run_dir / 'source-probes.enc'
        if labels_path.is_file():
            # Existing regression labels stay immutable across repairs.
            labels = load_store(labels_path, password)
        else:
            labels = source_probes(rag)
            # Questions and source facts never enter a plaintext fixture or log.
            save_store(labels_path, password, labels)
        started = time.perf_counter()
        def progress(done, total):
            status(store, 'preparing_semantics', completed_windows=done, total_windows=total)
        preparation = rag.prepare_semantic('personal', progress)
        preparation_seconds = time.perf_counter() - started
        status(store, 'testing_sources')
        sources = audit_sources(rag)
        checks = []
        for document in rag.payload['documents']:
            checks.append(document.get('text_sha256') == hashlib.sha256(document['text'].encode()).hexdigest())
        # Use a new session for timings; source-derived exact-match questions
        # cannot establish real paraphrase recall or answer correctness.
        status(store, 'testing_retrieval')
        load_started = time.perf_counter()
        baseline = load_private_rag(store, password, mode='bm25', use_index_cache=False)
        baseline_load_seconds = time.perf_counter() - load_started
        try:
            baseline_result = evaluate(baseline, labels, repetitions=1)
        finally:
            baseline.close()
        load_started = time.perf_counter()
        candidate = load_private_rag(store, password, mode='hybrid')
        candidate_load_seconds = time.perf_counter() - load_started
        try:
            candidate_result = evaluate(candidate, labels, repetitions=3)
        finally:
            candidate.close()
        for result, seconds in ((baseline_result, baseline_load_seconds), (candidate_result, candidate_load_seconds)):
            result['summary']['session_load_seconds'] = round(seconds, 4)
            result['summary']['first_query_seconds'] = result['questions'][0]['first_call_seconds']
            result['summary']['load_and_first_query_seconds'] = round(seconds + result['questions'][0]['first_call_seconds'], 4)
        baseline_result['summary']['index_cache'] = 'disabled_for_BM25_baseline'
        candidate_result['summary']['index_cache'] = 'authenticated_encrypted_sidecars'
        measured_starts = None
        if fresh_latency:
            status(store, 'testing_fresh_start')
            measured_starts = {mode: fresh_start(store, labels_path, password, mode)
                               for mode in ('bm25', 'hybrid')}
        detailed = {'version': 1, 'tested_at': now(), 'source_probes': labels,
                    'bm25': baseline_result, 'hybrid': candidate_result}
        save_store(run_dir / 'source-probe-results.enc', password, detailed)
        hybrid_summary = candidate_result['summary']
        summary = {'tested_at': now(), 'preparation': preparation,
                   'preparation_seconds': round(preparation_seconds, 4),
                   'personal_documents': sum(d['scope'] == 'personal' for d in rag.payload['documents']),
                   'personal_text_hashes_match': all(checks), 'source_audit': sources,
                   'content_review_counts': rag.payload.get('coverage', {}).get('content_gaps', {}),
                   'notebook': notebook, 'bm25_source_probes': baseline_result['summary'],
                   'hybrid_source_probes': hybrid_summary,
                   'fresh_process_start': measured_starts,
                   'natural_question_accuracy': None,
                   'boundary': 'Real corpus source-derived probes plus integrity checks. Natural user questions and answers require separately source-checked encrypted labels.'}
        from mr_private_maintenance import verify_native_ui
        summary['native_ui'] = verify_native_ui(store, password)
        summary['acceptance_checks'] = acceptance_checks(summary)
        passed = all(summary['acceptance_checks'].values())
        summary['passed'] = passed
        atomic_write(run_dir / 'summary.json', json.dumps(summary, indent=2).encode())
        final_state = 'failed_checks' if not passed else ('complete_with_content_gaps'
            if summary['content_review_counts'] or notebook.get('state') not in
            {'updated', 'no_linked_notebooks', 'not_checked'} else 'complete')
        status(store, final_state, **summary)
        return summary
    finally:
        rag.close()


def launch_password_window(store, collect=True):
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk()
    root.title('个人 RAG 优化与测试')
    root.geometry('560x250')
    outer = ttk.Frame(root, padding=20); outer.pack(fill='both', expand=True)
    ttk.Label(outer, text='个人 RAG 优化与测试', font=('Microsoft YaHei UI', 16, 'bold')).pack(anchor='w')
    message = ('在本机输入档案密码。更新资料、建立加密语义索引并测试。' if collect else
               '在本机输入档案密码。复用已有加密索引，以原来的题目复测检索修复。')
    ttk.Label(outer, text=message + '\n后台任务完成后释放密码；详细评测仍加密保存。',
              wraplength=500).pack(anchor='w', pady=12)
    password_entry = ttk.Entry(outer, show='●', width=32); password_entry.pack(anchor='w')
    label = tk.StringVar(value='密码不会写入配置或发送到聊天。')
    ttk.Label(outer, textvariable=label, wraplength=500).pack(anchor='w', pady=8)
    def start():
        password = password_entry.get(); password_entry.delete(0, 'end')
        if not password:
            return
        try:
            # Authenticate locally before transferring to the hidden worker.
            payload = load_store(store, password); del payload
            command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()),
                       '--store', str(store), '--stdin-password']
            if not collect:
                command.append('--no-collect')
            process = subprocess.Popen(command, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            process.stdin.write((password + '\n').encode()); process.stdin.close()
            password = None
            status(store, 'worker_started', pid=process.pid)
            button.configure(state='disabled'); password_entry.configure(state='disabled')
            label.set('后台任务已启动。这个窗口只显示进度，可以关闭。')
            root.after(1000, poll)
        except Exception as error:
            password = None
            label.set('未启动：请核对密码与加密文件。错误类型：' + type(error).__name__)
    names = {'authenticating': '正在解锁', 'collecting': '正在收录资料',
             'archive_members': '正在读取项目资料包', 'onenote': '正在读取已同步笔记',
             'publishing': '正在更新加密资料库', 'preparing_semantics': '正在准备语义检索',
             'testing_sources': '正在核对资料来源', 'testing_retrieval': '正在测试检索',
             'testing_fresh_start': '正在用新进程测首次查询',
             'complete': '优化测试已完成', 'complete_with_content_gaps': '优化测试已完成，部分来源仍需打开或解锁',
             'failed': '任务未完成，请查看本地状态', 'failed_checks': '评测未通过，不能作为验收完成'}
    def poll():
        path = Path(store).parent / 'optimization.status.json'
        if path.is_file():
            data = json.loads(path.read_text(encoding='utf-8'))
            text = names.get(data.get('state'), '后台任务已启动')
            if data.get('total_windows'):
                text += '：{}/{}'.format(data.get('completed_windows', 0), data['total_windows'])
            label.set(text)
            if data.get('state') in {'complete', 'complete_with_content_gaps', 'failed', 'failed_checks'}:
                return
        root.after(2000, poll)
    button = ttk.Button(outer, text='开始后台优化与测试', command=start); button.pack(anchor='w')
    password_entry.bind('<Return>', lambda event: start())
    password_entry.focus_set()
    root.mainloop()


def main():
    import getpass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=Path, default=default_store())
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--stdin-password', action='store_true')
    parser.add_argument('--no-collect', action='store_true')
    parser.add_argument('--latency-worker', choices=('bm25', 'hybrid'), help=argparse.SUPPRESS)
    parser.add_argument('--labels', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.gui:
        launch_password_window(args.store, collect=not args.no_collect)
        return 0
    password = sys.stdin.readline().rstrip('\r\n') if args.stdin_password else getpass.getpass('Archive password: ')
    try:
        if args.latency_worker:
            print(json.dumps(latency_worker(args.store, args.labels, password, args.latency_worker)))
            return 0
        result = run_checkup(args.store, password, collect=not args.no_collect, fresh_latency=True)
        return 0 if result['passed'] else 1
    except Exception as error:
        status(args.store, 'failed', error_type=type(error).__name__)
        return 2
    finally:
        password = None


if __name__ == '__main__':
    raise SystemExit(main())
