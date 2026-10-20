"""Pair BM25 and hybrid on the same encrypted source-checked question set.

Detailed outcomes and labels remain encrypted outside the public checkout.
Only aggregate measurements are exported. A holdout is consumed before search,
including a failed or interrupted run; reruns must explicitly be regressions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from mr_private import atomic_write, default_store, load_private_rag, load_store, now, save_store
from mr_private_eval import evaluate, labels_sha256, validate_labels


def consume_holdout(labels_path, password, labels):
    """Exclusive reservation prevents two workers consuming a holdout as fresh."""
    marker = Path(str(labels_path) + '.consumed.enc')
    marker.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    # Even if encryption fails, leave the exclusive reservation in place.
    save_store(marker, password, {'version': 1, 'labels_sha256': labels_sha256(labels),
                                 'started_at': now()})
    return marker


def source_signature(rag, labels):
    from mr_private_routing import route_query
    scopes = {route_query(q['query'])['scope'] if q.get('scope', 'auto') == 'auto'
              else q['scope'] for q in labels['questions']}
    for scope in scopes:
        rag.ensure_scope(scope)
    rows = sorted((doc['id'], hashlib.sha256(doc['text'].encode()).hexdigest())
                  for doc in rag.documents.values() if 'all' in scopes or doc.get('scope') in scopes)
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def code_fingerprint(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    names = ('mr_private.py', 'mr_private_index.py', 'mr_private_facts.py',
             'mr_private_semantic.py', 'mr_private_routing.py', 'mr_private_eval.py',
             'mr_private_natural.py', 'mr_private_benchmark.py', 'mr_index.py', 'mr_corpus.py')
    result = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    worker = root / 'scripts' / 'private_eval_worker.py'
    if worker.is_file():
        result['scripts/private_eval_worker.py'] = hashlib.sha256(worker.read_bytes()).hexdigest()
    return result


def isolated_baseline(backend, store, labels_path, output, password, mode):
    command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve().parent /
               'scripts' / 'private_eval_worker.py'), '--backend-code', str(backend),
               '--store', str(store), '--labels', str(labels_path), '--output', str(output), '--mode', mode]
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', MR_LOG='0')
    result = subprocess.run(command, input=(password + '\n').encode(), stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL, env=env, timeout=600,
                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise RuntimeError('Isolated baseline evaluation failed')
    return load_store(output, password)


def compare_reports(baseline, candidate):
    old = {row['id']: row for row in baseline['questions']}
    new = {row['id']: row for row in candidate['questions']}
    if set(old) != set(new) or baseline['summary']['labels_sha256'] != candidate['summary']['labels_sha256']:
        raise ValueError('Paired comparisons require identical questions and frozen labels')
    positives = [key for key, row in old.items() if row['positive']]
    def changes(field):
        is_ok = (lambda row: row['rank'] is not None) if field == 'file' else (lambda row: row['evidence_complete'])
        return {'gains': sum(not is_ok(old[key]) and is_ok(new[key]) for key in positives),
                'regressions': sum(is_ok(old[key]) and not is_ok(new[key]) for key in positives)}
    groups = sorted({old[key]['case_group'] for key in positives})
    group_counts = {}
    for name, rows in (('bm25', old), ('hybrid', new)):
        group_counts[name] = {'complete_evidence_groups': sum(
            all(rows[key]['evidence_complete'] for key in positives
                if rows[key]['case_group'] == group) for group in groups)}
    return {'positive_questions': len(positives), 'event_groups': len(groups),
            'file': changes('file'), 'evidence': changes('evidence'),
            'group_counts': group_counts,
            'hybrid_actually_used': candidate['summary']['modes'] == ['hybrid'],
            'answer_accuracy': None, 'refusal_accuracy': None}


def paired_evaluate(store, labels_path, output, password, *, repetitions=2, baseline_code=None):
    tested_code = code_fingerprint()
    labels = load_store(labels_path, password)
    validate_labels(labels)
    if any(q.get('split') == 'holdout' for q in labels['questions']):
        consume_holdout(labels_path, password, labels)
    frozen_hash = labels_sha256(labels)
    reports, signatures = {}, {}
    old_reports = {}
    baseline_hashes = code_fingerprint(baseline_code) if baseline_code is not None else None
    if baseline_code is not None:
        for mode in ('bm25', 'hybrid'):
            old_output = Path(output).with_name(Path(output).stem + '-baseline-' + mode + '.enc')
            measured = isolated_baseline(baseline_code, store, labels_path, old_output, password, mode)
            old_reports[mode] = measured['report']
            signatures['baseline-' + mode] = measured['source_signature']
    for mode in ('bm25', 'hybrid'):
        rag = load_private_rag(store, password, mode=mode, use_index_cache=True)
        try:
            signatures[mode] = source_signature(rag, labels)
            for question in labels['questions']:
                expected_hash = question.get('source_text_sha256')
                if expected_hash and len(question.get('expected_documents', [])) == 1:
                    original = rag.get(question['expected_documents'][0])
                    if not original.get('ok') or hashlib.sha256(original['text'].encode()).hexdigest() != expected_hash:
                        raise ValueError('A benchmark source changed after question authoring')
            reports[mode] = evaluate(rag, labels, topk=5, repetitions=repetitions)
        finally:
            rag.close()
    if (len(set(signatures.values())) != 1 or code_fingerprint() != tested_code
            or (baseline_code is not None and code_fingerprint(baseline_code) != baseline_hashes)
            or labels_sha256(load_store(labels_path, password)) != frozen_hash):
        raise ValueError('Sources or labels changed during paired evaluation')
    comparison = compare_reports(reports['bm25'], reports['hybrid'])
    summary = {'version': 1, 'tested_at': now(), 'topk': 5, 'labels_sha256': frozen_hash,
               'source_signature': signatures['bm25'], 'labels_unchanged': True,
               'sources_unchanged': True, 'code_unchanged': True, 'code_sha256': tested_code,
               'label_origin': labels.get('label_origin', 'externally_source_checked'),
               'bm25': reports['bm25']['summary'], 'hybrid': reports['hybrid']['summary'],
               'comparison': comparison, 'evaluation_complete': True,
               'retrieval_improved_in_this_sample': comparison['hybrid_actually_used'] and
                    comparison['evidence']['gains'] > comparison['evidence']['regressions'],
               'boundary': labels.get('boundary', 'Retrieval evidence only; no final-answer accuracy.')}
    if old_reports:
        summary.update({'baseline_code_sha256': baseline_hashes,
                        'baseline_bm25': old_reports['bm25']['summary'],
                        'baseline_hybrid': old_reports['hybrid']['summary'],
                        'before_after_bm25': compare_reports(old_reports['bm25'], reports['bm25']),
                        'before_after_hybrid': compare_reports(old_reports['hybrid'], reports['hybrid'])})
        changes = summary['before_after_hybrid']['evidence']
        # A baseline that fell back to keywords (for example because its vector
        # cache belongs to another encoder version) is not a hybrid comparison.
        summary['baseline_hybrid_actually_used'] = old_reports['hybrid']['summary']['modes'] == ['hybrid']
        summary['before_after_improved_in_this_sample'] = (comparison['hybrid_actually_used'] and
                                                          summary['baseline_hybrid_actually_used'] and
                                                          changes['gains'] > changes['regressions'])
    save_store(output, password, {'version': 1, 'labels': labels, 'reports': reports,
                                 'baseline_reports': old_reports, 'summary': summary})
    return summary


def run_authored(store, output_dir, password, baseline_code=None):
    from mr_private_natural import author_sets, authoring_counts, source_candidates
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / 'status.json'
    def state(value, **extra):
        atomic_write(state_path, json.dumps({'state': value, 'updated_at': now(), **extra}).encode())
    state('authoring')
    rag = load_private_rag(store, password, mode='bm25', use_index_cache=True)
    try:
        rag.ensure_scope('personal')
        probe_path = Path(store).parent / 'personal-checkup' / 'source-probes.enc'
        excluded = set()
        if probe_path.is_file():
            probes = load_store(probe_path, password)
            excluded.update(identifier for q in probes['questions'] for identifier in q.get('expected_documents', []))
        probe_excluded_count = len(excluded)
        prior_labels = []
        for prior in sorted((Path(store).parent / 'personal-checkup').glob('natural-*/*-labels.enc')):
            if prior.parent.resolve() == output_dir.resolve():
                continue
            previous = load_store(prior, password)
            excluded.update(identifier for q in previous['questions'] for identifier in q.get('expected_documents', []))
            prior_labels.append(previous)
        originals = list(rag.documents.values())
        candidates = source_candidates(originals, excluded)
        atomic_write(output_dir / 'source-selection.json', json.dumps(
            {'qualified_source_groups': len({r['case_group'] for r in candidates}),
             'field_families_available': len({r['family'] for r in candidates}),
             'previous_probe_documents_excluded': probe_excluded_count,
             'previous_natural_documents_excluded': len(excluded) - probe_excluded_count,
             'selection_consulted_retrieval': False}, indent=2).encode())
        labels = author_sets(originals, excluded)
    finally:
        rag.close()
    # Freeze both sets before any query is sent to either retrieval mode.
    for role, data in labels.items():
        path = output_dir / (role + '-labels.enc')
        if path.exists():
            raise FileExistsError('An authored benchmark must use a fresh directory')
        save_store(path, password, data)
    freeze = {'version': 1, 'frozen_at': now(),
              'code_sha256': code_fingerprint(),
              'labels': {role: {'sha256': labels_sha256(data), **authoring_counts(data)}
                         for role, data in labels.items()},
              'excluded_previous_evaluation_documents': len(excluded),
              'selection_consulted_retrieval': False,
              'boundary': labels['holdout']['boundary']}
    atomic_write(output_dir / 'freeze.json', json.dumps(freeze, indent=2).encode())
    summaries = {}
    probe_path = Path(store).parent / 'personal-checkup' / 'source-probes.enc'
    if probe_path.is_file():
        state('evaluating_source_regression')
        rag = load_private_rag(store, password, mode='hybrid', use_index_cache=True)
        try:
            source_labels = load_store(probe_path, password)
            measured = evaluate(rag, source_labels, topk=5, repetitions=1)
            save_store(output_dir / 'source-regression-results.enc', password, measured)
            atomic_write(output_dir / 'source-regression-summary.json',
                         json.dumps(measured['summary'], indent=2).encode())
        finally:
            rag.close()
    if prior_labels:
        regression_questions = []
        for number, previous in enumerate(prior_labels):
            regression_questions.extend({**q, 'id': 'previous-' + str(number) + '-' + q['id'],
                                         'split': 'regression'} for q in previous['questions'])
        regression = {'version': 1, 'questions': regression_questions,
                      'label_origin': 'previously_consumed_natural_questions_known_regression',
                      'boundary': 'Previously observed questions; no new holdout claim.'}
        # Original encrypted labels are left untouched. This separate rendering
        # changes only evaluation role and ids, never query/source/fact labels.
        save_store(output_dir / 'previous-natural-regression.enc', password, regression)
        state('evaluating_previous_regression')
        rag = load_private_rag(store, password, mode='hybrid', use_index_cache=True)
        try:
            measured = evaluate(rag, regression, topk=5, repetitions=1)
            save_store(output_dir / 'previous-natural-regression-results.enc', password, measured)
            atomic_write(output_dir / 'previous-natural-regression-summary.json',
                         json.dumps(measured['summary'], indent=2).encode())
        finally:
            rag.close()
    for role in ('development', 'holdout'):
        if code_fingerprint() != freeze['code_sha256']:
            raise ValueError('Code changed after label freeze')
        state('evaluating_' + role)
        summaries[role] = paired_evaluate(store, output_dir / (role + '-labels.enc'),
                                         output_dir / (role + '-results.enc'), password,
                                         baseline_code=baseline_code if role == 'holdout' else None)
        atomic_write(output_dir / (role + '-summary.json'), json.dumps(summaries[role], indent=2).encode())
    state('complete', evaluation_complete=True,
          improved_in_holdout=summaries['holdout'].get('before_after_improved_in_this_sample',
                                                      summaries['holdout']['retrieval_improved_in_this_sample']))
    return summaries


def launch_gui(args):
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk(); root.title('真实资料自然问法评测'); root.geometry('570x300')
    frame = ttk.Frame(root, padding=20); frame.pack(fill='both', expand=True)
    ttk.Label(frame, text='真实资料自然问法评测', font=('Microsoft YaHei UI', 15, 'bold')).pack(anchor='w')
    ttk.Label(frame, text='在本机解锁已有资料。先封存题目与证据，再比较关键词和混合检索。\n'
              '问题和逐题结果加密保存；这次不会重新收录或改动原始资料。',
              wraplength=520).pack(anchor='w', pady=12)
    entry = ttk.Entry(frame, show='●', width=34); entry.pack(anchor='w')
    message = tk.StringVar(value='密码只在这个窗口输入，不写入配置或聊天。')
    ttk.Label(frame, textvariable=message, wraplength=520).pack(anchor='w', pady=12)
    def start():
        password = entry.get(); entry.delete(0, 'end')
        if not password:
            return
        try:
            verified = load_store(args.store, password); del verified
            command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()),
                       '--store', str(args.store), '--output-dir', str(args.output_dir),
                       '--author', '--stdin-password']
            if args.baseline_code is not None:
                command.extend(['--baseline-code', str(args.baseline_code)])
            env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       HF_HUB_DISABLE_TELEMETRY='1', MR_LOG='0')
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, env=env,
                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            process.stdin.write((password + '\n').encode()); process.stdin.close(); password = None
            entry.configure(state='disabled'); button.configure(state='disabled')
            message.set('评测已启动，正在封存新的题目与证据。')
            def poll():
                path = args.output_dir / 'status.json'
                if path.is_file():
                    data = json.loads(path.read_text(encoding='utf-8'))
                    state = data.get('state')
                    names = {'authoring': '正在核对原文并封存新题目',
                             'evaluating_source_regression': '正在核对原有原文检索探针',
                             'evaluating_previous_regression': '正在复测此前暴露的问题',
                             'evaluating_development': '正在比较开发题集',
                             'evaluating_holdout': '正在比较首次留出题集',
                             'complete': '评测完成，密码已释放；详细结果仍加密保存。',
                             'failed': '评测未完成；错误类型：' + data.get('error_type', '')}
                    message.set(names.get(state, '后台评测进行中'))
                    if state in {'complete', 'failed'}:
                        return
                if process.poll() is not None:
                    message.set('评测进程已退出，请查看本机状态。'); return
                root.after(1000, poll)
            root.after(1000, poll)
        except Exception as error:
            password = None
            message.set('未启动，请核对密码。错误类型：' + type(error).__name__)
    button = ttk.Button(frame, text='解锁并开始评测', command=start); button.pack(anchor='w')
    entry.bind('<Return>', lambda event: start()); entry.focus_set()
    root.mainloop()


def main(argv=None):
    import getpass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=Path, default=default_store())
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--baseline-code', type=Path, help='Trusted previous code checkout for before/after comparison')
    parser.add_argument('--author', action='store_true')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--stdin-password', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parent
    private_paths = [p for p in (args.store, args.labels, args.output, args.output_dir) if p is not None]
    if any(path.resolve().is_relative_to(repo) for path in private_paths):
        parser.error('Private data, labels and detailed results must stay outside the checkout')
    if (args.author or args.gui) and args.output_dir is None:
        parser.error('Authoring or GUI requires --output-dir')
    if not (args.author or args.gui) and (args.labels is None or args.output is None):
        parser.error('Paired evaluation requires --labels and --output')
    if args.gui:
        launch_gui(args); return 0
    password = sys.stdin.readline().rstrip('\r\n') if args.stdin_password else getpass.getpass('Archive password: ')
    try:
        if args.author:
            summary = run_authored(args.store, args.output_dir, password, baseline_code=args.baseline_code)
        else:
            summary = paired_evaluate(args.store, args.labels, args.output, password, baseline_code=args.baseline_code)
            if args.summary:
                atomic_write(args.summary, json.dumps(summary, indent=2).encode())
        print(json.dumps(summary))
        return 0
    except Exception as error:
        if args.output_dir:
            atomic_write(args.output_dir / 'status.json', json.dumps(
                {'state': 'failed', 'updated_at': now(), 'error_type': type(error).__name__}).encode())
        print(json.dumps({'evaluation_complete': False, 'error_type': type(error).__name__}))
        return 2
    finally:
        password = None


if __name__ == '__main__':
    raise SystemExit(main())
