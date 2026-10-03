"""Collect, publish and verify the private corpus without archive packaging gates."""
import argparse
import datetime as dt
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from mr_private import (atomic_write, import_journals, load_private_rag, load_store,
                        now, read_journal, save_store)
from mr_private_policy import classify, policy_signature, SCOPES


def source_fingerprint(source, root=None, notebook_probe=None):
    """Check metadata before decrypting journals or rebuilding unchanged text."""
    root = Path(root or os.environ.get('OneDrive') or Path.home() / 'OneDrive')
    digest = hashlib.sha256(policy_signature().encode('ascii'))
    errors = 0
    for parent, directories, files in os.walk(root, followlinks=False):
        if Path(parent) == root:
            directories[:] = [d for d in directories if d.casefold() != '_private-memory-rag']
        directories.sort()
        for name in sorted(files):
            path = Path(parent) / name
            try:
                stat = path.stat()
                digest.update(str(path.relative_to(root)).encode('utf-8'))
                digest.update(('%s:%s;' % (stat.st_size, stat.st_mtime_ns)).encode('ascii'))
            except OSError:
                errors += 1
    for directory, names in ((Path(source) / 'tools', ('build_memory.py', 'mr_private_policy.py', 'mr_private_media.py', 'private-policy.json')),
                             (Path(__file__).parent, ('mr_private.py', 'mr_private_policy.py', 'mr_private_onenote.py', 'mr_private_media.py', 'mr_private_maintenance.py'))):
        for name in names:
            path = directory / name
            if path.is_file(): digest.update(path.read_bytes())
    try:
        if notebook_probe is None:
            from mr_private_onenote import application, _links
            hierarchy = application().GetHierarchy('', 4, 2) if _links(root) else ''
        else:
            hierarchy = notebook_probe()
        digest.update(hierarchy.encode('utf-8'))
    except Exception as error:
        digest.update(type(error).__name__.encode('ascii'))
    digest.update(str(errors).encode('ascii'))
    return digest.hexdigest()


class MaintenanceStopped(Exception):
    pass


def implementation_signature():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('mr_private*.py')):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def update_status(path, state, **values):
    atomic_write(path, json.dumps({'state': state, 'updated_at': now(), 'process_id': os.getpid(), **values},
                                 ensure_ascii=False, indent=2).encode('utf-8'))


def run_script(source, name, args, password, run_dir):
    if (source / 'stop.request').exists(): raise MaintenanceStopped()
    command = [sys.executable, '-X', 'utf8', str(source / 'tools' / name), *args, '--stdin-password']
    result = subprocess.run(command, input=(password + '\n').encode('utf-8'), stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if (source / 'stop.request').exists(): raise MaintenanceStopped()
    if result.returncode:
        save_store(run_dir / ('diagnostic-' + name.replace('.py', '') + '.enc'), password,
                   {'version': 1, 'returncode': result.returncode,
                    'stderr': result.stderr.decode('utf-8', errors='replace')})
        raise RuntimeError(name + ':exit_' + str(result.returncode))


def stop_legacy_watcher(store):
    marker = store.parent / 'stop.request'
    marker.touch()
    state = store.parent / 'watcher.status.json'
    pid = None
    if state.is_file():
        pid = json.loads(state.read_text(encoding='utf-8')).get('pid')
    if pid and pid != os.getpid():
        quoted = str(store).replace("'", "''")
        code = ("$ragTask=Get-CimInstance Win32_Process -Filter 'ProcessId=" + str(int(pid)) + "'; "
                "if($ragTask -and $ragTask.CommandLine -and $ragTask.CommandLine.Contains('mr_private.py') "
                "-and $ragTask.CommandLine.Contains('" + quoted + "')) { "
                "Start-Sleep -Seconds 2; $ragTask=Get-CimInstance Win32_Process -Filter 'ProcessId=" + str(int(pid)) + "'; "
                "if($ragTask -and $ragTask.CommandLine.Contains('mr_private.py') -and $ragTask.CommandLine.Contains('" + quoted + "')) "
                "{ Stop-Process -Id " + str(int(pid)) + " -Force } }")
        subprocess.run(['powershell.exe', '-NoProfile', '-Command', code], capture_output=True,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), check=True)
    if marker.exists(): marker.unlink()


def verify_live(store, password):
    started = time.perf_counter()
    rag = load_private_rag(store, password)
    unlock_seconds = time.perf_counter() - started
    violations = 0
    for document in rag.payload['documents']:
        if document.get('method') not in {'curated_memory', 'project_catalogue'} and classify(
                document['file'], text=document['text'])['scope'] != 'personal':
            violations += 1
    queries = ('家庭关系', '旅行', '护照', '地址', '税务', '工作', '生日')
    from mr_private_facts import compact_search
    checks, timings = [], []
    for query in queries:
        started = time.perf_counter()
        result = compact_search(rag, query, topk=5, preview_chars=700)
        seconds = time.perf_counter() - started
        checks.append({'query': query, 'hits': len(result['hits']), 'seconds': round(seconds, 4),
                       'all_personal': all(rag.documents[h['document_id']]['scope'] == 'personal' for h in result['hits'])})
        for _ in range(3):
            started = time.perf_counter()
            compact_search(rag, query, topk=5, preview_chars=700)
            timings.append(time.perf_counter() - started)
    # Dense retrieval returns nearest neighbours even for nonsense. This is
    # specifically a lexical no-match control, not a refusal-accuracy test.
    negative_empty = not rag.search('AbsentUniqueTerm991912xyz987', mode='bm25')['hits']
    shards_lazy = all(scope not in rag.loaded_scopes for scope in rag.payload.get('shards', {}))
    text_hashes = all(d.get('text_sha256') == hashlib.sha256(d['text'].encode('utf-8')).hexdigest()
                        for d in rag.payload['documents'])
    original_checked, original_failed, virtual_sources = 0, 0, 0
    source_root = Path(rag.payload.get('source_root') or os.environ.get('OneDrive') or Path.home() / 'OneDrive')
    for document in rag.payload['documents']:
        for item in [document, *document.get('duplicate_sources', [])]:
            name = item['file']
            if '!/' in name.replace('\\', '/') or not item.get('source_sha256'):
                virtual_sources += 1
                continue
            path = (source_root / name).resolve()
            if not path.is_relative_to(source_root.resolve()):
                original_failed += 1
                continue
            try:
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    while block := stream.read(4 * 1024 * 1024): digest.update(block)
                original_checked += 1
                original_failed += digest.hexdigest() != item['source_sha256']
            except OSError:
                original_failed += 1
    wrong_password_rejected = False
    try:
        load_store(store, 'synthetic-wrong-password-' + os.urandom(8).hex())
    except Exception as error:
        from cryptography.exceptions import InvalidTag
        wrong_password_rejected = isinstance(error, InvalidTag)
    report = {'verified_at': now(), 'snapshot_updated_at': rag.payload['updated_at'],
              'unlock_seconds': round(unlock_seconds, 4), 'documents': len(rag.payload['documents']),
              'chunks': len(rag.payload['chunks']), 'default_scope_violations': violations,
              'personal_text_hashes_verified': text_hashes, 'shards_loaded_only_on_request': shards_lazy,
              'original_files_sha256_checked': original_checked, 'original_files_sha256_failed': original_failed,
              'virtual_or_unhashed_sources': virtual_sources,
              'wrong_password_rejected': wrong_password_rejected, 'negative_query_empty': negative_empty,
              'negative_control_mode': 'bm25',
              'search_checks': checks, 'warm_search_p95_seconds': round(sorted(timings)[math.ceil(.95 * len(timings)) - 1], 4),
              'quality_boundary': 'Real retrieval smoke checks, not an independently labelled answer-accuracy evaluation.'}
    scope_checks = []
    for scope, metadata in rag.payload.get('shards', {}).items():
        started = time.perf_counter()
        rag.ensure_scope(scope)
        docs = [d for d in rag.documents.values() if d['scope'] == scope]
        invalid = sum(classify(d['file'], text=d['text'])['scope'] != scope for d in docs)
        from mr_index import tokenize
        probe = next((' '.join(tokenize(d['text'][:4000])[:3]) for d in docs if tokenize(d['text'][:4000])[:3]), '')
        result = rag.search(probe, topk=3, scope=scope) if probe else {'hits': []}
        scope_checks.append({'scope': scope, 'documents': len(docs), 'chunks': metadata['chunks'],
                             'policy_violations': invalid, 'probe_hits': len(result['hits']),
                             'first_load_and_query_seconds': round(time.perf_counter() - started, 4),
                             'count_matches': len(docs) == metadata['documents']})
    report['partition_checks'] = scope_checks
    report['native_ui'] = verify_native_ui(store, password)
    report['passed'] = (not violations and text_hashes and not original_failed and shards_lazy and negative_empty
                        and wrong_password_rejected and all(not c['policy_violations'] and c['count_matches']
                        and c['probe_hits'] for c in scope_checks) and report['native_ui']['passed'])
    return report


def verify_native_ui(store, password):
    """Exercise the deployed snapshot without showing private text on screen."""
    import tkinter as tk
    from mr_private_ui import PrivateMemoryApp
    root = tk.Tk()
    root.withdraw()
    app = PrivateMemoryApp(root, store)
    started = time.perf_counter()
    def until(predicate):
        deadline = time.monotonic() + 120
        while not predicate() and time.monotonic() < deadline:
            root.update()
            time.sleep(.02)
        if not predicate(): raise TimeoutError('Native private UI verification timed out')
    try:
        app.password_entry.insert(0, password)
        app.unlock()
        until(lambda: app.rag is not None)
        app.search_for('护照')
        until(lambda: app.status.get().startswith('找到 '))
        hits = len(app.rows)
        source_matches = False
        if app.rows:
            key = next(iter(app.rows)); row = app.rows[key]
            app.tree.selection_set(key)
            app.full_source()
            document = app.rag.get(row['document_id'])
            source_matches = app.detail.get('1.0', 'end-1c') == document['file'] + '\n\n' + document['text']
        app.coverage_review()
        until(lambda: app.status.get().startswith('资料核对清单：'))
        coverage_items = len(app.rows)
        app.lock()
        cleared = app.password is None and app.rag is None and not app.rows
        return {'passed': bool(hits and source_matches and cleared), 'retrieval_hits': hits,
                'full_source_matches': source_matches, 'coverage_items': coverage_items,
                'lock_clears_session': cleared, 'seconds': round(time.perf_counter() - started, 4)}
    finally:
        app.close()


def cycle(source, store, password, run_dir):
    status = run_dir / 'run.status.json'
    job_status = source / 'job-status.json'
    for path in (status, job_status): update_status(path, 'collecting', phase='source_collection')
    run_script(source, 'build_memory.py', ['collect'], password, run_dir)
    for path in (status, job_status): update_status(path, 'collecting', phase='archive_members')
    run_script(source, 'build_memory.py', ['archives'], password, run_dir)
    notebook = {}
    try:
        from mr_private_onenote import harvest
        notebook = harvest(Path(os.environ.get('OneDrive') or Path.home() / 'OneDrive'), source / 'onenote.enc', password)
    except Exception as error:
        notebook = {'state': 'unavailable', 'error_type': type(error).__name__}
    atomic_write(store.parent / 'onenote.status.json', json.dumps({'updated_at': now(), **notebook}, indent=2).encode('utf-8'))
    for path in (status, job_status): update_status(path, 'indexing', phase='publish_partitions', notebook=notebook)
    payload = import_journals(source, store, password, partitioned=True)
    coverage = payload['coverage']
    del payload
    update_status(status, 'verifying', phase='live_retrieval', notebook=notebook)
    verification = verify_live(store, password)
    atomic_write(store.parent / 'verification.json', json.dumps(verification, ensure_ascii=False, indent=2).encode('utf-8'))
    atomic_write(run_dir / 'live-verification.json', json.dumps(verification, ensure_ascii=False, indent=2).encode('utf-8'))
    if not verification['passed']: raise RuntimeError('Live private retrieval verification failed')
    summary = {'coverage': coverage, 'verification_passed': True, 'notebook': notebook,
               'archive_packaging': 'Existing encrypted archive retained; archive packing is independent of RAG collection.'}
    update_status(status, 'complete_with_content_gaps' if coverage['content_gaps'] or notebook.get('state') not in
                  {'updated', 'no_linked_notebooks'} else 'complete', **summary)
    update_status(job_status, 'monitoring_with_content_gaps' if coverage['content_gaps'] else 'monitoring', **summary)
    return summary


def main(argv=None):
    import getpass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--stdin-password', action='store_true')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--interval', type=int, default=300)
    args = parser.parse_args(argv)
    password = sys.stdin.readline().rstrip('\r\n') if args.stdin_password else getpass.getpass('Archive password: ')
    args.source, args.store = args.source.resolve(), args.store.resolve()
    loaded_implementation = implementation_signature()
    run_dir = (args.run_dir or args.source / 'rag-maintenance').resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.interval < 30: parser.error('Interval must be at least 30 seconds')
    if args.once and not args.worker:
        # Keep the locally supplied password available for a corrected retry,
        # rather than requiring the user to enter it again after an error.
        command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()),
                   '--source', str(args.source), '--store', str(args.store), '--run-dir', str(run_dir),
                   '--stdin-password', '--once', '--worker']
        if args.resume: command.append('--resume')
        while True:
            process = subprocess.run(command, input=(password + '\n').encode('utf-8'),
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if process.returncode == 0: return 0
            save_store(run_dir / 'worker-diagnostic.enc', password,
                       {'version': 1, 'stderr': process.stderr.decode('utf-8', errors='replace')})
            update_status(run_dir / 'supervisor.status.json', 'waiting_for_corrected_retry', worker_exit_code=process.returncode)
            marker = run_dir / 'retry.request'
            while not marker.exists():
                if (args.source / 'stop.request').exists(): return 2
                time.sleep(1)
            marker.unlink()
    try:
        next(read_journal(args.source / 'core.enc', password))
        stop_legacy_watcher(args.store)
        source_marker = args.source / 'stop.request'
        if source_marker.exists(): source_marker.unlink()
        if args.once:
            # The root is published before a long first collection. Subsequent
            # background cycles do not repeat this extra import.
            update_status(run_dir / 'run.status.json', 'indexing', phase='filter_existing_evidence')
            payload = import_journals(args.source, args.store, password, partitioned=True)
            del payload
        while not source_marker.exists():
            if (args.store.parent / 'stop.request').exists(): break
            if not args.once and implementation_signature() != loaded_implementation:
                command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--source', str(args.source),
                           '--store', str(args.store), '--run-dir', str(run_dir), '--stdin-password',
                           '--interval', str(args.interval)]
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                process.stdin.write((password + '\n').encode('utf-8')); process.stdin.close()
                update_status(args.store.parent / 'watcher.status.json', 'restarting', pid=process.pid)
                return 0
            fingerprint = source_fingerprint(args.source)
            checkpoint = args.store.parent / 'maintenance-checkpoint.json'
            previous = json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.is_file() else {}
            refresh = (args.once or fingerprint != previous.get('source_fingerprint') or
                       (previous.get('retryable_gaps', 0) and time.time() - previous.get('checked_at_epoch', 0) >= 3600))
            try:
                update_status(args.store.parent / 'watcher.status.json', 'updating' if refresh else 'monitoring', pid=os.getpid())
                if refresh:
                    summary = cycle(args.source, args.store, password, run_dir)
                    retryable = sum(summary['coverage']['content_gaps'].get(name, 0) for name in
                                    ('read_failed', 'extraction_failed', 'media_transcription_failed',
                                     'source_changed_since_extraction', 'archive_index_failed', 'archive_member_read_failed'))
                    retryable += summary.get('notebook', {}).get('failed_pages', 0)
                    atomic_write(checkpoint, json.dumps({'source_fingerprint': fingerprint,
                                 'checked_at_epoch': time.time(), 'retryable_gaps': retryable,
                                 'verified_at': now()}, indent=2).encode('utf-8'))
            except MaintenanceStopped:
                update_status(run_dir / 'run.status.json', 'stopped')
                return 0
            except Exception as error:
                update_status(run_dir / 'run.status.json', 'retrying', error_type=type(error).__name__)
                update_status(args.source / 'job-status.json', 'retrying', error_type=type(error).__name__)
                if args.once: raise
            if args.once: break
            for _ in range(args.interval):
                if source_marker.exists(): break
                if _ % 30 == 0:
                    update_status(args.store.parent / 'watcher.status.json', 'running', pid=os.getpid())
                time.sleep(1)
        if args.resume and args.once:
            command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--source', str(args.source),
                       '--store', str(args.store), '--run-dir', str(run_dir / 'background'), '--stdin-password']
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            process.stdin.write((password + '\n').encode('utf-8'))
            process.stdin.close()
            update_status(args.store.parent / 'watcher.status.json', 'running', pid=process.pid,
                          last_import=json.loads(args.store.with_suffix('.status.json').read_text())['updated_at'])
        return 0
    except Exception as error:
        frames = [{'file': Path(f.filename).name, 'line': f.lineno, 'function': f.name}
                  for f in traceback.extract_tb(error.__traceback__)]
        save_store(run_dir / 'runtime-diagnostic.enc', password,
                   {'version': 1, 'traceback': traceback.format_exc()})
        update_status(run_dir / 'run.status.json', 'failed', error_type=type(error).__name__, frames=frames)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
