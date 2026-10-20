"""Transfer a locally held update password to a verified fresh controller."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def handoff_if_requested(password, source, drive):
    source, drive = Path(source).resolve(), Path(drive).resolve()
    marker = source / 'rag-restart.request.json'
    if not marker.is_file(): return False
    request = json.loads(marker.read_text(encoding='utf-8'))
    parent_pid = os.getppid()
    if request['parent_pid'] != parent_pid: raise ValueError('Restart parent does not match')
    controller = Path(request['controller']).resolve()
    allowed = {Path.home() / 'projects' / 'memory-rag' / 'mr_private_maintenance.py',
               drive / '_Private-Memory-RAG' / 'app' / 'mr_private_maintenance.py'}
    if controller not in {p.resolve() for p in allowed}: raise ValueError('Untrusted restart controller')
    if hashlib.sha256(controller.read_bytes()).hexdigest() != request['controller_sha256']:
        raise ValueError('Restart controller hash mismatch')
    store, run_dir = Path(request['store']).resolve(), Path(request['run_dir']).resolve()
    if store != drive / '_Private-Memory-RAG' / 'corpus.enc': raise ValueError('Unexpected restart store')
    if not (run_dir.is_relative_to(source) or run_dir.is_relative_to(Path(r'D:\RAG-Optimization').resolve())):
        raise ValueError('Unexpected restart report directory')
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    lookup = ("Get-CimInstance Win32_Process -Filter 'ProcessId=" + str(parent_pid) + "' | "
              "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', lookup], capture_output=True,
                            text=True, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    parent = json.loads(result.stdout)
    command = parent.get('CommandLine') or ''
    if not all(value in command for value in ('mr_private_maintenance.py', str(source), str(store))):
        raise ValueError('Restart parent identity mismatch')
    process = subprocess.Popen([sys.executable, '-X', 'utf8', str(controller), '--source', str(source),
                               '--store', str(store), '--run-dir', str(run_dir), '--stdin-password'],
                               stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    process.stdin.write((password + '\n').encode('utf-8')); process.stdin.close()
    time.sleep(1)
    if process.poll() is not None: raise RuntimeError('Fresh update controller did not start')
    marker.unlink()
    stop = ("$ragParent=Get-CimInstance Win32_Process -Filter 'ProcessId=" + str(parent_pid) + "'; "
            "if($ragParent -and $ragParent.CommandLine -and $ragParent.CommandLine.Contains('mr_private_maintenance.py') "
            "-and $ragParent.CommandLine.Contains(" + quote(source) + ") -and $ragParent.CommandLine.Contains(" +
            quote(store) + ")) { Stop-Process -Id " + str(parent_pid) + " -Force }")
    subprocess.run(['powershell.exe', '-NoProfile', '-Command', stop], check=True,
                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    (run_dir / 'handoff.status.json').write_text(json.dumps({'state': 'restarted', 'previous_pid': parent_pid,
                                      'pid': process.pid, 'controller_sha256': request['controller_sha256']}), encoding='utf-8')
    return True
