"""Resume source collection and encrypted-index updates with one password entry."""
import argparse
import getpass
import json
from pathlib import Path
import subprocess
import sys

from mr_private import default_store, load_store


def running_processes(script, directory):
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    code = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" | "
            "Where-Object { $_.CommandLine -and $_.CommandLine -notmatch '\\s-c\\s' -and $_.CommandLine.Contains(" + quote(script) +
            ") -and $_.CommandLine.Contains(" + quote(directory) + ") } | "
            "Select-Object -ExpandProperty ProcessId")
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', code],
                            capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW, check=True)
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]


def start(command, password):
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    process.stdin.write((password + '\n').encode('utf-8'))
    process.stdin.close()
    return process.pid


def stop_updates(store, source):
    (store.parent / 'stop.request').touch()
    if source is None:
        return
    (source / 'stop.request').touch()
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    for script in ('mr_private_maintenance.py', 'finish_memory.py', 'build_memory.py'):
        for pid in running_processes(script, str(source)):
            code = ("$ragTask=Get-CimInstance Win32_Process -Filter 'ProcessId=" + str(pid) + "'; "
                    "if($ragTask -and $ragTask.CommandLine -and $ragTask.CommandLine.Contains(" + quote(script) +
                    ") -and $ragTask.CommandLine.Contains(" + quote(source) + ")) { Stop-Process -Id " + str(pid) + " -Force }")
            subprocess.run(['powershell.exe', '-NoProfile', '-Command', code], check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=Path, default=default_store())
    parser.add_argument('--stop', action='store_true')
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    if args.stop:
        stop_updates(args.store, args.source)
        print('Background updates stopped; encrypted files are retained.')
        return
    password = getpass.getpass('Archive password: ')
    payload = load_store(args.store, password)
    source = Path(payload['source'])
    controller = Path(__file__).with_name('mr_private_maintenance.py')
    launched = {}
    marker = args.store.parent / 'stop.request'
    if marker.is_file():
        marker.unlink()
    if controller.is_file() and (source / 'tools' / 'build_memory.py').is_file():
        marker = source / 'stop.request'
        if marker.is_file():
            marker.unlink()
        current = running_processes('mr_private_maintenance.py', str(source))
        if not current:
            launched['collection_pid'] = start([sys.executable, str(controller), '--source', str(source),
                                                '--store', str(args.store), '--stdin-password'], password)
        else:
            launched['collection_already_running'] = current
    else:
        print('The source collection tools are on the original computer; synced data remains searchable here.')
    print(json.dumps(launched))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Update control failed: ' + type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
