import ctypes
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import mr_daemon  # noqa: E402


def alive(pid):
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, int(pid))
    if not h:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return code.value == 259


class Shim:
    def __init__(self, env):
        self.proc = subprocess.Popen([sys.executable, "-X", "utf8", str(CODE / "mr_mcp_shim.py")], cwd=CODE, env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, encoding="utf-8")
        self.lines = queue.Queue()
        self.next_id = 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def request(self, method, params=None):
        self.next_id += 1
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.next_id, "method": method,
                                          "params": params or {}}, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.lines.get(timeout=120)
        assert line is not None, "shim closed stdout"
        reply = json.loads(line)
        assert reply["id"] == self.next_id
        return reply


@unittest.skipUnless(os.name == "nt", "Windows-only daemon")
class SharedDaemonTests(unittest.TestCase):
    def setUp(self):
        self.dir = self.enterContext(tempfile.TemporaryDirectory())
        self.env = {k: v for k, v in os.environ.items() if k not in ("MR_WARM", "MR_WARM_HEAVY")}
        self.env.update(MR_SHARED_DIR=self.dir, MR_LOG="0", MR_SHARED_READY_SEC="60")
        self.paths = mr_daemon.paths(self.env)
        self.shims = []
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for s in self.shims:
            if s.proc.poll() is None:
                s.proc.kill()
                s.proc.wait(timeout=10)
            s.proc.stdin.close()
            s.proc.stdout.close()
        state = mr_daemon.read_state(self.paths)
        if state:
            try:
                mr_daemon.request(state, {"op": "stop"}, timeout=30)
            except (OSError, ValueError):
                pass
            deadline = time.time() + 15
            while alive(state["pid"]) and time.time() < deadline:
                time.sleep(0.2)

    def shim(self):
        s = Shim(self.env)
        self.shims.append(s)
        return s

    def test_sessions_share_one_daemon_that_outlives_them(self):
        first = self.shim()
        self.assertEqual(first.request("initialize")["result"]["serverInfo"]["name"], "rh-memory")
        names = {t["name"] for t in first.request("tools/list")["result"]["tools"]}
        self.assertTrue({"memory_search", "memory_get", "memory_check_answer"} <= names)
        pid = mr_daemon.read_state(self.paths)["pid"]
        self.assertNotEqual(pid, first.proc.pid)

        second = self.shim()
        second.request("initialize")
        second.request("tools/list")
        self.assertEqual(mr_daemon.read_state(self.paths)["pid"], pid)

        called = second.request("tools/call", {"name": "no_such_tool", "arguments": {}})["result"]
        self.assertTrue(called["isError"])
        self.assertIn("no_such_tool", json.loads(called["content"][0]["text"])["error"])

        subprocess.run(["taskkill", "/T", "/F", "/PID", str(first.proc.pid)], capture_output=True)
        first.proc.wait(timeout=10)
        time.sleep(0.5)
        self.assertTrue(alive(pid))
        self.assertEqual(second.request("tools/list")["result"]["tools"][0]["name"], "memory_search")

    def test_stop_releases_state_and_next_session_respawns(self):
        s = self.shim()
        s.request("tools/list")
        state = mr_daemon.read_state(self.paths)
        mr_daemon.request(state, {"op": "stop"}, timeout=30)
        deadline = time.time() + 15
        while alive(state["pid"]) and time.time() < deadline:
            time.sleep(0.2)
        self.assertFalse(alive(state["pid"]))
        self.assertIsNone(mr_daemon.read_state(self.paths))
        s.request("tools/list")
        again = mr_daemon.read_state(self.paths)
        self.assertNotEqual(again["pid"], state["pid"])

    def test_bad_token_is_rejected(self):
        self.shim().request("tools/list")
        state = dict(mr_daemon.read_state(self.paths), token="wrong")
        self.assertEqual(mr_daemon.request(state, {"op": "list"}, timeout=10)["error"], "bad token")

    def test_env_changes_key(self):
        self.assertNotEqual(mr_daemon.env_key({"MR_PDF": "1"}), mr_daemon.env_key({"MR_PDF": "0"}))
        self.assertEqual(mr_daemon.env_key({"MR_PDF": "1", "PATH": "a"}), mr_daemon.env_key({"MR_PDF": "1", "PATH": "b"}))


if __name__ == "__main__":
    unittest.main()
