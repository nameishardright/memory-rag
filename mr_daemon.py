import hashlib
import json
import msvcrt
import os
import queue
import secrets
import socket
import socketserver
import subprocess
import sys
import threading
import time

BASE = os.path.dirname(os.path.abspath(__file__))
ENV_PREFIXES = ("MR_", "HF_", "TRANSFORMERS_", "TOKENIZERS_")
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def env_key(env):
    items = sorted((k, v) for k, v in env.items() if k.upper().startswith(ENV_PREFIXES))
    raw = json.dumps([os.path.normcase(BASE), items], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def state_dir(env):
    return env.get("MR_SHARED_DIR") or os.path.join(
        env.get("LOCALAPPDATA") or os.path.expanduser("~"), "memory-rag", "shared")


def paths(env):
    d = state_dir(env)
    key = env_key(env)
    return {"dir": d, "key": key,
            "lock": os.path.join(d, key + ".lock"),
            "state": os.path.join(d, key + ".json"),
            "log": os.path.join(d, key + ".log")}


def code_fingerprint():
    h = hashlib.sha256()
    names = sorted(n for n in os.listdir(BASE) if n.endswith(".py") or n == "corpus-policy.json")
    for n in names:
        st = os.stat(os.path.join(BASE, n))
        h.update(("%s|%d|%d\n" % (n, st.st_size, st.st_mtime_ns)).encode("utf-8"))
    return h.hexdigest()[:16]


def read_state(p):
    try:
        with open(p["state"], encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def request(state, payload, timeout=None):
    body = json.dumps(dict(payload, token=state["token"]), ensure_ascii=False) + "\n"
    with socket.create_connection(("127.0.0.1", int(state["port"])), timeout=5) as s:
        s.settimeout(timeout)
        s.sendall(body.encode("utf-8"))
        with s.makefile("rb") as f:
            line = f.readline()
    if not line:
        raise ConnectionError("shared daemon closed the connection")
    return json.loads(line.decode("utf-8"))


def spawn(env):
    exe = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    if not os.path.exists(exe):
        exe = sys.executable
    flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    cmd = [exe, "-X", "utf8", os.path.abspath(__file__), "--detach"]
    kw = dict(cwd=BASE, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
              stderr=subprocess.DEVNULL, close_fds=True)
    try:
        proc = subprocess.Popen(cmd, creationflags=flags | CREATE_BREAKAWAY_FROM_JOB, **kw)
    except OSError:
        proc = subprocess.Popen(cmd, creationflags=flags, **kw)
    proc.wait(timeout=30)


def _detach():
    p = paths(os.environ)
    os.makedirs(p["dir"], exist_ok=True)
    flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    with open(p["log"], "ab") as log:
        kw = dict(cwd=BASE, stdin=subprocess.DEVNULL, stdout=log, stderr=log, close_fds=True)
        cmd = [sys.executable, "-X", "utf8", os.path.abspath(__file__)]
        try:
            subprocess.Popen(cmd, creationflags=flags | CREATE_BREAKAWAY_FROM_JOB, **kw)
        except OSError:
            subprocess.Popen(cmd, creationflags=flags, **kw)


def _lock(p, wait_sec):
    os.makedirs(p["dir"], exist_ok=True)
    f = open(p["lock"], "a+b")
    deadline = time.time() + wait_sec
    while True:
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            return f
        except OSError:
            if time.time() >= deadline:
                f.close()
                return None
            time.sleep(0.5)


def _write_state(p, state):
    tmp = p["state"] + ".%d.tmp" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, p["state"])


def _drop_state(p):
    st = read_state(p)
    if st and st.get("pid") == os.getpid():
        try:
            os.remove(p["state"])
        except OSError:
            pass


def _warm(mr_tools):
    mr_tools._ensure("hybrid")
    if os.environ.get("MR_WARM_HEAVY"):
        pdf_model = os.environ.get("MR_EMBED_MODEL_PDF")
        if pdf_model:
            mr_tools._ensure("hybrid", model=pdf_model)
        if os.environ.get("MR_RERANK"):
            rk = mr_tools._get_reranker(None)
            if rk and rk.available:
                rk.score("预热", ["预热"])


def _log(msg):
    print("[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg), file=sys.stderr, flush=True)


def serve():
    p = paths(os.environ)
    lock = _lock(p, 15)
    if lock is None:
        return 0
    sys.path.insert(0, BASE)
    import mr_tools
    mr_tools.LOG_SRC = "mcp"
    token = secrets.token_hex(16)
    code = code_fingerprint()
    jobs = queue.Queue()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                req = json.loads(self.rfile.readline().decode("utf-8"))
            except (OSError, ValueError):
                return
            if not isinstance(req, dict) or req.get("token") != token:
                resp = {"ok": False, "error": "bad token"}
            elif req.get("op") == "ping":
                resp = {"ok": True, "pid": os.getpid(), "code": code}
            else:
                box, done = {}, threading.Event()
                jobs.put((req, box, done))
                done.wait()
                resp = box.get("resp") or {"ok": False, "error": "no response"}
            try:
                self.wfile.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
            except OSError:
                pass

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = False

    server = Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _write_state(p, {"pid": os.getpid(), "port": server.server_address[1], "token": token,
                     "code": code, "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    _log("started pid=%d port=%d key=%s code=%s" % (os.getpid(), server.server_address[1], p["key"], code))
    if os.environ.get("MR_WARM"):
        jobs.put(({"op": "warm"}, {}, threading.Event()))
    try:
        while True:
            req, box, done = jobs.get()
            op = req.get("op")
            try:
                if op == "list":
                    box["resp"] = {"ok": True, "tools": [
                        {"name": n, "description": t["desc"], "inputSchema": t["schema"]}
                        for n, t in mr_tools.REGISTRY.items()]}
                elif op == "call":
                    t0 = time.perf_counter()
                    result = mr_tools.call(req.get("name"), req.get("arguments") or {})
                    box["resp"] = {"ok": True, "result": result}
                    _log("call %s %.1fs ok=%s" % (req.get("name"), time.perf_counter() - t0, result.get("ok")))
                elif op == "warm":
                    _warm(mr_tools)
                    _log("warmed")
                elif op == "stop":
                    box["resp"] = {"ok": True, "pid": os.getpid()}
                    _log("stop requested")
                    break
                else:
                    box["resp"] = {"ok": False, "error": "unknown op %s" % op}
            except Exception as e:
                box["resp"] = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
                _log("op %s failed: %s" % (op, box["resp"]["error"]))
            finally:
                done.set()
    finally:
        _drop_state(p)
        time.sleep(0.3)
        server.server_close()
        lock.close()
    return 0


if __name__ == "__main__":
    if "--detach" in sys.argv:
        _detach()
    else:
        sys.exit(serve())
