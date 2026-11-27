import io
import json
import os
import sys
import time

_PROTO = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n")
sys.stdout = sys.stderr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mr_daemon  # noqa: E402

PATHS = mr_daemon.paths(os.environ)
READY_SEC = float(os.environ.get("MR_SHARED_READY_SEC", "90"))


def _ping(state):
    try:
        r = mr_daemon.request(state, {"op": "ping"}, timeout=5)
    except (OSError, ValueError):
        return None
    return r if r.get("ok") else None


def ensure():
    code = mr_daemon.code_fingerprint()
    state = mr_daemon.read_state(PATHS)
    if state:
        r = _ping(state)
        if r and r.get("code") == code:
            return state
        if r:
            print("[shim] daemon pid=%s runs old code, restarting" % r.get("pid"), file=sys.stderr)
            try:
                mr_daemon.request(state, {"op": "stop"}, timeout=600)
            except (OSError, ValueError):
                pass
    deadline = time.time() + READY_SEC
    next_spawn = 0.0
    while time.time() < deadline:
        if time.time() >= next_spawn:
            mr_daemon.spawn(dict(os.environ))
            next_spawn = time.time() + 20
        time.sleep(0.3)
        state = mr_daemon.read_state(PATHS)
        if state:
            r = _ping(state)
            if r and r.get("code") == code:
                return state
    raise RuntimeError("shared daemon not ready after %ds, log: %s" % (READY_SEC, PATHS["log"]))


def forward(payload):
    err = None
    for _ in range(2):
        state = ensure()
        try:
            r = mr_daemon.request(state, payload, timeout=None)
        except (OSError, ValueError) as e:
            err = e
            continue
        if not r.get("ok"):
            raise RuntimeError(r.get("error") or "daemon error")
        return r
    raise RuntimeError("shared daemon unreachable: %s: %s" % (type(err).__name__, err))


def reply(msg_id, result=None, error=None):
    m = {"jsonrpc": "2.0", "id": msg_id}
    if error:
        m["error"] = error
    else:
        m["result"] = result
    _PROTO.write(json.dumps(m, ensure_ascii=False) + "\n")
    _PROTO.flush()


def main():
    for line in io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid, method, params = req.get("id"), req.get("method", ""), req.get("params") or {}
        if method == "initialize":
            reply(mid, {"protocolVersion": params.get("protocolVersion", "2024-11-05"),
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "rh-memory", "version": "0.2-shared"}})
        elif method == "tools/list":
            try:
                reply(mid, {"tools": forward({"op": "list"})["tools"]})
            except Exception as e:
                reply(mid, error={"code": -32603, "message": "rh-memory shared daemon: %s" % e})
        elif method == "tools/call":
            try:
                r = forward({"op": "call", "name": params.get("name"),
                             "arguments": params.get("arguments") or {}})["result"]
            except Exception as e:
                r = {"ok": False, "error": "rh-memory shared daemon: %s" % e}
            reply(mid, {"content": [{"type": "text", "text": json.dumps(r, ensure_ascii=False, indent=1)}],
                        "isError": not r.get("ok", False)})
        elif mid is not None:
            reply(mid, error={"code": -32601, "message": "method not found: %s" % method})


if __name__ == "__main__":
    main()
