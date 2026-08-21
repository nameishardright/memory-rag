# -*- coding: utf-8 -*-
"""MCP stdio server(零 SDK,照 taskid ta_mcp_server 模式):换行分隔 JSON-RPC 2.0,三个方法。
铁纪律:stdout 是协议信道——进程一起来就把 sys.stdout 换成 stderr,协议走专用 writer,
库里任何手滑 print 都打不脏 JSON-RPC 流。
注册: claude mcp add -s user rh-memory -- py -3.12 <本文件绝对路径>
"""
import io
import json
import os
import sys

_PROTO = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n")
sys.stdout = sys.stderr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mr_tools  # noqa: E402

# 预热策略(分腿计时:冷启动 35.5s 里 88% 是 import torch,模型加载才 0.1s):
# ⚠️后台线程 import torch 在 Windows 慢一倍(实测 ~70s)——所以在【主线程】预热,
# 时机=答完 tools/list 之后(initialize/tools/list 是连接秒发的,必须秒回;tools/call
# 通常几分钟后才来)。预热期间来的消息在管道里排队,最坏等个尾巴,通常等于免费。
_WARMED = [False]


def _warm_after_list():
    if not _WARMED[0]:
        _WARMED[0] = True
        try:
            mr_tools._ensure("hybrid")
        except Exception as e:
            print("[warm] 预热失败,首查自己再初始化:", e, file=sys.stderr)


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
                        "serverInfo": {"name": "rh-memory", "version": "0.1"}})
        elif method == "tools/list":
            reply(mid, {"tools": [{"name": n, "description": t["desc"], "inputSchema": t["schema"]}
                                  for n, t in mr_tools.REGISTRY.items()]})
            _warm_after_list()   # 先回包再暖:客户端拿到工具表,我们用空闲期付 torch 税
        elif method == "tools/call":
            r = mr_tools.call(params.get("name"), params.get("arguments") or {})
            reply(mid, {"content": [{"type": "text", "text": json.dumps(r, ensure_ascii=False, indent=1)}],
                        "isError": not r.get("ok", False)})
        elif mid is not None:      # 请求必须应答;通知(无 id,如 notifications/initialized)静默忽略
            reply(mid, error={"code": -32601, "message": "method not found: %s" % method})


if __name__ == "__main__":
    main()
