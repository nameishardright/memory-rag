"""Render a delivery report only for passing results and a clean committed tree."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

CODE = Path(__file__).resolve().parents[1]


def corpus_change_text(unchanged):
    if type(unchanged) is not bool:
        raise ValueError("The live corpus change observation must be an explicit boolean")
    if unchanged:
        return "运行期间实时语料保持一致。"
    return "运行期间实时语料发生了变化。新旧算法使用相同的冻结快照，配对结论仍针对该快照成立。"


def delivery_state(repo):
    def git(*args, check=True):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", check=check)
    status = git("status", "--porcelain=v1", "--untracked-files=normal").stdout.strip()
    head = git("rev-parse", "--verify", "HEAD", check=False)
    return {"head": head.stdout.strip() if head.returncode == 0 else None,
            "branch": git("branch", "--show-current").stdout.strip(),
            "worktree_clean": not status, "status": status.splitlines(),
            "observed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def require_committed(state):
    if not state["head"] or not state["worktree_clean"]:
        raise RuntimeError("DELIVERY_NOT_COMMITTED: 交付报告要求已有提交且工作树干净；请先处理明确范围内的改动")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = (args.output or run / "implementation-report.zh.md").resolve()
    if output.is_relative_to(CODE):
        ignored = subprocess.run(["git", "-C", str(CODE), "check-ignore", "-q", "--", str(output.relative_to(CODE))])
        if ignored.returncode:
            raise ValueError("Write the report outside the tracked tree or in an ignored output directory")
    read = lambda name: json.loads((run / name).read_text(encoding="utf-8"))
    result, mcp, unit, integrity = (read(name) for name in (
        "verified-results.json", "mcp-results.json", "unit-results.json", "integrity-results.json"))
    if not (result["passed"] and mcp["passed"] and unit["passed"] and all(integrity["original_goldens_unchanged"].values())):
        raise RuntimeError("Verification or original-golden integrity check did not pass")
    for relative, digest in result["current_sha256"].items():
        path = (CODE / relative).resolve()
        if not path.is_relative_to(CODE) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Verified source changed: " + relative)
    for relative, digest in unit.get("files_sha256", {}).items():
        if hashlib.sha256((CODE / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Unit-tested file changed: " + relative)
    delivery = delivery_state(CODE)
    require_committed(delivery)
    drift = corpus_change_text(result["live_corpus_unchanged_during_run"])
    s = lambda version, suite: result["runs"][version][suite]["summary"]
    rows = []
    for label, suite in (("普通 hybrid", "notes_hybrid"), ("普通 BM25", "notes_bm25"), ("普通纯向量", "notes_vec"),
                         ("股票 hybrid（draft）", "stock_hybrid"), ("PDF bge-m3", "pdf_m3")):
        before, after = s("before", suite), s("after", suite)
        rows.append("| %s | %d/%d → %d/%d | %.3f → %.3f |" % (
            label, before["returned_hit_at5"], before["positives"], after["returned_hit_at5"], after["positives"],
            before["returned_mrr_at10"], after["returned_mrr_at10"]))
    before, after = s("before", "evidence_gate"), s("after", "evidence_gate")
    nb, na = s("before", "negative_gate"), s("after", "negative_gate")
    latency = mcp.get("latency", {})
    report = f"""# memory_rag 复核修正与交付

**交付状态：已提交，工作树干净。** 本地提交 `{delivery['head']}`，分支 `{delivery['branch']}`；状态核对时间 `{delivery['observed_at']}`。本报告记录本地交付，不代表远程发布。

**性能状态：{latency.get('budget_status', '未测，不作达标结论')}。** 功能通过与延迟预算分开报告。

单元与入口回归 {unit['tests_run']}/{unit['tests_run']}，配对检查 {len(result['checks'])}/{len(result['checks'])}，真实 MCP 检查 {len(mcp['checks'])}/{len(mcp['checks'])} 均通过。原有 59 题及其标签 SHA 保持一致，股票仍为 draft。比较基线是 2026-09-17 四项修复之前的源码，两版运行于本轮相同的冻结语料。

## 检索结果

| 配置 | 实际返回前五命中 | MRR |
|---|---:|---:|
{chr(10).join(rows)}

四个已标注答案段落的前五命中：{before['evidence_hit_at5']}/4 → {after['evidence_hit_at5']}/4。启用现有 base gate 时，12 道无答案题被标为 high：{nb['negative_high']}/12 → {na['negative_high']}/12。这是检索与置信标注评测，不是最终答案正确率；已观察过的题都是 regression，不是盲测。仅四条普通正例有严格段落标签，4/4 不能代表其余 19 条已通过段落验收。

## 复核项

- 上一轮改动已按运行时、评测模块与标签、测试与文档提交为三组；本轮复核修正另有提交。
- {drift} 实时 MCP 的调用验证另见运行记录。
- 评测前扫描全部已装载来源和本仓全部金标问句；未标 evaluation 的块出现原问句时，以非零状态终止，不再输出质量成绩。探测器不依赖文件名单，不自动改用途，也不宣称能发现改写过的问句。
- 本轮真实扫描曾拦住两篇漏标复盘及两页教材天气示例；核对正文后分别标注，教材其余页保留。已标材料仍可显式搜索或 memory_get 读取。
- memory_get 优先精确标题匹配，其次唯一前缀匹配；省略分片后缀后若仍有多个候选，会列出完整标题，不静默只取第一片。

## 耗时与剩余工作

MCP 新进程到首次股票 BM25 请求完成（rerank=false，包含服务预热）：{latency.get('first_search_ms', '未测')} ms。默认 hybrid 加首次 rerank 的完整冷启动尚未单独测量。暖查询使用 hybrid+base gate，样本数：{latency.get('warm_samples', '未测')}；样本 p50：{latency.get('warm_p50_ms', '未测')} ms，p95：{latency.get('warm_p95_ms', '未测')} ms。这些是本机有限样本；冷启动读缓存、机器负载会影响结果。质量对照中的缓存耗时不拿来替代这些实测。

暖查询目标预算为 p50 ≤ 2000 ms、p95 ≤ 4000 ms，新进程首查 ≤ 60000 ms。当前预算状态：{latency.get('budget_status', '未测，不作达标结论')}。本轮未修改模型截断长度、候选数或 hi/lo；预算没有达标时，性能优化仍是待办，不计入功能通过数。

股票词频耦合的重测触发条件、扩充段落标签、段落重叠、MQE/HyDE 缓存、盲测封存与禁止重做 heading_path 的记录，见仓库 [后续实验与验收条件]({(CODE / 'docs/next-experiments.zh.md').as_posix()})。

## 可核查记录

- 配对运行：`{result['run_at']}`；非 PDF 语料 SHA-256：`{result['corpus_sha256']}`。
- [配对逐题结果](verified-results.json)
- [真实 MCP 与耗时样本](mcp-results.json)
- [单测结果](unit-results.json)
- [原题完整性](integrity-results.json)
- [交付 Git 状态](delivery.json)
- [检索与标签用法]({(CODE / 'docs/retrieval-evaluation.zh.md').as_posix()})

已有 MCP 进程需重新连接才能加载本轮源码；本轮验证使用新启动的真实 stdio 子进程。
"""
    output.write_text(report, encoding="utf-8")
    (run / "delivery.json").write_text(json.dumps(delivery, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output), "head": delivery["head"], "worktree_clean": delivery["worktree_clean"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
