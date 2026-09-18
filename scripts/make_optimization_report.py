"""Render this optimization run with separate quality, latency, holdout and delivery states."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from scripts.make_report import delivery_state,require_committed,remote_state,remote_delivery_text


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir",type=Path,required=True)
    args=ap.parse_args()
    run=args.run_dir.resolve()
    read=lambda name:json.loads((run/name).read_text(encoding="utf-8"))
    quality,mcp,unit,blind,chunk,old_time=(read(n) for n in (
        "optimized-results.private.json","mcp-results.json","unit-results.json","blind-results.private.json",
        "chunk-results.private.json","baseline-latency.json"))
    if not (quality["passed"] and mcp["passed"] and unit["passed"] and read("frozen-source-integrity.json")["passed"]):
        raise RuntimeError("Functional verification or frozen corpus integrity failed")
    for artifact in (quality,mcp,blind):
        for name,expected in artifact["code_sha256"].items():
            if hashlib.sha256((CODE/name).read_bytes()).hexdigest()!=expected:
                raise RuntimeError("Delivered code differs from verified code: "+name)
    for name,expected in unit["files_sha256"].items():
        if hashlib.sha256((CODE/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError("Unit-tested file changed: "+name)
    protected=read("baseline-integrity.json")
    original={name:hashlib.sha256((CODE/"golden"/name).read_bytes()).hexdigest()==expected
              for name,expected in protected["golden_sha256"].items()}
    if not all(original.values()):raise RuntimeError("An original golden changed in the delivery tree")
    state=delivery_state(CODE)
    require_committed(state)
    state["remote"]=remote_state(CODE,state["head"])
    commits=subprocess.run(["git","-C",str(CODE),"log","--format=%h %s",protected["head"]+"..HEAD"],
                           capture_output=True,text=True,encoding="utf-8",check=True).stdout.strip()
    (run/"delivery.json").write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
    (run/"integrity-results.json").write_text(json.dumps({"original_goldens_unchanged":original,
        "original_cases":59,"stock_draft":True,"worktree_line_endings_explained":True},indent=2),encoding="utf-8")
    metrics=[]
    for name,label in (("notes_hybrid","普通 hybrid"),("notes_bm25","普通 BM25"),("notes_vec","普通纯向量"),
                       ("stock_hybrid","股票 hybrid（draft）")):
        before=chunk["runs"]["character"][name]["summary"]
        after=quality["runs"][name]["summary"]
        metrics.append("| %s | %d/%d → %d/%d | %.6f → %.6f |" % (label,
            before["returned_hit_at5"],before["positives"],after["returned_hit_at5"],after["positives"],
            before["returned_mrr_at10"],after["returned_mrr_at10"]))
    baseline_times=sorted(r["total_ms"] for r in old_time["samples"])
    old50=statistics.median(baseline_times)/1000
    old95=baseline_times[math.ceil(.95*len(baseline_times))-1]/1000
    latency=mcp["latency"]
    bp,apass=blind["baseline"]["summary"],blind["summary"]
    bnf=apass["returned_hit_at5"]>=bp["returned_hit_at5"]
    bne=apass["evidence_hit_at5"]>=bp["evidence_hit_at5"]
    blind_status="本次三题配对计数未退步" if bnf and bne else "本次三题出现退步，不能宣称泛化改善"
    corpus_text="本轮各次 MCP 搜索的语料 SHA 保持一致。" if mcp["corpus_stable_during_mcp"] else "本轮 MCP 期间语料 SHA 发生变化。"
    evidence=quality["runs"]["evidence_gate"]["summary"]
    negative=quality["runs"]["negative_gate"]["summary"]
    body=f"""# memory_rag 本轮优化与交付复核

**交付：本地已提交，工作树干净。** 分支 `{state['branch']}`，HEAD `{state['head']}`，核对时间 `{state['observed_at']}`。

{remote_delivery_text(state['remote'])}

**性能：{latency['budget_status']}。** 暖查询中位耗时明显下降，经验 p95 仍须按本批观测与预算比较，不能外推服务保证。

**质量边界：找对资料不等于答对。** 普通原题仍是文件级标签；新增私有标签覆盖 23/23，但实际预览带齐必需事实只有 {evidence['evidence_hit_at5']}/23。没有测最终答案正确率，也没有测调用 agent 在 low 时是否真的拒答。

## 大白话：具体改了什么

1. **资料尽量按完整段落切。** 超过上限再按句子/行处理，极长单句才硬切；保留有上限的整段重叠。普通笔记使用新切法，股票/PDF 保持旧边界。
2. **打分先少看，证据弱时补看。** 普通笔记先评前三个候选，弱到可能误拒时补到原前五个；每块两窗、每窗最多 256 token 输入。不能因用户只要 topk=2，就用没返回的第四条抬高置信。PDF/股票保留 512 token、五候选、三窗；两种预算共享权重，切换时显式恢复输入上限。
3. **检查从“书名对了”加严到“预览里真有必需事实”。** 原 59 题及原八项专项检查不改；另外核验全部 23 条普通正例。需要多段共同作答的题必须各组证据都出现。第 11 名以后不再给 MRR@10 加分。
4. **新复盘进入索引时就报警。** 已知问句出现在未标 evaluation 的块时，热更新告警；检查故障也明确显示 unavailable。报警不自动删材料或改排序，逐字引用也可能是正常知识，仍需复核。改写问题仍可能漏检。
5. **把挖题、验收和交付分清。** 新增不先检索的挖题入口；封存题走一次性配对入口。报告记录当前远端 SHA/公开性、独立的 MCP 语料身份与实际提交状态。

## 同一冻结资料上的回归结果

基线为 `{protected['head']}`，即上一轮四项修复已经存在的版本。两版读取相同冻结 Markdown 源文档，各自按自己的规则切块。原文共 1323 个文件；文本指纹按读取后的 UTF-8/统一换行计算。Git checkout 的 CRLF/LF 差异另行核对，主仓原题文件仍保持原始字节 SHA。

| 配置 | 实际返回前五命中 | MRR@10 |
|---|---:|---:|
{chr(10).join(metrics)}
| PDF bge-m3 | 12/12 → 12/12 | 0.916667 → 0.916667 |

新增 23 题严格预览标签：9/23 → {evidence['evidence_hit_at5']}/23；首条证据命中 7/23 → {evidence['evidence_hit_at1']}/23。旧四条段落题仍为 4/4。新标签由同一 agent 在检索实验前按原文核验，尚未经独立人工裁决。

base gate：普通正例 22 high / 1 medium / 0 low；阴性 {negative['negative_confidence']}，high 为 {negative['negative_high']}/12。未开 cross-encoder 的旧分数门仍有 3/12 阴性 high，不能混写成整套系统已经会拒答。

## 没有隐藏的退步与被否掉的方案

- **纯 BM25 文件命中 21/23 → 19/23。** 强行按文件去重可升到 22/23，但严格证据从 9/23 降到 6/23；每文件最多两块也会丢证据，均未采用。默认 hybrid 的收益不代表每种模式都改善；向量不可用时的 BM25 回退也受此限制。
- **静态 256/3/2 会误标一道正例为 low。** 没有为这道题下调 lo，而是加入通用、有限的补查后重新验证。
- **把短输入直接套到 PDF 会新增一道 low 正例。** 已撤回跨来源应用；PDF 的 high/medium/low 分布恢复为旧版 10/1/1。这也说明原来的页命中 12/12 不能证明置信或回答全部正确。
- **股票完全独立 IDF 再次退步：hybrid 11/12 → 10/12。** 保留混合统计，记录“hybrid 排序劣于 BM25”这个重测触发条件；股票依旧 draft。BM25 MRR 0.875000 与 hybrid 0.756944 的差距未被掩盖。
- **小 MRR 差不代表微调上限。** 当前普通集向量主要补覆盖；不能用本轮两条腿的差值推断 embedding 微调理论上限。

## 速度：真实调用，单列预算

基线和新版使用相同 12 题（八正四阴）各三次。基线为直接 Python 调用，新版为真实 stdio MCP，含协议开销；同一台 CPU，串行运行，无结果或 pair-score 缓存。参数筛选阶段另做过交错计时，质量脚本的缓存时间没有拿来冒充速度成绩。机器并非独占，重复问题也不是独立的随机用户样本。

| 观测 | 旧版 36 次 | 新版 36 次 |
|---|---:|---:|
| 暖查询 p50 | {old50:.3f} 秒 | {latency['warm_p50_ms']/1000:.3f} 秒 |
| 经验 p95（nearest-rank） | {old95:.3f} 秒 | {latency['warm_p95_ms']/1000:.3f} 秒 |

本次新 stdio 进程到第一次完整默认 hybrid + base gate 返回：**{latency['first_search_ms']/1000:.3f} 秒**，包含服务启动、预热和已装载的 PDF 源；此前那条“首个股票 BM25、rerank=false”的数字不是同一口径。冷启还受操作系统文件缓存与索引缓存影响。

预算：暖 p50 ≤2 秒、经验 p95 ≤4 秒、新进程首查 ≤60 秒。本次状态：**{latency['budget_status']}**。不因功能检查通过而把性能待办抹掉。

三参数八组比较后，最终普通笔记预算单独作一次粗网格标定，再复跑正例和阴性。`hi=0.5 / lo=0.1` 仍满足约束，因此数值沿用；这有新分布的记录支持，不是跳过标定。新增补查后的实际 MCP 速度单独测量，未使用静态三候选的快成绩替代。

{corpus_text} 每次调用均留有记录；这个观察只覆盖本轮 MCP，不能替过去另一时段的调用作保证。

## 封存后的三题：一次配对观察

从未检索的 transcript 候选中按原文核验，剔除已有题、重复来源/事件组与把问句直接抄进原文的候选。数据与参考集 SHA 在调参前封存；新旧两个固定版本在同一个被消耗的试验中各跑一次，之后不再调参。

| 指标 | 旧版 | 新版 |
|---|---:|---:|
| 文件 hit@5 | {bp['returned_hit_at5']}/3 | {apass['returned_hit_at5']}/3 |
| 严格预览 evidence@5 | {bp['evidence_hit_at5']}/3 | {apass['evidence_hit_at5']}/3 |
| MRR@10 | {bp['returned_mrr_at10']:.6f} | {apass['returned_mrr_at10']:.6f} |

{blind_status}。两版三题均标 high，不能据此宣称答案可靠。只有三道正例，由同一 agent 筛选并标注，只能叫小规模结果留出；不是第三方真盲测，也没有覆盖独立阴性或最终答案。看过结果后只可用于回归。通用 `mr.py eval` 不管理封存生命周期，正式留出试验须走 `sealed_eval.py`；消耗标记与原始结果留存备查。

## 提交与剩余风险

单测 {unit['tests_run']}/{unit['tests_run']}，质量/完整性检查 {len(quality['checks'])}/{len(quality['checks'])}，真实 MCP {len(mcp['checks'])}/{len(mcp['checks'])}。源码 SHA 与实际交付文件一致，原题 SHA 保持，股票仍为 draft。

```text
{commits}
```

已有公开历史包含内部来源文件名/事件描述，这项债仍在；本轮新增完整段落标签、封存题和原始检索结果只留私有忽略目录，没有改写公开历史或调整仓库可见性。现有 MCP 进程需要重连才能加载新源码；语料热更不等于代码热更。

仍要补：预览里的缺失事实、最终答案/引用/时效和真正拒答的评测、更大的独立留出集、p95 预算，以及纯 BM25 的退步。MQE/HyDE、recency 和微调尚未实施；完整 heading_path 已有负结果，本轮未重做。

## 本机原始记录

- [回归与统一标定](optimized-results.private.json)
- [切块消融](chunk-results.private.json) / [八组静态参数](gate-results.private.json)
- [参数交错计时](candidate-latency.json) / [旧版 36 次](baseline-latency.json)
- [真实 MCP 36 次与协议检查](mcp-results.json)
- [单次配对留出](blind-results.private.json) / [封存记录](blind-seal.private.json)
- [单测](unit-results.json) / [交付](delivery.json) / [原题完整性](integrity-results.json)
- [冻结文本完整性](frozen-source-integrity.json) / [依赖版本](dependencies.json)

这些记录中含私有问题与资料预览，不应提交到公开仓库。
"""
    output=run/"implementation-report.zh.md"
    output.write_text(body,encoding="utf-8")
    print(json.dumps({"report":str(output),"head":state["head"],"clean":state["worktree_clean"],
                      "remote":state["remote"],"blind_nonregression":bnf and bne},ensure_ascii=False))


if __name__=="__main__":main()
