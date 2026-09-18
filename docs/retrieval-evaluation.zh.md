# 检索范围、段落证据与评测

2026-09-17 更新。默认检索仍为 `case/book/mem`，股票用 `source="stock"`，教材原文用 `source="pdf"`。普通笔记的 BM25 统计排除股票、PDF 和已标记的评测内容；股票保留原有的全部非 PDF 统计。`include_pdf=true` 只增加 PDF，不顺带打开股票来源。

## 查询和取段落

```python
memory_search(query="财报后高开回落的历史判断", source="stock")
memory_get(file="<搜索返回的 file>", source="stock")

memory_search(query="cherry-pick 直合")
memory_get(file="<搜索返回的 file>", source="mem", heading="<搜索返回的 heading>")
```

`heading` 先精确匹配，再接受唯一前缀；省略 `(1)` 后缀后若仍唯一，会返回实际完整标题。找不到或有多个候选时，错误中给出 `candidates` 与 `candidate_count`；不会静默只取第一片。省略 heading 可读全文。PDF 仍用 `file="书名.pdf#p81"` 取该页及相邻页。

Hybrid 先融合文件排名，再在同文件的两路候选中选择正文。只有索引行时保留定位结果并标记 `evidence_kind="navigation"`。启用 gate 且当前结果均未达到 high 时，只对评分低于原有相关性下限的代表块补评同文件的一段词法正文；替代段落达到原有下限才替换，文件排名不变。已经达到 medium 的答案段落不会被高分主题概述挤走。这避免文件对另一主题的投票把已有答案的置信标签拉成 low。`snippet` 仍是预览，操作前应读完整段落，涉及历史变更时再核对全文中的后续更正。

两个 MCP 工具均声明 `stock`。`rerank` 接受布尔值或预设名；默认 `MR_RERANK_MODE=gate` 保留文件顺序和 `score`，另给 `rerank` 分与置信标签。普通笔记的 base 预设改为 256 token、每块最多两窗：先评前三条，低于 lo 时最多补到原前五条，不超过实际 topk。`expanded_candidates` 记录补看的文件数；原来的同文件词法段落补评仍由 `extra_passages` 记录，`pairs` 是总评分窗口数。弱证据不能仅因缩小预算就被静默放弃。PDF、stock、混入 PDF 的搜索和 v2-m3 保留原来的 512 token、五候选、三窗口预设；短输入曾导致一条 PDF 正例从 high 降到 low，不能跨来源直接复用。预算变体共享模型权重，在锁内按次设置 token 限制，不额外复制一套模型。只有显式 `MR_RERANK_MODE=rerank` 才按交叉编码器分数重排。相关分、`confidence=high` 都不等于最终答案正确。

`max_length`、`initial_candidates`、`max_windows`、`expanded_candidates` 会随实际评分结果返回；hi/lo 在 `gate` 中。改变预算仍需重新标定。256 token 并不意味着 450 字窗口全部进入模型；它可能截掉后半段，必须用段落与阴性检查共同验收。

## 评测内容的用途标签

`corpus-policy.json` 保存明确的文档和段落用途。评测复盘用 `purpose="evaluation"`，默认在 BM25 和向量召回前排除；纯教学正文保持可检索。混合文档可用 `sections[].heading_prefix` 只标记运行报告部分，覆盖该标题的所有硬切分块。

PDF 的演示/评测页放在 `pdf_pages` 扩展中；`p2` 只匹配该页及其分片，不匹配 p20。旧进程忽略此扩展，不会因热读新配置而报来源错误；重新连接后才启用 PDF 页标签。

新评测笔记可在 frontmatter 顶层写入：

```yaml
---
rag_purpose: evaluation
---
```

支持 `knowledge`、`evaluation` 两个值；sidecar 文档标签覆盖 frontmatter，段落标签再覆盖文档默认值。自定义标签文件可用 `MR_CORPUS_POLICY` 指定。标签文件变化也会触发已有的语料热更新。

需要查询评测方法或复盘时明确包含它们：

```python
memory_search(query="分数门的阴性对照复盘", source="case", include_evaluation=True)
```

返回内容会显示 `purpose="evaluation"` 并提醒其样例问题未必有事实答案。`memory_get` 仍可主动读取这些文档。过滤只依赖用途标签，golden 的期望答案不参与过滤、排序或置信判定；尚未标记的新增评测材料仍需补标签。金标题目、答案标签和结果报告应保留在本仓 `golden/`、隔离的评测目录，不加入普通事实语料。包含私有来源信息的扩充标签存入忽略的 `golden/private/`。

**自动验收检查**：`mr.py eval` 在预热模型和输出质量成绩前，扫描全部已装载来源，并检查本仓 `golden/*.json` 以及本次额外题集。任何未标 `evaluation` 的块，只要文件名、标题或正文逐字包含金标原问句，就以非零状态退出，列出位置。`mr.py audit-corpus` 可单独运行，不加载向量或重排模型；`MR_PDF=1` 时也扫描 PDF。配对验证脚本对冻结的笔记/股票与 PDF 快照都执行此检查。

首次装载和语料热更新也运行同一检查，但只在 stderr 输出 `[corpus-audit]` 告警，不中断检索、不自动改用途、不打印问句内容。每次 search 返回 `corpus_sha256` 和 `corpus_audit`，审计失败为 `unavailable`，不能冒充 clean。新复盘的逐字污染可在下一次热更新时发现；改写与新问法仍可能漏检。正常答疑也可能逐字引用问题，告警需要核对而非直接定罪；用途标签仍需明确设置。

## 分开衡量文件命中与答案段落

`mr.py eval` 同时报告：

- `returned@5` / MRR：实际返回条目中的首次命中。BM25、vec 可能返回同文件的多个块，这些块仍占真实名额。
- `document@5` / MRR：在本次最多十条返回结果内，按来源和文件去重后的历史口径。PDF 还检查原有页码范围。
- `evidence@1` / `evidence@5`：来源、文件、标题及预览内的必需事实同时满足。只找对文件不能通过。
- 阴性题的 low / medium / high 分布，以及标注的困难负例位置。

原有 59 道题与原标签保持不变，包括仍标为草稿的 12 道股票题。新增 `golden/golden_evidence.json` 有八条专项检查：四条原正例的段落标注、两条原阴性题的困难负例，以及两个主动检索评测笔记的问题。它们不是八道全新独立测试题。

正例 `evidence` 是可接受段落的列表；每个段落必须给出 `source`、`file`、`heading` 或 `heading_contains`，以及 `required_text`。必需事实之间是 AND，同一事实的可接受字面写法可以用列表表示 OR。所有必需事实都必须出现在实际 `snippet` 中，不能靠读取未返回的正文替一次失败的预览补分。

本轮在独立私有文件中为普通 23 题全部补了来源核验标签。原主集不因此变成段落集，必须显式运行 `--golden golden/private/notes-evidence.json`。`evidence_groups` 表示问题需要多个证据组，每组内部是可接受替代段落；只有各组都在返回结果中找到，才算完整命中。三个 `*_mrr_at10` 均严格截断：第 11 名以后按零计，不能沿用无限深度倒数排名。

`hard_negatives` 记录“相关但不能回答”的位置和理由，例如“已推 change 直合”与“叠错分支后补救”的区别，以及提到天气测试却没有实时天气的数据。标签只供评测和后续人工样本整理，不参与运行时排序，也不会自动触发模型训练。

这些已观察过的案例标为 `split="regression"`。以后建立独立测试集时，必须在调优前按 `case_group` 留出，同一事件的切片、副本和问法改写放在同一组。评测器会拒绝输入中跨 split 的同一组；不要把本轮已经用来选方案的题改名为 holdout。段落字符串检查也不是最终答题模型的准确率评测。

## 验证命令

```powershell
py -3.12 -X utf8 -m unittest discover -s tests -v
py -3.12 -X utf8 mr.py audit-corpus
py -3.12 -X utf8 mr.py eval --mode hybrid -v
py -3.12 -X utf8 mr.py eval --mode hybrid --golden golden/golden_evidence.json --rerank bge-reranker-base -v
py -3.12 -X utf8 mr.py eval --mode hybrid --golden golden/golden_negative.json --rerank bge-reranker-base -v
```

`scripts/verify_retrieval.py` 接受修改前的源码快照目录，在同一份内存语料、相同本地模型和原标签上运行新旧代码。完整模式还验证股票各模式的排序、PDF bge-m3 路径与置信门；`--quick` 跳过 PDF 和交叉编码器，供开发迭代。报告只保存位置、分数、命中结果及片段哈希，不复制语料正文。

```powershell
py -3.12 -X utf8 scripts/verify_retrieval.py --baseline <修改前源码目录> --cache .cache/verification/embeddings --output .cache/verification/paired.json
py -3.12 -X utf8 scripts/verify_mcp.py --cache .cache/verification/mcp-embeddings --output .cache/verification/mcp.json
```

第二条启动真实 stdio MCP 子进程，验证 `initialize`、`tools/list`、股票搜索和取文、直合答案段落、天气/菜谱置信以及评测笔记的主动访问。只关闭测试自己启动的进程。

`MR_CACHE_DIR` 可隔离 embedding 和查询日志的写入，模型权重仍从原 `.cache/` 或 `MR_MODELS_DIR` 读取。验证脚本关闭查询日志和模型联网下载。两版共享本地模型和相同查询的结果缓存，因此这些输出用于质量对照，不作为性能基准。其他会话可能继续写笔记，报告会记录实时语料是否变化；配对比较始终使用同一份冻结语料。

需要生成交付报告时，先将单测记录为 JSON：

```powershell
py -3.12 -X utf8 scripts/verify_unit.py --output <本轮结果目录>/unit-results.json
py -3.12 -X utf8 scripts/make_report.py --run-dir <本轮结果目录>
```

结果目录还应包含 `verified-results.json`、`mcp-results.json` 和原题 SHA 对照 `integrity-results.json`。生成器核对通过状态、被测文件 SHA，以及真实 Git HEAD/工作树；没有提交或尚有未提交文件时拒绝生成“已交付”报告。输出应在仓外或被忽略的结果目录。实时语料变动直接写成“运行期间实时语料发生了变化”，不再打印双重否定布尔值。原 2026-09-17 报告保留其当时的数值，并补记对应已提交基线。

MCP 验证另外记录新进程首查和 12 题各三次的串行暖查询，并与预算比较。每次保存语料 SHA；`--frozen` 可指定 Markdown 快照，`--models` 指向已有权重目录。质量对照的稳定性结论不覆盖另一时段的 MCP。应等质量实验结束后再测速度。36 次仍是有限、相关的样本，经验 p95 不能外推为服务承诺。性能未达标单列待办，不计入功能通过数。

`scripts/experiment_retrieval.py` 在同一份源文档快照上分别切块，区别于旧验证器的“同一批已切好的块”：`--stage chunks` 比较字符/段落边界，`--stage gates` 扫三参数八组组合。后者共享模型并缓存相同 pair 的分数，只用于质量与阈值分析。输出含私有原文预览，必须存到仓外或忽略目录，不公开提交。

`mine-transcripts --unranked` 不运行检索。经原文核验后，用 `scripts/sealed_eval.py --freeze` 封存 SHA 和事件组；正式运行在第一次检索前写入不可重复使用的标记。中断、失败或看过结果的题均只能作为后续回归。此机制保证一次使用，不证明标注者独立，也不替代最终回答与引用评测。

交付报告现在另外查询远端分支 SHA 和 GitHub 可见性，并记录独立观察时间。查询失败显示未知；不会从“本地已提交”推断发布状态，也不将现在看到的远端状态倒推为过去某时已发布。

## 让已有会话使用新代码

代码和 MCP 工具声明在进程启动时加载。已有 Claude Code 会话需重新连接 `rh-memory`，或打开新会话。语料与用途标签的热更新在加载新代码后继续生效。
