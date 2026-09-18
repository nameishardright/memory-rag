# 检索范围、段落证据与评测

2026-09-17 更新。默认检索仍为 `case/book/mem`，股票用 `source="stock"`，教材原文用 `source="pdf"`。普通笔记的 BM25 统计排除股票、PDF 和已标记的评测内容；股票保留原有的全部非 PDF 统计。`include_pdf=true` 只增加 PDF，不顺带打开股票来源。

## 查询和取段落

```python
memory_search(query="财报后高开回落的历史判断", source="stock")
memory_get(file="<搜索返回的 file>", source="stock")

memory_search(query="cherry-pick 直合")
memory_get(file="<搜索返回的 file>", source="mem", heading="<搜索返回的 heading>")
```

`heading` 精确选择一个完整 Markdown 段落；找不到或有多个同名段落时明确报错，省略它可读全文。PDF 仍用 `file="书名.pdf#p81"` 取该页及相邻页。

Hybrid 先融合文件排名，再在同文件的两路候选中选择正文。只有索引行时保留定位结果并标记 `evidence_kind="navigation"`。启用 gate 且当前结果均未达到 high 时，只对评分低于原有相关性下限的代表块补评同文件的一段词法正文；替代段落达到原有下限才替换，文件排名不变。已经达到 medium 的答案段落不会被高分主题概述挤走。这避免文件对另一主题的投票把已有答案的置信标签拉成 low。`snippet` 仍是预览，操作前应读完整段落，涉及历史变更时再核对全文中的后续更正。

两个 MCP 工具均声明 `stock`。`rerank` 接受布尔值或预设名；默认 `MR_RERANK_MODE=gate` 对前五条评分，保留文件顺序和 `score`，另给 `rerank` 分与置信标签。必要的同文件补评会增加延迟，`rerank.extra_passages` 和 `pairs` 记录额外工作量；查询已是 high 时不补评。只有显式 `MR_RERANK_MODE=rerank` 才按交叉编码器分数重排。相关分、`confidence=high` 都不等于最终答案正确。

## 评测内容的用途标签

`corpus-policy.json` 保存明确的文档和段落用途。评测复盘用 `purpose="evaluation"`，默认在 BM25 和向量召回前排除；纯教学正文保持可检索。混合文档可用 `sections[].heading_prefix` 只标记运行报告部分，覆盖该标题的所有硬切分块。

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

返回内容会显示 `purpose="evaluation"` 并提醒其样例问题未必有事实答案。`memory_get` 仍可主动读取这些文档。过滤只依赖用途标签，不根据天气、菜谱等问题做黑名单，也不读取 golden 标签；尚未标记的新增评测材料仍需补标签。金标题目、答案标签和结果报告应保留在本仓 `golden/`、隔离的评测目录，不加入普通事实语料。

## 分开衡量文件命中与答案段落

`mr.py eval` 同时报告：

- `returned@5` / MRR：实际返回条目中的首次命中。BM25、vec 可能返回同文件的多个块，这些块仍占真实名额。
- `document@5` / MRR：在本次最多十条返回结果内，按来源和文件去重后的历史口径。PDF 还检查原有页码范围。
- `evidence@1` / `evidence@5`：来源、文件、标题及预览内的必需事实同时满足。只找对文件不能通过。
- 阴性题的 low / medium / high 分布，以及标注的困难负例位置。

原有 59 道题与原标签保持不变，包括仍标为草稿的 12 道股票题。新增 `golden/golden_evidence.json` 有八条专项检查：四条原正例的段落标注、两条原阴性题的困难负例，以及两个主动检索评测笔记的问题。它们不是八道全新独立测试题。

正例 `evidence` 是可接受段落的列表；每个段落必须给出 `source`、`file`、`heading` 或 `heading_contains`，以及 `required_text`。必需事实之间是 AND，同一事实的可接受字面写法可以用列表表示 OR。所有必需事实都必须出现在实际 `snippet` 中，不能靠读取未返回的正文替一次失败的预览补分。

`hard_negatives` 记录“相关但不能回答”的位置和理由，例如“已推 change 直合”与“叠错分支后补救”的区别，以及提到天气测试却没有实时天气的数据。标签只供评测和后续人工样本整理，不参与运行时排序，也不会自动触发模型训练。

这些已观察过的案例标为 `split="regression"`。以后建立独立测试集时，必须在调优前按 `case_group` 留出，同一事件的切片、副本和问法改写放在同一组。评测器会拒绝输入中跨 split 的同一组；不要把本轮已经用来选方案的题改名为 holdout。段落字符串检查也不是最终答题模型的准确率评测。

## 验证命令

```powershell
py -3.12 -X utf8 -m unittest discover -s tests -v
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

## 让已有会话使用新代码

代码和 MCP 工具声明在进程启动时加载。已有 Claude Code 会话需重新连接 `rh-memory`，或打开新会话。语料与用途标签的热更新在加载新代码后继续生效。
