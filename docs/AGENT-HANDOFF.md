# OneDrive 私有 RAG：agent 接手说明

更新日期：2026-10-03。通用代码在 [`feat/encrypted-personal-corpus`](https://github.com/nameishardright/memory-rag/tree/feat/encrypted-personal-corpus) 分支。最后一次已评测的检索代码提交是 `a87c8b6`；文档更新后的完整部署提交与文件 SHA-256 以本机 `app-manifest.json` 为准。main 未合并此模块。

先读本页和[变更记录](CHANGELOG-RAG.md)，再按[私有版使用指南](private-corpus.md)操作。这些文件只包含通用实现、操作路径和汇总结果。

## 未部署的后续分支（2026-10-11）

`fix/private-natural-retrieval` 基于 `2bb85d1`，针对不带资料名的日常问法。改动、修改前后对照和部署步骤见[变更记录](CHANGELOG-RAG.md)第一节。本页其余内容描述的仍是 OneDrive 里正在运行的 `2bb85d1`。

合入并部署后，下面几处会变：

- 自动范围路由到 technical、stock 或 history 时同时查 personal，结果交替合并；尚未加载的大分区不自动打开，返回里带 `route.deferred_scope`。手动选择的范围不受影响。
- 私有版的关键词分词改用 `mr_private_index.tokenize`，不再直接用 `mr_index.tokenize`；`mr_index.rrf_files` 仍然复用，工作学习版没有改动。
- 长文件的 evidence 由关键词和向量一起选段，并带上相邻段。
- 向量缓存整体失效，需要按范围重新 prepare 一次；关键词缓存自动重建。
- 默认策略排除股票研究目录下的原始公告和行情缓存，下一次导入会重写分区。

这批改动是在另一台没有资料库、也没有口令的电脑上做的，只跑过虚构题集：单元检查 228 项、四份虚构题集 85 个正例的本地 E5 回归，以及新旧代码的成对对照。真实资料上的回归要在资料源电脑上完成后才能部署：

```powershell
py -3.12 -m unittest discover -s tests
py -3.12 scripts/verify_private_semantic.py --output .cache/private-semantic.json
py -3.12 scripts/compare_private_versions.py --baseline-code <2bb85d1 的目录> --regression-fixture tests/fixtures/private_natural_dev.json --regression-fixture tests/fixtures/private_natural_holdout.json --output .cache/compare/check.json
py -3.12 mr_private_benchmark.py --store <corpus.enc> --labels <已有的加密题集> --output <结果.enc> --summary <汇总.json> --baseline-code <2bb85d1 的目录>
```

两份新的虚构题集都已经用过：`private_natural_dev.json` 是开发集，`private_natural_holdout.json` 在 2026-10-11 作为留出集消耗了一次，之后都只能当回归。

## 本机位置和版本核对

下面使用 Windows 环境变量表示位置；这台电脑的生产目录是用户 OneDrive 下的 `_Private-Memory-RAG`。

| 位置 | 用途 |
|---|---|
| `%USERPROFILE%\projects\memory-rag` | 通用源码 checkout，分支为 `feat/encrypted-personal-corpus` |
| `%OneDrive%\_Private-Memory-RAG\app` | 私有桌面应用实际加载的 Python 文件 |
| `%OneDrive%\_Private-Memory-RAG\corpus.enc` | 已加密的资料库 |
| `%OneDrive%\_Private-Memory-RAG\app-manifest.json` | 部署提交、应用文件哈希、模型版本和验收结果 |
| `%OneDrive%\_Private-Memory-RAG\retrieval-settings.json` | 当前检索配置：hybrid，索引缓存加密保存 |
| `%OneDrive%\_Private-Memory-RAG\Start-RAG.cmd` | 现有桌面入口；重新打开应用后加载新代码 |
| `%LOCALAPPDATA%\memory-rag\models\multilingual-e5-small` | 固定版本的本地多语言向量模型 |

本次交付将本页、`CHANGELOG-RAG.md`、`private-corpus.md` 和 `AGENTS.md` 放到 OneDrive 应用根目录。GitHub 推送和本机部署是两个步骤；推送不会自动更新 OneDrive 应用。本机文件一致也不能证明 OneDrive 云端已经完成同步。

核对版本时，先确认源码的 Git 分支、提交和工作区状态，再比对 `app-manifest.json` 的 `source_git_commit`、`sha256` 和 `root_sha256`。生产 app 是部署副本，后续修改从源码 checkout 开始。

## 当前检索怎么工作

1. 先选择 personal、technical、history 或 stock 范围；保留手动选择。`all` 是关键词检索。自动范围判断只是辅助。
2. BM25 找精确词、姓名、编号；本地 E5 向量找中英文意思相近的段落。结果按文件融合，再挑证据段落。
3. 明确写出 `档案《资料名》` 或 `record named "source-name"` 时，先在所选范围精确定位文件，再在文件内检索。普通书名号不会自动指定文件。重名时使用完整相对路径；不存在、歧义、被排除或范围不符时返回无证据。
4. 找到文件不等于事实已证实；短预览经常省略关键数字和日期。继续读取完整 evidence，核对人物、时间、出处后才作答。

工作学习版原有 `mr_index.tokenize` 和 `mr_index.rrf_files` 被私有版复用。`mr_private_index.py` 为加密内容建立独立、按范围隔离的倒排索引；`mr_private_semantic.py` 在本机计算向量。两个资料库分别建立索引，工作学习版保留原有检索流程。

私有模型是 `intfloat/multilingual-e5-small`，revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`，ONNX int8、384 维，使用对应的 `query:` / `passage:` 前缀和长文重叠窗口。模型身份用于复现和缓存核对，不是模型训练。personal 向量已准备；technical/history 需要按范围另行 prepare。当前机器的工作学习版默认 BGE 模型和部分默认资料路径缺失，不能把私有版结果当作工作版性能验证。

## 评测记录在哪里，哪些可以继续用

生产目录的 `personal-checkup\natural-20261003T105233` 保存最终评测：

| 文件 | 内容 |
|---|---|
| `status.json` | 完成状态和是否在这次留出对照中提升 |
| `freeze.json`、`source-selection.json` | 搜索前冻结记录、代码指纹、来源分组与排除计数 |
| `holdout-summary.json` | 同一组题的旧/新 BM25 与 hybrid 汇总，top 5 |
| `development-summary.json` | 开发集汇总 |
| `source-regression-summary.json` | 原有 20 题来源探针对照 |
| `previous-natural-regression-summary.json` | 已用过的 40 题自然改写回归 |
| `*.enc`、`*.consumed.enc` | 加密标签、详细结果和留出集已使用标记 |

早先的 `personal-checkup\natural-20261003T101443` 是已用过的首次试验，保留原始退步结果。源码 `.cache\rag-github-natural-20261003` 保存公开代码测试结果、修改前代码快照、部署备份和本次本地交付记录；该目录不提交 Git。

只读汇总文件即可了解进度；需要具体问题或证据时在本机解锁加密结果，避免写到公共文档、终端输出或日志。本次留出集已经使用。后续只能作为回归；不要删除 consumed 标记、改标签或改目录后称为新测试。

最终 24 个正例来自 12 个新来源/正文组，每组中英文两问，另有 4 个负例；开发集为另外 8 组、16 问。相同正文/别名和语言版本保持同一组，并排除此前评测来源。按正文哈希分组不能保证不同文件描述的是完全独立的现实事件。

这些问句由 agent 根据真实原文的可核对字段改写，包含明确资料名。它们可以验证有来源上下文的检索，尚未证明无资料名日常问法、独立收集的人类提问、最终回答正确率或拒答正确率。负例中相关文件也可能返回为待核实候选。

## 继续修改与验证

通用检索代码变化先运行离线单元检查；本地模型已经安装时，再运行公开虚构资料的真实模型回归：

```powershell
py -3.12 -m unittest discover -s tests
py -3.12 scripts/verify_private_semantic.py --output .cache/private-semantic.json
git diff --check
```

当前已通过 206 项单元检查和 27 个虚构正例的本地 E5 回归。模型回归在推理期间禁止 socket 连接，不解锁真实资料库。可选来源收集器测试通过 `MR_PRIVATE_SOURCE_TOOLS` 指向已安装的本地工具；此收集器不包含在公开分支。

新增真实评测使用[指南](private-corpus.md)中的 `mr_private_benchmark.py --gui`，口令只进入本机界面和隐藏 worker 的 stdin。先保存未参与检索的新问题及可核对原文，把同一事件/来源的版本放到同一组，冻结标签后才搜索。详细题目、答案与逐题结果仍加密保存在 checkout 外。旧集可回归，新的未使用留出集才支持新的提升结论。

下一步优先定位剩余 6 个完整证据缺口，再用新收集、没有资料名提示的人类日常问题建立独立评测。先核对来源收录和段落覆盖，再决定扩大证据预算或调整排序。不要用已消耗的最终留出题继续调参数并报告同一集为新测试。

## 部署和回退

本次代码部署保留每个被替换文件的原件，备份位于上述本地交付目录内的 `deployment-backup-*`；`deployment-results.json` 记录完整备份路径、部署哈希和保护文件一致性。更新应用代码与文档时，不重新导入或重新加密资料库。

回退时读取该次备份清单，恢复原来存在的应用/文档文件；新增加的文件按清单移出到恢复目录。备份的 `app-manifest.json` 一起恢复。操作后核对哈希并重新打开应用。代码回退不回滚 `corpus.enc`、加密索引、冻结标签或 consumed 标记。

## 仍未补齐的来源

已有来源审计确认 OneDrive AI 备份和 GitHub 快照正文进入对应范围；这是本地快照覆盖，不证明云端所有内容或最新 GitHub 提交已收录。OneNote 链接未展开成页面，仍未收录。另有 4 个原始加密 PDF 需要各自的文件密码，资料库口令不能解锁它们。历史资料也不能直接证明当前工资、合同或家庭状态。
