# memory_rag — 给 zihao 三层记忆建的 RAG(毕业项目 2.0)

**触发时机**:排查/写方案/做设计决策前查"我以前踩过的同构坑";复习 agent 教材概念+自己的实战案例;面试前按主题捞素材。

## 语料(三源,启动时现读现建,无需手动重建索引)

| source | 目录 | 内容 |
|---|---|---|
| case | `Desktop\Agent学习计划\rh学习记忆\` | 实战原则笔记 |
| book | `Desktop\Agent学习计划\notes\` | 两本教材的章节笔记 |
| mem | `~\.claude\projects\C--Users-zihao\memory\` | CC 长期记忆(**凭据脱敏**:password/token/key 类的值→`<REDACTED>`) |

排除 `MEMORY.md`(纯索引)。切块:按 `#/##/###` 标题,超 1500 字符再硬切(重叠 150)。

## 入口

```bash
py -3.12 mr.py stats                                    # 语料/分词器/向量层状态
py -3.12 mr.py search "端到端绿了但组件没被测到" --mode bm25   # mode: bm25|vec|hybrid(默认)
py -3.12 mr.py eval --mode all -v                       # 金标评测三方对比,-v 看 miss
py -3.12 mr.py setup-vec                                # 装向量依赖(清华镜像)+预拉 bge 模型
```

MCP:已注册 user 级 `rh-memory`(工具 `memory_search`/`memory_get`),新 CC 会话直接可用。
更新命令:`claude mcp list` 看健康;卸载:`claude mcp remove -s user rh-memory` + 删本目录 + 删 `.cache\`。

## 性能(2026-08-21 分腿计时定案)

- 冷启动 35.5s 的 **88% 是 `import torch`**(31.4s,Windows DLL 加载),模型加载只 0.1s、暖查询 0ms——别优化错腿。
- **MCP server 已做主线程分段预热**(答完 tools/list 后暖):会话里首查 35s→**0.45s**。⚠️后台线程 import torch 慢一倍(~70s),别改回线程方案。
- CLI 一次性进程躲不开 torch 税:**快查用 `--mode bm25`**(4ms 级,17/18·.852 够用),要语义再上 hybrid。
- 后续候选:ONNX 化 encode(冷启动可到 ~6s),引新依赖需单独验证再上。

## 仓库

- git 私库 `nameishardright/memory-rag`(https,凭据走 Git Credential Manager,同 supplier-health);**改本体当轮必 commit+push**。`.cache/`(模型+embedding)不进 git,可重建。
- Google Drive 备份随「海马云」项 robocopy(`/XD .cache`),清单见 `G:\My Drive\AI备份\_备份清单.md`。

## 查询技巧(2026-08-21 三步实验定案:英文全偏→中文常用词到领域→中文罕见词精准命中)

- **用语料的语言问**:语料中文为主,`pr merge failed` 这种英文短查询词法腿零重合、向量腿(bge-zh)也弱。
- **用罕见词问**:`直合 cherry-pick 冲突` 一发命中 gerrit-workflow;`gerrit 合入` 这类高频实体词满语料都是,零区分度还淹掉 how-to。
- **已知文件直接 `get`**,检索是给"不知道在哪"用的。
- 多义词(merge/代理)检索分不清——看 `::` 块标题人裁,或换更具体的词重问。

## 坑

- **向量层没装不炸**:hybrid/vec 自动回退 bm25,结果里 `note` 会说明——别把回退结果当混合结果读。
- 模型下载走 `HF_ENDPOINT=https://hf-mirror.com`(代码里已默认);pip 走清华镜像。
- embedding 缓存在 `.cache\`,按 chunk md5 增量复用;改语料只重算改动块。换模型名会全量重算。
- `MR_TOKENIZER=bigram` 环境变量可强制关 jieba(分词器对比实验用)。
- MCP 纪律:`mr_mcp_server.py` 里 stdout 已换成 stderr,库里随便 print 不会打脏协议流——但别改掉开头那两行。
- 金标 `golden\golden_queries.json` **v1 已验收**(2026-08-21,draft=false)。官方基线:hybrid 18/18·MRR .880(937 chunks)。改金标必须重跑 eval 并在教学 doc 记新基线;语料里新长出"第二真源"时 expect 要跟着演化(q9 先例)。

教学 doc(设计决策×章节):`Desktop\Agent学习计划\notes\第5-6周-毕业项目2.0-记忆RAG.md`
