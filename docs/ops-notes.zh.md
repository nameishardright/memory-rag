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
- 金标 `golden\golden_queries.json` **v1 已验收**(2026-08-21,draft=false)。官方基线:hybrid 18/18·MRR .880(937 chunks,2026-08-21)→ **.838(1922 chunks,2026-09-16 重跑)**;同期 bm25 17/18 .852→.789、vec 16/18 .716 —— **语料长大只稀释 BM25 单腿,融合没掉**(1819→1922 块,.835→.838)。PDF 金标同日 hybrid 12/12 .917(MR_EMBED_MODEL_PDF=bge-m3),阴性 12 条 low 7/medium 4/high 1(原 8/3/1),MR_PDF 开/关两态主金标仍逐字一致。改金标必须重跑 eval 并在教学 doc 记新基线;语料里新长出"第二真源"时 expect 要跟着演化(q9 先例)。

教学 doc(设计决策×章节):`Desktop\Agent学习计划\notes\第5-6周-毕业项目2.0-记忆RAG.md`

## 2026-09-08 热更 + PDF 原文源(开关)

**热更**:原来 `mr_tools._ensure` 进程启动装一次索引就不看文件了,会话里新写的笔记查不到,要重启 Claude Code 才生效。现在每次查询前(最多每 `MR_RELOAD_SEC`=5 秒一次)比对语料指纹(`mr_corpus.signature()` = 各源 md 的名/mtime/size),变了就重建:BM25 按 md5 缓存分词只切新块、向量层按 md5 增量只算新块、模型实例复用。实测新笔记写入后下一次查询 2.0s 内命中(bm25#1+vec#1),删掉后消失;改造前重建要 12s,大头是 jieba 重切 1816 块,加分词缓存后 2s。

**PDF 原文源**(两层开关,缺一不可):
- 环境变量 `MR_PDF=1` 管**装不装**:装载 `Desktop\Agent学习计划\*.pdf` 按页切块(1296 块),文本抽取 `py -3.12 mr.py pdf-extract` 一次(指南 102s / HA 128s,pypdf + NFKC),缓存在 `.cache/pdf-text/`(gitignore,**书的原文绝不能进公开仓**),按 (size, mtime) 失效。
- 查询参数 `source="pdf"` 管**查不查**:默认检索只走 case/book/mem 三源,PDF 只有显式 `source=pdf`(或 `include_pdf=true`)才参与。`memory_get("xxx.pdf#p81")` 取那一页 ±1。
- 🔴 为什么不能全混进默认检索(数字):把 1296 页混进去,主金标 hybrid 18/18→17/18、MRR .835→.750,原文页把带 ⭐ 页码的笔记挤下去;**即便检索时剔掉 PDF,BM25 的 IDF/平均长度统计也被 PDF 拉偏**(.835→.806)。所以 BM25 按源集合**分别建索引**(`_SubBM25`:notes / pdf / all 懒建),向量层用掩码**先掩后切**。改完 MR_PDF 开/关两态主金标逐字一致。
- PDF 消费单元是「文件#页」不是文件:`rrf_files(unit="page")`,否则一本书 700 页永远只出一页;评测器去重同理(有 `expect_pages` 时按页)。这一处修完 PDF 金标 9/12→11/12。
- 数字(`golden/golden_pdf.json` 12 条,问法换说法、页码区间按笔记 ⭐ 页 ±2 反查核过):bm25 11/12 MRR .830 · vec 10/12 .628 · hybrid 11/12 .694 —— **原文页上 bm25 比 hybrid 强**(页块长、内容杂,向量腿拖后腿),12 条不够下调权的结论,先记着。MR_PDF=0 时 0/12(理应)。
- 定位:笔记答「对应哪一章」(带页码指针),原文答「书上原话怎么说」。让 rh-memory MCP 带上 PDF:`claude mcp remove -s user rh-memory && claude mcp add -s user -e MR_PDF=1 rh-memory -- py -3.12 <path>\mr_mcp_server.py`(要重启会话才换新进程)。

**观测留痕 + 金标候选(2026-09-08,HA ch8 笔记「对照」第 4 条 / 《指南》p70-71)**:每次 `memory_search` / `memory_get` 追加一行到 `.cache/query-log.jsonl`(gitignore;`MR_LOG=0` 关;eval/ranks 不记;MCP 记 `src=mcp`,CLI 记 `src=cli`)。`py -3.12 mr.py log [--days 7] [--src mcp]` 汇总:0 命中的真实问题(语料缺口或阴性题候选)、**search 后 120s 内 get 了命中文件 = 隐式正标签**,直接打成 `{"q","expect"}` 供挑进 golden。金标从 18/12 往 50+ 涨靠这个,不靠手编。

**三级置信 / 拒答门(2026-09-08,自测题 6 / 《指南》p70-71;观澜 kb.js 的门搬过来时改成了查询级)**:结果多两个字段 `confidence`(high / medium / low)和 `gate`(两腿最高分 + 门槛)。两条腿各取最高分:BM25 ≥ 8(笔记)/ 12(PDF,`MR_MIN_BM25*`)、余弦 ≥ `MODELS[*]["min_cos"]`(small .48 / bge-m3 .52,Qwen3 没标定不启用),两腿都过 = high,一条 = medium(note 说明),都没过 = low(note 明说「语料里很可能没有」;`MR_GATE=drop` 才清空 hits,默认 note)。
- 标定数据(`golden/golden_negative.json` 12 条无答案问题,故意混「今天/怎么/为什么/agent」):正例 18 + 12 条全部 high;阴性 low 2 / medium 7 / high 3。**分数门在个人通用语料上只能标不能拒**:无答案的问题靠常见词 BM25 也能到 19、靠语义近邻余弦也能到 .60,分差断崖(top1/top5、余弦领先幅度)也分不开;标成 high 的三条(量子计算 / agent 发音 / 英语时态)本来就贴着语料题材。kb.js 能一刀切是因为它语料小而专。
- 🔴 为什么不逐条过滤(kb.js 那样):正确答案常常只有一条腿看见(bm25#1 + 向量未进榜),逐条 AND 会把它删掉;「cherry-pick 直合」这类罕见词问法就是 medium(bm25 16 / cos .45),删了就是误杀。真要硬拒答要上重排模型(cross-encoder 直接给 query-passage 相关分,分布分得开),这是下一项。
- 🔴 一个假阴性:「今天天气怎么样」在语料里真的有(我自己的笔记里引用了 kb.js 的那个例子;教材里也有天气工具示例)——阴性题也会被自己的笔记污染,建阴性题要 grep 一遍语料。

**第三级 cross-encoder(2026-09-08 项 7,用户「重排 go」)——结论:当重排掉分,当置信门是最强信号,默认 gate 模式**
- 预设 `mr_index.RERANKERS`:`bge-reranker-base`(278M,1.1GB,默认)/ `bge-reranker-v2-m3`(568M,2.27GB);`pull-model bge-reranker-base` 拉。入口:进程 `MR_RERANK=<预设>`、单次 `memory_search(rerank=True/False/<预设>)`、CLI `--rerank`。`MR_RERANK_MODE=gate`(默认)只给 RRF 前 `MR_RERANK_G`=5 条打分定置信,**顺序不动**;`=rerank` 按分重排(对照用)。段落按 450 字开窗最多 3 窗取最大分(512 token 对中文≈480 字,不开窗 PDF 两条正例分 .003/.053)。
- 🔴 当重排用两套金标全掉(GPU 标定):笔记 hybrid .835 → .692(base)/ .764(v2-m3);PDF .917 → .748 / .817。RRF 前几名本来就对,cross-encoder 只看一块前 480 字、偏爱「像答案」的段落,把对的往下压。别开 rerank 模式。
- 当置信门用(gate 模式,阈值 hi .50 / lo .10):base——正例 18/18 + 12/12 里只有 PDF 2 条不是 high(「上下文包分区 State」.061 答案在图表里);阴性 low 8 / medium 3 / high 1(「今天天气怎么样」语料里真有)。v2-m3 更狠(阴性 low 10 / high 0)但把 4 条真正例标成 medium/low——**记忆工具里假拒绝比假放行贵**,选 base。对比两腿分数门(阴性 low 2 / medium 7 / high 3)是质变。
- 延迟:GPU 每查 +0.5s;**CPU 每查 +3s**(9–11 对,约 300ms/对;PDF 14 对 4.4s),首次加载 5s。MCP 注册已带 `MR_RERANK=bge-reranker-base`;重模型(bge-m3 / 重排)默认**懒加载**,`MR_WARM_HEAVY=1` 才预热——用户常开 4-5 个会话,每会话一个 MCP 进程,全预热 = 每进程 3.5GB。嫌慢:`MR_RERANK_G=3` / `MR_RERANK_MAXWIN=2` 大约减半,或单次 `rerank=false`。
- 结果字段:`rerank`={model, mode, n, pairs, top, ms},每条 hit 带 `rerank` 分;`gate` 里 `level_legs` 保留两腿的判断、`level` 以重排为准。

**embedding 模型一键切换(2026-09-08)**:预设表在 `mr_index.MODELS`:`bge-small-zh-v1.5`(默认,24M,512 token,92MB)/ `bge-m3`(568M,8192 token,2.27GB,仓里只有 pytorch_model.bin)/ `Qwen3-Embedding-0.6B`(0.6B,32k token,1.19GB);短名 small / m3 / qwen3 也认,任意 HF id 或本地权重目录也能传(前缀用 `MR_EMBED_PREFIX` 给)。
- 三个入口:进程默认 `MR_EMBED_MODEL=bge-m3`(MCP 用这个,要重启会话)/ 单次查询 `memory_search(model="bge-m3")` / CLI `--model bge-m3`(search / eval / ranks 都有)。`py -3.12 mr.py stats` 列出每个预设有没有权重、有没有缓存。
- 🔴 每个预设**自带 query 前缀**:bge-zh v1.5 要那句中文指令,bge-m3 不要,Qwen3 要 `Instruct: …\nQuery: `——换模型不换前缀会无声掉分,金标只会说"变差了"。
- 缓存按模型分文件 `.cache/emb-<模型>.npy` + `emb_meta-<模型>.json`(老的 emb.npy 首次自动改名迁移,不重算;⚠️ 改名要在 `with open()` 外面,Windows 上文件开着 os.replace 报 PermissionError,吞掉就成了静默全量重算),切回来零成本;每个模型的向量层在进程里各自懒建,内存按装了几个算(bge-m3 fp32 约 2.3GB)。
- 缓存键(v2):`mr_corpus.chunk_key` = md5(文件名+标题+正文),即真正被编码的文本;之前只哈希正文,改标题/文件名不触发重算(外部评审项②,当时暴露面 1 组)。`emb_meta` 带 `ver`,改 `_index_text`/分块规则时 `CACHE_VER`+1 整体作废,别新旧键混一个文件。文件级 RRF 的文档身份改成「源/文件名」(项④,跨源同名当时 0 个,潜伏)。两处改完主金标 18/18 .835、PDF 11/12 .694 逐字不变。
- 拉权重:`py -3.12 mr.py pull-model bge-m3`(curl -C - 断点续传,走 HF_ENDPOINT=hf-mirror.com)。🔴 镜像对仓里不存在的文件回 15 字节 "Entry not found" 而不是 404,pull-model 会识别并删掉;别手工 curl 留下假权重。没本地权重**不会**偷偷去 HF 拉(内网拉不动/代理劫持),hybrid 退 bm25 并在 note 里说 pull-model;真要联网拉设 `MR_EMBED_ALLOW_DOWNLOAD=1`。
- 数字(用户 GPU 对照,同一 3116 块,笔记 18 题 + PDF 12 题):笔记 hybrid small 18/18 .835 vs bge-m3 17/18 .822(一题之差,噪声级);PDF small 11/12 .694 vs bge-m3 12/12 .917;笔记查询中位延迟 49ms vs 199ms(GPU;CPU 上更慢)。**截尾的代价取决于块长和答案位置**:笔记按标题分段主题在段首,截掉的是尾巴;PDF 一页一块,答案在页中任何位置。同一天先试过按 512 token 切小块:hybrid .835→.801(800 字)→.764(420 字),多向量 max-pool .798——**一段一块本身就是信号,切小反而掉**。
- CPU 上的真实数(本机 16 线程,缓存命中,含查询编码):PDF 金标 hybrid bge-m3 12/12 .917 平均 256ms vs small 11/12 .694 35ms;笔记金标 bge-m3 17/18 .822 169ms vs small 18/18 .835 59ms(bge-m3 的 vec 单腿在笔记上 .804 > small .716,但 hybrid 不涨——BM25 已经把那部分捞回来了)。**建缓存别用 CPU**:bge-m3 3116 块 GPU 不到 3 分钟,CPU 小时级;Qwen3-Embedding-0.6B CPU 100 分钟没编完 1819 块,放弃。本机基础 env 是 CPU torch,另一会话装了隔离的 torch 2.11+cu128 在 `C:\Users\zihao\scratchpad\rag-model-compare-20260908-171900\cuda-runtime\`,`PYTHONPATH` 指过去就能 GPU 编码(`PYTHONPATH=<那个目录> MR_PDF=1 py -3.12 mr.py eval --golden golden/golden_pdf.json --model bge-m3`),缓存文件跟 env 无关,建完 MCP 进程照常用 CPU torch 查。
- **heading_path 实验(HA ch8 笔记「对照 memory_rag」2①,p289;2026-09-08 晚,GPU 编码同一主金标)——负结果,别做**:索引文本里把当前标题换成完整路径(H1 > H2 > H3):hybrid 18/18 .835→.769、bm25 .789→.778、vec .716→.636;去掉 H1(文件名已在索引文本里)只拼 H2 起:hybrid .780、vec .669,仍全面掉。原因和「切小块反而掉」同源:笔记的文件名已经是主题,当前标题已经是本节,再把上级标题塞进每一块只是给全文件所有块加同一串词——BM25 里它们不再区分块、还把块拉长,向量里它们稀释本节语义。观澜 kb.js 带 heading_path 有收益是语料形状不同:案例卡的概念名只在 H1 里、文件名是拼音/编号。**同一个技巧在两个语料上一正一负,只能各量各的。**
- 决策:笔记留 small,`source=pdf` 走 bge-m3——`MR_EMBED_MODEL_PDF=bge-m3`(没显式传 model 且 source=pdf 时生效)。MCP 注册:`claude mcp remove -s user rh-memory && claude mcp add -s user -e MR_PDF=1 -e MR_EMBED_MODEL_PDF=bge-m3 rh-memory -- py -3.12 <path>\mr_mcp_server.py`,之前先 `pull-model bge-m3` 并 `MR_PDF=1 py -3.12 mr.py eval --golden golden/golden_pdf.json --model bge-m3` 建一次缓存(CPU 上 1296 页要十几分钟,只算一次)。
