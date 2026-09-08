# memory-rag

**A personal-memory RAG that gives my AI coding agent long-term recall** — hybrid retrieval (BM25 + dense vectors + reciprocal-rank fusion) over three corpora of engineering notes, exposed to Claude Code through a custom [MCP](https://modelcontextprotocol.io) server, and held to a golden-set evaluation: **18/18 recall, MRR 0.880** across 937 chunks.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Retrieval](https://img.shields.io/badge/retrieval-BM25%20%2B%20bge%20%2B%20RRF-orange) ![MCP](https://img.shields.io/badge/protocol-MCP-8A2BE2) ![Eval](https://img.shields.io/badge/golden%20eval-18%2F18%20·%20MRR%20.880-brightgreen)

> Why: an AI agent forgets everything between sessions. I keep three growing bodies of notes — hard-won incident lessons, textbook study notes, and the agent's own long-term memory files. This project turns them into a retrieval service the agent queries *before* repeating a mistake I already paid for.

## Architecture

```mermaid
flowchart LR
    subgraph Corpora["3 corpora (read fresh at startup)"]
        A["case notes<br/>(incident lessons)"]
        B["book notes<br/>(agent-engineering texts)"]
        C["agent long-term memory<br/>(credential values redacted<br/>at ingestion)"]
    end
    Corpora --> CH["Heading-aware chunker<br/>≤1500 chars, 150 overlap"]
    CH --> IDX1["BM25 index<br/>(jieba tokenizer)"]
    CH --> IDX2["bge dense vectors<br/>(md5-incremental cache)"]
    IDX1 --> F["File-level<br/>Reciprocal-Rank Fusion"]
    IDX2 --> F
    F --> CLI["CLI<br/>(search / get / stats / eval)"]
    F --> MCP["MCP server<br/>(memory_search / memory_get)<br/>staged warm-up"]
    MCP --> CC["Claude Code sessions"]
```

## What's interesting about it

**Evaluation as the acceptance gate.** An 18-query golden set (`golden/golden_queries.json`, expectations judged at file level) is versioned in the repo and treated like a test suite: the current baseline is **hybrid 18/18, MRR 0.880 (937 chunks)**; BM25-only holds 17/18 at MRR 0.852. Any change to the golden set requires re-running the eval and recording a new baseline — the ruler stays stable even when a higher score is tempting.

**Cold-start profiling, then fixing the right leg.** First query used to take 35.5s. Per-stage timing showed **88% was `import torch`** (31.4s of Windows DLL loading) — model load was 0.1s and a warm query 0ms. Fix: the MCP server does a *staged main-thread warm-up* right after answering the protocol handshake, cutting the first in-session query from **35s to 0.45s**. (A background-thread import was tried and rejected: the GIL contention made it ~2× slower.)

**Honest degradation.** If vector dependencies aren't installed, hybrid/vec modes automatically fall back to BM25 — and the result explicitly *says so* in a `note` field, so a fallback result is never mistaken for a fusion result.

**Security at ingestion.** The agent-memory corpus may contain `password:`/`token:`/`api_key:` lines. A redaction pass keeps the *keys* searchable ("where do I keep the X credential?") while replacing every *value* with `<REDACTED>` before anything enters an index or leaves through a tool.

**Retrieval findings from a 3-step query-language experiment.** Query in the corpus's language (English queries lost both the lexical leg and the zh-tuned dense leg); rare domain terms beat frequent entity words (which match everything and discriminate nothing); and when you already know the file, `get` beats search.

**MCP hygiene.** stdout is reserved for the protocol stream; all library prints are rerouted to stderr, so no stray `print` can corrupt the JSON-RPC channel.

## Usage

```bash
py -3.12 mr.py stats                     # corpus / tokenizer / vector-layer status
py -3.12 mr.py search "<query>"          # --mode bm25|vec|hybrid (default hybrid)
py -3.12 mr.py eval --mode all -v        # golden-set eval, three modes side by side
py -3.12 mr.py setup-vec                 # install vector deps + pre-pull the bge model
```

Registered as a user-level MCP server (`memory_search` / `memory_get`), available to every new Claude Code session:

```bash
claude mcp add --scope user rh-memory -- py -3.12 <path>/mr_mcp_server.py
```

## Repo layout

| File | Role |
|---|---|
| `mr.py` | CLI entry: search / get / stats / eval |
| `mr_corpus.py` | Corpus loading, heading-aware chunking, credential redaction |
| `mr_index.py` | BM25 + dense index, md5-incremental embedding cache |
| `mr_tools.py` | Shared search/fusion logic for CLI and MCP |
| `mr_mcp_server.py` | MCP server with staged warm-up |
| `golden/golden_queries.json` | 18-query golden eval set (v1, accepted) |

The corpora themselves are personal notes and are **not** included; the golden set documents the retrieval contract. Original operational notes (Chinese): [`docs/ops-notes.zh.md`](docs/ops-notes.zh.md).

### Hot reload & optional PDF source (2026-09-08)

- **Hot reload**: the index now re-checks a corpus fingerprint (file name / mtime / size) at most every 5 s and rebuilds incrementally — new notes are searchable on the next query without restarting the MCP server (tokenisation and embeddings are md5-cached, so a rebuild costs ~2 s).
- **PDF source, two switches**: `MR_PDF=1` loads the textbook PDFs page-by-page (`py -3.12 mr.py pdf-extract` once; text cached under `.cache/`, never committed). Default search still covers only the three note corpora; pass `source="pdf"` to search the original pages, then `memory_get("<book>.pdf#p81")` for the page ±1.
- **Embedding model, one-key switch**: presets in `mr_index.MODELS` — `bge-small-zh-v1.5` (default, 24M, 512 tokens), `bge-m3` (568M, 8192 tokens), `Qwen3-Embedding-0.6B` (32k tokens) — each carrying its own query instruction (bge-zh needs the Chinese instruction, bge-m3 none, Qwen3 an `Instruct:` prefix; swapping the model without swapping the prefix silently degrades) and its own md5-incremental cache (`.cache/emb-<model>.npy`, so switching back costs nothing). Choose per process (`MR_EMBED_MODEL`), per query (`memory_search(model=...)`) or per CLI call (`--model`); `py -3.12 mr.py pull-model bge-m3` fetches weights from the HF mirror (resumable; detects the mirror's 15-byte "Entry not found" placeholders). Measured 2026-09-08 on the same 3,116 chunks: on the note corpus the models tie (small 18/18 · MRR .835 vs bge-m3 17/18 · .822), on textbook PDF pages bge-m3 wins clearly (12/12 · .917 vs 11/12 · .694) — truncation only costs when the unit is long and the answer is not in its lead. Hence `MR_EMBED_MODEL_PDF=bge-m3` routes only `source="pdf"` queries to the bigger model while notes stay on the 4× faster small one. Chunking to fit the 512-token window was tried first and *lost* (hybrid MRR .835 → .764 at 420 chars; multi-vector max-pool .798): the heading-sized chunk is itself the signal.
- Why two switches: mixing 1,296 pages into the default search dropped the main golden set from 18/18 · MRR .835 to 17/18 · .750 — and even with PDF hits filtered out, BM25's corpus statistics drifted (.806). Fix: per-source BM25 indexes + masked vector search. With PDFs loaded, the main golden set is byte-for-byte identical to the no-PDF run. `golden/golden_pdf.json` (12 paraphrased queries with expected page ranges): bm25 11/12 · hybrid 11/12.
