# Encrypted personal corpus

The private extension reads authenticated `FAMILYMEMORYv1` extraction journals and
publishes an encrypted local corpus with optional hybrid retrieval. Ordinary search, MCP tools, redaction,
evaluation labels and golden files retain their existing contracts. Personal
documents and their original values remain outside the Git checkout.

## What belongs in each scope

The **personal** scope contains identity, immigration, family, employment,
tax, property and learning records. Age alone does not remove a passport, contract
or tax record. They remain source evidence, not proof that a historical fact is current.

Project files under `06_代码项目` and imported project roots belong to **technical**;
stock research belongs to **stock**; designated archives and backup summaries belong
to **history**. A small project directory catalogue appears in personal search. Its
quoted README text is navigation, not evidence of current maintenance or authorship.

Dependencies, build outputs, raw logs, conversation dumps, database dumps, secret
configuration files and labelled evaluation material are excluded. Original files
and existing encrypted archives are retained. `private-policy.json` supports prefix
rules for source directories; hard exclusions take precedence. Never repurpose
evaluation material as facts or change goldens to obtain a passing result.

## Retrieval and storage

Production imports write `corpus.enc` with personal documents and generation-specific
`corpus.<generation>.<scope>.enc` partitions. Partitions are written first and the
root is replaced atomically last. A query decrypts other scopes only when requested.
Each selected scope has independent BM25 statistics and an in-memory inverted index.
The UI defaults to bounded rule-based **automatic** selection: pay and family questions
use personal sources, code/project terms use technical sources, and stock terms use
stock sources. Explicit choices always win. Unrecognized questions default to personal;
automatic selection can be wrong and is not an exhaustive search across every source.
Unchanged partitions are reused across root updates. Current and previous partition references are protected; authenticated generated partitions older than 24 hours may be retired. Source journals and original archives are retained.

File-level top-k is exact: it chooses each document's highest-scoring passage without
truncating the chunk candidate pool. Identical extracted text within a scope has one
body, while original paths and SHA-256 values remain in `duplicate_sources`.

AES-256-GCM authenticates compressed payloads. PBKDF2-HMAC-SHA256 derives keys with
a random 16-byte salt and 600,000 iterations. Temporary files contain ciphertext.
Passwords enter locally and remain in process memory; they do not enter arguments,
environment variables, configuration or logs. Query sessions lock after ten idle
minutes. Background updates have an independent password lifetime until stopped.

Hybrid search retains BM25 and bilingual lexical expansion for exact words, names
and identifiers. A fixed local `intfloat/multilingual-e5-small` encoder handles
paraphrases and multilingual matches. Only the original question is embedded;
expansion terms apply to BM25. Both legs reuse `mr_index.rrf_files` to fuse document
ranks. Heading splitting and tokenization also reuse the work/study retrieval base.
The public path and its plaintext vector cache are not changed or used for private data.

The CPU backend uses the model's published ONNX INT8 export at revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`. Setup verifies the full model file against
the pinned repository checksum; inference reads local files only. E5 uses `query:`
and `passage:` prefixes. Long heading chunks use overlapping tokenizer-sized windows
without changing the original source chunks. Model weights contain no personal data
and live outside OneDrive by default.

BM25 postings, vector rows and embeddings are saved in authenticated encrypted
sidecars next to the corpus. Cache identity includes the model manifest and source
chunk fingerprints; changed content is rebuilt, unchanged embedding windows reused.
An incomplete checkpoint cannot be searched. No plaintext vector, private query log
or plaintext corpus cache is written. Default persistent cache scopes are personal,
technical and history; the very large stock scope needs separate capacity planning.

Short previews are for scanning. Search also supplies up to 12,000 characters of
original evidence, with multiple relevant passages, source hashes, dates and extraction
warnings. The UI displays that evidence and still offers the complete original source.
Year/person/appointment and historical/current-state checks flag cases for review;
they do not certify facts or resolve contradictions. All candidates remain unverified
answers. OCR and automatic speech transcripts carry uncertainty warnings. This app
retrieves evidence; it does not generate final answers or claim calibrated refusal accuracy.

## Collecting missing content

The source collector strips HTML scripts and styles, supports `.7z` alongside other
archives, expands bounded nested ZIP members, and skips dependencies before reading or hashing them. Java compiled classes and dependency JAR contents are excluded. A bounded work
queue limits retained extraction text. Eligible personal audio/video is transcribed
using an existing local Whisper model, with timestamps and review warnings. Media
without audio or speech is distinguished from extraction failure.

Linked OneDrive notebooks are read through local desktop OneNote. The notebook must
be open and synced with the signed-in account. A `.url` shortcut itself is not notebook
content. Only notebooks associated with those source shortcuts are selected. Pages
and transcripts are appended directly to encrypted journals. Deleted pages and
removed archive members are omitted from subsequent imports.

`coverage-review.enc` lists unresolved sources and review warnings. The UI's
“收录缺口” button decrypts this report locally. Plaintext status files contain counts,
timestamps and state only. `cleanup-review.enc` is a separate, dated file-cleanup
review with no deletion action; logical duplicate sizes do not establish reclaimable
local storage. Original-password files, unavailable notebooks, unsynced directories
and Personal Vault require source access before their bodies can be collected.

## Updating and testing

`mr_private_maintenance.py` runs collection, archive-member extraction, OneNote
harvesting, partition publication and live verification. Full encrypted archive
packing is independent and cannot prevent RAG updates. Existing packed archives keep
their original coverage date. The background process checks source metadata and
notebook changes every five minutes, skips unchanged data, and performs an hourly
retry for coverage gaps. Reboot requires a fresh local password entry.

```powershell
python scripts/setup_private_semantic.py
python mr.py private --source C:\private\journals --store C:\private\corpus.enc import
python mr.py private --store C:\private\corpus.enc prepare --scope personal
python mr.py private --store C:\private\corpus.enc search "家庭关系" --compact --scope auto
python mr.py private --store C:\private\corpus.enc search "architecture" --scope technical
python mr.py private --store C:\private\corpus.enc search "earnings" --scope stock
python mr.py private --store C:\private\corpus.enc search "old notes" --scope history
python mr.py private --store C:\private\corpus.enc get DOCUMENT_ID
python mr.py private --store C:\private\corpus.enc evidence DOCUMENT_ID --query "合同开始和结束日期"
```

Before preparing, create `retrieval-settings.json` beside the encrypted store:

```json
{
  "version": 1,
  "mode": "hybrid",
  "encrypted_index_cache": true,
  "cache_scopes": ["personal", "technical", "history"]
}
```

No settings preserves the legacy BM25 API. Until the selected scope has a complete
semantic index, searches explicitly report `bm25(fallback)`. Build one scope at a time
with the UI's “准备语义检索” button or the CLI. `all` is an explicit BM25 search across
sources, not a claim that a million-chunk semantic index is ready. Another synced
computer needs the local model installation before it can use encrypted vectors.

The private adapter uses the work/study file-level RRF implementation. It fuses
the full BM25/vector candidate union before selecting the final results. An entire
original query of at least 12 characters found literally in a BM25 candidate's
body retains BM25 priority and its matching passage. Paraphrases still use fusion;
this phrase rule is a retrieval heuristic and does not certify an answer.

The shell supports `/stats`, `/get ID`, `/all QUERY`, `/reload` and `/quit`. The native
UI provides personal, technical, stock, history and all-source choices, bounded
previews, complete sources and separately extracted card records. Source labels,
format and Luhn checks cannot establish current card ownership or validity.

Base retrieval needs `requirements-private.txt`; optional hybrid retrieval needs
`requirements-private-semantic.txt`. The source computer's optional
OneNote and speech integrations use `requirements-private-integrations.txt` and an
already installed local speech model. Other synced computers can search without
installing transcription or OneNote integrations.

Run `scripts/verify_unit.py` with the repository's offline settings. Synthetic
regressions compare inverted-index scores and file ranks with the original BM25,
exercise authenticated partitions, changed/deleted sources, collector HTML/ZIP
extraction, native UI scopes, locking and update controls. Live verification checks
actual encrypted snapshots, scope isolation, original personal-file SHA-256 values,
partition integrity, a negative query and timed retrieval. These checks are not an
independently labelled answer-accuracy evaluation.

`scripts/verify_private_retrieval.py` compares the unchanged old app, new BM25 and
new hybrid on frozen fictional documents with the actual encoder. It separates
document hits, preview evidence and complete evidence. A new event-grouped holdout
is consumed once, including interrupted runs. Synthetic scores cannot establish
accuracy on personal documents.

`mr_private_checkup.py --gui --store C:\private\corpus.enc` provides one local
password entry for incremental collection, linked OneNote harvesting, publication,
personal semantic preparation and native UI verification. Verified public GitHub
ZIP snapshots must be collected and expanded before their contents count as indexed;
staging a ZIP or manifest alone does not prove coverage. OneNote pages require an
open, synced notebook. Source audits report actual document inclusion and cannot
prove all cloud material or the latest remote commit is covered.

The checkup derives literal source probes before retrieval and stores those questions
and detailed results encrypted. Aggregate summaries report integrity, mode, counts
and timings only. Literal probes test ingestion and retrieval plumbing, not natural
question accuracy. For that, author source-checked questions and required facts in
an encrypted versioned label set outside the checkout, keep related events in one
split, then use `mr_private_eval.py --labels labels.enc --output results.enc`.
Final answer correctness requires a separate answer-generation evaluation.
Existing encrypted source-probe labels stay unchanged on subsequent runs. Failed
retrieval checks produce `failed_checks`, even if source collection completed.
`--no-collect` retests the existing corpus and reuses its complete vectors. The GUI
also measures first-query startup in separate Python processes; reported process
times include loading the encrypted benchmark question and do not flush OS caches.
Startup workers explicitly use the same top-five budget. Startup baseline misses
are reported as baseline observations; acceptance still requires preserved
top-five file/evidence coverage and successful fresh-process hybrid retrieval.

## Explicit source names and natural rewording

An explicitly named file narrows retrieval before the global candidate cutoff:

```powershell
python mr.py private --store C:\private\corpus.enc search "档案《offer-example》里，每年固定能拿多少钱？" --scope personal
python mr.py private --store C:\private\corpus.enc search 'In the record named "offer-example", how much is the fixed yearly pay?' --scope personal
```

The reference must exactly identify a file stem, filename, or stored relative path
within the selected scope. Same-name files require a complete relative path.
Missing, ambiguous, excluded, or wrong-scope references return no evidence instead
of neighbouring files. Ordinary book-title quotation alone does not select a file.
Within an identified file, local semantic retrieval chooses a passage and the
original evidence is still available; selecting a file does not verify its facts.

`mr_private_benchmark.py` compares BM25 and hybrid using identical encrypted
questions, original-source hashes, code fingerprints and a top-five budget. It can
also compare a trusted previous private-feature code checkout in isolated workers.
The controller consumes a held-out set before any mode searches; failed or
interrupted runs cannot be renamed into fresh tests. Detailed reports stay encrypted.

```powershell
python -m pip install -r requirements-private.txt -r requirements-private-semantic.txt
python scripts/setup_private_semantic.py
python scripts/verify_private_semantic.py --output .cache/private-semantic.json
python mr_private_benchmark.py --gui --author --store C:\private\corpus.enc --output-dir C:\private\evaluation\natural-001
python mr_private_benchmark.py --store C:\private\corpus.enc --labels C:\private\labels.enc --output C:\private\results.enc --summary C:\private\summary.json --baseline-code C:\code\previous-private-feature
```

The optional authoring helper uses fixed everyday Chinese/English wording and
verifiable original source fields. It reads no retrieval results, excludes prior
evaluation documents and curated narratives, keeps related language variants in
one event group, and freezes the development and held-out labels before searching.
Those generated questions include source-name context. Their scores do not establish
unqualified conversational recall, independently collected user-question accuracy,
calibrated refusal, or final-answer correctness. Existing probe and natural labels
remain unchanged; earlier observations are rendered separately as known regressions.

The fictional public regression command never unlocks a real corpus and denies
socket connections during model inference. It also checks evidence at the end of a
long named source and rejects a nonexistent named source. The optional installed
source-collector integration tests use `MR_PRIVATE_SOURCE_TOOLS`; otherwise they skip.
