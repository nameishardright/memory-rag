# Agent entry points

For the optional encrypted personal RAG and its OneDrive deployment, read:

1. [OneDrive handoff](docs/AGENT-HANDOFF.md): current code, deployment, evaluation locations and remaining work.
2. [OneDrive change log](docs/CHANGELOG-RAG.md): measured changes and their limits.
3. [Private corpus guide](docs/private-corpus.md): commands, encryption and source scopes.

Keep work/study defaults separate from the optional private adapter. Private retrieval reuses the shared tokenizer and file-level fusion; encrypted storage, scoped indexes and the local multilingual model are private-adapter responsibilities.

Keep decrypted documents, actual private filenames, questions, expected facts, detailed results and credentials outside this public checkout and plaintext logs. Publish generic code, fictional fixtures and aggregate measurements only.

Preserve original labels and consumed holdout markers. Chinese/English variants and duplicate source bodies belong to the same split. Previously evaluated questions are regressions; collect and freeze new source groups before claiming a fresh evaluation. Read full source evidence before making factual claims.

Keep changes focused. For retrieval code changes, use the checks in the handoff. Documentation edits need link and diff checks; they do not require rerunning encrypted evaluations. User instructions take precedence over these repository notes.
