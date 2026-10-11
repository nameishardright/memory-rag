"""Password-unlocked, local-only retrieval over encrypted family-memory journals.

No corpus text, tokens, query logs, or passwords are persisted in plaintext.
This opt-in path does not change ordinary memory-rag redaction or corpora.
"""
import argparse
from collections import Counter, defaultdict
import datetime as dt
import getpass
import hashlib
import json
import os
import re
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import threading
import zlib

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import mr_corpus
from mr_index import rrf, rrf_files
from mr_private_index import PRIVATE_TOKENIZER, PrivateBM25
from mr_private_policy import (SCOPES, classify, load_policy, policy_signature, project_group)

JOURNAL_MAGIC = b"FAMILYMEMORYv1\x00"
STORE_MAGIC = b"MEMORYRAGPRIVATEv1\x00"
JOURNALS = ("inventory.enc", "bulk.enc", "extraction.enc", "refined.enc", "core.enc",
            "narrative.enc", "media.enc", "onenote.enc")
MAX_FRAME = 512 * 1024 * 1024
PERSONAL_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".rtf",
    ".txt", ".md", ".csv", ".html", ".htm", ".json", ".ini", ".config",
    ".env", ".log", ".url", ".sqlite", ".db", ".png", ".jpg", ".jpeg",
    ".bmp", ".tif", ".tiff", ".webp", ".gif",
}


def now():
    return dt.datetime.now().astimezone().isoformat()


def default_store():
    explicit = os.environ.get('MR_PRIVATE_STORE')
    if explicit:
        return Path(explicit)
    onedrive = Path(os.environ.get('OneDrive') or Path.home() / 'OneDrive')
    if onedrive.is_dir():
        return onedrive / '_Private-Memory-RAG' / 'corpus.enc'
    return Path.home() / 'Documents' / 'Private-Memory-RAG' / 'corpus.enc'


def key_for(password, salt):
    if not password:
        raise ValueError("An archive password is required")
    return PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt,
                     iterations=600000).derive(password.encode("utf-8"))


def read_journal(path, password):
    """Read only complete authenticated frames present when the file was opened.

    Collectors may append concurrently. Only a partial final frame is ignored;
    invalid authentication, headers, and oversized frames fail closed.
    """
    with Path(path).open("rb") as stream:
        limit = os.fstat(stream.fileno()).st_size
        if stream.read(len(JOURNAL_MAGIC)) != JOURNAL_MAGIC:
            raise ValueError("Unsupported source journal")
        salt = stream.read(16)
        if len(salt) != 16:
            raise ValueError("Incomplete source journal header")
        aes = AESGCM(key_for(password, salt))
        while stream.tell() < limit:
            if limit - stream.tell() < 4:
                break
            size = struct.unpack("<I", stream.read(4))[0]
            if not 28 <= size <= MAX_FRAME:
                raise ValueError("Invalid source frame length")
            if limit - stream.tell() < size:
                break
            data = stream.read(size)
            if len(data) != size:
                break
            yield json.loads(zlib.decompress(aes.decrypt(data[:12], data[12:], JOURNAL_MAGIC)))


def append_journal(path, password, rows):
    """Append authenticated frames; no plaintext temporary file is created."""
    path = Path(path)
    if path.exists():
        # Authenticate the existing journal before adding data or recovering
        # an incomplete tail. A corrupted complete frame is never repaired.
        for _row in read_journal(path, password):
            pass
        with path.open('r+b') as stream:
            stream.seek(len(JOURNAL_MAGIC))
            salt = stream.read(16)
            while True:
                start = stream.tell()
                header = stream.read(4)
                if not header:
                    break
                if len(header) != 4:
                    stream.truncate(start)
                    break
                size = struct.unpack('<I', header)[0]
                if len(stream.read(size)) != size:
                    stream.truncate(start)
                    break
    else:
        salt = os.urandom(16)
        with path.open('xb') as stream:
            stream.write(JOURNAL_MAGIC + salt)
    aes = AESGCM(key_for(password, salt))
    count = 0
    with path.open('ab') as stream:
        for row in rows:
            nonce = os.urandom(12)
            raw = zlib.compress(json.dumps(row, ensure_ascii=False).encode('utf-8'), 6)
            data = nonce + aes.encrypt(nonce, raw, JOURNAL_MAGIC)
            stream.write(struct.pack('<I', len(data)) + data)
            stream.flush()
            count += 1
    return count


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".pending", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_store(path, password, payload):
    salt, nonce = os.urandom(16), os.urandom(12)
    raw = zlib.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"), 6)
    sealed = AESGCM(key_for(password, salt)).encrypt(nonce, raw, STORE_MAGIC)
    atomic_write(path, STORE_MAGIC + salt + nonce + sealed)


def load_store(path, password):
    data = Path(path).read_bytes()
    n = len(STORE_MAGIC)
    if not data.startswith(STORE_MAGIC) or len(data) < n + 44:
        raise ValueError("Unsupported encrypted RAG store")
    raw = AESGCM(key_for(password, data[n:n + 16])).decrypt(
        data[n + 16:n + 28], data[n + 28:], STORE_MAGIC)
    payload = json.loads(zlib.decompress(raw))
    if payload.get("version") != 1:
        raise ValueError("Unsupported RAG payload version")
    return payload


def source_signature(source):
    result = []
    for name in JOURNALS:
        path = Path(source) / name
        if path.is_file():
            stat = path.stat()
            result.append([name, stat.st_size, stat.st_mtime_ns])
    return result


def _matches(row, target):
    return bool(target and row.get("mtime_ns") == target.get("mtime_ns")
                and row.get("size") == target.get("size"))


def make_chunks(documents):
    chunks = []
    for doc in documents:
        ordinal = 0
        for heading, text in mr_corpus._split_heading(doc["text"]):
            for head, body in mr_corpus._hard_wrap(heading, text, strategy="paragraph"):
                ordinal += 1
                digest = mr_corpus.chunk_key(doc["file"], head, body)
                chunks.append({"id": doc["id"] + ":" + str(ordinal), "document_id": doc["id"],
                               "source": "private_" + doc["scope"], "scope": doc["scope"],
                               "file": doc["file"], "heading": head, "text": body,
                               "md5": digest, "purpose": doc.get("purpose", "knowledge"),
                               "evidence_kind": "navigation" if doc.get("purpose") == "navigation" else "body"})
    return chunks


def _deduplicate(documents):
    unique, by_text, duplicates = [], {}, 0
    for document in documents:
        key = (document['scope'], document.get('purpose', 'knowledge'), document['text_sha256'])
        previous = by_text.get(key)
        if previous is None:
            by_text[key] = document
            unique.append(document)
        else:
            previous.setdefault('duplicate_sources', []).append(
                {k: document.get(k) for k in ('id', 'file', 'source_sha256', 'text_sha256', 'method')})
            duplicates += 1
    return unique, duplicates


def _project_catalogue(documents):
    groups = defaultdict(list)
    for document in documents:
        group = project_group(document['file'])
        if document['scope'] == 'technical' and group:
            groups[group].append(document)
    summaries = []
    for group, docs in sorted(groups.items()):
        readmes = [d for d in docs if Path(d['file'].replace('\\', '/')).name.casefold() in
                   {'readme.md', 'architecture.md', 'overview.md', '设计说明.md', '项目说明.md'}]
        body = '# 项目资料导航\n\n项目目录：' + group
        body += '\n已收录项目文件：' + str(len(docs))
        body += '\n这是目录导航，不证明项目仍在维护，也不证明个人贡献。完整源码请选择“代码项目”。\n'
        for document in readmes[:3]:
            body += '\n## 原说明文件：' + document['file'] + '\n\n' + document['text'][:5000] + '\n'
        summaries.append({'id': 'project-summary:' + hashlib.sha256(group.encode()).hexdigest()[:24],
                          'file': '@projects/' + group, 'scope': 'personal', 'purpose': 'navigation',
                          'text': body, 'text_sha256': hashlib.sha256(body.encode()).hexdigest(),
                          'source_sha256': None, 'method': 'project_catalogue', 'warnings': [],
                          'source_refs': [{'file': d['file'], 'id': d['id'], 'source_sha256': d.get('source_sha256')}
                                          for d in readmes[:3]]})
    return summaries


def partition_fingerprint(documents, policy_hash, source_root):
    digest = hashlib.sha256((policy_hash + str(source_root)).encode('utf-8'))
    for document in documents:
        metadata = {k: v for k, v in document.items() if k != 'text'}
        digest.update(json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode('utf-8'))
    return digest.hexdigest()


def prune_generated_partitions(store, password, protected, max_age=86400):
    """Only retire authenticated generated shards; never remove source journals."""
    store = Path(store).resolve()
    from cryptography.exceptions import InvalidTag
    pattern = re.compile(re.escape(store.stem) + r'\.([0-9a-f]{16})\.(technical|stock|history)\.enc')
    removed = 0
    for path in store.parent.glob(store.stem + '.*.*.enc'):
        match = pattern.fullmatch(path.name)
        if not match or path.name in protected or path.resolve().parent != store.parent or path.is_symlink(): continue
        if time.time() - path.stat().st_mtime < max_age: continue
        try:
            payload = load_store(path, password)
            valid = (payload.get('partitioned') and payload.get('generation') == match.group(1)
                     and payload.get('scope') == match.group(2))
            del payload
            if valid:
                path.unlink()
                removed += 1
        except (OSError, ValueError, InvalidTag):
            continue
    return removed


def import_journals(source, store, password, partitioned=False, policy=None):
    source, store = Path(source).resolve(), Path(store).resolve()
    repo = Path(__file__).resolve().parent
    if store.is_relative_to(repo):
        raise ValueError("Private store must be outside the source checkout")
    signature = source_signature(source)
    if not signature:
        raise ValueError("No encrypted source journals found")
    policy = policy or load_policy()
    previous_partitions = {}
    if partitioned and store.is_file():
        previous_payload = load_store(store, password)
        previous_partitions = previous_payload.get('shards', {})
        del previous_payload
    latest, current, inventory, narrative, external_inventories = {}, {}, None, None, {}
    archive_inventories = {}
    for name, _size, _mtime in signature:
        for row in read_journal(source / name, password):
            kind = row.get("record_type")
            if kind == "inventory":
                if inventory is None or row["time"] > inventory["time"]:
                    inventory = row
                    current = {r['path']: r for r in row['files']}
            elif kind in {"file", "archive_member"}:
                path = row['path']
                target = current.get(path)
                old = latest.get(path)
                def quality(record):
                    return (_matches(record, target), bool(record.get('read_ok')),
                            int(record.get('extractor_version', 0)), bool(record.get('text')))
                if old and quality(old) > quality(row):
                    continue
                route = classify(path, row.get('dependency', False), row.get('text', ''), policy)
                if route['scope'] == 'exclude':
                    row = {k: v for k, v in row.items() if k != 'text'}
                row['_private_route'] = route
                latest[path] = row
            elif kind == "curated_memory":
                if narrative is None or row.get("time", "") >= narrative.get("time", ""):
                    narrative = row
            elif kind == 'external_inventory':
                old = external_inventories.get(row['owner'])
                if old is None or row['time'] >= old['time']:
                    external_inventories[row['owner']] = row
            elif kind == 'archive_done':
                archive_inventories[row['archive']] = row
    if inventory is None:
        raise ValueError("No authenticated source inventory found")
    current = {r["path"]: r for r in inventory["files"]}
    documents, stale, excluded = [], 0, Counter()
    for path, row in sorted(latest.items()):
        if row['record_type'] == 'file' and path not in current:
            continue
        if row["record_type"] == "archive_member":
            owner = row.get("archive", path.split("!/")[0])
            if row.get('method') == 'onenote_com' and owner in external_inventories and path not in external_inventories[owner]['paths']:
                continue
            parent = latest.get(owner)
            if parent is None or not _matches(parent, current.get(owner)) or parent.get("changed_during_read"):
                stale += 1
                continue
            archive_inventory = archive_inventories.get(owner)
            if archive_inventory and (path not in archive_inventory.get('members', []) or
                                      archive_inventory.get('source_sha256') != parent.get('sha256')):
                stale += 1
                continue
            if row.get('archive_source_sha256') and row['archive_source_sha256'] != parent.get('sha256'):
                stale += 1
                continue
        route = row.get('_private_route') or classify(path, row.get('dependency', False), row.get('text', ''), policy)
        if route['scope'] == 'exclude':
            excluded[route['reason']] += 1
            continue
        if row.get('method') in {'shortcut_target_not_read', 'web_notebook_link_only',
                                'media_not_transcribed', 'binary_hashed', 'unsupported_binary',
                                'password_locked_pdf', 'policy_excluded'}:
            continue
        if row["record_type"] == "file" and (not _matches(row, current.get(path))
                                               or row.get("changed_during_read")):
            stale += 1
            continue
        if not row.get("read_ok") or not row.get("text", "").strip():
            continue
        scope = route['scope']
        document_id = hashlib.sha256(path.encode('utf-8')).hexdigest()[:24]
        if partitioned and scope != 'personal':
            document_id = scope + ':' + document_id
        documents.append({"id": document_id,
                          "file": path, "scope": scope, "text": row["text"],
                          'purpose': route['purpose'], 'route_reason': route['reason'],
                          "source_sha256": row.get("sha256"),
                          'source_size': row.get('size'), 'source_mtime_ns': row.get('mtime_ns'),
                          "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                          "method": row.get("method"), "warnings": row.get("warnings", [])})
        if row.get('source_updated_at'):
            documents[-1]['source_updated_at'] = row['source_updated_at']
    if narrative and narrative.get("text", "").strip():
        text = narrative["text"]
        documents.insert(0, {"id": "curated-family-memory", "file": "01_个人与家庭记忆.md",
                             "scope": "personal", "text": text, "method": "curated_memory",
                             "source_sha256": None, "warnings": [],
                             "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    documents, duplicate_count = _deduplicate(documents)
    documents.extend(_project_catalogue(documents))
    file_rows = [r for r in latest.values() if r['record_type'] == 'file' and r['path'] in current]
    read_ok = sum(bool(r.get("read_ok")) and _matches(r, current.get(r["path"]))
                  and not r.get("changed_during_read") for r in file_rows)
    eligible = {path for path, row in current.items()
                if (latest.get(path, {}).get('_private_route') or
                    classify(path, row.get('dependency', False), policy=policy))['scope'] != 'exclude'}
    eligible_rows = [r for r in file_rows if r['path'] in eligible]
    gaps = Counter()
    gaps_by_scope = defaultdict(Counter)
    gap_rows = []
    for row in eligible_rows:
        method = row.get('method', 'unknown')
        if not _matches(row, current.get(row['path'])) or row.get('changed_during_read'):
            gaps['source_changed_since_extraction'] += 1
            gaps_by_scope[row['_private_route']['scope']]['source_changed_since_extraction'] += 1
            gap_rows.append({'path': row['path'], 'method': 'source_changed_since_extraction', 'warnings': []})
        elif not row.get('read_ok') or not row.get('text', '').strip() or row.get('warnings'):
            if method == 'archive_contents_indexed' and not row.get('warnings') and row.get('archive_member_count', 0):
                continue
            if method == 'web_notebook_link_only' and external_inventories.get(row['path'], {}).get('paths'):
                continue
            gaps[method] += 1
            gaps_by_scope[row['_private_route']['scope']][method] += 1
            gap_rows.append({k: row.get(k) for k in ('path', 'method', 'warnings', 'read_ok')})
    for row in latest.values():
        if row.get('record_type') != 'archive_member': continue
        path = row['path']; owner = row.get('archive', path.split('!/')[0]); parent = latest.get(owner)
        if parent is None or not _matches(parent, current.get(owner)): continue
        route = row.get('_private_route') or classify(path, row.get('dependency', False), policy=policy)
        if route['scope'] == 'exclude': continue
        members = archive_inventories.get(owner)
        if members and (path not in members.get('members', []) or members.get('source_sha256') != parent.get('sha256')): continue
        if row.get('method') == 'onenote_com' and owner in external_inventories and path not in external_inventories[owner]['paths']: continue
        if row.get('method') == 'nested_zip_container' and row.get('nested_member_count', 0) and not row.get('warnings'): continue
        if not row.get('read_ok') or not row.get('text', '').strip() or row.get('warnings'):
            method = row.get('method', 'archive_member_not_extracted')
            gaps[method] += 1
            gaps_by_scope[route['scope']][method] += 1
            gap_rows.append({k: row.get(k) for k in ('path', 'method', 'warnings', 'read_ok')})
    scope_counts = Counter(d['scope'] for d in documents)
    coverage = {"visible_files": len(current), "read_success": read_ok,
                "read_failed": sum(not r.get("read_ok") for r in file_rows),
                "not_processed": len(set(current) - set(latest)), "stale_records_excluded": stale,
                "documents": len(documents),
                'eligible_files': len(eligible), 'excluded_files': len(current) - len(eligible),
                'eligible_not_processed': len(eligible - set(latest)),
                'excluded_index_records': dict(excluded), 'duplicate_documents_collapsed': duplicate_count,
                'content_gaps': dict(gaps), 'scope_documents': dict(scope_counts),
                'content_gaps_by_scope': {scope: dict(counts) for scope, counts in gaps_by_scope.items()},
                "personal_documents": sum(d["scope"] == "personal" for d in documents),
                "technical_documents": sum(d["scope"] == "technical" for d in documents),
                "text_characters": sum(len(d["text"]) for d in documents),
                "source_status": "incomplete" if (eligible - set(latest) or gaps) else "eligible_files_processed",
                "content_limits": "Only extracted text is searchable; locked files, unsynced files, "
                                  "Personal Vault, and untranscribed media remain subject to source coverage."}
    generation = os.urandom(8).hex()
    base = {"version": 1, "updated_at": now(), "source": str(source), 'generation': generation,
            'source_root': inventory.get('root'),
            'policy_sha256': policy_signature(policy), 'partitioned': partitioned,
            'coverage': coverage, 'mode': 'bm25_inverted', 'tokenizer': PRIVATE_TOKENIZER}
    partitions, total_chunks = {}, 0
    if partitioned:
        for scope in SCOPES:
            docs = [d for d in documents if d['scope'] == scope]
            fingerprint = partition_fingerprint(docs, base['policy_sha256'], base['source_root'])
            previous = previous_partitions.get(scope, {})
            name = previous.get('file', '')
            if scope != 'personal' and docs and previous.get('content_sha256') == fingerprint and Path(name).name == name:
                path = store.with_name(name)
                if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == previous.get('sha256'):
                    partitions[scope] = previous
                    total_chunks += previous['chunks']
                    continue
            chunks = make_chunks(docs)
            total_chunks += len(chunks)
            part = {**base, 'documents': docs, 'chunks': chunks, 'scope': scope, 'content_sha256': fingerprint}
            if scope == 'personal':
                payload = part
            elif docs:
                filename = store.stem + '.' + generation + '.' + scope + '.enc'
                path = store.with_name(filename)
                save_store(path, password, part)
                partitions[scope] = {'file': filename, 'documents': len(docs), 'chunks': len(chunks),
                                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                     'content_sha256': fingerprint, 'generation': generation}
        payload['shards'] = partitions
    else:
        chunks = make_chunks(documents)
        total_chunks = len(chunks)
        payload = {**base, "documents": documents, "chunks": chunks}
    coverage['chunks'] = total_chunks
    coverage['default_documents'] = sum(d['scope'] == 'personal' for d in documents)
    coverage['default_chunks'] = len(payload['chunks']) if partitioned else sum(c['scope'] == 'personal' for c in payload['chunks'])
    payload.update({
               "source_signature": signature,
               "coverage": coverage, "mode": "bm25", "tokenizer": PRIVATE_TOKENIZER}
    )
    # Publish the root last: readers either see the previous generation or a
    # complete new generation; never a mixture of sidecar partitions.
    save_store(store, password, payload)
    status = {k: payload[k] for k in ("version", "updated_at", "coverage", "mode", "tokenizer")}
    atomic_write(store.with_suffix(".status.json"), json.dumps(status, indent=2).encode("utf-8"))
    save_store(store.with_name('coverage-review.enc'), password,
               {'version': 1, 'updated_at': payload['updated_at'], 'coverage': coverage,
                'gaps': gap_rows,
                'not_processed': sorted(eligible - set(latest))})
    if partitioned:
        protected = {metadata['file'] for metadata in [*partitions.values(), *previous_partitions.values()]}
        retired = prune_generated_partitions(store, password, protected)
        atomic_write(store.with_name('index-retention.status.json'), json.dumps({'updated_at': now(),
                     'retired_generated_partitions': retired, 'minimum_age_hours': 24,
                     'current_and_previous_references_protected': True}).encode('utf-8'))
    return payload


class PrivateRAG:
    def __init__(self, payload, partition_loader=None, *, mode='bm25', index_loader=None,
                 index_saver=None, encoder_factory=None, auto_scope_chunk_limit=200000):
        self.payload = payload
        self.auto_scope_chunk_limit = auto_scope_chunk_limit
        self.documents = {d["id"]: d for d in payload["documents"]}
        self.aliases = {alias['id']: (document['id'], alias) for document in payload['documents']
                        for alias in document.get('duplicate_sources', [])}
        self.indexes = {}
        self.partition_loader = partition_loader
        self.chunks = list(payload['chunks'])
        self.loaded_scopes = {d['scope'] for d in payload['documents']} | {'personal'}
        self.mode = mode
        self.index_loader, self.index_saver = index_loader, index_saver
        self.encoder_factory = encoder_factory
        self.encoder = None
        self.vectors = {}
        self.semantic_states = {}
        self.cancelled = threading.Event()
        self.document_chunks = defaultdict(list)
        for chunk in self.chunks:
            self.document_chunks[chunk['document_id']].append(chunk)

    def _active(self):
        if self.cancelled.is_set():
            raise InterruptedError('Private session is locked')

    def ensure_scope(self, scope):
        self._active()
        requested = SCOPES if scope == 'all' else (scope,)
        for value in requested:
            if value in self.loaded_scopes:
                continue
            if self.partition_loader and value in self.payload.get('shards', {}):
                part = self.partition_loader(value)
                self.documents.update((d['id'], d) for d in part['documents'])
                self.aliases.update((alias['id'], (document['id'], alias)) for document in part['documents']
                                    for alias in document.get('duplicate_sources', []))
                self.chunks.extend(part['chunks'])
                for chunk in part['chunks']:
                    self.document_chunks[chunk['document_id']].append(chunk)
            self.loaded_scopes.add(value)

    def _index(self, scope):
        self._active()
        if scope not in self.indexes:
            self.ensure_scope(scope)
            chunks = [c for c in self.chunks if (scope == 'all' or c['scope'] == scope)
                      and c.get('purpose') not in {'evaluation', 'excluded'}]
            cached = self.index_loader(scope, 'lexical', chunks) if self.index_loader else None
            index = PrivateBM25.from_state(chunks, cached['lexical']) if cached else PrivateBM25(chunks)
            self.indexes[scope] = (chunks, index)
            if not cached and self.index_saver and scope != 'all':
                self.index_saver(scope, 'lexical', chunks, {'lexical': index.to_state()})
        return self.indexes[scope]

    def _vector(self, scope, chunks):
        if scope == 'all':
            self.semantic_states[scope] = 'choose_one_scope_for_semantics'
            return None
        if scope in self.vectors:
            return self.vectors[scope]
        if not self.encoder_factory:
            self.semantic_states[scope] = 'model_not_configured'
            return None
        try:
            from mr_private_semantic import PrivateVector
            if self.encoder is None:
                self.encoder = self.encoder_factory()
            cache = self.index_loader(scope, 'semantic_path', chunks) if self.index_loader else None
            loader = (lambda path: self.index_loader(scope, 'semantic', chunks)) if self.index_loader else None
            saver = (lambda path, data: self.index_saver(scope, 'semantic', chunks, data)) if self.index_saver else None
            vector = PrivateVector(chunks, self.encoder, cache=cache, loader=loader,
                                   saver=saver, cancelled=self.cancelled)
            self.vectors[scope] = vector
            self.semantic_states[scope] = vector.state
            return vector
        except (OSError, ValueError, ImportError, InvalidTag) as error:
            self.semantic_states[scope] = 'unavailable:' + type(error).__name__
            return None

    def _document_passages(self, scope, document_id, candidates, query, topk):
        """Dense ranking of one document's passages; empty unless a ready local index exists."""
        if self.mode != 'hybrid' or scope not in SCOPES or not query.strip():
            return []
        try:
            chunks, _index = self._index(scope)
            vector = self._vector(scope, chunks)
            if vector is None or not vector.ready:
                return []
            position = {chunk['id']: i for i, chunk in enumerate(candidates)}
            return [(position[vector.chunks[i]['id']], score)
                    for i, score in vector.passages(query, document_id, topk)
                    if vector.chunks[i]['id'] in position]
        except InterruptedError:
            raise
        except Exception as error:
            self.semantic_states[scope] = 'unavailable:' + type(error).__name__
            return []

    def prepare_semantic(self, scope='personal', progress=None):
        if scope not in SCOPES:
            raise ValueError('Prepare one source scope at a time')
        chunks, _index = self._index(scope)
        vector = self._vector(scope, chunks)
        if vector is None:
            raise RuntimeError('Local semantic model or cache unavailable')
        result = vector.prepare(progress)
        self.semantic_states[scope] = vector.state
        return {'scope': scope, **result}

    def _deferred(self, scope):
        """A very large partition is opened by an explicit choice, not by one routing keyword."""
        if scope in self.loaded_scopes:
            return False
        metadata = self.payload.get('shards', {}).get(scope) or {}
        return metadata.get('chunks', 0) > self.auto_scope_chunk_limit

    def search(self, query, topk=5, scope="personal", *, lexical_query=None, mode=None):
        self._active()
        if scope not in (*SCOPES, 'all', 'auto'):
            raise ValueError("Unknown source scope")
        if not query.strip() or not 1 <= topk <= 50:
            raise ValueError("A nonempty query and topk between 1 and 50 are required")
        selected_mode = mode or self.mode
        if selected_mode not in {'bm25', 'hybrid'}:
            raise ValueError('Unknown private retrieval mode')
        from mr_private_routing import route_query, named_source
        if scope != 'auto':
            return self._search_scope(query, topk, {'scope': scope, 'automatic': False, 'reason': 'explicit'},
                                      lexical_query, selected_mode)
        route = route_query(query)
        primary = route['scope']
        if primary == 'personal':
            return self._search_scope(query, topk, route, lexical_query, selected_mode)
        # Keyword routing is a guess: an offer letter mentions stock options and a
        # resume mentions projects. Personal records are therefore always consulted.
        if self._deferred(primary):
            return self._search_scope(query, topk, {**route, 'scope': 'personal', 'deferred_scope': primary},
                                      lexical_query, selected_mode)
        first = self._search_scope(query, topk, {**route, 'also_searched': ['personal']}, lexical_query, selected_mode)
        named = named_source(query) is not None
        if named and first['hits']:
            return first
        second = self._search_scope(query, topk, {**route, 'scope': 'personal', 'also_searched': [primary]},
                                    lexical_query, selected_mode)
        if named:
            return second if second['hits'] else first
        hits, seen = [], set()
        for position in range(max(len(first['hits']), len(second['hits']))):
            for result in (first, second):
                if position < len(result['hits']) and result['hits'][position]['document_id'] not in seen:
                    seen.add(result['hits'][position]['document_id'])
                    hits.append(result['hits'][position])
        hits = hits[:topk]
        return {**first, 'hits': hits, 'modes': {primary: first['mode'], 'personal': second['mode']},
                'answerability': 'unverified' if hits else 'no_evidence'}

    def _search_scope(self, query, topk, route, lexical_query, selected_mode):
        from mr_private_routing import named_source
        scope = route['scope']
        chunks, index = self._index(scope)
        reference = named_source(query)
        referenced_ids = None
        search_query = query
        if reference:
            from pathlib import PurePosixPath
            import unicodedata
            def matches_source(value):
                value = unicodedata.normalize('NFKC', value.replace('\\', '/')).casefold()
                path = PurePosixPath(value)
                return reference['name'] in {value, path.name, path.stem}
            referenced_ids = {doc['id'] for doc in self.documents.values()
                              if (scope == 'all' or doc['scope'] == scope)
                              and doc.get('purpose') not in {'evaluation', 'excluded', 'navigation'}
                              and any(matches_source(source['file']) for source in
                                      [doc, *doc.get('duplicate_sources', [])])}
            route = {**route, 'source_reference': 'unique' if len(referenced_ids) == 1 else
                     'missing' if not referenced_ids else 'ambiguous',
                     'source_reference_matches': len(referenced_ids)}
            if len(referenced_ids) != 1:
                return {'ok': True, 'mode': selected_mode, 'scope': scope, 'route': route,
                        'hits': [], 'semantic_state': 'not_searched', 'answerability': 'no_evidence',
                        'note': 'Explicit source is missing or ambiguous in the selected scope.'}
            search_query = reference['query']
            from mr_private_facts import expand_query
            lexical_query = expand_query(search_query)
        if referenced_ids is not None:
            # Search the named source before a global top-k cutoff. A relevant
            # source must not vanish behind unrelated high-scoring files.
            positions = [i for i, chunk in enumerate(chunks) if chunk['document_id'] in referenced_ids]
            local = PrivateBM25([chunks[i] for i in positions])
            bm = [(positions[i], score) for i, score in local.search(
                lexical_query or search_query, topk=max(50, topk), group_by_document=True)]
        else:
            bm = index.search(lexical_query or search_query, topk=max(50, topk), group_by_document=True)
        ranked = [(i, s, [rank, None]) for rank, (i, s) in enumerate(bm)]
        if referenced_ids is not None and not ranked and positions:
            ranked = [(positions[0], 0.0, [None, None])]
        actual_mode, vector_scores, literal_matches = 'bm25', {}, set()
        if selected_mode == 'hybrid':
            vector = self._vector(scope, chunks)
            if vector is not None and vector.ready:
                try:
                    dense = vector.search(search_query, topk=max(50, topk), group_by_document=True,
                                          **({'document_ids': referenced_ids} if referenced_ids is not None else {}))
                    vector_scores = dict(dense)
                    # Fuse the complete candidate union before truncation. Two
                    # weak votes must not displace an original literal phrase.
                    ranked = rrf_files(chunks, [bm, dense], topk=len(bm) + len(dense))
                    # Two files ranked [0, 1] and [1, 0] fuse to the same score. The
                    # question's own embedding settles that, not insertion order.
                    closeness = {chunks[i]['document_id']: score for i, score in dense}
                    ranked.sort(key=lambda row: (-round(row[1], 12),
                                                 -closeness.get(chunks[row[0]]['document_id'], -1.0)))
                    if referenced_ids is not None and dense:
                        # File identity is fixed by the user's explicit reference;
                        # use semantics to pick its answer passage rather than
                        # voting a different passage into the representative slot.
                        ranked = [(dense[0][0], ranked[0][1], ranked[0][2])]
                    needle = query.strip().casefold()
                    if len(needle) >= 12:
                        fused = {chunks[i]['document_id']: (score, legs) for i, score, legs in ranked}
                        protected = []
                        for i, _ in bm:
                            identifier = chunks[i]['document_id']
                            if needle in self.documents[identifier]['text'].casefold():
                                score, legs = fused[identifier]
                                protected.append((i, score, legs))
                                literal_matches.add(identifier)
                        # Use the BM25 passage for literal matches; a semantic
                        # passage elsewhere in the file may omit that phrase.
                        ranked = protected + [row for row in ranked
                            if chunks[row[0]]['document_id'] not in literal_matches]
                    actual_mode = 'hybrid'
                except InterruptedError:
                    raise
                except Exception as error:
                    self.semantic_states[scope] = 'unavailable:' + type(error).__name__
                    actual_mode = 'bm25(fallback)'
            else:
                actual_mode = 'bm25(fallback)'
        lexical_scores = dict(bm)
        hits, seen = [], set()
        for i, score, legs in ranked:
            chunk = chunks[i]
            if chunk["document_id"] in seen:
                continue
            seen.add(chunk["document_id"])
            hits.append({**chunk, "score": round(score, 6),
                         "source_sha256": self.documents[chunk["document_id"]].get("source_sha256"),
                         'duplicate_sources': self.documents[chunk['document_id']].get('duplicate_sources', []),
                         'lexical_score': round(lexical_scores.get(i, 0), 6),
                         'semantic_score': round(vector_scores[i], 6) if i in vector_scores else None,
                         'literal_query_match': chunk['document_id'] in literal_matches,
                         'retrieval_ranks': legs})
            if len(hits) == topk:
                break
        return {"ok": True, "mode": actual_mode, "scope": scope, 'route': route, "hits": hits,
                'semantic_state': self.semantic_states.get(scope, 'disabled'),
                'answerability': 'unverified' if hits else 'no_evidence',
                "note": "Local source retrieval. Related candidates are not verified answers; "
                        "read the evidence and check person, date, missing facts and source warnings."}

    def evidence(self, document_id, query='', max_chars=12000, max_passages=5, expected_md5=None):
        self._active()
        doc = self.get(document_id)
        if not doc.get('ok'):
            return doc
        canonical = doc.get('canonical_document_id', document_id)
        candidates = self.document_chunks.get(canonical, [])
        if expected_md5 and not any(c['md5'] == expected_md5 for c in candidates):
            return {'ok': False, 'error': 'Stale source reference; search again'}
        if not 200 <= max_chars <= 50000 or not 1 <= max_passages <= 10:
            raise ValueError('Evidence budget is outside supported bounds')
        complete = len(doc['text']) <= max_chars
        if complete:
            selected = candidates
            context = doc['text']
        else:
            from mr_private_facts import expand_query
            from mr_private_routing import named_source
            reference = named_source(query)
            question = reference['query'] if reference else query
            # A question in one language about a document in another has no
            # lexical match inside the file; the local vectors still rank its
            # passages. Without them this is the previous keyword selection.
            lexical = PrivateBM25(candidates).search(expand_query(question), topk=max_passages * 2)
            semantic = self._document_passages(doc['scope'], canonical, candidates, question, max_passages * 2)
            ranked = rrf([lexical, semantic], topk=max_passages * 2) if semantic else lexical
            positions = [i for i, _ in ranked] or [0]
            if expected_md5:
                positions.insert(0, next(i for i, c in enumerate(candidates) if c['md5'] == expected_md5))
            chosen, used = [], 0
            def take(i):
                nonlocal used
                size = len(candidates[i]['heading']) + len(candidates[i]['text']) + 4
                if i in chosen or used + size > max_chars:
                    return False
                chosen.append(i); used += size
                return True
            primary = []
            for i in dict.fromkeys(positions):
                if len(primary) == max_passages or not take(i):
                    break
                primary.append(i)
            # A value and its label are often split across a passage boundary.
            for i in primary[:2]:
                for neighbour in (i + 1, i - 1):
                    if 0 <= neighbour < len(candidates) and len(chosen) < max_passages + 4:
                        take(neighbour)
            chosen.sort()
            selected = [candidates[i] for i in chosen]
            context = '\n\n'.join(c['heading'] + '\n\n' + c['text'] for c in selected)
        metadata = {k: doc.get(k) for k in ('file', 'scope', 'source_sha256', 'text_sha256',
                    'source_mtime_ns', 'source_updated_at', 'method', 'warnings', 'purpose')}
        issues = []
        requested_years = set(re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)', query))
        source_years = set(re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)', context))
        if requested_years - source_years:
            issues.append('requested_year_not_in_evidence')
        if doc.get('warnings'):
            issues.append('source_extraction_requires_review')
        if doc.get('purpose') in {'navigation', 'historical'}:
            issues.append('navigation_or_historical_source')
        if re.search(r'(?i)现在|目前|当前|\bcurrent\b|\btoday\b', query):
            issues.append('current_status_requires_confirmation')
        roles = {'mother': r'(?i)mother|mum|妈妈|母亲', 'father': r'(?i)father|dad|爸爸|父亲',
                 'spouse': r'(?i)spouse|wife|husband|配偶|妻子|丈夫'}
        if any(re.search(pattern, query) and not re.search(pattern, context) for pattern in roles.values()):
            issues.append('requested_person_role_not_in_evidence')
        if re.search(r'(?i)appointment|预约', query) and not re.search(r'(?i)appointment|预约', context):
            issues.append('requested_appointment_not_in_evidence')
        if re.search(r'(?i)date of birth|birth date|生日|出生日期', query):
            if not re.search(r'(?i)(?:date of birth|birth date|dob|生日|出生日期)\s*[:：]\s*\S+|born on|出生于', context):
                issues.append('requested_birth_date_not_recorded')
        if re.search(r'(?i)market survey|industry estimate|target employer|no offer received|求职草稿|目标雇主', context):
            issues.append('draft_or_third_party_estimate')
        return {'ok': True, 'document_id': document_id, **metadata, 'context': context,
                'context_complete': complete,
                'passages': [{k: c[k] for k in ('id', 'heading', 'md5', 'text')} for c in selected],
                'review_reasons': issues, 'answerability': 'unverified',
                'note': 'Source dates are not proof of when a personal fact became effective.'}

    def get(self, document_id):
        self._active()
        if document_id not in self.documents and ':' in document_id:
            scope = document_id.split(':', 1)[0]
            if scope in SCOPES:
                self.ensure_scope(scope)
        doc = self.documents.get(document_id)
        if doc is None and document_id in self.aliases:
            canonical_id, alias = self.aliases[document_id]
            doc = {**self.documents[canonical_id], **alias, 'canonical_document_id': canonical_id}
        if doc is None:
            return {"ok": False, "error": "Unknown document ID"}
        return {"ok": True, **doc}

    def close(self):
        self.cancelled.set()
        for vector in self.vectors.values():
            vector.close()
        self.vectors.clear(); self.indexes.clear(); self.document_chunks.clear()
        self.documents.clear(); self.aliases.clear(); self.chunks.clear()
        self.payload = {}
        self.partition_loader = self.index_loader = self.index_saver = self.encoder_factory = None
        self.encoder = None


def load_private_rag(store, password, *, mode=None, use_index_cache=None):
    store = Path(store)
    payload = load_store(store, password)
    def loader(scope):
        metadata = payload['shards'][scope]
        name = metadata['file']
        if Path(name).name != name or not name.endswith('.' + scope + '.enc'):
            raise ValueError('Invalid private partition name')
        if metadata.get('sha256') and hashlib.sha256(store.with_name(name).read_bytes()).hexdigest() != metadata['sha256']:
            raise ValueError('Encrypted partition hash mismatch')
        part = load_store(store.with_name(name), password)
        if part.get('generation') != metadata.get('generation', payload.get('generation')) or part.get('scope') != scope:
            raise ValueError('Mismatched private partition generation')
        if metadata.get('content_sha256') and part.get('content_sha256') != metadata['content_sha256']:
            raise ValueError('Mismatched private partition content')
        return part
    from mr_private_semantic import LocalEncoder, chunk_signature
    settings_path = store.parent / 'retrieval-settings.json'
    settings = json.loads(settings_path.read_text(encoding='utf-8-sig')) if settings_path.is_file() else {}
    selected_mode = mode or settings.get('mode', 'bm25')
    def index_path(scope, kind):
        return store.with_name(store.stem + '.' + kind + '.' + scope + '.enc')
    cache_scopes = settings.get('cache_scopes', ['personal', 'technical', 'history'])
    def index_loader(scope, kind, chunks):
        if kind == 'semantic_path':
            return index_path(scope, 'semantic') if scope in cache_scopes else None
        if scope not in cache_scopes:
            return None
        path = index_path(scope, kind)
        if not path.is_file():
            return None
        data = load_store(path, password)
        if kind == 'lexical' and (data.get('chunk_signature') != chunk_signature(chunks) or data.get('tokenizer') != PRIVATE_TOKENIZER):
            return None
        return data
    def index_saver(scope, kind, chunks, data):
        if scope not in cache_scopes:
            return
        save_store(index_path(scope, kind), password,
                   {**data, 'version': 1, 'chunk_signature': chunk_signature(chunks), 'tokenizer': PRIVATE_TOKENIZER})
    # No settings means the previous BM25-only, no-cache API remains available.
    enabled = bool(settings.get('encrypted_index_cache')) if use_index_cache is None else bool(use_index_cache)
    return PrivateRAG(payload, loader, mode=selected_mode,
                      auto_scope_chunk_limit=int(settings.get('auto_scope_chunk_limit', 200000)),
                      index_loader=index_loader if enabled else None,
                      index_saver=index_saver if enabled else None,
                      encoder_factory=(lambda: LocalEncoder(settings.get('model_directory'))) if selected_mode == 'hybrid' else None)


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=default_store())
    parser.add_argument("--source", type=Path)
    parser.add_argument("--stdin-password", action="store_true", help="Read one password line from a private pipe")
    sub = parser.add_subparsers(dest="command", required=True)
    importer = sub.add_parser('import')
    importer.add_argument('--combined', action='store_true', help='Keep all scopes in one encrypted file')
    sub.add_parser("stats")
    sub.add_parser("shell")
    sub.add_parser("ui")
    cards = sub.add_parser('cards')
    cards.add_argument('--include-bank-accounts', action='store_true')
    watch = sub.add_parser("watch")
    watch.add_argument("--interval", type=int, default=300)
    watch.add_argument("--detach", action="store_true", help="Start a hidden background updater")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--scope", choices=[*SCOPES, 'all', 'auto'], default="personal")
    search.add_argument("--topk", type=int, default=5)
    search.add_argument('--compact', action='store_true', help='Return short evidence and source IDs only')
    get = sub.add_parser("get")
    get.add_argument("document_id")
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--scope', choices=SCOPES, default='personal')
    evidence = sub.add_parser('evidence')
    evidence.add_argument('document_id')
    evidence.add_argument('--query', default='')
    evidence.add_argument('--md5')
    args = parser.parse_args(argv)
    if args.command == 'ui':
        from mr_private_ui import launch
        launch(args.store)
        return 0
    password = sys.stdin.readline().rstrip("\r\n") if args.stdin_password else getpass.getpass("Archive password: ")
    if args.command in {"import", "watch"}:
        if args.source is None:
            parser.error("--source is required for import/watch")
        if args.command == "import":
            _print(import_journals(args.source, args.store, password, partitioned=not args.combined)["coverage"])
            return 0
        if args.interval < 30:
            parser.error("watch interval must be at least 30 seconds")
        if args.detach:
            # Authenticate before detaching. Transfer the password only through
            # the child's stdin pipe, never its command line or environment.
            load_store(args.store, password)
            if (args.store.parent / "stop.request").exists():
                raise ValueError("Remove stop.request before starting the updater")
            command = [sys.executable, str(Path(__file__).resolve()), "--source", str(args.source),
                       "--store", str(args.store), "--stdin-password", "watch",
                       "--interval", str(args.interval)]
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            proc.stdin.write((password + "\n").encode("utf-8"))
            proc.stdin.close()
            _print({"background_process_id": proc.pid, "interval_seconds": args.interval})
            return 0
        old_signature = None
        last_import = None
        stop = args.store.parent / "stop.request"
        watcher_status = args.store.parent / "watcher.status.json"
        def watcher_state(state, **extra):
            atomic_write(watcher_status, json.dumps({"state": state, "pid": os.getpid(),
                         "updated_at": now(), **extra}).encode("utf-8"))
        watcher_state("running")
        while not stop.exists():
            signature = (source_signature(args.source), policy_signature())
            if signature != old_signature:
                try:
                    payload = import_journals(args.source, args.store, password, partitioned=True)
                    old_signature = signature
                    last_import = payload['updated_at']
                    watcher_state("running", last_import=last_import)
                    _print({"updated_at": payload["updated_at"], **payload["coverage"]})
                    del payload
                except (InvalidTag, ValueError):
                    watcher_state("stopped_invalid_input")
                    raise
                except Exception as error:
                    watcher_state("retrying", error=type(error).__name__)
                    _print({"update_error": type(error).__name__, "will_retry": True})
            else:
                watcher_state('running', last_import=last_import)
            for _ in range(args.interval):
                if stop.exists():
                    watcher_state("stopped")
                    return 0
                time.sleep(1)
        watcher_state("stopped")
        return 0
    rag = load_private_rag(args.store, password)
    payload = rag.payload
    if args.command == "stats":
        _print({"updated_at": payload["updated_at"], **payload["coverage"]})
        return 0
    if args.command == 'cards':
        from mr_private_facts import compact_cards, extract_cards
        _print(compact_cards(extract_cards(payload['documents'], args.include_bank_accounts)))
    elif args.command == "search":
        if args.compact:
            from mr_private_facts import compact_search
            _print(compact_search(rag, args.query, args.topk, args.scope))
        else:
            _print(rag.search(args.query, args.topk, args.scope))
    elif args.command == "get":
        _print(rag.get(args.document_id))
    elif args.command == 'evidence':
        _print(rag.evidence(args.document_id, args.query, expected_md5=args.md5))
    elif args.command == 'prepare':
        _print(rag.prepare_semantic(args.scope))
    else:
        print("Private RAG unlocked locally. /stats, /all QUERY, /get ID, /reload, /quit")
        while True:
            try:
                query = input("RAG> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if query == "/quit":
                break
            if query == "/stats":
                _print({"updated_at": payload["updated_at"], **payload["coverage"]})
            elif query == "/reload":
                payload = load_store(args.store, password)
                rag = load_private_rag(args.store, password)
                payload = rag.payload
                _print({"reloaded_at": payload["updated_at"]})
            elif query.startswith("/get "):
                _print(rag.get(query[5:].strip()))
            elif query:
                _print(rag.search(query[5:] if query.startswith("/all ") else query,
                                  scope="all" if query.startswith("/all ") else "personal"))
    return 0


def entrypoint(argv=None):
    try:
        return main(argv)
    except InvalidTag:
        print("Unlock failed: incorrect password or corrupted encrypted data.", file=sys.stderr)
        return 2
    except (ValueError, OSError) as error:
        # Paths in OS exceptions may contain private source filenames.
        print("Private RAG operation failed: " + type(error).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(entrypoint())
