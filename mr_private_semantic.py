"""Local multilingual retrieval with authenticated encrypted vector storage.

Public model weights are read locally. Corpus windows, row identities, vectors
and incremental checkpoints are encrypted together, never written as .npy.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import threading

MODEL_ID = 'intfloat/multilingual-e5-small'
REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
VECTOR_VERSION = 1
ENCODER_VERSION = 1


def default_model_dir():
    base = Path(os.environ.get('LOCALAPPDATA') or Path.home() / '.local' / 'share')
    return base / 'memory-rag' / 'models' / 'multilingual-e5-small'


def chunk_signature(chunks):
    identity = [(c['document_id'], c['md5'], c.get('purpose', 'knowledge')) for c in chunks]
    return hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()


class LocalEncoder:
    """E5 requires different query/passage prefixes. No network during loading."""
    def __init__(self, directory=None):
        self.directory = Path(directory or default_model_dir()).resolve()
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        manifest = self.directory / 'model-manifest.json'
        if not manifest.is_file():
            raise FileNotFoundError('Local semantic model is not installed')
        metadata = json.loads(manifest.read_text(encoding='utf-8'))
        if metadata.get('model_id') != MODEL_ID or metadata.get('revision') != REVISION:
            raise ValueError('Unexpected semantic model revision')
        self.backend = metadata.get('backend', 'sentence_transformers')
        if self.backend not in {'onnx_int8', 'sentence_transformers'}:
            raise ValueError('Unexpected semantic backend')
        weight_file = self.directory / ('onnx/model_qint8_avx512_vnni.onnx' if self.backend == 'onnx_int8' else 'model.safetensors')
        if not weight_file.is_file() or not (self.directory / 'tokenizer.json').is_file():
            raise FileNotFoundError('Local semantic model download is incomplete')
        for path in (weight_file, self.directory / 'tokenizer.json'):
            expected = metadata.get('sha256', {}).get(path.relative_to(self.directory).as_posix())
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                while block := stream.read(4 * 1024 * 1024):
                    digest.update(block)
            if not expected or digest.hexdigest() != expected:
                raise ValueError('Local semantic model checksum mismatch')
        self.signature = hashlib.sha256(manifest.read_bytes() + str(ENCODER_VERSION).encode()).hexdigest()

    def _get(self):
        if self._model is None:
            if self.backend == 'onnx_int8':
                import onnxruntime as ort
                options = ort.SessionOptions()
                options.intra_op_num_threads = 4
                self._model = ort.InferenceSession(str(self.directory / 'onnx' / 'model_qint8_avx512_vnni.onnx'),
                                                    sess_options=options, providers=['CPUExecutionProvider'])
            else:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(str(self.directory), device='cpu',
                                                  local_files_only=True, trust_remote_code=False)
                self._model.max_seq_length = 512
        return self._model

    def _get_tokenizer(self):
        if self._tokenizer is None:
            if self.backend == 'onnx_int8':
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(str(self.directory),
                                                    local_files_only=True, trust_remote_code=False)
            else:
                self._tokenizer = self._get().tokenizer
        return self._tokenizer

    def windows(self, chunk):
        with self._lock:
            tokenizer = self._get_tokenizer()
            heading = chunk.get('heading', '')[:160]
            prefix = 'passage: ' + heading + '\n'
            budget = max(64, 508 - len(tokenizer(prefix, add_special_tokens=False)['input_ids']))
            body = chunk['text']
            encoded = tokenizer(body, add_special_tokens=False, return_offsets_mapping=True,
                                truncation=False, verbose=False)
            offsets = encoded['offset_mapping']
            if not offsets:
                return []
            result = []
            start = 0
            while start < len(offsets):
                end = min(len(offsets), start + budget)
                left, right = offsets[start][0], offsets[end - 1][1]
                text = prefix + body[left:right]
                key = hashlib.sha256((self.signature + '\0' + chunk['md5'] + '\0' + text).encode()).hexdigest()
                result.append({'key': key, 'start': left, 'end': right, 'text': text})
                if end == len(offsets):
                    break
                start = end - min(64, budget // 4)
            return result

    def encode(self, texts, query=False):
        with self._lock:
            import numpy as np
            values = ['query: ' + text for text in texts] if query else texts
            if self.backend == 'onnx_int8':
                session = self._get()
                encoded = self._get_tokenizer()(values, padding=True, truncation=True,
                                               max_length=512, return_tensors='np')
                if any(item.name == 'token_type_ids' for item in session.get_inputs()) and 'token_type_ids' not in encoded:
                    encoded['token_type_ids'] = np.zeros_like(encoded['input_ids'])
                feed = {item.name: np.asarray(encoded[item.name], dtype=np.int64)
                        for item in session.get_inputs()}
                outputs = session.run(None, feed)
                named = {item.name: value for item, value in zip(session.get_outputs(), outputs)}
                matrix = named.get('sentence_embedding')
                if matrix is None:
                    # A transformer's pooler_output is not E5's embedding.
                    # Use its attention-masked mean pooling over token states.
                    hidden = next((value for value in outputs if value.ndim == 3), None)
                    if hidden is None:
                        raise ValueError('Local model lacks token or sentence embeddings')
                    mask = np.asarray(encoded['attention_mask'], dtype=np.float32)[..., None]
                    matrix = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
                matrix = np.asarray(matrix, dtype='<f4')
                return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            return np.asarray(self._get().encode(values, batch_size=16,
                              normalize_embeddings=True, show_progress_bar=False,
                              convert_to_numpy=True), dtype='<f4')


class PrivateVector:
    def __init__(self, chunks, encoder, cache=None, loader=None, saver=None, cancelled=None):
        self.chunks, self.encoder = chunks, encoder
        self.cache, self.loader, self.saver = cache, loader, saver
        self.cancelled = cancelled or threading.Event()
        self.rows = []
        self.emb = None
        self.ready = False
        self.state = 'not_prepared'
        self._old = None
        if self.cache and self.cache.is_file() and self.loader:
            self._old = self.loader(self.cache)
            if self._old:
                self._restore(self._old)

    def _restore(self, payload):
        import numpy as np
        valid = (payload.get('vector_version') == VECTOR_VERSION and
                 payload.get('encoder_signature') == self.encoder.signature and
                 payload.get('chunk_signature') == chunk_signature(self.chunks) and
                 payload.get('complete'))
        if not valid:
            self.state = 'stale_or_incomplete'
            return
        rows = payload.get('rows', [])
        dim = payload.get('dimensions', 0)
        raw = base64.b64decode(payload.get('vectors', ''), validate=True)
        if not isinstance(dim, int) or not 1 <= dim <= 4096 or len(raw) != len(rows) * dim * 4:
            raise ValueError('Invalid encrypted vector shape')
        if any(not 0 <= r.get('chunk', -1) < len(self.chunks) for r in rows):
            raise ValueError('Invalid encrypted vector row')
        values = np.frombuffer(raw, dtype='<f4').reshape(len(rows), dim).copy()
        if not np.isfinite(values).all():
            raise ValueError('Invalid encrypted vector values')
        self.rows, self.emb = rows, values
        self._old = None
        self.ready, self.state = True, 'ready'

    def _payload(self, rows, values, complete):
        import numpy as np
        matrix = np.asarray(values, dtype='<f4')
        return {'version': 1, 'vector_version': VECTOR_VERSION,
                'encoder_signature': self.encoder.signature,
                'model_id': MODEL_ID, 'model_revision': REVISION,
                'chunk_signature': chunk_signature(self.chunks), 'complete': complete,
                'rows': rows, 'dimensions': matrix.shape[1],
                'vectors': base64.b64encode(matrix.tobytes()).decode('ascii')}

    def prepare(self, progress=None):
        import numpy as np
        if self.ready:
            if progress:
                progress(len(self.rows), len(self.rows))
            return {'windows': len(self.rows), 'encoded': 0, 'reused': len(self.rows)}
        old = {}
        if self._old and self._old.get('vector_version') == VECTOR_VERSION and self._old.get('encoder_signature') == self.encoder.signature:
            raw = base64.b64decode(self._old['vectors'], validate=True)
            dim = self._old['dimensions']
            matrix = np.frombuffer(raw, dtype='<f4').reshape(len(self._old['rows']), dim)
            if not np.isfinite(matrix).all():
                raise ValueError('Invalid checkpoint vectors')
            old = {row['key']: matrix[i] for i, row in enumerate(self._old['rows'])}
        windows = []
        for i, chunk in enumerate(self.chunks):
            if self.cancelled.is_set():
                raise InterruptedError('Session locked')
            for window in self.encoder.windows(chunk):
                windows.append({**window, 'chunk': i})
        rows, values, encoded, reused = [], [], 0, 0
        batch = []
        def checkpoint(complete=False):
            if self.cancelled.is_set():
                raise InterruptedError('Session locked')
            if values and self.cache and self.saver:
                self.saver(self.cache, self._payload(rows, values, complete))
        def flush():
            nonlocal encoded
            if not batch:
                return
            if self.cancelled.is_set():
                raise InterruptedError('Session locked')
            vectors = self.encoder.encode([w['text'] for w in batch])
            if len(vectors) != len(batch) or not np.isfinite(vectors).all():
                raise ValueError('Invalid model output')
            for window, vector in zip(batch, vectors):
                rows.append({k: window[k] for k in ('key', 'chunk', 'start', 'end')})
                values.append(vector)
            encoded += len(batch)
            batch.clear()
            if progress:
                progress(len(rows), len(windows))
            if len(rows) % 512 == 0:
                checkpoint()
        for window in windows:
            if self.cancelled.is_set():
                raise InterruptedError('Session locked')
            if window['key'] in old:
                flush()
                rows.append({k: window[k] for k in ('key', 'chunk', 'start', 'end')})
                values.append(old[window['key']])
                reused += 1
            else:
                batch.append(window)
                if len(batch) == 16:
                    flush()
        flush()
        if not values:
            self.state = 'empty'
            return {'windows': 0, 'encoded': 0, 'reused': 0}
        checkpoint(complete=True)
        if self.cancelled.is_set():
            raise InterruptedError('Session locked')
        self.rows, self.emb = rows, np.stack(values)
        self._old = None
        self.ready, self.state = True, 'ready'
        return {'windows': len(rows), 'encoded': encoded, 'reused': reused}

    def search(self, query, topk=50, group_by_document=False):
        if self.cancelled.is_set():
            raise InterruptedError('Session locked')
        if not self.ready or self.emb is None:
            return []
        vector = self.encoder.encode([query], query=True)[0]
        scores = self.emb @ vector
        best = {}
        for row, score in zip(self.rows, scores):
            i = row['chunk']
            unit = self.chunks[i]['document_id'] if group_by_document else i
            previous = best.get(unit)
            if previous is None or (-float(score), i) < (-previous[1], previous[0]):
                best[unit] = (i, float(score))
        return sorted(best.values(), key=lambda r: (-r[1], r[0]))[:topk]

    def close(self):
        self.cancelled.set()
        if self.emb is not None:
            self.emb.fill(0)
        self.emb, self._old = None, None
        self.rows.clear()
        self.ready = False
