"""In-memory inverted BM25, with exact file-level top-k and no plaintext cache."""
from array import array
from collections import Counter
import base64
import heapq
import math

from mr_index import tokenize


class PrivateBM25:
    K1, B = 1.5, 0.75

    def __init__(self, chunks):
        self.chunks = chunks
        self.N = len(chunks)
        self.lengths = array('I')
        self.postings = {}
        for i, chunk in enumerate(chunks):
            tokens = tokenize(chunk['file'] + ' ' + chunk['heading'] + ' ' + chunk['text'])
            self.lengths.append(len(tokens))
            for token, count in Counter(tokens).items():
                self.postings.setdefault(token, array('I')).extend((i, count))
        self.avgdl = sum(self.lengths) / max(1, self.N)

    def to_state(self):
        return {'version': 1, 'lengths': base64.b64encode(self.lengths.tobytes()).decode('ascii'),
                'postings': {token: base64.b64encode(values.tobytes()).decode('ascii')
                             for token, values in self.postings.items()}}

    @classmethod
    def from_state(cls, chunks, state):
        if state.get('version') != 1:
            raise ValueError('Unsupported private lexical cache')
        value = cls.__new__(cls)
        value.chunks, value.N = chunks, len(chunks)
        value.lengths = array('I')
        value.lengths.frombytes(base64.b64decode(state['lengths'], validate=True))
        if len(value.lengths) != value.N:
            raise ValueError('Invalid private lexical lengths')
        value.postings = {}
        for token, raw in state['postings'].items():
            posting = array('I')
            posting.frombytes(base64.b64decode(raw, validate=True))
            if len(posting) % 2 or any(i >= value.N for i in posting[::2]) or any(n == 0 for n in posting[1::2]):
                raise ValueError('Invalid private lexical posting')
            value.postings[token] = posting
        value.avgdl = sum(value.lengths) / max(1, value.N)
        return value

    def search(self, query, topk=50, mask=None, group_by_document=False):
        scores = {}
        for token, multiplicity in Counter(tokenize(query)).items():
            posting = self.postings.get(token)
            if not posting:
                continue
            df = len(posting) // 2
            idf = math.log(1 + (self.N - df + .5) / (df + .5))
            for j in range(0, len(posting), 2):
                i, frequency = posting[j], posting[j + 1]
                if mask is not None and i not in mask:
                    continue
                value = idf * frequency * (self.K1 + 1) / (
                    frequency + self.K1 * (1 - self.B + self.B * self.lengths[i] / self.avgdl))
                scores[i] = scores.get(i, 0.0) + multiplicity * value
        if group_by_document:
            best = {}
            for i, score in scores.items():
                document = self.chunks[i]['document_id']
                previous = best.get(document)
                if previous is None or (-score, i) < (-scores[previous], previous):
                    best[document] = i
            candidates = best.values()
        else:
            candidates = scores.keys()
        selected = heapq.nsmallest(topk, candidates, key=lambda i: (-scores[i], i))
        return [(i, scores[i]) for i in selected]
