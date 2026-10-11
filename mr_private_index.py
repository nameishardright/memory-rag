"""In-memory inverted BM25, with exact file-level top-k and no plaintext cache."""
from array import array
from collections import Counter
import base64
import heapq
import math
import re
import unicodedata

from mr_index import TOKENIZER, _ASCII, _CJK, _cjk_cut

# Cached postings are only valid for the tokenizer that produced them.
PRIVATE_TOKENIZER = TOKENIZER + '+form-v1'

_EDGE = './:-'
_SEPARATOR = re.compile(r'[./:\-_]+')
_GROUPED_NUMBER = re.compile(r'(?<![\d.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])')
_LATIN_EXTENDED = re.compile('[À-ɏḀ-ỿ]')
_COMBINING = re.compile('(?<=[A-Za-z])[̀-ͯ]+')
_FULLWIDTH = {code: code - 0xFEE0 for code in range(0xFF01, 0xFF5F)}
_UNDECOMPOSED = {'ß': 'ss', 'æ': 'ae', 'Æ': 'AE', 'œ': 'oe', 'Œ': 'OE', 'ø': 'o', 'Ø': 'O',
                 'đ': 'd', 'Đ': 'D', 'ł': 'l', 'Ł': 'L'}


def _plain_letter(match):
    char = match.group()
    return _UNDECOMPOSED.get(char) or unicodedata.normalize('NFKD', char).encode('ascii', 'ignore').decode('ascii') or char


def fold(text):
    """Full-width forms and accented Latin letters become plain ASCII; CJK text is untouched."""
    text = text.translate(_FULLWIDTH)
    if text.isascii():
        return text
    return _COMBINING.sub('', _LATIN_EXTENDED.sub(_plain_letter, text))


def split_tokens(text):
    """Return (words, alternates) for the private corpus.

    The work/study tokenizer keeps `salary:` or `2024.txt` as one token, so a
    label, a sentence-final value or the last word of a file name never matches
    the plain word. Here the surrounding punctuation is dropped. A compound
    stays one word, so an exact identifier still outweighs a partial one, and
    its parts are returned separately as alternates: they are searchable but do
    not make a record full of dates and reference numbers count as longer.
    """
    text = fold(text)
    words, alternates = [], []
    for raw in _ASCII.findall(text):
        if raw.isalnum():
            words.append(raw.lower())
            continue
        core = raw.strip(_EDGE).lower()
        if not core:
            continue
        words.append(core)
        if core.isalnum():
            continue
        parts = {core}
        for piece in core.split('.'):
            piece = piece.strip(_EDGE)
            if len(piece) > 1 and piece not in parts:
                parts.add(piece)
                alternates.append(piece)
        for atom in _SEPARATOR.split(core):
            if len(atom) > 1 and atom not in parts:
                parts.add(atom)
                alternates.append(atom)
    if ',' in text:
        # `128,500.00` is otherwise only `128` and `500.00`.
        for match in _GROUPED_NUMBER.finditer(text):
            joined = match.group().replace(',', '')
            alternates.append(joined)
            if '.' in joined:
                alternates.append(joined.split('.')[0])
    for run in _CJK.findall(text):
        words.extend(_cjk_cut(run))
    return words, alternates


def tokenize(text):
    words, alternates = split_tokens(text)
    return words + alternates


class PrivateBM25:
    K1, B = 1.5, 0.75

    def __init__(self, chunks, tokenizer=tokenize):
        self.chunks = chunks
        self.tokenize = tokenizer
        self.N = len(chunks)
        self.lengths = array('I')
        self.postings = {}
        for i, chunk in enumerate(chunks):
            text = chunk['file'] + ' ' + chunk['heading'] + ' ' + chunk['text']
            if tokenizer is tokenize:
                tokens, alternates = split_tokens(text)
                self.lengths.append(len(tokens))
                tokens += alternates
            else:
                tokens = tokenizer(text)
                self.lengths.append(len(tokens))
            for token, count in Counter(tokens).items():
                self.postings.setdefault(token, array('I')).extend((i, count))
        self.avgdl = sum(self.lengths) / max(1, self.N)

    def to_state(self):
        return {'version': 1, 'lengths': base64.b64encode(self.lengths.tobytes()).decode('ascii'),
                'postings': {token: base64.b64encode(values.tobytes()).decode('ascii')
                             for token, values in self.postings.items()}}

    @classmethod
    def from_state(cls, chunks, state, tokenizer=tokenize):
        if state.get('version') != 1:
            raise ValueError('Unsupported private lexical cache')
        value = cls.__new__(cls)
        value.chunks, value.N, value.tokenize = chunks, len(chunks), tokenizer
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
        for token, multiplicity in Counter(self.tokenize(query)).items():
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
