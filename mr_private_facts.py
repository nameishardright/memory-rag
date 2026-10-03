"""Local field extraction and previews linked to complete source evidence."""
import re
from mr_index import tokenize

PAN_PATTERNS = [re.compile(r'(?<!\d)(?=(\d(?:[ \t-]*\d){' + str(n - 1) + r'})(?!\d))')
                for n in (13, 15, 16, 19)]
LABEL_WORDS = re.compile(
    r'(?i)\b(?:american express|amex|hilton|honors|hiltion|aspire|surpass|platinum|plantum|plat|'
    r'business|bus|marriott|marriot|marr|aeroplan|areplan|aero|gold|cobalt|cashback|cash|'
    r'mbna|world elite|we|rogers|rbc|td|bmo|cibc|simplii|tangerine\w*|scotia|scotiabank|sco|sc|'
    r'amazon|triangle|gongshang|icbc|pc|money|acc|acct|account|cheque|cheqe|saving|client|'
    r'visa|mastercard|master card|new|wj|westjet|card number|card)\b|信用卡|卡号|工行|工商银行|借记卡')
GENERIC = {'card', 'card number', 'visa', 'mastercard', 'master card', 'account', '卡号'}
BANK_WORDS = {'money', 'acc', 'acct', 'cheque', 'cheqe', 'saving', 'client', '借记卡'}
CARD_INTENT = re.compile(r'(?i)信用卡|卡号|credit\s*cards?|card\s*numbers?')
ALIASES = {
    '家庭': 'family father mother spouse child', '父亲': 'father', '母亲': 'mother',
    '配偶': 'spouse wife husband', '旅行': 'travel trip itinerary flight',
    '出入境': 'travel entry exit arrival departure', '护照': 'passport',
    '生日': 'date of birth', '地址': 'address', '账号': 'account username login',
    '密码': 'password', '税务': 'tax CRA', '工作': 'employment employer',
}


def luhn(number):
    total = 0
    for i, char in enumerate(reversed(number)):
        n = int(char)
        if i % 2:
            n *= 2
            n -= 9 if n > 9 else 0
        total += n
    return total % 10 == 0


def network(number):
    if len(number) == 15 and number[:2] in {'34', '37'}:
        return 'American Express'
    if len(number) in {13, 16, 19} and number.startswith('4'):
        return 'Visa'
    if len(number) == 16 and (51 <= int(number[:2]) <= 55 or 2221 <= int(number[:4]) <= 2720):
        return 'Mastercard'
    if len(number) == 16 and (number.startswith(('6011', '65')) or 644 <= int(number[:3]) <= 649):
        return 'Discover'
    if number.startswith('62') and len(number) in {16, 19}:
        return 'UnionPay'
    return None


def valid_grouping(raw):
    sizes = tuple(len(group) for group in re.findall(r'\d+', raw))
    return len(sizes) == 1 or sizes in {(4, 4, 4, 4), (4, 6, 5), (4, 4, 4, 3),
                                       (4, 4, 4, 4, 3), (4, 4, 5)}


def label_tokens(text):
    # Only known labels may leave the source context. Adjacent passwords, CVVs,
    # dates, names, and unrelated identifiers are never part of this field.
    return list(dict.fromkeys(m.group().strip().lower() for m in LABEL_WORDS.finditer(text)))


def extract_cards(documents, include_bank_accounts=False):
    records = {}
    for doc in documents:
        if doc.get('scope') not in {None, 'personal'} or doc.get('purpose') in {'navigation', 'evaluation'}:
            continue
        previous = []
        for lineno, line in enumerate(doc['text'].splitlines(), 1):
            matches = []
            for pattern in PAN_PATTERNS:
                for match in pattern.finditer(line):
                    raw = match.group(1)
                    number = re.sub(r'\D', '', raw)
                    brand = network(number)
                    if brand and valid_grouping(raw) and luhn(number):
                        matches.append((match.start(1), match.end(1), number, brand))
            for start, end, number, brand in sorted(set(matches)):
                own = label_tokens(line[:start])
                labels = own
                if not labels or set(labels) <= {'card', 'card number', 'account', '卡号'}:
                    for prior in reversed(previous[-3:]):
                        prior_labels = label_tokens(prior)
                        if prior_labels and not set(prior_labels) <= GENERIC:
                            labels = prior_labels + own
                            break
                if not labels:
                    continue
                record = records.setdefault(number, {'number': number, 'network': brand,
                                            'labels': [], 'sources': [], 'bank_evidence': False})
                label = ' '.join(dict.fromkeys(labels))
                if label and label not in record['labels']:
                    record['labels'].append(label)
                evidence = {'file': doc['file'], 'line': lineno,
                            'document_id': doc['id'], 'text_sha256': doc.get('text_sha256')}
                if evidence not in record['sources']:
                    record['sources'].append(evidence)
                for duplicate in doc.get('duplicate_sources', []):
                    reference = {**evidence, 'file': duplicate['file'],
                                 'text_sha256': duplicate.get('text_sha256')}
                    if reference not in record['sources']:
                        record['sources'].append(reference)
                if set(labels) & BANK_WORDS:
                    record['bank_evidence'] = True
            if line.strip():
                previous.append(line.strip())
    result = []
    for record in records.values():
        if record['bank_evidence'] and not include_bank_accounts:
            continue
        useful = [label for label in record['labels'] if set(label.split()) - GENERIC]
        record['label'] = ' / '.join(useful or record['labels']) or record['network']
        record['classification'] = 'bank_or_account_record' if record['bank_evidence'] else 'card_record'
        record['status'] = 'source_record_only; current card type and validity not verified'
        result.append(record)
    return sorted(result, key=lambda r: (r['label'], r['number']))


def expand_query(query):
    additions = [value for key, value in ALIASES.items() if key in query]
    return query + (' ' + ' '.join(additions) if additions else '')


def compact_cards(records):
    sources = {}
    rows = []
    for record in records:
        refs = []
        for source in record['sources']:
            source_id = sources.setdefault(source['file'], 'S' + str(len(sources) + 1))
            refs.append(source_id + ':' + str(source['line']))
        rows.append({'label': record['label'], 'number': record['number'],
                     'type': record['classification'], 'sources': refs})
    return {'mode': 'local_field_extraction', 'cards': rows,
            'sources': {value: key for key, value in sources.items()},
            'note': 'Source records, not bank-confirmed active cards. Luhn checks syntax only.'}


def compact_search(rag, query, topk=3, scope='personal', preview_chars=220):
    result = rag.search(query, topk=topk, scope=scope, lexical_query=expand_query(query))
    hits = []
    tokens = list(dict.fromkeys(t for t in tokenize(expand_query(query)) if len(t) > 1))
    for hit in result['hits']:
        text = hit['text']
        matches = [text.casefold().find(token.casefold()) for token in tokens]
        starts = [pos for pos in matches if pos >= 0]
        start = max(0, min(starts) - preview_chars // 4) if starts else 0
        evidence = rag.evidence(hit['document_id'], query, expected_md5=hit['md5'])
        hits.append({key: hit[key] for key in ('document_id', 'file', 'scope', 'heading', 'md5',
                     'score', 'purpose', 'evidence_kind', 'duplicate_sources', 'source_sha256',
                     'lexical_score', 'semantic_score', 'retrieval_ranks') if key in hit}
                    | {'excerpt': text[start:start + preview_chars],
                       'excerpt_truncated': start > 0 or len(text) > preview_chars,
                       'evidence': evidence})
    return ({key: result[key] for key in ('ok', 'mode', 'scope', 'route', 'semantic_state', 'answerability')}
            | {'hits': hits, 'note': 'Preview is for navigation; use evidence.context and check review_reasons before answering.'})
