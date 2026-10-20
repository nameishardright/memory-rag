"""Author source-checked paraphrases without consulting retrieval results.

These are agent-authored, source-name-qualified questions over real documents,
not independently collected user questions or final-answer accuracy labels.
All instantiated questions and evidence must be saved only in encrypted files.
"""
from collections import Counter
import hashlib
from pathlib import PurePosixPath
import re
import unicodedata


MONEY = r'(?:[$£€]\s*[\d,]+(?:\.\d{1,2})?|\b(?:CAD|USD|CNY|RMB)\s*[$¥]?\s*[\d,]+(?:\.\d{1,2})?|\d[\d,.]*\s*(?:元|加元|美元))'
DATE = r'(?:\b(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b|\b[A-Za-z]{3,9}\s+\d{1,2},?\s+(?:19|20)\d{2}\b|\b\d{1,2}\s+[A-Za-z]{3,9}\s+(?:19|20)\d{2}\b|(?:19|20)\d{2}年\d{1,2}月\d{1,2}日)'
PERCENT = r'\b\d{1,3}(?:\.\d+)?\s*%'
IDENTIFIER = r'(?<![\w])(?:[A-Z]{1,4}[- ]?)?\d(?:[\d -]{3,18}\d)(?![\w])'

# Cues select verifiable source fields. Query wording is fixed before source
# selection and before either retriever is called.
RULES = [
    ('annual_pay', r'annual\s+(?:base\s+)?(?:salary|compensation)|base\s+(?:annual\s+)?salary|年薪|年度薪酬', MONEY,
     '每年固定能拿多少钱？', 'How much is the fixed yearly pay?'),
    ('monthly_pay', r'monthly\s+(?:salary|pay)|月薪', MONEY,
     '每个月固定发多少钱？', 'How much fixed pay is received each month?'),
    ('hourly_pay', r'hourly\s+(?:rate|pay)|per\s+hour|时薪', MONEY,
     '干一个小时能拿多少钱？', 'How much does an hour of work pay?'),
    ('bonus', r'(?:target|annual|performance)\s+bonus|奖金比例', PERCENT,
     '额外奖励占固定报酬的多大比例？', 'What fraction of fixed pay is the target incentive?'),
    ('job_start', r'(?:employment|commencement|start)\s+date|commenc(?:e|ing|ement)|入职日期|开始工作', DATE,
     '约定从哪一天开始上班？', 'When was work supposed to begin?'),
    ('job_end', r'(?:employment|contract)\s+(?:end|expiration)|termination\s+date|合同到期', DATE,
     '这段工作约定哪一天结束？', 'On what day does this work agreement finish?'),
    ('rent', r'monthly\s+rent|rent\s+(?:amount|payable)|月租|租金', MONEY,
     '每个月住在这里要交多少钱？', 'How much does living here cost each month?'),
    ('deposit', r'(?:security|damage|rental)\s+deposit|押金', MONEY,
     '入住时要先交多少可退的钱？', 'How much refundable money is paid before moving in?'),
    ('lease_start', r'(?:lease|tenancy)\s+(?:start|commenc)|租期开始|起租', DATE,
     '哪一天开始可以住进去？', 'When does the right to live here begin?'),
    ('lease_end', r'(?:lease|tenancy)\s+(?:end|expir)|租期结束|租约到期', DATE,
     '约定可以住到哪一天？', 'Until what day does the housing agreement last?'),
    ('insurance_cost', r'(?:annual|monthly|total)\s+premium|premium\s+(?:amount|due)|保费', MONEY,
     '这份保障要花多少钱？', 'How much does this coverage cost?'),
    ('deductible', r'deductible|自付额|免赔额', MONEY,
     '出事后需要先自己承担多少钱？', 'How much must be paid personally before coverage begins?'),
    ('insurance_end', r'(?:policy|coverage)\s+(?:end|expir)|保险到期', DATE,
     '这份保障有效到哪一天？', 'Until what day is this coverage valid?'),
    ('hotel_checkin', r'check[ -]?in(?:\s+date)?|入住日期', DATE,
     '旅馆是从哪一天开始住？', 'On what day does the hotel stay begin?'),
    ('hotel_checkout', r'check[ -]?out(?:\s+date)?|退房日期', DATE,
     '哪一天要离开旅馆？', 'On what day must the hotel room be vacated?'),
    ('flight_departure', r'(?:departure|flight)\s+date|depart(?:ing)?\s+on|出发日期|起飞日期', DATE,
     '这趟飞机哪一天出发？', 'On what day does this flight leave?'),
    ('trip_cost', r'(?:total|amount)\s+(?:paid|charged|due)|总金额|实付金额', MONEY,
     '这次一共花了多少钱？', 'How much was paid altogether?'),
    ('tax_refund', r'(?:tax\s+)?refund(?:\s+amount)?|退税', MONEY,
     '税务机关要退回来多少钱？', 'How much money is to be returned by the tax authority?'),
    ('tax_due', r'balance\s+owing|tax\s+(?:payable|due)|应缴税额', MONEY,
     '还需要补交多少钱给税务机关？', 'How much more money must be sent to the tax authority?'),
    ('tuition', r'tuition(?:\s+fees?)?|学费', MONEY,
     '上学需要交多少钱？', 'How much must be paid to attend school?'),
    ('invoice_due', r'(?:payment|invoice)\s+due\s+date|付款截止日期', DATE,
     '最晚哪一天要把钱付清？', 'By what day must the bill be paid?'),
    ('purchase_price', r'purchase\s+price|售价|购买价格', MONEY,
     '买下来需要多少钱？', 'How much money is needed to buy it?'),
    ('mortgage_payment', r'(?:monthly|regular)\s+(?:mortgage\s+)?payment|月供', MONEY,
     '每个月要还多少钱？', 'How much must be repaid each month?'),
    ('interest', r'(?:annual\s+)?interest\s+rate|利率', PERCENT,
     '借这笔钱每年按多大比例计息？', 'At what yearly percentage is borrowing charged?'),
    ('passport_expiry', r'(?:date\s+of\s+expiry|expiration\s+date|passport\s+expir)|护照有效期', DATE,
     '这本出国证件能用到哪一天？', 'Until what day can this travel document be used?'),
    ('birth_date', r'date\s+of\s+birth|出生日期', DATE,
     '这里记载的是哪一天出生？', 'What day was this person born according to this record?'),
    ('bank_routing', r'(?:routing|transit|institution)\s+(?:number|no)|银行代码', IDENTIFIER,
     '转钱过来时应该填哪串银行识别数字？', 'Which bank-identifying digits should be used for an incoming transfer?'),
    ('account_number', r'(?:account|member|policy)\s+(?:number|no)|账号|保单号', IDENTIFIER,
     '办理这笔业务时需要提供哪串编号？', 'Which reference digits are needed to handle this service?'),
    ('appointment', r'appointment\s+(?:date|on)|预约日期', DATE,
     '约好了哪一天去办理？', 'On what day was the visit arranged?'),
    ('coverage_limit', r'(?:coverage|benefit)\s+(?:amount|limit)|保额', MONEY,
     '这份保障最多能赔多少钱？', 'What is the maximum amount this coverage can pay?'),
]


def _digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _eligible(doc):
    return (doc.get('scope') == 'personal' and doc.get('method') != 'curated_memory'
            and doc.get('purpose') not in {'navigation', 'evaluation', 'excluded'}
            and len(doc.get('text', '').strip()) >= 80)


def source_candidates(documents, excluded_ids=()):
    """Read original source fields only; never search or consult outcome scores."""
    excluded = set(excluded_ids)
    excluded_hashes = {_digest(d.get('text', '')) for d in documents
                       if d['id'] in excluded}
    names = Counter()
    for doc in documents:
        if doc.get('scope') != 'personal' or doc.get('purpose') in {'navigation', 'evaluation', 'excluded'}:
            continue
        references = {unicodedata.normalize('NFKC', PurePosixPath(source['file'].replace('\\', '/')).stem).casefold()
                      for source in [doc, *doc.get('duplicate_sources', [])]}
        names.update(references)
    rows = []
    for doc in sorted(documents, key=lambda item: item['id']):
        if not _eligible(doc) or doc['id'] in excluded or _digest(doc['text']) in excluded_hashes:
            continue
        if any(alias.get('id') in excluded for alias in doc.get('duplicate_sources', [])):
            continue
        name = PurePosixPath(doc['file'].replace('\\', '/')).stem.strip()
        normalized_name = unicodedata.normalize('NFKC', name).casefold()
        if (not name or len(name) > 140 or names[normalized_name] != 1
                or any(char in name for char in '《》"“”')):
            continue
        for key, cue, value_pattern, zh, en in RULES:
            for match in re.finditer(cue, doc['text'], flags=re.I):
                # Require a literal value immediately following the named field,
                # rather than borrowing an unrelated fact elsewhere in the file.
                tail = doc['text'][match.end():match.end() + 100]
                value = re.search(value_pattern, tail, flags=re.I)
                if value is None or value.start() > 45:
                    continue
                rows.append({'document_id': doc['id'], 'source_name': name,
                             'case_group': 'source-' + _digest(doc['text']),
                             'family': key, 'required_text': [match.group(), value.group()],
                             'source_text_sha256': _digest(doc['text']),
                             'queries': ['档案《' + name + '》里，' + zh,
                                         'In the record named "' + name + '", ' + en]})
                break
    return rows


def author_sets(documents, excluded_ids=(), *, holdout_events=12, development_events=8):
    """Create grouped, frozen sets from a retrieval-independent sampling rule."""
    candidates = source_candidates(documents, excluded_ids)
    # Round-robin field families; hash order is deterministic and unrelated to
    # either retriever. One source body is used for at most one event.
    buckets = {}
    for row in candidates:
        buckets.setdefault(row['family'], []).append(row)
    for rows in buckets.values():
        rows.sort(key=lambda r: _digest(r['case_group'] + r['family']))
    selected, used = [], set()
    while len(selected) < holdout_events + development_events:
        progress = False
        for family in sorted(buckets):
            rows = buckets[family]
            while rows and rows[0]['case_group'] in used:
                rows.pop(0)
            if rows:
                row = rows.pop(0)
                selected.append(row); used.add(row['case_group']); progress = True
                if len(selected) == holdout_events + development_events:
                    break
        if not progress:
            break
    if len(selected) < holdout_events + development_events:
        raise ValueError('Insufficient source-checked events for the requested frozen split')
    selected.sort(key=lambda r: _digest('natural-split-v1:' + r['case_group']))
    groups = {'development': selected[:development_events],
              'holdout': selected[development_events:]}
    sets = {}
    for role, rows in groups.items():
        questions = []
        for event, row in enumerate(rows, 1):
            for language, query in zip(('zh', 'en'), row['queries']):
                if any(query.casefold() in d.get('text', '').casefold() for d in documents):
                    raise ValueError('Natural question must not be copied literally from a source')
                questions.append({'id': role + '-' + str(event) + '-' + language,
                                  'query': query, 'case_group': row['case_group'],
                                  'split': 'holdout' if role == 'holdout' else 'regression',
                                  'scope': 'personal', 'family': row['family'], 'language': language,
                                  'expected_documents': [row['document_id']],
                                  'required_text': row['required_text'],
                                  'source_text_sha256': row['source_text_sha256']})
        # These test missing recorded facts, not forecasts or calibrated refusal.
        absent_date = '2098-12-31'
        if any(absent_date in d.get('text', '') for d in documents):
            raise ValueError('Negative-case date unexpectedly appears in source bodies')
        negatives = ['档案里记载的 2098-12-31 实际到手收入是多少？',
                     'How much pay was actually received on 2098-12-31 according to the records?',
                     '档案里 2098-12-31 那次住旅馆实际付了多少钱？',
                     'How much was actually paid for a hotel stay on 2098-12-31 in the records?']
        # Do not expose these same absent-event questions in development first.
        for number, query in enumerate(negatives if role == 'holdout' else [], 1):
            questions.append({'id': role + '-negative-' + str(number), 'query': query,
                              'case_group': 'missing-recorded-date-' + absent_date,
                              'split': 'holdout' if role == 'holdout' else 'regression',
                              'positive': False, 'scope': 'personal',
                              'negative_reason': 'Specific recorded date absent from all source bodies'})
        sets[role] = {'version': 1,
                      'label_origin': 'agent_authored_source_name_qualified_natural_paraphrases',
                      'authoring_version': 2,
                      'boundary': ('Real original documents; natural rewording with source-name context. '
                                   'Not independently collected user questions or final-answer accuracy. '
                                   'Two languages per event are correlated, not independent samples.'),
                      'event_groups': len(rows), 'questions': questions}
    return sets


def authoring_counts(labels):
    """Only aggregates may leave encrypted benchmark files."""
    positives = [q for q in labels['questions'] if q.get('positive', True)]
    return {'positive_questions': len(positives),
            'negative_questions': len(labels['questions']) - len(positives),
            'event_groups': len({q['case_group'] for q in positives}),
            'families': dict(Counter(q['family'] for q in positives)),
            'languages': dict(Counter(q['language'] for q in positives))}
