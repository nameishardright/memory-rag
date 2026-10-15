"""Local field extraction and previews linked to complete source evidence."""
import re
from mr_private_index import tokenize

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
# Keyword bridge between Chinese questions and English records, and back. It only
# widens the BM25 query; the local vectors remain the general multilingual path.
# Each row: Chinese ways of saying it, then the English terms records use.
GLOSSARY = (
    # identity and immigration
    (('护照',), ('passport',)),
    (('签证',), ('visa',)),
    (('身份证',), ('identity card', 'ID card')),
    (('驾照', '驾驶证'), ('driver licence', 'driver license', 'driving licence')),
    (('出生地',), ('place of birth', 'birthplace', 'born')),
    (('生日', '出生日期'), ('date of birth', 'birthday')),
    (('出生证明', '出生公证'), ('birth certificate',)),
    (('户口',), ('household register', 'hukou')),
    (('公证',), ('notarization', 'notarized', 'notarial', 'notary')),
    (('结婚',), ('marriage', 'married')),
    (('永久居民', '枫叶卡'), ('permanent resident', 'PR card')),
    (('移民',), ('immigration', 'immigrant')),
    (('登陆', '入境'), ('landing', 'landed', 'port of entry')),
    (('工签', '工作许可'), ('work permit',)),
    (('学签', '学习许可'), ('study permit',)),
    (('社保号', '社会保险号'), ('social insurance number', 'SIN')),
    (('健康卡', '医保卡'), ('health card',)),
    (('到期', '过期', '有效期'), ('expiry', 'expires', 'expire', 'expiration', 'valid until')),
    (('签发',), ('issued', 'date of issue')),
    # family
    (('家庭',), ('family', 'father', 'mother', 'spouse', 'child')),
    (('父亲', '爸爸'), ('father',)),
    (('母亲', '妈妈'), ('mother',)),
    (('配偶', '妻子', '老婆', '丈夫', '老公', '爱人'), ('spouse', 'wife', 'husband')),
    (('女儿',), ('daughter', 'child')),
    (('儿子',), ('son', 'child')),
    (('孩子', '小孩'), ('child', 'children')),
    (('托儿所', '幼儿园', '日托'), ('daycare', 'childcare', 'kindergarten', 'preschool')),
    (('成绩',), ('grades', 'report card', 'transcript', 'GPA')),
    (('疫苗',), ('vaccine', 'vaccination', 'immunization')),
    # work and education
    (('工作',), ('employment', 'employer')),
    (('雇主',), ('employer',)),
    (('工资', '薪水', '年薪', '薪资', '薪酬'), ('salary', 'pay', 'compensation', 'remuneration')),
    (('收入',), ('income', 'earnings')),
    (('奖金',), ('bonus',)),
    (('期权', '股权', '股票'), ('stock options', 'equity', 'shares', 'share units')),
    (('年假', '假期', '带薪假', '休假'), ('vacation', 'paid time off', 'leave')),
    (('合同', '协议'), ('contract', 'agreement')),
    (('录用', '聘书'), ('offer of employment', 'offer letter')),
    (('简历',), ('resume', 'CV')),
    (('推荐信',), ('reference letter',)),
    (('绩效',), ('performance review', 'rating')),
    (('工号', '员工号'), ('employee number',)),
    (('福利',), ('benefits',)),
    (('学位', '学历', '毕业'), ('degree', 'diploma', 'graduation', 'bachelor', 'master')),
    (('大学',), ('university', 'college')),
    (('成绩单',), ('transcript',)),
    (('学费',), ('tuition',)),
    # money, tax and insurance
    (('税务', '报税', '税单'), ('tax', 'CRA')),
    (('退税',), ('refund',)),
    (('账单',), ('bill', 'statement')),
    (('发票', '收据'), ('invoice', 'receipt')),
    (('信用卡',), ('credit card',)),
    (('还款',), ('payment due', 'minimum payment')),
    (('银行',), ('bank',)),
    (('账号', '账户'), ('account', 'username', 'login')),
    (('密码',), ('password',)),
    (('余额',), ('balance',)),
    (('房贷', '贷款', '按揭'), ('mortgage', 'loan')),
    (('利率',), ('interest rate',)),
    (('保险', '保单'), ('insurance', 'policy')),
    (('免赔额',), ('deductible',)),
    (('责任险',), ('liability',)),
    (('保费',), ('premium',)),
    (('续保', '续期', '续签'), ('renewal', 'renew')),
    (('捐款', '捐赠'), ('donation',)),
    # housing and bills
    (('公寓',), ('condo', 'condominium', 'apartment')),
    (('房子', '房屋', '房产'), ('property', 'house', 'home')),
    (('成交价', '房价', '买房', '购房'), ('purchase price',)),
    (('交割', '交房', '过户'), ('closing', 'completion', 'possession')),
    (('定金', '订金', '押金'), ('deposit',)),
    (('租金', '月租', '房租'), ('rent',)),
    (('租约', '租赁'), ('lease', 'tenancy')),
    (('房东',), ('landlord',)),
    (('租客', '房客'), ('tenant',)),
    (('房产税', '地税'), ('property tax',)),
    (('物业费', '管理费'), ('common expenses', 'maintenance fee')),
    (('电费',), ('electricity', 'hydro')),
    (('燃气', '煤气'), ('gas',)),
    (('宽带', '网费'), ('internet',)),
    (('话费', '手机'), ('mobile', 'phone')),
    # travel and health
    (('旅行', '旅游'), ('travel', 'trip', 'itinerary', 'flight')),
    (('出入境',), ('travel', 'entry', 'exit', 'arrival', 'departure')),
    (('机票', '航班'), ('flight', 'itinerary', 'ticket')),
    (('起飞', '出发'), ('departure', 'departs')),
    (('到达', '抵达'), ('arrival',)),
    (('酒店',), ('hotel', 'booking', 'reservation')),
    (('租车',), ('car rental',)),
    (('体检',), ('medical exam', 'checkup')),
    (('化验',), ('laboratory', 'lab result')),
    (('牙科', '牙医'), ('dental', 'dentist')),
    (('验光', '眼科'), ('eye exam', 'optometry')),
    (('处方',), ('prescription',)),
    # other
    (('地址',), ('address',)),
    (('遗嘱',), ('will', 'executor')),
    (('执行人',), ('executor',)),
    (('车辆', '汽车'), ('vehicle', 'car')),
    (('保修',), ('warranty',)),
    (('会员',), ('membership',)),
)
# Ordinary English words that would attach an unrelated Chinese term to a question.
AMBIGUOUS_ENGLISH = {'will', 'pay', 'home', 'leave', 'bill', 'master', 'gas', 'car', 'son', 'house',
                     'property', 'policy', 'rating', 'completion', 'possession', 'closing', 'entry',
                     'exit', 'trip', 'phone', 'mobile', 'child', 'children', 'account', 'statement',
                     'balance', 'income', 'checkup', 'born', 'issued', 'rent', 'college', 'login'}
# Words that occur in almost every form; they identify a phrase in a question
# but would only add noise as separate search terms.
_GENERIC_ENGLISH = {'of', 'on', 'to', 'the', 'a', 'until', 'valid', 'date', 'number', 'card',
                    'time', 'off', 'paid', 'place', 'port', 'units'}
_CHINESE_TRIGGERS = [
    (chinese, ' '.join(dict.fromkeys(word for term in english for word in term.split()
                                     if word.lower() not in _GENERIC_ENGLISH)))
    for chinese, english in GLOSSARY]
_ENGLISH_TRIGGERS = [
    (re.compile(r'(?i)(?<![A-Za-z])(?:' + '|'.join(re.escape(term) for term in english
                                                   if term.lower() not in AMBIGUOUS_ENGLISH) + r')(?![A-Za-z])'),
     ' '.join(chinese))
    for chinese, english in GLOSSARY if any(term.lower() not in AMBIGUOUS_ENGLISH for term in english)]


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
    additions = [english for chinese, english in _CHINESE_TRIGGERS if any(term in query for term in chinese)]
    additions.extend(chinese for pattern, chinese in _ENGLISH_TRIGGERS if pattern.search(query))
    additions = list(dict.fromkeys(additions))
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
