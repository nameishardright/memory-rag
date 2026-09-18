"""Cheap sentence/citation checks, not a general semantic entailment model.

Provenance and deterministic counterexamples can fail a claim. Passing these
checks never certifies an arbitrary paraphrase; the caller must still read each
sentence with its own quotations. No gold labels or model calls are used here.
"""
import re
import unicodedata
from datetime import date
from decimal import Decimal

import mr_stock


def normal(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def sentences(text):
    return [part.strip() for part in re.split(r"(?<=[。！？!?；;])|\n+|(?<=[a-zA-Z])[.]\s+", text) if part.strip()]


_OUTCOME = re.compile(
    r"(?:尚未|未曾|从未|没有|未|没).{0,6}(?:触发|到位|成交|买入|突破|跌破)"
    r"|(?:已经|已|确实).{0,4}(?:触发|成交|买入)"
    r"|(?:两条腿|两腿).{0,6}(?:到位|触发)")
_UNCERTAIN = re.compile(r"不能|无法|不足以|不代表|不等于|不能证明|不确定|缺少|是否|不知|需.{0,4}核实")
_WINDOW = re.compile(r"整个.{0,6}(?:窗口|期间)|全程|全天|一直|从未|期间.{0,8}(?:未|没)")
_DATE = re.compile(r"(?<!\d)(?:(\d{4})[-/年])?(\d{1,2})[-/月](\d{1,2})日?(?!\d)")
_DATE_PAIR = re.compile(r"(?<!\d)(?:(\d{4})[-/年])?(\d{1,2})[-/月](\d{1,2})/(\d{1,2})(?!\d)")
_PRICE_PREFIX = re.compile(r"(?:入|入场|损|止损|站上|价|价格|区间|参考)\s*[$￥¥]?\s*$")
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?![\d.])")
_PRICE = re.compile(r"收盘|收在|开盘|最高价|最低价|成交均价|成交价|报价|现价")
_CONDITIONAL = re.compile(r"如果|假如|假设|若|一旦|\bif\b", re.I)
_PRICE_CONDITION = re.compile(
    r"如果|假如|若|一旦|估算|(?:收盘|报价|现价).{0,8}(?:低于|高于|站上|突破|跌破).{0,24}(?:就|则|退出|止损|减仓|触发)"
    r"|(?:收盘|现价).{0,12}(?:在|低于|高于).{0,15}(?:减仓线|止损线|计划线|区间)")


def _date_matches(text):
    for pattern in (_DATE, _DATE_PAIR):
        for match in pattern.finditer(text):
            year, month, *days = match.groups()
            # Bare price ranges such as 入 10-11 must not become October 11.
            if year is None and _PRICE_PREFIX.search(text[max(0,match.start()-12):match.start()]):
                continue
            values = []
            try:
                for day in days:
                    date(int(year or 2000), int(month), int(day))
                    values.append((int(year) if year else None, int(month), int(day)))
            except ValueError:
                continue
            yield match, values


def dates(text):
    return {value for _,values in _date_matches(text) for value in values}


def _context_dates(context, quote):
    metadata = dict(mr_stock.card_identity(context['file']))
    year = int(metadata['card_date'][:4]) if metadata else None
    return {(y or year,m,d) for y,m,d in dates(quote)}


def _same_date(left, right):
    return left[1:] == right[1:] and (left[0] is None or left[0] == right[0])


def numbers(text):
    chars = list(text)
    for match,_ in _date_matches(text):
        chars[match.start():match.end()] = ' ' * (match.end()-match.start())
    text = ''.join(chars)
    return {Decimal(m.group().replace(",", "")) for m in _NUMBER.finditer(text)}


def _clauses(text):
    return [p for p in re.split(r"[，；;。!?！？\n]|(?<!\d),(?!\d)", text) if p.strip()]


def _factual_fragments(text):
    for sentence in sentences(text):
        for part in re.split(r"但是|但|不过|然而|\bbut\b|\bhowever\b", sentence, flags=re.I):
            if not _CONDITIONAL.search(part):
                yield part


def _outcomes(text):
    facts = set()
    for sentence in _factual_fragments(text):
        for clause in _clauses(sentence):
            for match in _OUTCOME.finditer(clause):
                prefix = re.split(r"但是|但|不过", clause[:match.start()])[-1]
                if _UNCERTAIN.search(prefix):
                    continue
                value = match.group()
                action = next((word for word in ("触发", "到位", "成交", "买入", "突破", "跌破") if word in value), None)
                facts.add((action, not bool(re.search(r"未|没", value))))
    return facts


def _source_envelope(context, quote):
    """Recover full sentences at every occurrence, including multi-sentence quotes."""
    source = normal(context['text'])
    needle = normal(quote)
    start, spans = source.find(needle), []
    while start >= 0:
        spans.append((start,start+len(needle)))
        start = source.find(needle,start+1)
    cursor, enclosing = 0, []
    for sentence in sentences(context['text']):
        part = normal(sentence)
        pos = source.find(part, cursor)
        if pos < 0:
            continue
        cursor = pos + len(part)
        if any(pos < end and cursor > start for start,end in spans):
            enclosing.append(sentence)
    return '\n'.join(enclosing)


def _quoted_outcomes(accepted):
    facts = set()
    for context, quote in accepted:
        # A cropped quote must not turn an if-clause/uncertain statement into
        # an observed outcome. Recheck its containing source sentence as well.
        allowed = _outcomes(quote)
        allowed &= _outcomes(_source_envelope(context, quote))
        facts.update(allowed)
    return facts


def _price_units(text, context=None):
    """Only unambiguous date/metric/value triples are automatically supported.

A date before a comma carries through that sentence. Several dates or metrics
in one unsplit clause need review; set membership cannot establish pairing.
"""
    for fragment in _factual_fragments(text):
        current_dates = set()
        for clause in re.split(r"[、，；;。!?！？\n]|(?<!\d),(?!\d)", fragment):
            explicit = _context_dates(context,clause) if context else dates(clause)
            if explicit:
                current_dates = explicit
            metrics = list(_PRICE.finditer(clause))
            if not metrics or not numbers(clause) or _PRICE_CONDITION.search(clause):
                continue
            labels = {'收盘' if m.group() == '收在' else m.group() for m in metrics}
            value = re.match(r"[\s价为是报收记作：:约≈$￥¥]*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
                             clause[metrics[0].end():])
            if len(current_dates) != 1 or len(labels) != 1 or len(metrics) != 1 or value is None:
                yield None  # ambiguous is not a proven price observation
            else:
                yield (next(iter(current_dates)), next(iter(labels)), Decimal(value[1].replace(',','')))


def _price_date_supported(sentence, accepted):
    proven = set()
    for context, quote in accepted:
        units = set(_price_units(quote,context))
        units &= set(_price_units(_source_envelope(context,quote),context))
        proven.update(unit for unit in units if unit is not None)
    for wanted in _price_units(sentence):
        if wanted is None or not any(_same_date(wanted[0],p[0]) and wanted[1:] == p[1:] for p in proven):
            return False
    return True


def check_answer(answer, contexts, question=""):
    """Return an auditable row for every sentence, using only its own citations."""
    by_id = {c["id"]: c for c in contexts}
    if len(by_id) != len(contexts):
        raise ValueError("Duplicate evidence ids")
    requested_scope = mr_stock.query_filter(question, {c['file'] for c in contexts if c.get('source') == 'stock'})
    rows, issues = [], []
    for ci, claim in enumerate(answer["claims"]):
        accepted, quote_issues = [], []
        for citation in claim["citations"]:
            context = by_id.get(citation["evidence_id"])
            if not context or len(normal(citation["quote"])) < 4 or normal(citation["quote"]) not in normal(context["text"]):
                quote_issues.append(("invalid_quote", "引文不存在、过短或证据编号无效。"))
            else:
                accepted.append((context, citation["quote"]))
        for si, sentence in enumerate(sentences(claim["text"])):
            problems = list(quote_issues)
            quoted = "\n".join(quote for _, quote in accepted)
            stock = any(context.get("source") == "stock" for context, _ in accepted)
            if numbers(sentence) - numbers(quoted):
                problems.append(("number_not_quoted", "本句有数字未出现在本句的有效引文中；其他句的引用不能补证。计算或单位换算需另行核验。"))
            if stock:
                scopes = {mr_stock.card_identity(c["file"]) for c, _ in accepted if c.get("source") == "stock"}
                scopes.discard(())
                if (requested_scope
                        and any(any(dict(scope).get(k) != v for k,v in requested_scope.items()) for scope in scopes)):
                    problems.append(("outside_requested_card", "本句引用不属于问题明确限定的股票卡片；不能用其他日期或周期代替。"))
                if len(scopes) > 1:
                    problems.append(("mixed_card_scopes", "本句混引不同代码/卡片日期/周期；请拆开结论、逐条归属，或明确复核比较关系。"))
                dated = set().union(*(_context_dates(c,q) for c,q in accepted))
                if re.search(r"卡|复核|计划|记录|原判", sentence):
                    dated.update(d for scope in scopes for d in dates(dict(scope)['card_date']))
                if any(not any(_same_date(d,proven) for proven in dated) for d in dates(sentence)):
                    problems.append(("date_not_quoted", "本句新增了其引文及卡片身份未支持的日期；不能从前一日资料推出后一日已确认。"))
                if not _price_date_supported(sentence, accepted):
                    problems.append(("price_date_not_bound", "引文未把本句的价格指标、数字与观测日期直接对应；卡片日期不能替代价格发生日期。"))
                if _outcomes(sentence) - _quoted_outcomes(accepted):
                    problems.append(("trade_outcome_unproven", "价格快照/计划线不能证明整个交易窗口是否触发或成交；需明确的过程或结果记录。"))
                elif (_WINDOW.search(sentence) and _outcomes(sentence)
                      and not any(_WINDOW.search(s) and _outcomes(sentence) <= _outcomes(s)
                                  for _,q in accepted for s in sentences(q))):
                    problems.append(("window_coverage_unproven", "引文未覆盖结论所说的整个交易窗口，不能把当时状态扩展为全程状态。"))
            checks = [{"claim": ci, "sentence": si, "code": code, "reason": reason} for code, reason in problems]
            issues.extend(checks)
            rows.append({"claim": ci, "sentence": si, "text": sentence,
                         "evidence_ids": list(dict.fromkeys(c["id"] for c, _ in accepted)),
                         "checks_pass": not checks, "issues": checks,
                         "semantic_review_required": True})
    return {"checks_pass": not issues, "sentences": rows, "issues": issues,
            "requested_stock_scope": requested_scope,
            "semantic_review_required": bool(rows),
            "scope": "逐句引文出处与已知反例检查；通过不等于语义支持或外部事实正确。"}
