"""Stock card identity, separate from observations quoted inside a card."""
from datetime import date
from functools import lru_cache
import re


HORIZONS = {"short": "短线 短期", "mid": "中线 中期", "long": "长线 长期"}
_CARD = re.compile(
    r"card_([A-Z0-9][A-Z0-9.\-]*)(?:#[A-Z]+)*#(short|mid|long)_(\d{4}-\d{2}-\d{2})\.md", re.I)
_DATE = re.compile(r"(?<!\d)(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?(?!\d)")


@lru_cache(maxsize=8192)
def card_identity(file):
    """Return immutable identity pairs; legacy #MID markers are not tickers."""
    match = _CARD.fullmatch(file)
    if not match:
        return ()
    ticker, horizon, stamp = match.groups()
    try:
        date.fromisoformat(stamp)
    except ValueError:
        return ()
    return (("ticker", ticker.upper()), ("card_date", stamp), ("horizon", horizon.lower()))


def index_terms(chunk):
    if chunk.get("source") != "stock":
        return ""
    fields = dict(card_identity(chunk["file"]))
    if not fields:
        return ""
    terms = " ".join(fields.values()) + " " + HORIZONS[fields["horizon"]]
    # The exporter abbreviates numeric plan fields. Index their meanings only
    # on the passage containing the field, not on every chunk in the card.
    for short, aliases in (("入", "入场 入场区"), ("损", "止损 止损位"), ("站上", "站上线")):
        if re.search(r"(?:三个数[:：]|[/／])\s*" + short + r"\s*[\d.$]", chunk["text"]):
            terms += " " + aliases
    return terms


def query_filter(query, files, explicit=None):
    """Only auto-narrow a single dated/period card, not general case analogies.

Several dates/tickers/periods or temporal range language stay broad. Callers
can supply an explicit identity when natural-language scope is ambiguous.
"""
    if explicit is not None:
        if not isinstance(explicit, dict) or not explicit or set(explicit) - {"ticker", "card_date", "horizon"}:
            raise ValueError("stock_filter needs ticker, card_date and/or horizon")
        fields = dict(explicit)
        if "ticker" in fields:
            if not isinstance(fields["ticker"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.\-]*", fields["ticker"]):
                raise ValueError("Invalid stock ticker")
            fields["ticker"] = fields["ticker"].upper()
        if "card_date" in fields:
            if not isinstance(fields["card_date"], str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", fields["card_date"]):
                raise ValueError("card_date must be YYYY-MM-DD")
            date.fromisoformat(fields["card_date"])
        if "horizon" in fields and fields["horizon"] not in HORIZONS:
            raise ValueError("horizon must be short, mid or long")
        return fields
    known = {dict(card_identity(file)).get("ticker") for file in files} - {None}
    words = {w.upper() for w in re.findall(r"[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*", query)}
    tickers = known & words
    # Also recognize an explicit but absent ticker beside a date/period. Its
    # missing card must produce no hits, rather than another company's card.
    tickers.update(t.upper() for t in re.findall(
        r"(?<![A-Za-z0-9_.\-])([A-Za-z][A-Za-z0-9.\-]{0,11}|\d{4,6}\.[A-Za-z]{1,4})"
        r"(?=\s*(?:的\s*)?(?:\d{4}[-/年]|短线|短期|中线|中期|长线|长期))", query))
    dates = set()
    for y, m, d in _DATE.findall(query):
        try:
            dates.add(date(int(y), int(m), int(d)).isoformat())
        except ValueError:
            return {}
    horizons = {key for key, aliases in HORIZONS.items()
                if re.search(r"\b" + key + r"\b", query, re.I)
                or any(alias in query for alias in aliases.split())}
    if (len(tickers) != 1 or len(dates) > 1 or len(horizons) > 1
            or not (dates or horizons)
            or re.search(r"以来|之前|之后|以前|以后|截至|至今|期间|对比|比较|对照|since|before|after|between"
                         r"|(?:从|自).{0,24}(?:到|至)|\d{4}[-/年]\d{1,2}[-/月]\d{1,2}.{0,3}[到至~～—–]", query, re.I)):
        return {}
    fields = {"ticker": next(iter(tickers))}
    if dates:
        fields["card_date"] = next(iter(dates))
    if horizons:
        fields["horizon"] = next(iter(horizons))
    return fields


def matches(file, fields):
    metadata = dict(card_identity(file))
    return all(metadata.get(key) == value for key, value in fields.items())
