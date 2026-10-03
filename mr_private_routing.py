"""Bounded source selection. Explicit user choices always take precedence."""
import re

STOCK = re.compile(r'(?i)股票|股价|财报|市值|估值|期权|持仓|\bstocks?\b|\bshare price\b|\bearnings\b|\bticker\b')
TECHNICAL = re.compile(r'(?i)代码|源码|仓库|项目|编程|架构|检索|算法|报错|\b(?:github|rag|api|python|java|sqlite|code|repository|architecture|debug|system design)\b')
HISTORY = re.compile(r'(?i)历史备份|旧备份|备份档案|\b(?:historical backup|old backup|archived backup)\b')
PERSONAL_PAY = re.compile(r'(?i)工资|年薪|薪水|\b(?:my salary|my income|my earnings|do i earn|am i paid)\b')


def route_query(query):
    if PERSONAL_PAY.search(query):
        return {'scope': 'personal', 'reason': 'personal_pay_terms', 'automatic': True}
    if STOCK.search(query):
        return {'scope': 'stock', 'reason': 'stock_terms', 'automatic': True}
    if TECHNICAL.search(query):
        return {'scope': 'technical', 'reason': 'project_terms', 'automatic': True}
    if HISTORY.search(query):
        return {'scope': 'history', 'reason': 'backup_terms', 'automatic': True}
    return {'scope': 'personal', 'reason': 'personal_default', 'automatic': True}
