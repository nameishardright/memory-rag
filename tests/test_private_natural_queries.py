"""Natural questions without source names: tokens, routing, evidence and policy.

All documents, people and identifiers are fictional. The local model is not
loaded here; the dense leg is replaced by small deterministic stand-ins.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from mr_index import tokenize as shared_tokenize
from mr_private import PrivateRAG, load_private_rag, load_store, make_chunks, save_store
from mr_private_eval import evaluate
from mr_private_facts import compact_search, expand_query
from mr_private_index import PRIVATE_TOKENIZER, PrivateBM25, fold, split_tokens, tokenize
from mr_private_policy import classify, load_policy, policy_signature
from mr_private_semantic import LocalEncoder, PrivateVector, source_label

FIXTURES = Path(__file__).parent / 'fixtures'


def document(identifier, text, scope='personal', file=None, **metadata):
    return {'id': identifier, 'file': file or identifier + '.txt', 'scope': scope, 'text': text,
            'text_sha256': hashlib.sha256(text.encode()).hexdigest(), **metadata}


class PrivateTokenizerTests(unittest.TestCase):
    def test_labels_sentence_ends_and_file_names_match_the_plain_word(self):
        text = 'Annual salary: 100,000. Date of birth: 1997-06-25. Passport No.: EQ1234567. See offer.'
        self.assertIn('salary:', shared_tokenize(text))      # the work/study tokens are unchanged
        tokens = tokenize(text)
        for word in ('salary', 'birth', '1997-06-25', 'no', 'eq1234567', 'offer', '100000'):
            self.assertIn(word, tokens)
        self.assertFalse(any(token.endswith((':', '.')) for token in tokens))
        names = tokenize('04_finance\\tax\\Rowan Example T4 2024.txt')
        for word in ('finance', 'tax', 't4', '2024', '2024.txt'):
            self.assertIn(word, names)

    def test_accents_and_full_width_forms_fold_without_touching_cjk(self):
        self.assertEqual(fold('Montréal, Québec — Hélène'), 'Montreal, Quebec — Helene')
        self.assertEqual(fold('Straße Æther été'), 'Strasse AEther ete')
        self.assertEqual(fold('苏州市が'), '苏州市が')
        tokens = tokenize('编号：（2021）虚构证字第3391号，ＣＡＤ ８４，５００。')
        for word in ('2021', '3391', 'cad', '84500', '编号'):
            self.assertIn(word, tokens)

    def test_compound_remains_one_word_and_parts_do_not_lengthen_the_record(self):
        words, alternates = split_tokens('Licence number: SYN-DL-88231-40117. Due 2025-04-21 at 09:15.')
        self.assertEqual(words, ['licence', 'number', 'syn-dl-88231-40117', 'due', '2025-04-21', 'at', '09:15'])
        for part in ('88231', '40117', '2025', '04', '21', '09', '15'):
            self.assertIn(part, alternates)
        identifiers = document('ids', 'Vehicle registration expires 2032-08-29. Plate: TEST-5836. Reference CAR-2958.')
        prose = document('prose', 'Vehicle registration requirements and renewal schedules are general guidance for owners.')
        chunks = make_chunks([identifiers, prose])
        index = PrivateBM25(chunks)
        self.assertEqual(list(index.lengths), [len(split_tokens(c['file'] + ' ' + c['heading'] + ' ' + c['text'])[0])
                                               for c in chunks])
        # The exact identifier outranks a record that only shares one of its parts.
        docs = [document('exact', 'Policy SYN-INS-4471-0092 renews in March.'),
                document('partial', 'Locker 4471 is on the second floor. Locker 4471 is next to the lift.')]
        chunks = make_chunks(docs)
        index = PrivateBM25(chunks)
        self.assertEqual(chunks[index.search('SYN-INS-4471-0092')[0][0]]['document_id'], 'exact')
        self.assertEqual({chunks[i]['document_id'] for i, _ in index.search('4471')}, {'exact', 'partial'})

    def test_plain_words_find_form_fields_years_in_file_names_and_partial_identifiers(self):
        docs = [document('t4-2023', 'Box 14 - Employment income: 118,950.00', file='tax\\Example T4 2023.txt'),
                document('t4-2024', 'Box 14 - Employment income: 131,204.18', file='tax\\Example T4 2024.txt'),
                document('licence', 'Licence number: SYN-DL-88231-40117.\nExpiry date: 2029-08-14.'),
                document('transcript', 'Cumulative GPA: 3.62.\nDegree conferred: 2016-06-14.')]
        rag = PrivateRAG({'documents': docs, 'chunks': make_chunks(docs)})
        try:
            first = lambda query: rag.search(query)['hits'][0]['document_id']
            self.assertEqual(first('T4 2024 income'), 't4-2024')
            self.assertEqual(first('T4 2023 income'), 't4-2023')
            self.assertEqual(first('88231-40117'), 'licence')
            self.assertEqual(first('GPA'), 'transcript')
            self.assertEqual(first('131204.18'), 't4-2024')
        finally:
            rag.close()

    def test_lexical_cache_from_another_tokenizer_is_rebuilt(self):
        docs = [document('pay', 'Base salary: CAD 128,500.00 per year.')]
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory) / 'corpus.enc'
            save_store(store, 'synthetic-password', {'version': 1, 'documents': docs, 'chunks': make_chunks(docs),
                                                     'updated_at': '2026-01-01', 'coverage': {}})
            (store.parent / 'retrieval-settings.json').write_text(json.dumps(
                {'version': 1, 'mode': 'bm25', 'encrypted_index_cache': True}), encoding='utf-8')
            rag = load_private_rag(store, 'synthetic-password')
            self.assertTrue(rag.search('salary')['hits'])
            rag.close()
            cache = store.with_name('corpus.lexical.personal.enc')
            saved = load_store(cache, 'synthetic-password')
            self.assertEqual(saved['tokenizer'], PRIVATE_TOKENIZER)
            # A cache written by the previous tokenizer would have no `salary` posting.
            save_store(cache, 'synthetic-password', {**saved, 'tokenizer': 'jieba',
                                                     'lexical': {**saved['lexical'], 'postings': {}}})
            rag = load_private_rag(store, 'synthetic-password')
            try:
                self.assertTrue(rag.search('salary')['hits'])
            finally:
                rag.close()
            self.assertEqual(load_store(cache, 'synthetic-password')['tokenizer'], PRIVATE_TOKENIZER)


class PrivateGlossaryTests(unittest.TestCase):
    def test_bridge_works_in_both_directions_without_generic_words(self):
        expanded = expand_query('我的护照有效期到哪一天？')
        for word in ('passport', 'expiry', 'expires'):
            self.assertIn(word, expanded.split())
        self.assertNotIn('valid', expanded.split())
        self.assertNotIn('until', expanded.split())
        self.assertEqual(expand_query('生日').split()[1:], ['birth', 'birthday'])
        reverse = expand_query('What is the date of birth on my household register notarization?')
        for word in ('生日', '户口', '公证'):
            self.assertIn(word, reverse.split())
        # Ordinary English words do not attach an unrelated Chinese term.
        self.assertEqual(expand_query('I will pay the bill at home'), 'I will pay the bill at home')
        self.assertEqual(expand_query('SYN-X92003'), 'SYN-X92003')

    def test_chinese_question_reaches_an_english_record_by_keywords_alone(self):
        docs = [document('lease', 'Monthly rent: CAD 2,450.00.\nTenancy begins: 2025-02-01.'),
                document('policy', 'Deductible: CAD 2,000.\nRenewal date: 2026-03-15.'),
                document('trip', 'Departure: 2025-08-09 at 06:50.')]
        rag = PrivateRAG({'documents': docs, 'chunks': make_chunks(docs)})
        try:
            self.assertEqual(compact_search(rag, '房屋保险的免赔额是多少？')['hits'][0]['document_id'], 'policy')
            self.assertEqual(compact_search(rag, '月租是多少？')['hits'][0]['document_id'], 'lease')
        finally:
            rag.close()


class PrivateAutomaticScopeTests(unittest.TestCase):
    def docs(self):
        return [document('offer', '# Equity\nStock options: 2,400 options vesting over four years.'),
                document('resume', '# Experience\nLed the Harbor Scheduler project, cutting latency by 37%.'),
                document('readme', '# Harbor Scheduler\nStorage layer: SQLite. A scheduler project.', 'technical'),
                document('ticker', '# SYNQ options\nImplied volatility: 44%.', 'stock')]

    def test_personal_records_are_consulted_when_a_keyword_routes_elsewhere(self):
        docs = self.docs()
        rag = PrivateRAG({'documents': docs, 'chunks': make_chunks(docs)})
        try:
            options = rag.search('入职时公司给了我多少期权？', scope='auto', lexical_query=expand_query('入职时公司给了我多少期权？'))
            self.assertEqual(options['scope'], 'stock')
            self.assertEqual(options['route']['also_searched'], ['personal'])
            self.assertIn('offer', [hit['document_id'] for hit in options['hits']])
            self.assertEqual(options['modes'], {'stock': 'bm25', 'personal': 'bm25'})
            project = rag.search('scheduler 项目 latency', scope='auto')
            self.assertEqual([hit['document_id'] for hit in project['hits']], ['readme', 'resume'])
            # An explicit choice is never widened.
            self.assertEqual([hit['document_id'] for hit in rag.search('scheduler 项目 latency', scope='technical')['hits']], ['readme'])
            self.assertEqual([hit['document_id'] for hit in rag.search('options', scope='personal')['hits']], ['offer'])
            self.assertNotIn('also_searched', rag.search('my passport', scope='auto')['route'])
        finally:
            rag.close()

    def test_named_source_under_automatic_scope_is_found_in_personal_records(self):
        docs = self.docs()
        rag = PrivateRAG({'documents': docs, 'chunks': make_chunks(docs)})
        try:
            found = rag.search('档案《offer》里，有多少期权？', scope='auto')
            self.assertEqual([hit['document_id'] for hit in found['hits']], ['offer'])
            self.assertEqual(found['route']['source_reference'], 'unique')
            missing = rag.search('档案《not-present》里，有多少期权？', scope='auto')
            self.assertEqual(missing['hits'], [])
            self.assertEqual(missing['answerability'], 'no_evidence')
            routed = rag.search('档案《ticker》里，期权隐含波动率是多少？', scope='auto')
            self.assertEqual([hit['document_id'] for hit in routed['hits']], ['ticker'])
        finally:
            rag.close()

    def test_large_partition_is_not_opened_by_a_routing_keyword(self):
        personal = [document('offer', 'Stock options: 2,400 options vesting over four years.')]
        stock = [document('stock:note', 'SYNQ stock options implied volatility 44%.', 'stock')]
        loads = []
        def loader(scope):
            loads.append(scope)
            return {'documents': stock, 'chunks': make_chunks(stock)}
        payload = {'documents': personal, 'chunks': make_chunks(personal),
                   'shards': {'stock': {'file': 'corpus.x.stock.enc', 'documents': 1, 'chunks': 1038769}}}
        rag = PrivateRAG(payload, loader)
        try:
            result = rag.search('my stock options', scope='auto')
            self.assertEqual(loads, [])
            self.assertEqual(result['scope'], 'personal')
            self.assertEqual(result['route']['deferred_scope'], 'stock')
            self.assertEqual([hit['document_id'] for hit in result['hits']], ['offer'])
            explicit = rag.search('implied volatility', scope='stock')
            self.assertEqual(loads, ['stock'])
            self.assertEqual([hit['document_id'] for hit in explicit['hits']], ['stock:note'])
            # Once the user has opened it, automatic selection may use it.
            merged = rag.search('my stock options', scope='auto')
            self.assertEqual(merged['scope'], 'stock')
            self.assertEqual({hit['document_id'] for hit in merged['hits']}, {'stock:note', 'offer'})
        finally:
            rag.close()
        small = PrivateRAG({**payload, 'shards': {'stock': {**payload['shards']['stock'], 'chunks': 5000}}}, loader)
        try:
            self.assertEqual(small.search('my stock options', scope='auto')['scope'], 'stock')
            self.assertEqual(loads, ['stock', 'stock'])
        finally:
            small.close()

    def test_equal_fused_rank_is_settled_by_the_question_embedding(self):
        docs = [document('survey', '# Salary survey\nTypical salary and pay at Cedar Labs.'),
                document('confirmation', '# Employment confirmation\nEmployer: Cedar Labs. Annual remuneration: CAD 84500.')]
        chunks = make_chunks(docs)
        rag = PrivateRAG({'documents': docs, 'chunks': chunks}, mode='hybrid')
        position = {chunk['document_id']: i for i, chunk in enumerate(chunks)}
        class Vector:
            ready = True
            def search(self, *_args, **_kwargs):
                return [(position['confirmation'], .79), (position['survey'], .78)]
            def passages(self, *_args, **_kwargs):
                return []
        try:
            with patch.object(rag, '_vector', return_value=Vector()):
                result = rag.search('salary pay employer remuneration')
            self.assertEqual(result['mode'], 'hybrid')
            self.assertEqual([hit['retrieval_ranks'] for hit in result['hits']], [[1, 0], [0, 1]])
            self.assertEqual(result['hits'][0]['document_id'], 'confirmation')
        finally:
            rag.close()


class PrivateEvidenceSelectionTests(unittest.TestCase):
    def long_document(self):
        filler = 'Ordinary clause about notices, headings and governing wording. ' * 22
        sections = ['# Section ' + str(n) + '\n' + filler for n in range(1, 13)]
        sections.insert(8, '# Handover\nThe keys are handed over on 2025-06-27.')
        sections.insert(9, '# Consideration\nThe agreed amount is CAD 612,000.')
        return document('agreement', '\n\n'.join(sections))

    def test_dense_passages_recover_a_fact_that_shares_no_word_with_the_question(self):
        doc = self.long_document()
        chunks = make_chunks([doc])
        self.assertGreater(len(doc['text']), 12000)
        target = next(i for i, chunk in enumerate(chunks) if '2025-06-27' in chunk['text'])
        keyword_only = PrivateRAG({'documents': [doc], 'chunks': chunks})
        try:
            before = keyword_only.evidence('agreement', '公寓哪天交房？')
            self.assertFalse(before['context_complete'])
            self.assertNotIn('2025-06-27', before['context'])
        finally:
            keyword_only.close()
        rag = PrivateRAG({'documents': [doc], 'chunks': chunks}, mode='hybrid')
        class Vector:
            ready = True
            chunks = None
            def passages(self, query, document_id, topk):
                assert document_id == 'agreement' and query == '公寓哪天交房？'
                return [(target, .91)]
        vector = Vector(); vector.chunks = chunks
        try:
            with patch.object(rag, '_vector', return_value=vector):
                result = rag.evidence('agreement', '公寓哪天交房？')
            self.assertIn('2025-06-27', result['context'])
            # The neighbouring passage comes along, and passages read in document order.
            self.assertIn('CAD 612,000', result['context'])
            identifiers = [passage['id'] for passage in result['passages']]
            self.assertEqual(identifiers, sorted(identifiers, key=lambda value: int(value.split(':')[-1])))
            self.assertLessEqual(len(result['context']), 12000)
        finally:
            rag.close()

    def test_selection_keeps_the_matched_passage_and_respects_the_budget(self):
        doc = self.long_document()
        chunks = make_chunks([doc])
        rag = PrivateRAG({'documents': [doc], 'chunks': chunks})
        try:
            last = chunks[-1]
            result = rag.evidence('agreement', 'agreed amount', expected_md5=last['md5'], max_chars=4000, max_passages=2)
            self.assertIn(last['md5'], [passage['md5'] for passage in result['passages']])
            self.assertIn('CAD 612,000', result['context'])
            self.assertLessEqual(len(result['context']), 4000)
            self.assertLessEqual(len(result['passages']), 6)
            tight = rag.evidence('agreement', 'agreed amount', max_chars=200)
            self.assertIn('CAD 612,000', tight['context'])
            self.assertLessEqual(len(tight['context']), 200)
            # Nothing is returned rather than a passage cut to fit.
            self.assertEqual(rag.evidence('agreement', 'ordinary clause', max_chars=200)['passages'], [])
        finally:
            rag.close()

    def test_vector_ranks_passages_of_one_document_and_embeds_the_question_once(self):
        chunks = [{'document_id': 'one', 'id': 'one:1'}, {'document_id': 'one', 'id': 'one:2'},
                  {'document_id': 'two', 'id': 'two:1'}]
        class Encoder:
            calls = 0
            def encode(self, texts, query=False):
                Encoder.calls += 1
                return np.array([[1.0, 0.0]], dtype=np.float32)
        vector = PrivateVector(chunks, Encoder())
        vector.emb = np.array([[0.2, 0.9], [0.9, 0.1], [0.6, 0.0], [1.0, 0.0]], dtype=np.float32)
        vector.rows = [{'chunk': 0}, {'chunk': 1}, {'chunk': 1}, {'chunk': 2}]; vector.ready = True
        try:
            self.assertEqual([i for i, _ in vector.passages('fictional question', 'one')], [1, 0])
            self.assertAlmostEqual(vector.passages('fictional question', 'one')[0][1], 0.9, places=6)
            self.assertEqual(vector.passages('fictional question', 'missing'), [])
            self.assertEqual(vector.search('fictional question')[0][0], 2)
            self.assertEqual(Encoder.calls, 1)
            vector.search('another question')
            self.assertEqual(Encoder.calls, 2)
        finally:
            vector.close()
        self.assertIsNone(vector._last_query)
        cancelled = threading.Event(); cancelled.set()
        with self.assertRaises(InterruptedError):
            PrivateVector(chunks, Encoder(), cancelled=cancelled).passages('fictional question', 'one')

    def test_embedded_passage_starts_with_folder_and_file_name(self):
        self.assertEqual(source_label('04_finance\\tax\\Rowan Example T4 2024.txt'), 'tax / Rowan Example T4 2024')
        self.assertEqual(source_label('fictional-offer.txt'), 'fictional-offer')
        self.assertEqual(source_label('archive.zip!/inner/notes.v2.md'), 'inner / notes.v2')
        self.assertEqual(source_label(''), '')
        class Tokenizer:
            def __call__(self, text, **kwargs):
                offsets, start = [], 0
                for word in text.split(' '):
                    offsets.append((start, start + len(word))); start += len(word) + 1
                return {'input_ids': list(range(len(offsets))), 'offset_mapping': offsets}
        encoder = LocalEncoder.__new__(LocalEncoder)
        encoder._lock, encoder._tokenizer, encoder.signature = threading.Lock(), Tokenizer(), 'synthetic'
        chunk = {'file': 'tax\\Example T4 2024.txt', 'heading': 'Statement', 'text': 'Box 14 income', 'md5': 'm'}
        window = encoder.windows(chunk)[0]
        self.assertEqual(window['text'], 'passage: tax / Example T4 2024\nStatement\nBox 14 income')
        # The same body under another file name is a different embedded passage.
        self.assertNotEqual(window['key'], encoder.windows({**chunk, 'file': 'tax\\Example T4 2023.txt'})[0]['key'])
        self.assertEqual(encoder.windows({'heading': 'Statement', 'text': 'Box 14 income', 'md5': 'm'})[0]['text'],
                         'passage: Statement\nBox 14 income')


class PrivatePolicyComponentTests(unittest.TestCase):
    def test_component_rules_match_any_depth_and_respect_their_parent(self):
        policy = {'version': 1, 'rules': [
            {'component': 'originals', 'under': '股票分析', 'scope': 'exclude', 'reason': 'raw_public_filing'},
            {'component': '*-sources', 'under': '股票分析', 'scope': 'exclude', 'reason': 'raw_public_filing'},
            {'component': 'drafts', 'scope': 'history', 'purpose': 'historical'}]}
        raw = classify('导入/user/AI备份/股票分析/ABC-20260921/originals/form4/x.xml', policy=policy)
        self.assertEqual(raw, {'scope': 'exclude', 'purpose': 'excluded', 'reason': 'raw_public_filing'})
        self.assertEqual(classify('导入/u/股票分析/ABC/route3-sources/a.pdf', policy=policy)['scope'], 'exclude')
        self.assertEqual(classify('导入/u/股票分析/ABC/analysis.md', policy=policy)['scope'], 'stock')
        # A file merely named like the directory, or the same directory elsewhere, is untouched.
        self.assertEqual(classify('导入/u/股票分析/ABC/originals.md', policy=policy)['scope'], 'stock')
        self.assertEqual(classify('03_work/originals/contract.pdf', policy=policy)['scope'], 'personal')
        self.assertEqual(classify('03_work/Drafts/letter.docx', policy=policy),
                         {'scope': 'history', 'purpose': 'historical', 'reason': 'directory_rule'})

    def test_shipped_policy_drops_raw_filings_and_keeps_written_research(self):
        empty = {'version': 1, 'rules': []}
        path = '导入/user/AI备份/股票分析/ABC/originals/10-k.htm'
        self.assertEqual(classify(path, policy=empty)['scope'], 'stock')
        shipped = load_policy(Path(__file__).resolve().parents[1] / 'private-policy.json')
        self.assertEqual(classify(path, policy=shipped)['reason'], 'raw_public_filing')
        self.assertEqual(classify('导入/user/AI备份/股票分析/ABC/sources/main/companyfacts.json', policy=shipped)['scope'], 'exclude')
        self.assertEqual(classify('导入/user/AI备份/claude-skills/stock-analysis/scripts/px_cache/ABC.json', policy=shipped)['scope'], 'exclude')
        self.assertEqual(classify('导入/user/AI备份/股票分析/ABC/analysis.md', policy=shipped)['scope'], 'stock')
        self.assertEqual(classify('导入/user/AI备份/股票分析/cases/md/case-001.md', policy=shipped)['scope'], 'stock')
        self.assertEqual(classify('06_代码项目/old/README.md', policy=shipped)['scope'], 'technical')
        self.assertEqual(classify('90_历史归档/tax.pdf', policy=shipped)['scope'], 'history')
        self.assertNotEqual(policy_signature(shipped), policy_signature(empty))

    def test_invalid_rules_and_development_fixtures_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.json'
            for rule in ({'scope': 'exclude'}, {'prefix': 'a', 'component': 'b', 'scope': 'exclude'},
                         {'component': 'a', 'scope': 'unknown'}):
                path.write_text(json.dumps({'version': 1, 'rules': [rule]}), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_policy(path)
        for name in ('private_natural_dev.json', 'private_natural_holdout.json'):
            self.assertEqual(classify('tests/fixtures/' + name)['purpose'], 'evaluation')


class PrivateNaturalFixtureTests(unittest.TestCase):
    def test_development_questions_without_source_names_reach_their_records_by_keywords(self):
        fixture = json.loads((FIXTURES / 'private_natural_dev.json').read_text(encoding='utf-8'))
        self.assertTrue(fixture['synthetic'])
        self.assertFalse(any('《' in q['query'] or 'named "' in q['query'] for q in fixture['questions']))
        documents = fixture['documents']
        rag = PrivateRAG({'documents': documents, 'chunks': make_chunks(documents)})
        try:
            summary = evaluate(rag, fixture, repetitions=0)['summary']
        finally:
            rag.close()
        self.assertEqual(summary['positive_questions'], 24)
        self.assertEqual(summary['file_hits_at_k'], 24)
        self.assertEqual(summary['evidence_complete_at_k'], 24)
        self.assertEqual(summary['modes'], ['bm25'])


if __name__ == '__main__':
    unittest.main()
