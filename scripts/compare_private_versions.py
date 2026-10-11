"""Pair a baseline checkout with this checkout on fictional fixtures.

Both versions answer the same frozen questions in isolated processes, in BM25
and in hybrid with the local E5 model. Fixtures must be marked synthetic; this
tool never unlocks a private corpus. A fixture whose questions are a holdout is
reserved before the first search and cannot be rerun as fresh.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
CODE_FILES = ('mr_private.py', 'mr_private_index.py', 'mr_private_facts.py', 'mr_private_semantic.py',
              'mr_private_routing.py', 'mr_private_policy.py', 'mr_private_eval.py', 'mr_index.py', 'mr_corpus.py')


def code_fingerprint(root):
    return {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
            for name in CODE_FILES if (Path(root) / name).is_file()}


def padding_documents(count, forbidden):
    """Deterministic fictional filler for the personal scope.

    A real archive holds far more files than either retrieval leg returns, and
    fusion behaves differently there than in a fixture of a few dozen records.
    The filler repeats everyday record types over many months and years and
    never contains a labelled fact.
    """
    import random
    rng = random.Random(20261011)
    money = lambda low, high: 'CAD {:,.2f}'.format(rng.uniform(low, high))
    day = lambda: '%04d-%02d-%02d' % (rng.randint(2012, 2023), rng.randint(1, 12), rng.randint(1, 28))
    month = lambda: '%04d-%02d' % (rng.randint(2012, 2023), rng.randint(1, 12))
    reference = lambda prefix: 'PAD-%s-%06d' % (prefix, rng.randint(0, 999999))
    cities = ['Winnipeg', 'Regina', 'Victoria', 'Saskatoon', 'Fredericton', 'Kelowna', 'Sudbury', 'Moncton']
    stores = ['Fictional Grocer', 'Imaginary Hardware', 'Pretend Pharmacy', 'Sample Bookshop', 'Mock Outfitters']
    products = ['kettle', 'vacuum', 'router', 'thermostat', 'blender', 'printer', 'humidifier', 'toaster']
    topics = ['roadmap', 'architecture review', 'API migration', 'hiring plan', 'incident follow-up', 'budget']

    def pay():
        date, gross = day(), rng.uniform(2200, 4800)
        return ('03_work\\pay\\Pay statement ' + date + '.txt',
                '# Pay statement\nEmployer: Quillon Fictional Works.\nPay period ending: ' + date +
                '.\nGross pay: CAD {:,.2f}.\nNet pay: CAD {:,.2f}.\nReference: {}.'.format(gross, gross * .71, reference('PAY')))

    def card():
        period = month()
        return ('04_finance\\cards\\Fictional Summit Card statement ' + period + '.txt',
                '# Credit card statement\nStatement period: ' + period + '-01 to ' + period + '-28.\nNew balance: ' +
                money(80, 3900) + '.\nMinimum payment: CAD 10.00.\nPayment due date: ' + day() + '.')

    def bill():
        period, kind = month(), rng.choice(['water', 'internet', 'waste collection'])
        return ('04_finance\\bills\\Fictional ' + kind + ' bill ' + period + '.txt',
                '# ' + kind.capitalize() + ' bill\nAccount: ' + reference('ACC') + '.\nAmount due: ' + money(20, 190) +
                '.\nDue date: ' + day() + '.')

    def bank():
        period = month()
        return ('04_finance\\bank\\Fictional chequing statement ' + period + '.txt',
                '# Account statement\nOpening balance: ' + money(300, 9000) + '.\nClosing balance: ' + money(300, 9000) +
                '.\nStatement month: ' + period + '.')

    def newsletter():
        period = month()
        return ('02_family\\school\\Newsletter ' + period + '.txt',
                '# School newsletter\nThe library is collecting gently used books this month. Families are reminded that '
                'pick-up begins at half past three and that the parking lot closes during assemblies. Issue ' + period + '.')

    def clinic():
        date = day()
        return ('05_health\\visits\\Clinic visit ' + date + '.txt',
                '# Visit summary\nVisit date: ' + date + '.\nReason: seasonal check.\nFollow-up: none required.\nFile: ' +
                reference('VIS') + '.')

    def receipt():
        date, store = day(), rng.choice(stores)
        return ('05_misc\\receipts\\Receipt ' + store + ' ' + date + '.txt',
                '# Receipt\nStore: ' + store + '.\nDate: ' + date + '.\nTotal: ' + money(4, 260) + '.\nTransaction: ' +
                reference('TXN') + '.')

    def manual():
        product = rng.choice(products)
        return ('05_misc\\manuals\\' + product.capitalize() + ' manual ' + reference('MAN') + '.txt',
                '# ' + product.capitalize() + ' quick start\nUnpack the ' + product + ' and remove the protective film. '
                'Keep the appliance away from water and unplug it before cleaning. Register the product to activate support.')

    def trip():
        city, date = rng.choice(cities), day()
        return ('02_family\\travel\\Fictional trip ' + city + ' ' + date + '.txt',
                '# Trip confirmation\nDestination: ' + city + '.\nDeparture: ' + date + ' at %02d:%02d.\nBooking reference: %s.'
                % (rng.randint(5, 22), rng.choice([0, 10, 25, 40, 55]), reference('TRP')))

    def slip():
        year = rng.randint(2012, 2023)
        return ('04_finance\\tax\\Fictional T5 ' + str(year) + ' ' + reference('SLP') + '.txt',
                '# Statement of investment income\nPayer: Larch Fictional Savings.\nBox 13 - Interest: {:,.2f}\nYear: {}'
                .format(rng.uniform(3, 420), year))

    def notes():
        date, topic = day(), rng.choice(topics)
        return ('03_work\\notes\\Meeting notes ' + date + ' ' + topic + '.txt',
                '# Meeting notes\nTopic: ' + topic + '.\nThe team reviewed the open items for the project and agreed to '
                'revisit the plan next sprint. Action items were assigned and no decision about compensation was made.')

    def notice():
        number = rng.randint(1, 9999)
        return ('01_identity\\其他\\虚构社区通知 ' + str(number) + '.txt',
                '# 社区通知\n本周末小区进行例行消防检查，请各位住户保持楼道畅通。通知编号：虚构通字第' + str(number) + '号。')

    makers = [pay, pay, card, bill, bank, newsletter, clinic, receipt, manual, trip, slip, notes, notice]
    documents, files = [], set()
    while len(documents) < count:
        file, text = rng.choice(makers)()
        lowered = text.casefold()
        if file in files or any(value in lowered for value in forbidden):
            continue
        files.add(file)
        documents.append({'id': 'pad-%04d' % len(documents), 'file': file, 'scope': 'personal', 'text': text})
    return documents


def padded_fixture(path, count, directory):
    fixture = json.loads(path.read_text(encoding='utf-8'))
    forbidden = {value.casefold() for question in fixture['questions'] for value in question.get('required_text', [])}
    fixture['documents'] = fixture['documents'] + padding_documents(count, forbidden)
    fixture['padding_documents'] = count
    target = directory / (path.stem + '.padded-' + str(count) + '.json')
    content = json.dumps(fixture, ensure_ascii=False, indent=1)
    if target.exists() and target.read_text(encoding='utf-8') != content:
        raise ValueError('An existing padded fixture differs from the generated one')
    target.write_text(content, encoding='utf-8')
    return target


def worker(code, fixture_path, mode, repetitions):
    sys.path.insert(0, str(Path(code).resolve()))
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', MR_LOG='0')
    from unittest.mock import patch
    from mr_private import PrivateRAG, make_chunks
    from mr_private_eval import evaluate
    fixture = json.loads(Path(fixture_path).read_text(encoding='utf-8'))
    if not fixture.get('synthetic'):
        raise ValueError('Only explicitly fictional fixtures are allowed')
    documents = fixture['documents']
    encoder_factory = None
    if mode == 'hybrid':
        from mr_private_semantic import LocalEncoder
        encoder = LocalEncoder()
        encoder_factory = lambda: encoder
    rag = PrivateRAG({'documents': documents, 'chunks': make_chunks(documents)}, mode=mode,
                     encoder_factory=encoder_factory)
    try:
        with patch('socket.socket.connect', side_effect=AssertionError('Unexpected network access')), \
             patch('socket.create_connection', side_effect=AssertionError('Unexpected network access')):
            started = time.perf_counter()
            if mode == 'hybrid':
                for scope in sorted({d['scope'] for d in documents}):
                    rag.prepare_semantic(scope)
            preparation = time.perf_counter() - started
            report = evaluate(rag, fixture, repetitions=repetitions)
    finally:
        rag.close()
    report['preparation_seconds'] = round(preparation, 3)
    return report


def run_worker(code, fixture_path, mode, repetitions):
    command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--worker',
               '--code', str(code), '--fixture', str(fixture_path), '--mode', mode,
               '--repetitions', str(repetitions)]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', cwd=str(code))
    if result.returncode:
        raise RuntimeError('Worker failed for ' + str(code) + ' (' + mode + '): ' + result.stderr[-2000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def compare(before, after):
    old = {row['id']: row for row in before['questions']}
    new = {row['id']: row for row in after['questions']}
    if set(old) != set(new) or before['summary']['labels_sha256'] != after['summary']['labels_sha256']:
        raise ValueError('Paired comparisons require identical questions and frozen labels')
    positives = sorted(key for key, row in old.items() if row['positive'])
    negatives = sorted(key for key, row in old.items() if not row['positive'])
    def changed(ok):
        return {'gains': [key for key in positives if not ok(old[key]) and ok(new[key])],
                'regressions': [key for key in positives if ok(old[key]) and not ok(new[key])]}
    return {'positive_questions': len(positives),
            'file': changed(lambda row: row['rank'] is not None),
            'top1': changed(lambda row: row['rank'] == 1),
            'evidence': changed(lambda row: row['evidence_complete']),
            'still_missing_file': [key for key in positives if new[key]['rank'] is None],
            'still_missing_evidence': [key for key in positives if not new[key]['evidence_complete']],
            'negatives_with_candidates': {'before': sum(bool(old[key]['candidates']) for key in negatives),
                                          'after': sum(bool(new[key]['candidates']) for key in negatives)}}


def headline(summary):
    return {key: summary[key] for key in ('positive_questions', 'file_hits_at_k', 'evidence_complete_at_k',
                                           'preview_complete_at_k', 'mrr', 'negative_questions',
                                           'negative_related_candidates', 'modes', 'warm_p95_seconds')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-code', type=Path)
    parser.add_argument('--fixture', type=Path, action='append', default=[])
    parser.add_argument('--regression-fixture', type=Path, action='append', default=[],
                        help='Previously consumed fixture, reported only as a known regression')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', choices=('bm25', 'hybrid'), action='append')
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--pad', type=int, default=0,
                        help='Add this many generated fictional filler records to the personal scope')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--code', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.code, args.fixture[0], args.mode[0], args.repetitions), ensure_ascii=False))
        return 0
    if not args.baseline_code or not (args.fixture or args.regression_fixture) or not args.output:
        parser.error('--baseline-code, a fixture and --output are required')
    modes = args.mode or ['bm25', 'hybrid']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error('Refusing to overwrite an existing comparison report')
    baseline, candidate = args.baseline_code.resolve(), CODE
    frozen_code = {'baseline': code_fingerprint(baseline), 'candidate': code_fingerprint(candidate)}
    report = {'version': 1, 'boundary': 'Fictional fixtures only; not real private accuracy.',
              'baseline_code': str(baseline), 'code_sha256': frozen_code, 'datasets': {}}
    for path in [*args.fixture, *args.regression_fixture]:
        fixture = json.loads(path.read_text(encoding='utf-8'))
        if not fixture.get('synthetic'):
            raise ValueError('Only explicitly fictional fixtures are allowed')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        holdout = path in args.fixture and any(q.get('split') == 'holdout' for q in fixture['questions'])
        if holdout:
            # Exclusive creation: an interrupted run still counts as consumed.
            marker = args.output.parent / (path.stem + '.consumed.json')
            with marker.open('x', encoding='utf-8') as stream:
                json.dump({'fixture_sha256': digest, 'started_epoch': time.time()}, stream)
        # Padding is added after the reservation, so it cannot be used to rerun a holdout as fresh.
        measured = padded_fixture(path, args.pad, args.output.parent) if args.pad else path
        dataset = {'fixture_sha256': digest, 'documents': len(fixture['documents']) + args.pad,
                   'padding_documents': args.pad,
                   'evaluation_role': ('fresh_once_holdout' if holdout else
                                       'known_regression' if path in args.regression_fixture else 'development'),
                   'label_origin': fixture.get('label_origin'), 'modes': {}}
        for mode in modes:
            before = run_worker(baseline, measured.resolve(), mode, args.repetitions)
            after = run_worker(candidate, measured.resolve(), mode, args.repetitions)
            # Canonical label hash: unlike the file hash it does not depend on line endings.
            dataset['labels_sha256'] = after['summary']['labels_sha256']
            dataset['modes'][mode] = {'before': headline(before['summary']), 'after': headline(after['summary']),
                                      'changes': compare(before, after),
                                      'questions': {'before': before['questions'], 'after': after['questions']}}
            changes = dataset['modes'][mode]['changes']
            print(json.dumps({'fixture': path.name, 'mode': mode,
                              'before': {k: before['summary'][k] for k in ('file_hits_at_k', 'evidence_complete_at_k', 'mrr')},
                              'after': {k: after['summary'][k] for k in ('file_hits_at_k', 'evidence_complete_at_k', 'mrr')},
                              'positives': changes['positive_questions'],
                              'file_gains': len(changes['file']['gains']), 'file_regressions': len(changes['file']['regressions']),
                              'evidence_gains': len(changes['evidence']['gains']),
                              'evidence_regressions': len(changes['evidence']['regressions'])}, ensure_ascii=False), flush=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Fixture changed during the comparison')
        report['datasets'][path.name] = dataset
    if {'baseline': code_fingerprint(baseline), 'candidate': code_fingerprint(candidate)} != frozen_code:
        raise ValueError('Code changed during the comparison')
    regressions = sum(len(row['changes'][field]['regressions']) for data in report['datasets'].values()
                      for row in data['modes'].values() for field in ('file', 'evidence'))
    report['no_file_or_evidence_regressions'] = regressions == 0
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if regressions == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
