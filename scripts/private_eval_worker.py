"""Isolated encrypted benchmark worker for an explicitly selected code version."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-code', type=Path, required=True)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('bm25', 'hybrid'), required=True)
    args = parser.parse_args()
    public_checkout = Path(__file__).resolve().parents[1]
    if any(path.resolve().is_relative_to(public_checkout) for path in (args.store, args.labels, args.output)):
        parser.error('Private data, labels and detailed results must stay outside the checkout')
    sys.path.insert(0, str(args.backend_code.resolve()))
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', MR_LOG='0')
    from mr_private import load_private_rag, load_store, save_store
    from mr_private_eval import evaluate, validate_labels, labels_sha256
    from mr_private_benchmark import source_signature
    password = sys.stdin.readline().rstrip('\r\n')
    rag = None
    try:
        labels = load_store(args.labels, password)
        validate_labels(labels)
        if any(q.get('split') == 'holdout' for q in labels['questions']):
            marker = load_store(Path(str(args.labels) + '.consumed.enc'), password)
            if marker['labels_sha256'] != labels_sha256(labels):
                raise ValueError('Holdout is not reserved by the paired controller')
        rag = load_private_rag(args.store, password, mode=args.mode, use_index_cache=True)
        signature = source_signature(rag, labels)
        for question in labels['questions']:
            expected_hash = question.get('source_text_sha256')
            if expected_hash and len(question.get('expected_documents', [])) == 1:
                doc = rag.get(question['expected_documents'][0])
                if not doc.get('ok') or hashlib.sha256(doc['text'].encode()).hexdigest() != expected_hash:
                    raise ValueError('Source changed after question authoring')
        report = evaluate(rag, labels, topk=5, repetitions=1)
        if source_signature(rag, labels) != signature:
            raise ValueError('Source changed during evaluation')
        save_store(args.output, password, {'version': 1, 'report': report,
                                          'source_signature': signature})
        return 0
    except Exception as error:
        print(json.dumps({'error_type': type(error).__name__}))
        return 2
    finally:
        if rag is not None:
            rag.close()
        password = None


if __name__ == '__main__':
    raise SystemExit(main())
