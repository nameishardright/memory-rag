import hashlib
import unittest

from mr_private import PrivateRAG, make_chunks
from mr_private_facts import compact_cards, compact_search, expand_query, extract_cards


def document(name, text, scope='personal'):
    return {'id': name, 'file': name, 'text': text, 'scope': scope,
            'text_sha256': hashlib.sha256(text.encode()).hexdigest()}


class PrivateFactTests(unittest.TestCase):
    def test_card_fields_do_not_include_adjacent_authentication_values(self):
        docs = [document('notes.txt', 'Rogers 5555555555554444 999 12/30 password=SyntheticSecret\n'
                        'amex hilton 3782 822463 10005 1234\n'
                        'pc money acc 4111111111111111 888')]
        cards = extract_cards(docs)
        self.assertEqual({c['number'] for c in cards}, {'5555555555554444', '378282246310005'})
        compact = str(compact_cards(cards))
        for secret in ('SyntheticSecret', '999', '12/30', '1234', '888'):
            self.assertNotIn(secret, compact)
        self.assertEqual(len(extract_cards(docs, include_bank_accounts=True)), 3)

    def test_duplicate_sources_and_payment_form_account_label(self):
        docs = [document('notes.txt', 'Rogers 5555555555554444'),
                document('form.pdf', 'Credit Card Payment Details\nMastercard\nAccount Number: 5555555555554444')]
        cards = extract_cards(docs)
        self.assertEqual(len(cards), 1)
        self.assertEqual(len(cards[0]['sources']), 2)
        self.assertFalse(cards[0]['bank_evidence'])

    def test_luhn_is_not_enough_and_numeric_fields_cannot_be_concatenated(self):
        docs = [document('lesson.txt', 'Array sequence 4111111111111111'),
                document('notes.txt', 'visa 4111111111111112\nvisa 4111 1111 1111 1111 999'),
                document('dependency.js', 'visa 5555555555554444', scope='technical')]
        cards = extract_cards(docs)
        self.assertEqual([c['number'] for c in cards], ['4111111111111111'])
        self.assertEqual(cards[0]['sources'][0]['line'], 2)

    def test_source_labels_and_conflicts_are_preserved(self):
        docs = [document('old.txt', 'cobalt\n378282246310005'),
                document('new.txt', 'gold\n378282246310005')]
        card = extract_cards(docs)[0]
        self.assertIn('cobalt', card['label'])
        self.assertIn('gold', card['label'])
        self.assertEqual(len(card['sources']), 2)

    def test_previous_bank_line_does_not_reclassify_next_visa_record(self):
        docs = [document('notes.txt', 'saving client 4111111111111111\nvisa 4012888888881881')]
        cards = extract_cards(docs)
        self.assertEqual([c['number'] for c in cards], ['4012888888881881'])
        self.assertEqual(cards[0]['label'], 'visa')

    def test_compact_search_returns_bounded_evidence_with_source_identity(self):
        docs = [document('family.txt', '# Family\nfather mother spouse ' + 'evidence ' * 150)]
        rag = PrivateRAG({'documents': docs, 'chunks': make_chunks(docs)})
        result = compact_search(rag, '家庭', preview_chars=100)
        self.assertTrue(result['hits'])
        self.assertLessEqual(len(result['hits'][0]['excerpt']), 100)
        self.assertIn('document_id', result['hits'][0])
        self.assertIn('father', expand_query('家庭'))


if __name__ == '__main__':
    unittest.main()
