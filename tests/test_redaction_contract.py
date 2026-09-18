import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mr_corpus
import mr_tools
import mr_answer_eval
from scripts import eval_answers


class RedactionContractTests(unittest.TestCase):
    def test_inline_login_and_authenticated_url_are_redacted(self):
        text='新账号 `reader:Example12345`；地址 https://reader:Example12345@example.com'
        clean=mr_corpus.redact(text)
        self.assertNotIn('Example12345',clean)
        self.assertIn('reader:<REDACTED>',clean)

    def test_json_and_quoted_spaces_preserve_format_and_hide_values(self):
        raw='{"password": "example with spaces", "api_key":"Demo12345", "count":2}'
        clean=mr_corpus.redact(raw)
        self.assertEqual(json.loads(clean),{'password':'<REDACTED>','api_key':'<REDACTED>','count':2})

    def test_ports_timestamps_and_token_counts_are_not_credentials(self):
        text='`example.com:47021` 19:57:53 http://example.com:8080 token_count=512 max_tokens=2048'
        self.assertEqual(mr_corpus.redact(text),text)

    def test_auth_headers_and_redaction_are_idempotent(self):
        text='Authorization: Bearer Demo12345\n新账号 `reader:Example12345`\npassword="another example"'
        clean=mr_corpus.redact(text)
        self.assertNotIn('Demo12345',clean)
        self.assertEqual(mr_corpus.redact(clean),clean)

    def test_case_source_and_get_use_same_redacted_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'note.md').write_text('# 复盘\n新账号 `reader:Example12345`，随后验证恢复。',encoding='utf-8')
            with patch.object(mr_corpus,'SOURCES',[{'key':'case','dir':directory,'redact':False}]), \
                 patch.object(mr_corpus,'CORPUS_POLICY',str(root/'none.json')), \
                 patch.object(mr_corpus,'pdf_enabled',return_value=False), patch.object(mr_tools,'LOG_ON',False):
                chunks=mr_corpus.load_chunks()
                self.assertNotIn('Example12345',json.dumps(chunks))
                c=chunks[0]
                got=mr_tools.memory_get('note.md',source='case',heading=c['heading'],chunk_md5=c['md5'])
                self.assertTrue(got['ok'])
                self.assertEqual(got['text'],c['text'])

    def test_pdf_index_and_get_both_redact(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pdf=root/'book.pdf';pdf.touch()
            pages=['教材演示内容。'*10+'\napi_key="Example12345"']
            with patch.object(mr_corpus,'PDF_DIR',directory), patch.object(mr_corpus,'CORPUS_POLICY',str(root/'none.json')), \
                 patch.object(mr_corpus,'pdf_files',return_value=[str(pdf)]), \
                 patch.object(mr_corpus,'extract_pdf',return_value=pages), patch.object(mr_tools,'LOG_ON',False):
                self.assertNotIn('Example12345',json.dumps(mr_corpus.load_pdf_chunks()))
                self.assertNotIn('Example12345',mr_tools.memory_get('book.pdf#p1',source='pdf')['text'])

    def test_model_boundary_blocks_unredacted_context_before_launch(self):
        c={'id':'e1','source':'case','file':'x','heading':'h','text':'新账号 `reader:Example12345`'}
        with patch.object(eval_answers.subprocess,'run') as launch:
            with self.assertRaisesRegex(ValueError,'request blocked'):
                eval_answers.generate({'argv':['never-run']},{'q':'question'},[c],'.')
            launch.assert_not_called()

    def test_nested_model_envelope_is_sanitized_without_json_corruption(self):
        data={'result':json.dumps({'claims':[{'text':'新账号 `reader:Example12345`'}]},ensure_ascii=False),
              'usage':{'input_tokens':10},'password':'Example12345'}
        clean=mr_corpus.redact_values(data)
        self.assertNotIn('Example12345',json.dumps(clean))
        self.assertIsInstance(json.loads(clean['result']),dict)
        self.assertEqual(clean['usage']['input_tokens'],10)

    def test_context_truncation_does_not_turn_placeholder_into_a_false_secret_alarm(self):
        text='prefix password=<REDACTED> trailing words'
        hit={'source':'case','file':'x','heading':'h','snippet':text}
        result=mr_answer_eval.assemble_context({'hits':[hit]},policy='snippet',max_chars=21)
        self.assertLessEqual(result['characters'],21)
        for c in result['contexts']:
            self.assertEqual(mr_corpus.redact(c['text']),c['text'])


if __name__=='__main__':
    unittest.main()
