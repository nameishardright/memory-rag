import unittest
from unittest import mock

import mr_corpus
import mr_private


def headings(text):
    return [h for h, _ in mr_corpus._split_heading(text)]


def lost_lines(text, blocks):
    blob = "\n".join(h + "\n" + b for h, b in blocks)
    return [ln for ln in text.splitlines() if ln.strip() and ln.lstrip("#").strip() not in blob]


class FencedCodeTests(unittest.TestCase):
    CASES = {
        "issue_example": ("# Setup\n\nInstall the dependencies:\n\n```bash\n# install deps\npip install x\n```\n\n"
                          "Then run the tool.\n", ["Setup"]),
        "adjacent_comments": ("## Deploy\n\n```bash\n# stop the old service\n# copy the new jar\n"
                              "cp app.jar /srv/app.jar\n```\n", ["Deploy"]),
        "longer_outer_fence": ("# Guide\n\n````markdown\n```bash\n# not a heading\n```\n# still inside\n````\n\n"
                               "# Next\ntext\n", ["Guide", "Next"]),
        "tilde_fence_with_backticks": ("# Guide\n\n~~~\n```\n# comment\n~~~\n\n# Next\ntext\n", ["Guide", "Next"]),
        "inline_code_at_line_start": ("# Notes\n```x``` is inline code.\n\n# Next\ntext\n", ["Notes", "Next"]),
        "indented_fence": ("# Steps\n  ```bash\n# inside\n  ```\n# Next\ntext\n", ["Steps", "Next"]),
        "fence_on_list_marker": ("- ```bash\n  echo ok\n  ```\n\n# Outside\nbody\n\n```bash\necho next\n```\n",
                                 ["(开头)", "Outside"]),
        "unclosed_top_level": ("# A\nintro\n```python\nprint(1)\n# B\nsection b\n## C\nsection c\n", ["A"]),
        "unclosed_in_list_item": ("- ```\n  code\n\n# Outside\nbody\n", ["(开头)", "Outside"]),
        "sibling_list_items": ("- ```\n  a\n- ~~~\n  b\n  ~~~\n\n# Outside\nbody\n", ["(开头)", "Outside"]),
        "html_block": ("<div>\n- ```\n</div>\n\n# Outside\nbody\n", ["(开头)", "Outside"]),
        "html_comment": ("<!--\n- ```\n-->\n\n# Outside\nbody\n", ["(开头)", "Outside"]),
        "pre_in_list_item": ("- <pre>\n  ```\n  </pre>\n\n# Outside\nbody\n", ["(开头)", "Outside"]),
        "pre_block": ("# A\n<pre>\n```\n# not a heading\n</pre>\n\n# B\ntext\n", ["A", "B"]),
        "nbsp_after_closer": ("```\n``` \n# lost\n# kept\nbody\n", ["(开头)"]),
        "thematic_break": ("* * *\n    ```\n# Outside\nbody\n", ["(开头)", "Outside"]),
        "indented_code_block": ("    - ```\n# Outside\nbody\n", ["(开头)", "Outside"]),
        "lazy_continuation": ("- item\nlazy\n    ```\n    code\n  ```\n\n# Outside\nbody\n", ["(开头)", "Outside"]),
        "tab_indented_fence": ("- item\n\t```\n\tcode\n  ```\n\n# Outside\nbody\n", ["(开头)", "Outside"]),
        "blockquote_fence": ("# A\n> ```bash\n> # quoted comment\n> ```\n\n# B\ntext\n", ["A", "B"]),
        "ideographic_space_outside_code": ("```\ncode\n```\n#　全角标题\nbody\n", ["(开头)", "全角标题"]),
    }

    def test_headings_follow_commonmark_code_blocks(self):
        for name, (text, expected) in self.CASES.items():
            with self.subTest(name=name):
                self.assertEqual(headings(text), expected)

    def test_no_line_is_dropped(self):
        for name, (text, _expected) in self.CASES.items():
            with self.subTest(name=name):
                self.assertEqual(lost_lines(text, mr_corpus._split_heading(text)), [])

    def test_issue_example_keeps_the_fence_together(self):
        (head, body), = mr_corpus._split_heading(self.CASES["issue_example"][0])
        self.assertEqual(head, "Setup")
        self.assertIn("```bash\n# install deps\npip install x\n```", body)

    def test_notes_without_code_keep_their_previous_chunks(self):
        text = ("---\ntitle: demo\n---\nintro line\n# Title\n## Part one\nbody one\n### Part two\nbody two\n"
                "#### deeper is not split\nbody three\n")
        with mock.patch.object(mr_corpus, "_raw_ranges", side_effect=AssertionError("parser used")):
            blocks = mr_corpus._split_heading(text)
        self.assertEqual([h for h, _ in blocks], ["(开头)", "Part one", "Part two"])
        self.assertEqual(blocks[2][1], "body two\n#### deeper is not split\nbody three")


class PrivateCodeFileTests(unittest.TestCase):
    TWO_COMMENTS = "# first comment\n# second comment\nbody line\n"

    def chunks(self, file, text, **extra):
        doc = {"id": "d", "file": file, "scope": "technical", "text": text, **extra}
        return mr_private.make_chunks([doc])

    def test_code_and_config_files_are_not_split_at_comments(self):
        files = ["src/tool.py", "scripts/backup.sh", "scripts/setup.ps1", "Dockerfile", "Dockerfile.dev", "Makefile",
                 "conf/app.ini", "conf/nginx.conf", "requirements.txt", "lib/a.rb", "lib/a.pl", "settings.env",
                 "deploy/docker-compose.yml", "x.zip!/a.py", "SRC/A.PY"]
        for file in files:
            with self.subTest(file=file):
                chunks = self.chunks("06_demo/" + file, self.TWO_COMMENTS)
                self.assertEqual([c["heading"] for c in chunks], ["(开头)"])
                self.assertIn("# first comment\n# second comment", chunks[0]["text"])

    def test_shebang_script_without_suffix_is_not_split(self):
        chunks = self.chunks("06_demo/tools/run", "#!/bin/sh\n" + self.TWO_COMMENTS)
        self.assertEqual([c["heading"] for c in chunks], ["(开头)"])

    def test_markdown_like_documents_still_split(self):
        text = "# Plan\n\nIntro.\n\n## Step one\nDo it.\n"
        for file in ("notes/plan.md", "notes/Dockerfile.md", "notes/todo.txt", "notes/requirements-review.txt",
                     "notes/constraints-for-trip.txt"):
            with self.subTest(file=file):
                self.assertEqual([c["heading"] for c in self.chunks(file, text)], ["Plan", "Step one"])
        onenote = self.chunks("Notebook.one!/OneNote/page.py", "# Page\n\nPacking list\n", method="onenote_com")
        self.assertEqual([c["heading"] for c in onenote], ["Page"])

    def test_project_catalogue_is_split_as_markdown(self):
        readme = {"id": "r", "file": "06_projects/service.py/README.md", "scope": "technical",
                  "text": "# Intro\nhello\n## Details\nmore\n", "text_sha256": "x", "purpose": "knowledge"}
        catalogue = mr_private._project_catalogue([readme])
        self.assertEqual(len(catalogue), 1)
        found = [c["heading"] for c in mr_private.make_chunks(catalogue)]
        self.assertEqual(found, ["项目资料导航", "Intro", "Details"])

    def test_long_code_files_are_still_split_by_size(self):
        code = "\n\n".join("def f%d():\n    return %d\n" % (i, i) + "x" * 150 for i in range(25))
        chunks = self.chunks("06_demo/big.py", code)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c["heading"].startswith("(开头)") for c in chunks))

    def test_partition_fingerprint_follows_chunking_rules(self):
        docs = [{"id": "a", "file": "06_demo/a.py", "scope": "technical", "text": "x", "text_sha256": "t"}]
        current = mr_private.partition_fingerprint(docs, "policy", "root")
        with mock.patch.object(mr_corpus, "CHUNKER_VERSION", "other"):
            self.assertNotEqual(mr_private.partition_fingerprint(docs, "policy", "root"), current)
        with mock.patch.object(mr_private, "PRIVATE_CHUNK_RULES", "other"):
            self.assertNotEqual(mr_private.partition_fingerprint(docs, "policy", "root"), current)
        self.assertEqual(mr_private.partition_fingerprint(docs, "policy", "root"), current)


if __name__ == "__main__":
    unittest.main()
