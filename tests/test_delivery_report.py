from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.make_report import corpus_change_text, delivery_state, require_committed


class DeliveryReportTests(unittest.TestCase):
    def test_corpus_changed_is_an_affirmative_sentence(self):
        text = corpus_change_text(False)
        self.assertIn("发生了变化", text)
        self.assertIn("冻结快照", text)
        self.assertNotIn("False", text)
        self.assertNotIn("未变化", text)

    def test_stable_corpus_is_reported_plainly(self):
        self.assertEqual(corpus_change_text(True), "运行期间实时语料保持一致。")

    def test_unknown_corpus_state_is_not_rendered_as_stable(self):
        with self.assertRaises(ValueError):
            corpus_change_text(None)

    def test_actual_git_state_controls_delivery_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            def git(*args):
                subprocess.run(["git", "-C", str(repo), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                                "-c", "commit.gpgsign=false", *args], check=True, capture_output=True)
            git("init", "--quiet")
            with self.assertRaises(RuntimeError):
                require_committed(delivery_state(repo))
            file = repo / "code.txt"
            file.write_text("tested\n", encoding="utf-8")
            self.assertFalse(delivery_state(repo)["worktree_clean"])
            git("add", "--", "code.txt")
            git("commit", "--quiet", "-m", "fixture")
            state = delivery_state(repo)
            require_committed(state)
            self.assertEqual(len(state["head"]), 40)
            file.write_text("changed\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                require_committed(delivery_state(repo))


if __name__ == "__main__":
    unittest.main()
