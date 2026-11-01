import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from minigpt.tokenizer import CharTokenizer


class DataModuleTests(unittest.TestCase):
    def run_module(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "minigpt.data", *map(str, args)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=10,
        )

    def test_default_corpus_runs_as_module(self):
        result = self.run_module()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("训练字符数：141；验证字符数：16", result.stdout)
        self.assertIn("训练窗口数：133；验证窗口数：8", result.stdout)
        self.assertIn("训练集首个窗口：x=", result.stdout)
        self.assertIn("验证集首个窗口：x=", result.stdout)

    def test_abcd_module_demo_and_saved_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "input.txt"
            vocab = Path(directory) / "vocab.json"
            data.write_text("abcdabcd", encoding="utf-8")
            result = self.run_module(
                "--data", data, "--block-size", 2, "--train-fraction", 0.5,
                "--vocab-out", vocab,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("训练窗口数：2；验证窗口数：2", result.stdout)
            self.assertIn("x=[1, 2], y=[2, 3]", result.stdout)
            self.assertIn("解码：x='ab', y='bc'", result.stdout)
            self.assertEqual(CharTokenizer.load(vocab).encode("abcd?"), [1, 2, 3, 4, 0])

    def test_short_corpus_and_invalid_parameters_report_cli_errors(self):
        for args, message in (
            (("--block-size", 32), "验证集.*至少需要 33.*实际 16"),
            (("--block-size", 0), "block_size 必须是正整数"),
            (("--train-fraction", 1), "train_fraction 必须在 0 和 1 之间"),
        ):
            with self.subTest(args=args):
                result = self.run_module(*args)
                self.assertEqual(result.returncode, 2)
                self.assertRegex(result.stderr, message)
                self.assertNotIn("Traceback", result.stderr)

    def test_missing_or_non_utf8_file_reports_cli_error(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "input.txt"
            missing = self.run_module("--data", data)
            self.assertEqual(missing.returncode, 2)
            self.assertIn(str(data), missing.stderr)
            self.assertNotIn("Traceback", missing.stderr)
            data.write_bytes(b"\xff\xfe")
            invalid = self.run_module("--data", data)
            self.assertEqual(invalid.returncode, 2)
            self.assertIn("utf-8", invalid.stderr)
            self.assertNotIn("Traceback", invalid.stderr)
