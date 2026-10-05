import unittest
from pathlib import Path

from minigpt.data import make_windows, prepare_data, split_text
from minigpt.tokenizer import CharTokenizer


class DataTests(unittest.TestCase):
    def test_abcd_windows(self):
        tokenizer = CharTokenizer("abcd")
        self.assertEqual(make_windows(tokenizer.encode("abcd"), 2),
                         [([1, 2], [2, 3]), ([2, 3], [3, 4])])
        self.assertEqual(make_windows(tokenizer.encode("abcd"), 3), [([1, 2, 3], [2, 3, 4])])

    def test_short_text(self):
        for ids in ([], [1], [1, 2]):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "至少需要 3"):
                make_windows(ids, 2)
        with self.assertRaisesRegex(ValueError, "训练集.*文本太短"):
            prepare_data("abcd", 2, 0.5)
        with self.assertRaisesRegex(ValueError, "验证集.*文本太短"):
            prepare_data("abcdefgh", 2, 0.75)

    def test_split_before_windows_and_train_only_vocabulary(self):
        text = "abcdabcdWXYZ"
        self.assertEqual(split_text(text, 2 / 3), ("abcdabcd", "WXYZ"))
        tokenizer, train, val = prepare_data(text, 2, 2 / 3)
        self.assertEqual(tokenizer.vocabulary, tuple("abcd"))
        self.assertEqual(train, [([1, 2], [2, 3]), ([2, 3], [3, 4]),
                                ([3, 4], [4, 1]), ([4, 1], [1, 2]),
                                ([1, 2], [2, 3]), ([2, 3], [3, 4])])
        self.assertEqual(val, [([0, 0], [0, 0]), ([0, 0], [0, 0])])
        self.assertEqual(len(train) + len(val), 8)  # 边界上的两个窗口必须被排除。

    def test_preserves_whitespace(self):
        self.assertEqual(split_text(" a\n b\n", 0.5), (" a\n", " b\n"))

    def test_invalid_parameters(self):
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "正整数"):
                make_windows([1, 2, 3], size)
        for fraction in (0, 1, -1, True, "0.9", float("nan")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                split_text("abcd", fraction)
        with self.assertRaises(ValueError):
            make_windows([1, -1, 2], 2)

    def test_repository_corpus(self):
        path = Path(__file__).resolve().parents[1] / "data/tiny.txt"
        text = path.read_text(encoding="utf-8")
        tokenizer, train, val = prepare_data(text, 8)
        train_text, val_text = split_text(text)
        self.assertEqual(len(train), len(train_text) - 8)
        self.assertEqual(len(val), len(val_text) - 8)
        self.assertEqual(train[0], (tokenizer.encode(train_text[:8]), tokenizer.encode(train_text[1:9])))
