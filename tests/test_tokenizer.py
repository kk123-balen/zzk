import json
import tempfile
import unittest
from pathlib import Path

from minigpt.tokenizer import CharTokenizer


class TokenizerTests(unittest.TestCase):
    def test_deterministic_vocabulary_and_roundtrip(self):
        tokenizer = CharTokenizer("b你a\n b")
        self.assertEqual(tokenizer.stoi, CharTokenizer("你ba b\n").stoi)
        self.assertEqual(tokenizer.vocabulary, ("\n", " ", "a", "b", "你"))
        self.assertEqual(tokenizer.encode("ab你"), [3, 4, 5])
        text = "你 ab\nb你"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.vocab_size, 6)

    def test_unknown_characters_preserve_positions(self):
        tokenizer = CharTokenizer("ab")
        self.assertEqual(tokenizer.encode("a?你b"), [1, 0, 0, 2])
        self.assertEqual(tokenizer.decode([1, 0, 0, 2]), "a<UNK><UNK>b")
        self.assertEqual(tokenizer.vocab_size, 3)

    def test_save_load(self):
        tokenizer = CharTokenizer("你\n ab")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocab.json"
            tokenizer.save(path)
            loaded = CharTokenizer.load(path)
            self.assertEqual(loaded.stoi, tokenizer.stoi)
            self.assertEqual(loaded.itos, tokenizer.itos)
            self.assertEqual(loaded.encode("你? ab\n"), tokenizer.encode("你? ab\n"))
            self.assertEqual(loaded.decode(range(loaded.vocab_size)), tokenizer.decode(range(tokenizer.vocab_size)))
            loaded.save(path)
            self.assertEqual(CharTokenizer.load(path).stoi, tokenizer.stoi)

    def test_invalid_vocabulary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocab.json"
            for payload in ({}, {"version": 2, "unk_id": 0, "characters": []},
                            {"version": 1, "unk_id": 0, "characters": ["b", "a"]},
                            {"version": 1, "unk_id": 0, "characters": ["a", "a"]},
                            {"version": 1, "unk_id": 0, "characters": ["ab"]}):
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        CharTokenizer.load(path)

    def test_empty_and_invalid_inputs(self):
        tokenizer = CharTokenizer("")
        self.assertEqual(tokenizer.encode(""), [])
        self.assertEqual(tokenizer.decode([]), "")
        self.assertEqual(tokenizer.encode("a"), [0])
        for token_id in (-1, 1, True, 0.0, "0"):
            with self.subTest(token_id=token_id), self.assertRaises(ValueError):
                tokenizer.decode([token_id])
        with self.assertRaises(ValueError):
            CharTokenizer(["ab"])
