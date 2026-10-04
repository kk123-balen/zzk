import unittest

from minigpt.baseline import BigramModel


class BaselineTests(unittest.TestCase):
    def test_learns_next_character(self):
        model = BigramModel().fit("ababab")
        self.assertEqual(model.generate("a", 5), "ababab")

    def test_unknown_prompt_and_seed(self):
        model = BigramModel().fit("abcacb")
        generated = model.generate("?", 12, seed=7)
        self.assertEqual(len(generated), 13)
        self.assertEqual(generated, model.generate("?", 12, seed=7))
        self.assertTrue(set(generated[1:]) <= set("abc"))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            BigramModel().fit("a")
        with self.assertRaises(ValueError):
            BigramModel().generate("a")
        with self.assertRaises(ValueError):
            BigramModel().fit("ab").generate("a", -1)

    def test_refit_replaces_old_counts(self):
        model = BigramModel().fit("ab").fit("xyxy")
        self.assertEqual(model.generate("x", 3), "xyxy")
