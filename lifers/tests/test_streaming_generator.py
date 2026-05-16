"""Tests for lifers.streaming_generator."""
from __future__ import annotations

import unittest
from pathlib import Path

from lifers.markov_lm import MarkovWeights
from lifers.streaming_generator import iter_markov_chars, iter_transformer_chars
from lifers.transformer_lm import TinyTransformerWeights

ROOT = Path(__file__).resolve().parents[1]


class StreamingGeneratorTests(unittest.TestCase):
    def test_iter_markov_chars_length(self) -> None:
        wp = ROOT / "weights" / "lifers_markov.json"
        if not wp.is_file():
            self.skipTest("lifers_markov.json missing")
        w = MarkovWeights.load(wp)
        chars = list(iter_markov_chars(w, "hello", max_chars=7, seed=42, root=ROOT))
        self.assertEqual(len(chars), 7)
        self.assertTrue(all(isinstance(c, str) and len(c) == 1 for c in chars))

    def test_iter_transformer_chars_length(self) -> None:
        wp = ROOT / "weights" / "lifers_transformer.json"
        if not wp.is_file():
            self.skipTest("lifers_transformer.json missing")
        w = TinyTransformerWeights.load(wp)
        chars = list(iter_transformer_chars(w, "ab", max_chars=5, seed=1, root=ROOT))
        self.assertEqual(len(chars), 5)


if __name__ == "__main__":
    unittest.main()
