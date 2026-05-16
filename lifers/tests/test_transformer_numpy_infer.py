"""NumPy 推理路径与纯 Python ``forward`` 一致性（``generate_text`` / ``forward_np``）。"""
from __future__ import annotations

import os
import unittest

from lifers.transformer_lm import (
    TinyTransformerWeights,
    _np_tensors_from_weights,
    _try_numpy,
    forward,
    forward_np,
    generate_text,
    init_weights,
)


class TransformerNumpyInferTests(unittest.TestCase):
    def _tiny_weights(self) -> TinyTransformerWeights:
        vocab = ["a", "b", "c", "\n"]
        return init_weights(vocab, d_model=8, d_ff=16, n_heads=2, max_seq=32, seed=99)

    def test_forward_np_matches_forward(self) -> None:
        np_mod = _try_numpy()
        if np_mod is None:
            self.skipTest("numpy not installed")
        w = self._tiny_weights()
        ids = [0, 1, 2]
        logits_py = forward(w, ids)
        te, pe, Wq, Wk, Wv, Wo, W1, W2, Wlm = _np_tensors_from_weights(w, np_mod)
        logits_np = forward_np(w, te, pe, Wq, Wk, Wv, Wo, W1, W2, Wlm, ids, np_mod)
        self.assertEqual(len(logits_py), int(logits_np.shape[0]))
        self.assertEqual(len(logits_py[0]), int(logits_np.shape[1]))
        for i, row in enumerate(logits_py):
            for j, v in enumerate(row):
                self.assertAlmostEqual(v, float(logits_np[i, j]), places=5)

    def test_generate_text_numpy_matches_python_top1(self) -> None:
        np_mod = _try_numpy()
        if np_mod is None:
            self.skipTest("numpy not installed")
        prev = os.environ.get("LIFERS_USE_NUMPY")
        try:
            w = self._tiny_weights()
            os.environ.pop("LIFERS_USE_NUMPY", None)
            out_np = generate_text(w, "a", max_chars=8, seed=7, temperature=0.5, top_k=1)
            os.environ["LIFERS_USE_NUMPY"] = "0"
            out_py = generate_text(w, "a", max_chars=8, seed=7, temperature=0.5, top_k=1)
            self.assertEqual(out_np, out_py)
        finally:
            if prev is None:
                os.environ.pop("LIFERS_USE_NUMPY", None)
            else:
                os.environ["LIFERS_USE_NUMPY"] = prev


if __name__ == "__main__":
    unittest.main()
