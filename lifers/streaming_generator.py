"""本地 Markov / TinyTransformer 的逐字符流式生成（生成器）。

用于 lifers_gate `/v1/stream` 或脚本侧 UX；完整 taskflow 仍走 `bridge_turn.lifers_turn`。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, List

from lifers.fix_inference_prompt import clip_prompt_for_transformer
from lifers.markov_lm import MarkovWeights, _sample_next
from lifers.speed_env import use_numpy_training
from lifers.stack_env import load_stack
from lifers.transformer_lm import (
    TinyTransformerWeights,
    _np_tensors_from_weights,
    _topk_sample_token_id,
    _try_numpy,
    decode,
    encode,
    forward,
    forward_np,
)


def _sampling_from_stack(root: Path, backend: str) -> tuple[float, int]:
    defaults = {"transformer": (1.1, 80), "markov": (0.9, 80)}
    d_temp, d_top = defaults.get(backend, (1.1, 80))
    try:
        stack = load_stack(root)
        brain = stack.get("brain") or {}
        block = (brain.get("local_lm_sampling") or {}).get(backend) or {}
        if not isinstance(block, dict):
            return d_temp, d_top
        t = float(block.get("temperature", d_temp))
        k = int(block.get("top_k", d_top))
        return max(0.01, min(t, 4.0)), max(1, min(k, 256))
    except (TypeError, ValueError):
        return d_temp, d_top


def iter_transformer_chars(
    w: TinyTransformerWeights,
    prompt: str,
    *,
    max_chars: int,
    seed: int = 1,
    root: Path | None = None,
) -> Iterator[str]:
    """与 `transformer_lm.generate_text` 同分布，但每步 yield 一个新 token 对应字符。"""
    rng = random.Random(seed)
    temp, top_k = _sampling_from_stack(root, "transformer") if root is not None else (1.1, 80)
    stoi = {ch: i for i, ch in enumerate(w.vocab)}
    itos = w.vocab
    prompt = clip_prompt_for_transformer(prompt, w.max_seq)
    ids: List[int] = encode(prompt, stoi)
    np_mod = _try_numpy()
    use_np = use_numpy_training(np_mod is not None)

    if use_np and np_mod is not None:
        tok_emb, pos_emb, Wq, Wk, Wv, Wo, W1, W2, Wlm = _np_tensors_from_weights(w, np_mod)
        for _ in range(max_chars):
            chunk = ids[-w.max_seq :]
            logits_arr = forward_np(w, tok_emb, pos_emb, Wq, Wk, Wv, Wo, W1, W2, Wlm, chunk, np_mod)
            pick = _topk_sample_token_id(logits_arr[-1], rng, temp, top_k)
            ids.append(pick)
            ch = decode([pick], itos)
            if ch:
                yield ch
        return

    for _ in range(max_chars):
        logits = forward(w, ids[-w.max_seq :])
        pick = _topk_sample_token_id(logits[-1], rng, temp, top_k)
        ids.append(pick)
        ch = decode([pick], itos)
        if ch:
            yield ch


def iter_markov_chars(
    w: MarkovWeights,
    prompt: str,
    *,
    max_chars: int,
    seed: int = 1,
    root: Path | None = None,
) -> Iterator[str]:
    rng = random.Random(seed)
    temp, top_k = _sampling_from_stack(root, "markov") if root is not None else (0.9, 80)
    prev = prompt[-1] if prompt else "\n"
    for _ in range(max_chars):
        ch = _sample_next(w, prev=prev, rng=rng, temperature=temp, top_k=top_k)
        yield ch
        prev = ch
