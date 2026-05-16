"""
Standalone edge inference runner — requires only numpy + model weights.
No lifers module dependency. Designed for edge deployment (Raspberry Pi, cloud VM, etc.).

Usage:
  python edge_inference.py --model weights/lifers_deep_transformer.npz --prompt "Hello"
  python edge_inference.py --serve --port 8080   # HTTP server mode
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# GELU activation
# ---------------------------------------------------------------------------

def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embedding)
# ---------------------------------------------------------------------------

_rope_cache: Dict[tuple, tuple] = {}


def _rope_apply(x: np.ndarray) -> np.ndarray:
    """Apply RoPE. x shape: [n_heads, T, head_dim]."""
    n_heads, T, head_dim = x.shape
    d2 = head_dim // 2
    if d2 == 0:
        return x
    key = (T, head_dim)
    if key in _rope_cache:
        cos, sin = _rope_cache[key]
    else:
        pos = np.arange(T, dtype=np.float64).reshape(T, 1)
        dim = np.arange(d2, dtype=np.float64).reshape(1, d2)
        theta = 1.0 / (10000.0 ** (2.0 * dim / head_dim))
        freqs = pos @ theta
        cos = np.cos(freqs).reshape(1, T, d2)
        sin = np.sin(freqs).reshape(1, T, d2)
        _rope_cache[key] = (cos, sin)
    x_even = x[:, :, 0::2]
    x_odd = x[:, :, 1::2]
    x_rot_even = cos * x_even - sin * x_odd
    x_rot_odd = sin * x_even + cos * x_odd
    result = np.empty_like(x)
    result[:, :, 0::2] = x_rot_even
    result[:, :, 1::2] = x_rot_odd
    return result


# ---------------------------------------------------------------------------
# Pre-LayerNorm
# ---------------------------------------------------------------------------

def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


# ---------------------------------------------------------------------------
# Deep Transformer forward pass (ALBERT-style weight sharing)
# ---------------------------------------------------------------------------

def forward_deep(
    ids: List[int],
    tok_emb: np.ndarray,
    pos_emb: np.ndarray,
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    Wo: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    Wlm: np.ndarray,
    n_heads: int,
    n_layers: int,
    max_seq: int,
) -> np.ndarray:
    """Forward pass through N shared layers. Returns logits of shape [T, V]."""
    T = min(len(ids), max_seq)
    ids_trunc = ids[-T:] if len(ids) > max_seq else ids
    d_model = tok_emb.shape[1]
    head_dim = d_model // n_heads

    x = tok_emb[ids_trunc] + pos_emb[:T]

    for _ in range(n_layers):
        # Pre-LayerNorm
        residual = x
        x_norm = _layer_norm(x)
        # Multi-head self-attention
        q = x_norm @ Wq  # [T, d_model]
        k = x_norm @ Wk
        v = x_norm @ Wv
        q = q.reshape(T, n_heads, head_dim).transpose(1, 0, 2)  # [H, T, hd]
        k = k.reshape(T, n_heads, head_dim).transpose(1, 0, 2)
        v = v.reshape(T, n_heads, head_dim).transpose(1, 0, 2)
        q = _rope_apply(q)
        k = _rope_apply(k)
        scores = q @ k.transpose(0, 2, 1) / math.sqrt(head_dim)  # [H, T, T]
        mask = np.triu(np.ones((T, T), dtype=np.float64), k=1) * -1e9
        scores = scores + mask
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        attn_out = attn @ v  # [H, T, hd]
        attn_out = attn_out.transpose(1, 0, 2).reshape(T, d_model)
        x = residual + attn_out @ Wo
        # FFN with GELU
        residual = x
        x_norm = _layer_norm(x)
        x = residual + _gelu(x_norm @ W1) @ W2

    x = _layer_norm(x)
    return x @ Wlm  # [T, V]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi.get(ch, 0) for ch in text]


def decode(ids: List[int], itos: Dict[int, str]) -> str:
    return "".join(itos.get(i, "") for i in ids)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str) -> dict:
    """Load model from .json metadata (+ versioned .npz) or legacy .npz path."""
    npz_path = Path(model_path)
    json_path = npz_path.with_suffix(".json")

    # Load metadata
    if json_path.is_file():
        meta = json.loads(json_path.read_text("utf-8"))
        # Support versioned .npz filenames
        npz_rel = meta.get("_npz", npz_path.name)
        npz_path = json_path.parent / npz_rel
    elif npz_path.is_file():
        meta = {}
    else:
        raise FileNotFoundError(f"No model found at {model_path}")

    # Load weights
    arrs = np.load(npz_path)
    vocab = meta.get("vocab", [chr(i) for i in range(256)])

    return {
        "tok_emb": arrs["tok_emb"],
        "pos_emb": arrs["pos_emb"],
        "Wq": arrs["Wq"],
        "Wk": arrs["Wk"],
        "Wv": arrs["Wv"],
        "Wo": arrs["Wo"],
        "W1": arrs["W1"],
        "W2": arrs["W2"],
        "Wlm": arrs["Wlm"],
        "vocab": vocab,
        "n_heads": int(meta.get("n_heads", 4)),
        "n_layers": int(meta.get("n_layers", 2)),
        "max_seq": int(meta.get("max_seq", 64)),
        "d_model": int(arrs["tok_emb"].shape[1]),
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    model: dict,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    seed: int = 42,
) -> str:
    import random as _random
    rng = _random.Random(seed)
    vocab = model["vocab"]
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    ids = encode(prompt, stoi)

    for _ in range(max_new_tokens):
        logits = forward_deep(
            ids, model["tok_emb"], model["pos_emb"],
            model["Wq"], model["Wk"], model["Wv"], model["Wo"],
            model["W1"], model["W2"], model["Wlm"],
            model["n_heads"], model["n_layers"], model["max_seq"],
        )
        last = logits[-1]
        if temperature > 0:
            last = last / max(temperature, 0.01)
            probs = np.exp(last - last.max())
            probs /= probs.sum()
            next_id = int(rng.choices(range(len(probs)), weights=probs.tolist(), k=1)[0])
        else:
            next_id = int(np.argmax(last))
        ids.append(next_id)

    return decode(ids, itos)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Lifers Edge Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to .npz weights file")
    parser.add_argument("--prompt", type=str, help="Single-shot inference prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[edge] Loading model from {args.model}...", file=sys.stderr, flush=True)
    model = load_model(args.model)
    info = f"D={model['d_model']} L={model['n_layers']} H={model['n_heads']} V={len(model['vocab'])}"
    print(f"[edge] Model: {info}", file=sys.stderr, flush=True)

    if args.prompt:
        result = generate(model, args.prompt, args.max_tokens, args.temperature, args.seed)
        print(result)
    else:
        print("[edge] Interactive mode — type a prompt:", file=sys.stderr)
        while True:
            try:
                prompt = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt or prompt.lower() in ("exit", "quit", "q"):
                break
            result = generate(model, prompt, args.max_tokens, args.temperature, args.seed)
            print(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
