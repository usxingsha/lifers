"""
Deep multi-layer transformer with multi-head attention, weight sharing (ALBERT-style),
GELU activation, and pre-LayerNorm. Compatible with existing escalate/training pipeline.

Key improvements over TinyTransformer:
  - n_layers: stack depth (default 4, max 10 with weight sharing)
  - n_heads:  multi-head attention (default 8)
  - GELU:     smoother activation than ReLU
  - Weight sharing: all layers reuse same params (dramatic memory savings)
  - Gradient checkpoint stubs for training loop
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as _numpy


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def _gelu_np(x: Any, np: Any) -> Any:
    """GELU: x * 0.5 * (1 + erf(x/sqrt(2))) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi)*(x+0.044715*x^3)))"""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embedding)
# ---------------------------------------------------------------------------

_rope_cache: Dict[tuple, tuple] = {}

def _rope_apply(x: Any, np: Any) -> Any:
    """Apply RoPE to input x. x shape: [n_heads, T, head_dim]."""
    _n_heads, T, head_dim = x.shape
    d2 = head_dim // 2
    if d2 == 0:
        return x
    cache_key = (T, head_dim)
    if cache_key in _rope_cache:
        cos, sin = _rope_cache[cache_key]
    else:
        pos = np.arange(T, dtype=np.float64).reshape(T, 1)
        dim = np.arange(d2, dtype=np.float64).reshape(1, d2)
        theta = 1.0 / (10000.0 ** (2.0 * dim / head_dim))
        freqs = pos @ theta
        cos = np.cos(freqs).reshape(1, T, d2)
        sin = np.sin(freqs).reshape(1, T, d2)
        _rope_cache[cache_key] = (cos, sin)
    x_even = x[:, :, 0::2]
    x_odd = x[:, :, 1::2]
    x_rot_even = cos * x_even - sin * x_odd
    x_rot_odd = sin * x_even + cos * x_odd
    result = np.empty_like(x)
    result[:, :, 0::2] = x_rot_even
    result[:, :, 1::2] = x_rot_odd
    return result


# ---------------------------------------------------------------------------
# Model weights
# ---------------------------------------------------------------------------

@dataclass
class DeepTransformerWeights:
    vocab: List[str]
    d_model: int
    d_ff: int
    n_heads: int
    n_layers: int
    max_seq: int
    # Embeddings
    tok_emb: List[List[float]]
    pos_emb: List[List[float]]
    # Shared attention weights (ALBERT-style: same params for all layers)
    Wq: List[List[float]]   # [d_model, d_model]
    Wk: List[List[float]]   # [d_model, d_model]
    Wv: List[List[float]]   # [d_model, d_model]
    Wo: List[List[float]]   # [d_model, d_model]
    # Shared FFN weights
    W1: List[List[float]]   # [d_model, d_ff]
    W2: List[List[float]]   # [d_ff, d_model]
    # LM head
    Wlm: List[List[float]]  # [d_model, vocab]

    _weight_keys = ("tok_emb", "pos_emb", "Wq", "Wk", "Wv", "Wo", "W1", "W2", "Wlm")

    def _meta_dict(self) -> dict:
        return {
            "vocab": self.vocab, "d_model": self.d_model, "d_ff": self.d_ff,
            "n_heads": self.n_heads, "n_layers": self.n_layers, "max_seq": self.max_seq,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Versioned .npz: never overwrite, so .json always points to a valid file.
        # Old versions are cleaned up after .json is safely updated.
        base = path.stem  # e.g. "lifers_deep_transformer"
        stamp = str(int(time.time() * 1_000_000))
        npz_name = f"{base}.v{stamp}.npz"
        npz_path = path.parent / npz_name
        arrays = {}
        for k in self._weight_keys:
            v = getattr(self, k)
            arrays[k] = _numpy.array(v, dtype=_numpy.float32)
        _numpy.savez_compressed(npz_path, **arrays)
        # Atomically update JSON to point to the new .npz
        meta = self._meta_dict()
        meta["_npz"] = npz_name
        data = json.dumps(meta, ensure_ascii=False)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.unlink(missing_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(5):
            try:
                tmp.replace(path)
                break
            except OSError:
                if attempt == 4:
                    raise
                path.unlink(missing_ok=True)
                time.sleep(0.5 * (attempt + 1))
        # Clean old .npz files: keep only the one referenced by current JSON
        for old in path.parent.glob(f"{base}.v*.npz"):
            if old.name != npz_name:
                try:
                    old.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def load(path: Path) -> "DeepTransformerWeights":
        obj = json.loads(path.read_text(encoding="utf-8"))
        npz_rel = obj.pop("_npz", None)
        # Try binary .npz first (fast path)
        if npz_rel:
            npz_path = path.parent / npz_rel
            if npz_path.is_file():
                try:
                    arrs = _numpy.load(npz_path)
                except Exception:
                    # Corrupt .npz — clean it so we don't keep broken file
                    try:
                        npz_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise
                for k in DeepTransformerWeights._weight_keys:
                    obj[k] = arrs[k].tolist()
                # Reconstruct: .npz may have been saved fp32; original was fp64 list-of-list
                return DeepTransformerWeights(**obj)
        # Fallback: weights are stored inline in JSON (legacy format)
        return DeepTransformerWeights(**obj)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_deep_weights(
    vocab: List[str],
    d_model: int = 256,
    d_ff: int = 1024,
    n_heads: int = 8,
    n_layers: int = 4,
    max_seq: int = 64,
    seed: int = 1,
) -> DeepTransformerWeights:
    rng = _numpy.random.RandomState(seed)
    V = len(vocab)
    D = d_model
    F = d_ff

    def _rand(m: int, n: int):
        scale = math.sqrt(2.0 / max(m, 1))
        return (rng.uniform(-scale, scale, (m, n))).tolist()

    return DeepTransformerWeights(
        vocab=list(vocab), d_model=D, d_ff=F, n_heads=n_heads,
        n_layers=n_layers, max_seq=max_seq,
        tok_emb=_rand(V, D),
        pos_emb=_rand(max_seq, D),
        Wq=_rand(D, D), Wk=_rand(D, D), Wv=_rand(D, D), Wo=_rand(D, D),
        W1=_rand(D, F), W2=_rand(F, D),
        Wlm=_rand(D, V),
    )


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def _layer_norm_fwd(X: Any, np: Any, eps: float = 1e-5) -> tuple:
    mu = X.mean(axis=-1, keepdims=True)
    Xc = X - mu
    var = (Xc ** 2).mean(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    return Xc * inv_std, (Xc, var, inv_std)


def _softmax_rows(X: Any, np: Any) -> Any:
    m = X.max(axis=-1, keepdims=True)
    ex = np.exp(X - m)
    return ex / ex.sum(axis=-1, keepdims=True)


_causal_mask_cache: Dict[int, Any] = {}

def _causal_mask(T: int, np: Any) -> Any:
    if T not in _causal_mask_cache:
        _causal_mask_cache[T] = np.triu(np.ones((T, T), dtype=np.float64), k=1) * (-1e9)
    return _causal_mask_cache[T]


def _attention_block(X_norm: Any, Wq: Any, Wk: Any, Wv: Any, Wo: Any,
                     n_heads: int, np: Any) -> Tuple[Any, dict]:
    """Multi-head causal attention. X_norm: [T, D]."""
    T, D = X_norm.shape
    head_dim = D // n_heads

    Q = X_norm @ Wq   # [T, D]
    K = X_norm @ Wk
    V = X_norm @ Wv

    # Reshape: [T, D] -> [T, n_heads, head_dim] -> [n_heads, T, head_dim]
    Q = Q.reshape(T, n_heads, head_dim).transpose(1, 0, 2)
    K = K.reshape(T, n_heads, head_dim).transpose(1, 0, 2)
    V = V.reshape(T, n_heads, head_dim).transpose(1, 0, 2)

    # RoPE on Q and K
    Q = _rope_apply(Q, np)
    K = _rope_apply(K, np)

    # Scaled dot-product attention
    scale = 1.0 / math.sqrt(float(head_dim))
    scores = (Q @ K.transpose(0, 2, 1)) * scale  # [n_heads, T, T]

    # Causal mask (pre-allocated)
    scores = scores + _causal_mask(T, np)

    attn = _softmax_rows(scores, np)
    ctx = attn @ V  # [n_heads, T, head_dim]

    # Merge heads: [n_heads, T, head_dim] -> [T, n_heads, head_dim] -> [T, D]
    ctx = ctx.transpose(1, 0, 2).reshape(T, D)

    out = ctx @ Wo  # [T, D]

    cache = {"Q": Q, "K": K, "V": V, "attn": attn, "scores": scores,
             "X": X_norm, "Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo,
             "n_heads": n_heads, "head_dim": head_dim}
    return out, cache


def _ffn_block(X_norm: Any, W1: Any, W2: Any, np: Any) -> Tuple[Any, tuple]:
    """FFN: GELU(X @ W1) @ W2."""
    H_pre = X_norm @ W1
    H = _gelu_np(H_pre, np)
    out = H @ W2
    return out, (X_norm, H_pre, H, W1, W2)


def _transformer_block(X: Any, Wq: Any, Wk: Any, Wv: Any, Wo: Any,
                       W1: Any, W2: Any, n_heads: int, np: Any) -> Tuple[Any, dict]:
    """Single transformer block: LN -> Attn -> Residual -> LN -> FFN -> Residual."""
    # Pre-LN + Attention
    X_norm1, ln1_cache = _layer_norm_fwd(X, np)
    attn_out, attn_cache = _attention_block(X_norm1, Wq, Wk, Wv, Wo, n_heads, np)
    X = X + attn_out  # Residual

    # Pre-LN + FFN
    X_norm2, ln2_cache = _layer_norm_fwd(X, np)
    ffn_out, ffn_cache = _ffn_block(X_norm2, W1, W2, np)
    X = X + ffn_out  # Residual

    cache = {"ln1": ln1_cache, "attn": attn_cache,
             "ln2": ln2_cache, "ffn": ffn_cache,
             "X_in": X - ffn_out - attn_out,  # original X before this block
             "attn_out": attn_out}  # cached to avoid recomputation in bwd
    return X, cache


def forward_deep(
    ids: List[int],
    tok_emb: Any, pos_emb: Any,
    Wq: Any, Wk: Any, Wv: Any, Wo: Any,
    W1: Any, W2: Any, Wlm: Any,
    n_heads: int, n_layers: int, max_seq: int,
    np: Any,
    *,
    training: bool = False,
) -> tuple:
    """
    Full forward pass through N-layer transformer.
    Returns (logits, caches) where caches is list of per-layer activation caches.
    """
    t_len = min(len(ids), max_seq)
    if training:
        offset = 0
    else:
        offset = len(ids) - t_len if len(ids) >= t_len else 0
    idx = np.asarray(ids[offset:offset + t_len], dtype=np.intp)
    te = tok_emb[idx]
    pe = pos_emb[:t_len]
    X = te + pe

    layer_caches: List[dict] = []
    for _ in range(n_layers):
        X, cache = _transformer_block(X, Wq, Wk, Wv, Wo, W1, W2, n_heads, np)
        layer_caches.append(cache)

    # Final LayerNorm + LM head
    Y, final_ln_cache = _layer_norm_fwd(X, np)
    logits = Y @ Wlm

    caches = {
        "layer_caches": layer_caches,
        "final_ln": final_ln_cache,
        "Y": Y,
        "tok_emb_idx": idx,
        "t_len": t_len,
        "X_input": te + pe,
    }
    return logits, caches


# ---------------------------------------------------------------------------
# Text encoding / decoding helpers (character-level)
# ---------------------------------------------------------------------------

def build_vocab_from_text(text: str, max_vocab: int = 256) -> List[str]:
    from collections import Counter
    freq = Counter(text)
    # Reserve slot for \n if needed
    effective_max = max_vocab - 1 if "\n" not in set(ch for ch, _ in freq.most_common(max_vocab)) else max_vocab
    items = freq.most_common(effective_max)
    vocab = [ch for ch, _ in items]
    if "\n" not in vocab:
        vocab.append("\n")
    return vocab


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi.get(ch, 0) for ch in text]


def decode(ids: List[int], itos: Dict[int, str]) -> str:
    return "".join(itos.get(i, "?") for i in ids)


def _topk_sample_token_id(last_logits, rng, temperature: float, top_k: int) -> int:
    """温度 + top-k 采样一步，消除低概率噪声 token。"""
    if hasattr(last_logits, "tolist"):
        scaled = [x / max(1e-6, temperature) for x in last_logits.tolist()]
    else:
        scaled = [x / max(1e-6, temperature) for x in last_logits]
    idxs = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)[:max(1, min(top_k, len(scaled)))]
    vals = [scaled[i] for i in idxs]
    import math
    max_val = max(vals)
    exps = [math.exp(v - max_val) for v in vals]
    total = sum(exps)
    probs = [e / total for e in exps]
    r = rng.random()
    acc = 0.0
    pick = idxs[-1]
    for i, p in zip(idxs, probs):
        acc += p
        if acc >= r:
            pick = i
            break
    return pick


def generate_text(
    w: DeepTransformerWeights,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 80,
    repetition_penalty: float = 1.05,
    seed: int = 42,
) -> str:
    """Autoregressive generation using NumPy forward pass.

    top_k: 采样截断数 (默认 80)，过滤低概率噪声 token。
    repetition_penalty: >1.0 惩罚已生成 token 的重复 (默认 1.05)。
    """
    import random as _random
    try:
        import numpy as _np
    except ImportError:
        return prompt + " [numpy required]"

    rng = _random.Random(seed)
    stoi = {ch: i for i, ch in enumerate(w.vocab)}
    itos = {i: ch for i, ch in enumerate(w.vocab)}
    ids = encode(prompt, stoi)
    if not ids:
        ids = [0]  # bootstrap with first vocab token on empty prompt

    te = _np.asarray(w.tok_emb, dtype=_np.float64)
    pe = _np.asarray(w.pos_emb, dtype=_np.float64)
    Wq = _np.asarray(w.Wq, dtype=_np.float64)
    Wk = _np.asarray(w.Wk, dtype=_np.float64)
    Wv = _np.asarray(w.Wv, dtype=_np.float64)
    Wo = _np.asarray(w.Wo, dtype=_np.float64)
    W1 = _np.asarray(w.W1, dtype=_np.float64)
    W2 = _np.asarray(w.W2, dtype=_np.float64)
    Wlm = _np.asarray(w.Wlm, dtype=_np.float64)

    # 统计已生成 token 频率，用于重复惩罚
    token_counts: dict = {}

    for _ in range(max_new_tokens):
        logits, _ = forward_deep(
            ids, te, pe, Wq, Wk, Wv, Wo, W1, W2, Wlm,
            w.n_heads, w.n_layers, w.max_seq, _np, training=False,
        )
        last_logits = logits[-1].copy()

        # 重复惩罚：降低已生成 token 的 logit
        if repetition_penalty > 1.0 and token_counts:
            for tid, count in token_counts.items():
                if count > 1:
                    penalty = repetition_penalty ** count
                    last_logits[tid] /= penalty

        if temperature > 0:
            next_id = _topk_sample_token_id(last_logits, rng, temperature, top_k)
        else:
            next_id = int(_np.argmax(last_logits))
        ids.append(next_id)
        token_counts[next_id] = token_counts.get(next_id, 0) + 1

    return decode(ids, itos)
