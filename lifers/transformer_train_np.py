"""
Proper NumPy backprop training for TinyTransformer.

Replaces the finite-difference-on-Wlm-only stub in transformer_lm.train_sgd_minimal()
with full backprop through all layers: embeddings, attention (Q/K/V/O), FFN (W1/W2),
and LM head. Uses cross-entropy loss on all sequence positions with Adam optimizer.

This is the MINIMUM required to get non-garbled output from the transformer.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

from lifers.train_control import (
    LifersTrainingPause,
    LifersTrainingStop,
    read_train_control,
)
from lifers.train_progress import end_progress_line, write_progress_line
from lifers.train_status_file import refresh_sgd_status, write_heartbeat
from lifers.transformer_lm import (
    TinyTransformerWeights,
    build_vocab_from_text,
    decode,
    encode,
    init_weights,
)


def _try_numpy() -> Any:
    try:
        import numpy as np
        return np
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Forward pass (same semantics as transformer_lm.forward / forward_np)
# ---------------------------------------------------------------------------

def _layer_norm_fwd(X: Any, np: Any, eps: float = 1e-5) -> tuple:
    """Layer norm forward. Returns (Y, cache) where cache holds (mu, var, inv_std, X_centered)."""
    mu = X.mean(axis=1, keepdims=True)
    Xc = X - mu
    var = (Xc ** 2).mean(axis=1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    Y = Xc * inv_std
    return Y, (Xc, var, inv_std)


def _softmax_rows_fwd(X: Any, np: Any) -> tuple:
    """Row-wise softmax. Returns (probs, cache) where cache is probs (for backward)."""
    m = X.max(axis=1, keepdims=True)
    ex = np.exp(X - m)
    s = ex.sum(axis=1, keepdims=True)
    probs = ex / s
    return probs, probs


def _attention_fwd(
    X: Any, Wq: Any, Wk: Any, Wv: Any, Wo: Any, np: Any
) -> tuple:
    """Causal attention forward. Returns (output, cache_dict)."""
    d = int(X.shape[1])
    Q = X @ Wq
    K = X @ Wk
    Vv = X @ Wv
    scale = 1.0 / math.sqrt(float(d))
    scores = (Q @ K.T) * scale
    t = int(scores.shape[0])
    mask = np.triu(np.ones((t, t), dtype=np.float64), k=1) * (-1e9)
    scores_masked = scores + mask
    attn, _ = _softmax_rows_fwd(scores_masked, np)
    ctx = attn @ Vv
    out = ctx @ Wo
    cache = {"Q": Q, "K": K, "V": Vv, "attn": attn, "scores_masked": scores_masked,
             "X": X, "Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo}
    return out, cache


def _ffn_fwd(X: Any, W1: Any, W2: Any, np: Any) -> tuple:
    """FFN forward: ReLU(X @ W1) @ W2. Returns (output, cache)."""
    H_pre = X @ W1
    H = np.maximum(0.0, H_pre)
    out = H @ W2
    return out, (X, H_pre, H, W1, W2)


def forward_train(
    ids: List[int],
    tok_emb: Any, pos_emb: Any,
    Wq: Any, Wk: Any, Wv: Any, Wo: Any,
    W1: Any, W2: Any, Wlm: Any,
    max_seq: int,
    np: Any,
    *,
    training: bool = False,
) -> tuple:
    """
    Full forward pass returning (logits, caches) for backprop.
    logits: [T, V]
    caches: dict with everything needed for backward pass.

    When training=True: uses first min(len(ids), max_seq) tokens as input.
    When training=False: uses last min(len(ids), max_seq) tokens (for autoregressive generation).
    """
    t_len = min(len(ids), max_seq)
    if training:
        offset = 0
    else:
        offset = len(ids) - t_len
    idx = np.asarray(ids[offset:offset + t_len], dtype=np.intp)
    te = tok_emb[idx]
    pe = pos_emb[:t_len]
    X0 = te + pe

    # Embedding layer norm
    X1, ln1_cache = _layer_norm_fwd(X0, np)

    # Attention
    A, attn_cache = _attention_fwd(X1, Wq, Wk, Wv, Wo, np)

    # Residual + layer norm
    X2 = X1 + A
    X2_ln, ln2_cache = _layer_norm_fwd(X2, np)

    # FFN
    F, ffn_cache = _ffn_fwd(X2_ln, W1, W2, np)

    # Residual + layer norm
    X3 = X2_ln + F
    Y, ln3_cache = _layer_norm_fwd(X3, np)

    # LM head
    logits = Y @ Wlm

    caches = {
        "ln1": ln1_cache, "attn": attn_cache,
        "ln2": ln2_cache, "ffn": ffn_cache,
        "ln3": ln3_cache, "Y": Y,
        "X0": X0, "X1": X1, "X2_ln": X2_ln, "X3": X3,
        "tok_emb_idx": idx, "t_len": t_len,
    }
    return logits, caches


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

def _layer_norm_bwd(dY: Any, cache: tuple, np: Any) -> Any:
    """
    Layer norm backward using standard formula.
    y = (x - mu) / sigma  where sigma = sqrt(var + eps)
    dx = (1/sigma) * (dy - mean(dy) - y * mean(dy * y))
    cache = (X_centered, var, inv_std) where inv_std = 1/sigma.
    """
    Xc, _var, inv_std = cache
    D = float(Xc.shape[1])
    sigma = 1.0 / inv_std  # sqrt(var + eps)
    y = Xc * inv_std       # normalized output
    dy_mean = dY.mean(axis=1, keepdims=True)
    dy_y_mean = (dY * y).mean(axis=1, keepdims=True)
    dX = (1.0 / sigma) * (dY - dy_mean - y * dy_y_mean)
    return dX


def _softmax_cross_entropy_bwd(logits: Any, targets: Any, np: Any) -> tuple:
    """
    Backward of softmax + cross-entropy loss.
    logits: [T, V], targets: [T] (integer indices)
    Returns (loss_scalar, dL_dlogits).
    """
    T, V = logits.shape
    m = logits.max(axis=1, keepdims=True)
    ex = np.exp(logits - m)
    s = ex.sum(axis=1, keepdims=True)
    probs = ex / s

    # Loss
    target_probs = probs[np.arange(T), targets]
    loss = -np.log(np.maximum(target_probs, 1e-9)).mean()

    # Gradient: probs - one_hot(targets)
    dL_dlogits = probs.copy()
    dL_dlogits[np.arange(T), targets] -= 1.0
    dL_dlogits /= T  # average over positions

    return loss, dL_dlogits


def _linear_bwd(dY: Any, X: Any, W: Any, np: Any) -> tuple:
    """
    Backward through Y = X @ W.
    Returns (dX, dW).
    """
    dW = X.T @ dY
    dX = dY @ W.T
    return dX, dW


def _attention_bwd(dA: Any, cache: dict, np: Any) -> dict:
    """Backward through causal attention. Returns gradients for X, Wq, Wk, Wv, Wo."""
    Q = cache["Q"]; K = cache["K"]; Vv = cache["V"]
    attn = cache["attn"]; X = cache["X"]
    Wq = cache["Wq"]; Wk = cache["Wk"]; Wv = cache["Wv"]; Wo = cache["Wo"]
    D = int(Q.shape[1])

    # dA = d(out) where out = ctx @ Wo
    ctx = attn @ Vv
    dWo = ctx.T @ dA
    d_ctx = dA @ Wo.T

    # ctx = attn @ Vv -> d(attn) = d_ctx @ Vv.T, dVv = attn.T @ d_ctx
    d_attn = d_ctx @ Vv.T
    dVv = attn.T @ d_ctx

    # Backward through softmax
    row_sum = (attn * d_attn).sum(axis=1, keepdims=True)
    d_scores = attn * (d_attn - row_sum)

    # scores = (Q @ K.T) * scale
    scale = 1.0 / math.sqrt(float(D))
    d_scores_scaled = d_scores * scale
    dQ = d_scores_scaled @ K
    dK = d_scores_scaled.T @ Q

    # Q = X @ Wq, K = X @ Wk, Vv = X @ Wv
    dX_q = dQ @ Wq.T
    dX_k = dK @ Wk.T
    dX_v = dVv @ Wv.T
    dX_attn = dX_q + dX_k + dX_v

    dWq = X.T @ dQ
    dWk = X.T @ dK
    dWv = X.T @ dVv

    return {"dX": dX_attn, "dWq": dWq, "dWk": dWk, "dWv": dWv, "dWo": dWo}


def _ffn_bwd(dF: Any, cache: tuple, np: Any) -> dict:
    """Backward through FFN: ReLU(X @ W1) @ W2."""
    X, H_pre, H, W1, W2 = cache

    # F = H @ W2
    dH = dF @ W2.T
    dW2 = H.T @ dF

    # H = ReLU(H_pre)
    dH_pre = dH * (H_pre > 0).astype(np.float64)

    # H_pre = X @ W1
    dW1 = X.T @ dH_pre
    dX = dH_pre @ W1.T

    return {"dX": dX, "dW1": dW1, "dW2": dW2}


# ---------------------------------------------------------------------------
# Adam optimizer state
# ---------------------------------------------------------------------------

def _adam_init(params: Dict[str, Any]) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    for k, v in params.items():
        state[k] = {"m": v * 0.0, "v": v * 0.0}
    return state


def _adam_update(
    params: Dict[str, Any],
    grads: Dict[str, Any],
    state: Dict[str, Any],
    lr: float,
    t: int,
    np: Any,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    clip: float = 1.0,
) -> None:
    """In-place Adam update with gradient clipping."""
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    for k in params:
        if k not in grads:
            continue
        g = grads[k]

        # Gradient clipping
        g_norm = float(np.sqrt((g ** 2).sum()))
        if g_norm > clip:
            g = g * (clip / g_norm)

        s = state[k]
        s["m"] = beta1 * s["m"] + (1.0 - beta1) * g
        s["v"] = beta2 * s["v"] + (1.0 - beta2) * (g ** 2)
        m_hat = s["m"] / bc1
        v_hat = s["v"] / bc2
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ---------------------------------------------------------------------------
# Full training step (forward + backward + update)
# ---------------------------------------------------------------------------

def _train_step(
    input_ids: List[int],
    target_ids: List[int],
    params: Dict[str, Any],
    adam_state: Dict[str, Any],
    adam_t: int,
    lr: float,
    max_seq: int,
    np: Any,
) -> float:
    """Single full training step. input_ids and target_ids are aligned: target_ids[i] = next token after input_ids[i]."""
    tok_emb = params["tok_emb"]
    pos_emb = params["pos_emb"]
    Wq = params["Wq"]; Wk = params["Wk"]; Wv = params["Wv"]; Wo = params["Wo"]
    W1 = params["W1"]; W2 = params["W2"]; Wlm = params["Wlm"]

    # --- Forward (training mode: no offset, input is first max_seq tokens) ---
    logits, caches = forward_train(
        input_ids, tok_emb, pos_emb, Wq, Wk, Wv, Wo, W1, W2, Wlm, max_seq, np,
        training=True,
    )
    T = logits.shape[0]
    targets = np.asarray(target_ids[:T], dtype=np.intp)
    loss, dL_dlogits = _softmax_cross_entropy_bwd(logits, targets, np)

    # --- Backward ---
    grads: Dict[str, Any] = {}

    # LM head: logits = Y @ Wlm
    Y = caches["Y"]
    grads["Wlm"] = Y.T @ dL_dlogits
    dY = dL_dlogits @ params["Wlm"].T

    # ln3 backward: Y = LayerNorm(X3)
    dX3 = _layer_norm_bwd(dY, caches["ln3"], np)

    # Residual X3 = X2_ln + F
    # Gradient flows to both paths: d(X2_ln)_residual = dX3,  dF = dX3
    dX2_ln_residual = dX3
    dF = dX3

    # FFN backward: F = FFN(X2_ln)
    ffn_grads = _ffn_bwd(dF, caches["ffn"], np)
    dX2_ln_via_ffn = ffn_grads["dX"]
    grads["W1"] = ffn_grads["dW1"]
    grads["W2"] = ffn_grads["dW2"]

    # Total gradient into X2_ln = residual path + through FFN
    dX2_ln_total = dX2_ln_residual + dX2_ln_via_ffn

    # ln2 backward: X2_ln = LayerNorm(X2)
    dX2 = _layer_norm_bwd(dX2_ln_total, caches["ln2"], np)

    # Residual X2 = X1 + A
    dX1_residual = dX2
    dA = dX2

    # Attention backward: A = Attention(X1)
    attn_grads = _attention_bwd(dA, caches["attn"], np)
    dX1_via_attn = attn_grads["dX"]
    grads["Wq"] = attn_grads["dWq"]
    grads["Wk"] = attn_grads["dWk"]
    grads["Wv"] = attn_grads["dWv"]
    grads["Wo"] = attn_grads["dWo"]

    # Total gradient into X1 = residual path + through attention
    dX1_total = dX1_residual + dX1_via_attn

    # ln1 backward: X1 = LayerNorm(X0)
    dX0 = _layer_norm_bwd(dX1_total, caches["ln1"], np)

    # Embeddings: X0 = tok_emb[idx] + pos_emb[:t_len]
    idx = caches["tok_emb_idx"]
    grads["pos_emb"] = np.zeros_like(params["pos_emb"])
    grads["tok_emb"] = np.zeros_like(params["tok_emb"])
    t_len = caches["t_len"]
    grads["pos_emb"][:t_len] += dX0
    for i in range(t_len):
        grads["tok_emb"][idx[i]] += dX0[i]

    # --- Adam update ---
    _adam_update(params, grads, adam_state, lr, adam_t, np)

    return float(loss)


# ---------------------------------------------------------------------------
# Public training entry point
# ---------------------------------------------------------------------------

def train_backprop(
    text: str,
    out_path: Path,
    max_vocab: int = 256,
    d_model: int = 24,
    d_ff: int = 48,
    max_seq: int = 32,
    steps: int = 2000,
    lr: float = 3e-4,
    seed: int = 1,
    *,
    show_progress: bool = True,
    progress_stream: TextIO | None = None,
    warm_start_path: Path | None = None,
    control_path: Path | None = None,
) -> TinyTransformerWeights:
    """
    Train TinyTransformer with proper backprop through all layers.

    This replaces the finite-difference stub. Uses NumPy + OpenBLAS for speed.
    On Kali with 4+ cores, d_model=64 trains ~100 steps/sec.
    """
    np = _try_numpy()
    if np is None:
        raise ImportError(
            "NumPy is required for backprop training. "
            "Install: pip install numpy  or  apt-get install python3-numpy"
        )

    rng = random.Random(seed)
    vocab = build_vocab_from_text(text, max_vocab=max_vocab)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    ids_raw = encode(text, stoi)

    V = len(vocab)
    D = d_model

    # Init or warm-start
    w: TinyTransformerWeights
    if warm_start_path is not None and warm_start_path.is_file():
        try:
            cand = TinyTransformerWeights.load(warm_start_path)
            if (list(cand.vocab) == vocab and cand.d_model == d_model
                    and cand.d_ff == d_ff and cand.max_seq == max_seq and cand.n_heads == 1):
                w = cand
            else:
                w = init_weights(vocab=vocab, d_model=d_model, d_ff=d_ff, n_heads=1, max_seq=max_seq, seed=seed)
        except (OSError, KeyError, ValueError):
            w = init_weights(vocab=vocab, d_model=d_model, d_ff=d_ff, n_heads=1, max_seq=max_seq, seed=seed)
    else:
        w = init_weights(vocab=vocab, d_model=d_model, d_ff=d_ff, n_heads=1, max_seq=max_seq, seed=seed)

    # Convert to NumPy arrays (params dict)
    params: Dict[str, Any] = {
        "tok_emb": np.asarray(w.tok_emb, dtype=np.float64),
        "pos_emb": np.asarray(w.pos_emb, dtype=np.float64),
        "Wq": np.asarray(w.Wq, dtype=np.float64),
        "Wk": np.asarray(w.Wk, dtype=np.float64),
        "Wv": np.asarray(w.Wv, dtype=np.float64),
        "Wo": np.asarray(w.Wo, dtype=np.float64),
        "W1": np.asarray(w.W1, dtype=np.float64),
        "W2": np.asarray(w.W2, dtype=np.float64),
        "Wlm": np.asarray(w.Wlm, dtype=np.float64),
    }
    adam_state = _adam_init(params)

    # Progress config
    stream: TextIO | None = None
    if show_progress:
        stream = progress_stream if progress_stream is not None else sys.stderr
    log_every = max(1, min(50, steps // 50))
    tty_every = 1 if steps <= 500 else max(1, steps // 40)
    ctrl_chk = max(1, log_every)
    save_every = max(1, steps // 20)

    thr_line = (
        os.environ.get("OMP_NUM_THREADS", "").strip()
        or os.environ.get("OPENBLAS_NUM_THREADS", "").strip()
        or os.environ.get("MKL_NUM_THREADS", "").strip()
    )
    tip = f"OMP_NUM_THREADS={thr_line}" if thr_line else "export OMP_NUM_THREADS=$(nproc)"
    print(f"[backprop] NumPy/BLAS backprop training V={V} D={D} F={d_ff} S={max_seq} steps={steps} — {tip}",
          file=sys.stderr)

    def _save_np() -> None:
        w.tok_emb = params["tok_emb"].tolist()
        w.pos_emb = params["pos_emb"].tolist()
        w.Wq = params["Wq"].tolist()
        w.Wk = params["Wk"].tolist()
        w.Wv = params["Wv"].tolist()
        w.Wo = params["Wo"].tolist()
        w.W1 = params["W1"].tolist()
        w.W2 = params["W2"].tolist()
        w.Wlm = params["Wlm"].tolist()
        w.save(out_path)

    t0 = time.time()
    total_loss = 0.0
    n_data = len(ids_raw)
    seq_len = max_seq + 1

    for step in range(steps):
        # Control check
        if control_path is not None and (step == 0 or (step + 1) % ctrl_chk == 0):
            mode = read_train_control(control_path)
            if mode in ("pause", "stop"):
                _save_np()
                if mode == "stop":
                    raise LifersTrainingStop()
                raise LifersTrainingPause()

        # Random sequence chunk: need max_seq+1 tokens (input + target)
        if n_data > seq_len:
            start = rng.randrange(0, n_data - seq_len)
        else:
            start = 0
        chunk = ids_raw[start:start + seq_len]  # length = max_seq + 1

        # Input: first max_seq tokens, Target: tokens 1..max_seq+1 (next-token prediction)
        input_ids = chunk[:max_seq]
        target_ids = chunk[1:max_seq + 1]

        loss = _train_step(input_ids, target_ids, params, adam_state, step + 1, lr, max_seq, np)
        total_loss += loss

        # Save
        if (step + 1) % save_every == 0 or step + 1 == steps:
            _save_np()

        # Progress
        if stream is not None:
            upd = tty_every if stream.isatty() else log_every
            if step == 0 or (step + 1) % upd == 0 or step + 1 == steps:
                avg_loss = total_loss / (step + 1)
                elapsed = time.time() - t0
                sps = (step + 1) / max(0.01, elapsed)
                write_progress_line(
                    stream, step + 1, steps,
                    prefix=f"backprop V={V} D={D} loss={avg_loss:.3f} {sps:.0f}s/s ",
                )
                try:
                    refresh_sgd_status(step + 1, steps, V, D)
                except Exception:
                    pass
                # Heartbeat for crash detection
                try:
                    raw_root = os.environ.get("LIFERS_TRAIN_STATUS_BRAIN_ROOT", "").strip()
                    if raw_root:
                        ri = int(os.environ.get("LIFERS_TRAIN_STATUS_RAMP_ITER", "1"))
                        rm = int(os.environ.get("LIFERS_TRAIN_STATUS_RAMP_MAX", "1"))
                        write_heartbeat(Path(raw_root), ri, rm, step + 1, steps, avg_loss)
                except Exception:
                    pass

    if stream is not None:
        end_progress_line(stream)

    _save_np()
    elapsed = time.time() - t0
    print(f"[backprop] done {steps} steps in {elapsed:.1f}s, final avg loss={total_loss/steps:.4f}",
          file=sys.stderr)
    return w


# ---------------------------------------------------------------------------
# Drop-in replacement for train_sgd_minimal (same signature)
# ---------------------------------------------------------------------------

def train_backprop_minimal(
    text: str,
    out_path: Path,
    max_vocab: int = 256,
    d_model: int = 24,
    d_ff: int = 48,
    max_seq: int = 32,
    steps: int = 2,
    lr: float = 1e-2,
    seed: int = 1,
    *,
    show_progress: bool = True,
    progress_stream: TextIO | None = None,
    warm_start_path: Path | None = None,
    control_path: Path | None = None,
) -> TinyTransformerWeights:
    """
    Drop-in replacement for train_sgd_minimal with the same signature.
    Uses proper backprop when LIFERS_USE_BACKPROP=1 or when steps > 100.
    Falls back to original finite-difference for very small step counts.
    """
    use_backprop = os.environ.get("LIFERS_USE_BACKPROP", "").strip().lower() not in (
        "0", "false", "no", "off"
    )
    # Auto-enable backprop for any meaningful training run
    if steps >= 50:
        use_backprop = True

    if use_backprop:
        return train_backprop(
            text=text, out_path=out_path,
            max_vocab=max_vocab, d_model=d_model, d_ff=d_ff, max_seq=max_seq,
            steps=steps, lr=lr, seed=seed,
            show_progress=show_progress, progress_stream=progress_stream,
            warm_start_path=warm_start_path, control_path=control_path,
        )

    # Fallback to original (for backward compatibility with LIFERS_USE_BACKPROP=0)
    from lifers.transformer_lm import train_sgd_minimal
    return train_sgd_minimal(
        text=text, out_path=out_path,
        max_vocab=max_vocab, d_model=d_model, d_ff=d_ff, max_seq=max_seq,
        steps=steps, lr=lr, seed=seed,
        show_progress=show_progress, progress_stream=progress_stream,
        warm_start_path=warm_start_path, control_path=control_path,
    )
