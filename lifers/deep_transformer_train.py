"""
Full backprop training for DeepTransformer with N layers, multi-head attention,
full activation caching (for backward pass), and AdamW optimizer.

Memory: stores per-layer activation caches for backward pass.
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
from lifers.deep_transformer import (
    DeepTransformerWeights,
    build_vocab_from_text,
    encode,
    init_deep_weights,
    _gelu_np,
    _layer_norm_fwd,
    _softmax_rows,
    _attention_block,
    _ffn_block,
    _transformer_block,
    forward_deep,
    _rope_cache,
)


def _try_numpy() -> Any:
    """Legacy — use get_compute_backend() instead."""
    from lifers.core.compute_backend import get_compute_backend
    np_mod, _, _ = get_compute_backend()
    return np_mod


# ======================================================================
# Backward pass primitives
# ======================================================================

def _layer_norm_bwd(dY: Any, cache: tuple, np: Any) -> Any:
    """Standard LN backward."""
    Xc, _var, inv_std = cache
    sigma = 1.0 / inv_std
    y = Xc * inv_std
    D = float(Xc.shape[-1])
    dy_mean = dY.mean(axis=-1, keepdims=True)
    dy_y_mean = (dY * y).mean(axis=-1, keepdims=True)
    return (1.0 / sigma) * (dY - dy_mean - y * dy_y_mean)


def _gelu_bwd(dY: Any, H_pre: Any, np: Any) -> Any:
    """GELU backward: d/dx gelu(x) * dY."""
    # Approximate derivative of GELU
    cdf = 0.5 * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (H_pre + 0.044715 * H_pre**3)))
    pdf_factor = math.sqrt(2.0 / math.pi)
    pdf_inner = 1.0 + 0.044715 * 3 * H_pre**2
    pdf = pdf_factor * np.exp(-0.5 * H_pre**2) * pdf_inner
    grad = cdf + H_pre * pdf * 0.5  # simplified GELU derivative
    return dY * grad


def _softmax_cross_entropy_bwd(logits: Any, targets: Any, np: Any) -> tuple:
    T, V = logits.shape
    m = logits.max(axis=-1, keepdims=True)
    ex = np.exp(logits - m)
    probs = ex / ex.sum(axis=-1, keepdims=True)
    loss = -np.log(np.maximum(probs[np.arange(T), targets], 1e-9)).mean()
    dL = probs.copy()
    dL[np.arange(T), targets] -= 1.0
    dL /= T
    return float(loss), dL


def _rope_bwd(dY_rot: Any, cos: Any, sin: Any, np: Any) -> Any:
    """Backward through RoPE rotation."""
    dY_even = dY_rot[:, :, 0::2]
    dY_odd = dY_rot[:, :, 1::2]
    # Forward: out_even = cos*x_even - sin*x_odd, out_odd = sin*x_even + cos*x_odd
    # Backward: dx_even = cos*dout_even + sin*dout_odd, dx_odd = -sin*dout_even + cos*dout_odd
    dx_even = cos * dY_even + sin * dY_odd
    dx_odd = -sin * dY_even + cos * dY_odd
    result = np.empty_like(dY_rot)
    result[:, :, 0::2] = dx_even
    result[:, :, 1::2] = dx_odd
    return result


def _attention_bwd(dOut: Any, cache: dict, np: Any) -> dict:
    """Backward through multi-head causal attention with RoPE."""
    Q = cache["Q"]; K = cache["K"]; V = cache["V"]
    attn = cache["attn"]; X = cache["X"]
    Wq = cache["Wq"]; Wk = cache["Wk"]; Wv = cache["Wv"]; Wo = cache["Wo"]
    n_heads = cache["n_heads"]; head_dim = cache["head_dim"]
    T, D = X.shape

    # dOut: [T, D], reverse output projection
    # Wo backward: out_final = ctx_merged @ Wo -> dWo = ctx_merged.T @ dOut
    ctx_h = attn @ V  # [n_heads, T, head_dim]
    ctx_merged = ctx_h.transpose(1, 0, 2).reshape(T, D)
    dWo = ctx_merged.T @ dOut
    d_ctx_merged = dOut @ Wo.T  # [T, D]
    d_ctx = d_ctx_merged.reshape(T, n_heads, head_dim).transpose(1, 0, 2)  # [n_heads, T, head_dim]

    # ctx = attn @ V -> dV = attn.T @ d_ctx, d_attn = d_ctx @ V.T
    dV = attn.transpose(0, 2, 1) @ d_ctx
    d_attn = d_ctx @ V.transpose(0, 2, 1)

    # Softmax backward: attn = softmax(scores)
    row_sum = (attn * d_attn).sum(axis=-1, keepdims=True)
    d_scores = attn * (d_attn - row_sum)

    # Scale backward
    scale = 1.0 / math.sqrt(float(head_dim))
    d_scores = d_scores * scale

    # QK backward: scores = Q @ K.T * scale
    dQ_raw = d_scores @ K    # [n_heads, T, head_dim]
    dK_raw = d_scores.transpose(0, 2, 1) @ Q

    # RoPE backward on Q and K (reuse cache from forward pass)
    d2 = head_dim // 2
    cache_key = (T, head_dim)
    if cache_key in _rope_cache:
        cos, sin = _rope_cache[cache_key]
    else:
        float_dtype = np.float32 if device == "cuda" else np.float64
        pos = np.arange(T, dtype=float_dtype).reshape(T, 1)
        dim_arr = np.arange(d2, dtype=float_dtype).reshape(1, d2)
        theta = 1.0 / (10000.0 ** (2.0 * dim_arr / head_dim))
        freqs = pos @ theta
        cos = np.cos(freqs).reshape(1, T, d2)
        sin = np.sin(freqs).reshape(1, T, d2)
        _rope_cache[cache_key] = (cos, sin)

    dQ = _rope_bwd(dQ_raw, cos, sin, np)
    dK = _rope_bwd(dK_raw, cos, sin, np)

    # Input projection backward: Q = X @ Wq, K = X @ Wk, V = X @ Wv
    Q_merged = Q.transpose(1, 0, 2).reshape(T, D)
    K_merged = K.transpose(1, 0, 2).reshape(T, D)
    V_merged = V.transpose(1, 0, 2).reshape(T, D)
    dQ_merged = dQ.transpose(1, 0, 2).reshape(T, D)
    dK_merged = dK.transpose(1, 0, 2).reshape(T, D)
    dV_merged = dV.transpose(1, 0, 2).reshape(T, D)

    dWq = X.T @ dQ_merged
    dWk = X.T @ dK_merged
    dWv = X.T @ dV_merged

    dX = (dQ_merged @ Wq.T) + (dK_merged @ Wk.T) + (dV_merged @ Wv.T)

    return {"dX": dX, "dWq": dWq, "dWk": dWk, "dWv": dWv, "dWo": dWo}


def _ffn_bwd(dF: Any, cache: tuple, np: Any) -> dict:
    X, H_pre, H, W1, W2 = cache

    # dF = d(out) where out = H @ W2
    dW2 = H.T @ dF
    dH = dF @ W2.T

    # GELU backward
    dH_pre = _gelu_bwd(dH, H_pre, np)

    # H_pre = X @ W1
    dW1 = X.T @ dH_pre
    dX = dH_pre @ W1.T

    return {"dX": dX, "dW1": dW1, "dW2": dW2}


def _transformer_block_bwd(dX_out: Any, block_cache: dict, np: Any) -> Tuple[Any, dict]:
    """Backward through one transformer block. Returns (dX_in, weight_grads)."""
    # dX_out = gradient into output of this block (after residual FFN)
    # Block: X_mid = X_in + Attn(LN1(X_in)), X_out = X_mid + FFN(LN2(X_mid))

    # Reverse FFN residual + LN + FFN
    X_mid = block_cache["X_in"]  # This is actually X_mid after attention
    # Actually, let me re-derive: block_cache["X_in"] = X before LN1
    # After attention: X_after_attn = X_in + Attn(LN1(X_in))
    # After FFN:     X_out = X_after_attn + FFN(LN2(X_after_attn))
    # So dX_out flows to both: dX_after_attn_residual AND dFFN

    # Get the X_after_attn by reconstructing
    # We need to re-derive X_after_attn. Let me compute it from cache.
    # X_in = block_cache["X_in"] = X before this block
    X_in = block_cache["X_in"]

    # Use cached attn_out to avoid recomputation
    attn_out = block_cache["attn_out"]
    X_after_attn = X_in + attn_out
    attn_cache = block_cache["attn"]

    # dX_out -> splits into dX_after_attn (residual) and dFFN
    d_ffn_out = dX_out  # into FFN path
    d_residual_ffn = dX_out  # skip connection

    # FFN backward
    ffn_grads = _ffn_bwd(d_ffn_out, block_cache["ffn"], np)
    dX_after_attn_ffn = ffn_grads["dX"]

    # LN2 backward
    dX_after_attn_ln2 = _layer_norm_bwd(dX_after_attn_ffn, block_cache["ln2"], np)

    # Total into X_after_attn
    dX_after_attn = d_residual_ffn + dX_after_attn_ln2

    # Split at first residual: dX_in (skip) and d_attn_block
    dX_in_skip = dX_after_attn
    d_attn_block = dX_after_attn

    # Attention backward
    attn_grads = _attention_bwd(d_attn_block, attn_cache, np)
    dX_attn = attn_grads["dX"]

    # LN1 backward
    dX_ln1 = _layer_norm_bwd(dX_attn, block_cache["ln1"], np)

    # Total into X_in
    dX_in = dX_in_skip + dX_ln1

    grads = {
        "Wq": attn_grads["dWq"], "Wk": attn_grads["dWk"],
        "Wv": attn_grads["dWv"], "Wo": attn_grads["dWo"],
        "W1": ffn_grads["dW1"], "W2": ffn_grads["dW2"],
    }
    return dX_in, grads


# ======================================================================
# AdamW optimizer
# ======================================================================

def _adamw_init(params: Dict[str, Any]) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    for k, v in params.items():
        state[k] = {"m": v * 0.0, "v": v * 0.0}
    return state


def _adamw_update(
    params: Dict[str, Any],
    grads: Dict[str, Any],
    state: Dict[str, Any],
    lr: float,
    t: int,
    np: Any,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    clip: float = 1.0,
    weight_decay: float = 0.01,
) -> None:
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    for k in params:
        if k not in grads:
            continue
        g = grads[k]
        g_norm = float(np.sqrt((g ** 2).sum()))
        if g_norm > clip:
            g = g * (clip / max(g_norm, 1e-12))
        s = state[k]
        # AdamW: decay then Adam
        params[k] *= (1.0 - lr * weight_decay)
        s["m"] = beta1 * s["m"] + (1.0 - beta1) * g
        s["v"] = beta2 * s["v"] + (1.0 - beta2) * (g ** 2)
        m_hat = s["m"] / bc1
        v_hat = s["v"] / bc2
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ======================================================================
# Single training step
# ======================================================================

def _train_step_deep(
    input_ids: List[int],
    target_ids: List[int],
    params: Dict[str, Any],
    adam_state: Dict[str, Any],
    adam_t: int,
    lr: float,
    n_heads: int,
    n_layers: int,
    max_seq: int,
    np: Any,
    *,
    compute_dtype: Any = None,  # FP16 compute dtype for mixed precision
    loss_scale: float = 1.0,     # gradient scaling for FP16
) -> tuple:
    """Single training step: forward (可选FP16) → backward → AdamW update (FP32 master).
    Returns (loss, overflow) where overflow=True means NaN/Inf gradients detected."""

    master_dtype = np.float32
    if compute_dtype is not None and compute_dtype != master_dtype:
        # FP16 mixed precision: cast weights to compute dtype for forward/backward
        fp_params = {k: v.astype(compute_dtype) for k, v in params.items()}
    else:
        fp_params = params

    # --- Forward (in compute_dtype) ---
    logits, caches = forward_deep(
        input_ids,
        fp_params["tok_emb"], fp_params["pos_emb"],
        fp_params["Wq"], fp_params["Wk"], fp_params["Wv"], fp_params["Wo"],
        fp_params["W1"], fp_params["W2"], fp_params["Wlm"],
        n_heads, n_layers, max_seq, np,
        training=True,
    )
    T = logits.shape[0]
    targets = np.asarray(target_ids[:T], dtype=np.intp)

    # --- Loss (use float32 for numerical stability) ---
    logits_f32 = logits.astype(master_dtype) if compute_dtype is not None else logits
    loss, dL_dlogits = _softmax_cross_entropy_bwd(logits_f32, targets, np)
    dL_dlogits = dL_dlogits.astype(logits.dtype) if compute_dtype is not None else dL_dlogits

    # Scale loss for FP16 gradient preservation
    if loss_scale != 1.0:
        dL_dlogits = dL_dlogits * loss_scale

    # --- Backward (in compute_dtype) ---
    grads: Dict[str, Any] = {}

    Y = caches["Y"]
    grads["Wlm"] = Y.T @ dL_dlogits
    dY = dL_dlogits @ fp_params["Wlm"].T

    dX_final = _layer_norm_bwd(dY, caches["final_ln"], np)
    dX = dX_final

    shared_grads = {
        "Wq": np.zeros_like(fp_params["Wq"]),
        "Wk": np.zeros_like(fp_params["Wk"]),
        "Wv": np.zeros_like(fp_params["Wv"]),
        "Wo": np.zeros_like(fp_params["Wo"]),
        "W1": np.zeros_like(fp_params["W1"]),
        "W2": np.zeros_like(fp_params["W2"]),
    }

    for layer_idx in range(n_layers - 1, -1, -1):
        dX, layer_grads = _transformer_block_bwd(dX, caches["layer_caches"][layer_idx], np)
        for k in shared_grads:
            shared_grads[k] += layer_grads[k]

    grads.update(shared_grads)

    idx = caches["tok_emb_idx"]
    t_len = caches["t_len"]
    grads["pos_emb"] = np.zeros_like(fp_params["pos_emb"])
    grads["tok_emb"] = np.zeros_like(fp_params["tok_emb"])
    grads["pos_emb"][:t_len] += dX
    np.add.at(grads["tok_emb"], idx, dX)

    # --- Unscale + Check for overflow (NaN/Inf in gradients) ---
    overflow = False
    if loss_scale != 1.0:
        inv_scale = 1.0 / loss_scale
        for k in grads:
            g = grads[k]
            if np.any(~np.isfinite(g)):
                overflow = True
                break
            grads[k] = g * inv_scale

    # Cast gradients to master dtype for Adam update
    if compute_dtype is not None:
        for k in grads:
            grads[k] = grads[k].astype(master_dtype)

    if not overflow:
        _adamw_update(params, grads, adam_state, lr, adam_t, np)

    return float(loss), overflow


# ======================================================================
# Public training entry point
# ======================================================================

def train_deep_backprop(
    text: str,
    out_path: Path,
    max_vocab: int = 256,
    d_model: int = 256,
    d_ff: int = 1024,
    n_heads: int = 8,
    n_layers: int = 4,
    max_seq: int = 64,
    steps: int = 2000,
    lr: float = 3e-4,
    seed: int = 1,
    *,
    show_progress: bool = True,
    progress_stream: TextIO | None = None,
    warm_start_path: Path | None = None,
    control_path: Path | None = None,
) -> DeepTransformerWeights:
    """
    Train DeepTransformer with full backprop through all layers and heads.
    """
    from lifers.core.compute_backend import get_compute_backend, get_cpu_numpy, print_backend_info
    np, device, gpu_info = get_compute_backend()
    cpu_np = get_cpu_numpy()

    # FP16 混合精度: GPU 环境下自动启用（可手动关闭 LIFERS_FP16=0）
    use_fp16 = (
        device == "cuda"
        and os.environ.get("LIFERS_FP16", "1").strip().lower() not in ("0", "false", "no", "off")
    )
    compute_dtype = np.float16 if use_fp16 else (np.float32 if device == "cuda" else np.float64)
    master_dtype = np.float32  # master weights always float32 for stability

    if use_fp16:
        print(f"[deep-backprop] FP16 mixed precision enabled (GPU={gpu_info.get('name','?') if gpu_info else '?'})",
              file=sys.stderr)

    print_backend_info()

    rng = random.Random(seed)

    # Try warm-start first — use saved vocab if weights match architecture
    w: DeepTransformerWeights | None = None
    warm_vocab: List[str] | None = None
    if warm_start_path is not None and warm_start_path.is_file():
        try:
            cand = DeepTransformerWeights.load(warm_start_path)
            if (cand.d_model == d_model and cand.d_ff == d_ff
                    and cand.n_heads == n_heads and cand.n_layers == n_layers
                    and cand.max_seq == max_seq):
                w = cand
                warm_vocab = list(cand.vocab)
        except Exception:
            pass

    # Vocab: prefer saved vocab (matches weights), fall back to corpus
    if warm_vocab is not None:
        vocab = warm_vocab
        stoi = {ch: i for i, ch in enumerate(vocab)}
    else:
        vocab = build_vocab_from_text(text, max_vocab=max_vocab)
        stoi = {ch: i for i, ch in enumerate(vocab)}
    ids_raw = encode(text, stoi)
    V = len(vocab)

    if w is None:
        w = init_deep_weights(vocab, d_model, d_ff, n_heads, n_layers, max_seq, seed)

    # Convert to params dict — master weights in float32, compute in fp16/fp32/fp64
    params: Dict[str, Any] = {
        "tok_emb": np.asarray(w.tok_emb, dtype=master_dtype),
        "pos_emb": np.asarray(w.pos_emb, dtype=master_dtype),
        "Wq": np.asarray(w.Wq, dtype=master_dtype),
        "Wk": np.asarray(w.Wk, dtype=master_dtype),
        "Wv": np.asarray(w.Wv, dtype=master_dtype),
        "Wo": np.asarray(w.Wo, dtype=master_dtype),
        "W1": np.asarray(w.W1, dtype=master_dtype),
        "W2": np.asarray(w.W2, dtype=master_dtype),
        "Wlm": np.asarray(w.Wlm, dtype=master_dtype),
    }
    # FP16 compute copy (如果需要混合精度)
    if use_fp16:
        params_fp16 = {k: v.astype(np.float16) for k, v in params.items()}
    else:
        params_fp16 = params
    adam_state = _adamw_init(params)

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
    dev_label = f"GPU:{gpu_info['name']}" if device == "cuda" else "CPU"
    print(f"[deep-backprop] V={V} D={d_model} F={d_ff} H={n_heads} L={n_layers} S={max_seq} "
          f"steps={steps} device={dev_label} — {tip}", file=sys.stderr)

    def _save() -> None:
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

    def _save_adam(step_num: int) -> None:
        """Save AdamW state for crash recovery within same tier (GPU-safe I/O)."""
        adam_path = out_path.parent / "lifers_deep_adam.npz"
        d = {}
        for k in params:
            val_m = adam_state[k]["m"]
            val_v = adam_state[k]["v"]
            d[f"{k}_m"] = val_m if isinstance(val_m, cpu_np.ndarray) else val_m.get()
            d[f"{k}_v"] = val_v if isinstance(val_v, cpu_np.ndarray) else val_v.get()
        d["_step"] = cpu_np.array([step_num], dtype=cpu_np.int64)
        cpu_np.savez_compressed(adam_path, **d)

    def _load_adam() -> int | None:
        """Load AdamW state if available for this tier (GPU-safe). Returns loaded step or None.
        Returns None when shapes don't match (avoids false positive resume)."""
        adam_path = out_path.parent / "lifers_deep_adam.npz"
        if not adam_path.is_file():
            return None
        try:
            arrs = cpu_np.load(adam_path)
            loaded_any = False
            for k in params:
                key_m = f"{k}_m"
                key_v = f"{k}_v"
                if key_m in arrs and key_v in arrs:
                    loaded_m = arrs[key_m]
                    loaded_v = arrs[key_v]
                    if loaded_m.shape == adam_state[k]["m"].shape:
                        # Transfer to device if on GPU
                        adam_state[k]["m"] = np.asarray(loaded_m) if device == "cuda" else loaded_m
                        adam_state[k]["v"] = np.asarray(loaded_v) if device == "cuda" else loaded_v
                        loaded_any = True
                    else:
                        return None  # shape mismatch — don't risk corrupt state
            if not loaded_any:
                return None
            return int(arrs.get("_step", cpu_np.array([0]))[0])
        except Exception:
            return None

    def _clean_adam() -> None:
        """Remove Adam checkpoint (tier complete or shape mismatch)."""
        adam_path = out_path.parent / "lifers_deep_adam.npz"
        adam_path.unlink(missing_ok=True)

    # Try load AdamW checkpoint (crash recovery within same tier)
    resumed_step = _load_adam()
    start_step = resumed_step if resumed_step else 0
    if resumed_step:
        print(f"[deep-backprop] AdamW state loaded (resume from step {resumed_step})",
              file=sys.stderr)

    t0 = time.time()
    total_loss = 0.0
    n_data = len(ids_raw)
    seq_len = max_seq + 1

    # FP16 动态 loss scale: 初始 2^16，溢出折半，稳定则翻倍
    loss_scale = float(65536) if use_fp16 else 1.0
    overflow_steps = 0

    for step in range(start_step, steps):
        if control_path is not None and (step == start_step or (step + 1) % ctrl_chk == 0):
            mode = read_train_control(control_path)
            if mode in ("pause", "stop"):
                _save()
                _save_adam(step + 1)
                if mode == "stop":
                    raise LifersTrainingStop()
                raise LifersTrainingPause()

        if n_data > seq_len:
            start = rng.randrange(0, n_data - seq_len)
        else:
            start = 0
        chunk = ids_raw[start:start + seq_len]
        input_ids = chunk[:max_seq]
        target_ids = chunk[1:max_seq + 1]

        loss_val, overflow = _train_step_deep(
            input_ids, target_ids, params, adam_state, step + 1, lr,
            n_heads, n_layers, max_seq, np,
            compute_dtype=compute_dtype, loss_scale=loss_scale,
        )

        # FP16 动态 loss scale 调整
        if use_fp16:
            if overflow or not np.isfinite(loss_val):
                loss_scale = max(loss_scale * 0.5, 1.0)
                overflow_steps = 0
                continue  # 跳过这个 step 的 loss 累加
            else:
                overflow_steps += 1
                if overflow_steps >= 2000:
                    loss_scale = min(loss_scale * 2.0, 2.0 ** 24)
                    overflow_steps = 0

        total_loss += loss_val

        if (step + 1) % save_every == 0 or step + 1 == steps:
            _save()
            _save_adam(step + 1)

        if stream is not None:
            upd = tty_every if stream.isatty() else log_every
            if step == start_step or (step + 1) % upd == 0 or step + 1 == steps:
                steps_done = step + 1 - start_step
                avg_loss = total_loss / max(1, steps_done)
                elapsed = time.time() - t0
                sps = steps_done / max(0.01, elapsed)
                write_progress_line(
                    stream, step + 1, steps,
                    prefix=f"deep V={V} D={d_model} L={n_layers} loss={avg_loss:.3f} {sps:.1f}s/s ",
                )
                try:
                    refresh_sgd_status(step + 1, steps, V, d_model)
                except Exception:
                    pass
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

    _save()
    _clean_adam()  # tier complete, Adam state won't transfer to next tier
    elapsed = time.time() - t0
    print(f"[deep-backprop] done {steps} steps in {elapsed:.1f}s, final loss={total_loss/steps:.4f}",
          file=sys.stderr)
    return w
