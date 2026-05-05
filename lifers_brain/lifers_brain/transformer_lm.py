from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, TextIO, Tuple

from lifers_brain.speed_env import max_speed_enabled, train_control_every_steps, train_save_every, use_numpy_training
from lifers_brain.train_control import LifersTrainingPause, LifersTrainingStop, read_train_control
from lifers_brain.train_progress import end_progress_line, write_progress_line
from lifers_brain.train_status_file import refresh_sgd_status


def _softmax(xs: List[float]) -> List[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def _matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    # A: [m x k], B: [k x n] -> [m x n]
    m = len(A)
    k = len(A[0])
    n = len(B[0])
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        Ai = A[i]
        for kk in range(k):
            a = Ai[kk]
            Bkk = B[kk]
            for j in range(n):
                out[i][j] += a * Bkk[j]
    return out


def _transpose(A: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*A)]


def _add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def _layer_norm(X: List[List[float]], eps: float = 1e-5) -> List[List[float]]:
    out = []
    for row in X:
        mu = sum(row) / len(row)
        var = sum((x - mu) ** 2 for x in row) / len(row)
        inv = 1.0 / math.sqrt(var + eps)
        out.append([(x - mu) * inv for x in row])
    return out


def _relu(X: List[List[float]]) -> List[List[float]]:
    return [[x if x > 0 else 0.0 for x in row] for row in X]


def _randn(rng: random.Random, scale: float = 0.02) -> float:
    # Box-Muller
    u1 = max(1e-9, rng.random())
    u2 = rng.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return z * scale


@dataclass
class TinyTransformerWeights:
    vocab: List[str]
    d_model: int
    d_ff: int
    n_heads: int
    max_seq: int
    # Embeddings
    tok_emb: List[List[float]]  # [vocab x d_model]
    pos_emb: List[List[float]]  # [max_seq x d_model]
    # Attention (single block)
    Wq: List[List[float]]  # [d_model x d_model]
    Wk: List[List[float]]
    Wv: List[List[float]]
    Wo: List[List[float]]
    # FFN
    W1: List[List[float]]  # [d_model x d_ff]
    W2: List[List[float]]  # [d_ff x d_model]
    # LM head
    Wlm: List[List[float]]  # [d_model x vocab]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = {
            "vocab": self.vocab,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "n_heads": self.n_heads,
            "max_seq": self.max_seq,
            "tok_emb": self.tok_emb,
            "pos_emb": self.pos_emb,
            "Wq": self.Wq,
            "Wk": self.Wk,
            "Wv": self.Wv,
            "Wo": self.Wo,
            "W1": self.W1,
            "W2": self.W2,
            "Wlm": self.Wlm,
        }
        data = json.dumps(obj, ensure_ascii=False)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def load(path: Path) -> "TinyTransformerWeights":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return TinyTransformerWeights(**obj)


def build_vocab_from_text(text: str, max_vocab: int = 256) -> List[str]:
    freq: Dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    vocab = [ch for ch, _ in items[:max_vocab]]
    if "\n" not in vocab:
        vocab.append("\n")
    return vocab


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    unk = stoi.get("\n", 0)
    return [stoi.get(ch, unk) for ch in text]


def decode(ids: List[int], itos: List[str]) -> str:
    return "".join(itos[i] for i in ids if 0 <= i < len(itos))


def init_weights(vocab: List[str], d_model: int = 64, d_ff: int = 128, n_heads: int = 4, max_seq: int = 128, seed: int = 1) -> TinyTransformerWeights:
    rng = random.Random(seed)
    V = len(vocab)

    def mat(r: int, c: int) -> List[List[float]]:
        return [[_randn(rng) for _ in range(c)] for _ in range(r)]

    return TinyTransformerWeights(
        vocab=vocab,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        max_seq=max_seq,
        tok_emb=mat(V, d_model),
        pos_emb=mat(max_seq, d_model),
        Wq=mat(d_model, d_model),
        Wk=mat(d_model, d_model),
        Wv=mat(d_model, d_model),
        Wo=mat(d_model, d_model),
        W1=mat(d_model, d_ff),
        W2=mat(d_ff, d_model),
        Wlm=mat(d_model, V),
    )


def _attention_causal(X: List[List[float]], Wq: List[List[float]], Wk: List[List[float]], Wv: List[List[float]], Wo: List[List[float]]) -> List[List[float]]:
    # Single-head simplified attention for minimalism.
    Q = _matmul(X, Wq)  # [T x D]
    K = _matmul(X, Wk)
    Vv = _matmul(X, Wv)
    T = len(X)
    D = len(X[0])

    KT = _transpose(K)  # [D x T]
    scores = _matmul(Q, KT)  # [T x T]
    scale = 1.0 / math.sqrt(D)
    for i in range(T):
        for j in range(T):
            s = scores[i][j] * scale
            if j > i:
                s = -1e9
            scores[i][j] = s
    attn = []
    for i in range(T):
        attn.append(_softmax(scores[i]))

    # attn [T x T] * V [T x D] => [T x D]
    out = [[0.0] * D for _ in range(T)]
    for i in range(T):
        for j in range(T):
            a = attn[i][j]
            Vj = Vv[j]
            for d in range(D):
                out[i][d] += a * Vj[d]
    return _matmul(out, Wo)


def forward(w: TinyTransformerWeights, ids: List[int]) -> List[List[float]]:
    # Returns logits per position: [T x V]
    T = min(len(ids), w.max_seq)
    D = w.d_model
    V = len(w.vocab)

    X = []
    for t in range(T):
        tok = w.tok_emb[ids[t]]
        pos = w.pos_emb[t]
        X.append([tok[d] + pos[d] for d in range(D)])

    X = _layer_norm(X)
    A = _attention_causal(X, w.Wq, w.Wk, w.Wv, w.Wo)
    X2 = _layer_norm(_add(X, A))
    H = _relu(_matmul(X2, w.W1))
    F = _matmul(H, w.W2)
    Y = _layer_norm(_add(X2, F))
    logits = _matmul(Y, w.Wlm)  # [T x V]
    return logits


def _try_numpy():  # type: ignore[no-any-unimported]
    try:
        import numpy as np  # noqa: PLC0415

        return np
    except ImportError:
        return None


def _layer_norm_np(X, np: Any) -> Any:
    mu = X.mean(axis=1, keepdims=True)
    var = ((X - mu) ** 2).mean(axis=1, keepdims=True)
    return (X - mu) / np.sqrt(var + 1e-5)


def _softmax_rows_np(scores: Any, np: Any) -> Any:
    m = scores.max(axis=1, keepdims=True)
    ex = np.exp(scores - m)
    return ex / ex.sum(axis=1, keepdims=True)


def _attention_causal_np(
    X: Any,
    Wq: Any,
    Wk: Any,
    Wv: Any,
    Wo: Any,
    np: Any,
) -> Any:
    d = int(X.shape[1])
    Q = X @ Wq
    K = X @ Wk
    Vv = X @ Wv
    scale = 1.0 / math.sqrt(float(d))
    scores = (Q @ K.T) * scale
    t = int(scores.shape[0])
    if t > 1:
        tri = np.triu(np.ones((t, t), dtype=np.float64), k=1)
        scores = np.where(tri > 0, -1e9, scores)
    attn = _softmax_rows_np(scores, np)
    out = attn @ Vv
    return out @ Wo


def forward_np(
    w: TinyTransformerWeights,
    tok_emb: Any,
    pos_emb: Any,
    Wq: Any,
    Wk: Any,
    Wv: Any,
    Wo: Any,
    W1: Any,
    W2: Any,
    Wlm: Any,
    ids: List[int],
    np: Any,
) -> Any:
    """Same semantics as forward(); tensors are ndarray views (Wlm may be updated in-place)."""
    t_len = min(len(ids), w.max_seq)
    d_model = w.d_model
    d_ff = w.d_ff
    idx = np.asarray(ids[:t_len], dtype=np.intp)
    te = tok_emb[idx]
    pe = pos_emb[:t_len]
    X = te + pe
    X = _layer_norm_np(X, np)
    A = _attention_causal_np(X, Wq, Wk, Wv, Wo, np)
    X2 = _layer_norm_np(X + A, np)
    H = np.maximum(0.0, X2 @ W1)
    F = H @ W2
    Y = _layer_norm_np(X2 + F, np)
    return Y @ Wlm  # [T, V]


def _np_tensors_from_weights(w: TinyTransformerWeights, np: Any) -> tuple[Any, ...]:
    return (
        np.asarray(w.tok_emb, dtype=np.float64),
        np.asarray(w.pos_emb, dtype=np.float64),
        np.asarray(w.Wq, dtype=np.float64),
        np.asarray(w.Wk, dtype=np.float64),
        np.asarray(w.Wv, dtype=np.float64),
        np.asarray(w.Wo, dtype=np.float64),
        np.asarray(w.W1, dtype=np.float64),
        np.asarray(w.W2, dtype=np.float64),
        np.asarray(w.Wlm, dtype=np.float64),
    )


def _tiny_from_np_tensors(
    w: TinyTransformerWeights,
    tok_emb: Any,
    pos_emb: Any,
    Wq: Any,
    Wk: Any,
    Wv: Any,
    Wo: Any,
    W1: Any,
    W2: Any,
    Wlm: Any,
) -> TinyTransformerWeights:
    return TinyTransformerWeights(
        vocab=w.vocab,
        d_model=w.d_model,
        d_ff=w.d_ff,
        n_heads=w.n_heads,
        max_seq=w.max_seq,
        tok_emb=tok_emb.tolist(),
        pos_emb=pos_emb.tolist(),
        Wq=Wq.tolist(),
        Wk=Wk.tolist(),
        Wv=Wv.tolist(),
        Wo=Wo.tolist(),
        W1=W1.tolist(),
        W2=W2.tolist(),
        Wlm=Wlm.tolist(),
    )


def generate_text(
    w: TinyTransformerWeights,
    prompt: str,
    max_chars: int = 160,
    seed: int = 1,
    temperature: float = 1.0,
    top_k: int = 50,
) -> str:
    rng = random.Random(seed)
    stoi = {ch: i for i, ch in enumerate(w.vocab)}
    itos = w.vocab
    ids = encode(prompt, stoi)

    for _ in range(max_chars):
        logits = forward(w, ids[-w.max_seq :])
        last = logits[-1]
        # temperature + top-k sampling
        scaled = [x / max(1e-6, temperature) for x in last]
        idxs = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)[: max(1, min(top_k, len(scaled)))]
        vals = [scaled[i] for i in idxs]
        probs = _softmax(vals)
        r = rng.random()
        acc = 0.0
        pick = idxs[-1]
        for i, p in zip(idxs, probs):
            acc += p
            if acc >= r:
                pick = i
                break
        ids.append(pick)

    return decode(ids[len(encode(prompt, stoi)) :], itos)


def train_sgd_minimal(
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
    Minimal self-contained trainer (VERY small):
    - uses a crude finite-difference gradient on LM head only (to keep it dependency-free)
    - goal: produce a non-trivial self-made weight file, not SOTA.

    This keeps dependencies minimal, but training quality is limited.
    """
    rng = random.Random(seed)
    vocab = build_vocab_from_text(text, max_vocab=max_vocab)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    ids = encode(text, stoi)
    w: TinyTransformerWeights
    if warm_start_path is not None and warm_start_path.is_file():
        try:
            cand = TinyTransformerWeights.load(warm_start_path)
            if (
                list(cand.vocab) == vocab
                and cand.d_model == d_model
                and cand.d_ff == d_ff
                and cand.max_seq == max_seq
                and cand.n_heads == 1
            ):
                w = cand
            else:
                w = init_weights(vocab=vocab, d_model=d_model, d_ff=d_ff, n_heads=1, max_seq=max_seq, seed=seed)
        except (json.JSONDecodeError, OSError, TypeError, KeyError):
            w = init_weights(vocab=vocab, d_model=d_model, d_ff=d_ff, n_heads=1, max_seq=max_seq, seed=seed)
    else:
        w = init_weights(vocab=vocab, d_model=d_model, d_ff=d_ff, n_heads=1, max_seq=max_seq, seed=seed)

    V = len(vocab)
    D = d_model

    # Finite-difference update on Wlm only (tiny & slow but dependency-free).
    # Keep it deliberately small so the full pipeline stays fast.
    stream: TextIO | None = None
    if show_progress:
        stream = progress_stream if progress_stream is not None else sys.stderr
    log_every = max(1, min(50, max(1, steps // 25)))
    tty_every = 1 if steps <= 500 else max(1, steps // 40)
    if max_speed_enabled():
        log_every = max(1, steps // 20)
        tty_every = max(1, steps // 25)
    ctrl_chk = train_control_every_steps(log_every)
    save_every = train_save_every(steps)

    np_mod = _try_numpy()
    if (
        os.environ.get("LIFERS_USE_NUMPY", "").strip().lower() in ("1", "true", "yes", "on")
        and np_mod is None
    ):
        print(
            "[train_sgd] LIFERS_USE_NUMPY=1 but numpy is not installed; falling back to pure Python",
            file=sys.stderr,
        )
    use_np = use_numpy_training(np_mod is not None)

    if use_np:
        assert np_mod is not None
        tok_emb, pos_emb, Wq, Wk, Wv, Wo, W1, W2, Wlm = _np_tensors_from_weights(w, np_mod)
        thr_line = (
            os.environ.get("OMP_NUM_THREADS", "").strip()
            or os.environ.get("OPENBLAS_NUM_THREADS", "").strip()
            or os.environ.get("MKL_NUM_THREADS", "").strip()
        )
        tip = f"OMP_NUM_THREADS={thr_line}" if thr_line else "export OMP_NUM_THREADS=$(nproc)  # Linux: saturate CPU"
        print(f"[train_sgd] NumPy/BLAS forward path — {tip}", file=sys.stderr)

        def loss_np(start: int) -> float:
            seq = ids[start : start + max_seq + 1]
            x = seq[:-1]
            y = seq[1:]
            logits = forward_np(w, tok_emb, pos_emb, Wq, Wk, Wv, Wo, W1, W2, Wlm, x, np_mod)
            last = logits[-1]
            m = float(last.max())
            ex = np_mod.exp(last - m)
            pr = ex / ex.sum()
            return -math.log(max(1e-9, float(pr[y[-1]])))

        def save_np() -> None:
            _tiny_from_np_tensors(w, tok_emb, pos_emb, Wq, Wk, Wv, Wo, W1, W2, Wlm).save(out_path)

        for step in range(steps):
            if control_path is not None and (step == 0 or (step + 1) % ctrl_chk == 0):
                mode = read_train_control(control_path)
                if mode in ("pause", "stop"):
                    save_np()
                    if mode == "stop":
                        raise LifersTrainingStop()
                    raise LifersTrainingPause()

            start = rng.randrange(0, max(1, len(ids) - (max_seq + 2)))
            base_loss = loss_np(start)

            for _ in range(3):
                i = rng.randrange(0, D)
                j = rng.randrange(0, V)
                old = float(Wlm[i, j])
                eps = 1e-3
                Wlm[i, j] = old + eps
                l2 = loss_np(start)
                grad = (l2 - base_loss) / eps
                Wlm[i, j] = old - lr * grad

            if (step + 1) % save_every == 0 or step + 1 == steps:
                save_np()
            if stream is not None:
                upd = tty_every if stream.isatty() else log_every
                if max_speed_enabled():
                    upd = max(upd, max(1, steps // 55))
                if step == 0 or (step + 1) % upd == 0 or step + 1 == steps:
                    write_progress_line(
                        stream,
                        step + 1,
                        steps,
                        prefix=f"train_sgd V={V} D={D} ",
                    )
                    try:
                        refresh_sgd_status(step + 1, steps, V, D)
                    except Exception:
                        pass

        if stream is not None:
            end_progress_line(stream)

        save_np()
        return _tiny_from_np_tensors(w, tok_emb, pos_emb, Wq, Wk, Wv, Wo, W1, W2, Wlm)

    def loss_for_batch(start: int) -> float:
        seq = ids[start : start + max_seq + 1]
        x = seq[:-1]
        y = seq[1:]
        logits = forward(w, x)
        # NLL of last position only to keep it light
        last = logits[-1]
        probs = _softmax(last)
        return -math.log(max(1e-9, probs[y[-1]]))

    for step in range(steps):
        if control_path is not None and (step == 0 or (step + 1) % ctrl_chk == 0):
            mode = read_train_control(control_path)
            if mode in ("pause", "stop"):
                w.save(out_path)
                if mode == "stop":
                    raise LifersTrainingStop()
                raise LifersTrainingPause()

        start = rng.randrange(0, max(1, len(ids) - (max_seq + 2)))
        base_loss = loss_for_batch(start)

        # Update a tiny random subset of Wlm entries.
        for _ in range(3):
            i = rng.randrange(0, D)
            j = rng.randrange(0, V)
            old = w.Wlm[i][j]
            eps = 1e-3
            w.Wlm[i][j] = old + eps
            l2 = loss_for_batch(start)
            grad = (l2 - base_loss) / eps
            w.Wlm[i][j] = old - lr * grad

        if (step + 1) % save_every == 0 or step + 1 == steps:
            w.save(out_path)
        if stream is not None:
            upd = tty_every if stream.isatty() else log_every
            if max_speed_enabled():
                upd = max(upd, max(1, steps // 55))
            if step == 0 or (step + 1) % upd == 0 or step + 1 == steps:
                write_progress_line(
                    stream,
                    step + 1,
                    steps,
                    prefix=f"train_sgd V={V} D={D} ",
                )
                try:
                    refresh_sgd_status(step + 1, steps, V, D)
                except Exception:
                    pass

    if stream is not None:
        end_progress_line(stream)

    w.save(out_path)
    return w

