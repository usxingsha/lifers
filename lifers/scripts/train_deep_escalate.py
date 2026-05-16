"""
Deep Transformer escalate ramp — build depth and width gradually.

Architecture ramp (configurable via env):
  LIFERS_DEEP_MIN_LAYERS (default 2)  → LIFERS_DEEP_MAX_LAYERS (default 10)
  LIFERS_DEEP_MIN_HEADS  (default 4)  → LIFERS_DEEP_MAX_HEADS  (default 8)
  d_model: 96 → 2048   d_ff: max(256, d_model*4)  → capped at 8192
  vocab: 128 → 4096    seq: 64 → 256   steps: proportional to params

With weight sharing (ALBERT-style), parameter count stays roughly the same
regardless of layer count — only activations consume more RAM.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from lifers.speed_env import pause_poll_seconds
from lifers.train_control import (
    LifersTrainingPause,
    LifersTrainingStop,
    control_file_path,
    read_train_control,
    write_default_run,
    write_train_control,
)
from lifers.train_progress import end_progress_line, write_progress_line
from lifers.train_status_file import (
    detect_crash,
    finalize_train_status,
    publish_escalate_snapshot,
    recover_stale_tmp,
)
from lifers.deep_transformer import (
    DeepTransformerWeights,
    build_vocab_from_text,
    init_deep_weights,
)
from lifers.deep_transformer_train import train_deep_backprop


# ---------------------------------------------------------------------------
# Env hparams
# ---------------------------------------------------------------------------

def _snap_d_model(raw: int, n_heads: int) -> int:
    """Round d_model down to nearest multiple of 2*n_heads (RoPE needs even head_dim)."""
    step = 2 * n_heads
    if raw < step:
        return step
    return (raw // step) * step


def _read_deep_env() -> tuple:
    """Return (d_model, d_ff, max_vocab, max_seq, steps, n_layers, n_heads)."""
    n_heads = int(os.environ.get("LIFERS_DEEP_MIN_HEADS", "4"))
    d_model_raw = int(os.environ.get("LIFERS_DEEP_D_START", "96"))
    d_model = _snap_d_model(d_model_raw, n_heads)
    d_ff = int(os.environ.get("LIFERS_DEEP_FF_START", str(max(256, d_model * 4))))
    max_vocab = int(os.environ.get("LIFERS_DEEP_V_START", "128"))
    max_seq = int(os.environ.get("LIFERS_DEEP_S_START", "64"))
    steps = int(os.environ.get("LIFERS_DEEP_STEPS_START", "200"))
    n_layers = int(os.environ.get("LIFERS_DEEP_MIN_LAYERS", "2"))
    return d_model, d_ff, max_vocab, max_seq, steps, n_layers, n_heads


def _grow_tier_params(it: int, max_vocab: int, d_model: int, d_ff: int,
                      max_seq: int, steps: int, n_layers: int, n_heads: int,
                      target: float = float("inf")):
    """Grow tier parameters for the next escalate iteration. Shared logic."""
    grow = 1.25 if (not math.isfinite(target) or _rough_est(max_vocab, d_model, d_ff, n_layers) < target * 0.01) else 1.08
    max_heads = int(os.environ.get("LIFERS_DEEP_MAX_HEADS", "8"))
    if it > 0 and (it + 1) % 5 == 0 and n_heads < max_heads:
        n_heads += 2
    d_model = _snap_d_model(int(min(2048, max(d_model + 8, int(d_model * grow)))), n_heads)
    d_ff = int(min(8192, max(d_ff, int(d_model * 4))))
    max_vocab = int(min(4096, max_vocab + 64))
    max_seq = int(min(256, max_seq + 8))
    steps = int(min(12000, int(steps * 1.05)))
    max_layers = int(os.environ.get("LIFERS_DEEP_MAX_LAYERS", "10"))
    if it > 0 and (it + 1) % 4 == 0 and n_layers < max_layers:
        n_layers += 1
    return max_vocab, d_model, d_ff, max_seq, steps, n_layers, n_heads


def _auto_lr(d_model: int, base_lr: float = 3e-4, base_d: int = 256) -> float:
    """Scale LR down as model grows: lr = base_lr * sqrt(base_d / d_model)."""
    if d_model <= base_d:
        return base_lr
    return base_lr * math.sqrt(base_d / d_model)


def _configure_threads(d_model: int) -> int:
    """Dynamically set OpenBLAS thread count based on model size.
    Small models (D<1024): 1 thread (overhead dominates)
    Medium (D 1024-1536): 2 threads
    Large (D>=1536): 4 threads (matrices large enough for parallel speedup)

    Returns the number of threads configured.
    """
    auto = os.environ.get("LIFERS_AUTO_THREADS", "1").strip()
    if auto in ("0", "false", "no"):
        return 0  # user wants manual control

    if d_model < 1536:
        threads = 1
    elif d_model < 1920:
        threads = 2
    else:
        threads = 4

    # Runtime OpenBLAS config (works even after numpy init)
    try:
        import ctypes
        for libname in ("libopenblas.so.0", "libopenblas.so", "libopenblas64.so.0"):
            try:
                lib = ctypes.cdll.LoadLibrary(libname)
                lib.openblas_set_num_threads.argtypes = [ctypes.c_int]
                lib.openblas_set_num_threads(threads)
                break
            except Exception:
                continue
    except Exception:
        pass

    # Also update env for child processes and numpy internal checks
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    return threads


def _rough_est(max_vocab: int, d_model: int, d_ff: int, n_layers: int) -> float:
    """Rough float count for deep transformer (with weight sharing)."""
    v, d, f, n = float(max_vocab), float(d_model), float(d_ff), float(n_layers)
    tok_emb = v * d
    pos_emb = 256.0 * d  # max_seq upper bound
    # Per-block (shared across layers): 4 attention projections + 2 FFN
    attn = 4.0 * d * d
    ffn = 2.0 * d * f
    head = d * v
    # Activations roughly scale with layers (but not stored all at once)
    return tok_emb + pos_emb + attn + ffn + head


def _iter_deep_ramp(
    d_model: int, d_ff: int, max_vocab: int, max_seq: int,
    steps: int, n_layers: int, n_heads: int,
    target_stop: float, max_iters: int,
):
    """Yield (it, vocab, D, FF, seq, steps, layers, heads, est) at each tier."""
    for it in range(max_iters):
        est = _rough_est(max_vocab, d_model, d_ff, n_layers)
        yield it, max_vocab, d_model, d_ff, max_seq, steps, n_layers, n_heads, est
        if est >= target_stop:
            break
        max_vocab, d_model, d_ff, max_seq, steps, n_layers, n_heads = (
            _grow_tier_params(it, max_vocab, d_model, d_ff, max_seq, steps, n_layers, n_heads, target_stop)
        )


# ---------------------------------------------------------------------------
# Resume detection
# ---------------------------------------------------------------------------

def _infer_deep_resume(
    out: Path, max_iters: int,
) -> tuple | None:
    """Match existing weights to ramp tier."""
    try:
        w = DeepTransformerWeights.load(out)
    except Exception:
        return None
    key = (w.d_model, w.d_ff, w.n_layers, w.n_heads, w.max_seq)
    d0, ff0, v0, s0, st0, l0, h0 = _read_deep_env()
    for row in _iter_deep_ramp(d0, ff0, v0, s0, st0, l0, h0, float("inf"), max(max_iters, 4096)):
        it, mv, dm, df, ms, st, nl, nh, est = row
        if (dm, df, nl, nh, ms) == key:
            return it, mv, dm, df, ms, st, nl, nh, est
    return None


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def _load_jsonl_inputs(path: Path) -> str:
    buf = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                buf.append(str(obj.get("input", "")))
            except json.JSONDecodeError:
                buf.append(line)
    return "\n".join(buf)


def _load_corpus(root: Path, suite: Path) -> str:
    # Primary: training_corpus.txt (dense, comprehensive)
    corpus_txt = root / "weights" / "training_corpus.txt"
    if corpus_txt.is_file() and corpus_txt.stat().st_size > 10240:
        return corpus_txt.read_text(encoding="utf-8")
    # Fallback: JSONL suite
    text = []
    if suite.is_dir():
        for p in sorted(suite.glob("*.jsonl")):
            text.append(_load_jsonl_inputs(p))
    corpus = "\n".join(text) + "\n"
    if not corpus.strip() and corpus_txt.is_file():
        corpus = corpus_txt.read_text(encoding="utf-8")
    return corpus


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _maybe_checkpoint(root: Path, out: Path, cumulative: float,
                      last_floor: int, it: int, est: float) -> int:
    every_b = float(os.environ.get("LIFERS_CHECKPOINT_EVERY_B", "2").strip() or "2")
    if every_b <= 0:
        return last_floor
    new_floor = int(cumulative // (every_b * 1e9))
    if new_floor <= last_floor:
        return last_floor
    cp_dir = root / "weights" / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    for bf in range(last_floor + 1, new_floor + 1):
        tag = time.strftime("%Y%m%dT%H%M%S")
        dest = cp_dir / f"deep_chunk_{bf}B_iter{it + 1}_{tag}.json"
        shutil.copy2(out, dest)
        print(f"[deep-escalate] checkpoint B>={bf} -> {dest.name}", flush=True)
    return new_floor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    root = Path(__file__).resolve().parent.parent
    raw_suite = os.environ.get("LIFERS_TRAIN_SUITE_DIR", "").strip()
    suite = Path(raw_suite).expanduser() if raw_suite else (root / "eval" / "suites" / "v001")
    if not suite.is_absolute():
        suite = (root / suite).resolve()
    corpus = _load_corpus(root, suite)
    if not corpus.strip():
        print("[deep-escalate] fatal: no corpus", file=sys.stderr, flush=True)
        return 1

    out = root / "weights" / "lifers_deep_transformer.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Crash recovery
    if recover_stale_tmp(out):
        print("[deep-escalate] recovered .tmp from crash", flush=True)
    crash = detect_crash(root)
    if crash:
        print(f"[deep-escalate] previous crash at ramp {crash.get('ramp_iter','?')}", flush=True)

    ctl_path = control_file_path(out.parent)
    write_train_control(ctl_path, "run")  # always start in run mode

    state_path = out.parent / ".deep_train_state.json"

    unlimited = os.environ.get("LIFERS_ESCALATE_UNLIMITED", "0").strip() in ("1", "true", "yes")
    target_b = float(os.environ.get("LIFERS_TARGET_PARAM_B", "10").strip() or "10")
    if unlimited or target_b <= 0:
        target = float("inf")
    else:
        target = max(target_b, 0.001) * 1e9

    default_mi = "999999" if unlimited else "60"
    max_iters = int(os.environ.get("LIFERS_RAMP_MAX_ITERS", default_mi).strip() or default_mi)
    max_iters = max(1, min(max_iters, 10_000_000))

    d_model, d_ff, max_vocab, max_seq, steps, n_layers, n_heads = _read_deep_env()

    cumulative_est = 0.0
    last_b_floor = 0
    if state_path.is_file():
        try:
            s = json.loads(state_path.read_text(encoding="utf-8"))
            cumulative_est = float(s.get("cumulative_est", 0.0))
            last_b_floor = int(s.get("last_b_floor", 0))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    start_it = 0
    warm_path: Path | None = None
    resume_on = os.environ.get("LIFERS_ESCALATE_RESUME", "1").strip() not in ("0", "false", "no")
    if resume_on and out.is_file():
        inferred = _infer_deep_resume(out, max_iters)
        if inferred is not None:
            start_it, max_vocab, d_model, d_ff, max_seq, steps, n_layers, n_heads, _ = inferred
            warm_path = out
            print(f"[deep-escalate] resume tier {start_it + 1}/{max_iters} "
                  f"D={d_model} L={n_layers} H={n_heads} (warm-start)", flush=True)
        elif out.stat().st_size > 1024:
            print("[deep-escalate] weights exist but shape mismatch — cold start", flush=True)
    elif not out.is_file():
        print("[deep-escalate] no weights — cold start", flush=True)

    print(f"[deep-escalate] target≈{target_b if math.isfinite(target) else '∞'}B "
          f"max_iters={max_iters} ctl={ctl_path} "
          f"start D={d_model} L={n_layers} H={n_heads} V={max_vocab} seq={max_seq}",
          flush=True)

    publish_escalate_snapshot(root, phase="ramp_ready", ramp_max=max_iters,
                              cumulative_est_g=cumulative_est / 1e9,
                              message="deep escalate ready")

    max_tier_cap = 0
    raw_cap = os.environ.get("LIFERS_ESCALATE_MAX_TIER", "").strip()
    if raw_cap:
        try:
            max_tier_cap = max(0, int(raw_cap))
        except ValueError:
            pass

    last_err: str | None = None

    for it in range(start_it, max_iters):
        mode = read_train_control(ctl_path)
        if mode == "stop":
            print("[deep-escalate] control=stop", flush=True)
            break
        if mode == "pause":
            print("[deep-escalate] control=pause — waiting", flush=True)
            while read_train_control(ctl_path) == "pause":
                time.sleep(pause_poll_seconds())
            if read_train_control(ctl_path) == "stop":
                break

        tier_display = it + 1
        if max_tier_cap > 0 and tier_display > max_tier_cap:
            print(f"[deep-escalate] tier cap {max_tier_cap} reached", flush=True)
            break

        est = _rough_est(max_vocab, d_model, d_ff, n_layers)
        est_m = est / 1e6

        print(f"[deep-escalate] tier {tier_display}/{max_iters} "
              f"D={d_model} F={d_ff} L={n_layers} H={n_heads} V={max_vocab} "
              f"seq={max_seq} steps={steps} ~{est_m:.1f}M", flush=True)

        # Set env for status updates
        os.environ["LIFERS_TRAIN_STATUS_BRAIN_ROOT"] = str(root)
        os.environ["LIFERS_TRAIN_STATUS_RAMP_ITER"] = str(tier_display)
        os.environ["LIFERS_TRAIN_STATUS_RAMP_MAX"] = str(max_iters)
        os.environ["LIFERS_TRAIN_STATUS_TIER_EST_M"] = str(est_m)
        os.environ["LIFERS_TRAIN_STATUS_MAX_V"] = str(max_vocab)
        os.environ["LIFERS_TRAIN_STATUS_D"] = str(d_model)
        os.environ["LIFERS_TRAIN_STATUS_DFF"] = str(d_ff)
        os.environ["LIFERS_TRAIN_STATUS_MS"] = str(max_seq)
        os.environ["LIFERS_TRAIN_STATUS_STEPS"] = str(steps)

        publish_escalate_snapshot(
            root, phase="tier_sgd", ramp_iter=tier_display, ramp_max=max_iters,
            tier_est_m=est_m, max_vocab=max_vocab, d_model=d_model, d_ff=d_ff,
            max_seq=max_seq, steps=steps, cumulative_est_g=cumulative_est / 1e9,
            message=f"deep tier {tier_display}/{max_iters} L={n_layers}",
        )

        # Reload corpus each tier so auto_expand additions are picked up
        corpus = _load_corpus(root, suite)

        try:
            base_lr = float(os.environ.get("TT_LR", "3e-4"))
            scaled_lr = _auto_lr(d_model, base_lr=base_lr)
            if scaled_lr < base_lr:
                print(f"[deep-escalate] lr auto-scaled: {base_lr:.1e} → {scaled_lr:.1e} "
                      f"(D={d_model})", flush=True)
            th = _configure_threads(d_model)
            if th:
                print(f"[deep-escalate] threads={th} (auto, D={d_model})", flush=True)
            train_deep_backprop(
                corpus, out,
                max_vocab=max_vocab, d_model=d_model, d_ff=d_ff,
                n_heads=n_heads, n_layers=n_layers, max_seq=max_seq,
                steps=steps,
                lr=scaled_lr,
                warm_start_path=(warm_path if warm_path is not None and it == start_it else None),
                control_path=ctl_path,
            )
            write_progress_line(sys.stdout, tier_display, max_iters,
                                prefix=f"[deep-escalate] done ~{est_m:.1f}M | ")

        except LifersTrainingPause:
            print("[deep-escalate] paused mid-tier", flush=True)
            end_progress_line(sys.stdout)
            _save_state(state_path, cumulative_est, last_b_floor)
            finalize_train_status(root, "paused", "pause mid-tier")
            return 0
        except LifersTrainingStop:
            print("[deep-escalate] stopped mid-tier", flush=True)
            end_progress_line(sys.stdout)
            _save_state(state_path, cumulative_est, last_b_floor)
            finalize_train_status(root, "stopped", "stop mid-tier")
            return 0
        except MemoryError:
            print("[deep-escalate] OOM — keeping last weights", flush=True)
            write_train_control(ctl_path, "pause")
            break
        except Exception as e:
            last_err = str(e)
            print(f"[deep-escalate] error: {e}\n{traceback.format_exc()}", flush=True)
            break

        cumulative_est += est
        last_b_floor = _maybe_checkpoint(root, out, cumulative_est, last_b_floor, it, est)
        _save_state(state_path, cumulative_est, last_b_floor)

        if math.isfinite(target) and est >= target:
            print(f"[deep-escalate] reached target {target:.4f}B", flush=True)
            break

        # Grow for next tier
        max_vocab, d_model, d_ff, max_seq, steps, n_layers, n_heads = (
            _grow_tier_params(it, max_vocab, d_model, d_ff, max_seq, steps, n_layers, n_heads, target)
        )

    end_progress_line(sys.stdout)

    ok = out.is_file()
    if ok:
        print(f"[deep-escalate] wrote {out}", flush=True)
        phase_end = "completed" if not last_err else "error"
        msg_end = "ramp ok" if not last_err else last_err[:400]
        finalize_train_status(root, phase_end, msg_end)
        return 0
    finalize_train_status(root, "failed", "no weights")
    return 1


def _save_state(path: Path, cumulative: float, floor: int) -> None:
    try:
        path.write_text(json.dumps(
            {"cumulative_est": cumulative, "last_b_floor": floor}, indent=2),
            encoding="utf-8")
    except OSError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
