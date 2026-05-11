"""
Ramp TinyTransformer training until rough float-param budget or MemoryError.

- Writes weights/lifers_transformer.json (overwrites each successful tier = last wins).
- LIFERS_TARGET_PARAM_B: target in **billions of approximate float weights** (default 20).
  Pure-Python tiny LM will OOM / cap iterations long before 20B; that is expected — we climb until the machine says stop.
- LIFERS_RAMP_MAX_ITERS: safety cap (default 48, or 999999 when LIFERS_ESCALATE_UNLIMITED=1).
- LIFERS_ESCALATE_RESUME: default 1 — if weights/lifers_transformer.json exists and matches a ramp tier,
  continue from that tier with warm-start (set 0 to always cold-start).
- LIFERS_ESCALATE_UNLIMITED=1 — never stop on target B; only OOM, stop control, or max_iters.
- Control file (default weights/.train_control): first word run | pause | stop (see scripts/lifers_train_ctl.sh).
- LIFERS_CHECKPOINT_EVERY_B (default 1): after each tier, cumulative rough float est crosses N×B×1e9 →
  copy weights/checkpoints/chunk_{N}B_*.json, append manifest, run LIFERS_POST_CHECKPOINT_CMD if set.
- LIFERS_PAUSE_ON_CHECKPOINT=1: after any new B-floor checkpoint(s) and post-cmd, write control=pause so you can
  sync editors / pull artifacts; set run again to continue (no fixed “tier cap” — use LIFERS_ESCALATE_UNLIMITED).
- LIFERS_ESCALATE_MAX_TIER=N: never **start** training for ramp display iter > N (1-based, same as log `iter K/...`).
  Use on low-RAM hosts to stay at e.g. tier 16 instead of OOM on the next growth step (~38M→~68M params is typical).
- LIFERS_ESCALATE_PAUSE_ON_MEMERROR: default 1 — on MemoryError write control=pause so `kali_train_escalate_loop.sh`
  does not immediately restart and re-run the same resume tier (looks like “tier 16 twice forever”). Set 0 to keep
  hot-loop retry.
- LIFERS_TRAIN_SUITE_DIR: optional corpus directory (jsonl). Default eval/suites/v001. Use with capability queue script.
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

from lifers_brain.speed_env import pause_poll_seconds
from lifers_brain.train_control import (
    LifersTrainingPause,
    LifersTrainingStop,
    control_file_path,
    read_train_control,
    write_default_run,
    write_train_control,
)
from lifers_brain.train_progress import end_progress_line, write_progress_line
from lifers_brain.train_status_file import finalize_train_status, publish_escalate_snapshot
from lifers_brain.transformer_lm import TinyTransformerWeights, build_vocab_from_text, train_sgd_minimal


def _load_jsonl_inputs(path: Path) -> str:
    buf = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        buf.append(str(obj.get("input", "")))
    return "\n".join(buf)


def _read_ramp_env_hparams() -> tuple[bool, int, int, int, int, int]:
    honor_tt = os.environ.get("LIFERS_ESCALATE_HONOR_TT_STEPS", "").strip().lower() in ("1", "true", "yes")
    if honor_tt:
        d_model = int(os.environ.get("TT_DMODEL", "24"))
        d_ff = int(os.environ.get("TT_DFF", str(max(48, d_model * 2))))
        max_vocab = int(os.environ.get("TT_VOCAB", "96"))
        max_seq = int(os.environ.get("TT_MAXSEQ", "32"))
        steps = int(os.environ.get("TT_STEPS", "96"))
    else:
        d_model = int(os.environ.get("LIFERS_ESCALATE_D_START", "24"))
        d_ff = int(os.environ.get("LIFERS_ESCALATE_FF_START", str(max(48, d_model * 2))))
        max_vocab = int(os.environ.get("LIFERS_ESCALATE_V_START", "96"))
        max_seq = int(os.environ.get("LIFERS_ESCALATE_S_START", "32"))
        steps = int(os.environ.get("LIFERS_ESCALATE_STEPS_START", "96"))
    return honor_tt, d_model, d_ff, max_vocab, max_seq, steps


def _iter_ramp_schedule(
    corpus: str,
    target_stop: float,
    max_iters: int,
    d_model: int,
    d_ff: int,
    max_vocab: int,
    max_seq: int,
    steps: int,
):
    """Yield (it, max_vocab, d_model, d_ff, max_seq, steps, est, vlen) at each tier start (before train)."""
    for it in range(max_iters):
        est = rough_param_estimate(max_vocab, d_model, d_ff, max_seq)
        vlen = len(build_vocab_from_text(corpus, max_vocab=max_vocab))
        yield it, max_vocab, d_model, d_ff, max_seq, steps, est, vlen
        if est >= target_stop:
            break
        grow = (
            1.35
            if (not math.isfinite(target_stop) or est < target_stop * 0.01)
            else 1.12
        )
        d_model = int(min(4096, max(d_model + 4, int(d_model * grow))))
        d_ff = int(min(12288, max(d_ff, int(d_model * 2.2))))
        max_vocab = int(min(2048, max_vocab + 16))
        max_seq = int(min(192, max_seq + 4))
        steps = int(min(8000, int(steps * 1.08)))


def _infer_resume_tier(
    corpus: str,
    out: Path,
    max_iters: int,
) -> tuple[int, int, int, int, int, int] | None:
    """Match weights file to a ramp tier; return (it, max_vocab, d_model, d_ff, max_seq, steps) or None."""
    try:
        w = TinyTransformerWeights.load(out)
    except (json.JSONDecodeError, OSError, TypeError, KeyError):
        return None
    key = (len(w.vocab), w.d_model, w.d_ff, w.max_seq)
    honor_tt, d0, ff0, v0, s0, st0 = _read_ramp_env_hparams()
    _ = honor_tt
    max_sim = min(max(max_iters, 2048), 2_000_000)
    matches: list[tuple[int, int, int, int, int, int]] = []
    for row in _iter_ramp_schedule(corpus, float("inf"), max_sim, d0, ff0, v0, s0, st0):
        it, mv, dm, df, ms, st, _est, vlen = row
        if (vlen, dm, df, ms) == key:
            matches.append((it, mv, dm, df, ms, st))
    if not matches:
        return None
    return max(matches, key=lambda x: x[0])


def _escalate_max_tier_display() -> int:
    """Max ramp iter to start (1-based, matches printed `iter K/...`). 0 = no cap."""
    raw = os.environ.get("LIFERS_ESCALATE_MAX_TIER", "").strip()
    if not raw:
        return 0
    try:
        n = int(raw)
        return max(0, n)
    except ValueError:
        return 0


def rough_param_estimate(max_vocab: int, d_model: int, d_ff: int, max_seq: int) -> float:
    """Order-of-magnitude float count for TinyTransformer layout (single block, n_heads=1)."""
    v, d, f, s = float(max_vocab), float(d_model), float(d_ff), float(max_seq)
    tok = v * d
    pos = s * d
    attn = 4.0 * d * d
    ffn = d * f + f * d
    head = d * v
    return tok + pos + attn + ffn + head


def _load_train_state(path: Path) -> tuple[float, int]:
    try:
        o = json.loads(path.read_text(encoding="utf-8"))
        return float(o.get("cumulative_est", 0.0)), int(o.get("last_b_floor", 0))
    except (json.JSONDecodeError, OSError, TypeError, KeyError, ValueError):
        return 0.0, 0


def _save_train_state(path: Path, cumulative_est: float, last_b_floor: int) -> None:
    try:
        path.write_text(
            json.dumps({"cumulative_est": cumulative_est, "last_b_floor": last_b_floor}, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def _append_manifest(manifest: Path, row: dict) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _maybe_checkpoint_by_budget(
    *,
    root: Path,
    out: Path,
    cumulative_est: float,
    last_b_floor: int,
    it: int,
    est: float,
    max_iters: int,
) -> int:
    """Return updated last_b_floor; copy shards + optional shell sync."""
    every_b = float(os.environ.get("LIFERS_CHECKPOINT_EVERY_B", "1").strip() or "1")
    if every_b <= 0 or not math.isfinite(cumulative_est):
        return last_b_floor
    new_floor = int(cumulative_est // (every_b * 1e9))
    if new_floor <= last_b_floor:
        return last_b_floor
    cp_dir = root / "weights" / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    manifest = cp_dir / "manifest.jsonl"
    for bf in range(last_b_floor + 1, new_floor + 1):
        tag = time.strftime("%Y%m%dT%H%M%S")
        dest = cp_dir / f"chunk_{bf}B_iter{it + 1}_{tag}.json"
        shutil.copy2(out, dest)
        _append_manifest(
            manifest,
            {
                "ts": tag,
                "b_floor": bf,
                "tier_iter": it + 1,
                "tier_max_iters": max_iters,
                "tier_est": est,
                "cumulative_est": cumulative_est,
                "path": str(dest),
            },
        )
        print(f"[lifers-escalate] checkpoint B≥{bf} -> {dest.name}", flush=True)
        cmd = os.environ.get("LIFERS_POST_CHECKPOINT_CMD", "").strip()
        if cmd:
            env = os.environ.copy()
            env["LIFERS_BRAIN_ROOT"] = str(root)
            env["LIFERS_CHECKPOINT_JSON"] = str(dest)
            env["LIFERS_CHECKPOINT_B"] = str(bf)
            env["LIFERS_CUMULATIVE_EST"] = str(cumulative_est)
            try:
                subprocess.run(cmd, shell=True, cwd=str(root), env=env, timeout=3600, check=False)
            except subprocess.TimeoutExpired:
                print("[lifers-escalate] LIFERS_POST_CHECKPOINT_CMD timed out", flush=True)
    return new_floor


def _load_corpus_from_suite_dir(root: Path, suite: Path) -> str:
    text = []
    if not suite.is_dir():
        return ""
    for p in sorted(suite.glob("*.jsonl")):
        text.append(_load_jsonl_inputs(p))
    return "\n".join(text) + "\n"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    raw_suite = os.environ.get("LIFERS_TRAIN_SUITE_DIR", "").strip()
    suite = Path(raw_suite).expanduser() if raw_suite else (root / "eval" / "suites" / "v001")
    if not suite.is_absolute():
        suite = (root / suite).resolve()
    corpus = _load_corpus_from_suite_dir(root, suite)
    if not corpus.strip():
        fallback = root / "eval" / "suites" / "v001"
        print(
            f"[lifers-escalate] warn: empty corpus at {suite}; falling back to {fallback}",
            flush=True,
        )
        suite = fallback
        corpus = _load_corpus_from_suite_dir(root, suite)
    if not corpus.strip():
        print("[lifers-escalate] fatal: no jsonl corpus", file=sys.stderr, flush=True)
        return 1

    out = root / "weights" / "lifers_transformer.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    ctl_path = control_file_path(out.parent)
    write_default_run(ctl_path)

    unlimited = os.environ.get("LIFERS_ESCALATE_UNLIMITED", "").strip().lower() in ("1", "true", "yes")
    target_b = float(os.environ.get("LIFERS_TARGET_PARAM_B", "20").strip() or "20")
    if unlimited or target_b <= 0:
        target = float("inf")
        target_b_disp = float("inf")
    else:
        target = max(target_b, 0.001) * 1e9
        target_b_disp = target_b

    default_mi = "999999" if unlimited else "48"
    max_iters = int(os.environ.get("LIFERS_RAMP_MAX_ITERS", default_mi).strip() or default_mi)
    max_iters = max(1, min(max_iters, 10_000_000))

    honor_tt, d_model, d_ff, max_vocab, max_seq, steps = _read_ramp_env_hparams()
    _ = honor_tt

    state_path = out.parent / ".lifers_train_state.json"
    cumulative_est, last_b_floor = _load_train_state(state_path)

    resume_on = os.environ.get("LIFERS_ESCALATE_RESUME", "1").strip().lower() not in ("0", "false", "no", "")
    start_it = 0
    warm_path: Path | None = None
    if resume_on and out.is_file():
        inferred = _infer_resume_tier(corpus, out, max_iters)
        if inferred is not None:
            start_it, max_vocab, d_model, d_ff, max_seq, steps = inferred
            warm_path = out
            print(
                f"[lifers-escalate] resume tier iter {start_it + 1}/{max_iters} "
                f"Vcap={max_vocab} D={d_model} F={d_ff} S={max_seq} steps={steps} (warm-start from {out.name})",
                flush=True,
            )
        else:
            print(
                f"[lifers-escalate] note: {out.name} exists but shape does not match current ramp schedule "
                f"(corpus/vocab changed or LIFERS_ESCALATE_* env differs) — cold start from tier 1. "
                f"To ignore file: move it aside or set LIFERS_ESCALATE_RESUME=0.",
                flush=True,
            )
    elif out.is_file() and not resume_on:
        print("[lifers-escalate] LIFERS_ESCALATE_RESUME=0 — cold start (existing weights will be overwritten).", flush=True)
    elif not out.is_file():
        print("[lifers-escalate] no weights file yet — cold start from tier 1.", flush=True)

    tgt_msg = "∞ (no target cap)" if not math.isfinite(target) else f"{target_b_disp}B"
    print(
        f"[lifers-escalate] target≈{tgt_msg} floats, max_iters={max_iters}, "
        f"control={ctl_path} suite={suite} cumulative≈{cumulative_est / 1e9:.4f}B est, start "
        f"V={max_vocab} D={d_model} F={d_ff} S={max_seq} steps={steps}",
        flush=True,
    )
    publish_escalate_snapshot(
        root,
        phase="ramp_ready",
        ramp_max=max_iters,
        cumulative_est_g=cumulative_est / 1e9,
        message="see weights/.train_status.json for live progress",
    )

    max_tier_cap = _escalate_max_tier_display()
    if max_tier_cap > 0:
        print(
            f"[lifers-escalate] LIFERS_ESCALATE_MAX_TIER={max_tier_cap} — will not start training past that tier.",
            flush=True,
        )

    last_err: str | None = None
    for it in range(start_it, max_iters):
        mode = read_train_control(ctl_path)
        if mode == "stop":
            print("[lifers-escalate] control=stop — exiting before next tier.", flush=True)
            break
        if mode == "pause":
            print("[lifers-escalate] control=pause — waiting at tier boundary (set run to continue).", flush=True)
            while read_train_control(ctl_path) == "pause":
                time.sleep(pause_poll_seconds())
            if read_train_control(ctl_path) == "stop":
                print("[lifers-escalate] control=stop while paused — exit.", flush=True)
                break

        tier_display = it + 1
        if max_tier_cap > 0 and tier_display > max_tier_cap:
            print(
                f"[lifers-escalate] stop: tier {tier_display} exceeds LIFERS_ESCALATE_MAX_TIER={max_tier_cap} "
                f"(holds last weights; avoids OOM on small RAM). Add RAM/swap, raise cap, or tune ramp env, then "
                f"lifers_train_ctl.sh run.",
                flush=True,
            )
            break

        est = rough_param_estimate(max_vocab, d_model, d_ff, max_seq)
        print(
            f"[lifers-escalate] iter {it + 1}/{max_iters} est_params≈{est / 1e6:.3f}M  train…",
            flush=True,
        )
        os.environ["LIFERS_TRAIN_STATUS_BRAIN_ROOT"] = str(root)
        os.environ["LIFERS_TRAIN_STATUS_RAMP_ITER"] = str(it + 1)
        os.environ["LIFERS_TRAIN_STATUS_RAMP_MAX"] = str(max_iters)
        os.environ["LIFERS_TRAIN_STATUS_TIER_EST_M"] = str(est / 1e6)
        os.environ["LIFERS_TRAIN_STATUS_MAX_V"] = str(max_vocab)
        os.environ["LIFERS_TRAIN_STATUS_D"] = str(d_model)
        os.environ["LIFERS_TRAIN_STATUS_DFF"] = str(d_ff)
        os.environ["LIFERS_TRAIN_STATUS_MS"] = str(max_seq)
        os.environ["LIFERS_TRAIN_STATUS_STEPS"] = str(steps)
        publish_escalate_snapshot(
            root,
            phase="tier_sgd",
            ramp_iter=it + 1,
            ramp_max=max_iters,
            tier_est_m=est / 1e6,
            max_vocab=max_vocab,
            d_model=d_model,
            d_ff=d_ff,
            max_seq=max_seq,
            steps=steps,
            sgd_step=0,
            sgd_total=steps,
            cumulative_est_g=cumulative_est / 1e9,
            message=f"tier {it + 1}/{max_iters} — SGD running",
        )
        try:
            train_sgd_minimal(
                corpus,
                out_path=out,
                steps=steps,
                max_vocab=max_vocab,
                d_model=d_model,
                d_ff=d_ff,
                max_seq=max_seq,
                lr=float(os.environ.get("TT_LR", "1e-2")),
                warm_start_path=(warm_path if warm_path is not None and it == start_it else None),
                control_path=ctl_path,
            )
            write_progress_line(
                sys.stdout,
                it + 1,
                max_iters,
                prefix=f"[lifers-escalate] ramp done est≈{est / 1e6:.2f}M floats | ",
            )
        except LifersTrainingPause:
            print("[lifers-escalate] paused (control mid-tier); weights saved. Re-run or set control=run.", flush=True)
            end_progress_line(sys.stdout)
            _save_train_state(state_path, cumulative_est, last_b_floor)
            finalize_train_status(root, "paused", "control pause mid-tier")
            return 0
        except LifersTrainingStop:
            print("[lifers-escalate] stopped (control mid-tier); weights saved.", flush=True)
            end_progress_line(sys.stdout)
            _save_train_state(state_path, cumulative_est, last_b_floor)
            finalize_train_status(root, "stopped", "control stop mid-tier")
            return 0
        except MemoryError:
            last_err = "MemoryError"
            print("[lifers-escalate] MemoryError — stopping ramp, keeping last successful weights file.", flush=True)
            pause_mem = os.environ.get("LIFERS_ESCALATE_PAUSE_ON_MEMERROR", "1").strip().lower() not in (
                "0",
                "false",
                "no",
                "off",
            )
            if pause_mem:
                write_train_control(ctl_path, "pause")
                print(
                    "[lifers-escalate] wrote control=pause (LIFERS_ESCALATE_PAUSE_ON_MEMERROR default on) so the "
                    "outer train loop will not instantly restart into the same resume tier. "
                    "Tip: export LIFERS_ESCALATE_MAX_TIER=16 to cap ramp on ~5GiB RAM hosts.",
                    flush=True,
                )
            break
        except Exception as e:
            last_err = str(e)
            print(f"[lifers-escalate] train error: {e}\n{traceback.format_exc()}", flush=True)
            break

        cumulative_est += est
        prev_b_floor = last_b_floor
        last_b_floor = _maybe_checkpoint_by_budget(
            root=root,
            out=out,
            cumulative_est=cumulative_est,
            last_b_floor=last_b_floor,
            it=it,
            est=est,
            max_iters=max_iters,
        )
        _save_train_state(state_path, cumulative_est, last_b_floor)
        pause_on_cp = os.environ.get("LIFERS_PAUSE_ON_CHECKPOINT", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if pause_on_cp and last_b_floor > prev_b_floor:
            write_train_control(ctl_path, "pause")
            print(
                f"[lifers-escalate] LIFERS_PAUSE_ON_CHECKPOINT: wrote pause -> {ctl_path}; "
                "sync/publish weights, then lifers_train_ctl.sh run (or echo run > control).",
                flush=True,
            )

        if math.isfinite(target) and est >= target:
            print(f"[lifers-escalate] reached est≥target ({est / 1e9:.4f}B).", flush=True)
            break

        # Grow architecture / work; bounded growth per step
        grow = 1.35 if (not math.isfinite(target) or est < target * 0.01) else 1.12
        d_model = int(min(4096, max(d_model + 4, int(d_model * grow))))
        d_ff = int(min(12288, max(d_ff, int(d_model * 2.2))))
        max_vocab = int(min(2048, max_vocab + 16))
        max_seq = int(min(192, max_seq + 4))
        steps = int(min(8000, int(steps * 1.08)))

    end_progress_line(sys.stdout)

    ok_weights = out.is_file()
    if ok_weights:
        print(f"[lifers-escalate] wrote {out}", flush=True)
        if last_err:
            print(f"[lifers-escalate] note: {last_err}", flush=True)
    else:
        print("[lifers-escalate] failed: no output weights", file=sys.stderr, flush=True)

    if ok_weights:
        phase_end = "completed"
        msg_end = "ramp finished ok"
        if last_err == "MemoryError":
            phase_end = "oom"
            msg_end = "MemoryError — last successful weights kept"
        elif last_err:
            phase_end = "error"
            msg_end = last_err[:400]
        finalize_train_status(root, phase_end, msg_end)
        return 0
    finalize_train_status(root, "failed", "no output weights")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
