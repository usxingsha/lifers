#!/usr/bin/env python3
"""Readable summary of live training progress from weights/.train_status.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Show Lifers train progress snapshot")
    ap.add_argument(
        "brain_root",
        nargs="?",
        default="",
        help="lifers_brain root (default: parent of scripts/)",
    )
    args = ap.parse_args()
    root = Path(args.brain_root).resolve() if args.brain_root else Path(__file__).resolve().parent.parent
    status_p = root / "weights" / ".train_status.json"
    if not status_p.is_file():
        print(f"(no {status_p} yet — start train_lifers_escalate or check path)")
        return 1
    try:
        data = json.loads(status_p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"read error: {e}")
        return 1

    ts = data.get("updated_at", "?")
    phase = data.get("phase", "?")
    msg = data.get("message", "")
    print(f"updated_at: {ts}")
    print(f"phase: {phase}")
    if msg:
        print(f"message: {msg}")

    ramp = data.get("ramp") or {}
    if ramp:
        print(
            f"ramp: {ramp.get('iter', '?')}/{ramp.get('max', '?')} "
            f"({ramp.get('pct', '?')}%)"
        )

    sgd = data.get("sgd") or {}
    if sgd:
        print(
            f"sgd: step {sgd.get('step', '?')}/{sgd.get('total_steps', '?')} "
            f"({sgd.get('pct', '?')}%)  V={sgd.get('vocab_size', '?')} D={sgd.get('d_model', '?')}"
        )

    if data.get("overall_pct_approx") is not None:
        print(f"overall_pct_approx: {data['overall_pct_approx']}% (ramp × inner SGD rough)")

    te = data.get("tier_est_params_m")
    if te is not None:
        print(f"tier_est_params_m: {te}")

    ce = data.get("cumulative_est_g")
    if ce is not None:
        print(f"cumulative_est_g: {ce} (rough float·G seen so far)")

    arch = data.get("architecture") or {}
    if arch:
        print(
            "architecture: "
            + ", ".join(f"{k}={arch[k]}" for k in sorted(arch.keys()) if arch.get(k) is not None)
        )

    print(f"\n(raw file: {status_p})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
