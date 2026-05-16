#!/usr/bin/env python3
"""
Daily Lifers AI health: logic/coherence (eval suite), full_system_check, bridge latency,
context throughput vs LIFERS_CONTEXT_MAX_FILES, optional auto-remediation into local workspace_custom.json.

Run from repo (portable root or lifers):
  cd lifers && python eval/daily_ai_health.py

Environment:
  LIFERS_DAILY_HEALTH_AUTO_REMEDIATE=1  — bump lifers.context* in ../config/workspace_custom.json (gitignored) when safe
  LIFERS_FULL_CHECK_BRIDGE=0             — faster pass (skip agent_bridge in full_system_check subprocess)
  MODEL=markov|transformer              — same as rest of stack
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


BRAIN_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BRAIN_ROOT.parent
EVAL_DIR = BRAIN_ROOT / "eval"
STATE_DIR = BRAIN_ROOT / "state" / "daily_health"
PROBE_DIR_REL = "state/daily_health_ctx_probe"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def effective_workspace_limits() -> Dict[str, int]:
    """Effective lifers.context* after integrated_layout + optional workspace_custom."""
    layout_path = REPO_ROOT / "config/integrated_layout.json"
    settings: Dict[str, Any] = {}
    if layout_path.is_file():
        layout = _load_json(layout_path)
        ws = layout.get("workspace_settings") or {}
        if isinstance(ws, dict):
            settings = {k: v for k, v in ws.items() if not str(k).startswith("_")}
    custom_rel = "config/workspace_custom.json"
    if layout_path.is_file():
        try:
            layout = _load_json(layout_path)
            custom_rel = str(layout.get("workspace_custom_file") or custom_rel).strip()
        except Exception:
            pass
    custom_path = REPO_ROOT / custom_rel.replace("\\", "/")
    if custom_path.is_file():
        try:
            custom = _load_json(custom_path)
            if isinstance(custom, dict):
                for k, v in custom.items():
                    if k == "folders" or str(k).startswith("_"):
                        continue
                    if isinstance(v, dict) and isinstance(settings.get(k), dict):
                        settings[k] = {**settings[k], **v}
                    else:
                        settings[k] = v
        except Exception:
            pass
    def _iget(key: str, default: int) -> int:
        raw = settings.get(key, default)
        try:
            return int(raw)
        except Exception:
            return default

    return {
        "lifers.contextMaxFiles": _iget("lifers.contextMaxFiles", 48),
        "lifers.bridgeContextMaxFiles": _iget("lifers.bridgeContextMaxFiles", 32),
        "lifers.bridgeTimeoutMs": _iget("lifers.bridgeTimeoutMs", 900000),
    }


def run_full_system_check_subprocess() -> Dict[str, Any]:
    env = os.environ.copy()
    env["LIFERS_ROOT"] = str(BRAIN_ROOT)
    env.setdefault("MODEL", os.environ.get("MODEL", "markov"))
    env.setdefault("SANDBOX", "1")
    t0 = time.perf_counter()
    cmd = [sys.executable, str(EVAL_DIR / "full_system_check.py")]
    p = subprocess.run(
        cmd,
        cwd=str(BRAIN_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    elapsed = time.perf_counter() - t0
    ok = p.returncode == 0 and "ALL CHECKS PASSED" in (p.stdout or "")
    return {
        "ok": ok,
        "returncode": p.returncode,
        "elapsed_s": round(elapsed, 3),
        "stdout_tail": (p.stdout or "")[-1200:],
        "stderr_tail": (p.stderr or "")[-1200:],
    }


def run_eval_report() -> Dict[str, Any]:
    sys.path.insert(0, str(BRAIN_ROOT))
    sys.path.insert(0, str(EVAL_DIR))
    os.environ.setdefault("LIFERS_ROOT", str(BRAIN_ROOT))
    from run_eval import run_suite

    suite_dir = EVAL_DIR / "suites" / "v001"
    try:
        report = run_suite(suite_dir)
        report["ok"] = float(report.get("pass_rate", 0)) >= 0.75
        return report
    except Exception as e:
        return {"ok": False, "error": str(e), "pass_rate": 0.0, "cases": 0, "passed": 0}


def bridge_roundtrip_ms(text: str, context_files: List[str], runs: int = 5) -> Dict[str, Any]:
    env = os.environ.copy()
    env["LIFERS_ROOT"] = str(BRAIN_ROOT)
    env.setdefault("MODEL", os.environ.get("MODEL", "markov"))
    env.setdefault("SANDBOX", "1")
    env.setdefault("LIFERS_TASKFLOW_LEARN", "0")
    cmd = [sys.executable, str(BRAIN_ROOT / "scripts" / "agent_bridge_once.py")]
    times: List[float] = []
    last_out: Dict[str, Any] = {}
    for _ in range(max(1, runs)):
        body = json.dumps({"text": text, "contextFiles": context_files}, ensure_ascii=False)
        t0 = time.perf_counter()
        p = subprocess.run(
            cmd,
            input=body,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(BRAIN_ROOT),
            timeout=240,
        )
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)
        try:
            last_out = json.loads((p.stdout or "").strip() or "{}")
        except json.JSONDecodeError:
            last_out = {"ok": False, "parse_error": True}
        if p.returncode != 0 or not last_out.get("ok"):
            break
    return {
        "median_ms": round(statistics.median(times), 2) if times else None,
        "min_ms": round(min(times), 2) if times else None,
        "max_ms": round(max(times), 2) if times else None,
        "last_ok": bool(last_out.get("ok")),
        "runs": len(times),
    }


def prepare_probe_files(n_files: int = 96) -> List[str]:
    base = BRAIN_ROOT / PROBE_DIR_REL
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    rels: List[str] = []
    for i in range(n_files):
        rel = f"{PROBE_DIR_REL}/f{i:04d}.txt"
        (BRAIN_ROOT / rel).write_text(f"probe-{i}\n" + ("x" * 80) + "\n", encoding="utf-8")
        rels.append(rel.replace("\\", "/"))
    return rels


def probe_context_throughput(all_rels: List[str]) -> Dict[str, Any]:
    """Measure bridge with increasing effective max context files (env LIFERS_CONTEXT_MAX_FILES)."""
    rows = []
    limits_to_try = [8, 16, 32, 48, 64, 96]
    for lim in limits_to_try:
        if lim > len(all_rels):
            continue
        os.environ["LIFERS_CONTEXT_MAX_FILES"] = str(lim)
        # Use first `lim` files so payload scales with limit
        subset = all_rels[:lim]
        r = bridge_roundtrip_ms("用一句话列出上文里出现的 probe 编号范围。", subset, runs=2)
        rows.append({"max_files_env": lim, **r})
    # restore
    os.environ.pop("LIFERS_CONTEXT_MAX_FILES", None)
    worst_median = max((x["median_ms"] or 0) for x in rows) if rows else 0
    return {"matrix": rows, "worst_median_ms": round(worst_median, 2)}


def maybe_remediate(report: Dict[str, Any], limits_before: Dict[str, int]) -> Dict[str, Any]:
    if os.environ.get("LIFERS_DAILY_HEALTH_AUTO_REMEDIATE", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return {"applied": False, "reason": "LIFERS_DAILY_HEALTH_AUTO_REMEDIATE not set"}

    if not report.get("full_system_check", {}).get("ok"):
        return {"applied": False, "reason": "full_system_check failed — no auto changes"}

    eval_ok = float(report.get("eval_suite", {}).get("pass_rate", 0)) >= 0.85
    if not eval_ok:
        return {"applied": False, "reason": "eval pass_rate below 0.85 — review before raising limits"}

    bridge_info = report.get("bridge_speed_short", {}) or {}
    median_ms = bridge_info.get("median_ms") or 0
    ctx_worst = float(report.get("context_probe", {}).get("worst_median_ms") or 0)

    caps = {"lifers.contextMaxFiles": 192, "lifers.bridgeContextMaxFiles": 160}

    updates: Dict[str, Any] = {}
    cur_ctx = limits_before["lifers.contextMaxFiles"]
    cur_bridge = limits_before["lifers.bridgeContextMaxFiles"]
    cur_to = limits_before["lifers.bridgeTimeoutMs"]

    # Slow bridge with heavy context: extend timeout first (safe UX fix)
    if ctx_worst > 45000 or median_ms > 45000:
        new_to = min(max(cur_to, 900000), 3600000)
        if new_to > cur_to:
            updates["lifers.bridgeTimeoutMs"] = new_to

    # High latency under load but checks pass: modestly raise file caps for editor-side UX (matches bridge_turn env on Kali via manual sync)
    if ctx_worst > 8000 and ctx_worst < 120000:
        if cur_ctx < caps["lifers.contextMaxFiles"]:
            updates["lifers.contextMaxFiles"] = min(cur_ctx + 16, caps["lifers.contextMaxFiles"])
        if cur_bridge < caps["lifers.bridgeContextMaxFiles"]:
            updates["lifers.bridgeContextMaxFiles"] = min(cur_bridge + 16, caps["lifers.bridgeContextMaxFiles"])

    if not updates:
        return {"applied": False, "reason": "no remediation thresholds matched"}

    custom_path = REPO_ROOT / "config/workspace_custom.json"
    custom_path.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if custom_path.is_file():
        try:
            existing = json.loads(custom_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing["_daily_health"] = {
        "updated": _utc_stamp(),
        "note": "merged by lifers/eval/daily_ai_health.py",
    }
    existing.update(updates)
    custom_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    mat = REPO_ROOT / "tools" / "materialize_integrated_workspace.py"
    mat_note = ""
    if mat.is_file():
        mp = subprocess.run(
            [sys.executable, str(mat)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        mat_note = (mp.stderr or "")[-400:]
        if mp.returncode != 0:
            return {"applied": True, "updates": updates, "materialize_rc": mp.returncode, "materialize_err": mat_note}

    return {"applied": True, "updates": updates, "path": str(custom_path), "materialize_stderr_tail": mat_note}


def main() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    limits_before = effective_workspace_limits()

    summary: Dict[str, Any] = {
        "stamp": stamp,
        "brain_root": str(BRAIN_ROOT),
        "repo_root": str(REPO_ROOT),
        "effective_workspace_limits_before": limits_before,
        "model": os.environ.get("MODEL", "markov"),
    }

    summary["eval_suite"] = run_eval_report()

    summary["full_system_check"] = run_full_system_check_subprocess()

    summary["bridge_speed_short"] = bridge_roundtrip_ms("仅回复：ok。", [], runs=6)

    all_rels = prepare_probe_files(96)
    summary["context_probe"] = probe_context_throughput(all_rels)

    # Aggregate scores for dashboards
    er = summary["eval_suite"]
    pass_rate = float(er.get("pass_rate", 0))
    fsc_ok = bool(summary["full_system_check"].get("ok"))
    med = summary["bridge_speed_short"].get("median_ms")
    summary["scores"] = {
        "logic_coherence_pass_rate": round(pass_rate, 4),
        "pipeline_integrity": 1.0 if fsc_ok else 0.0,
        "bridge_latency_median_ms": med,
        "context_worst_case_median_ms": summary["context_probe"].get("worst_median_ms"),
        "intelligence_proxy": round(0.55 * pass_rate + 0.35 * (1.0 if fsc_ok else 0.0) + 0.10 * min(1.0, 120000 / max(1.0, float(med or 1))), 4),
    }

    summary["remediation"] = maybe_remediate(summary, limits_before)

    out_path = STATE_DIR / f"report_{stamp}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = STATE_DIR / "latest.json"
    latest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out_path), "latest": str(latest), "scores": summary["scores"]}, ensure_ascii=False))
    return 0 if summary["full_system_check"]["ok"] and er.get("pass_rate", 0) >= 0.75 else 2


if __name__ == "__main__":
    raise SystemExit(main())
