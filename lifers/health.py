"""
Startup health diagnostics for Lifers edge deployment.

Called once at agent initialisation.  Checks critical paths, files, and
environment — reports warnings and errors to stderr so operators see them
without crashing.  Designed to be harmless on every platform (checks are
read-only, no side effects).
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from lifers.constants import (
    CONFIG_STACK,
    MEMORY_DB,
    STACK_SCHEMA,
    WEIGHTS_MARKOV,
    WEIGHTS_MARKOV_FALLBACK,
    WEIGHTS_TRANSFORMER,
)


@dataclass
class HealthIssue:
    severity: str  # "error" | "warn" | "info"
    message: str
    remediation: str = ""


def _check_python() -> List[HealthIssue]:
    issues: List[HealthIssue] = []
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        issues.append(HealthIssue(
            severity="error",
            message=f"Python {v.major}.{v.minor}.{v.micro} — Lifers needs >= 3.10",
            remediation="Install Python 3.10+ from python.org or your package manager",
        ))
    return issues


def _check_dirs(root: Path) -> List[HealthIssue]:
    issues: List[HealthIssue] = []
    required = {"config", "state", "weights", "memory", "logs"}
    for name in required:
        p = root / name
        if not p.is_dir():
            issues.append(HealthIssue(
                severity="warn",
                message=f"Missing directory: {p}",
                remediation=f"mkdir -p {p}",
            ))
    return issues


def _check_stack(root: Path) -> List[HealthIssue]:
    issues: List[HealthIssue] = []
    p = root / CONFIG_STACK
    if not p.is_file():
        issues.append(HealthIssue(
            severity="warn",
            message=f"Stack config missing: {p}",
            remediation="Create config/stack.json (self_heal will create minimal if LIFERS_SELF_HEAL=1)",
        ))
        return issues
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        issues.append(HealthIssue(
            severity="error",
            message=f"Stack config corrupt: {p} — {e}",
            remediation="Restore from backup or delete and let self_heal recreate",
        ))
        return issues

    for key, expected in STACK_SCHEMA.items():
        parts = key.split(".")
        val: Any = data
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part)
            else:
                val = None
                break
        if val is not None and not isinstance(val, expected):
            issues.append(HealthIssue(
                severity="warn",
                message=f"stack.{key}: expected {expected.__name__}, got {type(val).__name__} = {val!r}",
                remediation=f"Fix config/stack.json: set {key} to a {expected.__name__} value",
            ))

    # Check NPC config if present
    npc_list = ((data.get("embodied_world") or {}).get("dynamic_npc") or [])
    if isinstance(npc_list, list):
        for i, item in enumerate(npc_list):
            if not isinstance(item, dict):
                issues.append(HealthIssue(
                    severity="warn",
                    message=f"stack.embodied_world.dynamic_npc[{i}]: expected object, got {type(item).__name__}",
                    remediation=f"Fix entry #{i} in embodied_world.dynamic_npc",
                ))
                continue
            if not item.get("name") or not item.get("persona"):
                issues.append(HealthIssue(
                    severity="warn",
                    message=f"NPC #{i} missing 'name' or 'persona' in dynamic_npc",
                    remediation="Each NPC entry needs at least 'name' and 'persona' fields",
                ))
    elif npc_list:
        issues.append(HealthIssue(
            severity="warn",
            message="stack.embodied_world.dynamic_npc: expected array",
            remediation="Set dynamic_npc to an array of NPC profile objects",
        ))
    return issues


def _check_weights(root: Path) -> List[HealthIssue]:
    issues: List[HealthIssue] = []
    patterns = [WEIGHTS_TRANSFORMER, WEIGHTS_MARKOV, WEIGHTS_MARKOV_FALLBACK]
    found_any = False
    for rel in patterns:
        p = (root / rel).resolve()
        if p.is_file():
            found_any = True
            sz = p.stat().st_size
            if sz == 0:
                issues.append(HealthIssue(
                    severity="error",
                    message=f"Weight file is empty (0 bytes): {rel}",
                    remediation="Re-run training pipeline or sync weights",
                ))
            # Try to validate markov JSON
            if "markov" in rel:
                try:
                    raw = p.read_bytes()
                    data = json.loads(raw[: min(len(raw), 512 * 1024)])
                    if not isinstance(data, dict):
                        issues.append(HealthIssue(
                            severity="warn",
                            message=f"Markov weight file has unexpected top-level type: {type(data).__name__}",
                            remediation="Re-run training pipeline",
                        ))
                except json.JSONDecodeError:
                    issues.append(HealthIssue(
                        severity="warn",
                        message=f"Markov weight file is not valid JSON (may be truncated): {rel}",
                        remediation="Re-run training pipeline or re-sync weights; the file may be corrupt",
                    ))
                except (MemoryError, OSError):
                    pass
            if "transformer" in rel and sz < 1_000_000:
                issues.append(HealthIssue(
                    severity="warn",
                    message=f"Transformer weight file is unusually small ({sz} bytes): {rel}",
                    remediation="The file may be incomplete; re-run training pipeline",
                ))
            issues.append(HealthIssue(
                severity="info",
                message=f"Weight file found: {rel} ({sz // (1024*1024)} MB)",
                remediation="",
            ))
    if not found_any:
        issues.append(HealthIssue(
            severity="error",
            message="No weight files found under weights/",
            remediation="Run scripts/run_pipeline.py to train, or download/symlink weights",
        ))
    return issues


def _check_disk(root: Path) -> List[HealthIssue]:
    issues: List[HealthIssue] = []
    try:
        usage = shutil.disk_usage(root)
        free_mb = usage.free // (1024 * 1024)
        if free_mb < 100:
            issues.append(HealthIssue(
                severity="error",
                message=f"Low disk space: {free_mb} MB free (Lifers needs >= 100 MB for DB + state)",
                remediation="Free disk space or move project to a volume with more space",
            ))
        elif free_mb < 500:
            issues.append(HealthIssue(
                severity="warn",
                message=f"Low disk space: {free_mb} MB free",
                remediation="Monitor disk usage; SQLite and training logs can grow large",
            ))
    except OSError:
        pass  # can't check disk usage on all platforms
    return issues


def _check_training_status(root: Path) -> List[HealthIssue]:
    issues: List[HealthIssue] = []
    p = root / "weights" / ".train_status.json"
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            phase = data.get("phase", "")
            pid = data.get("pid")
            updated = data.get("updated_at", "")
            if phase in ("escalate", "sgd"):
                sgd = data.get("sgd", {})
                pct = sgd.get("pct", 0) if isinstance(sgd, dict) else 0
                issues.append(HealthIssue(
                    severity="warn",
                    message=f"Training in progress (phase={phase}, pid={pid}, sgd={pct}%, updated={updated})",
                    remediation="Wait for training to finish, or stop it before chatting for best results",
                ))
            else:
                issues.append(HealthIssue(
                    severity="info",
                    message=f"Last training phase={phase} (pid={pid}, updated={updated})",
                    remediation="",
                ))
        except (json.JSONDecodeError, OSError):
            pass
    return issues


def _check_memory_db(root: Path) -> List[HealthIssue]:
    issues: List[HealthIssue] = []
    p = root / MEMORY_DB
    if p.is_file():
        sz = p.stat().st_size
        mb = sz // (1024 * 1024)
        if mb > 500:
            issues.append(HealthIssue(
                severity="warn",
                message=f"Long-term memory DB is large: {mb} MB — {p}",
                remediation="Run kb_prune or adjust deep_steward.global_forget thresholds",
            ))
    return issues


def check_health(root: Path) -> List[HealthIssue]:
    """Run all startup checks.  Returns list of issues (empty = healthy)."""
    issues: List[HealthIssue] = []
    try:
        issues.extend(_check_python())
        issues.extend(_check_dirs(root))
        issues.extend(_check_stack(root))
        issues.extend(_check_weights(root))
        issues.extend(_check_training_status(root))
        issues.extend(_check_disk(root))
        issues.extend(_check_memory_db(root))
    except Exception as exc:
        issues.append(HealthIssue(
            severity="error",
            message=f"Health check itself failed: {exc}",
            remediation="Report this as a bug",
        ))
    return issues


def emit_health_report(root: Path) -> int:
    """Run checks, write to stderr, return error count."""
    issues = check_health(root)
    errs = 0
    for iss in issues:
        line = f"LIFERS_HEALTH {iss.severity.upper()} {iss.message}"
        if iss.remediation:
            line += f" | fix: {iss.remediation}"
        sys.stderr.write(line + "\n")
        if iss.severity == "error":
            errs += 1
    if not issues:
        sys.stderr.write("LIFERS_HEALTH INFO All checks passed\n")
    sys.stderr.flush()
    return errs
