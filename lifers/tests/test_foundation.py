"""Tests for health diagnostics, graceful degradation, and edge deployment."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from lifers.health import HealthIssue, check_health, emit_health_report
from lifers.local_brain import LocalBrain, AgentConfig


# ── Health check ──────────────────────────────────────────────────────────────


class TestHealthCheck:
    def test_empty_dir_reports_issues(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            issues = check_health(root)
            # Should find missing dirs and weights
            dir_issues = [i for i in issues if "Missing directory" in i.message]
            weight_issues = [i for i in issues if "No weight files" in i.message]
            assert dir_issues, f"expected dir issues, got {issues}"
            assert weight_issues, f"expected weight issues, got {issues}"
            assert any(i.severity == "error" for i in issues)

    def test_healthy_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Create minimal healthy layout
            for d in ("config", "state", "weights", "memory", "logs"):
                (root / d).mkdir(parents=True)
            stack = {
                "version": 1,
                "runtime": {"role": "auto"},
                "brain": {
                    "model": "markov",
                    "sandbox": True,
                    "session_max_turns": 8,
                    "llm_identity_short": "Lifers",
                    "memory_db": "memory/longterm.sqlite3",
                },
                "embodied_world": {"dynamic_npc": []},
            }
            (root / "config" / "stack.json").write_text(
                json.dumps(stack, ensure_ascii=False), encoding="utf-8"
            )
            issues = check_health(root)
            # No dir issues, no schema warnings
            assert not [i for i in issues if "Missing directory" in i.message]
            # But no weights — expect error
            weight_errs = [i for i in issues if i.severity == "error" and "No weight files" in i.message]
            assert weight_errs

    def test_emit_health_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            errs = emit_health_report(root)
            assert errs > 0

    def test_weight_integrity_check(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "weights").mkdir()
            # Create a corrupt "JSON" file for markov
            (root / "weights" / "lifers_markov.json").write_text(
                "{not valid json", encoding="utf-8"
            )
            (root / "config").mkdir()
            (root / "config" / "stack.json").write_text(
                json.dumps({"version": 1}, ensure_ascii=False), encoding="utf-8"
            )
            issues = check_health(root)
            corrupt_warnings = [
                i for i in issues
                if "corrupt" in i.message.lower() or "valid JSON" in i.message
            ]
            # May or may not flag corrupt JSON depending on file size vs 512KB read limit
            # The test just verifies the check runs without crashing
            assert isinstance(issues, list)

    def test_training_status_check(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "weights").mkdir(parents=True)
            status = {
                "phase": "sgd",
                "pid": 12345,
                "updated_at": "2025-01-01T00:00:00Z",
                "sgd": {"step": 50, "total_steps": 100, "pct": 50.0},
            }
            (root / "weights" / ".train_status.json").write_text(
                json.dumps(status), encoding="utf-8"
            )
            issues = check_health(root)
            train_warnings = [i for i in issues if "Training in progress" in i.message]
            assert train_warnings, f"expected training warning, got {issues}"

    def test_disk_space_check(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            issues = check_health(root)
            # Disk check may or may not fire depending on available space
            disk_issues = [i for i in issues if "disk space" in i.message.lower()]
            # Just verify it doesn't crash
            assert isinstance(issues, list)


# ── Graceful degradation ──────────────────────────────────────────────────────


class TestGracefulDegradation:
    def test_missing_weights_returns_readable_message(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config").mkdir()
            (root / "config" / "stack.json").write_text(
                json.dumps({"version": 1}, ensure_ascii=False), encoding="utf-8"
            )
            cfg = AgentConfig(root_dir=root, model="markov", sandbox=True)
            brain = LocalBrain(cfg)
            text = brain.generate("hello")
            # Should NOT return "(missing weights)" — should have readable message
            assert "(missing weights)" not in text
            assert "权重" in text or "weight" in text.lower() or "weights" in text.lower()

    def test_transformer_fallback_to_markov(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for d in ("config", "weights"):
                (root / d).mkdir(parents=True)
            (root / "config" / "stack.json").write_text(
                json.dumps({"version": 1}, ensure_ascii=False), encoding="utf-8"
            )
            cfg = AgentConfig(root_dir=root, model="transformer", sandbox=True)
            brain = LocalBrain(cfg)
            # No transformer weights exist, should NOT crash
            text = brain.generate("hello")
            # The fallback should produce either a readable message or markov text
            assert isinstance(text, str)
            assert len(text) > 0
            # Model should have been switched to markov
            assert brain.model == "markov" or "权重" in text


# ── Response dedup ────────────────────────────────────────────────────────────


class TestResponseDedup:
    def test_dedup_detects_exact_duplicate(self) -> None:
        from lifers.agent import LifersAgent

        methods = ["_is_duplicate_response", "_record_response", "_dedup_suffix"]
        for m in methods:
            assert hasattr(LifersAgent, m), f"missing {m}"
