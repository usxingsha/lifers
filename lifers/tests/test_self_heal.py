"""Tests for the self_heal module (stack.json repair and default key merge)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from lifers.self_heal import (
    _merge_missing,
    _package_template_stack,
    heal_stack_at_startup,
)


class TestMergeMissing:
    def test_merge_into_empty(self) -> None:
        dst = {}
        changed = _merge_missing(dst, {"a": 1, "b": {"c": 2}})
        assert changed
        assert dst == {"a": 1, "b": {"c": 2}}

    def test_merge_does_not_overwrite(self) -> None:
        dst = {"a": 99, "b": {"c": 1}}
        changed = _merge_missing(dst, {"a": 1, "b": {"c": 2, "d": 3}})
        assert changed  # 'd' is new
        assert dst["a"] == 99  # not overwritten
        assert dst["b"]["c"] == 1  # not overwritten
        assert dst["b"]["d"] == 3  # added

    def test_merge_no_changes(self) -> None:
        dst = {"a": 1, "b": 2}
        changed = _merge_missing(dst, {"a": 1})
        assert not changed

    def test_merge_non_dict_dst(self) -> None:
        assert not _merge_missing("hello", {"a": 1})

    def test_merge_non_dict_src(self) -> None:
        dst = {"a": 1}
        assert not _merge_missing(dst, "hello")


class TestPackageTemplateStack:
    def test_template_path_is_relative_to_project_root(self) -> None:
        tpl = _package_template_stack()
        assert tpl.name == "stack.json"
        assert tpl.parent.name == "config"


class TestHealStackAtStartup:
    def test_skipped_when_env_disabled(self) -> None:
        os.environ["LIFERS_SELF_HEAL"] = "0"
        try:
            report = heal_stack_at_startup(Path("/nonexistent"))
            assert report.get("skipped") is True
        finally:
            os.environ.pop("LIFERS_SELF_HEAL", None)

    def test_creates_minimal_when_no_template(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = heal_stack_at_startup(root)
            stack_file = root / "config" / "stack.json"
            assert stack_file.is_file()
            data = json.loads(stack_file.read_text(encoding="utf-8"))
            assert data.get("version") == 1
            assert "brain" in data
            # May be created_from_template (if project config/stack.json exists)
            # or created_minimal (if not)
            assert report.get("created_minimal") or report.get("created_from_template")

    def test_merges_default_keys_into_existing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config").mkdir(parents=True)
            existing = {"version": 1, "runtime": {"role": "auto"}}
            (root / "config" / "stack.json").write_text(
                json.dumps(existing, ensure_ascii=False), encoding="utf-8"
            )
            report = heal_stack_at_startup(root)
            data = json.loads(
                (root / "config" / "stack.json").read_text(encoding="utf-8")
            )
            # Should have merged embodied_world (from _DEFAULT_STACK_KEYS)
            assert "embodied_world" in data
            assert "dynamic_npc" in data["embodied_world"]
            assert report.get("merged_default_keys") is True

    def test_repairs_corrupt_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config").mkdir(parents=True)
            (root / "config" / "stack.json").write_text(
                "{invalid json}", encoding="utf-8"
            )
            report = heal_stack_at_startup(root)
            assert report.get("corrupt_renamed") or report.get("restored_minimal") or report.get("restored_from_template")
            data = json.loads(
                (root / "config" / "stack.json").read_text(encoding="utf-8")
            )
            assert data.get("version") == 1

    def test_healthy_stack_no_corruption(self) -> None:
        """User values are preserved even when default keys are merged in."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config").mkdir(parents=True)
            healthy = {
                "version": 1,
                "runtime": {"role": "auto"},
                "brain": {"model": "test", "sandbox": True, "session_max_turns": 8, "llm_identity_short": "T", "memory_db": "m.db"},
                "embodied_world": {"dynamic_npc": []},
            }
            (root / "config" / "stack.json").write_text(
                json.dumps(healthy, ensure_ascii=False), encoding="utf-8"
            )
            report = heal_stack_at_startup(root)
            # Should NOT have created/restored anything (file existed)
            assert not report.get("created_minimal")
            assert not report.get("created_from_template")
            assert not report.get("restored_minimal")
            assert not report.get("restored_from_template")
            assert not report.get("corrupt_renamed")
            # May have merged default keys (e.g. remote_infer, self_code) — that's fine
            # Key assertion: user values were NOT overwritten
            data = json.loads(
                (root / "config" / "stack.json").read_text(encoding="utf-8")
            )
            assert data["runtime"]["role"] == "auto"  # user value preserved
            assert data["brain"]["model"] == "test"   # user value preserved
            assert data["brain"]["sandbox"] is True    # user value preserved
