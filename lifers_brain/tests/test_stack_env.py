"""stack_env.load_stack：LIFERS_ROOT 回退与 JSON 容错。"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from lifers_brain.stack_env import load_stack


class StackEnvTests(unittest.TestCase):
    def test_load_stack_prefers_root_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "config"
            cfg.mkdir(parents=True)
            want = {"version": 99, "brain": {"model": "lifers"}}
            (cfg / "stack.json").write_text(json.dumps(want), encoding="utf-8")
            self.assertEqual(load_stack(root).get("version"), 99)

    def test_load_stack_falls_back_to_lifers_root_env(self) -> None:
        with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
            brain = Path(td_a)
            wrong = Path(td_b)
            cfg = brain / "config"
            cfg.mkdir(parents=True)
            want = {"version": 42, "x": True}
            (cfg / "stack.json").write_text(json.dumps(want), encoding="utf-8")
            prev = os.environ.get("LIFERS_ROOT")
            try:
                os.environ["LIFERS_ROOT"] = str(brain)
                got = load_stack(wrong)
                self.assertEqual(got.get("version"), 42)
                self.assertTrue(got.get("x"))
            finally:
                if prev is None:
                    os.environ.pop("LIFERS_ROOT", None)
                else:
                    os.environ["LIFERS_ROOT"] = prev

    def test_load_stack_returns_empty_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            prev = os.environ.get("LIFERS_ROOT")
            try:
                os.environ.pop("LIFERS_ROOT", None)
                self.assertEqual(load_stack(Path(td)), {})
            finally:
                if prev is None:
                    os.environ.pop("LIFERS_ROOT", None)
                else:
                    os.environ["LIFERS_ROOT"] = prev


if __name__ == "__main__":
    unittest.main()
