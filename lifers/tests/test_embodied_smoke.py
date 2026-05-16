"""物化 tick、dynamic_npc 配置块、CLI 烟测（与 Agents 并行路径）。"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class EmbodiedSmokeTests(unittest.TestCase):
    def _root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def test_stack_has_embodied_dynamic_npc_shape(self) -> None:
        root = self._root()
        p = root / "config" / "stack.json"
        self.assertTrue(p.is_file())
        data = json.loads(p.read_text(encoding="utf-8"))
        emb = data.get("embodied_world")
        self.assertIsInstance(emb, dict, msg="stack.json 缺少 embodied_world 对象")
        self.assertIn("enabled", emb)
        self.assertIn("state_relpath", emb)
        dn = emb.get("dynamic_npc")
        self.assertIsInstance(dn, (dict, list),
                              msg="embodied_world.dynamic_npc 应为对象（占位多 NPC）或空列表")
        if isinstance(dn, dict):
            self.assertIn("enabled", dn)

    def test_run_embodied_tick_skipped_when_disabled(self) -> None:
        from lifers.embodied import run_embodied_tick

        root = self._root()
        out = run_embodied_tick(root)
        self.assertTrue(out.get("ok"), msg=out)
        if not (out.get("skipped")):
            self.fail("默认 stack 应关闭 embodied_world.enabled，此处应 skipped")
        self.assertIn("as_of_unix_ms", out)

    def test_embodied_tick_once_script_exits_zero(self) -> None:
        root = self._root()
        script = root / "scripts" / "embodied_tick_once.py"
        self.assertTrue(script.is_file())
        p = subprocess.run(
            [sys.executable, str(script), "--root", str(root)],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        self.assertEqual(p.returncode, 0, msg=p.stderr or p.stdout)
        body = json.loads((p.stdout or "").strip())
        self.assertTrue(body.get("ok"), msg=body)

    def test_embodied_tick_runs_in_isolated_tmpdir(self) -> None:
        """启用 embodied_world 时真实 step + 写 state，不污染仓库默认 state/。"""
        from lifers.embodied import run_embodied_tick

        stack_min = {
            "embodied_world": {
                "enabled": True,
                "dt_sec": 0.05,
                "state_relpath": "state/embodied_isolated_unittest.json",
                "dynamic_npc": {"enabled": False, "note": "unittest sandbox"},
                "vision": {"enabled": False},
                "decision": {"policy": "heuristic_v1"},
            }
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config").mkdir(parents=True)
            (root / "weights").mkdir(parents=True)
            (root / "config" / "stack.json").write_text(json.dumps(stack_min, ensure_ascii=False), encoding="utf-8")
            (root / "weights" / ".train_control").write_text("run\n", encoding="utf-8")

            out1 = run_embodied_tick(root)
            self.assertTrue(out1.get("ok"), msg=out1)
            self.assertFalse(out1.get("skipped"), msg=out1)
            self.assertEqual(out1.get("tick"), 1)
            st = root / "state" / "embodied_isolated_unittest.json"
            self.assertTrue(st.is_file(), msg="tick 后应写入 state_relpath")
            raw = json.loads(st.read_text(encoding="utf-8"))
            self.assertIn("world", raw)
            self.assertEqual(raw.get("control", {}).get("tick"), 1)

            out2 = run_embodied_tick(root)
            self.assertTrue(out2.get("ok"), msg=out2)
            self.assertFalse(out2.get("skipped"), msg=out2)
            self.assertEqual(out2.get("tick"), 2)

            (root / "weights" / ".train_control").write_text("pause\n", encoding="utf-8")
            out_pause = run_embodied_tick(root)
            self.assertTrue(out_pause.get("ok"), msg=out_pause)
            self.assertTrue(out_pause.get("skipped"), msg=out_pause)
            self.assertIn("pause", str(out_pause.get("reason", "")).lower())

    def test_embodied_tick_once_cli_on_isolated_root(self) -> None:
        stack_min = {
            "embodied_world": {
                "enabled": True,
                "dt_sec": 0.05,
                "state_relpath": "state/embodied_cli_isolated.json",
                "dynamic_npc": {"enabled": False},
                "vision": {"enabled": False},
                "decision": {"policy": "heuristic_v1"},
            }
        }
        script = self._root() / "scripts" / "embodied_tick_once.py"
        self.assertTrue(script.is_file())
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config").mkdir(parents=True)
            (root / "weights").mkdir(parents=True)
            (root / "config" / "stack.json").write_text(json.dumps(stack_min, ensure_ascii=False), encoding="utf-8")
            (root / "weights" / ".train_control").write_text("run\n", encoding="utf-8")
            p = subprocess.run(
                [sys.executable, str(script), "--root", str(root)],
                cwd=str(self._root()),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            self.assertEqual(p.returncode, 0, msg=p.stderr or p.stdout)
            body = json.loads((p.stdout or "").strip())
            self.assertTrue(body.get("ok"), msg=body)
            self.assertEqual(body.get("tick"), 1)
            self.assertTrue((root / "state" / "embodied_cli_isolated.json").is_file())


if __name__ == "__main__":
    unittest.main()
