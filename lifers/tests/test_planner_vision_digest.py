"""Planner inserts vision_digest for image paths under LIFERS_ROOT."""
from __future__ import annotations

import os
import unittest
from pathlib import Path

from lifers.agent import Planner


class PlannerVisionDigestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._prev = os.environ.get("LIFERS_ROOT")

    def tearDown(self) -> None:
        if self._prev is None:
            os.environ.pop("LIFERS_ROOT", None)
        else:
            os.environ["LIFERS_ROOT"] = self._prev

    def test_relative_image_under_root(self) -> None:
        root = Path(__file__).resolve().parents[1]
        os.environ["LIFERS_ROOT"] = str(root)
        rel = "weights/planner_probe.png"
        msg = f"请看 {rel} 里是什么"
        calls = Planner().plan(msg)
        self.assertTrue(any(c.name == "vision_digest" for c in calls))
        vd = [c for c in calls if c.name == "vision_digest"][0]
        self.assertEqual(vd.args.get("rel_path"), rel)

    def test_absolute_image_under_root(self) -> None:
        root = Path(__file__).resolve().parents[1]
        os.environ["LIFERS_ROOT"] = str(root)
        abs_path = str((root / "weights" / "planner_probe.png").resolve())
        msg = f"分析 {abs_path}"
        calls = Planner().plan(msg)
        self.assertTrue(any(c.name == "vision_digest" for c in calls))
        vd = [c for c in calls if c.name == "vision_digest"][0]
        self.assertEqual(vd.args.get("rel_path"), "weights/planner_probe.png")


if __name__ == "__main__":
    unittest.main()
