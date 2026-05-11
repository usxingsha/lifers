"""lifers_ai_pipeline.json：结构完整、阶段齐全。"""
from __future__ import annotations

import json
import unittest
from pathlib import Path


class AiPipelineJsonTests(unittest.TestCase):
    def test_pipeline_has_core_stages(self) -> None:
        root = Path(__file__).resolve().parents[1]
        p = root / "config" / "lifers_ai_pipeline.json"
        self.assertTrue(p.is_file(), msg=f"missing {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        stages = data.get("stages")
        self.assertIsInstance(stages, list)
        ids = {s.get("id") for s in stages if isinstance(s, dict)}
        for need in (
            "input",
            "semantic",
            "inference",
            "output",
            "agent_runtime",
            "edge",
            "embodied_npc",
        ):
            self.assertIn(need, ids, msg=f"missing stage {need}")
        self.assertIn("config/lifers_llm_bootstrap.json", data.get("related_maps", []))


if __name__ == "__main__":
    unittest.main()
