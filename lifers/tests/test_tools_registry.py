"""domains.json 等配置中列出的工具名须落在默认 ToolRegistry 内（防文档与实现漂移）。"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from lifers.tools import build_default_registry


def _domain_tool_names() -> set[str]:
    root = Path(__file__).resolve().parents[1]
    p = root / "config" / "domains.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    out: set[str] = set()
    for d in data.get("domains", []):
        tools = d.get("tools")
        if not isinstance(tools, list):
            continue
        for t in tools:
            if isinstance(t, str) and t and not t.startswith("（"):
                out.add(t)
    return out


def _organ_tool_names() -> set[str]:
    root = Path(__file__).resolve().parents[1]
    p = root / "config" / "organ_capabilities.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    out: set[str] = set()
    for o in data.get("organs", []):
        tools = o.get("tools")
        if not isinstance(tools, list):
            continue
        for t in tools:
            if not isinstance(t, str) or not t:
                continue
            if t.startswith("（"):
                continue
            out.add(t)
    return out


class ToolsRegistryTests(unittest.TestCase):
    def test_default_registry_count(self) -> None:
        reg = build_default_registry()
        names = {s.name for s in reg.list_specs()}
        self.assertEqual(len(names), 19, msg=sorted(names))

    def test_domains_tools_subset_of_registry(self) -> None:
        reg = build_default_registry()
        registered = {s.name for s in reg.list_specs()}
        unknown = _domain_tool_names() - registered
        self.assertFalse(unknown, msg=f"domains.json references unknown tools: {sorted(unknown)}")

    def test_organ_tools_subset_of_registry(self) -> None:
        reg = build_default_registry()
        registered = {s.name for s in reg.list_specs()}
        unknown = _organ_tool_names() - registered
        self.assertFalse(unknown, msg=f"organ_capabilities unknown tools: {sorted(unknown)}")


if __name__ == "__main__":
    unittest.main()
