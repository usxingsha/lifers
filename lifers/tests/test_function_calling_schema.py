"""Tests for lifers.function_calling_schema."""
from __future__ import annotations

import unittest

from lifers.function_calling_schema import (
    extract_json_values,
    parse_tool_invocation,
    tool_spec_to_openai_function,
    validate_object,
)
from lifers.tools import ToolSpec


class FunctionCallingSchemaTests(unittest.TestCase):
    def test_validate_object_ok(self) -> None:
        schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
        ok, errs = validate_object({"q": "hi"}, schema)
        self.assertTrue(ok)
        self.assertEqual(errs, [])

    def test_validate_object_missing_required(self) -> None:
        schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
        ok, errs = validate_object({}, schema)
        self.assertFalse(ok)
        self.assertTrue(any("missing required" in e for e in errs))

    def test_tool_spec_to_openai_function(self) -> None:
        spec = ToolSpec(
            name="web_search",
            args_schema={"query": "string"},
            permissions=["network:outbound"],
            supports_modes=("dry_run", "execute", "verify", "rollback"),
            risk_level="low",
        )
        fn = tool_spec_to_openai_function(spec)
        self.assertEqual(fn["type"], "function")
        self.assertEqual(fn["function"]["name"], "web_search")
        self.assertIn("parameters", fn["function"])

    def test_extract_json_values_fence(self) -> None:
        text = 'prefix\n```json\n{"a": 1}\n```\n'
        vals = extract_json_values(text)
        self.assertEqual(vals, [{"a": 1}])

    def test_parse_tool_invocation(self) -> None:
        p = parse_tool_invocation({"tool": "fs_read", "arguments": {"path": "/x"}})
        self.assertIsNotNone(p)
        assert p is not None
        self.assertEqual(p[0], "fs_read")
        self.assertEqual(p[1]["path"], "/x")


if __name__ == "__main__":
    unittest.main()
