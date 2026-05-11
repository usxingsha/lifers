"""realtime_anchor：地名配置与栈读取。"""
from __future__ import annotations

import unittest

from lifers_brain.realtime_anchor import geo_context_line


class RealtimeAnchorTests(unittest.TestCase):
    def test_geo_label_from_stack(self) -> None:
        s = {"brain": {"geo_label": "  上海, CN  "}}
        line = geo_context_line(s)
        self.assertIn("上海", line)
        self.assertIn("【实时·定位】", line)

    def test_geo_label_env_overrides_empty_stack(self) -> None:
        import os

        prev = os.environ.get("LIFERS_GEO_LABEL")
        try:
            os.environ["LIFERS_GEO_LABEL"] = "Tokyo"
            line = geo_context_line({"brain": {}})
            self.assertIn("Tokyo", line)
        finally:
            if prev is None:
                os.environ.pop("LIFERS_GEO_LABEL", None)
            else:
                os.environ["LIFERS_GEO_LABEL"] = prev


if __name__ == "__main__":
    unittest.main()
