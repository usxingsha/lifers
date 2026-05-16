"""CHAT_QUICK 本地 generate 墙钟秒数：Windows Bridge 默认等。"""
from __future__ import annotations

import os
import unittest
from unittest import mock

import lifers.local_brain as lb_mod


class QuickGenerateWallclockTests(unittest.TestCase):
    def tearDown(self) -> None:
        for k in ("LIFERS_QUICK_GENERATE_TIMEOUT_SEC", "LIFERS_AGENTS_UI_BRIDGE"):
            os.environ.pop(k, None)

    def test_explicit_zero_disables(self) -> None:
        os.environ["LIFERS_QUICK_GENERATE_TIMEOUT_SEC"] = "0"
        os.environ["LIFERS_AGENTS_UI_BRIDGE"] = "1"
        self.assertEqual(lb_mod._quick_generate_wallclock_sec(), 0.0)

    def test_explicit_numeric(self) -> None:
        os.environ["LIFERS_QUICK_GENERATE_TIMEOUT_SEC"] = "42"
        self.assertEqual(lb_mod._quick_generate_wallclock_sec(), 42.0)

    def test_bridge_windows_default(self) -> None:
        os.environ["LIFERS_AGENTS_UI_BRIDGE"] = "1"
        with mock.patch.object(lb_mod.os, "name", "nt"):
            self.assertEqual(lb_mod._quick_generate_wallclock_sec(), 120.0)

    def test_bridge_posix_default(self) -> None:
        os.environ["LIFERS_AGENTS_UI_BRIDGE"] = "1"
        with mock.patch.object(lb_mod.os, "name", "posix"):
            self.assertEqual(lb_mod._quick_generate_wallclock_sec(), 120.0)

    def test_non_bridge_posix(self) -> None:
        with mock.patch.object(lb_mod.os, "name", "posix"):
            self.assertEqual(lb_mod._quick_generate_wallclock_sec(), 120.0)

    def test_non_bridge_non_posix(self) -> None:
        with mock.patch.object(lb_mod.os, "name", "java"):
            self.assertEqual(lb_mod._quick_generate_wallclock_sec(), 0.0)


if __name__ == "__main__":
    unittest.main()
