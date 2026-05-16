from __future__ import annotations

import json
import unittest
from pathlib import Path

from tools.lifers_gui_host.vscodium_settings import gui_theme_from_defaults, load_vscodium_defaults_json, repo_root_from_brain


class GuiHostSettingsTests(unittest.TestCase):
    def test_repo_has_vscodium_defaults(self) -> None:
        brain = Path(__file__).resolve().parents[1]
        repo = repo_root_from_brain(brain)
        raw = load_vscodium_defaults_json(repo)
        self.assertIn("editor.fontSize", raw)
        th = gui_theme_from_defaults(raw)
        self.assertGreater(th["fontSizePx"], 0)
        self.assertIn("fontFamily", th)

    def test_theme_json_serializable(self) -> None:
        brain = Path(__file__).resolve().parents[1]
        repo = repo_root_from_brain(brain)
        raw = load_vscodium_defaults_json(repo)
        json.dumps(gui_theme_from_defaults(raw))


if __name__ == "__main__":
    unittest.main()
