"""fix_inference_prompt：长 SYSTEM + USER/ASSISTANT prompt 裁剪。"""
from __future__ import annotations

import unittest

from lifers.fix_inference_prompt import clip_prompt_for_transformer, patch_agent_generate_call


class FixInferencePromptTests(unittest.TestCase):
    def test_short_unchanged(self) -> None:
        s = "SYSTEM:\nhi\nUSER:\nok\nASSISTANT:\n"
        self.assertEqual(clip_prompt_for_transformer(s, 128), s)

    def test_long_keeps_user_assistant(self) -> None:
        head = "SYSTEM:\n" + ("x" * 500) + "\n"
        tail = "USER:\nquestion here\nASSISTANT:\n"
        p = head + tail
        out = clip_prompt_for_transformer(p, 120)
        self.assertLessEqual(len(out), 120)
        self.assertIn("USER:", out)
        self.assertIn("ASSISTANT:", out)

    def test_patch_alias(self) -> None:
        self.assertEqual(patch_agent_generate_call("a", 99), "a")


if __name__ == "__main__":
    unittest.main()
