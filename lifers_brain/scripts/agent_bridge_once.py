"""
One-shot stdin/stdout bridge for the Lifers Agents UI extension.

stdin:  one JSON object UTF-8: {"text": str, "contextFiles": [relative paths optional]}
stdout: one JSON object: {"ok": bool, "text": str, "error": str optional}

Reads optional workspace files (under LIFERS_ROOT only) and prefixes context to the user message.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    root = Path(os.environ.get("LIFERS_ROOT", "").strip() or Path(__file__).resolve().parents[1])
    sys.path.insert(0, str(root))

    # Windows 管道下 sys.stdin/stdout 可能不是 UTF-8；用二进制读写避免 JSON/中文乱码。
    raw_bytes = sys.stdin.buffer.read()
    raw = raw_bytes.decode("utf-8", errors="replace")
    from lifers_brain.bridge_turn import lifers_turn_from_json_body

    out = lifers_turn_from_json_body(root, raw)
    payload = (json.dumps(out, ensure_ascii=False) + "\n").encode("utf-8")
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
