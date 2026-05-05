#!/usr/bin/env python3
"""Merge-update extensions.json for lifers.lifers-agents-ui (keep other extensions)."""
from __future__ import annotations

import json
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: repair_lifers_extensions_index.py <extensions_parent_dir> <version>", file=sys.stderr)
        return 2
    ext_parent = pathlib.Path(sys.argv[1]).expanduser().resolve()
    ver = sys.argv[2].strip()
    bundle = f"lifers.lifers-agents-ui-{ver}"
    target = ext_parent / bundle
    if not target.is_dir():
        print(f"missing {target}", file=sys.stderr)
        return 1
    jf = ext_parent / "extensions.json"
    rows: list = []
    if jf.is_file():
        try:
            rows = json.loads(jf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            rows = []
    if not isinstance(rows, list):
        rows = []
    rows = [x for x in rows if x.get("identifier", {}).get("id") != "lifers.lifers-agents-ui"]
    rows.append(
        {
            "identifier": {"id": "lifers.lifers-agents-ui"},
            "version": ver,
            "location": {"$mid": 1, "path": str(target), "scheme": "file"},
            "relativeLocation": bundle,
        }
    )
    jf.parent.mkdir(parents=True, exist_ok=True)
    jf.write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    obs = ext_parent / ".obsolete"
    if obs.is_file():
        try:
            od = json.loads(obs.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            od = {}
        if isinstance(od, dict):
            od = {k: v for k, v in od.items() if not str(k).startswith("lifers.lifers-agents-ui")}
            obs.write_text(json.dumps(od, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            print("OK", obs)
    print("OK", jf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
