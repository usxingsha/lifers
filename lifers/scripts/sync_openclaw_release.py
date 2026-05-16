#!/usr/bin/env python3
"""
对照上游 GitHub latest release，更新 config/stack.json 的 openclaw.compat_ref（本地锚点镜像）。

不安装、不调用 OpenClaw；仅保持与上游发行标签同步，供 AI 边界与 drift 校验。
可选写入 config/openclaw_manifest.json 的 last_synced_tag。

用法:
  python scripts/sync_openclaw_release.py
  python scripts/sync_openclaw_release.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from lifers.openclaw_compat import UPSTREAM_REPO, fetch_latest_release_tag

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tag = fetch_latest_release_tag(20.0)
    if not tag:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "could not fetch GitHub latest tag (network or set LIFERS_HTTP_DIRECT=1 if proxy returns 10061)",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    tag = str(tag).strip()
    if not tag:
        print(json.dumps({"ok": False, "error": "no tag_name"}, ensure_ascii=False, indent=2))
        return 1

    compat_ref = f"{UPSTREAM_REPO}@{tag}"
    stack_path = root / "config" / "stack.json"
    if not stack_path.is_file():
        print(json.dumps({"ok": False, "error": f"missing {stack_path}"}, ensure_ascii=False, indent=2))
        return 1

    stack = json.loads(stack_path.read_text(encoding="utf-8"))
    oc = stack.get("openclaw")
    if not isinstance(oc, dict):
        oc = {}
        stack["openclaw"] = oc
    old = oc.get("compat_ref")
    oc["compat_ref"] = compat_ref

    man_path = root / "config" / "openclaw_manifest.json"
    man_updated = False
    if man_path.is_file():
        try:
            man = json.loads(man_path.read_text(encoding="utf-8"))
            man["last_synced_tag"] = tag
            man["last_synced_at_ms"] = int(__import__("time").time() * 1000)
            if not args.dry_run:
                man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            man_updated = True
        except Exception:
            pass

    out = {
        "ok": True,
        "tag": tag,
        "compat_ref": compat_ref,
        "previous_compat_ref": old,
        "dry_run": args.dry_run,
        "manifest_updated": man_updated,
    }

    if args.dry_run:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    stack_path.write_text(json.dumps(stack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
