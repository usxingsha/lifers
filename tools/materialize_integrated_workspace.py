#!/usr/bin/env python3
"""Read config/integrated_layout.json (portable root next to tools/) and write VS Code multi-root workspace files.

Optional config/workspace_custom.json:
- Top-level keys except `folders` and `_*` merge into workspace settings (deep-merge for dicts).
- If `folders` is a non-empty array of {name, path}, it replaces roots from integrated_layout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def _strip_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if not str(k).startswith("_") and str(k) != "comment"}


def main() -> int:
    rs_root = Path(__file__).resolve().parent.parent
    layout_path = rs_root / "config" / "integrated_layout.json"
    if not layout_path.is_file():
        print(f"missing {layout_path}", file=sys.stderr)
        return 1
    data = json.loads(layout_path.read_text(encoding="utf-8"))

    roots = data.get("roots") or []
    folders: list[dict[str, str]] = []
    for r in roots:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name", r.get("id", "folder"))).strip()
        rel = str(r.get("path", ".")).strip()
        folders.append({"name": name, "path": rel})
    # integrated_layout.json 历史上用 workspaces[] 而未有 roots[] 时，避免写出空 folders。
    if not folders:
        for w in data.get("workspaces") or []:
            if not isinstance(w, dict):
                continue
            name = str(w.get("id") or w.get("name") or "folder").strip()
            rel = str(w.get("path", ".")).strip()
            folders.append({"name": name, "path": rel})

    raw_ws = data.get("workspace_settings") or {}
    settings: Dict[str, Any] = {}
    if isinstance(raw_ws, dict):
        settings = _strip_meta(dict(raw_ws))

    custom_rel = str(data.get("workspace_custom_file") or "config/workspace_custom.json").strip()
    custom_path = rs_root / custom_rel.replace("\\", "/")
    if custom_path.is_file():
        try:
            custom = json.loads(custom_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"invalid JSON {custom_path}: {e}", file=sys.stderr)
            return 1
        if not isinstance(custom, dict):
            print(f"{custom_path} must be a JSON object", file=sys.stderr)
            return 1

        cf = custom.get("folders")
        if isinstance(cf, list) and len(cf) > 0:
            folders = []
            for item in cf:
                if not isinstance(item, dict):
                    continue
                nm = str(item.get("name", "folder")).strip()
                pt = str(item.get("path", ".")).strip()
                folders.append({"name": nm, "path": pt})

        for k, v in custom.items():
            if k == "folders" or str(k).startswith("_"):
                continue
            if isinstance(v, dict) and isinstance(settings.get(k), dict):
                settings[k] = _deep_merge(settings[k], v)
            else:
                settings[k] = v

        print(f"merged {custom_path}", file=sys.stderr)

    out = {"folders": folders, "settings": settings}
    body = json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    for fn in data.get("workspace_outputs") or ["lifers.code-workspace"]:
        p = rs_root / str(fn).strip()
        if not p.name.endswith(".code-workspace"):
            continue
        p.write_text(body, encoding="utf-8")
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
