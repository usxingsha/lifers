"""
可选：从 VSCodium 上游拉取少量公开元数据（product.json 等）到 state/vscodium_upstream/，
便于自有 GUI 与发行版溯源对齐；不替代 tools/vscodium_editor_defaults.json（仍为本仓主配置源）。
需网络；CI 勿依赖。
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


URLS = {
    "vscodium_product.json": "https://raw.githubusercontent.com/VSCodium/vscodium/master/product.json",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--brain-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    out_dir = (args.brain_root / "state" / "vscodium_upstream").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        req = urllib.request.Request(url, headers={"User-Agent": "LifersBrain/fetch-upstream"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
        except Exception as e:
            print(json.dumps({"ok": False, "file": name, "error": str(e)}, ensure_ascii=False), flush=True)
            return 1
        (out_dir / name).write_bytes(data)
        print(json.dumps({"ok": True, "wrote": str(out_dir / name)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
