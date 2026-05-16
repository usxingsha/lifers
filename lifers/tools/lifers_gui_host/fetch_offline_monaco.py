#!/usr/bin/env python3
"""
将 monaco-editor npm 包中的 min/vs 解压到 static/vendor/monaco/min/vs，
供自研 GUI 离线加载（app.js 优先 /static/vendor/...，缺失则回退 jsDelivr）。

用法（在 lifers 目录，需联网一次）:
  PYTHONPATH=. python tools/lifers_gui_host/fetch_offline_monaco.py

版本号须与 static/app.js 中 MONACO_CDN 的 npm 版本一致（当前 0.52.0）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def _here() -> Path:
    return Path(__file__).resolve().parent


def _download(url: str, dest: Path, chunk: int = 256 * 1024) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "LifersBrain/monaco-vendor"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as out:
        shutil.copyfileobj(r, out, length=chunk)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="0.52.0", help="monaco-editor npm version")
    ap.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="解压目标（默认 tools/lifers_gui_host/static/vendor/monaco/min/vs）",
    )
    args = ap.parse_args()
    base = _here()
    dest_vs = args.dest or (base / "static" / "vendor" / "monaco" / "min" / "vs")
    dest_vs = dest_vs.resolve()

    meta_url = f"https://registry.npmjs.org/monaco-editor/{args.version}"
    print(f"fetch meta {meta_url}", flush=True)
    with urllib.request.urlopen(meta_url, timeout=60) as r:
        meta = json.loads(r.read().decode("utf-8"))
    tarball = meta.get("dist", {}).get("tarball")
    if not tarball:
        print("no dist.tarball in npm meta", file=sys.stderr)
        return 2

    tmpdir = tempfile.mkdtemp(prefix="monaco_dl_")
    tgz = Path(tmpdir) / "monaco.tgz"
    try:
        print(f"download {tarball}", flush=True)
        _download(tarball, tgz)
        prefix = "package/min/vs/"
        dest_vs.parent.mkdir(parents=True, exist_ok=True)
        if dest_vs.exists():
            shutil.rmtree(dest_vs)
        dest_vs.mkdir(parents=True, exist_ok=True)
        print(f"extract {prefix}* -> {dest_vs}", flush=True)
        with tarfile.open(tgz, "r:gz") as tar:
            for m in tar.getmembers():
                if not m.isfile():
                    continue
                if not m.name.startswith(prefix):
                    continue
                rel = m.name[len(prefix) :]
                if not rel or rel.endswith("/"):
                    continue
                out = dest_vs / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                bio = tar.extractfile(m)
                if bio is None:
                    continue
                out.write_bytes(bio.read())
        loader = dest_vs / "loader.js"
        if not loader.is_file():
            print(f"missing {loader}", file=sys.stderr)
            return 3
        print(json.dumps({"ok": True, "vs": str(dest_vs), "bytes": sum(f.stat().st_size for f in dest_vs.rglob("*") if f.is_file())}, ensure_ascii=False), flush=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
