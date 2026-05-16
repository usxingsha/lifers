#!/usr/bin/env python3
"""
Lifers 聚合自检：unittest（含 tests/test_inference_comprehensive.py）+ GUI 宿主 HTTP 烟测 + eval/full_system_check.py（可选关网桥）。

在 lifers 目录：  PYTHONPATH=. python scripts/lifers_run_all_checks.py

含：unittest、边缘路由烟测、LLM 就绪、full_system_check（强制 LIFERS_FULL_CHECK_BRIDGE=1 跑 agent_bridge_once）、embodied_tick_once。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    prev_pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(root) + (os.pathsep + prev_pp if prev_pp else "")

    print("== unittest discover (tests/test_*.py) ==", flush=True)
    u = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
        cwd=str(root),
        env=os.environ.copy(),
    )
    if u.returncode != 0:
        return u.returncode

    print("== scripts/verify_edge_agent_pipeline.py ==", flush=True)
    v = subprocess.run([sys.executable, str(root / "scripts" / "verify_edge_agent_pipeline.py")], cwd=str(root), env=os.environ.copy())
    if v.returncode != 0:
        return int(v.returncode)

    print("== scripts/check_lifers_llm_ready.py ==", flush=True)
    h = subprocess.run([sys.executable, str(root / "scripts" / "check_lifers_llm_ready.py")], cwd=str(root), env=os.environ.copy())
    if h.returncode != 0:
        return int(h.returncode)

    print("== eval/full_system_check.py ==", flush=True)
    env = os.environ.copy()
    env.setdefault("LIFERS_FULL_CHECK_NETWORK", "0")
    # 父 shell 若曾 export LIFERS_FULL_CHECK_BRIDGE=0，setdefault 不会覆盖；聚合自检在此显式打开桥接烟测。
    env["LIFERS_FULL_CHECK_BRIDGE"] = "1"
    f = subprocess.run([sys.executable, str(root / "eval" / "full_system_check.py")], cwd=str(root), env=env)
    if f.returncode != 0:
        return int(f.returncode)

    print("== scripts/embodied_tick_once.py ==", flush=True)
    e = subprocess.run([sys.executable, str(root / "scripts" / "embodied_tick_once.py")], cwd=str(root), env=os.environ.copy())
    return int(e.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
