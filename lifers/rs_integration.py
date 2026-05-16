"""
rs 侧「整合」运行时：与 junction 并列，表达「纳入本树后改编自用」的流程。

- 读 rs/config/integrated_layout.json 的 bootstrap 段；
- 在 agent / pipeline 入口 apply_stack_env 时自动执行（可关）；
- 物化多根 .code-workspace；按清单安全删除过时路径（仅 remove_if_present 白名单）。

不把外部仓库密钥或云端模型写进本树；大宗目录复制请用手工或自建脚本，此处只做引导与配置产物同步。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _brain_to_rs_root(brain_root: Path) -> Path:
    return brain_root.parent


def _safe_path_under(rs_root: Path, rel: str) -> Optional[Path]:
    rel = rel.strip().replace("\\", "/")
    if not rel or rel.startswith(("/", "\\")):
        return None
    parts = Path(rel).parts
    if ".." in parts:
        return None
    cand = (rs_root / rel).resolve()
    rs_r = rs_root.resolve()
    try:
        cand.relative_to(rs_r)
    except ValueError:
        return None
    return cand


def _materialize_needed(rs_root: Path, layout_path: Path, data: Dict[str, Any]) -> bool:
    boot = data.get("bootstrap") or {}
    ref_name = str(boot.get("materialize_if_layout_newer_than") or "").strip()
    outs = data.get("workspace_outputs") or []
    if not ref_name and outs:
        ref_name = str(outs[0]).strip()
    if not ref_name:
        return True
    ref = rs_root / ref_name
    if not ref.is_file():
        return True
    try:
        return layout_path.stat().st_mtime > ref.stat().st_mtime
    except OSError:
        return True


def run_rs_integration_bootstrap(brain_root: Path) -> Dict[str, Any]:
    """
    在 lifers 根目录已解析的前提下调用。失败不抛异常到上层（由调用方决定是否吞掉）。
    """
    rs_root = _brain_to_rs_root(brain_root)
    layout_path = rs_root / "config" / "integrated_layout.json"
    out: Dict[str, Any] = {"ok": True, "actions": []}
    if not layout_path.is_file():
        return out
    try:
        data = json.loads(layout_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {**out, "ok": False, "error": "integrated_layout.json invalid JSON"}

    boot = data.get("bootstrap") or {}
    if boot.get("on_agent_startup") is False:
        return out

    steps: List[str] = list(boot.get("steps") or ["materialize_workspace", "prune_paths"])
    verbose = os.environ.get("LIFERS_INTEGRATION_VERBOSE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    if "materialize_workspace" in steps:
        script = rs_root / "tools" / "materialize_integrated_workspace.py"
        if script.is_file() and _materialize_needed(rs_root, layout_path, data):
            try:
                r = subprocess.run(
                    [sys.executable, str(script)],
                    cwd=str(rs_root),
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                out["actions"].append("materialize_workspace")
                if verbose and r.stdout:
                    print(r.stdout, end="", file=sys.stderr)
            except subprocess.CalledProcessError as e:
                out["ok"] = False
                err = (e.stderr or e.stdout or str(e)) or ""
                out["error"] = str(err)[:800]
            except OSError as e:
                out["ok"] = False
                out["error"] = str(e)[:800]
        elif verbose:
            print("[rs_integration] materialize_workspace skipped (fresh)", file=sys.stderr)

    if "prune_paths" in steps:
        removed: List[str] = []
        for rel in boot.get("remove_if_present") or []:
            p = _safe_path_under(rs_root, str(rel))
            if p is None or not p.exists():
                continue
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                removed.append(str(rel).replace("\\", "/"))
            except OSError as e:
                out["ok"] = False
                out.setdefault("prune_errors", []).append(f"{rel}: {e}")
        if removed:
            out["actions"].append({"prune_paths": removed})

    return out


def bootstrap_summary_for_hints(data: Dict[str, Any]) -> str:
    """供 openclaw_compat 拼进 AI 提示的一小段。"""
    boot = data.get("bootstrap") or {}
    if boot.get("on_agent_startup") is False:
        return "启动引导: 已关闭"
    steps = boot.get("steps") or ["materialize_workspace", "prune_paths"]
    return "启动引导: " + ", ".join(str(s) for s in steps) + "（见 integrated_layout.json#bootstrap）"
