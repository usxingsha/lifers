"""可选插件：在 `lifers/tools/plugins/<name>/plugin.py` 中实现 `register_plugin_tools(registry, root)`。

默认关闭（`stack.json` → `plugins.enabled`）；开启后于 `build_default_registry` 末尾加载。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List

from lifers.stack_env import load_stack
from lifers.tools import ToolRegistry


def plugins_enabled_for_root(root: Path) -> bool:
    pl = load_stack(root).get("plugins") or {}
    if isinstance(pl, dict) and (pl.get("enabled") is True or str(pl.get("enabled")).lower() == "true"):
        return True
    import os

    return os.environ.get("LIFERS_PLUGINS", "").strip().lower() in ("1", "true", "yes", "on")


def plugins_dir(root: Path) -> Path:
    pl = load_stack(root).get("plugins") or {}
    rel = "tools/plugins"
    if isinstance(pl, dict):
        r = str(pl.get("rel_dir") or pl.get("dir") or "").strip()
        if r:
            rel = r.replace("\\", "/")
    return (root / rel).resolve()


def load_plugin_modules(root: Path) -> List[str]:
    """返回成功加载的插件目录名（失败写入 stderr，不抛）。"""
    if not plugins_enabled_for_root(root):
        return []
    base = plugins_dir(root)
    if not base.is_dir():
        return []
    loaded: list[str] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith((".", "_")):
            continue
        plug = child / "plugin.py"
        if not plug.is_file():
            continue
        mod_name = f"lifers_plugins_{child.name}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, plug)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
            fn = getattr(mod, "register_plugin_tools", None)
            if not callable(fn):
                continue
            loaded.append(child.name)
        except Exception as e:  # pragma: no cover - best effort
            print(f"[plugins] skip {child.name}: {e}", file=sys.stderr, flush=True)
    return loaded


def register_plugins(registry: ToolRegistry, root: Path) -> List[str]:
    """执行各插件的 register_plugin_tools。"""
    if not plugins_enabled_for_root(root):
        return []
    base = plugins_dir(root)
    out: list[str] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        plug = child / "plugin.py"
        if not plug.is_file():
            continue
        mod_name = f"lifers_plugins_{child.name}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, plug)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
            fn = getattr(mod, "register_plugin_tools", None)
            if callable(fn):
                fn(registry, root)
                out.append(child.name)
        except Exception as e:
            print(f"[plugins] register {child.name} failed: {e}", file=sys.stderr, flush=True)
    return out
