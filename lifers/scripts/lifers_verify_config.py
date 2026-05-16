"""
Validate config/stack.json and stack_env wiring (智脑 / 仿真人 / 机器人).
Exit 0 if JSON + imports OK; exit 2 if optional asset paths missing (weights/tasks).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "config" / "stack.json"
    out: dict = {"root": str(root), "errors": [], "warnings": [], "stack_ok": False}

    if not cfg_path.is_file():
        out["errors"].append(f"missing {cfg_path}")
        print(json.dumps({"ok": False, **out}, ensure_ascii=False, indent=2))
        return 1

    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        out["errors"].append(f"invalid JSON: {e}")
        print(json.dumps({"ok": False, **out}, ensure_ascii=False, indent=2))
        return 1

    out["stack_ok"] = True
    brain = data.get("brain") or {}
    robot = data.get("robot") or {}
    hs = data.get("human_sim") or {}

    for role, section in (
        ("brain", brain),
        ("human_sim", hs),
        ("robot", robot),
    ):
        if not isinstance(section, dict):
            out["errors"].append(f"{role} must be an object")

    sys.path.insert(0, str(root))
    from lifers.model_names import resolve_existing_weight_file

    wmap = brain.get("weights") or {}
    for key in ("markov", "transformer"):
        rel = wmap.get(key)
        if isinstance(rel, str) and rel.strip():
            p = (root / rel.strip()).resolve()
            if not p.is_file() and not resolve_existing_weight_file(root, key):
                out["warnings"].append(f"brain.weights.{key}: missing file {p} (run pipeline / train)")

    mem_rel = str(brain.get("memory_db", "memory/longterm.sqlite3"))
    mp = Path(mem_rel) if Path(mem_rel).is_absolute() else (root / mem_rel).resolve()
    if not mp.exists():
        out["warnings"].append(f"brain.memory_db: DB not created yet {mp}")

    tf = str(robot.get("tasks_file", "sim/tasks/tasks_v001.jsonl")).strip() or "sim/tasks/tasks_v001.jsonl"
    tp = Path(tf) if Path(tf).is_absolute() else (root / tf).resolve()
    if not tp.is_file():
        out["warnings"].append(f"robot.tasks_file: missing {tp}")

    audit_rel = str(brain.get("audit_log", "logs/audit.jsonl"))
    ap = Path(audit_rel) if Path(audit_rel).is_absolute() else (root / audit_rel).resolve()
    if not ap.parent.is_dir():
        out["warnings"].append(f"brain.audit_log: parent dir missing {ap.parent} (created on first audit)")

    try:
        from lifers.stack_env import apply_stack_env, load_stack

        apply_stack_env(root)
        loaded = load_stack(root)
        if not loaded:
            out["errors"].append("load_stack returned empty after reading file")
        from lifers.runtime_mode import resolve_runtime

        out["runtime_resolved"] = resolve_runtime(root, loaded)
        out["LIFERS_RUNTIME"] = os.environ.get("LIFERS_RUNTIME")
        from lifers.llm_ops_context import format_llm_ops_context
        from lifers.openclaw_compat import summary_for_verify, verify_upstream_drift

        lo = loaded.get("llm_ops") or {}
        if isinstance(lo, dict):
            blob = format_llm_ops_context(loaded, root)
            out["llm_ops"] = {
                "enabled": lo.get("enabled", True) is not False,
                "inject_chars": len(blob),
                "max_inject_chars": lo.get("max_inject_chars"),
                "context_pack_for_brain_fallback": bool(lo.get("context_pack_for_brain_fallback")),
                "context_pack_max_prompt_chars": lo.get("context_pack_max_prompt_chars"),
            }

        ocfg = loaded.get("openclaw") or {}
        out["openclaw"] = summary_for_verify(ocfg)
        man_p = root / "config" / "openclaw_manifest.json"
        if not man_p.is_file():
            out["warnings"].append("config/openclaw_manifest.json 缺失，OpenClaw 分项对照将仅用语义块；可从仓库补全。")
        claw_man = root / "config" / "claw_code_rust_vendor.json"
        claw_cargo = root.parent / "third_party" / "claw_code_rust" / "Cargo.toml"
        if claw_man.is_file() and not claw_cargo.is_file():
            out["warnings"].append(
                "claw_code_rust: config/claw_code_rust_vendor.json 存在但 third_party/claw_code_rust/Cargo.toml 缺失（vendor 未同步）"
            )
        gitmodules = root.parent / ".gitmodules"
        oc_pkg = root.parent / "third_party" / "openclaw" / "package.json"
        if gitmodules.is_file() and not oc_pkg.is_file():
            out["warnings"].append(
                "openclaw: 根目录有 .gitmodules 但 third_party/openclaw 未检出，请执行: git submodule update --init --depth 1"
            )
        check_remote = bool(ocfg.get("check_remote_on_verify")) or os.environ.get("LIFERS_CHECK_OPENCLAW", "").strip() in (
            "1",
            "true",
            "yes",
        )
        if check_remote and ocfg.get("enabled"):
            drift = verify_upstream_drift(ocfg)
            if drift:
                out["warnings"].append(drift)

        emb = loaded.get("embodied_world") or {}
        if isinstance(emb, dict):
            dn = emb.get("dynamic_npc") if isinstance(emb.get("dynamic_npc"), dict) else {}
            out["embodied_world"] = {
                "enabled": bool(emb.get("enabled")),
                "state_relpath": emb.get("state_relpath"),
                "dynamic_npc_enabled": bool(dn.get("enabled")),
                "tick_script": "scripts/embodied_tick_once.py",
            }

        from lifers.tools import build_default_registry

        reg = build_default_registry()
        tnames = sorted(s.name for s in reg.list_specs())
        out["tool_registry"] = {
            "count": len(tnames),
            "names": tnames,
            "source": "lifers/tools.py build_default_registry",
        }

        pipe_p = root / "config" / "lifers_ai_pipeline.json"
        if pipe_p.is_file():
            try:
                pj = json.loads(pipe_p.read_text(encoding="utf-8"))
                stages = pj.get("stages") if isinstance(pj.get("stages"), list) else []
                out["ai_pipeline"] = {
                    "map_relpath": "config/lifers_ai_pipeline.json",
                    "stage_count": len(stages),
                    "stage_ids": [s.get("id") for s in stages if isinstance(s, dict)],
                }
            except json.JSONDecodeError:
                out["warnings"].append("config/lifers_ai_pipeline.json：JSON 损坏")
        else:
            out["warnings"].append("缺少 config/lifers_ai_pipeline.json（输入→输出全链路索引）")
    except Exception as e:
        out["errors"].append(f"stack_env: {e}")

    ok = len(out["errors"]) == 0
    print(json.dumps({"ok": ok, **out}, ensure_ascii=False, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
