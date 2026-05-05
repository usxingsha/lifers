#!/usr/bin/env python3
"""
Lifers 全链自检：stack、记忆、任务流分类、Planner、工具 dry_run、安全写盘、steward 参数、桥接一轮。

用法（在 lifers_brain 仓库根）:
  python eval/full_system_check.py

环境（可选）:
  LIFERS_FULL_CHECK_BRIDGE=0  — 跳过桥接整轮（最快）
  LIFERS_FULL_CHECK_NETWORK=1 — 桥接阶段允许 SANDBOX=0 做一次联网（可能慢/失败）
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent.parent


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _ok(name: str) -> None:
    print(f"ok  {name}")


def check_stack() -> None:
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("LIFERS_ROOT", str(ROOT))
    from lifers_brain.stack_env import apply_stack_env, load_stack

    apply_stack_env(ROOT)
    st = load_stack(ROOT)
    if not isinstance(st, dict):
        _fail("load_stack not dict")
    _ok("stack_env + load_stack")


def check_memory() -> None:
    from lifers_brain.memory import LongTermMemory, MemoryItem
    import time

    d = Path(tempfile.mkdtemp())
    db = d / "t.sqlite3"
    m = LongTermMemory(db)
    assert m.count_all() == 0
    m.add(
        MemoryItem(
            type="preference",
            content={"k": "v"},
            importance=0.9,
            source="check",
            ts_ms=int(time.time() * 1000),
        )
    )
    assert m.count_all() == 1
    hits = m.search("v", k=3)
    assert len(hits) >= 1
    pr = m.prune_type_older_than("episode", older_than_days=0, limit=10)
    assert "deleted" in pr
    _ok("memory add/search/count/prune_type")


def check_safe_file() -> None:
    from lifers_brain.safe_file_backup import commit_journal, restore_from_hint, safe_replace_file_text

    base = Path(tempfile.mkdtemp())
    os.environ["LIFERS_ROOT"] = str(base)
    f = base / "a.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("orig", encoding="utf-8")
    r = safe_replace_file_text(f, "new", encoding="utf-8")
    if not r.get("ok"):
        _fail(f"safe_replace: {r}")
    assert f.read_text() == "new"
    restore_from_hint(
        {
            "journal_dir": r.get("journal_dir"),
            "backup_path": r.get("backup_path"),
            "restore_to": r.get("restore_to"),
            "was_new": r.get("was_new"),
        }
    )
    assert f.read_text() == "orig"
    if r.get("journal_dir"):
        commit_journal(str(r["journal_dir"]))
    _ok("safe_file_backup write/rollback")


def check_classify_planner() -> None:
    from lifers_brain.agent import Planner
    from lifers_brain.taskflow.classify import classify_task
    from lifers_brain.taskflow.kinds import TaskKind

    p = Planner()
    cases: List[tuple[str, TaskKind]] = [
        ("search x", TaskKind.WEB_SEARCH),
        ("搜索 y", TaskKind.WEB_SEARCH),
        ("总结 abc", TaskKind.FULL_PIPELINE),
        ("cmd dir", TaskKind.CMD_SHELL),
        ("https://example.com", TaskKind.URL_FETCH),
    ]
    for text, want in cases:
        got = classify_task(text, False)
        if got != want:
            _fail(f"classify {text!r} want {want} got {got}")
    c1 = p.plan("search q")
    assert c1 and c1[0].name == "web_search"
    c2 = p.plan("搜索 z")
    assert c2 and c2[0].name == "web_search"
    c3 = p.plan("rel_write p/x.txt\nbody")
    assert c3 and c3[0].name == "lifers_workspace_write"
    _ok("classify + planner")


def check_tools_dry_run() -> None:
    from lifers_brain.tools import ToolCall, build_default_registry

    reg = build_default_registry()
    args_map: Dict[str, Dict[str, Any]] = {
        "web_search": {"query": "ping", "limit": 1},
        "web_fetch": {"url": "https://example.com"},
        "extract_evidence": {"text": "hello world", "max_snippets": 2},
        "fs_read": {"path": str(ROOT / "config" / "stack.json")},
        "fs_write_patch": {"path": str(ROOT / "README.md"), "new_text": "x"},
        "lifers_workspace_write": {"rel_path": "tmp_check.txt", "new_text": "x"},
        "cmd_run": {"cmd": "echo ok"},
        "kb_upsert": {
            "items": [
                {"type": "fact", "content": {"full_system_check": True}, "importance": 0.22, "source": "eval"}
            ]
        },
        "kb_search": {"query": "note", "k": 2},
        "kb_prune": {"min_importance": 0.01, "older_than_days": 1, "limit": 1},
        "kb_compact": {"url": "https://example.com", "k": 2},
        "sim_run": {"task_id": "t0", "runs": 1},
        "sense_snapshot": {},
        "motion_plan": {"goal": {"x": 1}},
        "motion_execute": {"trajectory": []},
        "manipulate": {"action": "pick", "target": {"id": "noop"}},
        "safety_stop": {},
        "real_world": {"action": "clock"},
    }
    for spec in reg.list_specs():
        name = spec.name
        args = args_map.get(name)
        if args is None:
            _fail(f"dry_run args missing for tool {name}")
        r = reg.dispatch(ToolCall(name=name, args=args, mode="dry_run"))
        if not r.ok:
            _fail(f"dry_run {name}: {r.error}")
    _ok("tools dry_run (all registered)")


def check_steward_math() -> None:
    from lifers_brain.memory import LongTermMemory, MemoryItem
    from lifers_brain.steward import _resolve_global_forget_params
    import time

    class Ag:
        longterm: LongTermMemory

    d = Path(tempfile.mkdtemp())
    a = Ag()
    a.longterm = LongTermMemory(d / "m.sqlite3")
    gf: Dict[str, Any] = {
        "enabled": True,
        "min_importance": 0.14,
        "older_than_days": 48,
        "limit": 260,
        "auto_threshold": {"enabled": True, "rows_soft_cap": 500},
    }
    mi0, _, lim0, _ = _resolve_global_forget_params(gf, a)  # type: ignore[arg-type]
    for i in range(3000):
        a.longterm.add(
            MemoryItem(
                type="episode",
                content={"i": i},
                importance=0.04,
                source="t",
                ts_ms=int(time.time() * 1000) - 100 * 24 * 3600 * 1000,
            )
        )
    mi1, _, lim1, dbg = _resolve_global_forget_params(gf, a)  # type: ignore[arg-type]
    if mi1 >= mi0 and lim1 <= lim0:
        _fail(f"auto_threshold expected more aggressive: {mi0}->{mi1} lim {lim0}->{lim1} dbg={dbg}")
    _ok("steward auto_threshold")


def check_taskflow_dispatch_smoke() -> None:
    from lifers_brain.agent import AgentConfig, LifersAgent
    from lifers_brain.taskflow.handlers import build_default_dispatcher
    from lifers_brain.taskflow.context import TaskContext
    from lifers_brain.taskflow.classify import classify_task
    from lifers_brain.taskflow.kinds import TaskKind

    os.environ["SANDBOX"] = "1"
    os.environ["LIFERS_ROOT"] = str(ROOT)
    agent = LifersAgent(AgentConfig(root_dir=ROOT, model="markov", sandbox=True))
    disp = build_default_dispatcher()
    for text, kind_want in [
        ("hello", TaskKind.CHAT_QUICK),
        ("search unittest", TaskKind.WEB_SEARCH),
    ]:
        k = classify_task(text, False)
        if k != kind_want:
            _fail(f"classify mismatch {text} {k} vs {kind_want}")
        ctx = TaskContext(agent=agent, agent_input=text, user_text=text, kind=k)
        res = disp.dispatch(k, ctx)
        if not res.reply or not str(res.reply).strip():
            _fail(f"empty reply kind={k}")
    _ok("taskflow dispatch (markov+sandbox)")


def check_bridge_once() -> None:
    if os.environ.get("LIFERS_FULL_CHECK_BRIDGE", "1").strip().lower() in ("0", "false", "no", "off"):
        print("skip bridge (LIFERS_FULL_CHECK_BRIDGE=0)")
        return
    env = os.environ.copy()
    env["LIFERS_ROOT"] = str(ROOT)
    env.setdefault("MODEL", "markov")
    env.setdefault("SANDBOX", "1")
    env.setdefault("LIFERS_TASKFLOW_LEARN", "0")
    raw = json.dumps({"text": "hi", "contextFiles": []}, ensure_ascii=False)
    cmd = [sys.executable, str(ROOT / "scripts" / "agent_bridge_once.py")]
    p = subprocess.run(cmd, input=raw, text=True, capture_output=True, env=env, cwd=str(ROOT), timeout=120)
    if p.returncode != 0:
        _fail(f"agent_bridge rc={p.returncode} stderr={p.stderr[:500]}")
    try:
        out = json.loads(p.stdout.strip())
    except json.JSONDecodeError as e:
        _fail(f"bridge json: {e} stdout={p.stdout[:400]}")
    if not out.get("ok"):
        _fail(f"bridge ok=false {out}")
    if not str(out.get("text", "")).strip():
        _fail("bridge empty text")
    _ok("agent_bridge_once (markov SANDBOX=1)")


def check_tools_execute_sandbox() -> None:
    from lifers_brain.tools import ToolCall, build_default_registry

    os.environ["LIFERS_ROOT"] = str(ROOT)
    os.environ["SANDBOX"] = "1"
    reg = build_default_registry()
    r = reg.dispatch(ToolCall(name="web_search", args={"query": "ping", "limit": 1}, mode="execute"))
    if not r.ok or not r.data.get("results"):
        _fail(f"web_search sandbox execute: {r}")
    r2 = reg.dispatch(ToolCall(name="real_world", args={"action": "clock"}, mode="execute"))
    if not r2.ok:
        _fail(f"real_world clock: {r2}")
    _ok("tools execute (sandbox web_search + real_world clock)")


def check_self_code_queue() -> None:
    import shutil

    from lifers_brain.self_code_runner import process_self_code_queue

    tmp = Path(tempfile.mkdtemp())
    (tmp / "config").mkdir(parents=True, exist_ok=True)
    (tmp / "state" / "self_code_queue").mkdir(parents=True, exist_ok=True)
    stack = {"version": 1, "brain": {"self_code": {"enabled": True, "auto_consume_queue": True}}}
    (tmp / "config" / "stack.json").write_text(json.dumps(stack), encoding="utf-8")
    qf = tmp / "state" / "self_code_queue" / "z1.json"
    qf.write_text(json.dumps({"rel_path": "_queue_write_test.txt", "new_text": "from_queue"}), encoding="utf-8")
    os.environ["LIFERS_ROOT"] = str(tmp)
    os.environ["SANDBOX"] = "0"
    out = process_self_code_queue(tmp)
    if out.get("skipped"):
        _fail(f"self_code queue skipped: {out}")
    tgt = tmp / "_queue_write_test.txt"
    if not tgt.is_file() or tgt.read_text(encoding="utf-8") != "from_queue":
        _fail(f"self_code queue write failed: {out}")
    shutil.rmtree(tmp, ignore_errors=True)
    os.environ["LIFERS_ROOT"] = str(ROOT)
    _ok("self_code_runner queue consume")


def check_bridge_network_optional() -> None:
    if os.environ.get("LIFERS_FULL_CHECK_NETWORK", "").strip().lower() not in ("1", "true", "yes", "on"):
        return
    env = os.environ.copy()
    env["LIFERS_ROOT"] = str(ROOT)
    env["MODEL"] = "markov"
    env["SANDBOX"] = "0"
    env["LIFERS_TASKFLOW_LEARN"] = "0"
    raw = json.dumps({"text": "search lifers python test", "contextFiles": []}, ensure_ascii=False)
    cmd = [sys.executable, str(ROOT / "scripts" / "agent_bridge_once.py")]
    p = subprocess.run(cmd, input=raw, text=True, capture_output=True, env=env, cwd=str(ROOT), timeout=180)
    out = json.loads(p.stdout.strip())
    if not out.get("ok"):
        _fail(f"network bridge: {out}")
    _ok("agent_bridge_once (web_search real network)")


def main() -> None:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    check_stack()
    check_memory()
    check_safe_file()
    check_classify_planner()
    check_tools_dry_run()
    check_steward_math()
    check_taskflow_dispatch_smoke()
    check_tools_execute_sandbox()
    check_self_code_queue()
    check_bridge_once()
    check_bridge_network_optional()
    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
