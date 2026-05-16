#!/usr/bin/env python3
"""
检查 Lifers 本体 LLM 能否完成「对话→回复」最小闭环（权重文件 + vendor 对照树可选）。

- 不启动 OpenClaw / claw-code 网关；仅核对 third_party 目录是否存在（子模块未检出时告警）。
- 退出码：至少存在 markov 或 transformer 权重之一则 0，否则 1。

用法（在 lifers 根）:
  python scripts/check_lifers_llm_ready.py
  LIFERS_ROOT=/path/to/lifers python scripts/check_lifers_llm_ready.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    brain = Path(os.environ.get("LIFERS_ROOT", str(Path(__file__).resolve().parent.parent))).resolve()
    portable = brain.parent

    markov = brain / "weights" / "lifers_markov.json"
    trans = brain / "weights" / "lifers_transformer.json"

    op = portable / "third_party" / "openclaw"
    claw = portable / "third_party" / "claw_code_rust"

    map_path = brain / "config" / "lifers_llm_bootstrap.json"
    bootstrap_note = ""
    if map_path.is_file():
        try:
            raw = json.loads(map_path.read_text(encoding="utf-8"))
            bootstrap_note = str(raw.get("description_zh", ""))[:200]
        except json.JSONDecodeError:
            bootstrap_note = "(lifers_llm_bootstrap.json invalid JSON)"

    out: dict = {
        "ok": markov.is_file() or trans.is_file(),
        "brain_root": str(brain),
        "weights": {"markov": markov.is_file(), "transformer": trans.is_file()},
        "vendor_trees": {
            "third_party_openclaw": op.is_dir(),
            "third_party_claw_code_rust": claw.is_dir(),
        },
        "hints": [],
        "bootstrap_map": "config/lifers_llm_bootstrap.json" if map_path.is_file() else "missing lifers_llm_bootstrap.json",
    }
    if not out["ok"]:
        out["hints"].append("缺少 weights/lifers_markov.json 与 lifers_transformer.json：在 lifers 执行 python scripts/run_pipeline.py 或按 playbook 训练/同步权重。")
    if not op.is_dir():
        out["hints"].append("未检出 third_party/openclaw：在便携根执行 git submodule update --init --depth 1 third_party/openclaw（仅对照上游，非运行时）。")
    if not claw.is_dir():
        out["hints"].append("缺少 third_party/claw_code_rust：从 Kali 对照路径同步 vendor 或解压 dist（见 claw_code_rust_vendor.json）。")
    if bootstrap_note:
        out["policy_snippet_zh"] = bootstrap_note

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
