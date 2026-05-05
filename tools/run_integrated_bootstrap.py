#!/usr/bin/env python3
"""Run integrated_layout.json bootstrap via lifers_brain.rs_integration (from portable root).

Steps: conditional materialize_workspace + prune_paths (remove_if_present whitelist).
Requires lifers_brain next to tools/ (folder may be named lifers or rs historically).
"""
from __future__ import annotations

import sys
from pathlib import Path

RS_ROOT = Path(__file__).resolve().parents[1]
BRAIN = RS_ROOT / "lifers" if (RS_ROOT / "lifers" / "scripts" / "agent_bridge_once.py").is_file() else RS_ROOT / "lifers_brain"
sys.path.insert(0, str(BRAIN))


def main() -> int:
    if not BRAIN.is_dir():
        print(f"missing {BRAIN}", file=sys.stderr)
        return 1
    try:
        from lifers_brain.rs_integration import run_rs_integration_bootstrap
    except ImportError as e:
        print(f"import lifers_brain failed: {e}", file=sys.stderr)
        return 1

    out = run_rs_integration_bootstrap(BRAIN.resolve())
    actions = out.get("actions") or []
    if actions:
        print("[integrated_bootstrap]", actions)
    else:
        print("[integrated_bootstrap] no actions (workspace up to date; nothing pruned)")
    if not out.get("ok", True):
        err = out.get("error") or out.get("prune_errors") or "unknown"
        print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
