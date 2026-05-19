"""
tools/run_integrated_bootstrap.py
──────────────────────────────────
Integrated bootstrap: prune stale files, conditionally materialize
missing core modules, validate pipeline integrity.
Called from tasks.json "lifers: Integrated bootstrap only".
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT   = Path(__file__).parent.parent
BRAIN  = ROOT / "lifers"
CONFIG = BRAIN / "config"

REQUIRED_FILES = [
    BRAIN / "config" / "stack.json",
    BRAIN / "config" / "tokenizer.json",
    BRAIN / "scripts" / "run_agent.py",
    BRAIN / "scripts" / "agent_bridge.py",
    BRAIN / "core"    / "input_validator.py",
    BRAIN / "core"    / "inference_router.py",
    BRAIN / "core"    / "output_formatter.py",
    BRAIN / "core"    / "npc" / "persona.py",
    BRAIN / "core"    / "npc" / "state_machine.py",
    BRAIN / "core"    / "npc" / "npc_manager.py",
]

STALE_FILES = [
    ROOT / "rs.code-workspace",          # superseded by lifers.code-workspace
    BRAIN / "weights" / "tiny_transformer_v001.json",  # README says deleted
]

REQUIRED_DIRS = [
    BRAIN / "config" / "personas",
    BRAIN / "core"   / "npc",
    BRAIN / "scripts",
    BRAIN / "memory",
    BRAIN / "weights",
    ROOT  / "config",
    ROOT  / "data" / "extensions",
    ROOT  / "data" / "user-data",
]


def banner(msg: str) -> None:
    print(f"\n{'─'*50}\n  {msg}\n{'─'*50}")


def prune() -> list[str]:
    pruned = []
    for p in STALE_FILES:
        if p.exists():
            p.unlink()
            pruned.append(str(p))
            print(f"  [pruned] {p}")
    return pruned


def materialize_dirs() -> None:
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    # Ensure __init__.py exists in Python packages
    for pkg in [BRAIN, BRAIN / "core", BRAIN / "core" / "npc", BRAIN / "scripts"]:
        init = pkg / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")


def validate() -> list[str]:
    missing = []
    for f in REQUIRED_FILES:
        if not f.exists():
            missing.append(str(f.relative_to(ROOT)))
            print(f"  [MISSING] {f.relative_to(ROOT)}")
        else:
            print(f"  [  OK  ] {f.relative_to(ROOT)}")
    return missing


def validate_stack() -> bool:
    stack_path = CONFIG / "stack.json"
    if not stack_path.exists():
        print("  [SKIP] stack.json not found")
        return False
    try:
        with stack_path.open(encoding="utf-8") as f:
            stack = json.load(f)
        required_keys = ["model", "gate", "bridge", "npc", "output"]
        for k in required_keys:
            if k not in stack:
                print(f"  [WARN] stack.json missing key: {k}")
        print("  [OK] stack.json parsed successfully")
        return True
    except json.JSONDecodeError as e:
        print(f"  [ERR] stack.json invalid JSON: {e}")
        return False


def main() -> int:
    banner("1. Prune stale files")
    pruned = prune()
    print(f"  Pruned {len(pruned)} file(s)")

    banner("2. Materialize directories + __init__.py")
    materialize_dirs()
    print("  Directories ready")

    banner("3. Validate required files")
    missing = validate()

    banner("4. Validate stack.json")
    validate_stack()

    banner("Summary")
    if missing:
        print(f"  ✗ {len(missing)} file(s) missing:")
        for m in missing:
            print(f"      {m}")
        print("\n  Run powershell -File tools/bootstrap_lifers.ps1 to regenerate missing files.")
        return 1
    print("  ✓ All required files present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
