"""
lifers/scripts/run_agent.py
──────────────────────────────────
Main agent entry point. Called by tools/run_lifers_agent.bat.
Starts the persistent bridge server and handles graceful shutdown.

Environment variables (set by .bat launcher):
  PYTHONUTF8=1
  SANDBOX=1        (optional — enables extra safety checks)
  MODEL=transformer
"""
from __future__ import annotations
import json, logging, os, sys, time
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent              # lifers/ 包根
PROJECT_ROOT = ROOT.parent                        # 项目根
sys.path.insert(0, str(ROOT))

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "memory" / "agent.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("run_agent")


def load_stack() -> dict:
    path = PROJECT_ROOT / "config" / "stack.json"
    if not path.exists():
        log.error("stack.json not found at %s", path)
        sys.exit(1)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def preflight(stack: dict) -> bool:
    """Validate critical files before starting bridge."""
    ok = True
    checks = [
        ROOT / "config" / "tokenizer.json",
        ROOT / "core"   / "input_validator.py",
        ROOT / "core"   / "inference_router.py",
        ROOT / "core"   / "output_formatter.py",
        ROOT / "core"   / "npc" / "npc_manager.py",
    ]
    for f in checks:
        if not f.exists():
            log.error("MISSING: %s", f.relative_to(ROOT))
            ok = False
        else:
            log.info("OK: %s", f.relative_to(ROOT))

    brain_s = stack.get("brain") or {}
    wmap = brain_s.get("weights") or {}
    weights_rel = wmap.get(stack.get("brain", {}).get("model", "transformer"), "weights/lifers_transformer.json")
    weights = ROOT / weights_rel
    if not weights.exists():
        log.warning("Weights not found: %s — local inference will use stub", weights)

    return ok


def main() -> None:
    log.info("=== Lifers Agent starting ===")
    log.info("SANDBOX=%s  MODEL=%s", os.getenv("SANDBOX", "0"), os.getenv("MODEL", "?"))

    stack = load_stack()
    log.info("Stack version: %s", stack.get("version", "?"))

    if not preflight(stack):
        log.error("Preflight failed — aborting")
        sys.exit(1)

    # Import bridge here (after preflight confirms deps exist)
    from scripts.agent_bridge import main as run_bridge
    run_bridge()


if __name__ == "__main__":
    main()
