from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from lifers_brain.stack_env import apply_stack_env

    apply_stack_env(root)
    from lifers_brain.agent import AgentConfig, LifersAgent
    from lifers_brain.model_names import canonical_brain_model

    raw_model = os.environ.get("MODEL", "markov").strip().lower()
    model = canonical_brain_model(raw_model)
    sandbox = os.environ.get("SANDBOX", "1") == "1"

    agent = LifersAgent(AgentConfig(root_dir=root, model=raw_model, sandbox=sandbox))
    print(
        f"lifers_agent | MODEL={raw_model} (backend={model}) | SANDBOX={'1' if sandbox else '0'} | Ctrl+C exit\n"
    )

    while True:
        try:
            msg = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not msg:
            continue
        out = agent.step(msg)
        print(out)


if __name__ == "__main__":
    main()

