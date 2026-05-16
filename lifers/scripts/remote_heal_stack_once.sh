#!/usr/bin/env bash
# Run on Kali (or any Linux) from lifers root: merge stack defaults via self_heal.
set -euo pipefail
BR="${LIFERS_BRAIN:-$HOME/lifers/lifers}"
cd "$BR"
export PYTHONPATH="$BR"
python3 -c "import json,sys; from pathlib import Path; sys.path.insert(0, r'.'); from lifers.self_heal import heal_stack_at_startup; print(json.dumps(heal_stack_at_startup(Path('.').resolve()), ensure_ascii=False, indent=2))"
