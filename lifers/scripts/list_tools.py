"""Print registered lifers tools (JSON)."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from lifers.tools import build_default_registry, print_specs

    print(print_specs(build_default_registry()))


if __name__ == "__main__":
    main()
