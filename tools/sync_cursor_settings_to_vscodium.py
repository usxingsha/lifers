"""
tools/sync_cursor_settings_to_vscodium.py
──────────────────────────────────────────
Copies Cursor user settings → lifers VSCodium user-data directory.
Skips keys that are Cursor-exclusive (AI/account/telemetry).
Called from tasks.json "lifers: Sync Cursor settings to VSCodium user-data".
"""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT         = Path(__file__).parent.parent
VSCODIUM_UD  = ROOT / "data" / "user-data" / "User"

# Cursor user settings locations (try both)
_CURSOR_CANDIDATES = [
    Path(os.environ.get("APPDATA", "")) / "Cursor" / "User" / "settings.json",
    Path(os.environ.get("USERPROFILE", "")) / ".cursor" / "User" / "settings.json",
]

# Keys to drop (Cursor-specific, not valid in VSCodium)
_DROP_PREFIXES = (
    "cursor.",
    "github.copilot",
    "telemetry.",
    "workbench.enableExperiments",
    "update.",
)


def find_cursor_settings() -> Path | None:
    for p in _CURSOR_CANDIDATES:
        if p.exists():
            return p
    return None


def filter_settings(raw: dict) -> dict:
    return {
        k: v for k, v in raw.items()
        if not any(k.startswith(pfx) for pfx in _DROP_PREFIXES)
    }


def merge_settings(cursor: dict, existing: dict) -> dict:
    """Cursor settings are base; existing VSCodium settings win on conflict."""
    merged = {**cursor, **existing}
    return merged


def main() -> int:
    cursor_path = find_cursor_settings()
    if not cursor_path:
        print("Cursor settings not found — nothing to sync")
        return 0

    print(f"Cursor settings: {cursor_path}")
    with cursor_path.open(encoding="utf-8") as f:
        cursor_settings = json.load(f)

    filtered = filter_settings(cursor_settings)
    print(f"  Kept {len(filtered)}/{len(cursor_settings)} keys (dropped Cursor-only)")

    VSCODIUM_UD.mkdir(parents=True, exist_ok=True)
    target = VSCODIUM_UD / "settings.json"

    existing: dict = {}
    if target.exists():
        try:
            with target.open(encoding="utf-8") as f:
                existing = json.load(f)
            print(f"  Existing VSCodium settings: {len(existing)} keys")
        except json.JSONDecodeError:
            print("  Existing VSCodium settings corrupted — overwriting")

    merged = merge_settings(filtered, existing)

    # Backup
    if target.exists():
        shutil.copy2(target, target.with_suffix(".json.bak"))
        print(f"  Backup: {target.with_suffix('.json.bak')}")

    with target.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"  Written {len(merged)} keys → {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
