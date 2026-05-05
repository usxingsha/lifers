"""
Merge portable Cursor user settings into VSCodium user-data settings.

- Reads:   rs/data/cursor-user-data/User/settings.json (if present)
- Defaults: rs/tools/vscodium_editor_defaults.json
- Writes:   rs/data/user-data/User/settings.json

Filters out Cursor/cloud/Copilot/login/paid/API-oriented keys that should not
be carried to an offline, no-account editor profile (rs portable policy).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURSOR_USER = ROOT / "data" / "cursor-user-data" / "User"
VSC_USER = ROOT / "data" / "user-data" / "User"
CURSOR_SETTINGS = CURSOR_USER / "settings.json"
OUT = VSC_USER / "settings.json"
DEFAULTS = Path(__file__).with_name("vscodium_editor_defaults.json")

BLOCK_PREFIXES: tuple[str, ...] = (
    "cursor.",
    "anysphere.",
    "github.copilot",
    "extensions.experimental.affinity",
    "openai.",
    "anthropic.",
    "telemetry.",
    "aws.telemetry",
    "redhat.telemetry",
)

BLOCK_KEYS: frozenset[str] = frozenset(
    {
        "github.copilot.enable",
        "http.proxyAuthorization",
    }
)

BLOCK_REGEX = [
    re.compile(r"^sync\."),
    re.compile(r"^chat\."),
    re.compile(r"^github\.copilot"),
    re.compile(r"apikey"),
    re.compile(r"api_key"),
]


def should_drop(key: str) -> bool:
    kl = key.lower()
    if kl.startswith(BLOCK_PREFIXES):
        return True
    if key in BLOCK_KEYS:
        return True
    for bad in ("apikey", "api_key", "secret", "token", "password", "bearer", "oauth"):
        if bad not in kl:
            continue
        if any(
            x in kl
            for x in (
                "openai",
                "anthropic",
                "anysphere",
                "cursor",
                "github",
                "copilot",
                "tabnine",
                "codeium",
                "cody",
            )
        ):
            return True
    for rx in BLOCK_REGEX:
        if rx.search(kl):
            return True
    return False


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    return json.loads(text)


def copy_optional_user_files() -> None:
    """Copy keybindings + snippets when present (VS Code compatible)."""
    kb_src = CURSOR_USER / "keybindings.json"
    kb_dst = VSC_USER / "keybindings.json"
    if kb_src.is_file():
        VSC_USER.mkdir(parents=True, exist_ok=True)
        shutil.copy2(kb_src, kb_dst)
        print(f"Copied: {kb_dst}")

    snip_src = CURSOR_USER / "snippets"
    snip_dst = VSC_USER / "snippets"
    if snip_src.is_dir():
        if snip_dst.exists():
            shutil.rmtree(snip_dst)
        shutil.copytree(snip_src, snip_dst)
        print(f"Copied: {snip_dst}")


def main() -> int:
    defaults = load_json(DEFAULTS)
    cursor_user = load_json(CURSOR_SETTINGS)

    merged = dict(defaults)
    dropped = []
    for k, v in cursor_user.items():
        if should_drop(k):
            dropped.append(k)
            continue
        merged[k] = v

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote: {OUT}")
    copy_optional_user_files()
    if dropped:
        print(f"Dropped {len(dropped)} Cursor/cloud-only keys (sample): {dropped[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
