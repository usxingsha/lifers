"""Canonical brain backend name — only Lifers Deep."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def canonical_brain_model(raw: str) -> str:
    """All models map to lifers — the only backend."""
    return "lifers"


def env_model_token_for_process(canonical: str) -> str:
    """MODEL env token is always lifers."""
    return "lifers"


def default_weight_paths(canonical: str) -> tuple[str, ...]:
    """Only lifers_deep_transformer.json."""
    return ("weights/lifers_deep_transformer.json",)


def resolve_existing_weight_file(root: Path, canonical: str) -> Optional[Path]:
    for rel in default_weight_paths(canonical):
        p = (root / rel).resolve()
        if p.is_file():
            return p
    return None
