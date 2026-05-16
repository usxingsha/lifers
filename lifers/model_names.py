"""Canonical brain backend names vs product-facing MODEL token (lifers)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def canonical_brain_model(raw: str) -> str:
    """Map stack/env MODEL to LocalBrain backend: lifers | transformer | markov."""
    k = (raw or "").strip().lower()
    if k in ("lifers", "deep"):
        return "lifers"
    if k in ("markov", "transformer"):
        return k
    return "lifers"


def env_model_token_for_process(canonical: str) -> str:
    """What we keep in os.environ['MODEL'] for subprocesses that only understand lifers|transformer|markov."""
    return canonical if canonical in ("lifers", "transformer", "markov") else "lifers"


def default_weight_paths(canonical: str) -> tuple[str, ...]:
    """Trained / pipeline output only — 不再带仓库内「玩具」tiny_transformer 后备。"""
    if canonical == "lifers":
        return ("weights/lifers_deep_transformer.json",)
    if canonical == "transformer":
        return ("weights/lifers_transformer.json",)
    return ("weights/lifers_markov.json", "weights/markov_v001.json")


def resolve_existing_weight_file(root: Path, canonical: str) -> Optional[Path]:
    for rel in default_weight_paths(canonical):
        p = (root / rel).resolve()
        if p.is_file():
            return p
    return None
