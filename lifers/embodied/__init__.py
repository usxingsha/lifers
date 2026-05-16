"""Embodied loop: toy physics + vision summary + realtime decision; syncs with train_control."""

from lifers.embodied.coordinator import EmbodiedCoordinator, run_embodied_tick
from lifers.embodied.physics import PhysBody, PhysWorld
from lifers.embodied.vision import VisionSummary, observe

__all__ = [
    "EmbodiedCoordinator",
    "PhysBody",
    "PhysWorld",
    "VisionSummary",
    "observe",
    "run_embodied_tick",
]

