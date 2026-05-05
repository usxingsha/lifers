"""Embodied loop: toy physics + vision summary + realtime decision; syncs with train_control."""

from lifers_brain.embodied.coordinator import EmbodiedCoordinator, run_embodied_tick
from lifers_brain.embodied.physics import PhysBody, PhysWorld
from lifers_brain.embodied.vision import VisionSummary, observe

__all__ = [
    "EmbodiedCoordinator",
    "PhysBody",
    "PhysWorld",
    "VisionSummary",
    "observe",
    "run_embodied_tick",
]

