"""Lightweight 2D kinematics for embodied loop (engineering toy, not a physics engine)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple


Circle = Tuple[float, float, float]  # cx, cy, r


@dataclass
class PhysBody:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 0.12
    heading: float = 0.0  # rad


@dataclass
class PhysWorld:
    """Axis-aligned arena + circular obstacles."""

    width: float = 4.0
    height: float = 4.0
    body: PhysBody = field(default_factory=PhysBody)
    obstacles: List[Circle] = field(default_factory=list)

    def _clamp_pos(self) -> None:
        m = self.body.radius
        self.body.x = max(m, min(self.width - m, self.body.x))
        self.body.y = max(m, min(self.height - m, self.body.y))

    def _circle_hit(self, cx: float, cy: float, r: float) -> bool:
        dx = self.body.x - cx
        dy = self.body.y - cy
        dist = math.hypot(dx, dy)
        return dist < self.body.radius + r + 1e-6

    def step(self, dt: float, thrust: float, yaw_rate: float) -> None:
        """thrust m/s^2 along heading; yaw_rate rad/s."""
        h = self.body.heading + yaw_rate * dt
        self.body.heading = (h + math.pi) % (2 * math.pi) - math.pi
        ax = math.cos(self.body.heading) * thrust
        ay = math.sin(self.body.heading) * thrust
        self.body.vx += ax * dt
        self.body.vy += ay * dt
        # light drag
        drag = max(0.0, 1.0 - 0.35 * dt)
        self.body.vx *= drag
        self.body.vy *= drag
        self.body.x += self.body.vx * dt
        self.body.y += self.body.vy * dt
        self._clamp_pos()
        for cx, cy, r in self.obstacles:
            if self._circle_hit(cx, cy, r):
                # elastic-ish push-out along radial
                dx = self.body.x - cx
                dy = self.body.y - cy
                dist = max(1e-6, math.hypot(dx, dy))
                push = (self.body.radius + r - dist) + 1e-4
                self.body.x += dx / dist * push
                self.body.y += dy / dist * push
                self.body.vx *= -0.2
                self.body.vy *= -0.2
                self._clamp_pos()

    def to_dict(self) -> dict:
        return {
            "w": self.width,
            "h": self.height,
            "body": {
                "x": self.body.x,
                "y": self.body.y,
                "vx": self.body.vx,
                "vy": self.body.vy,
                "r": self.body.radius,
                "hd": self.body.heading,
            },
            "obstacles": [list(o) for o in self.obstacles],
        }

    @staticmethod
    def from_dict(d: dict) -> "PhysWorld":
        b = d.get("body") or {}
        body = PhysBody(
            x=float(b.get("x", 0)),
            y=float(b.get("y", 0)),
            vx=float(b.get("vx", 0)),
            vy=float(b.get("vy", 0)),
            radius=float(b.get("r", 0.12)),
            heading=float(b.get("hd", 0)),
        )
        obs = [tuple(map(float, x)) for x in (d.get("obstacles") or []) if len(x) >= 3]
        return PhysWorld(
            width=float(d.get("w", 4)),
            height=float(d.get("h", 4)),
            body=body,
            obstacles=obs,
        )
