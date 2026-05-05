"""Eyes: optional image path or tiny camera grab → compact summary for decisions."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class VisionSummary:
    mode: str  # none | path | camera
    brightness: float  # 0..1 coarse
    motion_hint: float  # 0..1 stub when no prev
    digest8: str


def _brightness_from_bytes(blob: bytes) -> float:
    if not blob:
        return 0.0
    n = min(len(blob), 4096)
    s = sum(blob[:n]) / max(1, n)
    return max(0.0, min(1.0, s / 255.0))


def summarize_path(image_path: Path | None) -> VisionSummary:
    if not image_path or not image_path.is_file():
        return VisionSummary("none", 0.0, 0.0, "00000000")
    try:
        blob = image_path.read_bytes()[:65536]
    except OSError:
        return VisionSummary("none", 0.0, 0.0, "00000000")
    br = _brightness_from_bytes(blob)
    dg = hashlib.sha256(blob).hexdigest()[:8]
    return VisionSummary("path", br, 0.15, dg)


def try_camera_frame(camera_index: int = 0) -> Optional[VisionSummary]:
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(int(camera_index))
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        br = float(gray.mean()) / 255.0
        dg = hashlib.sha256(gray.tobytes()[:4096]).hexdigest()[:8]
        return VisionSummary("camera", br, 0.25, dg)
    except Exception:
        return None


def observe(cfg: Dict[str, Any], root: Path) -> VisionSummary:
    v = cfg.get("vision") or {}
    if not v.get("enabled"):
        return VisionSummary("none", 0.0, 0.0, "00000000")
    src = str(v.get("frame_source") or "none").lower()
    if src == "camera":
        s = try_camera_frame(int(v.get("camera_index") or 0))
        return s or VisionSummary("camera", 0.0, 0.0, "00000000")
    if src == "path":
        rel = str(v.get("watch_path") or "").strip()
        p = Path(rel) if rel and Path(rel).is_absolute() else (root / rel if rel else None)
        return summarize_path(p)
    return VisionSummary("none", 0.0, 0.0, "00000000")
