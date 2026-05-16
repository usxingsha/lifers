"""
Camera / screen capture skills for Lifers AI.

Provides:
- ``camera_snapshot()`` — grab a frame from webcam → VisionSummary
- ``screen_grab()`` — stub for screen capture (platform-dependent)
- ``image_metadata()`` — existing vision_support wrapper
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def camera_snapshot(
    input_text: str = "",
    context: Optional[Dict[str, Any]] = None,
    camera_index: int = 0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Grab one frame from the camera and return a compact summary.
    Used by the 'camera_snapshot' skill.
    """
    try:
        import cv2
    except ImportError:
        return {"ok": False, "error": "opencv-python not installed (pip install opencv-python)"}

    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        return {"ok": False, "error": f"Camera {camera_index} cannot be opened"}
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            return {"ok": False, "error": "Camera read failed"}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        brightness = float(gray.mean()) / 255.0

        import hashlib
        digest = hashlib.sha256(gray.tobytes()[:4096]).hexdigest()[:8]

        return {
            "ok": True,
            "mode": "camera",
            "camera_index": camera_index,
            "width": w,
            "height": h,
            "brightness": round(brightness, 3),
            "digest8": digest,
            "caption_zh": f"[摄像头 {camera_index}] 帧 {w}x{h}，亮度 {brightness:.1%}",
        }
    finally:
        cap.release()


def screen_grab(
    input_text: str = "",
    context: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Stub: screen capture requires platform-specific tooling (e.g., mss/PIL on Windows).
    Returns a descriptive message.
    """
    return {
        "ok": False,
        "error": "screen_grab not yet implemented — use external screenshot tool or enable embodied vision with watch_path",
    }


def image_metadata(
    input_text: str = "",
    context: Optional[Dict[str, Any]] = None,
    rel_path: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Summarize an image file under LIFERS_ROOT.
    Delegates to ``vision_support.summarize_image_under_root``.
    """
    if not rel_path:
        return {"ok": False, "error": "Missing rel_path parameter"}
    from lifers.vision_support import summarize_image_under_root

    root = Path(context.get("lifers_root", ".")) if context else Path.cwd()
    info = summarize_image_under_root(root, rel_path)
    if not info.get("ok"):
        return {"ok": False, "error": str(info.get("error", "vision_digest failed"))}
    return {"ok": True, "info": info, "caption_zh": info.get("caption_zh", "")}
