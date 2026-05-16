"""
Advanced image reading skill for Lifers AI.

Provides real image understanding using OpenCV:
- OCR (text extraction via Tesseract if available)
- Face detection (Haar cascades)
- Scene analysis (colors, brightness, contrast)
- Object detection stubs
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, List


def _safe_path(root: Path, rel: str) -> Optional[Path]:
    """Resolve and validate an image path under root."""
    rel_n = rel.replace("\\", "/").lstrip("/")
    target = (root / rel_n).resolve()
    try:
        if not str(target).startswith(str(root.resolve())):
            return None
    except OSError:
        return None
    if not target.is_file():
        return None
    low = rel_n.lower()
    if not any(low.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
        return None
    return target


def read_image_text(
    input_text: str = "",
    context: Optional[Dict[str, Any]] = None,
    rel_path: str = "",
    lang: str = "chi_sim+eng",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    OCR: extract text from an image using Tesseract.

    Parameters
    ----------
    rel_path : str
        Image file path relative to LIFERS_ROOT.
    lang : str
        Tesseract language code (default: chi_sim+eng for Chinese + English).

    Returns
    -------
    dict with keys: ok, text, confidence, error
    """
    if not rel_path:
        return {"ok": False, "error": "Missing rel_path"}

    root = Path(context.get("lifers_root", ".")) if context else Path.cwd()
    p = _safe_path(root, rel_path)
    if p is None:
        return {"ok": False, "error": f"Invalid path or unsupported type: {rel_path}"}

    try:
        import pytesseract
        from PIL import Image

        img = Image.open(p)
        data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)

        words = []
        confs = []
        for i, text in enumerate(data.get("text", [])):
            txt = text.strip()
            conf = int(data.get("conf", [0])[i] or 0)
            if txt and conf > 0:
                words.append(txt)
                confs.append(conf)

        text = " ".join(words)
        avg_conf = sum(confs) / max(len(confs), 1)

        return {
            "ok": True,
            "text": text,
            "word_count": len(words),
            "confidence": round(avg_conf, 1),
            "lang": lang,
            "path": rel_path,
        }
    except ImportError:
        return {"ok": False, "error": "pytesseract not installed (pip install pytesseract)"}
    except Exception as e:
        return {"ok": False, "error": f"OCR failed: {e}"}


def detect_faces(
    input_text: str = "",
    context: Optional[Dict[str, Any]] = None,
    rel_path: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Face detection using OpenCV Haar cascades.

    Parameters
    ----------
    rel_path : str
        Image file path relative to LIFERS_ROOT.

    Returns
    -------
    dict with ok, face_count, faces (list of bounding boxes), error
    """
    if not rel_path:
        return {"ok": False, "error": "Missing rel_path"}

    root = Path(context.get("lifers_root", ".")) if context else Path.cwd()
    p = _safe_path(root, rel_path)
    if p is None:
        return {"ok": False, "error": f"Invalid path: {rel_path}"}

    try:
        import cv2
        import numpy as np

        img = cv2.imread(str(p))
        if img is None:
            return {"ok": False, "error": "Failed to read image"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        face_list = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                     for (x, y, w, h) in faces]

        return {
            "ok": True,
            "face_count": len(faces),
            "faces": face_list,
            "path": rel_path,
        }
    except ImportError:
        return {"ok": False, "error": "OpenCV not available"}
    except Exception as e:
        return {"ok": False, "error": f"Face detection failed: {e}"}


def analyze_scene(
    input_text: str = "",
    context: Optional[Dict[str, Any]] = None,
    rel_path: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Analyze image scene: colors, brightness, contrast, edges, dominant colors.

    Parameters
    ----------
    rel_path : str
        Image file path relative to LIFERS_ROOT.

    Returns
    -------
    dict with image analysis results
    """
    if not rel_path:
        return {"ok": False, "error": "Missing rel_path"}

    root = Path(context.get("lifers_root", ".")) if context else Path.cwd()
    p = _safe_path(root, rel_path)
    if p is None:
        return {"ok": False, "error": f"Invalid path: {rel_path}"}

    try:
        from PIL import Image
        import numpy as np

        img = Image.open(p)
        arr = np.array(img)

        h, w = arr.shape[:2]

        # Color analysis
        if len(arr.shape) == 3:
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            brightness = float(arr.mean()) / 255.0
            contrast = float(arr.std()) / 255.0

            # Dominant colors (simple quantization)
            flat = arr.reshape(-1, 3)
            # Sample every 10th pixel for speed
            sampled = flat[::10]
            # Simple average color per channel
            avg_color = {
                "r": int(sampled[:, 0].mean()),
                "g": int(sampled[:, 1].mean()),
                "b": int(sampled[:, 2].mean()),
            }
        else:
            brightness = float(arr.mean()) / 255.0
            contrast = float(arr.std()) / 255.0
            avg_color = {"gray": int(arr.mean())}

        # Estimate if it's a photo, document, or dark
        if brightness < 0.15:
            scene_type = "dark/low-light"
        elif contrast < 0.1:
            scene_type = "flat/uniform"
        elif contrast > 0.4:
            scene_type = "high-contrast"
        else:
            scene_type = "normal"

        # Basic edge detection for complexity
        if len(arr.shape) == 3:
            try:
                import cv2
                gray_cv = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray_cv, 100, 200)
                edge_ratio = float(edges.sum()) / (h * w * 255)
            except Exception:
                edge_ratio = 0.0
        else:
            edge_ratio = 0.0

        digest = hashlib.sha256(np.ascontiguousarray(arr).tobytes()[:4096]).hexdigest()[:8]

        return {
            "ok": True,
            "width": w,
            "height": h,
            "brightness": round(brightness, 3),
            "contrast": round(contrast, 3),
            "edge_ratio": round(edge_ratio, 3),
            "avg_color": avg_color,
            "scene_type": scene_type,
            "digest8": digest,
            "path": rel_path,
        }
    except Exception as e:
        return {"ok": False, "error": f"Scene analysis failed: {e}"}


def full_image_read(
    input_text: str = "",
    context: Optional[Dict[str, Any]] = None,
    rel_path: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Full image reading: scene analysis + face detection + OCR.
    Combines all available analysis into one comprehensive result.
    """
    result = {
        "path": rel_path,
        "scene": None,
        "faces": None,
        "ocr": None,
        "errors": [],
    }

    # Scene analysis
    scene = analyze_scene(input_text=input_text, context=context, rel_path=rel_path)
    if scene.get("ok"):
        result["scene"] = {k: scene[k] for k in ["width", "height", "brightness", "contrast", "scene_type", "avg_color", "edge_ratio"]}
    else:
        result["errors"].append(f"scene: {scene.get('error')}")

    # Face detection
    faces = detect_faces(input_text=input_text, context=context, rel_path=rel_path)
    if faces.get("ok"):
        result["faces"] = {"count": faces["face_count"], "boxes": faces["faces"]}
    else:
        result["errors"].append(f"face: {faces.get('error')}")

    # OCR
    ocr = read_image_text(input_text=input_text, context=context, rel_path=rel_path)
    if ocr.get("ok"):
        txt = ocr.get("text", "").strip()
        result["ocr"] = {
            "text": txt,
            "word_count": ocr["word_count"],
            "confidence": ocr["confidence"],
        }
    else:
        result["errors"].append(f"ocr: {ocr.get('error')}")

    result["ok"] = bool(result["scene"]) or bool(result["faces"]) or bool(result["ocr"])
    return result
