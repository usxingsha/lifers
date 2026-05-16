"""图像路径 → 可注入 prompt 的轻量摘要（无 torch/CLIP 依赖）。

可选 Pillow 读取尺寸；否则仅文件大小 + 头字节哈希。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict


def _safe_image_path(root: Path, rel: str) -> Path | None:
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


def summarize_image_under_root(root: Path, rel: str, *, max_bytes: int = 262_144) -> Dict[str, Any]:
    """
    返回 dict：含 `caption_zh` 一行说明，供工具或 prompt 前缀使用。
    """
    p = _safe_image_path(root, rel)
    if p is None:
        return {"ok": False, "error": "invalid path or unsupported type"}
    try:
        raw = p.read_bytes()[:max_bytes]
    except OSError as e:
        return {"ok": False, "error": str(e)}
    dg = hashlib.sha256(raw).hexdigest()[:16]
    width = height = None
    try:
        from PIL import Image  # type: ignore

        with Image.open(p) as im:
            width, height = im.size
    except Exception:
        pass
    cap = (
        f"[图像摘要 path={rel}] 字节≈{len(raw)}，sha256[:16]={dg}"
        + (f"，尺寸 {width}x{height}" if width else "")
        + "。（无 CLIP：仅元数据/尺寸；语义理解仍靠主模型与上下文。）"
    )
    return {
        "ok": True,
        "path": rel.replace("\\", "/").lstrip("/"),
        "bytes_read": len(raw),
        "sha16": dg,
        "width": width,
        "height": height,
        "caption_zh": cap,
    }
