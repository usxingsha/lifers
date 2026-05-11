"""
本机时间以外的「时空锚点」：可选用户/栈配置地名，或短时缓存的 wttr.in 粗定位（与 IP 出口相关）。

- 不替代 GPS；仅作对话上下文锚定，避免模型臆造当地日期以外的「你在哪里」。
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any, Dict

_cache_mono: float = 0.0
_cache_text: str = ""


def _truthy(raw: Any) -> bool:
    if raw is True:
        return True
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return False


def _http_get_json(url: str, timeout: float) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "LifersBrain/1.0 (realtime_anchor; https://github.com/)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _nearest_area_label(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    area = (data.get("nearest_area") or [{}])[0]
    if not isinstance(area, dict):
        return ""
    parts = []
    an = area.get("areaName")
    if isinstance(an, list) and an:
        parts.append(str(an[0]))
    elif isinstance(an, str) and an.strip():
        parts.append(an.strip())
    rg = area.get("region")
    if isinstance(rg, list) and rg:
        parts.append(str(rg[0]))
    elif isinstance(rg, str) and rg.strip():
        parts.append(rg.strip())
    return ", ".join(x for x in parts if x)


def wttr_ip_area_cached(ttl_sec: float = 600.0, timeout: float = 8.0) -> str:
    """请求 wttr.in 无地名路径，用出口 IP 粗定位；结果缓存 ttl_sec 秒。"""
    global _cache_mono, _cache_text
    now = time.monotonic()
    if _cache_text and (now - _cache_mono) < ttl_sec:
        return _cache_text
    try:
        raw_ttl = os.environ.get("LIFERS_REALTIME_GEO_CACHE_SEC", "").strip()
        ttl = float(raw_ttl) if raw_ttl else ttl_sec
        # 显式设 LIFERS_REALTIME_GEO_CACHE_SEC 时可压到 30s；默认分支仍用 ttl_sec（常见 600）不在此强制抬高。
        ttl = max(30.0, min(ttl, 86_400.0))
    except ValueError:
        ttl = ttl_sec
    try:
        data = _http_get_json("https://wttr.in/?format=j1", timeout=timeout)
        label = _nearest_area_label(data)
        _cache_mono = now
        _cache_text = label
        return label
    except Exception:
        _cache_mono = now
        _cache_text = ""
        return ""


def geo_context_line(stack: Dict[str, Any]) -> str:
    """
    单行中文，供注入 INSTINCT_AUTONOMIC；无可用信息时返回空串。
    优先级：stack.brain.geo_label / LIFERS_GEO_LABEL → realtime_geo_quick / LIFERS_REALTIME_GEO_QUICK → wttr 缓存。

    缓存：`LIFERS_REALTIME_GEO_CACHE_SEC`（秒，下限 30）覆盖默认 TTL；与天气同源，非 GPS。
    """
    brain = stack.get("brain") if isinstance(stack.get("brain"), dict) else {}
    label = str(brain.get("geo_label") or os.environ.get("LIFERS_GEO_LABEL", "")).strip()
    if label:
        return f"【实时·定位】{label}（来自 stack / 环境变量）"
    quick = _truthy(brain.get("realtime_geo_quick")) or _truthy(os.environ.get("LIFERS_REALTIME_GEO_QUICK"))
    if not quick:
        return ""
    area = wttr_ip_area_cached()
    if not area:
        return "【实时·定位】粗定位暂不可用（网络或 wttr.in）；可设 LIFERS_GEO_LABEL 或 stack.brain.geo_label。"
    return f"【实时·定位（粗/IP）】{area}（wttr.in，与天气同源；非精确 GPS）"
