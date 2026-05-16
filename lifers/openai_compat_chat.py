"""OpenAI-compatible chat/completions over HTTPS (stdlib only: urllib, json).

Supports any OpenAI-compatible endpoint:
- Ollama:       http://localhost:11434/v1/chat/completions
- llama.cpp:    http://localhost:8080/v1/chat/completions
- LM Studio:    http://localhost:1234/v1/chat/completions
- vLLM:         http://localhost:8000/v1/chat/completions
- NVIDIA NIM:   https://integrate.api.nvidia.com/v1/chat/completions
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from lifers.speed_env import http_timeout_seconds


def _is_localhost_url(url: str) -> bool:
    """Check if a URL points to localhost (no API key needed, safe for local-only mode)."""
    try:
        p = urlparse(url.strip())
        host = (p.hostname or "").lower()
        return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")
    except Exception:
        return False


def _post_json(url: str, headers: Dict[str, str], body: bytes, timeout: float) -> Tuple[Optional[Dict[str, Any]], str]:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    t = http_timeout_seconds(timeout)
    try:
        with urllib.request.urlopen(req, timeout=t, context=ctx) as resp:  # noqa: S310 — fixed URL from config
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")[:2000]
        except Exception:
            err_body = str(e)
        return None, f"HTTP {e.code} {e.reason}: {err_body}"
    except Exception as e:
        return None, str(e)[:2000]
    try:
        return json.loads(raw), ""
    except json.JSONDecodeError as e:
        return None, f"invalid json: {e}"


def chat_completion_text(
    *,
    url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 1024,
    temperature: float = 0.35,
    timeout_sec: float = 120.0,
) -> Tuple[Optional[str], str]:
    """
    POST /v1/chat/completions style. Returns (assistant_text, error_or_empty).
    """
    if not api_key.strip():
        return None, "empty API key"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max(1, min(int(max_tokens), 8192)),
        "temperature": float(temperature),
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
        "Accept": "application/json",
    }
    data, err = _post_json(url, headers, body, timeout_sec)
    if data is None:
        return None, err
    err_obj = data.get("error")
    if isinstance(err_obj, dict) and err_obj.get("message"):
        return None, str(err_obj.get("message", err_obj))[:2000]
    chs = data.get("choices")
    if not isinstance(chs, list) or not chs:
        return None, f"no choices: {str(data)[:500]}"
    c0 = chs[0]
    if not isinstance(c0, dict):
        return None, "invalid choice"
    msg = c0.get("message")
    if isinstance(msg, dict) and msg.get("content") is not None:
        return str(msg.get("content") or "").strip(), ""
    if c0.get("text"):
        return str(c0.get("text")).strip(), ""
    return None, f"unexpected shape: {str(c0)[:400]}"


def resolve_api_key(url: str = "") -> str:
    """Key from LIFERS_CHAT_API_KEY (direct) or os.environ[ LIFERS_CHAT_API_KEY_ENV ].

    For localhost URLs (Ollama, llama.cpp, LM Studio, etc.), returns a placeholder
    key so the caller doesn't need to set a real API key.
    """
    direct = os.environ.get("LIFERS_CHAT_API_KEY", "").strip()
    if direct:
        return direct
    if url and _is_localhost_url(url):
        return "lifers-local"
    env_name = (os.environ.get("LIFERS_CHAT_API_KEY_ENV") or "NVIDIA_API_KEY").strip() or "NVIDIA_API_KEY"
    return (os.environ.get(env_name) or "").strip()
