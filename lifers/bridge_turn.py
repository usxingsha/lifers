"""
Shared single-turn agent invocation for stdin bridge and lifers_gate HTTP service.

CHAT_QUICK（Agents UI）在 **MODEL=transformer** 且边缘 CPU 上时，agent 会默认收紧
`LIFERS_QUICK_SESSION_CONTEXT_CHARS` / `LIFERS_QUICK_PACK_MAX_CHARS`，减轻本地生成超时；
可用环境变量按需放宽。

Agents Chat 设 `LIFERS_AGENTS_UI_BRIDGE=1` 时，默认在快路径回复末追加 **【本轮·生成锚】**（`agent._append_quick_reply_time_footer`）；
可用 `LIFERS_QUICK_TIME_FOOTER=0` 关闭或 `=1` 强制开启（无 Bridge 时亦可在终端试验）。

多模态输入支持：contextFiles 可包含图片文件（png/jpg/gif/webp/bmp），
自动识别并作为 base64 data URL 注入上下文，供模型参考。
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".svg"})
IMAGE_MIME_MAP = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".ico": "image/x-icon", ".svg": "image/svg+xml",
}
TEXT_EXTENSIONS = frozenset({
    ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".jsonl",
    ".html", ".css", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".sh", ".bat", ".ps1", ".c", ".cpp", ".h", ".hpp", ".rs", ".go",
    ".java", ".kt", ".swift", ".r", ".sql", ".csv", ".tsv", ".log",
    ".vue", ".svelte", ".rb", ".php", ".pl", ".lua", ".scala", ".clj",
})


def _is_image_ext(suffix: str) -> bool:
    return suffix.lower() in IMAGE_EXTENSIONS


def _is_text_ext(suffix: str) -> bool:
    return suffix.lower() in TEXT_EXTENSIONS


def _read_context_file(root: Path, rel: str, max_text_chars: int = 8000,
                       max_image_bytes: int = 5 * 1024 * 1024) -> Dict[str, Any]:
    """读取单个上下文文件，返回结构化信息。图片返回 base64 data URL，文本返回内容。"""
    target = (root / rel).resolve()
    try:
        root_res = root.resolve()
        if not str(target).startswith(str(root_res)):
            return {"ok": False, "error": "path outside root"}
        if not target.is_file():
            return {"ok": False, "error": "not found"}
        suffix = target.suffix
        if _is_image_ext(suffix):
            raw = target.read_bytes()[:max_image_bytes]
            b64 = base64.b64encode(raw).decode("ascii")
            mime = IMAGE_MIME_MAP.get(suffix.lower(), "application/octet-stream")
            return {
                "ok": True, "type": "image", "name": target.name,
                "mime": mime, "data_base64": b64, "size": len(raw),
            }
        content = target.read_text(encoding="utf-8", errors="ignore")[:max_text_chars]
        return {"ok": True, "type": "text", "name": target.name, "content": content}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _safe_read_under_root(root: Path, rel: str, max_chars: int = 6000) -> str:
    target = (root / rel).resolve()
    try:
        root_res = root.resolve()
        if not str(target).startswith(str(root_res)):
            return ""
        if not target.is_file():
            return ""
        suffix = target.suffix
        if _is_image_ext(suffix):
            return f"[图片文件: {target.name}]"
        return target.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def _strip_api_key_nag_when_local_only(reply: str) -> str:
    """Agents Chat 已强制本地时，去掉模型复述的「须配 NVIDIA 密钥」类段落。"""
    if os.environ.get("LIFERS_FORCE_LOCAL_ONLY", "").strip().lower() not in ("1", "true", "yes", "on"):
        return reply
    if not reply or not reply.strip():
        return reply
    needles = (
        "NVIDIA_API_KEY",
        "LIFERS_CHAT_API_KEY",
        "未检测到 API 密钥",
        "未检测到API密钥",
        "未检测到 NVIDIA",
    )
    lines = reply.split("\n")
    kept: list[str] = []
    for ln in lines:
        if any(n in ln for n in needles):
            continue
        if "远程推理" in ln and "密钥" in ln:
            continue
        if "云端" in ln and "密钥" in ln and ("须" in ln or "必须" in ln or "请设置" in ln):
            continue
        kept.append(ln)
    out = "\n".join(kept).strip()
    return out if out else reply


def _sanitize_bridge_input(text: str) -> str:
    """移除 NUL 等会破坏 JSON/日志的控制字符（运行时最小输入清洗，非 XSS 沙箱）。"""
    if not text:
        return text
    return text.replace("\x00", "").replace("\ufeff", "")


def lifers_turn(root: Path, text: str, context_files: List[Any],
                images: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Run one LifersAgent step. Returns {"ok": bool, "text": str, "error": optional str}.
    支持多模态：context_files 中的图片自动识别，images 参数传递 base64 图片数据。
    """
    from lifers.stack_env import apply_stack_env

    apply_stack_env(root)
    text = _sanitize_bridge_input(text)
    if len(text) > 80_000:
        return {"ok": False, "text": "", "error": f"input too long ({len(text)} chars, max 80000)"}
    if len(text) > 40_000:
        sys.stderr.write(f"LIFERS_PROGRESS long_input chars={len(text)}\n")
    # Bridge（扩展）默认注入 LIFERS_FORCE_LOCAL_ONLY：禁用云端 chat/completions，仅用本地权重 + 联网工具。
    if os.environ.get("LIFERS_FORCE_LOCAL_ONLY", "").strip().lower() in ("1", "true", "yes", "on"):
        from lifers.openai_compat_chat import _is_localhost_url
        chat_url = os.environ.get("LIFERS_CHAT_URL", "").strip()
        if not chat_url or not _is_localhost_url(chat_url):
            os.environ.pop("LIFERS_REMOTE_CHAT", None)
    ui = os.environ.get("LIFERS_AGENTS_UI_BRIDGE", "").strip().lower() in ("1", "true", "yes", "on")
    respect = os.environ.get("LIFERS_QUICK_WEB_RESPECT_HOST", "").strip().lower() in ("1", "true", "yes", "on")
    if not ui and not respect:
        os.environ["LIFERS_QUICK_WEB"] = "0"

    ctx_files = context_files if isinstance(context_files, list) else []
    max_ctx = int(os.environ.get("LIFERS_CONTEXT_MAX_FILES", "32").strip() or "32")
    if max_ctx < 1:
        max_ctx = 32

    prefix_parts: list[str] = []
    image_refs: list[str] = []
    total_text_chars = 0
    max_total_ctx = int(os.environ.get("LIFERS_CONTEXT_MAX_CHARS", "32000").strip() or "32000")

    for rel in ctx_files[:max_ctx]:
        if not isinstance(rel, str) or not rel.strip():
            continue
        rel_n = rel.replace("\\", "/").lstrip("/")
        info = _read_context_file(root, rel_n)

        if info.get("type") == "image":
            image_refs.append(
                f"[图片: {info['name']} ({info.get('size', 0)} bytes) "
                f"data:{info['mime']};base64,{info['data_base64'][:200]}...]"
            )
            continue

        if info.get("type") == "text" and info.get("content"):
            content = info["content"]
            full_block = f"--- context file: {rel_n} ---\n{content}"
            if total_text_chars + len(full_block) > max_total_ctx:
                remaining = max_total_ctx - total_text_chars
                if remaining > 200:
                    content = content[:remaining] + "\n... (截断)"
                    full_block = f"--- context file: {rel_n} (truncated) ---\n{content}"
                else:
                    break
            prefix_parts.append(full_block)
            total_text_chars += len(full_block)

    # 追加直接传入的图片（来自 images 参数，已经是 base64）
    if images and isinstance(images, list):
        for img in images[:8]:
            name = img.get("name", "image")
            mime = img.get("mime", "image/png")
            b64 = img.get("data_base64", img.get("data", ""))
            if b64:
                image_refs.append(f"[图片: {name} data:{mime};base64,{b64[:200]}...]")

    # 组装最终上下文
    parts: list[str] = []
    if image_refs:
        parts.append("--- attached images ---\n" + "\n".join(image_refs))
    if prefix_parts:
        parts.append("\n\n".join(prefix_parts))
    if parts:
        full = "\n\n".join(parts) + "\n\n--- user message ---\n" + text
    else:
        full = text

    from lifers.agent import LifersAgent
    from lifers.local_brain import AgentConfig
    from lifers.model_names import canonical_brain_model
    from lifers.taskflow.orchestrator import run_lifers_turn

    model = canonical_brain_model(os.environ.get("MODEL", "transformer"))
    sandbox = os.environ.get("SANDBOX", "1") == "1"
    agent = LifersAgent(AgentConfig(root_dir=root, model=model, sandbox=sandbox))
    try:
        print("LIFERS_PROGRESS bridge LifersAgent（taskflow 或 step）开始 …", file=sys.stderr, flush=True)
        use_tf = os.environ.get("LIFERS_TASKFLOW", "1").strip().lower() not in ("0", "false", "no", "off")
        reply = run_lifers_turn(agent, full) if use_tf else agent.step(full)
        reply = _strip_api_key_nag_when_local_only(reply)
        return {"ok": True, "text": reply}
    except Exception as e:
        return {"ok": False, "text": "", "error": str(e)}


_STREAM_CACHE: dict = {}  # {(root, model): (brain, weights)}


def iter_stream_simple_chars(root: Path, text: str, *, max_chars: int = 200) -> Iterator[str]:
    """
    流式生成：使用 Lifers Deep Transformer。
    供 lifers_gate  使用。
    """
    from lifers.local_brain import AgentConfig, LocalBrain
    from lifers.stack_env import apply_stack_env

    apply_stack_env(root)
    text = _sanitize_bridge_input(text)
    if not text.strip():
        yield ""
        return

    cache_key = (str(root.resolve()), "lifers")
    if cache_key in _STREAM_CACHE:
        brain, w = _STREAM_CACHE[cache_key]
    else:
        brain = LocalBrain(AgentConfig(root_dir=root, model="lifers", sandbox=True))
        wp = brain._weights_path()
        if not wp.is_file():
            yield "（未找到 Lifers Deep 权重文件）"
            return
        w = brain._deep_weights(wp)
        _STREAM_CACHE[cache_key] = (brain, w)

    from lifers.deep_transformer import generate_text
    out = generate_text(w, text, max_new_tokens=max(50, max_chars // 2),
                        temperature=0.8, seed=42)
    for ch in out:
        yield ch


def lifers_turn_from_json_body(root: Path, raw_json: str) -> Dict[str, Any]:
    """Parse stdin-like JSON body and run lifers_turn. 支持多模态输入。"""
    if not raw_json.strip():
        return {"ok": False, "text": "", "error": "empty body"}
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        return {"ok": False, "text": "", "error": f"invalid json: {e}"}
    text = str(data.get("text", "")).strip()
    ctx = data.get("contextFiles") or []
    images = data.get("images") or data.get("attachedImages") or None
    return lifers_turn(root, text, ctx, images=images)
