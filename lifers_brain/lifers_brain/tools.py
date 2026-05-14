from __future__ import annotations

"""
Lifers 可执行工具（`ToolRegistry`）。

**单一实现源**：本模块各类 `*Tool` 的 `spec.name` + `build_default_registry()` 的 `register(...)` 顺序。
`agent.Planner` / `agent.LifersAgent._step_core` 只产出已注册 **name** 的 `ToolCall`；`config/domains.json` 与
`config/organ_capabilities.json` 的 `tools` 数组为**文档映射**，不得再实现一套执行逻辑。

默认注册名（与 `build_default_registry` 一致，共 19 项）::

    web_search, web_fetch, extract_evidence, fs_read, fs_write_patch,
    lifers_workspace_write, cmd_run, kb_upsert, kb_search, kb_prune, kb_compact,
    sim_run, sense_snapshot, motion_plan, motion_execute, manipulate,
    safety_stop, real_world, vision_digest

编排入口（非新工具名）：`smart`/`智搜`、`流程`/`workflow`、`方案`/`plan`、`cmd`、`sim_run`、`kb_*`、含 URL/路径
的自然语言等 — 见 `agent.py` 中 `Planner` 与 `_step_core`。
"""

import json
import os
import re
import subprocess
import sys
import time
import hashlib
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

Mode = Literal["dry_run", "execute", "verify", "rollback"]


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 30_000
    expected_effect: str = ""
    mode: Mode = "execute"


@dataclass
class ToolResult:
    ok: bool
    data: Dict[str, Any] = field(default_factory=dict)
    evidence_snippets: List[Dict[str, Any]] = field(default_factory=list)
    side_effects: List[Dict[str, Any]] = field(default_factory=list)
    rollback_hint: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    args_schema: Dict[str, Any]
    permissions: List[str]
    supports_modes: Tuple[Mode, ...]
    risk_level: Literal["low", "medium", "high"]
    estimated_cost: Dict[str, Any] = field(default_factory=dict)


class Tool:
    spec: ToolSpec

    def dry_run(self, call: ToolCall) -> ToolResult:  # pragma: no cover
        raise NotImplementedError

    def execute(self, call: ToolCall) -> ToolResult:  # pragma: no cover
        raise NotImplementedError

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:  # pragma: no cover
        raise NotImplementedError

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:  # pragma: no cover
        raise NotImplementedError


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.spec.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.spec.name}")
        self._tools[tool.spec.name] = tool

    def list_specs(self) -> List[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as e:
            raise KeyError(f"Unknown tool: {name}") from e

    def dispatch(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        tool = self.get(call.name)
        if call.mode not in tool.spec.supports_modes:
            return ToolResult(ok=False, error=f"Unsupported mode {call.mode} for tool {call.name}")

        # Minimal permission gate placeholder (structure-first).
        # You can later enforce allowlists/role-based policies here.
        sandbox = os.environ.get("SANDBOX", "0") == "1"
        # By default SANDBOX blocks high-risk *real* execution, but cmd_run is allowed to run
        # (it will still self-sandbox inside CmdRunTool when SANDBOX=1).
        if sandbox and tool.spec.risk_level == "high" and call.mode == "execute" and call.name != "cmd_run":
            return ToolResult(
                ok=False,
                error="SANDBOX=1 blocks high-risk execute. Use dry_run or set SANDBOX=0.",
            )

        fn = {
            "dry_run": tool.dry_run,
            "execute": tool.execute,
            "verify": lambda c: tool.verify(c, prior=prior),
            "rollback": lambda c: tool.rollback(c, prior=prior),
        }[call.mode]
        res = fn(call)
        # Audit (best-effort, no hard dependency).
        try:
            from .audit import audit_log

            audit_log(
                {
                    "kind": "tool_call",
                    "tool": call.name,
                    "mode": call.mode,
                    "expected_effect": call.expected_effect,
                    "ok": res.ok,
                    "error": res.error,
                    "side_effects": res.side_effects,
                }
            )
        except Exception:
            pass
        return res


def _now_ms() -> int:
    return int(time.time() * 1000)


def _outbound_opener() -> urllib.request.OpenerDirector:
    """
    出站 HTTP：尊重 HTTPS_PROXY / HTTP_PROXY / ALL_PROXY 及系统代理（urllib.getproxies）。
    不绑定特定浏览器；由系统与 Python 运行时决定走哪条链路。

    若代理指向本机已关闭的端口（常见 WinError 10061），可设 LIFERS_HTTP_DIRECT=1 跳过代理直连。
    """
    if os.environ.get("LIFERS_HTTP_DIRECT", "").strip().lower() in ("1", "true", "yes", "on"):
        return urllib.request.build_opener()
    try:
        px = urllib.request.getproxies()
    except Exception:
        px = {}
    if px:
        return urllib.request.build_opener(urllib.request.ProxyHandler(px))
    return urllib.request.build_opener()


def _http_open(req: urllib.request.Request, timeout: float):
    from lifers_brain.speed_env import http_timeout_seconds

    return _outbound_opener().open(req, timeout=http_timeout_seconds(timeout))


def _ddg_instant_results(q: str, limit: int) -> Optional[List[Dict[str, Any]]]:
    """DuckDuckGo Instant Answer JSON（不爬 HTML），部分网络下比 html 端点更可达。"""
    api = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({"q": q, "format": "json", "no_html": "1"})
    req = urllib.request.Request(
        api,
        headers={
            "User-Agent": "LifersBrain/1.0 (research; local)",
            "Accept": "application/json",
        },
    )
    try:
        with _http_open(req, timeout=12.0) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    out: List[Dict[str, Any]] = []

    def add_topic(t: Dict[str, Any]) -> None:
        if len(out) >= limit:
            return
        url = str(t.get("FirstURL", "")).strip()
        if not url:
            return
        txt = str(t.get("Text", "")).strip()
        out.append({"title": txt[:120] or url, "url": url, "snippet": txt[:500]})

    def walk_related(items: Any) -> None:
        if not isinstance(items, list) or len(out) >= limit:
            return
        for t in items:
            if len(out) >= limit:
                break
            if not isinstance(t, dict):
                continue
            if "Topics" in t:
                walk_related(t.get("Topics"))
            else:
                add_topic(t)

    au = str(data.get("AbstractURL", "")).strip()
    if au:
        out.append(
            {
                "title": str(data.get("Heading", "") or "Instant answer").strip() or au,
                "url": au,
                "snippet": str(data.get("AbstractText", "") or "").strip()[:800],
            }
        )
    walk_related(data.get("RelatedTopics") or [])
    return out if out else None


def _snippet(source: str, text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    s = {"source": source, "text": text}
    if meta:
        s.update(meta)
    return s


def _lifers_root() -> str:
    root = os.environ.get("LIFERS_ROOT", "").strip()
    if root:
        return root
    # Fallback to current working directory.
    return os.getcwd()


def _memory_db_path() -> str:
    from pathlib import Path as _Path

    from .stack_env import load_stack

    r = _Path(_lifers_root()).resolve()
    rel = str((load_stack(r).get("brain") or {}).get("memory_db", "memory/longterm.sqlite3"))
    p = _Path(rel)
    return str(p.resolve()) if p.is_absolute() else str((r / rel).resolve())


class WebSearchTool(Tool):
    spec = ToolSpec(
        name="web_search",
        args_schema={"query": "string", "limit": "int (default 5)"},
        permissions=["network:outbound"],
        supports_modes=("dry_run", "execute", "verify", "rollback"),
        risk_level="low",
        estimated_cost={"time_ms": 1500},
    )

    def dry_run(self, call: ToolCall) -> ToolResult:
        q = str(call.args.get("query", "")).strip()
        limit = int(call.args.get("limit", 5))
        return ToolResult(ok=True, data={"query": q, "limit": limit, "note": "Would perform a web search."})

    def execute(self, call: ToolCall) -> ToolResult:
        # Minimal, dependency-free: use DuckDuckGo HTML endpoint.
        # In strict environments this may be blocked; keep as best-effort tool.
        q = str(call.args.get("query", "")).strip()
        limit = int(call.args.get("limit", 5))
        if not q:
            return ToolResult(ok=False, error="Missing args.query")

        sandbox = os.environ.get("SANDBOX", "0") == "1"
        if sandbox:
            return ToolResult(
                ok=True,
                data={
                    "results": [
                        {"title": "SANDBOX result", "url": "https://example.com", "snippet": f"query={q}"}
                    ][:limit]
                },
                evidence_snippets=[_snippet("sandbox", f"web_search({q}) mocked")],
            )

        results: List[Dict[str, Any]] = []
        engine = ""
        err_parts: List[str] = []

        ia = _ddg_instant_results(q, limit)
        if ia:
            results = ia[:limit]
            engine = "duckduckgo_instant"

        if not results:
            url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": q})
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "LifersBrain/1.0 (local research)", "Accept": "text/html,*/*"},
            )
            try:
                with _http_open(req, timeout=call.timeout_ms / 1000) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
            except Exception as e:
                err_parts.append(f"html:{e}")
            else:
                links = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html)
                snippets = re.findall(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', html)
                for i, (href, title_html) in enumerate(links[:limit]):
                    title = re.sub(r"<[^>]+>", "", title_html).strip()
                    snip = re.sub(r"<[^>]+>", "", snippets[i]).strip() if i < len(snippets) else ""
                    results.append({"title": title, "url": href, "snippet": snip})
                if results:
                    engine = "duckduckgo_html"

        if not results:
            msg = "web_search: no results"
            if err_parts:
                msg += " (" + "; ".join(err_parts) + ")"
            return ToolResult(ok=False, error=msg)

        return ToolResult(
            ok=True,
            data={"results": results, "engine": engine, "fetched_at_ms": _now_ms()},
            evidence_snippets=[_snippet("web_search", f"query={q}")],
        )

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        if prior is None:
            return ToolResult(ok=False, error="verify requires prior ToolResult")
        results = prior.data.get("results", [])
        ok = isinstance(results, list) and len(results) >= 0
        return ToolResult(ok=ok, data={"results_count": len(results)})

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        return ToolResult(ok=True, data={"note": "No rollback needed for read-only tool."})


class WebFetchTool(Tool):
    spec = ToolSpec(
        name="web_fetch",
        args_schema={"url": "string", "max_bytes": "int (default 200000)"},
        permissions=["network:outbound"],
        supports_modes=("dry_run", "execute", "verify", "rollback"),
        risk_level="low",
        estimated_cost={"time_ms": 1500},
    )

    def dry_run(self, call: ToolCall) -> ToolResult:
        url = str(call.args.get("url", "")).strip()
        max_bytes = int(call.args.get("max_bytes", 200_000))
        return ToolResult(ok=True, data={"url": url, "max_bytes": max_bytes, "note": "Would fetch URL content."})

    def execute(self, call: ToolCall) -> ToolResult:
        url = str(call.args.get("url", "")).strip()
        max_bytes = int(call.args.get("max_bytes", 200_000))
        if not url:
            return ToolResult(ok=False, error="Missing args.url")

        sandbox = os.environ.get("SANDBOX", "0") == "1"
        if sandbox:
            return ToolResult(
                ok=True,
                data={"url": url, "status": 200, "text": "SANDBOX page", "truncated": False},
                evidence_snippets=[_snippet("sandbox", f"web_fetch({url}) mocked")],
            )

        req = urllib.request.Request(url, headers={"User-Agent": "lifers_brain/0.1"})
        try:
            with _http_open(req, timeout=call.timeout_ms / 1000) as resp:
                status = getattr(resp, "status", 200)
                raw = resp.read(max_bytes + 1)
        except Exception as e:
            return ToolResult(ok=False, error=f"web_fetch failed: {e}")

        truncated = len(raw) > max_bytes
        if truncated:
            raw = raw[:max_bytes]
        text = raw.decode("utf-8", errors="ignore")
        return ToolResult(
            ok=True,
            data={"url": url, "status": status, "text": text, "truncated": truncated},
            evidence_snippets=[_snippet(url, text[:500], {"truncated": truncated})],
        )

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        if prior is None:
            return ToolResult(ok=False, error="verify requires prior ToolResult")
        status = prior.data.get("status")
        ok = status == 200 or status == 301 or status == 302
        return ToolResult(ok=ok, data={"status": status})

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        return ToolResult(ok=True, data={"note": "No rollback needed for read-only tool."})


class ExtractEvidenceTool(Tool):
    spec = ToolSpec(
        name="extract_evidence",
        args_schema={"text": "string", "max_snippets": "int (default 5)"},
        permissions=[],
        supports_modes=("dry_run", "execute", "verify", "rollback"),
        risk_level="low",
        estimated_cost={"time_ms": 50},
    )

    def dry_run(self, call: ToolCall) -> ToolResult:
        max_snippets = int(call.args.get("max_snippets", 5))
        return ToolResult(ok=True, data={"max_snippets": max_snippets, "note": "Would extract evidence snippets."})

    def execute(self, call: ToolCall) -> ToolResult:
        text = str(call.args.get("text", ""))
        max_snippets = int(call.args.get("max_snippets", 5))
        if not text:
            return ToolResult(ok=False, error="Missing args.text")
        # Simple heuristic: split paragraphs and keep non-empty top chunks.
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        snippets = parts[:max_snippets]
        ev = [_snippet("extract", s[:800], {"i": i}) for i, s in enumerate(snippets)]
        return ToolResult(ok=True, data={"snippets": ev}, evidence_snippets=ev)

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        if prior is None:
            return ToolResult(ok=False, error="verify requires prior ToolResult")
        snips = prior.data.get("snippets", [])
        ok = isinstance(snips, list) and len(snips) > 0
        return ToolResult(ok=ok, data={"snippets_count": len(snips)})

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        return ToolResult(ok=True, data={"note": "No rollback needed."})


class FsReadTool(Tool):
    spec = ToolSpec(
        name="fs_read",
        args_schema={"path": "string", "max_bytes": "int (default 200000)"},
        permissions=["fs:read"],
        supports_modes=("dry_run", "execute", "verify", "rollback"),
        risk_level="low",
    )

    def dry_run(self, call: ToolCall) -> ToolResult:
        path = str(call.args.get("path", "")).strip()
        return ToolResult(ok=True, data={"path": path, "note": "Would read file or list directory."})

    def execute(self, call: ToolCall) -> ToolResult:
        import pathlib

        path = str(call.args.get("path", "")).strip()
        max_bytes = int(call.args.get("max_bytes", 200_000))
        if not path:
            return ToolResult(ok=False, error="Missing args.path")
        p = pathlib.Path(path)
        if not p.exists():
            return ToolResult(ok=False, error=f"Path not found: {path}")
        if p.is_dir():
            items = []
            for child in p.iterdir():
                items.append({"name": child.name, "is_dir": child.is_dir()})
            return ToolResult(ok=True, data={"path": path, "type": "dir", "items": items})
        try:
            raw = p.read_bytes()
        except OSError as e:
            return ToolResult(ok=False, error=f"fs_read: {e}")
        truncated = len(raw) > max_bytes
        if truncated:
            raw = raw[:max_bytes]
        text = raw.decode("utf-8", errors="ignore")
        return ToolResult(ok=True, data={"path": path, "type": "file", "text": text, "truncated": truncated})

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        if prior is None:
            return ToolResult(ok=False, error="verify requires prior ToolResult")
        ok = prior.ok and ("type" in prior.data)
        return ToolResult(ok=ok, data={"type": prior.data.get("type")})

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        return ToolResult(ok=True, data={"note": "No rollback for read-only."})


class VisionDigestTool(Tool):
    """图像路径（LIFERS_ROOT 下）→ 轻量元数据摘要，供 prompt 注入。"""

    spec = ToolSpec(
        name="vision_digest",
        args_schema={"rel_path": "string under LIFERS_ROOT (.png/.jpg/.webp/…)"},
        permissions=["fs:read"],
        supports_modes=("dry_run", "execute", "verify", "rollback"),
        risk_level="low",
    )

    def dry_run(self, call: ToolCall) -> ToolResult:
        rel = str(call.args.get("rel_path", "")).strip()
        return ToolResult(ok=True, data={"rel_path": rel, "note": "Would summarize image metadata under LIFERS_ROOT."})

    def execute(self, call: ToolCall) -> ToolResult:
        from lifers_brain.vision_support import summarize_image_under_root

        rel = str(call.args.get("rel_path", "")).strip()
        if not rel:
            return ToolResult(ok=False, error="Missing args.rel_path")
        root = Path(_lifers_root()).resolve()
        info = summarize_image_under_root(root, rel)
        if not info.get("ok"):
            return ToolResult(ok=False, error=str(info.get("error") or "vision_digest failed"))
        return ToolResult(
            ok=True,
            data=info,
            evidence_snippets=[_snippet("vision_digest", str(info.get("caption_zh") or ""))],
        )

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        if prior is None:
            return ToolResult(ok=False, error="verify requires prior ToolResult")
        ok = prior.ok and bool(prior.data.get("caption_zh"))
        return ToolResult(ok=ok, data={"has_caption": ok})

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        return ToolResult(ok=True, data={"note": "No rollback for read-only vision_digest."})


class FsWritePatchTool(Tool):
    spec = ToolSpec(
        name="fs_write_patch",
        args_schema={"path": "string", "new_text": "string"},
        permissions=["fs:write"],
        supports_modes=("dry_run", "execute", "verify", "rollback"),
        risk_level="high",
    )

    def dry_run(self, call: ToolCall) -> ToolResult:
        path = str(call.args.get("path", "")).strip()
        new_text = str(call.args.get("new_text", ""))
        return ToolResult(ok=True, data={"path": path, "new_bytes": len(new_text.encode("utf-8"))})

    def execute(self, call: ToolCall) -> ToolResult:
        import pathlib

        from lifers_brain.safe_file_backup import safe_replace_file_text

        path = str(call.args.get("path", "")).strip()
        new_text = str(call.args.get("new_text", ""))
        if not path:
            return ToolResult(ok=False, error="Missing args.path")
        p = pathlib.Path(path).expanduser()
        try:
            p = p.resolve()
        except OSError:
            p = pathlib.Path(path).expanduser()
        rec = safe_replace_file_text(p, new_text, encoding="utf-8")
        if not rec.get("ok"):
            return ToolResult(ok=False, error=str(rec.get("error") or "write failed"))
        hint = {
            "journal_dir": rec.get("journal_dir"),
            "backup_path": rec.get("backup_path"),
            "restore_to": rec.get("restore_to"),
            "was_new": rec.get("was_new"),
        }
        return ToolResult(
            ok=True,
            data={**hint, "path": rec.get("path", path)},
            side_effects=[{"type": "file_write", "path": rec.get("path", path)}],
            rollback_hint=hint,
        )

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        import pathlib

        from lifers_brain.safe_file_backup import commit_journal

        if prior is None:
            return ToolResult(ok=False, error="verify requires prior ToolResult")
        path = prior.data.get("path")
        if not path:
            return ToolResult(ok=False, error="Missing prior.data.path")
        p = pathlib.Path(path)
        ok = p.exists() and p.is_file()
        if ok:
            jd = prior.data.get("journal_dir") or (prior.rollback_hint or {}).get("journal_dir")
            if jd:
                commit_journal(str(jd))
        return ToolResult(ok=ok, data={"path": str(p), "exists": p.exists()})

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        from lifers_brain.safe_file_backup import restore_from_hint, rollback_hint_from_prior

        if prior is None:
            return ToolResult(ok=False, error="rollback requires prior ToolResult")
        hint = rollback_hint_from_prior(prior)
        if not hint.get("journal_dir") and not hint.get("backup_path") and hint.get("was_new") is not True:
            return ToolResult(ok=True, data={"note": "No backup available; nothing to rollback."})
        res = restore_from_hint(hint)
        return ToolResult(ok=bool(res.get("ok", True)), data=res)


class LifersWorkspaceWriteTool(Tool):
    """
    以 LIFERS_ROOT 为根的相对路径整文件写入（与 fs_write_patch 等价，但便于只给 rel_path）。
    SANDBOX=1 时由注册表拒绝 execute；非沙盒下可改自身 Python。
    """

    spec = ToolSpec(
        name="lifers_workspace_write",
        args_schema={"rel_path": "string under LIFERS_ROOT", "new_text": "string full file"},
        permissions=["fs:write", "lifers:self_code"],
        supports_modes=("dry_run", "execute", "verify", "rollback"),
        risk_level="high",
    )

    def dry_run(self, call: ToolCall) -> ToolResult:
        rel = str(call.args.get("rel_path", "")).strip()
        new_text = str(call.args.get("new_text", ""))
        return ToolResult(ok=True, data={"rel_path": rel, "new_bytes": len(new_text.encode("utf-8"))})

    def execute(self, call: ToolCall) -> ToolResult:
        import pathlib

        from lifers_brain.safe_file_backup import safe_replace_file_text

        rel = str(call.args.get("rel_path", "")).strip().replace("\\", "/").lstrip("/")
        new_text = str(call.args.get("new_text", ""))
        if not rel:
            return ToolResult(ok=False, error="Missing args.rel_path")
        if ".." in pathlib.Path(rel).parts:
            return ToolResult(ok=False, error="rel_path must not contain ..")
        root_s = os.environ.get("LIFERS_ROOT", "").strip() or "."
        root = pathlib.Path(root_s).resolve()
        target = (root / rel).resolve()
        if not str(target).startswith(str(root)):
            return ToolResult(ok=False, error="rel_path escapes LIFERS_ROOT")
        rec = safe_replace_file_text(target, new_text, encoding="utf-8")
        if not rec.get("ok"):
            return ToolResult(ok=False, error=str(rec.get("error") or "write failed"))
        hint = {
            "journal_dir": rec.get("journal_dir"),
            "backup_path": rec.get("backup_path"),
            "restore_to": rec.get("restore_to"),
            "was_new": rec.get("was_new"),
        }
        return ToolResult(
            ok=True,
            data={"rel_path": rel, **hint, "path": rec.get("path", str(target))},
            side_effects=[{"type": "lifers_workspace_write", "path": rec.get("path", str(target))}],
            rollback_hint=hint,
        )

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        import pathlib

        from lifers_brain.safe_file_backup import commit_journal

        if prior is None:
            return ToolResult(ok=False, error="verify requires prior ToolResult")
        path = prior.data.get("path")
        if not path:
            return ToolResult(ok=False, error="Missing prior.data.path")
        p = pathlib.Path(path)
        ok = p.is_file()
        if ok:
            jd = prior.data.get("journal_dir") or (prior.rollback_hint or {}).get("journal_dir")
            if jd:
                commit_journal(str(jd))
        return ToolResult(ok=ok, data={"path": str(p), "exists": p.is_file()})

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        from lifers_brain.safe_file_backup import restore_from_hint, rollback_hint_from_prior

        if prior is None:
            return ToolResult(ok=False, error="rollback requires prior ToolResult")
        hint = rollback_hint_from_prior(prior)
        if not hint.get("journal_dir") and not hint.get("backup_path") and hint.get("was_new") is not True:
            return ToolResult(ok=True, data={"note": "No backup available; nothing to rollback."})
        res = restore_from_hint(hint)
        return ToolResult(ok=bool(res.get("ok", True)), data=res)


class CmdRunTool(Tool):
    spec = ToolSpec(
        name="cmd_run",
        args_schema={"cmd": "string", "cwd": "string (optional)"},
        permissions=["proc:exec"],
        supports_modes=("dry_run", "execute", "verify", "rollback"),
        risk_level="high",
    )

    def dry_run(self, call: ToolCall) -> ToolResult:
        return ToolResult(ok=True, data={"cmd": call.args.get("cmd", ""), "cwd": call.args.get("cwd", "")})

    def execute(self, call: ToolCall) -> ToolResult:
        import subprocess

        cmd = str(call.args.get("cmd", "")).strip()
        cwd = str(call.args.get("cwd", "")).strip() or None
        if not cmd:
            return ToolResult(ok=False, error="Missing args.cmd")
        sandbox = os.environ.get("SANDBOX", "0") == "1"
        if sandbox:
            return ToolResult(ok=True, data={"cmd": cmd, "cwd": cwd, "exit_code": 0, "stdout": "SANDBOX"})
        # Minimal safety: allowlist via regex if provided.
        allow = os.environ.get("CMD_ALLOW_REGEX", "").strip()
        if allow:
            if re.search(allow, cmd) is None:
                return ToolResult(ok=False, error="cmd_run blocked by CMD_ALLOW_REGEX")
        try:
            p = subprocess.run(
                cmd,
                cwd=cwd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=call.timeout_ms / 1000,
            )
        except Exception as e:
            return ToolResult(ok=False, error=f"cmd_run failed: {e}")
        return ToolResult(
            ok=p.returncode == 0,
            data={"cmd": cmd, "cwd": cwd, "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr},
            side_effects=[{"type": "process", "cmd": cmd, "cwd": cwd}],
            rollback_hint={"note": "Command side-effects may be irreversible; use filesystem backups or git."},
        )

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        if prior is None:
            return ToolResult(ok=False, error="verify requires prior ToolResult")
        return ToolResult(ok=prior.data.get("exit_code", 1) == 0, data={"exit_code": prior.data.get("exit_code")})

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        return ToolResult(ok=True, data={"note": "No generic rollback for arbitrary commands."})


# Simulation/robot tools: start as sandbox-friendly stubs; upgrade to real sim/ROS later.
class SenseSnapshotTool(Tool):
    spec = ToolSpec(
        name="sense_snapshot",
        args_schema={"source": "string (optional)"},
        permissions=["robot:sense"],
        supports_modes=("dry_run", "execute", "verify", "rollback"),
        risk_level="low",
    )

    def dry_run(self, call: ToolCall) -> ToolResult:
        return ToolResult(ok=True, data={"note": "Would capture sensor snapshot."})

    def execute(self, call: ToolCall) -> ToolResult:
        cmd = os.environ.get("ROBOT_SENSE_CMD", "").strip()
        if cmd:
            try:
                payload = json.dumps({"tool": "sense_snapshot", "args": call.args}, ensure_ascii=False)
                p = subprocess.run(
                    cmd,
                    shell=True,
                    input=payload,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if p.returncode != 0:
                    return ToolResult(ok=False, error=f"ROBOT_SENSE_CMD rc={p.returncode} {p.stderr[:400]}")
                out = (p.stdout or "").strip()
                if out:
                    data = json.loads(out)
                    return ToolResult(ok=True, data=data, evidence_snippets=[_snippet("robot_sense", "external")])
            except Exception as e:
                return ToolResult(ok=False, error=f"sense_snapshot external failed: {e}")

        sandbox = os.environ.get("SANDBOX", "0") == "1"
        if sandbox:
            return ToolResult(ok=True, data={"pose": {"x": 0, "y": 0}, "objects": [], "humans": []})
        return ToolResult(ok=True, data={"pose": {"x": 0, "y": 0}, "objects": [], "humans": []})

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        if prior is None:
            return ToolResult(ok=False, error="verify requires prior ToolResult")
        ok = "pose" in prior.data
        return ToolResult(ok=ok, data={"has_pose": ok})

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        return ToolResult(ok=True, data={"note": "No rollback."})


class MotionPlanTool(Tool):
    spec = ToolSpec(
        name="motion_plan",
        args_schema={"goal": "object", "constraints": "object (optional)"},
        permissions=["robot:plan"],
        supports_modes=("dry_run", "execute", "verify", "rollback"),
        risk_level="low",
    )

    def dry_run(self, call: ToolCall) -> ToolResult:
        return ToolResult(ok=True, data={"note": "Would produce a trajectory plan."})

    def execute(self, call: ToolCall) -> ToolResult:
        goal = call.args.get("goal", {})
        traj = [{"t": 0.0, "x": 0.0, "y": 0.0}, {"t": 1.0, "x": 1.0, "y": 1.0}]
        return ToolResult(ok=True, data={"goal": goal, "trajectory": traj})

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        if prior is None:
            return ToolResult(ok=False, error="verify requires prior ToolResult")
        ok = isinstance(prior.data.get("trajectory"), list)
        return ToolResult(ok=ok, data={"trajectory_len": len(prior.data.get("trajectory", []))})

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        return ToolResult(ok=True, data={"note": "No rollback for planning."})


class MotionExecuteTool(Tool):
    spec = ToolSpec(
        name="motion_execute",
        args_schema={"trajectory": "array", "safety": "object (optional)"},
        permissions=["robot:act"],
        supports_modes=("dry_run", "execute", "verify", "rollback"),
        risk_level="high",
    )

    def dry_run(self, call: ToolCall) -> ToolResult:
        return ToolResult(ok=True, data={"note": "Would execute trajectory.", "steps": len(call.args.get("trajectory", []))})

    def execute(self, call: ToolCall) -> ToolResult:
        sandbox = os.environ.get("SANDBOX", "0") == "1"
        if sandbox:
            return ToolResult(ok=True, data={"executed": True, "events": [], "sandbox": True}, side_effects=[{"type": "robot_motion", "sandbox": True}])
        cmd = os.environ.get("ROBOT_ACT_CMD", "").strip()
        if cmd:
            try:
                payload = json.dumps({"tool": "motion_execute", "args": call.args}, ensure_ascii=False)
                p = subprocess.run(
                    cmd,
                    shell=True,
                    input=payload,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if p.returncode != 0:
                    return ToolResult(ok=False, error=f"ROBOT_ACT_CMD rc={p.returncode} {p.stderr[:400]}")
                out = (p.stdout or "").strip()
                if out:
                    data = json.loads(out)
                    return ToolResult(ok=True, data=data, side_effects=[{"type": "robot_motion", "external": True}])
            except Exception as e:
                return ToolResult(ok=False, error=f"motion_execute external failed: {e}")
        return ToolResult(ok=True, data={"executed": True, "events": []}, side_effects=[{"type": "robot_motion"}])

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        if prior is None:
            return ToolResult(ok=False, error="verify requires prior ToolResult")
        return ToolResult(ok=bool(prior.data.get("executed")), data={"executed": prior.data.get("executed")})

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        return ToolResult(ok=True, data={"note": "Rollback = stop and return to safe pose (to be implemented)."})


class ManipulateTool(Tool):
    spec = ToolSpec(
        name="manipulate",
        args_schema={"action": "pick|place|handover", "target": "object"},
        permissions=["robot:act"],
        supports_modes=("dry_run", "execute", "verify", "rollback"),
        risk_level="high",
    )

    def dry_run(self, call: ToolCall) -> ToolResult:
        return ToolResult(ok=True, data={"note": "Would manipulate object.", "action": call.args.get("action")})

    def execute(self, call: ToolCall) -> ToolResult:
        sandbox = os.environ.get("SANDBOX", "0") == "1"
        if sandbox:
            return ToolResult(ok=True, data={"success": True, "sandbox": True}, side_effects=[{"type": "manipulate", "sandbox": True}])
        cmd = os.environ.get("ROBOT_ACT_CMD", "").strip()
        if cmd:
            try:
                payload = json.dumps({"tool": "manipulate", "args": call.args}, ensure_ascii=False)
                p = subprocess.run(
                    cmd,
                    shell=True,
                    input=payload,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if p.returncode != 0:
                    return ToolResult(ok=False, error=f"ROBOT_ACT_CMD rc={p.returncode} {p.stderr[:400]}")
                out = (p.stdout or "").strip()
                if out:
                    data = json.loads(out)
                    return ToolResult(ok=True, data=data, side_effects=[{"type": "manipulate", "external": True}])
            except Exception as e:
                return ToolResult(ok=False, error=f"manipulate external failed: {e}")
        return ToolResult(ok=True, data={"success": True}, side_effects=[{"type": "manipulate"}])

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        if prior is None:
            return ToolResult(ok=False, error="verify requires prior ToolResult")
        return ToolResult(ok=bool(prior.data.get("success")), data={"success": prior.data.get("success")})

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        return ToolResult(ok=True, data={"note": "Rollback = release and retreat (to be implemented)."})


class SafetyStopTool(Tool):
    spec = ToolSpec(
        name="safety_stop",
        args_schema={},
        permissions=["robot:stop"],
        supports_modes=("dry_run", "execute", "verify", "rollback"),
        risk_level="high",
    )

    def dry_run(self, call: ToolCall) -> ToolResult:
        return ToolResult(ok=True, data={"note": "Would emergency stop."})

    def execute(self, call: ToolCall) -> ToolResult:
        return ToolResult(ok=True, data={"stopped": True}, side_effects=[{"type": "emergency_stop"}])

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        if prior is None:
            return ToolResult(ok=False, error="verify requires prior ToolResult")
        return ToolResult(ok=bool(prior.data.get("stopped")), data={"stopped": prior.data.get("stopped")})

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        return ToolResult(ok=True, data={"note": "Recovery procedure depends on hardware; implement later."})


class KbUpsertTool(Tool):
    spec = ToolSpec(
        name="kb_upsert",
        args_schema={
            "items": "array of {type,content,importance,source,key(optional)}",
        },
        permissions=["kb:write"],
        supports_modes=("dry_run", "execute", "verify", "rollback"),
        risk_level="medium",
    )

    def dry_run(self, call: ToolCall) -> ToolResult:
        items = call.args.get("items", [])
        n = len(items) if isinstance(items, list) else 0
        return ToolResult(ok=True, data={"count": n, "note": "Would write items into long-term memory (SQLite)."})

    def execute(self, call: ToolCall) -> ToolResult:
        import sqlite3

        items = call.args.get("items", [])
        if not isinstance(items, list) or not items:
            return ToolResult(ok=False, error="Missing args.items (non-empty list)")

        db_path = _memory_db_path()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        inserted: List[int] = []
        skipped_hashes: List[str] = []
        updated_keys: List[str] = []
        ts = _now_ms()
        with sqlite3.connect(db_path) as con:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  type TEXT NOT NULL,
                  content_hash TEXT,
                  mem_key TEXT,
                  content_json TEXT NOT NULL,
                  content_text TEXT NOT NULL,
                  importance REAL NOT NULL,
                  source TEXT NOT NULL,
                  ts_ms INTEGER NOT NULL
                );
                """
            )
            cols = [r[1] for r in con.execute("PRAGMA table_info(memories)").fetchall()]
            if "content_hash" not in cols:
                con.execute("ALTER TABLE memories ADD COLUMN content_hash TEXT;")
            if "mem_key" not in cols:
                con.execute("ALTER TABLE memories ADD COLUMN mem_key TEXT;")
            # Legacy UNIQUE(content_hash) breaks keyed UPDATEs when another row already has the new hash.
            row = con.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
                ("idx_mem_hash",),
            ).fetchone()
            if row and row[0] and "UNIQUE" in row[0].upper():
                con.execute("DROP INDEX IF EXISTS idx_mem_hash")
            con.execute("CREATE INDEX IF NOT EXISTS idx_mem_hash ON memories(content_hash);")
            rowk = con.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
                ("idx_mem_key",),
            ).fetchone()
            if rowk and rowk[0] and "UNIQUE" in rowk[0].upper():
                con.execute("DROP INDEX IF EXISTS idx_mem_key")
            con.execute("CREATE INDEX IF NOT EXISTS idx_mem_key ON memories(mem_key);")
            for it in items:
                t = str(it.get("type", "fact"))
                content = it.get("content", "")
                imp = float(it.get("importance", 0.5))
                src = str(it.get("source", "tool"))
                key = it.get("key")
                cj = json.dumps(content, ensure_ascii=False).replace("\x00", "")
                ct = cj if isinstance(content, (dict, list)) else str(content).replace("\x00", "")
                ch = hashlib.sha256((t + "\n" + cj).encode("utf-8")).hexdigest()

                # Upsert by key (e.g. URL) to support "update" semantics.
                if isinstance(key, str) and key.strip():
                    key = key.strip()
                    exists_k = con.execute("SELECT id FROM memories WHERE mem_key = ?", (key,)).fetchone()
                    if exists_k:
                        row_id = int(exists_k[0])
                        dup_h = con.execute(
                            "SELECT id FROM memories WHERE content_hash = ? AND id <> ?",
                            (ch, row_id),
                        ).fetchone()
                        if dup_h:
                            skipped_hashes.append(ch)
                            continue
                        con.execute(
                            "UPDATE memories SET type=?, content_hash=?, content_json=?, content_text=?, importance=?, source=?, ts_ms=? WHERE mem_key=?",
                            (t, ch, cj, ct, imp, src, ts, key),
                        )
                        updated_keys.append(key)
                        continue

                exists = con.execute("SELECT id FROM memories WHERE content_hash = ?", (ch,)).fetchone()
                if exists:
                    skipped_hashes.append(ch)
                    continue
                cur = con.execute(
                    "INSERT INTO memories(type,content_hash,mem_key,content_json,content_text,importance,source,ts_ms) VALUES(?,?,?,?,?,?,?,?)",
                    (t, ch, key, cj, ct, imp, src, ts),
                )
                inserted.append(int(cur.lastrowid))
        return ToolResult(
            ok=True,
            data={
                "inserted_ids": inserted,
                "skipped": len(skipped_hashes),
                "updated": len(updated_keys),
                "db_path": db_path,
            },
            side_effects=[{"type": "kb_write", "count": len(inserted), "db_path": db_path}],
            rollback_hint={"delete_ids": inserted, "db_path": db_path},
        )

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        if prior is None:
            return ToolResult(ok=False, error="verify requires prior ToolResult")
        ids = prior.data.get("inserted_ids", [])
        ok = isinstance(ids, list) and len(ids) > 0
        return ToolResult(ok=ok, data={"inserted_count": len(ids)})

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        import sqlite3

        if prior is None:
            return ToolResult(ok=False, error="rollback requires prior ToolResult")
        ids = prior.rollback_hint.get("delete_ids") or prior.data.get("inserted_ids") or []
        db_path = prior.rollback_hint.get("db_path") or prior.data.get("db_path")
        if not db_path or not ids:
            return ToolResult(ok=True, data={"note": "No kb rollback needed."})
        with sqlite3.connect(db_path) as con:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute(
                f"DELETE FROM memories WHERE id IN ({','.join(['?']*len(ids))})",
                tuple(ids),
            )
        return ToolResult(ok=True, data={"deleted_ids": ids})


class KbSearchTool(Tool):
    spec = ToolSpec(
        name="kb_search",
        args_schema={"query": "string", "k": "int (default 6)"},
        permissions=["kb:read"],
        supports_modes=("dry_run", "execute", "verify", "rollback"),
        risk_level="low",
    )

    def dry_run(self, call: ToolCall) -> ToolResult:
        return ToolResult(ok=True, data={"query": call.args.get("query", ""), "note": "Would search long-term memory."})

    def execute(self, call: ToolCall) -> ToolResult:
        from lifers_brain.memory import fts5_search

        q = str(call.args.get("query", "")).strip()
        k = int(call.args.get("k", 6))
        if not q:
            return ToolResult(ok=False, error="Missing args.query")
        db_path = _memory_db_path()
        if not os.path.exists(db_path):
            return ToolResult(ok=True, data={"items": [], "db_path": db_path})

        items = fts5_search(db_path, q, k=k)
        return ToolResult(ok=True, data={"items": items, "db_path": db_path}, evidence_snippets=[_snippet("kb_search", f"query={q}")])

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        if prior is None:
            return ToolResult(ok=False, error="verify requires prior ToolResult")
        items = prior.data.get("items", [])
        ok = isinstance(items, list)
        return ToolResult(ok=ok, data={"count": len(items)})

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        return ToolResult(ok=True, data={"note": "No rollback needed."})


class SimRunTool(Tool):
    spec = ToolSpec(
        name="sim_run",
        args_schema={"task_id": "string", "runs": "int (default 10)"},
        permissions=["sim:run"],
        supports_modes=("dry_run", "execute", "verify", "rollback"),
        risk_level="medium",
    )

    def dry_run(self, call: ToolCall) -> ToolResult:
        return ToolResult(ok=True, data={"task_id": call.args.get("task_id", ""), "note": "Would run sim task."})

    def execute(self, call: ToolCall) -> ToolResult:
        from pathlib import Path as _Path

        from .sim_exec import default_executor, load_tasks

        from .stack_env import load_stack

        task_id = str(call.args.get("task_id", "")).strip()
        root = _Path(_lifers_root())
        def_runs = int((load_stack(root).get("robot") or {}).get("default_sim_runs", 10))
        runs = int(call.args.get("runs", def_runs))
        if not task_id:
            return ToolResult(ok=False, error="Missing args.task_id")

        tasks = load_tasks(root)
        task = next((t for t in tasks if t.get("id") == task_id), None)
        if task is None:
            return ToolResult(ok=False, error=f"Unknown task_id: {task_id}")

        ex = default_executor()
        r = ex.run(task, runs=runs)
        report = {
            "task_id": task_id,
            "runs": runs,
            "success_rate": r.success_rate,
            "metrics": r.metrics,
            "task": task,
        }
        return ToolResult(ok=True, data=report, side_effects=[{"type": "sim_run", "task_id": task_id, "runs": runs}])

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        if prior is None:
            return ToolResult(ok=False, error="verify requires prior ToolResult")
        sr = float(prior.data.get("success_rate", 0.0))
        return ToolResult(ok=sr >= 0.0, data={"success_rate": sr})

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        return ToolResult(ok=True, data={"note": "No rollback for sim runs."})


class KbPruneTool(Tool):
    spec = ToolSpec(
        name="kb_prune",
        args_schema={"min_importance": "float (default 0.15)", "older_than_days": "int (default 30)", "limit": "int (default 500)"},
        permissions=["kb:write"],
        supports_modes=("dry_run", "execute", "verify", "rollback"),
        risk_level="medium",
    )

    def dry_run(self, call: ToolCall) -> ToolResult:
        return ToolResult(ok=True, data={"note": "Would prune long-term memory.", **call.args})

    def execute(self, call: ToolCall) -> ToolResult:
        from pathlib import Path as _Path

        from .memory import LongTermMemory

        ltm = LongTermMemory(_Path(_memory_db_path()))
        res = ltm.prune(
            min_importance=float(call.args.get("min_importance", 0.15)),
            older_than_days=int(call.args.get("older_than_days", 30)),
            limit=int(call.args.get("limit", 500)),
        )
        return ToolResult(ok=True, data=res, side_effects=[{"type": "kb_prune", "deleted": res.get("deleted", 0)}])

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        if prior is None:
            return ToolResult(ok=False, error="verify requires prior ToolResult")
        return ToolResult(ok=True, data={"deleted": prior.data.get("deleted", 0)})

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        return ToolResult(ok=True, data={"note": "Prune rollback not supported (append-only design)."})


class KbCompactTool(Tool):
    spec = ToolSpec(
        name="kb_compact",
        args_schema={"url": "string", "k": "int (default 6)"},
        permissions=["kb:write"],
        supports_modes=("dry_run", "execute", "verify", "rollback"),
        risk_level="medium",
    )

    def dry_run(self, call: ToolCall) -> ToolResult:
        return ToolResult(ok=True, data={"note": "Would compact tool_results into a fact.", **call.args})

    def execute(self, call: ToolCall) -> ToolResult:
        import sqlite3

        url = str(call.args.get("url", "")).strip()
        k = int(call.args.get("k", 6))
        if not url:
            return ToolResult(ok=False, error="Missing args.url")

        db_path = _memory_db_path()
        if not os.path.exists(db_path):
            return ToolResult(ok=False, error="KB not initialized")

        with sqlite3.connect(db_path) as con:
            con.execute("PRAGMA journal_mode=WAL;")
            rows = con.execute(
                "SELECT content_json FROM memories WHERE content_text LIKE ? ORDER BY ts_ms DESC LIMIT ?",
                (f"%{url}%", k),
            ).fetchall()
        if not rows:
            return ToolResult(ok=False, error="No matching memories to compact")

        snippets: List[str] = []
        for (cj,) in rows:
            try:
                obj = json.loads(cj)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("event") == "evidence":
                for s in obj.get("snippets", [])[:5]:
                    t = str(s.get("text", "")).strip()
                    if t:
                        snippets.append(t[:240])
        snippets = snippets[:5]

        fact = {"event": "fact_summary", "url": url, "bullets": snippets or ["(no evidence snippets captured)"]}

        up = KbUpsertTool()
        res = up.execute(
            ToolCall(
                name="kb_upsert",
                args={
                    "items": [
                        {
                            "type": "fact",
                            "key": f"{url}#fact",
                            "content": fact,
                            "importance": 0.85,
                            "source": "tool:kb_compact",
                        }
                    ]
                },
                expected_effect="write fact summary",
                mode="execute",
            )
        )
        return ToolResult(
            ok=res.ok,
            data={"url": url, "fact_written": res.data},
            side_effects=res.side_effects,
            error=res.error,
        )

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        if prior is None:
            return ToolResult(ok=False, error="verify requires prior ToolResult")
        return ToolResult(ok=prior.ok, data={"ok": prior.ok})

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        return ToolResult(ok=True, data={"note": "Compact rollback not supported (write-only)."})


def _http_get_text(url: str, timeout: float = 10.0) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "LifersBrain/1.0 (local; https://github.com/)",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    with _http_open(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


class RealWorldTool(Tool):
    """本机时钟 + wttr.in 天气 + OSM/Nominatim 地图检索（无 API Key，适合本能实时操作）。"""

    spec = ToolSpec(
        name="real_world",
        args_schema={"action": "clock|weather|map", "query": "string optional", "location": "string optional"},
        permissions=["net:read", "time:read"],
        supports_modes=("dry_run", "execute", "verify", "rollback"),
        risk_level="low",
    )

    def dry_run(self, call: ToolCall) -> ToolResult:
        act = str(call.args.get("action", "clock")).strip().lower()
        return ToolResult(ok=True, data={"would": act})

    def execute(self, call: ToolCall) -> ToolResult:
        act = str(call.args.get("action", "clock")).strip().lower()
        if act == "clock":
            import datetime as _dt

            now = _dt.datetime.now().astimezone()
            wd_zh = "一二三四五六日"[now.weekday()]
            return ToolResult(
                ok=True,
                data={
                    "real_world": "clock",
                    "iso": now.isoformat(),
                    "local": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "tz": str(now.tzname() or ""),
                    "weekday_zh": wd_zh,
                    "unix": int(now.timestamp()),
                },
            )

        if act == "weather":
            loc = str(call.args.get("location") or "").strip()
            q = str(call.args.get("query") or "").strip()
            if not loc and q:
                loc = q
            try:
                if loc:
                    seg = urllib.parse.quote(loc)
                    url = f"https://wttr.in/{seg}?format=j1"
                else:
                    url = "https://wttr.in/?format=j1"
                raw = _http_get_text(url, timeout=12.0)
                data = json.loads(raw)
                cur_raw = data.get("current_condition") or [{}]
                if isinstance(cur_raw, dict):
                    c0 = cur_raw
                else:
                    c0 = cur_raw[0] if isinstance(cur_raw, list) and cur_raw else {}
                temp_c = c0.get("temp_C") or c0.get("Temp_C") or c0.get("FeelsLikeC") or "?"
                wd = c0.get("weatherDesc")
                desc = ""
                if isinstance(wd, list) and wd:
                    desc = str(wd[0].get("value", "") if isinstance(wd[0], dict) else wd[0])
                elif isinstance(wd, str):
                    desc = wd
                area = (data.get("nearest_area") or [{}])[0]
                name = ""
                if isinstance(area, dict):
                    name = ", ".join(
                        filter(
                            None,
                            [
                                str(area.get("areaName", [""])[0] if isinstance(area.get("areaName"), list) else ""),
                                str(area.get("region", [""])[0] if isinstance(area.get("region"), list) else ""),
                            ],
                        )
                    )
                hum = c0.get("humidity") or ""
                lines = [f"气温约 {temp_c}°C", desc or "天气", f"湿度 {hum}" if hum else "", name or "（wttr.in）"]
                summary = " · ".join(x for x in lines if x)
                return ToolResult(
                    ok=True,
                    data={
                        "real_world": "weather",
                        "summary": summary,
                        "detail": data,
                        "source": "wttr.in",
                        "fetched_at_ms": _now_ms(),
                    },
                )
            except Exception as e:
                return ToolResult(ok=False, error=f"weather fetch failed: {e}")

        if act == "map":
            q = str(call.args.get("query") or "").strip()
            if len(q) < 2:
                return ToolResult(ok=False, error="map 需要地点关键词（query）")
            try:
                enc = urllib.parse.quote(q)
                url = f"https://nominatim.openstreetmap.org/search?q={enc}&format=json&limit=3"
                raw = _http_get_text(url, timeout=12.0)
                arr = json.loads(raw)
                if not isinstance(arr, list) or not arr:
                    return ToolResult(ok=False, error="未找到地点（换关键词）")
                top = arr[0]
                lat = float(top.get("lat", 0))
                lon = float(top.get("lon", 0))
                disp = str(top.get("display_name", ""))[:500]
                osm = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}&zoom=13"
                return ToolResult(
                    ok=True,
                    data={
                        "real_world": "map",
                        "display_name": disp,
                        "lat": lat,
                        "lon": lon,
                        "openstreetmap_url": osm,
                        "source": "nominatim",
                        "fetched_at_ms": _now_ms(),
                    },
                )
            except Exception as e:
                return ToolResult(ok=False, error=f"map geocode failed: {e}")

        return ToolResult(ok=False, error=f"unknown action: {act}")

    def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        if prior is None:
            return ToolResult(ok=False, error="verify requires prior ToolResult")
        return ToolResult(ok=prior.ok, data={"ok": prior.ok})

    def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
        return ToolResult(ok=True, data={"note": "No rollback for read-only real_world."})


def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(WebSearchTool())
    reg.register(WebFetchTool())
    reg.register(ExtractEvidenceTool())
    reg.register(FsReadTool())
    reg.register(FsWritePatchTool())
    reg.register(LifersWorkspaceWriteTool())
    reg.register(CmdRunTool())
    reg.register(KbUpsertTool())
    reg.register(KbSearchTool())
    reg.register(KbPruneTool())
    reg.register(KbCompactTool())
    reg.register(SimRunTool())
    reg.register(SenseSnapshotTool())
    reg.register(MotionPlanTool())
    reg.register(MotionExecuteTool())
    reg.register(ManipulateTool())
    reg.register(SafetyStopTool())
    reg.register(RealWorldTool())
    reg.register(VisionDigestTool())
    try:
        from lifers_brain.plugin_loader import register_plugins

        register_plugins(reg, Path(_lifers_root()).resolve())
    except Exception as exc:
        sys.stderr.write(f"LIFERS_PROGRESS plugin_loader error: {exc}\n")
        sys.stderr.flush()
    return reg


def print_specs(reg: ToolRegistry) -> str:
    specs = []
    for s in reg.list_specs():
        specs.append(
            {
                "name": s.name,
                "risk_level": s.risk_level,
                "supports_modes": list(s.supports_modes),
                "args_schema": s.args_schema,
                "permissions": s.permissions,
            }
        )
    return json.dumps(specs, ensure_ascii=False, indent=2)

