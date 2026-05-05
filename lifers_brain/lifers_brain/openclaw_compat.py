"""
上游对照：以 OpenClaw（github.com/openclaw/openclaw）发行与能力域为**参照**，同步 compat_ref / manifest 到本仓库。

默认 **不** 使用、不调用 OpenClaw 运行时或 npm 网关；仅把锚点与边界写进 AI 上下文。
可选 `use_external_openclaw_runtime`：若你本机另行安装 OpenClaw，才注入 OPENCLAW_WORKSPACE 等桥接 env。

compat_ref 由 scripts/sync_openclaw_release.py 对照 GitHub latest release 更新（本地镜像，非安装上游）。
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set

from .rs_integration import bootstrap_summary_for_hints


def _outbound_opener() -> urllib.request.OpenerDirector:
    """与 tools._outbound_opener 一致：LIFERS_HTTP_DIRECT=1 时跳过代理，避免假代理 10061。"""
    if os.environ.get("LIFERS_HTTP_DIRECT", "").strip().lower() in ("1", "true", "yes", "on"):
        return urllib.request.build_opener()
    try:
        px = urllib.request.getproxies()
    except Exception:
        px = {}
    if px:
        return urllib.request.build_opener(urllib.request.ProxyHandler(px))
    return urllib.request.build_opener()

# ---------------------------------------------------------------------------
# 能力命名空间（去重）
# ---------------------------------------------------------------------------

LIFERS_CONFIG_SOURCE: FrozenSet[str] = frozenset(
    {
        "brain.model_weights_path",
        "brain.sandbox",
        "runtime.LIFERS_RUNTIME",
        "robot.SIM_EXEC_CMD",
        "robot.ROBOT_SENSE_CMD",
        "robot.ROBOT_ACT_CMD",
        "instincts.automation",
        "memory.sqlite_kb",
    }
)

# 下列为「上游产品常见能力域」命名空间，用于 manifest 对照；本仓库不实现这些运行时。
OPENCLAW_UPSTREAM_SURFACE: FrozenSet[str] = frozenset(
    {
        "gateway.control_plane",
        "channels.inbox",
        "providers.hosted_llm",
        "providers.local_discovery",
        "skills.registry",
        "canvas.host",
        "cron.webhooks",
        "voice.ui",
        "migrate.cli",
    }
)

UPSTREAM_REPO = "openclaw/openclaw"
GITHUB_LATEST_API = "https://api.github.com/repos/openclaw/openclaw/releases/latest"


def resolve_openclaw_effective(ocfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    归一化 stack.openclaw：
    - track_upstream 默认 True：对照上游发行更新锚点/manifest，不运行 OpenClaw
    - use_external_openclaw_runtime 默认 False：不向本进程注入外部网关依赖
    """
    if not isinstance(ocfg, dict):
        return {}
    out = dict(ocfg)
    if not out.get("enabled"):
        return out
    out.setdefault("track_upstream", True)
    out.setdefault("use_external_openclaw_runtime", False)
    return out


def dedupe_tags(tags: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for t in tags:
        t = str(t).strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def merge_capability_sets(*lists: Iterable[str]) -> List[str]:
    return dedupe_tags([x for it in lists for x in it])


def load_openclaw_manifest(root: Path) -> Dict[str, Any]:
    p = root / "config" / "openclaw_manifest.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def openclaw_vendor_tree_path(brain_root: Path) -> Optional[Path]:
    """可选：rs/third_party/openclaw（子模块或浅克隆），仅对照阅读。"""
    p = brain_root.parent / "third_party" / "openclaw"
    try:
        if p.is_dir() and (p / ".git").exists():
            return p
    except OSError:
        pass
    return None


def rs_integrated_layout_path(brain_root: Path) -> Optional[Path]:
    """lifers_brain 上级 rs/config/integrated_layout.json（合并联接清单）。"""
    p = brain_root.parent / "config" / "integrated_layout.json"
    return p if p.is_file() else None


def claw_code_rust_vendor_root(brain_root: Path) -> Optional[Path]:
    """rs/third_party/claw_code_rust：Kali claw-code/rust 合并树（无运行时接入）。"""
    p = brain_root.parent / "third_party" / "claw_code_rust"
    return p if p.is_dir() and (p / "Cargo.toml").is_file() else None


def load_claw_code_rust_vendor_manifest(brain_root: Path) -> Dict[str, Any]:
    p = brain_root / "config" / "claw_code_rust_vendor.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def claw_code_rust_vendor_hint(brain_root: Path) -> str:
    """并入清单摘要（无密钥、无 API 配置）。"""
    root_v = claw_code_rust_vendor_root(brain_root)
    if not root_v:
        return ""
    man = load_claw_code_rust_vendor_manifest(brain_root)
    cov = str((man.get("completeness") or {}).get("approximate_coverage") or "").strip()
    merged = str(man.get("merged_at") or "").strip()
    tail = f" 并入≈{cov}（文件口径）" if cov else ""
    date_p = f"，{merged}" if merged else ""
    merge_md = root_v / "LIFERS_MERGE.md"
    doc = f"；审计说明 {merge_md.name}" if merge_md.is_file() else ""
    return (
        f"【claw-code/rust vendor】{brain_root.parent.as_posix()}/third_party/claw_code_rust"
        f"{date_p}{tail}{doc}；清单 config/claw_code_rust_vendor.json；不在 Lifers 内启用其 HTTP/API 运行时。"
    )


def merged_rs_openclaw_section(brain_root: Path) -> str:
    """
    将 rs 侧「合并工作区」写入 AI 上下文：lifers_brain + lifers + OpenClaw 对照合一说明。
    不包含任何 API Key / 外部模型配置。
    """
    sections: List[str] = []
    path = rs_integrated_layout_path(brain_root)
    if path and path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
        else:
            phil = str(data.get("philosophy", "")).strip()
            lines: List[str] = [
                "",
                "【合并联接 rs】集百家之长：把各来源纳入本 portable 根（等价于复制进树后再改编），保留自有配置；代码与资源按本仓需求改写自用，不是单纯目录指向。",
            ]
            desc = str(data.get("description", "")).strip()
            if desc:
                lines.append(desc)
            if phil:
                lines.append(phil)
            lines.append(f"  · {bootstrap_summary_for_hints(data)}")
            for r in data.get("roots") or []:
                if not isinstance(r, dict):
                    continue
                rid = str(r.get("id", "")).strip()
                role = str(r.get("role", "")).strip()
                name = str(r.get("name", "")).strip()
                j = " [junction]" if r.get("requires_junction") else ""
                lines.append(f"  · {(name or rid) + j}: {role}")
            cur = data.get("cursor") or {}
            if isinstance(cur, dict) and (
                str(cur.get("mode", "")).strip()
                or str(cur.get("note", "")).strip()
                or str(cur.get("workspace_excluded", "")).strip()
            ):
                lines.append(
                    f"  · Cursor（离线整合）: {cur.get('mode', '')} —— {cur.get('note', '')}"
                )
                if str(cur.get("settings_merge_script", "")).strip():
                    lines.append(f"    合并脚本: {cur.get('settings_merge_script')}")
                if str(cur.get("workspace_excluded", "")).strip():
                    lines.append(f"    工作区: {cur.get('workspace_excluded')}")
            oc_m = data.get("openclaw") or {}
            if isinstance(oc_m, dict) and (str(oc_m.get("mode", "")).strip() or str(oc_m.get("note", "")).strip()):
                lines.append(
                    f"  · OpenClaw 合并方式: {oc_m.get('mode', '')} —— {oc_m.get('note', '仅对照清单与锚点')}"
                )
            cons = data.get("constraints") or []
            if cons:
                lines.append("  · 约束: " + " | ".join(str(c) for c in cons[:6]))
            lines.append(f"  · 清单文件: {path.name}")
            sections.append("\n".join(lines))

    claw_h = claw_code_rust_vendor_hint(brain_root)
    if claw_h:
        sections.append(claw_h)

    return "\n\n".join(sections).strip()


def format_rs_integrated_layout_hint(brain_root: Path) -> str:
    """
    rs/config/integrated_layout.json 摘要（含 Cursor 策略等），不依赖 OpenClaw 是否启用。
    供 _context_pack / 本地 LLM 使用。
    """
    return merged_rs_openclaw_section(brain_root)


def bridge_context_line(ocfg: Dict[str, Any]) -> str:
    """单行 SYSTEM 提示。"""
    oc = resolve_openclaw_effective(ocfg)
    if not oc.get("enabled"):
        return ""
    ref = str(oc.get("compat_ref", "openclaw/openclaw main")).strip()
    core = (
        f"【上游对照 OpenClaw】仅以发行锚点 {ref} 与 manifest 对齐能力边界说明；"
        "本 Lifers 进程不调用、不依赖 OpenClaw 运行时或 npm 网关。"
    )
    if oc.get("use_external_openclaw_runtime"):
        core += (
            " （可选）若本机另行安装 OpenClaw，渠道与模型仅在其独立工作区配置，勿写入 stack.brain。"
        )
    return core


def _manifest_domains_lines(manifest: Dict[str, Any]) -> List[str]:
    domains = manifest.get("integration_domains") or []
    lines: List[str] = []
    for d in domains:
        if not isinstance(d, dict):
            continue
        oid = str(d.get("id", "")).strip()
        oc = str(d.get("upstream_ref") or d.get("openclaw", "")).strip()
        lf = str(d.get("lifers", "")).strip()
        if not oc and not lf:
            continue
        label = f"[{oid}] " if oid else ""
        lines.append(f"  · {label}上游参照: {oc} | Lifers 实现: {lf}")
    return lines


def ai_integration_block(ocfg: Dict[str, Any], root: Path) -> str:
    """
    给本地 Agent 的完整说明：对照上游能力域，落实仅在 Lifers（无密钥）。
    """
    oc = resolve_openclaw_effective(ocfg)
    base = bridge_context_line(oc)
    if not base:
        return ""

    manifest = load_openclaw_manifest(root)
    lines: List[str] = [
        base,
        "",
        "【能力边界 · 由对照清单维护】",
        "上游 OpenClaw 常见形态含网关、频道、托管模型、Skills/Canvas/Cron 等 —— 此处仅作文档参照；本仓库不安装、不执行上游二进制。",
        "Lifers 实际承担：LocalBrain 权重、Planner 工具、runtime/本能/机器人钩子、SQLite 记忆与 audit。",
    ]

    md_lines = _manifest_domains_lines(manifest)
    if md_lines:
        lines.append("")
        lines.append("【分项对照 · config/openclaw_manifest.json】")
        lines.extend(md_lines)

    merged = merged_rs_openclaw_section(root)
    if merged:
        lines.append(merged)

    vend = openclaw_vendor_tree_path(root)
    if vend:
        lines.append("")
        lines.append(
            f"【本地对照源码树】{vend}（git 子模块或浅克隆，只读；rs 根 git submodule update --init；或 scripts/vendor_openclaw_reference.ps1 / vendor_openclaw_reference.sh）"
        )

    cr = claw_code_rust_vendor_root(root)
    if cr:
        lines.append("")
        lines.append(
            f"【claw-code/rust vendor 源码树】{cr}（见 third_party/README.txt、config/claw_code_rust_vendor.json；非运行时）"
        )

    extra = str(manifest.get("extra_hints", "") or ocfg.get("extra_hints", "") or "").strip()
    if extra:
        lines.extend(["", "【补充】", extra])

    text = "\n".join(lines)
    max_chars = int(oc.get("ai_hints_max_chars", 2800) or 2800)
    if len(text) > max_chars:
        return text[: max_chars - 20].rstrip() + "\n…(OpenClaw 提示已截断)"
    return text


def integration_context_for_agent(ocfg: Dict[str, Any], root: Path) -> str:
    """入口：完整块或单行。"""
    oc = resolve_openclaw_effective(ocfg)
    if not oc.get("enabled"):
        return ""
    if oc.get("ai_hints_full", True):
        return ai_integration_block(oc, root)
    return bridge_context_line(oc)


def summary_for_verify(ocfg: Dict[str, Any]) -> Dict[str, Any]:
    oc = resolve_openclaw_effective(ocfg)
    return {
        "enabled": bool(oc.get("enabled")),
        "track_upstream": bool(oc.get("track_upstream", True)),
        "use_external_openclaw_runtime": bool(oc.get("use_external_openclaw_runtime")),
        "compat_ref": str(oc.get("compat_ref", "")).strip() or None,
        "workspace_linked": bool(str(oc.get("workspace_path", "")).strip()),
        "ai_hints_full": bool(oc.get("ai_hints_full", True)),
    }


def fetch_latest_release_tag(timeout_sec: float = 15.0) -> Optional[str]:
    """GitHub releases/latest → tag_name；失败返回 None。"""
    try:
        from lifers_brain.speed_env import http_timeout_seconds

        req = urllib.request.Request(
            GITHUB_LATEST_API,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "lifers_brain-openclaw-sync",
            },
        )
        t = http_timeout_seconds(timeout_sec)
        with _outbound_opener().open(req, timeout=t) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        tag = str(data.get("tag_name", "")).strip()
        return tag or None
    except Exception:
        return None


def compat_ref_tag(compat_ref: str) -> Optional[str]:
    """从 'openclaw/openclaw@v2026.4.26' 取出标签部分（含 v 前缀）。"""
    s = str(compat_ref).strip()
    if "@" in s:
        return s.split("@", 1)[-1].strip() or None
    return None


def verify_upstream_drift(ocfg: Dict[str, Any]) -> Optional[str]:
    """
    若远端 tag 与 compat_ref 不一致，返回警告文案；否则 None。
    """
    oc = resolve_openclaw_effective(ocfg)
    if not oc.get("enabled") or not oc.get("track_upstream", True):
        return None
    ref = str(oc.get("compat_ref", "")).strip()
    local_tag = compat_ref_tag(ref) if ref else None
    remote = fetch_latest_release_tag()
    if not remote or not local_tag:
        return None
    rn = remote.lstrip("v")
    ln = local_tag.lstrip("v")
    if rn != ln:
        return (
            f"OpenClaw 远端最新 release 为 {remote}，stack.openclaw.compat_ref 锚点为 {ref}。"
            "可运行: python scripts/sync_openclaw_release.py"
        )
    return None
