"""
LifersAgent: the main orchestration class for the Lifers AI agent.

Originally monolithic; now delegates to:
- ``Planner`` (``planner.py``) — tool chain planning
- ``LocalBrain`` / ``AgentConfig`` (``local_brain.py``) — LM inference
"""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from lifers_brain.instincts import load_instinct_state, sync_prev_user_ts, tick_instincts_end, tick_instincts_start
from lifers_brain.inference_pipeline import log_inference
from lifers_brain.health import emit_health_report
from lifers_brain.llm_ops_context import format_llm_ops_context
from lifers_brain.local_brain import (
    AgentConfig,
    LocalBrain,
    _generate_with_wallclock_timeout,
    _quick_fast_enabled,
    _quick_skip_kb,
)
from lifers_brain.memory import LongTermMemory, MemoryItem, Scratchpad, SessionMemory
from lifers_brain.model_names import canonical_brain_model
from lifers_brain.npc_engine import NpcEngine
from lifers_brain.openclaw_compat import format_rs_integrated_layout_hint, integration_context_for_agent
from lifers_brain.organ_system import format_organ_system_context
from lifers_brain.physiology_sim import update_physiology_and_format
from lifers_brain.planner import Planner
from lifers_brain.runtime_mode import resolve_runtime, runtime_label_from_role, runtime_system_line
from lifers_brain.speed_env import local_lm_max_chars as speed_local_lm_max_chars
from lifers_brain.stack_env import load_stack
from lifers_brain.tools import ToolCall, ToolRegistry, ToolResult, build_default_registry

# 仅首次：避免每轮 stderr 刷屏（Kali 上若误设 LIFERS_REMOTE_CHAT=1 且无 key）
_REMOTE_INFER_SKIP_LOGGED = False

_WEB_HINT_RE = re.compile(
    r"(什么|怎么|为什么|多少|哪里|哪儿|是否|哪个|谁|几种|如何|能否|能不能|是不是|"
    r"是啥|是什么意思|是何|能做什么|做什么|如何做|如何使用|怎么用|如何安装|推荐|比较|区别|原理|"
    r"历史|最新|资料|查查|搜一下|搜索一下|帮我搜|帮我找|不了解|不知道|讲讲|说说|有哪些|用处|功能|介绍|能力|作用|"
    r"官网|教程|例子|案例|对比|排名|价格|定义)"
)

_META_SELF_RE = re.compile(
    r"(你能做什么|你会做什么|你会啥|你能干啥|你会干什么|你是谁|你叫什么|介绍下自己|你是干什么的|"
    r"你做什么|你干啥|你是干啥的|你是做什么的|"
    r"你有什么功能|你有什么用|怎么用你|如何使用你|你的能力|能帮我什么|"
    r"what\s+can\s+you|who\s+are\s+you)",
    re.I,
)


def _quick_chat_wants_web(user_line: str) -> bool:
    """Knowledge/factual bias: auto web search under quick-chat path."""
    u = user_line.strip()
    if not u:
        return False
    if _META_SELF_RE.search(u):
        return False
    if len(u) < 4:
        return False
    return bool(_WEB_HINT_RE.search(u))


def _quick_output_sane(reply: str, user_line: str) -> bool:
    """Reject gibberish from local tiny models."""
    if not reply or len(reply) > 2400:
        return False
    u = user_line.strip()
    cjk_u = len(re.findall(r"[一-鿿]", u))
    cjk_r = len(re.findall(r"[一-鿿]", reply))
    if cjk_u == 0 and len(u) <= 16:
        if len(reply) > 180:
            return False
        if cjk_r > 16:
            return False
        if len(reply) > 72 and (cjk_r + len(re.findall(r"[A-Za-z]", reply))) > len(reply) * 0.72:
            return False
        if re.fullmatch(r"[\d\s]{1,16}", u) and len(u) >= 1 and len(reply.strip()) <= 3 and re.match(
            r"^[\d\s.]*$", reply.strip()
        ):
            return False
    if cjk_u >= max(2, len(u) // 3):
        if cjk_r < max(4, min(20, len(reply) // 5)):
            return False
    junk = sum(reply.count(x) for x in "{}_<>[]/\\:|")
    if junk > max(6, len(reply) // 12):
        return False
    ascii_let = len(re.findall(r"[A-Za-z]", reply))
    if cjk_u >= 2 and ascii_let > max(6, len(reply) // 6):
        return False
    if cjk_u >= 2 and re.search(r"[A-Za-z]{2,}[_/\\][A-Za-z0-9]{3,}", reply):
        return False
    if len(reply) >= 120:
        uniq_cjk = len(set(re.findall(r"[一-鿿]", reply)))
        if cjk_r >= 40 and uniq_cjk < max(12, cjk_r // 6):
            return False
    return True


def _detect_repetition(text: str, ngram: int = 4, threshold: float = 0.4) -> bool:
    """Check if output has excessive repeated n-gram overlaps (typical of degenerate local models)."""
    if len(text) < ngram * 6:
        return False
    seen: set[str] = set()
    total = 0
    repeat = 0
    for i in range(len(text) - ngram + 1):
        seg = text[i : i + ngram]
        total += 1
        if seg in seen:
            repeat += 1
        else:
            seen.add(seg)
    return total > 0 and (repeat / total) > threshold


def _enforce_output_length(text: str, max_chars: int = 16384) -> str:
    """Truncate at the last sentence boundary within max_chars."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Try to break at sentence boundary
    for sep in ("。", "！", "？", "\n", ".", "!", "?"):
        idx = truncated.rfind(sep)
        if idx > max_chars // 2:
            return truncated[: idx + 1]
    return truncated


def _quick_chat_trivial_local_reply(user_line: str) -> Optional[str]:
    """Short ASCII-only probe input: skip markov to avoid long blocking."""
    u = user_line.strip()
    if not u or len(u) > 24:
        return None
    if re.search(r"[一-鿿]", u):
        return None
    if re.fullmatch(r"\d{1,24}", u):
        return f"收到「{u}」。请用完整中文句子描述你的问题或任务，我就能更好地回答。"
    if not re.fullmatch(r"[\x20-\x7E]{1,24}", u):
        return None
    safe = u.replace("\r", " ").replace("\n", " ")[:48]
    return f"收到「{safe}」。请用完整中文句子描述你的问题或任务，我就能更好地回答。"


_QUICK_GREET_RE = re.compile(
    r"^(你好|您好|在吗|在么|哈喽|嗨|早上好|中午好|下午好|晚上好|谢谢|感谢|多谢|再见|拜拜)"
    r"[！!。.…?？~\s]*$"
)


def _quick_chat_cn_greeting_reply(user_line: str) -> Optional[str]:
    u = user_line.strip()
    if not u or len(u) > 24:
        return None
    if not _QUICK_GREET_RE.match(u):
        return None
    return (
        "你好，我是 **Lifers** 本地助手（当前为轻量快路径）。"
        "请用一两句完整中文说明你的问题或任务；若要查资料可发 **search 关键词**。"
    )


_META_CAP_SHORT = re.compile(
    r"^(能做什么|你会什么|会什么|能帮我什么|有啥功能|什么功能|介绍一下自己|自我介绍|干嘛的|干什么的|你做什么)"
    r"[！!。.…?？~\s]*$"
)


def _quick_chat_meta_capability_reply(user_line: str) -> Optional[str]:
    """User asks about agent capabilities/identity: local template, no Markov/web dependency."""
    u = user_line.strip()
    if not u or len(u) > 400:
        return None
    if _META_SELF_RE.search(u) or _META_CAP_SHORT.match(u):
        return (
            "我是 **Lifers** 本地助手（对话任务类型由「对话推理分发器」分到 CHAT_QUICK / 工具链等）。\n\n"
            "**常见用法：**\n"
            "- **日常聊天**：直接中文提问（权重在 `weights/`，体量大会慢，见 README）。\n"
            "- **联网检索**：发 **`search 关键词`** 或「帮我搜索…」类句式（需联网且 **lifers.sandbox** 非纯沙盒）。\n"
            "- **文件上下文**：侧栏 / **`@`** 把路径加入会话后再问。\n"
            "- **方案预览**：行首 **`方案`** 或 **`plan`**。\n\n"
            "扩展默认 **LIFERS_FORCE_LOCAL_ONLY**：不接云端 Chat API；更强模型见 **stack.remote_infer**（终端高级用法）。"
        )
    return None


class LifersAgent:
    def __init__(self, cfg: AgentConfig, registry: Optional[ToolRegistry] = None) -> None:
        self.cfg = cfg
        cfg.model = canonical_brain_model(cfg.model)
        os.environ["SANDBOX"] = "1" if cfg.sandbox else "0"
        os.environ["MODEL"] = cfg.model
        os.environ["LIFERS_ROOT"] = str(cfg.root_dir)

        stack = load_stack(cfg.root_dir)
        self._runtime = resolve_runtime(cfg.root_dir, stack)
        os.environ["LIFERS_RUNTIME"] = self._runtime
        self._runtime_system_line = runtime_system_line(self._runtime)
        brain_s = stack.get("brain") or {}
        hs = stack.get("human_sim") or {}
        if brain_s.get("max_tool_steps"):
            cfg.max_tool_steps = int(brain_s["max_tool_steps"])
        smax = int(brain_s.get("session_max_turns", 8))
        if hs.get("local_lm_max_chars") is not None:
            try:
                cfg.local_lm_max_chars = max(32, int(hs["local_lm_max_chars"]))
            except (TypeError, ValueError):
                pass

        self.tools = registry or build_default_registry()
        self.brain = LocalBrain(cfg)
        self.scratch = Scratchpad()
        self.session = SessionMemory(max_turns=smax)
        mem_rel = str(brain_s.get("memory_db", "memory/longterm.sqlite3"))
        self.longterm = LongTermMemory((cfg.root_dir / mem_rel).resolve())
        self._human_prompt_extra = ""
        if hs.get("enabled", True):
            pn = str(hs.get("persona_name", "Lifers")).strip() or "Lifers"
            ex = str(hs.get("system_prompt_extra", "")).strip()
            lines = [f"【仿真人】名称：{pn}。"]
            if ex:
                lines.append(ex)
            self._human_prompt_extra = "\n".join(lines)
        short = str(brain_s.get("llm_identity_short", "Lifers")).strip() or "Lifers"
        prod = str(brain_s.get("llm_product_name", "")).strip()
        self._llm_identity = f"{short}（{prod}）" if prod else short
        self.planner = Planner()
        self._instinct_state = load_instinct_state(cfg.root_dir)
        self._instinct_turn_notes: List[str] = []
        self._openclaw_line = integration_context_for_agent(stack.get("openclaw") or {}, cfg.root_dir)
        self._llm_ops_line = format_llm_ops_context(stack, cfg.root_dir)
        self._rs_layout_line = format_rs_integrated_layout_hint(cfg.root_dir)
        self._organ_line = format_organ_system_context(stack, cfg.root_dir)
        self._physio_line = ""
        self._quick_route_reason: str = ""
        self._quick_route_notes_zh: str = ""
        self.npc_engine = NpcEngine.from_stack(stack, cfg.root_dir)
        self._recent_responses: List[str] = []  # last N assistant replies for dedup
        # Startup health diagnostics (non-fatal; errors go to stderr)
        if os.environ.get("LIFERS_SKIP_HEALTH_CHECK", "").strip().lower() not in ("1", "true", "yes", "on"):
            emit_health_report(cfg.root_dir)

    def _format_plan_header(
        self,
        user_input: str,
        calls: List[ToolCall],
        recalled: List[Dict[str, Any]],
        *,
        note: str = "",
        preview_only: bool = False,
    ) -> str:
        rt_zh = runtime_label_from_role(self._runtime)
        try:
            anchor = datetime.now().astimezone()
            ts_line = (
                f"【上下文时效】本方案头生成于 {anchor.strftime('%Y-%m-%d %H:%M:%S')} "
                f"{anchor.tzname() or ''}（每轮推理刷新；工具结果见各步 fetched 时刻）"
            )
        except Exception:
            ts_line = f"【上下文时效】本方案头生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}（每轮推理刷新）"
        lines = [
            "──────── 【大脑方案】 ────────",
            ts_line,
            f"三合一宿主: {rt_zh}（LIFERS_RUNTIME={os.environ.get('LIFERS_RUNTIME', '')}）",
        ]
        ix = getattr(self, "_instinct_turn_notes", None) or []
        if ix:
            lines.append("本能（自动，内化语气即可）:")
            for s in ix[:6]:
                lines.append(f"  · {s}")
        lines.append(f"输入: {user_input[:280]}")
        if note:
            lines.append(f"策略说明: {note}")
        if recalled:
            lines.append(
                f"【记忆快照】本地知识库预检索已召回 {len(recalled)} 条（SQLite 历史片段，非互联网实时行情/天气）。"
            )
        if not calls:
            lines.append("步骤: （无匹配的工具链）→ 将给出指令提示或仅用本地小模型补一句。")
        else:
            lines.append("步骤:")
            for i, c in enumerate(calls[: self.cfg.max_tool_steps], start=1):
                lines.append(f"  {i}. [{c.name}] {c.expected_effect}")
            hr = [c.name for c in calls if c.name in ("fs_write_patch", "cmd_run", "motion_execute", "manipulate")]
            if hr:
                lines.append(f"高风险工具: {', '.join(hr)}（SANDBOX={os.environ.get('SANDBOX')}）→ 需要你自己判断是否放行。")
        if preview_only:
            lines.append("状态: 仅预览，未执行。")
        lines.append("────────────────────────────")
        return "\n".join(lines)

    def _npc_context_block(self) -> str:
        lines = self.npc_engine.active_context_lines()
        if not lines:
            return ""
        return "NPC_STATES:\n" + "\n".join(lines) + "\n\n"

    def _npc_react_for_turn(self, user_text: str, tool_result_ok: bool = True) -> None:
        """Detect active NPC and trigger emotional reaction after a turn."""
        active = self.npc_engine.detect_active_npc(user_text)
        if active:
            st = self.npc_engine.states.get(active)
            if st:
                st.react(user_text, tool_result_ok)
                # Store conversation memory (keep last 16 exchanges)
                st.dialogue_history.append(user_text[:240])
                if len(st.dialogue_history) > 16:
                    st.dialogue_history = st.dialogue_history[-16:]
        elif self.npc_engine.states:
            # No specific NPC addressed — gentle decay for all
            for st in self.npc_engine.states.values():
                st.emotion.decay(0.98)

    def _npc_dialogue_hint(self, user_text: str) -> str:
        """If active NPC has a dialogue tree match, return the NPC's scripted reply."""
        active = self.npc_engine.detect_active_npc(user_text)
        if not active:
            return ""
        # Check for greeting first (only on first encounter this session)
        greeting = self.npc_engine.greeting_for(active)
        if greeting:
            return f"\n【{active}·初次问候】{greeting}\n"
        _, reply = self.npc_engine.dialogue_match(active, user_text)
        if reply:
            return f"\n【{active}·对话树命中】{reply}\n"
        return ""

    def _stack_context_body(self) -> str:
        return (
            f"{self._runtime_system_line}\n"
            f"{self._context_block(self._llm_ops_line, self._rs_layout_line)}"
            f"{self._openclaw_line or ''}\n\n"
            f"{self._organ_line or ''}"
            f"{self._physio_line or ''}"
            f"{self._npc_context_block()}"
            f"{self._instinct_block()}"
        )

    @staticmethod
    def _context_block(*parts: str) -> str:
        joined = "\n\n".join(p for p in parts if p)
        return (joined + "\n\n") if joined else ""

    def _instinct_block(self) -> str:
        notes = getattr(self, "_instinct_turn_notes", None) or []
        if not notes:
            return ""
        return "INSTINCT_AUTONOMIC:\n" + "\n".join(notes[:8]) + "\n\n"

    def _dialogue_route_hint_block(self) -> str:
        rr = (getattr(self, "_quick_route_reason", None) or "").strip()
        rz = (getattr(self, "_quick_route_notes_zh", None) or "").strip()
        if not rr and not rz:
            return ""
        lines = ["DIALOGUE_ROUTE:", f"reason: {rr}", f"notes_zh: {rz}"]
        extra = (
            "\n【推理策略】用户询问助手身份或能力：请用简洁中文说明你能提供的帮助类型"
            "（日常对话、search 检索、文件上下文、方案预览等）。\n\n"
            if rr == "assistant_meta_intent"
            else "\n"
        )
        return "\n".join(lines) + extra

    def _quick_session_context_chars(self) -> int:
        raw = os.environ.get("LIFERS_QUICK_SESSION_CONTEXT_CHARS", "").strip()
        if raw:
            try:
                return max(400, min(int(raw), 80_000))
            except ValueError:
                pass
        return 4800 if self.brain.model == "transformer" else 14_000

    def _quick_stack_body_chars(self) -> int:
        raw = os.environ.get("LIFERS_QUICK_STACK_BODY_CHARS", "").strip()
        if raw:
            try:
                return max(800, min(int(raw), 100_000))
            except ValueError:
                pass
        return 4000 if self.brain.model == "transformer" else 12_000

    def _clip_quick_stack_body(self, body: str) -> str:
        cap = self._quick_stack_body_chars()
        if len(body) <= cap:
            return body
        log_inference("stack_body_clip", orig_chars=len(body), cap=cap, model=self.brain.model)
        head = max(400, cap - 160)
        return body[:head].rstrip() + "\n…【栈上下文过长已截断（CHAT_QUICK；可调 LIFERS_QUICK_STACK_BODY_CHARS）】\n"

    def _quick_chat_inference_pack(self, user_line: str, recalled: List[Dict[str, Any]]) -> str:
        try:
            mem_anchor = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            mem_anchor = time.strftime("%Y-%m-%d %H:%M:%S")
        mem_hdr = f"（以下检索于 {mem_anchor}；为本地库历史，非联网实时事实）\n"
        mem_txt = mem_hdr + "\n".join([f"- {m['type']}: {m['content']}" for m in recalled])[:2000]
        persona = (self._human_prompt_extra + "\n\n") if getattr(self, "_human_prompt_extra", "") else ""
        body = self._clip_quick_stack_body(self._stack_context_body())
        route = self._dialogue_route_hint_block()
        npc_hint = self._npc_dialogue_hint(user_line)
        tail_identity = (
            f"You are {self._llm_identity}. "
            "Reply in natural Chinese when the user writes Chinese; be concise and safe. "
            "This turn is CHAT_QUICK (route → local brain): no tools ran yet — use LONGTERM_RECALL and session.\n"
        )
        sess = self.session.context_text()
        smax = self._quick_session_context_chars()
        if len(sess) > smax:
            sess = sess[-smax:]
        return (
            "SYSTEM:\n"
            f"{body}"
            f"{route}"
            f"{npc_hint}"
            f"{tail_identity}"
            f"{persona}"
            f"{sess}\n\n"
            f"LONGTERM_RECALL:\n{mem_txt}\n\n"
            f"TOOL_OBSERVATIONS:\n(none — CHAT_QUICK path)\n\n"
            f"USER:\n{user_line}\n"
            "ASSISTANT:\n"
        )

    def _clip_quick_inference_prompt(self, prompt: str) -> str:
        if self.brain.model == "transformer":
            default_cap = "7200" if _quick_fast_enabled() else "11000"
        else:
            default_cap = "12000" if _quick_fast_enabled() else "28000"
        try:
            raw = os.environ.get("LIFERS_QUICK_PACK_MAX_CHARS", default_cap).strip()
            cap = int(raw) if raw else int(default_cap)
        except ValueError:
            cap = int(default_cap)
        cap = max(6000, min(cap, 250000))
        if len(prompt) <= cap:
            return prompt
        head_n = (cap * 3) // 5
        tail_n = cap - head_n - 96
        if tail_n < 512:
            tail_n = 512
        log_inference("prompt_clip", orig_chars=len(prompt), cap=cap)
        return (
            prompt[:head_n].rstrip()
            + "\n…\n【上文过长已截断；以下为尾部上下文（含 USER）】\n…\n"
            + prompt[-tail_n:]
        )

    def _context_pack(self, user_input: str, recalled: List[Dict[str, Any]], tool_obs: List[ToolResult]) -> str:
        try:
            mem_anchor = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            mem_anchor = time.strftime("%Y-%m-%d %H:%M:%S")
        mem_hdr = f"（以下检索于 {mem_anchor}；为本地库历史，非联网实时事实）\n"
        mem_txt = mem_hdr + "\n".join([f"- {m['type']}: {m['content']}" for m in recalled])[:2000]
        obs_txt = "\n".join([str(o.data) for o in tool_obs if o.ok])[:2000]
        persona = (self._human_prompt_extra + "\n\n") if getattr(self, "_human_prompt_extra", "") else ""
        body = self._stack_context_body()
        npc_hint = self._npc_dialogue_hint(user_input)
        return (
            "SYSTEM:\n"
            f"{body}"
            f"{npc_hint}"
            f"You are {self._llm_identity}. No cloud LLM API. Use tools if helpful. Be human-like, concise, and safe.\n"
            f"{persona}"
            f"{self.session.context_text()}\n\n"
            f"LONGTERM_RECALL:\n{mem_txt}\n\n"
            f"TOOL_OBSERVATIONS:\n{obs_txt}\n\n"
            f"USER:\n{user_input}\n"
            "ASSISTANT:\n"
        )

    def _tool_first_answer(self, user_input: str, tool_results: List[ToolResult]) -> str | None:
        for r in tool_results:
            if not r.ok:
                continue
            rw = r.data.get("real_world")
            if rw == "clock":
                d = r.data
                return f"【实时时间】{d.get('local')} 周{d.get('weekday_zh')}（{d.get('tz')}）\nISO: {d.get('iso')}"
            if rw == "weather":
                base = "【天气】" + str(r.data.get("summary", "")).strip()
                ts_ms = r.data.get("fetched_at_ms")
                if isinstance(ts_ms, (int, float)) and ts_ms > 0:
                    try:
                        snap = datetime.fromtimestamp(float(ts_ms) / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
                        base += f"\n（wttr.in 快照约 {snap}；极端行情请以当地实况为准）"
                    except (OSError, ValueError, OverflowError):
                        pass
                return base
            if rw == "map":
                lines = [
                    f"【地图】{r.data.get('display_name', '')}",
                    f"坐标: {r.data.get('lat')}, {r.data.get('lon')}",
                    f"在浏览器打开: {r.data.get('openstreetmap_url', '')}",
                ]
                ts_ms = r.data.get("fetched_at_ms")
                if isinstance(ts_ms, (int, float)) and ts_ms > 0:
                    try:
                        snap = datetime.fromtimestamp(float(ts_ms) / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
                        lines.append(f"（Nominatim 检索快照约 {snap}）")
                    except (OSError, ValueError, OverflowError):
                        pass
                return "\n".join(lines)

        for r in tool_results:
            if not r.ok:
                continue
            if "results" in r.data and isinstance(r.data.get("results"), list):
                items = r.data["results"][:5]
                lines = ["我查到这些结果（摘要）："]
                for i, it in enumerate(items, start=1):
                    title = str(it.get("title", "")).strip()
                    url = str(it.get("url", "")).strip()
                    snip = str(it.get("snippet", "")).strip()
                    lines.append(f"- {i}. {title} ({url})")
                    if snip:
                        lines.append(f"  摘要：{snip}")
                lines.append("我会自动点开第 1 条并提取要点。")
                ts_ms = r.data.get("fetched_at_ms")
                if isinstance(ts_ms, (int, float)) and ts_ms > 0:
                    try:
                        snap = datetime.fromtimestamp(float(ts_ms) / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
                        lines.append(f"（检索快照约 {snap}；网页时效以站点为准）")
                    except (OSError, ValueError, OverflowError):
                        pass
                return "\n".join(lines)

        for r in tool_results:
            if not r.ok:
                continue
            items = r.data.get("items")
            if isinstance(items, list):
                if not items:
                    return "长期记忆里没有找到匹配内容。你可以换个关键词，或让我先联网搜集并写入记忆。"
                lines = ["我在长期记忆里找到这些内容（前几条；为本地库历史，非联网实时）："]
                for it in items[:6]:
                    lines.append(f"- ({it.get('type')}) {it.get('content')}")
                return "\n".join(lines)

        for r in tool_results:
            if not r.ok:
                continue
            if r.data.get("task_id") and "success_rate" in r.data:
                return f"仿真任务 {r.data.get('task_id')} 已运行：success_rate={r.data.get('success_rate')}, runs={r.data.get('runs')}"

        for r in tool_results:
            if not r.ok:
                continue
            if "deleted" in r.data and ("cutoff_ts_ms" in r.data or r.side_effects):
                return f"已清理长期记忆：deleted={r.data.get('deleted')}（older_than_days 门槛已应用）"

        for r in tool_results:
            if not r.ok:
                continue
            if "exit_code" in r.data and "stdout" in r.data:
                return f"命令执行完成：exit_code={r.data.get('exit_code')}\nstdout:\n{str(r.data.get('stdout',''))[:800]}"

        fetched = None
        evidence = None
        for r in tool_results:
            if r.ok and "url" in r.data and "text" in r.data:
                fetched = r
            if r.ok and "snippets" in r.data:
                evidence = r
        if fetched is not None:
            url = str(fetched.data.get("url", ""))
            if evidence and isinstance(evidence.data.get("snippets"), list):
                snips = evidence.data["snippets"][:5]
                lines = [f"我已抓取页面：{url}", "关键片段："]
                for s in snips:
                    lines.append(f"- {str(s.get('text',''))[:200]}")
                return "\n".join(lines)
            return f"我已抓取页面：{url}。如果你要我提取要点，请告诉我关注点（定义/步骤/风险/价格等）。"

        for r in tool_results:
            if r.ok and r.data.get("type") == "dir":
                items = r.data.get("items", [])[:40]
                names = [it["name"] + ("/" if it.get("is_dir") else "") for it in items]
                return "目录内容（部分）：\n" + "\n".join([f"- {n}" for n in names])
            if r.ok and r.data.get("type") == "file":
                text = str(r.data.get("text", ""))[:800]
                return "文件内容（节选）：\n" + text

        return None

    def _quick_web_enabled(self) -> bool:
        return os.environ.get("LIFERS_QUICK_WEB", "0").strip().lower() not in ("0", "false", "no", "off")

    def _web_search_reply(self, query: str, user_line: str) -> str:
        from .tools import ToolCall

        u0 = user_line.strip()
        q = (query or u0).strip()[:240] or u0[:240]
        if _META_SELF_RE.search(u0):
            q = (u0 + " AI助手 功能 对话 工具").strip()[:240]
        sys.stderr.write(f"LIFERS_PROGRESS quick_chat auto_web_search q={q[:100]!r}\n")
        sys.stderr.flush()
        r = self.tools.dispatch(
            ToolCall(
                name="web_search",
                args={"query": q, "limit": 5},
                expected_effect="auto web when local reply weak or factual question",
                mode="execute",
            ),
            prior=None,
        )
        if not r.ok:
            err = (r.error or "").strip()[:220]
            low = err.lower()
            proxy_hint = ""
            if (
                "10061" in err
                or "refused" in low
                or "拒绝" in err
                or "actively refused" in low
                or "connection refused" in low
            ):
                proxy_hint = (
                    " 常见原因：系统或环境变量里的代理（HTTPS_PROXY/HTTP_PROXY）指向本机未监听的端口。"
                    "可关闭代理软件后重试，或在本机/启动脚本里设 **LIFERS_HTTP_DIRECT=1** 让 Lifers 出站直连（仍受防火墙限制）。"
                )
            return (
                "我按你的问题尝试联网检索，但没有拿到结果。"
                + (f"（{err}）" if err else "")
                + proxy_hint
                + "你可以换个关键词重试；若需走公司代理请正确设置 HTTPS_PROXY/HTTP_PROXY。"
                + "也可在输入行使用：search …（任意检索引擎由本机网络栈解析）。"
            )
        ans = self._tool_first_answer(user_line, [r])
        if ans and ans.strip():
            return "已自动联网查找，摘要如下：\n" + ans.strip()
        return "已联网检索到一些链接，但未能整理成可读摘要。可缩短问题或发：search " + user_line.strip()[:60]

    def _remote_quick_chat_attempt(self, user_line: str) -> Optional[str]:
        global _REMOTE_INFER_SKIP_LOGGED
        force_local = os.environ.get("LIFERS_FORCE_LOCAL_ONLY", "").strip().lower() in ("1", "true", "yes", "on")
        remote_on = os.environ.get("LIFERS_REMOTE_CHAT", "").strip().lower() in ("1", "true", "yes", "on")
        if not remote_on:
            return None
        from lifers_brain.openai_compat_chat import _is_localhost_url, chat_completion_text, resolve_api_key

        url = (os.environ.get("LIFERS_CHAT_URL") or "https://integrate.api.nvidia.com/v1/chat/completions").strip()
        is_local = _is_localhost_url(url)
        if force_local and not is_local:
            return None

        key = resolve_api_key(url)
        if not key:
            if not _REMOTE_INFER_SKIP_LOGGED:
                _REMOTE_INFER_SKIP_LOGGED = True
                sys.stderr.write(
                    "LIFERS_PROGRESS quick_chat remote_infer skipped: no API key; falling back to local brain.\n"
                )
                sys.stderr.flush()
            return None
        model = (os.environ.get("LIFERS_CHAT_MODEL") or "meta/llama-3.1-8b-instruct").strip()
        try:
            mxt = int(os.environ.get("LIFERS_CHAT_MAX_TOKENS", "1024").strip() or "1024")
        except ValueError:
            mxt = 1024
        mxt = max(64, min(mxt, 8192))
        try:
            to = float(os.environ.get("LIFERS_CHAT_TIMEOUT_SEC", "120").strip() or "120")
        except ValueError:
            to = 120.0
        persona = (getattr(self, "_human_prompt_extra", "") or "").strip()
        sess = self.session.context_text()
        if len(sess) > 4000:
            sess = sess[-4000:]
        intro = f"You are {self._llm_identity}. Reply in natural Chinese unless the user clearly uses another language. Be concise and correct."
        parts: list[str] = [intro]
        if persona:
            parts.append("User tone/preferences: " + persona)
        if sess.strip():
            parts.append("Recent context:\n" + sess)
        system_content = "\n\n".join(parts)[:12000]
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_line},
        ]
        sys.stderr.write(f"LIFERS_PROGRESS quick_chat remote_openai model={model!r}\n")
        sys.stderr.flush()
        text, err = chat_completion_text(
            url=url, api_key=key, model=model, messages=messages,
            max_tokens=mxt, temperature=0.35, timeout_sec=to,
        )
        if text:
            return text
        no_local = os.environ.get("LIFERS_LOCAL_FALLBACK", "1").strip().lower() in ("0", "false", "no", "off")
        if no_local:
            return (
                f"远程模型调用失败：{err or 'unknown'}。可检查网络/代理，或设 **LIFERS_HTTP_DIRECT=1** 后重试。"
            )
        return None

    def _quick_chat_reply(self, user_line: str) -> str:
        u = user_line.strip()
        if not u:
            return "发一句你想问的内容就好。"

        trivial_on = os.environ.get("LIFERS_QUICK_TRIVIAL_ASCII", "1").strip().lower() in ("1", "true", "yes", "on")
        if trivial_on:
            trivial = _quick_chat_trivial_local_reply(u)
            if trivial is not None:
                log_inference("shortcut", kind="trivial_ascii")
                return trivial

        rem = self._remote_quick_chat_attempt(u)
        if rem is not None:
            log_inference("generate", channel="remote_openai_compat")
            return rem

        shortcuts_tpl = os.environ.get("LIFERS_QUICK_TEMPLATE_SHORTCUTS", "0").strip().lower() in ("1", "true", "yes", "on")
        if shortcuts_tpl:
            greet = _quick_chat_cn_greeting_reply(u)
            if greet is not None:
                log_inference("shortcut", kind="greeting_template")
                return greet
            meta_cap = _quick_chat_meta_capability_reply(u)
            if meta_cap is not None:
                log_inference("shortcut", kind="meta_capability_template")
                return meta_cap

        rr = (getattr(self, "_quick_route_reason", None) or "").strip()
        if rr == "assistant_meta_intent":
            use_lm = os.environ.get("LIFERS_QUICK_META_USE_LOCAL_BRAIN", "0").strip().lower() in ("1", "true", "yes", "on")
            if not use_lm:
                meta_routed = _quick_chat_meta_capability_reply(u)
                if meta_routed is not None:
                    log_inference("reply", kind="meta_capability_by_route", route_reason=rr)
                    return meta_routed
                log_inference("meta_route_no_pattern_match", route_reason=rr, chars=len(u))

        if self._quick_web_enabled() and _quick_chat_wants_web(u):
            log_inference("tool", kind="auto_web_search")
            return self._web_search_reply(u, u)

        q = u[:64]
        if _quick_skip_kb():
            recalled: List[Dict[str, Any]] = []
            log_inference(
                "context_pack", kb_hits=0, kb_skipped=True,
                route_reason=(getattr(self, "_quick_route_reason", None) or "") or None,
            )
        else:
            recalled = self.longterm.search(q, k=6)
            log_inference(
                "context_pack", kb_hits=len(recalled),
                route_reason=(getattr(self, "_quick_route_reason", None) or "") or None,
            )

        legacy = os.environ.get("LIFERS_QUICK_LEGACY_PROMPT", "0").strip().lower() in ("1", "true", "yes", "on")
        if legacy:
            persona = (getattr(self, "_human_prompt_extra", "") or "").strip()
            inst_list = getattr(self, "_instinct_turn_notes", None) or []
            inst = "\n".join(inst_list[:4]) if inst_list else ""
            sess = self.session.context_text()
            try:
                qctx = int(os.environ.get("LIFERS_QUICK_CHAT_CONTEXT_CHARS", "960").strip() or "960")
            except ValueError:
                qctx = 960
            qctx = max(400, min(qctx, 2400))
            if len(sess) > qctx:
                sess = sess[-qctx:]
            zh_hint = (
                "只输出与 USER 话题相关的中文短答（一两句），禁止乱码、禁止大段符号与无意义拉丁串。"
                "若用户提及「刚才、上文、之前说过」，须结合 Recent context 作答，勿敷衍。"
            )
            if self.brain.model == "markov":
                zh_hint += "（当前为字符级 markov，务必简短、纯中文。）"
            lines = [
                "SYSTEM:",
                f"You are {self._llm_identity}. Reply in concise natural Chinese. No meta headers, no tool lecture.",
                zh_hint,
            ]
            if persona:
                lines.append(persona)
            if inst:
                lines.append("INSTINCT_AUTONOMIC:\n" + inst)
            lines.append(sess)
            lines.append(f"USER:\n{user_line}\nASSISTANT:\n")
            prompt = "\n\n".join(lines)
            log_inference("prompt", mode="legacy_narrow")
        else:
            prompt = self._quick_chat_inference_pack(u, recalled)
            log_inference("prompt", mode="full_stack_pack")

        prompt = self._clip_quick_inference_prompt(prompt)
        log_inference(
            "prompt_ready", model=self.brain.model, prompt_chars=len(prompt),
            route_reason=(getattr(self, "_quick_route_reason", None) or "") or None,
        )

        try:
            raw_out = os.environ.get("LIFERS_QUICK_CHAT_OUT_CHARS", "").strip()
            qout = int(raw_out) if raw_out else 0
        except ValueError:
            qout = 0
        if qout <= 0:
            qout = speed_local_lm_max_chars(int(getattr(self.brain.cfg, "local_lm_max_chars", 200) or 200))
        else:
            qout = max(48, min(qout, 16384))
        if _quick_fast_enabled() and self.brain.model == "markov":
            try:
                cap_m = int(os.environ.get("LIFERS_QUICK_FAST_MARKOV_CHARS", "140").strip() or "140")
            except ValueError:
                cap_m = 140
            cap_m = max(48, min(cap_m, 512))
            qout = min(qout, cap_m)

        t0 = time.perf_counter()
        text = _generate_with_wallclock_timeout(self.brain, prompt, qout)
        elapsed = int((time.perf_counter() - t0) * 1000)

        # Retry once if output is empty or trivially short
        if len(text.strip()) < 2 and elapsed < 45_000:
            log_inference("retry", reason="empty_or_short", first_elapsed_ms=elapsed, first_chars=len(text))
            text = _generate_with_wallclock_timeout(self.brain, prompt, qout)
            elapsed = int((time.perf_counter() - t0) * 1000)

        log_inference(
            "generate_local", elapsed_ms=elapsed,
            model=self.brain.model, prompt_chars=len(prompt),
        )

        # Post-generation guardrails: length enforcement, repetition detection
        text = _enforce_output_length(text, qout)
        if _detect_repetition(text):
            log_inference("output", repetition_detected=True, chars=len(text))
            # Degenerate output: re-route to web if enabled, else append warning
            if self._quick_web_enabled():
                return self._web_search_reply(u, u)
            text += "\n\n（以上本地模型可能生成了重复内容，建议用完整中文句重试或发 search…）"

        strict_sane = os.environ.get("LIFERS_QUICK_STRICT_SANE", "0").strip().lower() in ("1", "true", "yes", "on")

        if (not text or text.startswith("(missing weights)")) and self._quick_web_enabled():
            log_inference("fallback", kind="web_missing_weights_or_empty")
            return self._web_search_reply(u, u)
        if not text or text.startswith("(missing weights)"):
            return "本地模型不可用（缺权重）。可配置 weights 或联网后重试。"
        if not _quick_output_sane(text, u):
            if self._quick_web_enabled():
                log_inference("fallback", kind="web_insane_output")
                return self._web_search_reply(u, u)
            if strict_sane:
                return (
                    "本地 **Lifers 权重（lifers_transformer.json / Markov）** 由你的训练流水线写入，"
                    "体量与效果取决于训练进度；短输入或分布外话题仍可能不理想。\n"
                    "建议：用**完整中文句**提问；事实检索发 **search …**；需要云端大模型再接 **stack.json → remote_infer**；"
                    "若已关沙盒仍异常，检查 **LIFERS_QUICK_WEB=1** 与网络/代理。"
                    "训练过程中同一权重文件更新后，下一句对话会自动读最新（mtime）。"
                )
            log_inference("generate", sane=False, accepted_non_strict=True)
            return text + "\n\n（以上为本地权重生成；若不理想，请用完整中文句描述，或发 **search …**。）"
        log_inference("generate", sane=True, out_chars=len(text))
        return text

    def _quick_reply_time_footer_enabled(self) -> bool:
        raw = os.environ.get("LIFERS_QUICK_TIME_FOOTER", "").strip().lower()
        if raw in ("0", "false", "no", "off"):
            return False
        if raw in ("1", "true", "yes", "on"):
            return True
        return os.environ.get("LIFERS_AGENTS_UI_BRIDGE", "").strip().lower() in ("1", "true", "yes", "on")

    def _quick_reply_time_footer_line(self) -> str:
        if not self._quick_reply_time_footer_enabled():
            return ""
        try:
            now = datetime.now().astimezone()
            tz = str(now.tzname() or "").strip()
            tz_part = f" {tz}" if tz else ""
            return f"\n\n— 【本轮·生成锚】{now.strftime('%Y-%m-%d %H:%M:%S')}{tz_part}（CHAT_QUICK 本地生成时刻）"
        except Exception:
            return f"\n\n— 【本轮·生成锚】{time.strftime('%Y-%m-%d %H:%M:%S')}（CHAT_QUICK）"

    def _append_quick_reply_time_footer(self, reply: str) -> str:
        if not reply or not str(reply).strip():
            return reply
        line = self._quick_reply_time_footer_line()
        if not line:
            return reply
        if "【本轮·生成锚】" in reply:
            return reply
        return str(reply).rstrip() + line

    def _is_duplicate_response(self, reply: str) -> bool:
        """Check if reply is near-identical to any of the last N responses."""
        if not self._recent_responses:
            return False
        sig = reply[:100].strip()
        if not sig:
            return False
        for prev in self._recent_responses:
            if prev[:100].strip() == sig:
                return True
        return False

    def _record_response(self, reply: str) -> None:
        """Track this reply for dedup (keep window of 6)."""
        if reply:
            self._recent_responses.append(reply)
            if len(self._recent_responses) > 6:
                self._recent_responses.pop(0)

    def _dedup_suffix(self, reply: str) -> str:
        """Append a note if this response repeats a recent one (non-empty to help LM)."""
        if self._is_duplicate_response(reply):
            return "\n\n（注：以上回复与之前某轮相同；可能是本地模型对类似问题的模式重复。）"
        return ""

    def _inject_realtime_spacetime_context(self, stack: Dict[str, Any]) -> None:
        from datetime import datetime
        from lifers_brain.realtime_anchor import geo_context_line

        try:
            now = datetime.now().astimezone()
            wd = "一二三四五六日"[now.weekday()]
            line = (
                f"【实时·时钟】{now.strftime('%Y-%m-%d %H:%M:%S')} "
                f"{now.tzname() or ''} 周{wd}（涉及当前时间/日期须引用本条）"
            )
        except Exception:
            line = f"【实时·时钟】{time.strftime('%Y-%m-%d %H:%M:%S')}"
        ins = getattr(self, "_instinct_turn_notes", None) or []
        if not isinstance(ins, list):
            ins = []
        insert_at = 0
        if not any("【实时·时钟】" in str(x) for x in ins):
            ins.insert(insert_at, line)
            insert_at += 1
        geo = geo_context_line(stack)
        if geo and not any("【实时·定位" in str(x) for x in ins):
            ins.insert(insert_at, geo)
        self._instinct_turn_notes = ins

    def _non_tool_fallback(self, user_input: str) -> str:
        return (
            "未匹配工具链。可用：方案/plan、smart …（先记忆再网）、流程/workflow、search、URL、kb_search、cmd …；"
            "工作区整文件写入：首行 rel_write|workspace_write|self_write <相对路径> 换行后正文。"
        )

    def _brain_fallback_with_context_pack(
        self, user_input: str, recalled: List[Dict[str, Any]],
        tool_obs: List[ToolResult], stack: Dict[str, Any],
    ) -> Optional[str]:
        lo = stack.get("llm_ops") or {}
        if not isinstance(lo, dict) or not lo.get("context_pack_for_brain_fallback"):
            return None
        pack = self._context_pack(user_input, recalled, tool_obs)
        try:
            cap = int(lo.get("context_pack_max_prompt_chars", 8000) or 8000)
        except (TypeError, ValueError):
            cap = 8000
        cap = max(2000, min(cap, 64000))
        if len(pack) > cap:
            pack = pack[: cap - 40].rstrip() + "\n…(context_pack 已截断)\nASSISTANT:\n"
        text = self.brain.generate(pack).strip()
        if not text:
            return None
        return text

    def step(self, user_input: str) -> str:
        if len(user_input) > 80_000:
            return f"输入过长（{len(user_input)} 字符，上限 80000）。请精简后重试。"
        if len(user_input) > 40_000:
            log_inference("long_input", chars=len(user_input), endpoint="step")
        stack = load_stack(self.cfg.root_dir)
        now_ms = int(time.time() * 1000)
        idle_sec = (
            (now_ms - self._instinct_state.prev_user_ts_ms) / 1000.0
            if self._instinct_state.prev_user_ts_ms > 0
            else 0.0
        )
        self._openclaw_line = integration_context_for_agent(stack.get("openclaw") or {}, self.cfg.root_dir)
        self._llm_ops_line = format_llm_ops_context(stack, self.cfg.root_dir)
        self._rs_layout_line = format_rs_integrated_layout_hint(self.cfg.root_dir)
        self._organ_line = format_organ_system_context(stack, self.cfg.root_dir)
        self._physio_line = update_physiology_and_format(
            self.cfg.root_dir, stack, idle_sec,
            bool(user_input and user_input.strip()),
        )

        self._instinct_turn_notes = tick_instincts_start(self, idle_sec, stack)
        reply = ""
        _ok = False
        try:
            reply = self._step_core(user_input, stack)
            _ok = True
            return reply
        finally:
            if _ok:
                self._record_response(reply)
            tick_instincts_end(self, stack, user_input)
            self._npc_react_for_turn(user_input)
            self.npc_engine.save_all(self.cfg.root_dir)
            sync_prev_user_ts(self, int(time.time() * 1000))

    def quick_chat(
        self, user_line: str, *,
        dialogue_route_reason: str = "",
        dialogue_route_notes_zh: str = "",
    ) -> str:
        if len(user_line) > 80_000:
            return "输入过长，请精简后重试。"
        if len(user_line) > 40_000:
            log_inference("long_input", chars=len(user_line), endpoint="quick_chat")
        stack = load_stack(self.cfg.root_dir)
        now_ms = int(time.time() * 1000)
        idle_sec = (
            (now_ms - self._instinct_state.prev_user_ts_ms) / 1000.0
            if self._instinct_state.prev_user_ts_ms > 0
            else 0.0
        )
        self._openclaw_line = integration_context_for_agent(stack.get("openclaw") or {}, self.cfg.root_dir)
        self._llm_ops_line = format_llm_ops_context(stack, self.cfg.root_dir)
        self._rs_layout_line = format_rs_integrated_layout_hint(self.cfg.root_dir)
        self._organ_line = format_organ_system_context(stack, self.cfg.root_dir)
        self._physio_line = update_physiology_and_format(
            self.cfg.root_dir, stack, idle_sec,
            bool(user_line and user_line.strip()),
        )

        self._instinct_turn_notes = tick_instincts_start(self, idle_sec, stack)
        try:
            self._quick_route_reason = dialogue_route_reason or ""
            self._quick_route_notes_zh = dialogue_route_notes_zh or ""
            self.session.add_turn("user", user_line)
            stripped = user_line.strip()
            self._inject_realtime_spacetime_context(stack)
            reply = self._append_quick_reply_time_footer(self._quick_chat_reply(stripped))
            dedup = self._dedup_suffix(reply)
            if dedup:
                reply += dedup
            self._record_response(reply)
            self.session.add_turn("assistant", reply)
            return reply
        finally:
            self._quick_route_reason = ""
            self._quick_route_notes_zh = ""
            tick_instincts_end(self, stack, user_line)
            self._npc_react_for_turn(user_line)
            self.npc_engine.save_all(self.cfg.root_dir)
            sync_prev_user_ts(self, int(time.time() * 1000))

    def _step_core(self, user_input: str, _stack: Dict[str, Any]) -> str:
        self.session.add_turn("user", user_input)
        stripped = user_input.strip()
        low = stripped.lower()
        self._inject_realtime_spacetime_context(_stack)

        inner_preview: Optional[str] = None
        if stripped.startswith("方案"):
            inner_preview = stripped[2:].strip()
        elif low.startswith("plan "):
            inner_preview = stripped[5:].strip()
        if inner_preview is not None:
            qpv = (inner_preview or stripped)[:64]
            recalled_pv = self.longterm.search(qpv, k=6)
            calls = self.planner.plan(inner_preview)
            hdr = self._format_plan_header(inner_preview or stripped, calls, recalled_pv, preview_only=True)
            tip = (
                "\n提示：以上为预览。要执行请去掉「方案/plan」，直接发口令，例如：\n"
                f"  {inner_preview if inner_preview else 'search … / smart …'}"
            )
            out = hdr + tip
            self.session.add_turn("assistant", out)
            return out

        tool_results: List[ToolResult] = []
        plan_note = ""

        rw_calls = self.planner.plan_real_world_instinct(stripped)

        smart_q: Optional[str] = None
        if low.startswith("smart "):
            smart_q = stripped[6:].strip()
        elif stripped.startswith("智搜"):
            smart_q = stripped[2:].strip()

        plan: List[ToolCall] = []
        if not smart_q:
            plan = self.planner.plan(user_input)
            if not rw_calls and not plan:
                reply = self._quick_chat_reply(stripped)
                self.session.add_turn("assistant", reply)
                return reply

        q = stripped[:64]
        recalled = self.longterm.search(q, k=6)

        rw_cap = min(len(rw_calls), 3, self.cfg.max_tool_steps)

        def run_tx(call: ToolCall) -> List[ToolResult]:
            out: List[ToolResult] = []
            dr = self.tools.dispatch(ToolCall(name=call.name, args=call.args, expected_effect=call.expected_effect, mode="dry_run"))
            out.append(dr)
            ex = self.tools.dispatch(ToolCall(name=call.name, args=call.args, expected_effect=call.expected_effect, mode="execute"), prior=dr)
            out.append(ex)
            if not ex.ok:
                rb = self.tools.dispatch(ToolCall(name=call.name, args=call.args, expected_effect=call.expected_effect, mode="rollback"), prior=ex)
                out.append(rb)
                return out
            vr = self.tools.dispatch(ToolCall(name=call.name, args=call.args, expected_effect=call.expected_effect, mode="verify"), prior=ex)
            out.append(vr)
            if not vr.ok:
                rb = self.tools.dispatch(ToolCall(name=call.name, args=call.args, expected_effect=call.expected_effect, mode="rollback"), prior=ex)
                out.append(rb)
            return out

        for idx, call in enumerate(rw_calls[:rw_cap]):
            sys.stderr.write(f"LIFERS_PROGRESS [本能·实时 {idx+1}/{rw_cap}] {call.name} {call.args.get('action', '')}\n")
            sys.stderr.flush()
            if call.name in ("fs_write_patch", "lifers_workspace_write", "cmd_run", "motion_execute", "manipulate"):
                tx = run_tx(call)
                tool_results.extend(tx)
                continue
            r = self.tools.dispatch(call, prior=(tool_results[-1] if tool_results else None))
            tool_results.append(r)
            self.scratch.add(
                MemoryItem(
                    type="tool_result",
                    content={"tool": call.name, "ok": r.ok, "data": r.data},
                    importance=0.3,
                    source="tool",
                )
            )

        slots_left = max(0, self.cfg.max_tool_steps - rw_cap)

        if smart_q:
            plan_note = "先查长期记忆；若无命中再联网搜索。"
            if rw_calls:
                plan_note = "本能（实时环境）+ " + plan_note
            kb_call = ToolCall(
                name="kb_search",
                args={"query": smart_q, "k": 6},
                expected_effect="search long-term memory",
                mode="execute",
            )
            kb_r = self.tools.dispatch(kb_call)
            tool_results.append(kb_r)
            items = kb_r.data.get("items") if kb_r.ok else None
            if isinstance(items, list) and len(items) > 0:
                hdr = self._format_plan_header(stripped, rw_calls + [kb_call], recalled, note=plan_note)
                tool_ans = self._tool_first_answer(stripped, tool_results)
                answer = hdr + "\n\n【执行结果】\n" + (tool_ans or "(无摘要)")
                self.session.add_turn("assistant", answer)
                return answer
            web_call = ToolCall(
                name="web_search",
                args={"query": smart_q, "limit": 5},
                expected_effect="search web (memory miss)",
                mode="execute",
            )
            plan = [web_call]
            header = self._format_plan_header(stripped, rw_calls + [kb_call, web_call], recalled, note=plan_note)
            slots_left = max(0, slots_left - 1)
        else:
            header = self._format_plan_header(stripped, rw_calls + plan, recalled, note=plan_note)

        for i, call in enumerate(plan[:slots_left]):
            sys.stderr.write(f"LIFERS_PROGRESS [步骤 {i+1}] {call.name}\n")
            sys.stderr.flush()
            if call.name == "extract_evidence" and tool_results:
                prev_text = tool_results[-1].data.get("text", "")
                call = ToolCall(name="extract_evidence", args={"text": prev_text, "max_snippets": 5}, expected_effect=call.expected_effect, mode=call.mode)

            if call.name in ("fs_write_patch", "lifers_workspace_write", "cmd_run", "motion_execute", "manipulate"):
                tx = run_tx(call)
                tool_results.extend(tx)
                continue

            r = self.tools.dispatch(call, prior=(tool_results[-1] if tool_results else None))
            tool_results.append(r)
            self.scratch.add(
                MemoryItem(
                    type="tool_result",
                    content={"tool": call.name, "ok": r.ok, "data": r.data},
                    importance=0.3,
                    source="tool",
                )
            )

        for r in list(tool_results):
            if not r.ok:
                continue
            results = r.data.get("results")
            if isinstance(results, list) and results:
                top_url = str(results[0].get("url", "")).strip()
                if top_url:
                    fetched = self.tools.dispatch(
                        ToolCall(name="web_fetch", args={"url": top_url}, expected_effect="fetch top search result", mode="execute"),
                        prior=r,
                    )
                    tool_results.append(fetched)
                    if fetched.ok and "text" in fetched.data:
                        ev = self.tools.dispatch(
                            ToolCall(name="extract_evidence", args={"text": fetched.data.get("text", ""), "max_snippets": 5}, expected_effect="extract evidence", mode="execute"),
                            prior=fetched,
                        )
                        tool_results.append(ev)
                break

        kb_items: List[Dict[str, Any]] = []
        last_url: str | None = None
        for r in tool_results:
            if not r.ok:
                continue
            if r.data.get("type") == "file":
                kb_items.append({
                    "type": "episode",
                    "content": {"event": "fs_read", "path": r.data.get("path"), "note": "file read"},
                    "importance": 0.2,
                    "source": "tool:fs_read",
                })
            if "results" in r.data and isinstance(r.data.get("results"), list):
                kb_items.append({
                    "type": "tool_result",
                    "content": {"event": "web_search", "query": user_input, "results": r.data.get("results")[:5]},
                    "importance": 0.35,
                    "source": "tool:web_search",
                })
            if "url" in r.data and "text" in r.data:
                last_url = str(r.data.get("url", "")).strip() or last_url
                kb_items.append({
                    "type": "tool_result",
                    "key": str(r.data.get("url", "")).strip() or None,
                    "content": {"event": "web_fetch", "url": r.data.get("url"), "truncated": r.data.get("truncated", False)},
                    "importance": 0.55,
                    "source": "tool:web_fetch",
                })
            if "snippets" in r.data and isinstance(r.data.get("snippets"), list):
                kb_items.append({
                    "type": "fact",
                    "key": (f"{last_url}#evidence" if last_url else None),
                    "content": {"event": "evidence", "snippets": r.data.get("snippets")[:5]},
                    "importance": 0.75,
                    "source": "tool:extract_evidence",
                })

        if kb_items:
            kb_res = self.tools.dispatch(ToolCall(name="kb_upsert", args={"items": kb_items}, expected_effect="store KB items", mode="execute"))
            tool_results.append(kb_res)

        if last_url:
            comp = self.tools.dispatch(
                ToolCall(name="kb_compact", args={"url": last_url, "k": 6}, expected_effect="compact KB", mode="execute")
            )
            tool_results.append(comp)

        tool_ans = self._tool_first_answer(user_input, tool_results)
        if tool_ans:
            answer = header + "\n\n【执行结果】\n" + tool_ans
            self.session.add_turn("assistant", answer)
            return answer

        if rw_calls and not plan and not smart_q:
            err_line = ""
            for r in tool_results:
                if not r.ok and (r.error or "").strip():
                    err_line = str(r.error).strip()
                    break
            hint = (
                "【本能·实时】时钟/天气/地图请求未返回可用摘要（多为外网、DNS 或 wttr.in 超时）。\n"
                "请检查网络与代理；或在扩展设置将 **lifers.model** 改为 **markov** 做日常秒回。"
            )
            if err_line:
                hint += f"\n细节：{err_line[:500]}"
            answer = header + "\n\n【执行结果】\n" + hint
            self.session.add_turn("assistant", answer)
            return answer

        fallback_body = self._non_tool_fallback(user_input)
        brain_try = self._brain_fallback_with_context_pack(user_input, recalled, tool_results, _stack)
        if brain_try is not None:
            fallback_body = brain_try

        answer = header + "\n\n【执行结果】\n" + fallback_body
        self.session.add_turn("assistant", answer)

        if "记住" in user_input or "以后" in user_input:
            self.longterm.add(MemoryItem(type="preference", content=user_input, importance=0.8, source="user", ts_ms=int(time.time()*1000)))

        return answer
