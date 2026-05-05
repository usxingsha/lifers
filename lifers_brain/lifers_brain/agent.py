from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .instincts import load_instinct_state
from .llm_ops_context import format_llm_ops_context
from .organ_system import format_organ_system_context
from .physiology_sim import update_physiology_and_format
from .openclaw_compat import format_rs_integrated_layout_hint, integration_context_for_agent
from .daily_intents import cn_web_query_line, parse_workspace_write_message
from .memory import LongTermMemory, MemoryItem, Scratchpad, SessionMemory
from .runtime_mode import resolve_runtime, runtime_label_from_role, runtime_system_line
from .stack_env import load_stack
from .tools import ToolCall, ToolRegistry, ToolResult, build_default_registry
from .markov_lm import MarkovWeights, generate as markov_generate
from .model_names import canonical_brain_model, default_weight_paths
from .speed_env import local_lm_max_chars as speed_local_lm_max_chars
from .transformer_lm import TinyTransformerWeights, generate_text as tt_generate


_WEB_HINT_RE = re.compile(
    r"(什么|怎么|为什么|多少|哪里|哪儿|是否|哪个|谁|几种|如何|能否|能不能|是不是|"
    r"是啥|是什么意思|是何|能做什么|做什么|如何做|如何使用|怎么用|如何安装|推荐|比较|区别|原理|"
    r"历史|最新|资料|查查|搜一下|搜索一下|帮我搜|帮我找|不了解|不知道|讲讲|说说|有哪些|用处|功能|介绍|能力|作用|"
    r"官网|教程|例子|案例|对比|排名|价格|定义)"
)

# 「你能做什么」里「什么」与「做」不相邻，仅靠 _WEB_HINT_RE 可能漏网；另覆盖自我/能力元问题。
_META_SELF_RE = re.compile(
    r"(你能做什么|你会做什么|你会啥|你能干啥|你会干什么|你是谁|你叫什么|介绍下自己|你是干什么的|"
    r"你有什么功能|你有什么用|怎么用你|如何使用你|你的能力|能帮我什么|"
    r"what\s+can\s+you|who\s+are\s+you)",
    re.I,
)


def _quick_chat_wants_web(user_line: str) -> bool:
    """偏知识/检索或「问助手自身」：闲聊快路径下优先自动联网。"""
    u = user_line.strip()
    if not u:
        return False
    if _META_SELF_RE.search(u):
        return True
    if len(u) < 4:
        return False
    return bool(_WEB_HINT_RE.search(u))


def _quick_output_sane(reply: str, user_line: str) -> bool:
    """本地小模型乱码或与中文输入严重不匹配时判为不合格。"""
    if not reply or len(reply) > 2400:
        return False
    u = user_line.strip()
    cjk_u = len(re.findall(r"[\u4e00-\u9fff]", u))
    cjk_r = len(re.findall(r"[\u4e00-\u9fff]", reply))
    # 纯 ASCII/数字等极短输入：微型字符 LM 极易吐长串中英混杂噪声，不能当「正常对话」。
    if cjk_u == 0 and len(u) <= 16:
        if len(reply) > 180:
            return False
        if cjk_r > 16:
            return False
        if len(reply) > 72 and (cjk_r + len(re.findall(r"[A-Za-z]", reply))) > len(reply) * 0.72:
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
    # 长回复里「单字碎片」过多（典型随机采样噪声）
    if len(reply) >= 120:
        uniq_cjk = len(set(re.findall(r"[\u4e00-\u9fff]", reply)))
        if cjk_r >= 40 and uniq_cjk < max(12, cjk_r // 6):
            return False
    return True


@dataclass
class AgentConfig:
    root_dir: Path
    model: str = "markov"  # markov|transformer
    sandbox: bool = True
    max_tool_steps: int = 6
    local_lm_max_chars: int = 200


class LocalBrain:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.model = canonical_brain_model(cfg.model)

    def _weights_path(self) -> Path:
        stack = load_stack(self.cfg.root_dir)
        brain_s = stack.get("brain") or {}
        wmap = brain_s.get("weights") or {}
        rel = wmap.get(self.model)
        if isinstance(rel, str) and rel.strip():
            p = (self.cfg.root_dir / rel.strip()).resolve()
            if p.is_file():
                return p
        root = self.cfg.root_dir
        for rel in default_weight_paths(self.model):
            cand = (root / rel).resolve()
            if cand.is_file():
                return cand
        return (root / default_weight_paths(self.model)[0]).resolve()

    def generate(self, prompt: str) -> str:
        wp = self._weights_path()
        mc = speed_local_lm_max_chars(int(getattr(self.cfg, "local_lm_max_chars", 200) or 200))
        if not wp.exists():
            return "(missing weights) run scripts/run_pipeline.py first"
        if self.model == "transformer":
            w = TinyTransformerWeights.load(wp)
            # 温度略降：纯 Python 小模型采样过散时更像乱码墙。
            return tt_generate(w, prompt=prompt, max_chars=mc, seed=3, temperature=0.72, top_k=40).strip()
        w = MarkovWeights.load(wp)
        return markov_generate(w, prompt=prompt, max_chars=max(mc, 120), seed=3, temperature=0.9, top_k=80).strip()


class Planner:
    """
    Minimal planner (dependency-free):
    - If input looks like "search ..." or contains a URL -> plan web tools
    - If input mentions file path -> plan fs_read
    - Otherwise respond directly
    """

    def plan(self, user_input: str) -> List[ToolCall]:
        text = user_input.strip()
        calls: List[ToolCall] = []
        low = text.lower()

        # Fixed two-step: local KB then web (always runs both when you use this prefix).
        if text.startswith("流程") and len(text) > 2:
            q = text[2:].strip()
            if q:
                calls.append(
                    ToolCall(name="kb_search", args={"query": q, "k": 6}, expected_effect="search long-term memory first", mode="execute")
                )
                calls.append(
                    ToolCall(name="web_search", args={"query": q, "limit": 5}, expected_effect="then search the web", mode="execute")
                )
                return calls
        if low.startswith("workflow "):
            q = text[len("workflow ") :].strip()
            if q:
                calls.append(
                    ToolCall(name="kb_search", args={"query": q, "k": 6}, expected_effect="search long-term memory first", mode="execute")
                )
                calls.append(
                    ToolCall(name="web_search", args={"query": q, "limit": 5}, expected_effect="then search the web", mode="execute")
                )
                return calls

        q_cn = cn_web_query_line(text)
        if q_cn:
            calls.append(
                ToolCall(
                    name="web_search",
                    args={"query": q_cn, "limit": 5},
                    expected_effect="search web (Chinese surface intent)",
                    mode="execute",
                )
            )
            return calls

        ws = parse_workspace_write_message(text)
        if ws:
            rel, body = ws
            calls.append(
                ToolCall(
                    name="lifers_workspace_write",
                    args={"rel_path": rel, "new_text": body},
                    expected_effect="write file under LIFERS_ROOT (self-code allowed)",
                    mode="execute",
                )
            )
            return calls

        if low.startswith("kb_search "):
            q = text[len("kb_search ") :].strip()
            calls.append(
                ToolCall(name="kb_search", args={"query": q, "k": 6}, expected_effect="search KB", mode="execute")
            )
            return calls

        if low.startswith("kb_prune"):
            # Example: kb_prune 0.15 30
            parts = text.split()
            min_imp = float(parts[1]) if len(parts) >= 2 else 0.15
            days = int(parts[2]) if len(parts) >= 3 else 30
            calls.append(
                ToolCall(
                    name="kb_prune",
                    args={"min_importance": min_imp, "older_than_days": days, "limit": 500},
                    expected_effect="prune KB",
                    mode="execute",
                )
            )
            return calls

        if low.startswith("kb_compact "):
            url = text[len("kb_compact ") :].strip()
            calls.append(
                ToolCall(
                    name="kb_compact",
                    args={"url": url, "k": 6},
                    expected_effect="compact KB",
                    mode="execute",
                )
            )
            return calls

        if low.startswith("sim_run "):
            tid = text[len("sim_run ") :].strip()
            root = Path(os.environ.get("LIFERS_ROOT", ".")).resolve()
            dr = int((load_stack(root).get("robot") or {}).get("default_sim_runs", 10))
            calls.append(
                ToolCall(name="sim_run", args={"task_id": tid, "runs": dr}, expected_effect="run sim task", mode="execute")
            )
            return calls

        if low.startswith("cmd "):
            cmd = text[len("cmd ") :].strip()
            calls.append(ToolCall(name="cmd_run", args={"cmd": cmd}, expected_effect="run command", mode="execute"))
            return calls

        if "http://" in text or "https://" in text:
            # Fetch first URL found.
            url = None
            for token in text.split():
                if token.startswith("http://") or token.startswith("https://"):
                    url = token
                    break
            if url:
                calls.append(ToolCall(name="web_fetch", args={"url": url}, expected_effect="fetch web page", mode="execute"))
                calls.append(ToolCall(name="extract_evidence", args={"text": ""}, expected_effect="extract evidence", mode="execute"))
        if text.lower().startswith("search "):
            q = text[7:].strip()
            calls.append(ToolCall(name="web_search", args={"query": q, "limit": 5}, expected_effect="search web", mode="execute"))
        if ":" in text and ("\\" in text or "/" in text):
            # naive windows path detection
            for token in text.split():
                if (":\\" in token) or (token.startswith("/") and "/" in token):
                    calls.append(ToolCall(name="fs_read", args={"path": token.strip("\"'")}, expected_effect="read file/dir", mode="execute"))
                    break
        return calls

    def plan_real_world_instinct(self, user_input: str) -> List[ToolCall]:
        """
        本能层：时间 / 天气 / 地图 — 不依赖用户记口令，自动走 real_world 工具拉实时数据。
        """
        text = user_input.strip()
        if not text:
            return []
        low = text.lower()
        out: List[ToolCall] = []

        # 不要用泛化的「在哪」触发地图（例如「你在哪」会误打 Nominatim，闲聊极慢）。
        if any(k in text for k in ("地图", "导航", "路线", "经纬", "坐标", "定位", "geocode")):
            q = text
            for p in ("搜地图", "查地图", "地图", "导航到", "打开地图", "定位"):
                if text.startswith(p):
                    q = text[len(p) :].strip()
                    break
            if len(q) >= 2:
                out.append(
                    ToolCall(
                        name="real_world",
                        args={"action": "map", "query": q[:400]},
                        expected_effect="地图 / 地点（OpenStreetMap）",
                        mode="execute",
                    )
                )

        if any(k in text for k in ("天气", "气温", "下雨", "下雪", "温度")) or "weather" in low:
            loc = ""
            toks = text.replace("，", " ").replace("。", " ").split()
            for i, w in enumerate(toks):
                if "天气" in w or w in ("气温", "温度"):
                    if i > 0 and 2 <= len(toks[i - 1]) <= 14:
                        loc = toks[i - 1]
                    break
            out.append(
                ToolCall(
                    name="real_world",
                    args={"action": "weather", "location": loc, "query": text[:280]},
                    expected_effect="天气（wttr.in）",
                    mode="execute",
                )
            )

        clk = any(
            k in text
            for k in (
                "几点",
                "几号",
                "星期几",
                "周几",
                "日期",
                "时区",
                "当前时间",
                "现在几点",
                "现在时间",
                "什么时间",
            )
        ) or low in ("what time", "current time", "today date", "date today")
        if clk:
            out.insert(
                0,
                ToolCall(name="real_world", args={"action": "clock"}, expected_effect="本机实时时钟", mode="execute"),
            )

        seen: set = set()
        deduped: List[ToolCall] = []
        for c in out:
            key = (c.name, str(c.args.get("action")))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)
        return deduped[:4]


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
        self._step_stack: Dict[str, Any] = {}
        self._openclaw_line = integration_context_for_agent(stack.get("openclaw") or {}, cfg.root_dir)
        self._llm_ops_line = format_llm_ops_context(stack, cfg.root_dir)
        self._rs_layout_line = format_rs_integrated_layout_hint(cfg.root_dir)
        self._organ_line = format_organ_system_context(stack, cfg.root_dir)
        self._physio_line = ""

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
        lines = [
            "──────── 【大脑方案】 ────────",
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
            lines.append(f"长期记忆预检索: 已召回 {len(recalled)} 条用作上下文。")
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

    def _context_pack(self, user_input: str, recalled: List[Dict[str, Any]], tool_obs: List[ToolResult]) -> str:
        mem_txt = "\n".join([f"- {m['type']}: {m['content']}" for m in recalled])[:2000]
        obs_txt = "\n".join([str(o.data) for o in tool_obs if o.ok])[:2000]
        persona = (self._human_prompt_extra + "\n\n") if getattr(self, "_human_prompt_extra", "") else ""
        host_hint = self._runtime_system_line + "\n"
        llm_ops = ""
        lol = getattr(self, "_llm_ops_line", "") or ""
        rs_hint = getattr(self, "_rs_layout_line", "") or ""
        parts = [p for p in (lol, rs_hint) if p]
        if parts:
            llm_ops = "\n\n".join(parts) + "\n\n"
        oc = ""
        ol = getattr(self, "_openclaw_line", "") or ""
        if ol:
            oc = ol + "\n\n"
        _org = getattr(self, "_organ_line", "") or ""
        organ_part = _org if _org.strip() else ""
        _phy = getattr(self, "_physio_line", "") or ""
        physio_part = _phy if _phy.strip() else ""
        inst = ""
        if getattr(self, "_instinct_turn_notes", None):
            inst = "INSTINCT_AUTONOMIC:\n" + "\n".join(self._instinct_turn_notes[:8]) + "\n\n"
        return (
            "SYSTEM:\n"
            f"{host_hint}"
            f"{llm_ops}"
            f"{oc}"
            f"{organ_part}"
            f"{physio_part}"
            f"{inst}"
            f"You are {self._llm_identity}. No cloud LLM API. Use tools if helpful. Be human-like, concise, and safe.\n"
            f"{persona}"
            f"{self.session.context_text()}\n\n"
            f"LONGTERM_RECALL:\n{mem_txt}\n\n"
            f"TOOL_OBSERVATIONS:\n{obs_txt}\n\n"
            f"USER:\n{user_input}\n"
            "ASSISTANT:\n"
        )

    def _tool_first_answer(self, user_input: str, tool_results: List[ToolResult]) -> str | None:
        """
        Because the self-made tiny models are not yet reliable for readable text,
        we generate a human-like, tool-grounded answer when tools ran.
        """
        for r in tool_results:
            if not r.ok:
                continue
            rw = r.data.get("real_world")
            if rw == "clock":
                d = r.data
                return f"【实时时间】{d.get('local')} 周{d.get('weekday_zh')}（{d.get('tz')}）\nISO: {d.get('iso')}"
            if rw == "weather":
                return "【天气】" + str(r.data.get("summary", "")).strip()
            if rw == "map":
                return (
                    f"【地图】{r.data.get('display_name', '')}\n"
                    f"坐标: {r.data.get('lat')}, {r.data.get('lon')}\n"
                    f"在浏览器打开: {r.data.get('openstreetmap_url', '')}"
                )

        # web_search summary
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
                return "\n".join(lines)

        # kb_search summary
        for r in tool_results:
            if not r.ok:
                continue
            items = r.data.get("items")
            if isinstance(items, list):
                if not items:
                    return "长期记忆里没有找到匹配内容。你可以换个关键词，或让我先联网搜集并写入记忆。"
                lines = ["我在长期记忆里找到这些内容（前几条）："]
                for it in items[:6]:
                    lines.append(f"- ({it.get('type')}) {it.get('content')}")
                return "\n".join(lines)

        # sim_run summary
        for r in tool_results:
            if not r.ok:
                continue
            if r.data.get("task_id") and "success_rate" in r.data:
                return f"仿真任务 {r.data.get('task_id')} 已运行：success_rate={r.data.get('success_rate')}, runs={r.data.get('runs')}"

        # kb_prune summary
        for r in tool_results:
            if not r.ok:
                continue
            if "deleted" in r.data and ("cutoff_ts_ms" in r.data or r.side_effects):
                return f"已清理长期记忆：deleted={r.data.get('deleted')}（older_than_days 门槛已应用）"

        # cmd_run summary
        for r in tool_results:
            if not r.ok:
                continue
            if "exit_code" in r.data and "stdout" in r.data:
                return f"命令执行完成：exit_code={r.data.get('exit_code')}\nstdout:\n{str(r.data.get('stdout',''))[:800]}"

        # web_fetch + extract_evidence summary
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

        # fs_read summary
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
        return os.environ.get("LIFERS_QUICK_WEB", "1").strip().lower() not in ("0", "false", "no", "off")

    def _web_search_reply(self, query: str, user_line: str) -> str:
        """闲聊快路径下的自动联网摘要（非固定模板）。"""
        from .tools import ToolCall

        u0 = user_line.strip()
        q = (query or u0).strip()[:240] or u0[:240]
        if _META_SELF_RE.search(u0):
            # 用户问「你能做什么」等：用更可检索的查询词，避免只搜一句口语
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
        """stack.remote_infer / LIFERS_REMOTE_CHAT：OpenAI 兼容 HTTPS（仅标准库），密钥来自环境变量。"""
        if os.environ.get("LIFERS_REMOTE_CHAT", "").strip().lower() not in ("1", "true", "yes", "on"):
            return None
        from lifers_brain.openai_compat_chat import chat_completion_text, resolve_api_key

        key = resolve_api_key()
        if not key:
            sys.stderr.write(
                "LIFERS_PROGRESS quick_chat remote_infer skipped: no API key; falling back to local brain.\n"
            )
            sys.stderr.flush()
            return None
        url = (os.environ.get("LIFERS_CHAT_URL") or "https://integrate.api.nvidia.com/v1/chat/completions").strip()
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
            url=url,
            api_key=key,
            model=model,
            messages=messages,
            max_tokens=mxt,
            temperature=0.35,
            timeout_sec=to,
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
        """无工具链时的轻量路径：先本地小模型；知识型或乱码/失败则自动联网（非硬编码寒暄）。"""
        u = user_line.strip()
        if not u:
            return "发一句你想问的内容就好。"

        rem = self._remote_quick_chat_attempt(u)
        if rem is not None:
            return rem

        if self._quick_web_enabled() and _quick_chat_wants_web(u):
            return self._web_search_reply(u, u)

        persona = (getattr(self, "_human_prompt_extra", "") or "").strip()
        inst_list = getattr(self, "_instinct_turn_notes", None) or []
        inst = "\n".join(inst_list[:4]) if inst_list else ""
        sess = self.session.context_text()
        if len(sess) > 1600:
            sess = sess[-1600:]
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
        text = self.brain.generate(prompt).strip()
        if (not text or text.startswith("(missing weights)")) and self._quick_web_enabled():
            return self._web_search_reply(u, u)
        if not text or text.startswith("(missing weights)"):
            return "本地模型不可用（缺权重）。可配置 weights 或联网后重试。"
        if not _quick_output_sane(text, u) and self._quick_web_enabled():
            return self._web_search_reply(u, u)
        if not _quick_output_sane(text, u):
            return (
                "本地 **TinyTransformer/Markov** 只是仓库自带的**极小字符语言模型**（训练目标为可跑通流水线，"
                "不是 ChatGPT 级对话模型），短输入或分布外话题很容易看起来像乱码。\n"
                "建议：用**完整中文句**提问；需要事实检索发 **search …**；或接 **stack.json → llm_ops** 里的外部大模型；"
                "若已关沙盒仍异常，检查 **LIFERS_QUICK_WEB=1** 与网络/代理。"
            )
        return text

    def _inject_realtime_clock_line(self) -> None:
        """每轮注入本机时间一行，供回答「现在几点」等时不臆造。"""
        from datetime import datetime

        try:
            now = datetime.now().astimezone()
            wd = "一二三四五六日"[now.weekday()]
            line = f"【实时·时钟】{now.strftime('%Y-%m-%d %H:%M:%S')} {now.tzname() or ''} 周{wd}（涉及当前时间/日期须引用本条）"
        except Exception:
            line = f"【实时·时钟】{time.strftime('%Y-%m-%d %H:%M:%S')}"
        ins = getattr(self, "_instinct_turn_notes", None) or []
        if not isinstance(ins, list):
            ins = []
        if not any("【实时·时钟】" in str(x) for x in ins):
            ins.insert(0, line)
        self._instinct_turn_notes = ins

    def _non_tool_fallback(self, user_input: str) -> str:
        return "未匹配工具链。可用：方案/plan、smart …（先记忆再网）、流程/workflow、search、URL、kb_search、cmd …"

    def _brain_fallback_with_context_pack(
        self,
        user_input: str,
        recalled: List[Dict[str, Any]],
        tool_obs: List[ToolResult],
        stack: Dict[str, Any],
    ) -> Optional[str]:
        """无结构化工具摘要时，可选把完整 SYSTEM 包交给 LocalBrain.generate。"""
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
            self.cfg.root_dir,
            stack,
            idle_sec,
            bool(user_input and user_input.strip()),
        )
        from .instincts import sync_prev_user_ts, tick_instincts_end, tick_instincts_start

        self._instinct_turn_notes = tick_instincts_start(self, idle_sec, stack)
        self._step_stack = stack
        try:
            return self._step_core(user_input, stack)
        finally:
            tick_instincts_end(self, stack, user_input)
            sync_prev_user_ts(self, int(time.time() * 1000))

    def quick_chat(self, user_line: str) -> str:
        """
        任务流「闲聊快路径」：与 step() 相同的前奏/收尾（本能、生理、会话），仅走本地小模型短回复。
        由 taskflow 分发器在分类为 CHAT_QUICK 时调用；勿用于含工具链的输入。
        """
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
            self.cfg.root_dir,
            stack,
            idle_sec,
            bool(user_line and user_line.strip()),
        )
        from .instincts import sync_prev_user_ts, tick_instincts_end, tick_instincts_start

        self._instinct_turn_notes = tick_instincts_start(self, idle_sec, stack)
        self._step_stack = stack
        try:
            self.session.add_turn("user", user_line)
            stripped = user_line.strip()
            self._inject_realtime_clock_line()
            reply = self._quick_chat_reply(stripped)
            self.session.add_turn("assistant", reply)
            return reply
        finally:
            tick_instincts_end(self, stack, user_line)
            sync_prev_user_ts(self, int(time.time() * 1000))

    def _step_core(self, user_input: str, _stack: Dict[str, Any]) -> str:
        self.session.add_turn("user", user_input)

        stripped = user_input.strip()
        low = stripped.lower()
        self._inject_realtime_clock_line()

        # --- 只出方案、不执行 ---
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
            """
            Transaction wrapper: dry_run -> execute -> verify; rollback on failure.
            """
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
            sys.stderr.write(
                f"LIFERS_PROGRESS [本能·实时 {idx+1}/{rw_cap}] {call.name} {call.args.get('action', '')}\n"
            )
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

        # --- 智搜：先 KB，无命中再联网 ---
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
            # If planner inserted extract_evidence placeholder, fill from previous fetch.
            if call.name == "extract_evidence" and tool_results:
                prev_text = tool_results[-1].data.get("text", "")
                call = ToolCall(name="extract_evidence", args={"text": prev_text, "max_snippets": 5}, expected_effect=call.expected_effect, mode=call.mode)

            # High-risk tools always run as a transaction.
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

        # After web_search, auto-open the top result and extract evidence (no need to ask).
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

        # Auto: persist web evidence into long-term memory (offline KB) via tool.
        kb_items: List[Dict[str, Any]] = []
        last_url: str | None = None
        for r in tool_results:
            if not r.ok:
                continue
            if r.data.get("type") == "file":
                # Don't auto-store file contents; too risky/noisy. Store a small reference only.
                kb_items.append(
                    {
                        "type": "episode",
                        "content": {"event": "fs_read", "path": r.data.get("path"), "note": "file read"},
                        "importance": 0.2,
                        "source": "tool:fs_read",
                    }
                )
            if "results" in r.data and isinstance(r.data.get("results"), list):
                kb_items.append(
                    {
                        "type": "tool_result",
                        "content": {"event": "web_search", "query": user_input, "results": r.data.get("results")[:5]},
                        "importance": 0.35,
                        "source": "tool:web_search",
                    }
                )
            if "url" in r.data and "text" in r.data:
                last_url = str(r.data.get("url", "")).strip() or last_url
                kb_items.append(
                    {
                        "type": "tool_result",
                        "key": str(r.data.get("url", "")).strip() or None,
                        "content": {"event": "web_fetch", "url": r.data.get("url"), "truncated": r.data.get("truncated", False)},
                        "importance": 0.55,
                        "source": "tool:web_fetch",
                    }
                )
            if "snippets" in r.data and isinstance(r.data.get("snippets"), list):
                kb_items.append(
                    {
                        "type": "fact",
                        "key": (f"{last_url}#evidence" if last_url else None),
                        "content": {"event": "evidence", "snippets": r.data.get("snippets")[:5]},
                        "importance": 0.75,
                        "source": "tool:extract_evidence",
                    }
                )

        if kb_items:
            kb_res = self.tools.dispatch(ToolCall(name="kb_upsert", args={"items": kb_items}, expected_effect="store KB items", mode="execute"))
            tool_results.append(kb_res)

        # Auto-compact: if we fetched a URL and extracted evidence, write/update a fact summary.
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

        fallback_body = self._non_tool_fallback(user_input)
        brain_try = self._brain_fallback_with_context_pack(user_input, recalled, tool_results, _stack)
        if brain_try is not None:
            fallback_body = brain_try

        answer = header + "\n\n【执行结果】\n" + fallback_body

        self.session.add_turn("assistant", answer)

        # Minimal memory write policy: store explicit preferences/commitments markers.
        if "记住" in user_input or "以后" in user_input:
            self.longterm.add(MemoryItem(type="preference", content=user_input, importance=0.8, source="user", ts_ms=int(time.time()*1000)))

        return answer

