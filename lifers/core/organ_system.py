"""
lifers/core/organ_system.py
───────────────────────────────────
Human organ → AI module mapping system.
Each organ maps to a real AI subsystem with status check + config.
Organs can be enabled/disabled/configured individually.
"""
from __future__ import annotations
import json, logging, time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Callable

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "organ_config.json"


class OrganStatus(str, Enum):
    HEALTHY   = "healthy"
    DEGRADED  = "degraded"
    OFFLINE   = "offline"
    UNKNOWN   = "unknown"


@dataclass
class OrganDiag:
    organ_id:   str
    status:     OrganStatus
    latency_ms: float = 0.0
    message:    str   = ""
    checked_at: float = field(default_factory=time.time)


@dataclass
class Organ:
    id:          str
    name_zh:     str          # 中文名
    name_en:     str          # English name
    system:      str          # body system (神经系统/循环系统/…)
    ai_module:   str          # which Python module it maps to
    ai_role:     str          # what it does in the AI pipeline
    enabled:     bool  = True
    config:      dict  = field(default_factory=dict)
    notes:       str   = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ── Organ definitions ─────────────────────────────────────────────────────────

BUILTIN_ORGANS: list[Organ] = [

    # ── 神经系统 Nervous ──────────────────────────────────────────────────────
    Organ("brain",          "大脑",     "Brain",
          "神经系统", "core.inference_router",
          "主推理中枢：意图分类 + 路由决策",
          config={"max_context_turns": 20, "intent_threshold": 0.6}),

    Organ("cerebellum",     "小脑",     "Cerebellum",
          "神经系统", "core.output_formatter",
          "输出协调：格式化、流式、幻觉过滤",
          config={"stream": True, "hallucination_filter": True}),

    Organ("brainstem",      "脑干",     "Brainstem",
          "神经系统", "scripts.agent_bridge",
          "基础生命维持：HTTP桥接服务器，持续运行",
          config={"port": 55555, "host": "127.0.0.1"}),

    Organ("spinal_cord",    "脊髓",     "Spinal Cord",
          "神经系统", "core.input_validator",
          "反射弧：输入快速校验与拦截",
          config={"max_chars": 4096, "strict": False}),

    Organ("synapse",        "突触",     "Synapse",
          "神经系统", "core.npc.state_machine",
          "神经信号传递：NPC状态转换触发",
          config={"emotion_decay": 0.85}),

    # ── 感觉系统 Sensory ──────────────────────────────────────────────────────
    Organ("eyes",           "眼睛",     "Eyes",
          "感觉系统", "core.vision",
          "视觉感知：图像/截图输入处理",
          enabled=False,
          config={"ocr_enabled": False, "screenshot_interval_ms": 0},
          notes="vision模块待实现"),

    Organ("ears",           "耳朵",     "Ears",
          "感觉系统", "core.voice_manager",
          "听觉输入：语音识别(STT)",
          config={"stt_engine": "whisper", "language": "zh-CN"}),

    Organ("mouth",          "嘴巴",     "Mouth",
          "感觉系统", "core.voice_manager",
          "语音输出：TTS语音合成",
          config={"tts_engine": "edge-tts",
                  "default_voice": "zh-CN-XiaoxiaoNeural",
                  "rate": "+0%", "pitch": "+0Hz"}),

    Organ("skin",           "皮肤",     "Skin",
          "感觉系统", "core.input_validator",
          "触觉感知：外部刺激检测与过滤（输入边界）",
          config={"sensitivity": "normal"}),

    Organ("nose",           "鼻子",     "Nose",
          "感觉系统", "core.context_sniffer",
          "嗅觉：上下文环境探测（文件类型/语言检测）",
          enabled=False,
          notes="context_sniffer待实现"),

    # ── 循环系统 Circulatory ──────────────────────────────────────────────────
    Organ("heart",          "心脏",     "Heart",
          "循环系统", "core.npc.persona",
          "情感泵：驱动NPC情感状态更新",
          config={"emotion_decay": 0.85, "beat_interval_turns": 1}),

    Organ("blood",          "血液",     "Blood",
          "循环系统", "core.session_bus",
          "数据流：session消息在模块间流通",
          enabled=False,
          notes="session_bus待实现"),

    Organ("veins",          "血管",     "Veins",
          "循环系统", "scripts.agent_bridge",
          "传输管道：HTTP/SSE消息通道",
          config={"timeout_ms": 30000, "retry": 3}),

    # ── 消化系统 Digestive ────────────────────────────────────────────────────
    Organ("stomach",        "胃",       "Stomach",
          "消化系统", "core.input_validator",
          "消化/解析：文件上下文收集与预处理",
          config={"max_files": 48, "max_depth": 6}),

    Organ("intestine",      "肠道",     "Intestine",
          "消化系统", "core.inference_router",
          "营养吸收：tokenizer分词，提取有效token",
          config={"vocab_size": 32000, "max_length": 4096}),

    Organ("liver",          "肝脏",     "Liver",
          "消化系统", "core.output_formatter",
          "解毒/过滤：去除幻觉、过滤低置信度输出",
          config={"confidence_threshold": 0.4}),

    # ── 骨骼肌肉 Musculoskeletal ──────────────────────────────────────────────
    Organ("skeleton",       "骨骼",     "Skeleton",
          "骨骼系统", "core.skill_system",
          "支撑结构：stack.json + 各config文件",
          config={"config_file": "config/stack.json"}),

    Organ("muscles",        "肌肉",     "Muscles",
          "骨骼系统", "core.skill_system",
          "执行力：技能系统，主动行动执行",
          config={"max_concurrent_skills": 4}),

    Organ("hands",          "双手",     "Hands",
          "骨骼系统", "tools.agent_tools",
          "操作能力：工具调用、文件操作、代码执行",
          enabled=False,
          notes="agent_tools目录待实现"),

    # ── 免疫系统 Immune ───────────────────────────────────────────────────────
    Organ("immune",         "免疫系统", "Immune System",
          "免疫系统", "core.input_validator",
          "安全防护：prompt注入检测、内容过滤",
          config={"strict": False, "injection_block": True}),

    Organ("lymph",          "淋巴",     "Lymph Nodes",
          "免疫系统", "core.threat_logger",
          "威胁记录：安全事件日志与审计",
          enabled=False,
          notes="threat_logger待实现"),

    # ── 记忆/存储 Memory ──────────────────────────────────────────────────────
    Organ("hippocampus",    "海马体",   "Hippocampus",
          "记忆系统", "core.npc.npc_manager",
          "记忆形成：NPC长期记忆写入SQLite",
          config={"db_path": "memory/lifers.sqlite3", "retrieval_top_k": 5}),

    Organ("cortex",         "皮质",     "Cortex",
          "记忆系统", "core.inference_router",
          "长期推理：高阶语义理解与模式识别",
          config={"context_window": 4096}),

    Organ("amygdala",       "杏仁核",   "Amygdala",
          "记忆系统", "core.npc.state_machine",
          "情绪记忆：触发情绪状态机转换",
          config={"anger_threshold": 0.25}),

    # ── 内分泌 Endocrine ──────────────────────────────────────────────────────
    Organ("hormones",       "激素",     "Hormones",
          "内分泌系统", "core.npc.persona",
          "情绪调节器：emotion.valence/arousal衰减与更新",
          config={"decay": 0.85, "update_per_turn": True}),

    # ── 生殖/再生 Regenerative ────────────────────────────────────────────────
    Organ("stem_cells",     "干细胞",   "Stem Cells",
          "再生系统", "core.input_validator",
          "自我修复：bootstrap脚本，重建缺失文件",
          config={"auto_repair": True}),
]


# ── Organ System Manager ──────────────────────────────────────────────────────

class OrganSystem:
    """
    Manages all organ→module mappings, runs health checks, supports custom organs.

    Usage
    -----
    os = OrganSystem()
    report = os.check_all()
    os.configure("mouth", {"default_voice": "ja-JP-NanamiNeural"})
    os.add_custom(Organ(...))
    os.save()
    """

    def __init__(self, config_path: Path = _CONFIG_PATH) -> None:
        self._path   = config_path
        self._organs: dict[str, Organ] = {o.id: o for o in BUILTIN_ORGANS}
        self._load_custom()

    # ── Query ──────────────────────────────────────────────────────────────────

    def get(self, organ_id: str) -> Optional[Organ]:
        return self._organs.get(organ_id)

    def list_all(self) -> list[Organ]:
        return list(self._organs.values())

    def list_by_system(self, system: str) -> list[Organ]:
        return [o for o in self._organs.values()
                if o.system == system]

    def list_systems(self) -> list[str]:
        return sorted({o.system for o in self._organs.values()})

    def list_enabled(self) -> list[Organ]:
        return [o for o in self._organs.values() if o.enabled]

    def list_disabled(self) -> list[Organ]:
        return [o for o in self._organs.values() if not o.enabled]

    # ── Config ─────────────────────────────────────────────────────────────────

    def configure(self, organ_id: str, config: dict) -> bool:
        if o := self._organs.get(organ_id):
            o.config.update(config)
            log.info("Configured organ %s: %s", organ_id, config)
            return True
        return False

    def enable(self, organ_id: str) -> None:
        if o := self._organs.get(organ_id):
            o.enabled = True

    def disable(self, organ_id: str) -> None:
        if o := self._organs.get(organ_id):
            o.enabled = False

    # ── Custom organ CRUD ──────────────────────────────────────────────────────

    def add_custom(self, organ: Organ) -> None:
        self._organs[organ.id] = organ
        log.info("Added custom organ: %s", organ.id)

    def remove(self, organ_id: str) -> bool:
        builtin_ids = {o.id for o in BUILTIN_ORGANS}
        if organ_id in builtin_ids:
            log.warning("Cannot remove builtin organ: %s", organ_id)
            return False
        return bool(self._organs.pop(organ_id, None))

    # ── Health check ───────────────────────────────────────────────────────────

    def check_all(self, brain_root: Optional[Path] = None) -> list[OrganDiag]:
        root = brain_root or Path(__file__).parent.parent
        results = []
        for organ in self._organs.values():
            results.append(self._check_organ(organ, root))
        return results

    def check_one(self, organ_id: str,
                  brain_root: Optional[Path] = None) -> OrganDiag:
        root  = brain_root or Path(__file__).parent.parent
        organ = self._organs.get(organ_id)
        if not organ:
            return OrganDiag(organ_id, OrganStatus.UNKNOWN,
                             message="Organ not found")
        return self._check_organ(organ, root)

    def _check_organ(self, organ: Organ, root: Path) -> OrganDiag:
        if not organ.enabled:
            return OrganDiag(organ.id, OrganStatus.OFFLINE,
                             message="Disabled by config")
        t0 = time.perf_counter()
        # Map module path → file path
        rel = organ.ai_module.replace(".", "/") + ".py"
        path = root / rel
        # Also check without .py (directory package)
        path_pkg = root / organ.ai_module.replace(".", "/") / "__init__.py"

        if path.exists():
            status = OrganStatus.HEALTHY
            msg    = f"Module found: {rel}"
        elif path_pkg.exists():
            status = OrganStatus.HEALTHY
            msg    = f"Package found: {rel}"
        else:
            # Degraded if it's a known planned module, offline if unknown
            if organ.notes and "待实现" in organ.notes:
                status = OrganStatus.DEGRADED
                msg    = f"Planned module not yet implemented: {rel}"
            else:
                status = OrganStatus.OFFLINE
                msg    = f"Module missing: {rel}"

        latency = (time.perf_counter() - t0) * 1000
        return OrganDiag(organ.id, status, round(latency, 2), msg)

    def health_report(self, brain_root: Optional[Path] = None) -> dict:
        diags  = self.check_all(brain_root)
        report = {
            "timestamp": time.time(),
            "summary": {
                OrganStatus.HEALTHY:  0,
                OrganStatus.DEGRADED: 0,
                OrganStatus.OFFLINE:  0,
                OrganStatus.UNKNOWN:  0,
            },
            "organs": [],
        }
        for d in diags:
            report["summary"][d.status] += 1
            report["organs"].append({
                "id":         d.organ_id,
                "status":     d.status.value,
                "latency_ms": d.latency_ms,
                "message":    d.message,
            })
        return report

    # ── Persist ────────────────────────────────────────────────────────────────

    def save(self) -> None:
        builtin_ids = {o.id for o in BUILTIN_ORGANS}
        custom   = [o.to_dict() for o in self._organs.values()
                    if o.id not in builtin_ids]
        overrides = {o.id: {"config": o.config, "enabled": o.enabled}
                     for o in self._organs.values()
                     if o.id in builtin_ids}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump({"custom_organs": custom,
                       "overrides": overrides}, f,
                      ensure_ascii=False, indent=2)
        log.info("OrganSystem saved (%d custom, %d overrides)",
                 len(custom), len(overrides))

    def _load_custom(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for d in data.get("custom_organs", []):
                self._organs[d["id"]] = Organ(**d)
            for oid, ov in data.get("overrides", {}).items():
                if o := self._organs.get(oid):
                    o.config.update(ov.get("config", {}))
                    o.enabled = ov.get("enabled", o.enabled)
        except Exception as e:
            log.warning("organ_config.json load error: %s", e)
