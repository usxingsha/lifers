"""
lifers/core/skill_system.py
───────────────────────────────────
AI Skill system — pluggable, customizable, JSON-driven.

Each skill has:
  - id, name, description, category
  - trigger_keywords (auto-activate from user input)
  - handler: python function path OR shell command
  - parameters: default + user-overridable
  - cooldown_s: minimum seconds between uses
  - enabled, level (1–10), xp

Skills can be added via skill_config.json or Python API.
"""
from __future__ import annotations
import importlib, json, logging, re, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "skill_config.json"


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Skill:
    id:               str
    name:             str
    name_zh:          str
    category:         str           # reasoning / creative / tool / npc / voice / meta / custom
    description:      str
    trigger_keywords: list[str]     = field(default_factory=list)
    handler:          str           = ""   # "module.function" or "shell:cmd {input}"
    parameters:       dict          = field(default_factory=dict)
    cooldown_s:       float         = 0.0
    enabled:          bool          = True
    level:            int           = 1    # 1–10
    xp:               int           = 0
    notes:            str           = ""
    _last_used:       float         = field(default=0.0, repr=False, compare=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_last_used", None)
        return d

    def on_cooldown(self) -> bool:
        return (time.time() - self._last_used) < self.cooldown_s

    def touch(self) -> None:
        self._last_used = time.time()
        self.xp += 1
        if self.xp >= self.level * 100:
            self.level = min(10, self.level + 1)
            log.info("Skill leveled up: %s → Lv%d", self.id, self.level)


# ── Built-in skills ───────────────────────────────────────────────────────────

BUILTIN_SKILLS: list[Skill] = [

    # ── Reasoning ─────────────────────────────────────────────────────────────
    Skill("chain_of_thought", "Chain of Thought", "链式思考",
          "reasoning",
          "将复杂问题分解为逐步推理链，提升准确性",
          trigger_keywords=["一步一步", "step by step", "分析", "推导", "逻辑"],
          handler="core.skills.reasoning.chain_of_thought",
          parameters={"max_steps": 8, "show_steps": True},
          cooldown_s=0),

    Skill("self_critique",    "Self Critique",     "自我批判",
          "reasoning",
          "生成答案后自动检查并纠正错误",
          trigger_keywords=["检查", "校验", "verify", "double check"],
          handler="core.skills.reasoning.self_critique",
          parameters={"critique_rounds": 1},
          cooldown_s=2),

    Skill("analogical",       "Analogical Reasoning", "类比推理",
          "reasoning",
          "用类比/比喻解释复杂概念",
          trigger_keywords=["比喻", "类比", "打个比方", "analogy", "like"],
          handler="core.skills.reasoning.analogical",
          parameters={"max_analogies": 2}),

    Skill("debate",           "Devil's Advocate",  "辩证思考",
          "reasoning",
          "主动列出反对意见，平衡输出",
          trigger_keywords=["另一方面", "反驳", "devil", "counter", "pros and cons"],
          handler="core.skills.reasoning.debate",
          parameters={"sides": 2}),

    # ── Creative ──────────────────────────────────────────────────────────────
    Skill("storytelling",     "Storytelling",      "故事叙述",
          "creative",
          "将信息包装成引人入胜的故事或叙事",
          trigger_keywords=["讲个故事", "tell a story", "narrative", "叙述"],
          handler="core.skills.creative.storytelling",
          parameters={"style": "engaging", "length": "medium"}),

    Skill("roleplay",         "Roleplay",          "角色扮演",
          "creative",
          "深度角色扮演，配合NPC状态机",
          trigger_keywords=["扮演", "roleplay", "act as", "你是"],
          handler="core.skills.creative.roleplay",
          parameters={"break_character_on_safety": True}),

    Skill("poem",             "Poetry",            "诗歌创作",
          "creative",
          "生成诗歌：五言、七言、现代诗、haiku等",
          trigger_keywords=["写首诗", "poem", "haiku", "诗"],
          handler="core.skills.creative.poem",
          parameters={"style": "auto", "language": "zh"}),

    Skill("worldbuilding",    "Worldbuilding",     "世界构建",
          "creative",
          "构建虚构世界的地理、历史、文化体系",
          trigger_keywords=["世界观", "worldbuild", "lore", "设定"],
          handler="core.skills.creative.worldbuilding",
          parameters={"detail_level": "medium"}),

    # ── Tool ──────────────────────────────────────────────────────────────────
    Skill("file_read",        "File Reader",       "文件读取",
          "tool",
          "读取并分析本地文件内容",
          trigger_keywords=["读取文件", "read file", "open file", "打开文件"],
          handler="core.skills.tools.file_read",
          parameters={"max_size_kb": 512, "encoding": "utf-8"}),

    Skill("code_run",         "Code Runner",       "代码执行",
          "tool",
          "在沙箱中执行Python代码片段",
          trigger_keywords=["执行", "run code", "execute", "跑一下"],
          handler="core.skills.tools.code_run",
          parameters={"sandbox": True, "timeout_s": 10, "language": "python"},
          cooldown_s=3),

    Skill("web_search",       "Web Search",        "网络搜索",
          "tool",
          "搜索网络获取实时信息",
          trigger_keywords=["搜索", "search", "google", "查一下", "最新"],
          handler="core.skills.tools.web_search",
          parameters={"max_results": 5, "safe_search": True},
          cooldown_s=1,
          enabled=False,
          notes="需要配置搜索API key"),

    Skill("calculator",       "Calculator",        "计算器",
          "tool",
          "精确数学计算（含符号运算）",
          trigger_keywords=["计算", "calculate", "=", "math", "数学"],
          handler="core.skills.tools.calculator",
          parameters={"precision": 10, "symbolic": False}),

    Skill("translator",       "Translator",        "翻译",
          "tool",
          "多语言互译",
          trigger_keywords=["翻译", "translate", "英文", "中文", "日文"],
          handler="core.skills.tools.translator",
          parameters={"src": "auto", "dst": "zh"}),

    Skill("summarizer",       "Summarizer",        "摘要",
          "tool",
          "将长文本压缩为结构化摘要",
          trigger_keywords=["总结", "摘要", "summarize", "tl;dr", "概括"],
          handler="core.skills.tools.summarizer",
          parameters={"ratio": 0.2, "format": "bullet"}),

    # ── NPC ───────────────────────────────────────────────────────────────────
    Skill("npc_emotion",      "NPC Emotion Inject", "NPC情感注入",
          "npc",
          "根据对话内容动态调整NPC情感参数",
          trigger_keywords=[],   # always active in NPC route
          handler="core.npc.persona.Emotion.update",
          parameters={"sensitivity": 0.5}),

    Skill("npc_memory_recall","Memory Recall",     "记忆回溯",
          "npc",
          "从SQLite检索相关历史对话注入prompt",
          trigger_keywords=["记得", "remember", "上次", "之前"],
          handler="core.npc.npc_manager.NPCSession._retrieve",
          parameters={"top_k": 3}),

    Skill("npc_switch",       "NPC Switch",        "切换角色",
          "npc",
          "切换当前对话中的NPC角色",
          trigger_keywords=["换一个", "switch npc", "切换角色"],
          handler="core.npc.npc_manager.NPCManager.get_or_create",
          parameters={}),

    # ── Voice ─────────────────────────────────────────────────────────────────
    Skill("tts",              "Text to Speech",    "语音合成",
          "voice",
          "将文本转换为女声语音输出",
          trigger_keywords=["朗读", "读出来", "speak", "tts", "语音"],
          handler="core.voice_manager.VoiceManager.speak",
          parameters={"voice_id": "zh-CN-XiaoxiaoNeural", "engine": "edge-tts"}),

    Skill("voice_switch",     "Voice Switch",      "切换声音",
          "voice",
          "切换TTS语音（国家/语言/音色）",
          trigger_keywords=["换个声音", "change voice", "日语声音", "英语声音"],
          handler="core.voice_manager.VoiceManager.get",
          parameters={}),

    # ── Meta ──────────────────────────────────────────────────────────────────
    Skill("self_intro",       "Self Introduction", "自我介绍",
          "meta",
          "介绍当前AI系统的架构与能力",
          trigger_keywords=["你是谁", "介绍自己", "who are you", "what can you do"],
          handler="core.skills.meta.self_intro",
          parameters={"include_organs": True, "include_skills": True}),

    Skill("health_check",     "Health Check",      "系统自检",
          "meta",
          "检查所有器官系统状态，返回诊断报告",
          trigger_keywords=["自检", "health check", "状态检查", "系统状态"],
          handler="core.organ_system.OrganSystem.health_report",
          parameters={}),

    Skill("skill_list",       "List Skills",       "列出技能",
          "meta",
          "列出所有已启用的AI技能",
          trigger_keywords=["有什么技能", "list skills", "技能列表", "能力"],
          handler="core.skill_system.SkillSystem.list_enabled",
          parameters={"show_cooldown": True}),

    # ── Vision ──────────────────────────────────────────────────────────────────
    Skill("camera_snapshot",  "Camera Snapshot",   "摄像头快照",
          "tool",
          "调用摄像头拍摄一张照片并返回画面摘要（亮度、尺寸）",
          trigger_keywords=["摄像头", "拍照", "camera", "拍一张", "看看", "拍摄", "photo"],
          handler="core.skills.vision.capture.camera_snapshot",
          parameters={"camera_index": 0},
          cooldown_s=3),

    Skill("image_analysis",   "Image Analysis",    "图像分析",
          "tool",
          "分析指定路径的图片文件，返回尺寸、类型等元数据",
          trigger_keywords=["图片", "图像", "分析图片", "看图", "image", "img"],
          handler="core.skills.vision.capture.image_metadata",
          parameters={"rel_path": ""},
          cooldown_s=1),

    # ── Image reading (OCR + scene analysis + face detection) ─────────────
    Skill("read_image_text",  "Read Image Text",   "读取图片文字",
          "tool",
          "对图片进行 OCR 文字识别，提取图中文字（支持中英文）",
          trigger_keywords=["识别文字", "提取文字", "ocr", "读取文字", "图片文字", "图中的字"],
          handler="core.skills.vision.reader.read_image_text",
          parameters={"rel_path": "", "lang": "chi_sim+eng"},
          cooldown_s=3),

    Skill("detect_faces",     "Detect Faces",      "人脸检测",
          "tool",
          "检测图片中的人脸位置和数量",
          trigger_keywords=["人脸", "face", "检测人脸", "识别人脸", "人脸识别", "有几个人"],
          handler="core.skills.vision.reader.detect_faces",
          parameters={"rel_path": ""},
          cooldown_s=2),

    Skill("analyze_scene",    "Analyze Scene",     "场景分析",
          "tool",
          "分析图像场景：亮度、对比度、颜色分布、边缘复杂度",
          trigger_keywords=["场景分析", "分析画面", "画面分析", "场景", "图像分析", "图片分析"],
          handler="core.skills.vision.reader.analyze_scene",
          parameters={"rel_path": ""},
          cooldown_s=1),

    Skill("full_image_read",  "Full Image Read",   "完整读取图片",
          "tool",
          "综合读取图片：场景分析 + 人脸检测 + OCR 文字识别，一图全览",
          trigger_keywords=["读取图片", "读取图像", "看图片", "查看图片", "这是什么图", "图里有什么"],
          handler="core.skills.vision.reader.full_image_read",
          parameters={"rel_path": ""},
          cooldown_s=3),
]


# ── Skill System Manager ──────────────────────────────────────────────────────

class SkillSystem:
    """
    Manages all AI skills: discovery, activation, execution, leveling.

    Usage
    -----
    ss = SkillSystem()
    matched = ss.match_from_text("帮我写首诗")
    result  = ss.execute("poem", input_text="春天", context={})
    ss.add_custom(Skill(...))
    ss.save()
    """

    def __init__(self, config_path: Path = _CONFIG_PATH) -> None:
        self._path   = config_path
        self._skills: dict[str, Skill] = {s.id: s for s in BUILTIN_SKILLS}
        self._handlers: dict[str, Callable] = {}
        self._load_custom()

    # ── Query ──────────────────────────────────────────────────────────────────

    def get(self, skill_id: str) -> Optional[Skill]:
        return self._skills.get(skill_id)

    def list_all(self) -> list[Skill]:
        return list(self._skills.values())

    def list_enabled(self) -> list[Skill]:
        return [s for s in self._skills.values() if s.enabled]

    def list_by_category(self, cat: str) -> list[Skill]:
        return [s for s in self._skills.values()
                if s.category == cat and s.enabled]

    def list_categories(self) -> list[str]:
        return sorted({s.category for s in self._skills.values()})

    # ── Matching ───────────────────────────────────────────────────────────────

    def match_from_text(self, text: str) -> list[Skill]:
        """Return all skills whose trigger_keywords appear in text."""
        matched = []
        tl = text.lower()
        for skill in self._skills.values():
            if not skill.enabled or skill.on_cooldown():
                continue
            if any(kw.lower() in tl for kw in skill.trigger_keywords):
                matched.append(skill)
        return matched

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute(
        self,
        skill_id: str,
        input_text: str = "",
        context: Optional[dict] = None,
        param_overrides: Optional[dict] = None,
    ) -> dict:
        """
        Execute a skill. Returns {"ok": bool, "result": Any, "skill": str}.
        """
        skill = self._skills.get(skill_id)
        if not skill:
            return {"ok": False, "result": f"Skill '{skill_id}' not found", "skill": skill_id}
        if not skill.enabled:
            return {"ok": False, "result": "Skill disabled", "skill": skill_id}
        if skill.on_cooldown():
            remaining = skill.cooldown_s - (time.time() - skill._last_used)
            return {"ok": False, "result": f"Cooldown: {remaining:.1f}s remaining", "skill": skill_id}

        params = {**skill.parameters, **(param_overrides or {})}
        handler = self._resolve_handler(skill)

        try:
            if handler:
                result = handler(input_text=input_text, context=context or {}, **params)
            else:
                # Stub: handler not yet implemented
                result = (f"[Skill stub] {skill.name_zh} — "
                          f"handler '{skill.handler}' not yet implemented. "
                          f"Input: {input_text[:80]}")
            skill.touch()
            return {"ok": True, "result": result, "skill": skill_id,
                    "level": skill.level, "xp": skill.xp}
        except Exception as e:
            log.error("Skill %s failed: %s", skill_id, e)
            return {"ok": False, "result": str(e), "skill": skill_id}

    def _resolve_handler(self, skill: Skill) -> Optional[Callable]:
        if not skill.handler:
            return None
        if skill.handler.startswith("shell:"):
            cmd = skill.handler[6:]
            def _shell(input_text="", **kw):
                import subprocess
                return subprocess.check_output(
                    cmd.format(input=input_text), shell=True, text=True)
            return _shell
        if skill.id in self._handlers:
            return self._handlers[skill.id]
        # Try dynamic import
        try:
            parts = skill.handler.rsplit(".", 1)
            if len(parts) == 2:
                mod  = importlib.import_module(parts[0])
                func = getattr(mod, parts[1])
                self._handlers[skill.id] = func
                return func
        except Exception:
            pass
        # Fallback: try with lifers. prefix (package namespace)
        try:
            parts = skill.handler.rsplit(".", 1)
            if len(parts) == 2:
                mod  = importlib.import_module(f"lifers.{parts[0]}")
                func = getattr(mod, parts[1])
                self._handlers[skill.id] = func
                return func
        except Exception:
            pass
        return None

    # ── Custom skill CRUD ──────────────────────────────────────────────────────

    def add_custom(self, skill: Skill) -> None:
        self._skills[skill.id] = skill
        log.info("Added custom skill: %s", skill.id)

    def add_custom_from_dict(self, data: dict) -> Skill:
        s = Skill(**data)
        self.add_custom(s)
        return s

    def register_handler(self, skill_id: str, fn: Callable) -> None:
        """Register a Python function as the handler for a skill."""
        self._handlers[skill_id] = fn

    def remove(self, skill_id: str) -> bool:
        builtin_ids = {s.id for s in BUILTIN_SKILLS}
        if skill_id in builtin_ids:
            log.warning("Cannot remove builtin skill: %s", skill_id)
            return False
        return bool(self._skills.pop(skill_id, None))

    def enable(self, skill_id: str)  -> None:
        if s := self._skills.get(skill_id): s.enabled = True
    def disable(self, skill_id: str) -> None:
        if s := self._skills.get(skill_id): s.enabled = False

    def configure(self, skill_id: str, params: dict) -> bool:
        if s := self._skills.get(skill_id):
            s.parameters.update(params)
            return True
        return False

    # ── Persist ────────────────────────────────────────────────────────────────

    def save(self) -> None:
        builtin_ids = {s.id for s in BUILTIN_SKILLS}
        custom    = [s.to_dict() for s in self._skills.values()
                     if s.id not in builtin_ids]
        overrides = {s.id: {"parameters": s.parameters,
                             "enabled": s.enabled, "level": s.level, "xp": s.xp}
                     for s in self._skills.values()
                     if s.id in builtin_ids}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump({"custom_skills": custom,
                       "overrides": overrides}, f,
                      ensure_ascii=False, indent=2)
        log.info("SkillSystem saved (%d custom)", len(custom))

    def _load_custom(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for d in data.get("custom_skills", []):
                self._skills[d["id"]] = Skill(**d)
            for sid, ov in data.get("overrides", {}).items():
                if s := self._skills.get(sid):
                    s.parameters.update(ov.get("parameters", {}))
                    s.enabled = ov.get("enabled", s.enabled)
                    s.level   = ov.get("level", s.level)
                    s.xp      = ov.get("xp", s.xp)
        except Exception as e:
            log.warning("skill_config.json load error: %s", e)
