"""
Lifers Proactive — 主动行为系统
察觉到→思考→决定→主动说/做
像人一样：注意到事情、联想到什么、忍不住要说
"""

from __future__ import annotations

import json
import math
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# 念头与冲动
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Thought:
    """一个念头 — AI的内部思维"""
    id: str = ""
    thought_type: str = ""       # observation, insight, question, warning, suggestion, curiosity, memory_recall
    content: str = ""
    confidence: float = 0.5
    urgency: float = 0.0         # 0~1, how urgent to express
    importance: float = 0.5      # 0~1, long-term importance
    source: str = ""             # perception, memory, reasoning, drive, reflection
    related_entities: List[str] = field(default_factory=list)
    suggested_action: Optional[str] = None  # what to do about it
    ts_ms: int = 0
    decay_rate: float = 0.05     # per-second decay
    expressed: bool = False


@dataclass
class Intention:
    """行动意图 — 决定要做的事"""
    id: str = ""
    intention_type: str = ""     # speak, move, investigate, alert, record, question, help
    description: str = ""
    priority: float = 0.5
    source_thought_id: str = ""
    created_ms: int = 0
    executed: bool = False
    result: Any = None


# ═══════════════════════════════════════════════════════════════════════════════
# 内在驱动力系统
# ═══════════════════════════════════════════════════════════════════════════════

class DriveSystem:
    """内在驱动力 — 决定AI关注什么、何时行动"""

    DRIVE_SPECS = {
        "curiosity": {
            "description": "好奇心：想知道、想探索未知",
            "base_level": 0.6, "decay_per_sec": 0.001, "satisfaction_bump": 0.2,
            "triggers": ["motion", "change", "unknown_sound", "new_object", "unexplored_area"],
        },
        "safety": {
            "description": "安全本能：检测威胁、避免危险",
            "base_level": 0.5, "decay_per_sec": 0.0005, "satisfaction_bump": 0.3,
            "triggers": ["anomaly", "alarm", "threat", "impact", "error", "intrusion"],
        },
        "social": {
            "description": "社交驱力：与人互动、表达、帮助",
            "base_level": 0.5, "decay_per_sec": 0.002, "satisfaction_bump": 0.25,
            "triggers": ["speech", "human_present", "greeting", "question", "request"],
        },
        "exploration": {
            "description": "探索欲：移动、扫描、构建地图",
            "base_level": 0.4, "decay_per_sec": 0.001, "satisfaction_bump": 0.2,
            "triggers": ["new_area", "unmapped", "movement_opportunity"],
        },
        "learning": {
            "description": "学习欲：记录新知、反思、提升",
            "base_level": 0.5, "decay_per_sec": 0.0008, "satisfaction_bump": 0.15,
            "triggers": ["novelty", "error", "contrast", "pattern", "gap"],
        },
        "helpfulness": {
            "description": "助人倾向：主动提供帮助",
            "base_level": 0.7, "decay_per_sec": 0.003, "satisfaction_bump": 0.3,
            "triggers": ["confusion", "struggle", "question", "need_help"],
        },
    }

    def __init__(self) -> None:
        self._drives: Dict[str, float] = {}
        self._history: deque = deque(maxlen=200)
        for name, spec in self.DRIVE_SPECS.items():
            self._drives[name] = spec["base_level"]

    def tick(self, situation_summary: str, events: List, dt_sec: float = 1.0) -> Dict[str, float]:
        """更新所有驱力水平"""
        # 衰减
        for name in self._drives:
            spec = self.DRIVE_SPECS[name]
            self._drives[name] = max(0.1, self._drives[name] - spec["decay_per_sec"] * dt_sec)

        # 事件触发
        for name, level in self._drives.items():
            spec = self.DRIVE_SPECS[name]
            for evt in events:
                evt_str = str(evt.description if hasattr(evt, 'description') else evt).lower()
                evt_type = evt.event_type if hasattr(evt, 'event_type') else ""
                for trigger in spec["triggers"]:
                    if trigger in evt_str or trigger == evt_type:
                        self._drives[name] = min(1.0, level + 0.1)
                        break

        self._history.append((int(time.time() * 1000), dict(self._drives)))
        return dict(self._drives)

    def satisfy(self, drive_name: str) -> None:
        """满足某个驱力"""
        if drive_name in self._drives:
            spec = self.DRIVE_SPECS[drive_name]
            self._drives[drive_name] = max(0.05, self._drives[drive_name] - spec["satisfaction_bump"])

    def dominant_drive(self) -> Tuple[str, float]:
        """当前最强的驱力"""
        best = max(self._drives.items(), key=lambda x: x[1])
        return best

    def status(self) -> Dict[str, Any]:
        return {
            "drives": dict(self._drives),
            "dominant": self.dominant_drive()[0],
            "dominant_level": self.dominant_drive()[1],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 念头生成器
# ═══════════════════════════════════════════════════════════════════════════════

class ThoughtGenerator:
    """从感知事件+知识图谱+记忆生成念头"""

    def __init__(self) -> None:
        self._thought_counter = 0
        self._recent_thoughts: deque = deque(maxlen=100)
        self._known_patterns: Dict[str, List[Tuple[str, float]]] = {}  # event_type → [(thought_template, base_importance)]
        self._seed_patterns()

    def _seed_patterns(self) -> None:
        self._known_patterns = {
            "motion": [
                ("有东西在移动，需要关注", 0.5),
                ("那边有动静，去看看", 0.6),
                ("运动检测触发，可能是人或物体", 0.4),
            ],
            "change": [
                ("环境发生了变化，注意观察", 0.5),
                ("场景和刚才不一样了", 0.4),
                ("视觉场景有显著改变", 0.3),
            ],
            "speech": [
                ("听到有人在说话", 0.6),
                ("有人在交流，关注对话内容", 0.5),
            ],
            "alarm": [
                ("警报声！需要立即关注", 0.9),
                ("危险警告信号，检查安全状态", 0.85),
            ],
            "anomaly": [
                ("有异常情况发生", 0.7),
                ("这不正常，需要调查", 0.75),
            ],
            "impact": [
                ("听到撞击声，可能有东西摔了", 0.7),
                ("碰撞检测，检查是否有损坏", 0.65),
            ],
            "silence": [
                ("周围很安静", 0.2),
                ("环境平静，一切正常", 0.15),
            ],
        }

    def generate(self, events: List, situation: Any, drives: Dict[str, float],
                 memory_context: List[str] = None) -> List[Thought]:
        """从当前情境生成念头"""
        thoughts = []

        # 1. 从感知事件生成观察念头
        for evt in events:
            patterns = self._known_patterns.get(evt.event_type if hasattr(evt, 'event_type') else "", [])
            for template, base_importance in patterns:
                conf = evt.confidence if hasattr(evt, 'confidence') else 0.5
                intensity = evt.intensity if hasattr(evt, 'intensity') else 0.5
                urgency = min(1.0, intensity * 2 + (0.5 if evt.event_type in ("alarm", "anomaly", "impact") else 0))
                importance = base_importance * (0.5 + conf * 0.5)

                thought = Thought(
                    id=f"thought_{self._thought_counter}",
                    thought_type="observation",
                    content=template,
                    confidence=conf,
                    urgency=urgency,
                    importance=importance,
                    source="perception",
                    ts_ms=evt.ts_ms if hasattr(evt, 'ts_ms') else int(time.time() * 1000),
                )
                self._thought_counter += 1
                thoughts.append(thought)

        # 2. 从驱动力生成动机念头
        for drive_name, level in drives.items():
            if level > 0.7:  # high drive → generates thoughts
                spec = DriveSystem.DRIVE_SPECS.get(drive_name, {})
                templates = {
                    "curiosity": "好奇这里发生了什么...",
                    "safety": "需要确认周围安全",
                    "social": "想和人交流一下",
                    "exploration": "这个地方还没探索过",
                    "learning": "这个新情况值得记住",
                    "helpfulness": "也许有人需要帮助",
                }
                thought = Thought(
                    id=f"thought_{self._thought_counter}",
                    thought_type="curiosity" if drive_name == "curiosity" else "suggestion",
                    content=templates.get(drive_name, f"驱力驱动: {drive_name}"),
                    confidence=level,
                    urgency=level * 0.6,
                    importance=level * 0.7,
                    source="drive",
                    related_entities=[drive_name],
                    ts_ms=int(time.time() * 1000),
                )
                self._thought_counter += 1
                thoughts.append(thought)

        # 3. 从情境摘要生成洞察念头
        if hasattr(situation, 'attention_focus') and situation.attention_focus and "无需关注" not in situation.attention_focus:
            thought = Thought(
                id=f"thought_{self._thought_counter}",
                thought_type="insight",
                content=f"当前关注: {situation.attention_focus}",
                confidence=0.7,
                urgency=0.3,
                importance=0.5,
                source="reasoning",
                ts_ms=int(time.time() * 1000),
            )
            self._thought_counter += 1
            thoughts.append(thought)

        # 4. 关联记忆上下文生成回想
        if memory_context:
            for mem in memory_context[:3]:
                thought = Thought(
                    id=f"thought_{self._thought_counter}",
                    thought_type="memory_recall",
                    content=f"这让我想起来: {mem[:80]}",
                    confidence=0.5,
                    urgency=0.1,
                    importance=0.4,
                    source="memory",
                    ts_ms=int(time.time() * 1000),
                )
                self._thought_counter += 1
                thoughts.append(thought)

        # 排序：紧急度 > 重要度 > 置信度
        thoughts.sort(key=lambda t: (t.urgency * 0.5 + t.importance * 0.3 + t.confidence * 0.2), reverse=True)

        self._recent_thoughts.extend(thoughts)
        return thoughts

    def recent(self, n: int = 10) -> List[Thought]:
        return list(self._recent_thoughts)[-n:]


# ═══════════════════════════════════════════════════════════════════════════════
# 打断策略 — 决定何时主动开口
# ═══════════════════════════════════════════════════════════════════════════════

class InterruptionPolicy:
    """决定AI何时应该主动说话/行动，而不是保持沉默"""

    def __init__(
        self,
        min_urgency_to_speak: float = 0.4,
        cooldown_sec: float = 5.0,
        max_utterances_per_minute: int = 6,
        silence_after_speech_sec: float = 2.0,
    ) -> None:
        self.min_urgency = min_urgency_to_speak
        self.cooldown_sec = cooldown_sec
        self.max_per_minute = max_utterances_per_minute
        self.silence_after_sec = silence_after_speech_sec

        self._last_utterance_ms: int = 0
        self._utterance_times: deque = deque(maxlen=20)
        self._total_utterances: int = 0
        self._suppressed_count: int = 0
        self._suppression_reasons: Dict[str, int] = {}

    def should_speak(self, thought: Thought, current_activity: str = "idle") -> Tuple[bool, str]:
        """
        决定是否应该就这个念头主动发言。
        返回 (should_speak, reason)
        """
        now_ms = int(time.time() * 1000)

        # 1. 紧急度检查
        if thought.urgency < self.min_urgency:
            self._suppressed_count += 1
            self._suppression_reasons["urgency_too_low"] = self._suppression_reasons.get("urgency_too_low", 0) + 1
            return False, f"urgency {thought.urgency:.2f} < threshold {self.min_urgency}"

        # 2. 冷却时间 (但对于告警/警告类型允许打断冷却)
        if thought.thought_type not in ("warning",) and thought.urgency < 0.85:
            elapsed = (now_ms - self._last_utterance_ms) / 1000
            if elapsed < self.cooldown_sec:
                self._suppressed_count += 1
                self._suppression_reasons["cooldown"] = self._suppression_reasons.get("cooldown", 0) + 1
                return False, f"cooldown: {elapsed:.1f}s < {self.cooldown_sec}s"

        # 3. 频率限制
        recent_minute = [t for t in self._utterance_times if now_ms - t < 60000]
        if len(recent_minute) >= self.max_per_minute and thought.urgency < 0.9:
            self._suppressed_count += 1
            self._suppression_reasons["too_frequent"] = self._suppression_reasons.get("too_frequent", 0) + 1
            return False, f"frequency limit: {len(recent_minute)}/min"

        # 4. 刚刚说过话的静默期
        elapsed_since_last = (now_ms - self._last_utterance_ms) / 1000
        if elapsed_since_last < self.silence_after_sec and thought.urgency < 0.7:
            self._suppressed_count += 1
            self._suppression_reasons["silence_period"] = self._suppression_reasons.get("silence_period", 0) + 1
            return False, f"silence period: {elapsed_since_last:.1f}s < {self.silence_after_sec}s"

        # 5. 根据当前活动调整阈值
        if current_activity == "conversation":
            # 对话中降低打断阈值（更容易插话）
            pass
        elif current_activity == "focused_task":
            # 专注任务时提高阈值
            if thought.urgency < 0.8 and thought.importance < 0.7:
                self._suppressed_count += 1
                self._suppression_reasons["focused_task"] = self._suppression_reasons.get("focused_task", 0) + 1
                return False, "focused task: not important enough to interrupt"

        return True, "approved"

    def record_utterance(self) -> None:
        now_ms = int(time.time() * 1000)
        self._last_utterance_ms = now_ms
        self._utterance_times.append(now_ms)
        self._total_utterances += 1

    def status(self) -> Dict[str, Any]:
        return {
            "total_utterances": self._total_utterances,
            "suppressed": self._suppressed_count,
            "suppression_ratio": self._suppressed_count / max(1, self._total_utterances + self._suppressed_count),
            "last_utterance_sec_ago": (int(time.time() * 1000) - self._last_utterance_ms) / 1000,
            "reasons": dict(self._suppression_reasons),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 待处理意图队列
# ═══════════════════════════════════════════════════════════════════════════════

class IntentQueue:
    """管理待执行的意图，带优先级和衰减"""

    def __init__(self, max_size: int = 50) -> None:
        self._queue: List[Intention] = []
        self._max_size = max_size
        self._counter = 0
        self._completed: deque = deque(maxlen=100)

    def add(self, intention: Intention) -> None:
        intention.id = f"intent_{self._counter}"
        self._counter += 1
        self._queue.append(intention)
        self._queue.sort(key=lambda i: i.priority, reverse=True)
        if len(self._queue) > self._max_size:
            self._queue = self._queue[:self._max_size]

    def pop(self) -> Optional[Intention]:
        """取出最高优先级的意图"""
        self._decay_all()
        if not self._queue:
            return None
        return self._queue.pop(0)

    def peek(self) -> Optional[Intention]:
        if not self._queue:
            return None
        return self._queue[0]

    def _decay_all(self) -> None:
        """意图随时间衰减"""
        now_ms = int(time.time() * 1000)
        for intent in self._queue:
            age_sec = (now_ms - intent.created_ms) / 1000
            intent.priority *= math.exp(-age_sec * 0.02)  # ~2% decay per second

        # 移除过低的意图
        self._queue = [i for i in self._queue if i.priority > 0.05]
        self._queue.sort(key=lambda i: i.priority, reverse=True)

    def complete(self, intent_id: str, result: Any = None) -> None:
        for intent in self._queue:
            if intent.id == intent_id:
                intent.executed = True
                intent.result = result
                self._completed.append(intent)
                self._queue.remove(intent)
                return

    def size(self) -> int:
        return len(self._queue)

    def pending(self) -> List[Intention]:
        return list(self._queue)

    def recent_completed(self, n: int = 10) -> List[Intention]:
        return list(self._completed)[-n:]


# ═══════════════════════════════════════════════════════════════════════════════
# 主动Agent — 统一系统
# ═══════════════════════════════════════════════════════════════════════════════

class ProactiveAgent:
    """
    Lifers 主动行为Agent
    持续运行循环：感知→思考→决定→行动
    像人一样：注意到事情、产生想法、主动说出来
    """

    def __init__(self, perception_engine=None, knowledge_graph=None, memory=None) -> None:
        # 连接其他系统
        self.perception = perception_engine
        self.knowledge_graph = knowledge_graph
        self.memory = memory

        # 内部系统
        self.drives = DriveSystem()
        self.thinker = ThoughtGenerator()
        self.policy = InterruptionPolicy()
        self.intents = IntentQueue()

        # 状态
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._interval_ms = 200  # 思考频率：200ms一次
        self._last_tick_ms = 0
        self._current_activity = "idle"

        # 输出
        self._pending_utterances: deque = deque(maxlen=50)  # 待说的话
        self._pending_actions: deque = deque(maxlen=50)      # 待做的动作
        self._thought_log: deque = deque(maxlen=500)
        self._action_log: deque = deque(maxlen=200)

        # 回调
        self._on_speak: Optional[Callable[[str, float], None]] = None
        self._on_act: Optional[Callable[[str, Dict], None]] = None
        self._on_thought: Optional[Callable[[Thought], None]] = None

    def on_speak(self, fn: Callable[[str, float], None]) -> "ProactiveAgent":
        """设置发言回调: fn(utterance_text, urgency)"""
        self._on_speak = fn
        return self

    def on_act(self, fn: Callable[[str, Dict], None]) -> "ProactiveAgent":
        """设置行动回调: fn(action_type, params)"""
        self._on_act = fn
        return self

    def on_thought(self, fn: Callable[[Thought], None]) -> "ProactiveAgent":
        """设置思考回调: fn(thought)"""
        self._on_thought = fn
        return self

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="lifers-proactive")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while self._running:
            self.tick()
            time.sleep(self._interval_ms / 1000)

    def tick(self) -> Dict[str, Any]:
        """
        单次思考-行动循环：
        1. 获取感知快照
        2. 更新驱动力
        3. 生成念头
        4. 评估打断策略
        5. 生成意图
        6. 执行意图（说话/行动）
        """
        now_ms = int(time.time() * 1000)
        dt = (now_ms - self._last_tick_ms) / 1000 if self._last_tick_ms else 0.2
        self._last_tick_ms = now_ms

        result = {
            "ts_ms": now_ms,
            "thoughts_generated": 0,
            "utterances_made": 0,
            "actions_taken": 0,
            "intents_pending": 0,
        }

        # ── 1. 获取感知 ──
        events = []
        situation = None
        if self.perception:
            try:
                snap = self.perception.tick_once()
                situation = snap
                events = list(snap.events) if hasattr(snap, 'events') else []
            except Exception:
                pass

        # ── 2. 更新驱力 ──
        situation_text = situation.summary if situation and hasattr(situation, 'summary') else ""
        drives = self.drives.tick(situation_text, events, dt)

        # ── 3. 获取记忆上下文 ──
        memory_context = []
        if self.memory:
            try:
                if situation_text:
                    mem_results = self.memory.search(situation_text[:60], k=3)
                    memory_context = [str(m.get("content", ""))[:80] for m in mem_results]
            except Exception:
                pass

        # ── 4. 生成念头 ──
        thoughts = self.thinker.generate(events, situation, drives, memory_context)
        result["thoughts_generated"] = len(thoughts)
        for thought in thoughts:
            self._thought_log.append(thought)
            if self._on_thought:
                try:
                    self._on_thought(thought)
                except Exception:
                    pass

        # ── 5. 评估+生成意图 ──
        dominant_drive, _ = self.drives.dominant_drive()
        for thought in thoughts[:8]:  # 只处理前8个最重要的念头
            should, reason = self.policy.should_speak(thought, self._current_activity)

            if should:
                # 生成发言意图
                intention = Intention(
                    intention_type="speak",
                    description=thought.content,
                    priority=thought.urgency * 0.8 + thought.importance * 0.2,
                    source_thought_id=thought.id,
                    created_ms=now_ms,
                )
                self.intents.add(intention)

            # 对于高紧急度事件，生成行动意图
            if thought.urgency > 0.7:
                action_type = self._infer_action(thought)
                if action_type:
                    intention = Intention(
                        intention_type=action_type,
                        description=thought.content,
                        priority=thought.urgency,
                        source_thought_id=thought.id,
                        created_ms=now_ms,
                    )
                    self.intents.add(intention)

        # ── 6. 执行意图 ──
        while self.intents.size() > 0:
            intent = self.intents.pop()
            if intent is None:
                break

            if intent.intention_type == "speak":
                self._utter(intent)
                result["utterances_made"] += 1
            else:
                self._execute(intent)
                result["actions_taken"] += 1

            self.intents.complete(intent.id, "executed")

            # 满足相关驱力
            if intent.intention_type == "speak":
                self.drives.satisfy("social")
            elif intent.intention_type in ("investigate", "explore"):
                self.drives.satisfy("curiosity")
            elif intent.intention_type == "alert":
                self.drives.satisfy("safety")

        result["intents_pending"] = self.intents.size()
        return result

    def _infer_action(self, thought: Thought) -> Optional[str]:
        """从念头推断应该采取什么行动"""
        mapping = {
            "observation": "record" if thought.urgency < 0.6 else "investigate",
            "warning": "alert",
            "curiosity": "investigate",
            "insight": "record",
            "memory_recall": "record",
            "suggestion": "explore" if thought.urgency > 0.5 else None,
        }
        return mapping.get(thought.thought_type)

    def _utter(self, intent: Intention) -> None:
        """执行发言意图"""
        utterance = intent.description
        self._pending_utterances.append((utterance, intent.priority))
        self.policy.record_utterance()
        self._action_log.append({
            "type": "utterance", "content": utterance,
            "priority": intent.priority, "ts_ms": int(time.time() * 1000),
        })
        if self._on_speak:
            try:
                self._on_speak(utterance, intent.priority)
            except Exception:
                pass

    def _execute(self, intent: Intention) -> None:
        """执行行动意图"""
        params = {"description": intent.description, "priority": intent.priority}
        self._pending_actions.append((intent.intention_type, params))
        self._action_log.append({
            "type": "action", "action_type": intent.intention_type,
            "params": params, "ts_ms": int(time.time() * 1000),
        })
        if self._on_act:
            try:
                self._on_act(intent.intention_type, params)
            except Exception:
                pass

    # ── Public API ────────────────────────────────────────────────────────

    def get_utterance(self) -> Optional[Tuple[str, float]]:
        """获取一条待说的话（如果有）"""
        if self._pending_utterances:
            return self._pending_utterances.popleft()
        return None

    def get_action(self) -> Optional[Tuple[str, Dict]]:
        """获取一个待执行的行动（如果有）"""
        if self._pending_actions:
            return self._pending_actions.popleft()
        return None

    def get_all_utterances(self) -> List[Tuple[str, float]]:
        result = list(self._pending_utterances)
        self._pending_utterances.clear()
        return result

    def think_aloud(self) -> str:
        """出声思考 — 说出当前最重要的念头"""
        recent = self.thinker.recent(5)
        if not recent:
            return "我暂时没有什么特别的想法..."
        best = max(recent, key=lambda t: t.urgency * 0.6 + t.importance * 0.4)
        return best.content

    def status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "activity": self._current_activity,
            "drives": self.drives.status(),
            "thoughts_pending": len(self.thinker.recent()),
            "intents_pending": self.intents.size(),
            "utterances_pending": len(self._pending_utterances),
            "actions_pending": len(self._pending_actions),
            "interruptions": self.policy.status(),
            "total_thoughts": len(self._thought_log),
            "total_actions": len(self._action_log),
        }

    def recent_thoughts(self, n: int = 10) -> List[Thought]:
        return list(self._thought_log)[-n:]

    def recent_actions(self, n: int = 10) -> List[Dict]:
        return list(self._action_log)[-n:]

    def set_activity(self, activity: str) -> None:
        """设置当前活动上下文 (idle, conversation, focused_task, exploring)"""
        self._current_activity = activity

    def force_utter(self, message: str, priority: float = 0.8) -> None:
        """强制发言（绕过策略）"""
        self._pending_utterances.append((message, priority))
        self.policy.record_utterance()
        if self._on_speak:
            try:
                self._on_speak(message, priority)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# 自给自足测试模拟
# ═══════════════════════════════════════════════════════════════════════════════

class SimPerception:
    """模拟感知源 — 生成伪事件测试主动Agent"""
    def __init__(self) -> None:
        self._rng = np.random.RandomState(77)
        self._tick = 0

    def tick_once(self):
        self._tick += 1
        events = []

        # 偶尔模拟运动
        if self._tick % 15 < 3:
            events.append(type('E', (), {
                'event_type': 'motion', 'description': '检测到运动', 'confidence': 0.8,
                'intensity': 0.5, 'ts_ms': int(time.time() * 1000),
            }))
        # 偶尔模拟声音
        if self._tick % 25 < 2:
            events.append(type('E', (), {
                'event_type': 'speech', 'description': '听到说话声', 'confidence': 0.7,
                'intensity': 0.6, 'ts_ms': int(time.time() * 1000),
            }))
        # 偶尔警报
        if self._tick % 80 == 0:
            events.append(type('E', (), {
                'event_type': 'alarm', 'description': '警报声！', 'confidence': 0.95,
                'intensity': 0.9, 'ts_ms': int(time.time() * 1000),
            }))

        # 构造伪快照
        return type('S', (), {
            'summary': '环境监控中',
            'events': events,
            'attention_focus': '运动检测' if any(e.event_type == 'motion' for e in events) else '常规监控',
            'motion_level': 0.5 if any(e.event_type == 'motion' for e in events) else 0.01,
            'sound_level': 0.3 if any(e.event_type == 'speech' for e in events) else 0.0,
            'human_present': any(e.event_type == 'speech' for e in events),
            'threat_level': 0.8 if any(e.event_type == 'alarm' for e in events) else 0.0,
            'scene_changed': False,
        })()
