"""
Lifers Senses — 实时环境感知与情境分析
像人类一样：持续看、听、分析、理解周围发生的一切
纯 NumPy 实现，自给自足
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
# 事件基元
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PerceptEvent:
    """感知事件 — 五感输入的统一表示"""
    id: str = ""
    source: str = ""          # vision, audio, tactile, thermal
    event_type: str = ""      # motion, sound, change, presence, anomaly
    description: str = ""
    confidence: float = 0.5
    location: Tuple[float, float, float] = (0, 0, 0)
    intensity: float = 0.5    # 0~1
    duration_ms: int = 0
    ts_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SituationSnapshot:
    """当前情境快照 — AI对「现在发生了什么」的理解"""
    ts_ms: int = 0
    summary: str = ""                    # 一句话概括
    detected_objects: List[str] = field(default_factory=list)
    detected_sounds: List[str] = field(default_factory=list)
    motion_level: float = 0.0            # 0~1
    sound_level: float = 0.0             # 0~1
    human_present: bool = False
    scene_changed: bool = False
    attention_focus: str = ""            # 当前关注焦点
    threat_level: float = 0.0            # 0~1
    mood_estimate: str = "neutral"       # neutral, alert, relaxed, tense
    events: List[PerceptEvent] = field(default_factory=list)
    contextual_memory: List[str] = field(default_factory=list)  # 关联记忆


# ═══════════════════════════════════════════════════════════════════════════════
# 视觉分析器
# ═══════════════════════════════════════════════════════════════════════════════

class VisualAnalyzer:
    """图像分析：运动检测、变化感知、亮度/颜色分布、简易物体存在检测"""

    def __init__(self, motion_threshold: float = 0.03, change_threshold: float = 0.05) -> None:
        self.motion_threshold = motion_threshold
        self.change_threshold = change_threshold
        self._prev_frame: Optional[np.ndarray] = None
        self._baseline_frame: Optional[np.ndarray] = None
        self._baseline_age: int = 0
        self._frame_history: deque = deque(maxlen=30)
        self._motion_history: deque = deque(maxlen=100)
        self._change_events: deque = deque(maxlen=50)

    def analyze(self, frame: np.ndarray, ts_ms: int) -> Dict[str, Any]:
        """分析一帧图像，返回所有视觉洞察"""
        if frame is None or frame.size == 0:
            return self._empty_result()

        # 预处理：转灰度、降采样加速
        if frame.ndim == 3:
            gray = np.mean(frame, axis=2).astype(np.float32)
        else:
            gray = frame.astype(np.float32)

        # 降采样到 128x128 以内
        h, w = gray.shape
        scale = min(1.0, 128.0 / max(h, w))
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            indices_h = (np.linspace(0, h - 1, new_h)).astype(int)
            indices_w = (np.linspace(0, w - 1, new_w)).astype(int)
            gray = gray[indices_h][:, indices_w]

        self._frame_history.append(gray)

        result = {
            "brightness": float(np.mean(gray) / 255.0),
            "contrast": float(np.std(gray) / 128.0),
            "motion_detected": False,
            "motion_level": 0.0,
            "motion_hotspots": [],
            "change_detected": False,
            "change_level": 0.0,
            "scene_stable": True,
            "dominant_color": self._dominant_color(frame),
            "visual_attention_score": 0.0,
            "events": [],
        }

        # 运动检测（帧差法）
        if self._prev_frame is not None:
            diff = np.abs(gray - self._prev_frame)
            motion_mask = diff > (255 * self.motion_threshold)
            motion_level = float(np.mean(motion_mask))
            result["motion_level"] = motion_level
            self._motion_history.append((ts_ms, motion_level))

            if motion_level > 0.01:
                result["motion_detected"] = True
                hotspots = self._find_hotspots(motion_mask, gray.shape)
                result["motion_hotspots"] = hotspots[:5]

                if motion_level > 0.08:
                    result["events"].append(self._make_event(
                        "vision", "motion", f"检测到明显运动 (强度{motion_level:.2f})",
                        confidence=min(1.0, motion_level * 10), intensity=motion_level, ts_ms=ts_ms,
                        metadata={"hotspots": len(hotspots), "max_cluster": hotspots[0] if hotspots else {}},
                    ))

        # 场景变化检测（与基线比较）
        if self._baseline_frame is None or self._baseline_age > 300:
            self._baseline_frame = gray.copy()
            self._baseline_age = 0
        else:
            self._baseline_age += 1
            change_diff = np.abs(gray - self._baseline_frame)
            change_level = float(np.mean(change_diff > (255 * self.change_threshold)))
            result["change_level"] = change_level
            result["scene_stable"] = change_level < 0.02

            if change_level > 0.05:
                result["change_detected"] = True
                result["events"].append(self._make_event(
                    "vision", "change", f"场景显著变化 (变化度{change_level:.2f})",
                    confidence=min(1.0, change_level * 8), intensity=change_level, ts_ms=ts_ms,
                ))
                self._baseline_frame = gray.copy()
                self._baseline_age = 0

        # 视觉注意力评分
        novelty = result["change_level"] + result["motion_level"]
        surprise = abs(result["brightness"] - 0.5) + result["contrast"]
        result["visual_attention_score"] = min(1.0, novelty * 3 + surprise * 0.5)

        self._prev_frame = gray.copy()
        return result

    def _find_hotspots(self, motion_mask: np.ndarray, shape: Tuple[int, int]) -> List[Dict]:
        """简易连通区域查找运动热点"""
        h, w = shape
        # 降采样找热区
        block_h, block_w = max(1, h // 16), max(1, w // 16)
        hotspots = []
        for by in range(0, h, block_h):
            for bx in range(0, w, block_w):
                block = motion_mask[by:by + block_h, bx:bx + block_w]
                activity = float(np.mean(block))
                if activity > 0.05:
                    hotspots.append({
                        "y": (by + block_h // 2) / h,
                        "x": (bx + block_w // 2) / w,
                        "activity": activity,
                    })
        hotspots.sort(key=lambda x: x["activity"], reverse=True)
        return hotspots

    def _dominant_color(self, frame: np.ndarray) -> str:
        if frame.ndim < 3:
            return "gray"
        h, w = frame.shape[:2]
        # 采样中心区域
        cy, cx = h // 2, w // 2
        patch = frame[max(0, cy - 20):cy + 20, max(0, cx - 20):cx + 20]
        if patch.size == 0:
            return "unknown"
        mean_rgb = np.mean(patch, axis=(0, 1))
        r, g, b = mean_rgb
        if r > g + 30 and r > b + 30: return "warm_red"
        if g > r + 20 and g > b + 20: return "green"
        if b > r + 20 and b > g + 20: return "cool_blue"
        if r > 180 and g > 180 and b > 180: return "bright_white"
        if r < 50 and g < 50 and b < 50: return "dark"
        return "neutral"

    def _make_event(self, source: str, etype: str, desc: str, confidence: float, intensity: float, ts_ms: int, **kw) -> PerceptEvent:
        return PerceptEvent(
            source=source, event_type=etype, description=desc,
            confidence=confidence, intensity=intensity, ts_ms=ts_ms, **kw,
        )

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "brightness": 0, "contrast": 0, "motion_detected": False,
            "motion_level": 0, "motion_hotspots": [], "change_detected": False,
            "change_level": 0, "scene_stable": True, "dominant_color": "unknown",
            "visual_attention_score": 0, "events": [],
        }

    def motion_trend(self, window_sec: float = 5.0) -> Dict[str, float]:
        if not self._motion_history:
            return {"trend": "stable", "mean": 0, "recent": 0}
        now_ms = int(time.time() * 1000)
        recent = [m for t, m in self._motion_history if now_ms - t < window_sec * 1000]
        if not recent:
            return {"trend": "stable", "mean": 0, "recent": 0}
        mean_all = float(np.mean([m for _, m in self._motion_history]))
        mean_recent = float(np.mean(recent))
        if mean_recent > mean_all * 1.5:
            trend = "increasing"
        elif mean_recent < mean_all * 0.5:
            trend = "decreasing"
        else:
            trend = "stable"
        return {"trend": trend, "mean": mean_all, "recent": mean_recent}


# ═══════════════════════════════════════════════════════════════════════════════
# 音频分析器
# ═══════════════════════════════════════════════════════════════════════════════

class AudioAnalyzer:
    """环境声音分析：音量监控、声音分类(语音/噪声/警报/静默)、异常检测"""

    SOUND_CATEGORIES = ["silence", "speech", "noise", "alarm", "music", "impact", "nature", "mechanical"]

    def __init__(self, sr: int = 16000) -> None:
        self.sr = sr
        self._level_history: deque = deque(maxlen=200)
        self._baseline_level: float = 0.001
        self._baseline_samples: int = 0
        self._recent_events: deque = deque(maxlen=100)

    def analyze(self, audio: np.ndarray, ts_ms: int) -> Dict[str, Any]:
        """分析音频块，返回声音洞察"""
        if audio is None or len(audio) == 0:
            return self._empty_result()

        audio = np.asarray(audio, dtype=np.float32)
        rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-6)
        peak = float(np.max(np.abs(audio)))
        zcr = self._zero_crossing_rate(audio)

        self._level_history.append((ts_ms, rms, zcr))

        # 更新基线（自适应环境噪声）
        if self._baseline_samples < 200:
            self._baseline_level = (self._baseline_level * self._baseline_samples + rms) / (self._baseline_samples + 1)
            self._baseline_samples += 1
        else:
            self._baseline_level = self._baseline_level * 0.995 + rms * 0.005

        rel_level = rms / (self._baseline_level + 1e-6)
        category = self._classify_sound(rms, zcr, peak, rel_level)
        is_anomaly = rel_level > 5.0 or (category in ("alarm", "impact") and rms > 0.05)

        result = {
            "rms_level": rms,
            "peak_level": peak,
            "relative_level": rel_level,
            "zero_crossing_rate": zcr,
            "category": category,
            "is_silence": rms < self._baseline_level * 1.5,
            "is_anomaly": is_anomaly,
            "attention_score": min(1.0, (rel_level - 1) * 0.3 + (1.0 if is_anomaly else 0) * 0.5),
            "events": [],
        }

        if is_anomaly:
            result["events"].append(PerceptEvent(
                source="audio", event_type="anomaly",
                description=f"异常声音检测: {category} (相对音量{rel_level:.1f}x)",
                confidence=min(1.0, rel_level / 10), intensity=min(1.0, rms * 20),
                ts_ms=ts_ms,
                metadata={"category": category, "rms": rms, "relative_level": rel_level},
            ))
        elif category in ("speech", "alarm"):
            result["events"].append(PerceptEvent(
                source="audio", event_type=category,
                description=f"检测到{'语音' if category == 'speech' else '警报声'}",
                confidence=0.7 if category == "speech" else 0.9,
                intensity=min(1.0, rms * 20), ts_ms=ts_ms,
            ))

        self._recent_events.extend(result["events"])
        return result

    def _classify_sound(self, rms: float, zcr: float, peak: float, rel_level: float) -> str:
        if rms < self._baseline_level * 1.3:
            return "silence"
        if zcr > 0.3 and rel_level > 3:
            return "alarm"
        if 0.1 < zcr < 0.35 and rel_level > 2:
            return "speech"
        if peak > rms * 8:
            return "impact"
        if 0.05 < zcr < 0.2 and rel_level < 3:
            return "music"
        if zcr < 0.15 and rel_level > 4:
            return "mechanical"
        if zcr < 0.05 and rel_level < 2:
            return "nature"
        return "noise"

    def _zero_crossing_rate(self, signal: np.ndarray) -> float:
        if len(signal) < 2:
            return 0.0
        crossings = np.sum(np.abs(np.diff(np.sign(signal)))) > 0
        return float(np.mean(np.abs(np.diff(np.signbit(signal)))))

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "rms_level": 0, "peak_level": 0, "relative_level": 0,
            "zero_crossing_rate": 0, "category": "silence",
            "is_silence": True, "is_anomaly": False, "attention_score": 0, "events": [],
        }

    def sound_level_trend(self, window_sec: float = 10.0) -> Dict[str, float]:
        if not self._level_history:
            return {"trend": "stable", "mean_db": -100, "recent_db": -100}
        now_ms = int(time.time() * 1000)
        recent = [r for t, r, _ in self._level_history if now_ms - t < window_sec * 1000]
        all_levels = [r for _, r, _ in self._level_history]
        if not recent:
            return {"trend": "stable", "mean_db": self._to_db(np.mean(all_levels)), "recent_db": -100}
        mean_all = np.mean(all_levels)
        mean_recent = np.mean(recent)
        if mean_recent > mean_all * 1.8:
            trend = "rising"
        elif mean_recent < mean_all * 0.4:
            trend = "falling"
        else:
            trend = "stable"
        return {"trend": trend, "mean_db": self._to_db(mean_all), "recent_db": self._to_db(mean_recent)}

    @staticmethod
    def _to_db(linear: float) -> float:
        return float(20 * math.log10(max(linear, 1e-10)))


# ═══════════════════════════════════════════════════════════════════════════════
# 情境模型 — AI的「当前理解」
# ═══════════════════════════════════════════════════════════════════════════════

class SituationModel:
    """维护AI对「现在正在发生什么」的持续理解"""

    def __init__(self) -> None:
        self._current = SituationSnapshot()
        self._history: deque = deque(maxlen=200)
        self._objects_seen: Dict[str, float] = {}      # object → last_seen_ts
        self._sounds_heard: Dict[str, float] = {}
        self._attention_score: float = 0.0
        self._threat_history: deque = deque(maxlen=50)
        self._context_buffer: List[str] = []

    def update(self, visual: Dict, audio: Dict, ts_ms: int) -> SituationSnapshot:
        """整合视觉+听觉，更新情境理解"""
        snap = SituationSnapshot(ts_ms=ts_ms)

        # 运动/声音水平
        snap.motion_level = visual.get("motion_level", 0)
        snap.sound_level = audio.get("relative_level", 0) - 1.0
        snap.sound_level = max(0, snap.sound_level)

        # 检测到的物体和声音
        if visual.get("motion_detected"):
            snap.detected_objects.append("moving_entity")
        if visual.get("change_detected"):
            snap.detected_objects.append("scene_change")
        if audio.get("category") not in ("silence", "noise"):
            snap.detected_sounds.append(audio.get("category", ""))

        # 人类存在推断
        if audio.get("category") == "speech" or (visual.get("motion_level", 0) > 0.06 and audio.get("category") == "speech"):
            snap.human_present = True

        # 场景变化
        snap.scene_changed = visual.get("change_detected", False)

        # 收集事件
        snap.events = []
        for evt in visual.get("events", []):
            snap.events.append(evt)
        for evt in audio.get("events", []):
            snap.events.append(evt)

        # 注意力评分
        self._attention_score = (
            visual.get("visual_attention_score", 0) * 0.6 +
            audio.get("attention_score", 0) * 0.4
        )
        snap.attention_focus = self._determine_focus(snap)

        # 威胁评估
        self._threat_history.append(self._attention_score)
        snap.threat_level = self._assess_threat(snap)

        # 情绪推测
        snap.mood_estimate = self._estimate_mood(snap)

        # 情境摘要
        snap.summary = self._generate_summary(snap)

        # 关联上下文记忆
        snap.contextual_memory = list(self._context_buffer[-5:])

        # 保存历史
        self._history.append(snap)
        self._current = snap

        # 更新对象/声音跟踪
        for obj in snap.detected_objects:
            self._objects_seen[obj] = ts_ms
        for snd in snap.detected_sounds:
            self._sounds_heard[snd] = ts_ms

        return snap

    def _determine_focus(self, snap: SituationSnapshot) -> str:
        focuses = []
        if snap.motion_level > 0.05:
            focuses.append("运动检测")
        if snap.sound_level > 0.5:
            focuses.append("异常声音")
        if snap.human_present:
            focuses.append("人类活动")
        if snap.scene_changed:
            focuses.append("环境变化")
        if not focuses:
            if self._attention_score < 0.1:
                return "环境平静，无需关注"
            return "常规监控"
        return "、".join(focuses)

    def _assess_threat(self, snap: SituationSnapshot) -> float:
        threat = 0.0
        # 运动+异常声音=高威胁
        if snap.motion_level > 0.08 and snap.sound_level > 0.6:
            threat += 0.4
        # 警报声
        if "alarm" in snap.detected_sounds:
            threat += 0.5
        # 撞击声
        if "impact" in snap.detected_sounds:
            threat += 0.3
        # 持续上升的注意力
        if len(self._threat_history) >= 10:
            recent = list(self._threat_history)[-10:]
            if np.mean(recent[-5:]) > np.mean(recent[:5]) * 1.5:
                threat += 0.2
        # 人类+异常
        if snap.human_present and self._attention_score > 0.5:
            threat += 0.3
        return min(1.0, threat)

    def _estimate_mood(self, snap: SituationSnapshot) -> str:
        if snap.threat_level > 0.6:
            return "alert"
        if snap.threat_level > 0.3:
            return "tense"
        if snap.motion_level < 0.02 and snap.sound_level < 0.2:
            return "relaxed"
        if snap.human_present and snap.threat_level < 0.2:
            return "engaged"
        return "neutral"

    def _generate_summary(self, snap: SituationSnapshot) -> str:
        parts = []
        if snap.human_present:
            parts.append("附近有人类活动")
        if snap.motion_level > 0.05:
            parts.append(f"检测到运动(强度{snap.motion_level:.2f})")
        if snap.detected_sounds:
            sounds_str = "、".join(snap.detected_sounds)
            parts.append(f"听到{sounds_str}")
        if snap.scene_changed:
            parts.append("场景发生了变化")
        if not parts:
            return "环境平静，无显著事件"
        return "；".join(parts)

    @property
    def current(self) -> SituationSnapshot:
        return self._current

    def recent_snapshots(self, n: int = 10) -> List[SituationSnapshot]:
        return list(self._history)[-n:]

    def has_significant_change(self, threshold: float = 0.3) -> bool:
        if len(self._history) < 3:
            return False
        recent = list(self._history)[-3:]
        scores = [s.motion_level + s.sound_level for s in recent]
        return max(scores) - min(scores) > threshold


# ═══════════════════════════════════════════════════════════════════════════════
# 持续感知引擎
# ═══════════════════════════════════════════════════════════════════════════════

class PerceptionEngine:
    """持续感知引擎：驱动「看→听→分析→理解」循环"""

    def __init__(self, camera=None, microphone=None, interval_ms: int = 100) -> None:
        self.visual = VisualAnalyzer()
        self.audio = AudioAnalyzer()
        self.situation = SituationModel()

        # 硬件抽象
        self._camera = camera
        self._microphone = microphone

        self.interval_ms = interval_ms
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # 统计
        self._frames_processed: int = 0
        self._chunks_processed: int = 0
        self._events_generated: int = 0
        self._start_ms: int = 0

        # 回调
        self._on_event: Optional[Callable[[PerceptEvent], None]] = None
        self._on_snapshot: Optional[Callable[[SituationSnapshot], None]] = None
        self._on_alert: Optional[Callable[[str, float], None]] = None  # (message, threat_level)

        # 最近的事件日志
        self._event_log: deque = deque(maxlen=500)

    def set_camera(self, camera) -> "PerceptionEngine":
        self._camera = camera
        return self

    def set_microphone(self, mic) -> "PerceptionEngine":
        self._microphone = mic
        return self

    def on_event(self, fn: Callable[[PerceptEvent], None]) -> "PerceptionEngine":
        self._on_event = fn
        return self

    def on_snapshot(self, fn: Callable[[SituationSnapshot], None]) -> "PerceptionEngine":
        self._on_snapshot = fn
        return self

    def on_alert(self, fn: Callable[[str, float], None]) -> "PerceptionEngine":
        self._on_alert = fn
        return self

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_ms = int(time.time() * 1000)
        self._thread = threading.Thread(target=self._loop, daemon=True, name="lifers-perception")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while self._running:
            tick_start = time.perf_counter()
            ts_ms = int(time.time() * 1000)

            # ── 采集视觉 ──
            frame = None
            if self._camera:
                try:
                    frame = self._camera.capture()
                except Exception:
                    pass
            visual_result = self.visual.analyze(frame, ts_ms)

            # ── 采集听觉 ──
            audio_chunk = None
            if self._microphone:
                try:
                    audio_chunk = self._microphone.read_chunk()
                except Exception:
                    pass
            audio_result = self.audio.analyze(audio_chunk, ts_ms)

            # ── 更新情境理解 ──
            snap = self.situation.update(visual_result, audio_result, ts_ms)

            # ── 事件处理 ──
            for evt in snap.events:
                self._events_generated += 1
                evt.id = f"percept_{ts_ms}_{self._events_generated}"
                self._event_log.append(evt)
                if self._on_event:
                    try:
                        self._on_event(evt)
                    except Exception:
                        pass

            # ── 快照回调 ──
            if self._on_snapshot:
                try:
                    self._on_snapshot(snap)
                except Exception:
                    pass

            # ── 威胁告警 ──
            if snap.threat_level > 0.5 and self._on_alert:
                try:
                    self._on_alert(snap.summary, snap.threat_level)
                except Exception:
                    pass

            self._frames_processed += 1
            self._chunks_processed += 1

            # ── 维持帧率 ──
            elapsed = (time.perf_counter() - tick_start) * 1000
            sleep_ms = max(1, self.interval_ms - elapsed)
            time.sleep(sleep_ms / 1000)

    def tick_once(self) -> SituationSnapshot:
        """单次同步 tick（不启动线程）"""
        ts_ms = int(time.time() * 1000)
        frame = self._camera.capture() if self._camera else None
        audio_chunk = self._microphone.read_chunk() if self._microphone else None
        visual_result = self.visual.analyze(frame, ts_ms)
        audio_result = self.audio.analyze(audio_chunk, ts_ms)
        snap = self.situation.update(visual_result, audio_result, ts_ms)
        self._frames_processed += 1
        self._chunks_processed += 1
        # Process events from this tick
        for evt in snap.events:
            self._events_generated += 1
            evt.id = f"percept_{ts_ms}_{self._events_generated}"
            self._event_log.append(evt)
            if self._on_event:
                try:
                    self._on_event(evt)
                except Exception:
                    pass
        if self._on_snapshot:
            try:
                self._on_snapshot(snap)
            except Exception:
                pass
        if snap.threat_level > 0.5 and self._on_alert:
            try:
                self._on_alert(snap.summary, snap.threat_level)
            except Exception:
                pass
        return snap

    def status(self) -> Dict[str, Any]:
        uptime = (int(time.time() * 1000) - self._start_ms) / 1000 if self._start_ms else 0
        return {
            "running": self._running,
            "uptime_sec": uptime,
            "frames": self._frames_processed,
            "chunks": self._chunks_processed,
            "events": self._events_generated,
            "current_situation": {
                "summary": self.situation.current.summary,
                "motion": self.situation.current.motion_level,
                "sound": self.situation.current.sound_level,
                "human_present": self.situation.current.human_present,
                "threat": self.situation.current.threat_level,
                "mood": self.situation.current.mood_estimate,
                "focus": self.situation.current.attention_focus,
            },
            "motion_trend": self.visual.motion_trend(),
            "sound_trend": self.audio.sound_level_trend(),
        }

    def recent_events(self, n: int = 20) -> List[PerceptEvent]:
        return list(self._event_log)[-n:]

    def recent_snapshots(self, n: int = 10) -> List[SituationSnapshot]:
        return self.situation.recent_snapshots(n)


# ═══════════════════════════════════════════════════════════════════════════════
# 简易相机/麦克风模拟器（自给自足）
# ═══════════════════════════════════════════════════════════════════════════════

class SimCamera:
    """模拟相机：生成带噪声和伪运动的图像"""
    def __init__(self, width: int = 320, height: int = 240) -> None:
        self.width = width
        self.height = height
        self._rng = np.random.RandomState()
        self._frame_count = 0
        self._bg = self._rng.randint(0, 100, (height, width, 3)).astype(np.uint8)

    def capture(self) -> np.ndarray:
        self._frame_count += 1
        frame = self._bg.copy().astype(np.float32)
        # 添加噪声
        noise = self._rng.randn(self.height, self.width, 3) * 15
        frame += noise
        # 模拟偶尔的运动（移动物体）
        if self._frame_count % 30 < 5:  # 每30帧有5帧有运动
            x = int(self._rng.uniform(0, self.width - 40))
            y = int(self._rng.uniform(0, self.height - 40))
            frame[y:y + 30, x:x + 30] += self._rng.randint(30, 80)
        # 模拟光照变化
        brightness_variation = math.sin(self._frame_count / 100) * 20
        frame += brightness_variation
        return np.clip(frame, 0, 255).astype(np.uint8)


class SimMicrophone:
    """模拟麦克风：生成带环境噪声和伪事件的音频"""
    def __init__(self, sr: int = 16000, chunk_ms: int = 100) -> None:
        self.sr = sr
        self.chunk_size = sr * chunk_ms // 1000
        self._rng = np.random.RandomState()
        self._chunk_count = 0

    def read_chunk(self) -> np.ndarray:
        self._chunk_count += 1
        # 基础环境噪声
        audio = self._rng.randn(self.chunk_size).astype(np.float32) * 0.002
        # 偶发的响声
        if self._chunk_count % 100 < 3:
            t = np.linspace(0, 1, self.chunk_size)
            audio += np.sin(2 * math.pi * 440 * t) * 0.05 * np.exp(-t * 5)
        # 模拟语音
        if self._chunk_count % 200 < 8:
            t = np.linspace(0, 1, self.chunk_size)
            audio += (np.sin(2 * math.pi * 200 * t) + np.sin(2 * math.pi * 600 * t) * 0.5) * 0.03 * np.exp(-t * 3)
        return audio.astype(np.float32)
