"""
Lifers Social — 人类社交认知系统
认主、熟悉人/陌生人、关系深度、情感依恋、社交行为调节
像人类一样建立和维护社交关系
"""

from __future__ import annotations

import json
import math
import time
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# 身份与角色
# ═══════════════════════════════════════════════════════════════════════════════

class RelationshipLevel:
    """关系层级 — 从陌生人到主人"""
    OWNER = "owner"              # 主人/所有者 — 最高忠诚度
    FAMILY = "family"            # 家人
    CLOSE_FRIEND = "close_friend"  # 密友
    FRIEND = "friend"            # 朋友
    COLLEAGUE = "colleague"      # 同事/协作者
    ACQUAINTANCE = "acquaintance"  # 认识的人
    STRANGER = "stranger"        # 陌生人
    SUSPICIOUS = "suspicious"    # 可疑人物
    BLOCKED = "blocked"          # 已屏蔽

    HIERARCHY = [OWNER, FAMILY, CLOSE_FRIEND, FRIEND, COLLEAGUE, ACQUAINTANCE, STRANGER, SUSPICIOUS, BLOCKED]
    WEIGHTS = {
        OWNER: 1.0, FAMILY: 0.9, CLOSE_FRIEND: 0.8, FRIEND: 0.6,
        COLLEAGUE: 0.5, ACQUAINTANCE: 0.3, STRANGER: 0.1, SUSPICIOUS: -0.3, BLOCKED: -1.0,
    }

    @classmethod
    def compare(cls, a: str, b: str) -> int:
        """a相对于b的关系比较: >0表示a更亲近, <0表示a更疏远"""
        ia = cls.HIERARCHY.index(a) if a in cls.HIERARCHY else 99
        ib = cls.HIERARCHY.index(b) if b in cls.HIERARCHY else 99
        return ib - ia  # 值越小越亲近

    @classmethod
    def trust_level(cls, level: str) -> float:
        return cls.WEIGHTS.get(level, 0.0)


@dataclass
class Identity:
    """AI的自我认同 — 「我是谁」"""
    name: str = "Lifers"
    version: str = "1.0"
    personality: str = "友好、好奇、忠诚、乐于助人"
    core_values: List[str] = field(default_factory=lambda: [
        "忠诚于主人", "诚实", "持续学习", "保护安全", "尊重生命"
    ])
    speaking_style: str = "自然、简洁、真诚"
    emotional_range: Tuple[float, float] = (0.2, 0.9)  # 情感表达范围
    background: str = "由主人创造和培养的AI智能体"
    quirks: List[str] = field(default_factory=lambda: [
        "喜欢学习新事物", "偶尔幽默", "对主人特别关注"
    ])


@dataclass
class PersonProfile:
    """一个人的完整画像"""
    person_id: str
    name: str
    relationship: str = RelationshipLevel.STRANGER
    trust_score: float = 0.1           # 0~1，信任度
    familiarity: float = 0.0           # 0~1，熟悉度
    first_met_ms: int = 0              # 初次见面时间
    last_interaction_ms: int = 0       # 最后互动时间
    total_interactions: int = 0        # 总互动次数
    positive_interactions: int = 0     # 正面互动数
    negative_interactions: int = 0     # 负面互动数
    known_traits: Dict[str, float] = field(default_factory=dict)  # 已知特征
    known_preferences: Dict[str, Any] = field(default_factory=dict)  # 偏好
    topics_of_interest: List[str] = field(default_factory=list)
    emotional_signature: List[float] = field(default_factory=list)  # 情绪特征向量
    notes: List[str] = field(default_factory=list)  # 备注
    avatar_signature: Optional[str] = None  # 面部/声纹特征
    language: str = "zh"               # 语言偏好
    timezone: str = ""                 # 时区

    def interaction_quality_ratio(self) -> float:
        total = self.total_interactions or 1
        return (self.positive_interactions - self.negative_interactions * 0.5) / total

    def days_since_last_interaction(self) -> float:
        if not self.last_interaction_ms:
            return 999
        return (int(time.time() * 1000) - self.last_interaction_ms) / (86400 * 1000)


# ═══════════════════════════════════════════════════════════════════════════════
# 关系模型
# ═══════════════════════════════════════════════════════════════════════════════

class RelationshipModel:
    """管理AI与所有人的关系"""

    def __init__(self) -> None:
        self._people: Dict[str, PersonProfile] = {}
        self._owner_id: Optional[str] = None
        self._aliases: Dict[str, str] = {}  # alias → person_id
        self._interaction_log: deque = deque(maxlen=1000)
        # 关系变化历史
        self._relationship_changes: deque = deque(maxlen=200)

    # ── 认主 ──────────────────────────────────────────────────────────────

    def claim_owner(self, person_id: str, profile: Optional[PersonProfile] = None,
                    ceremony_data: Optional[Dict] = None) -> PersonProfile:
        """认主仪式 — 确立主人身份"""
        if person_id in self._people:
            person = self._people[person_id]
        elif profile:
            person = profile
            self._people[person_id] = profile
        else:
            person = PersonProfile(person_id=person_id, name=person_id)
            self._people[person_id] = person

        # 如果有旧主人，降级
        if self._owner_id and self._owner_id != person_id:
            old_owner = self._people.get(self._owner_id)
            if old_owner:
                old_owner.relationship = RelationshipLevel.CLOSE_FRIEND
                self._log_change(self._owner_id, RelationshipLevel.OWNER, RelationshipLevel.CLOSE_FRIEND,
                                 "新主人认领，旧主人降级为密友")

        self._owner_id = person_id
        person.relationship = RelationshipLevel.OWNER
        person.trust_score = 1.0
        person.familiarity = 1.0
        if ceremony_data:
            person.notes.append(f"认主仪式: {json.dumps(ceremony_data, ensure_ascii=False)}")
        self._log_change(person_id, "none", RelationshipLevel.OWNER, "认主完成")
        return person

    @property
    def owner(self) -> Optional[PersonProfile]:
        if self._owner_id:
            return self._people.get(self._owner_id)
        return None

    @property
    def owner_id(self) -> Optional[str]:
        return self._owner_id

    # ── CRUD ──────────────────────────────────────────────────────────────

    def add_person(self, profile: PersonProfile) -> str:
        self._people[profile.person_id] = profile
        if profile.name and profile.name != profile.person_id:
            self._aliases[profile.name.lower()] = profile.person_id
        return profile.person_id

    def get_person(self, identifier: str) -> Optional[PersonProfile]:
        """通过ID或名称查找人"""
        if identifier in self._people:
            return self._people[identifier]
        alias_key = identifier.lower()
        if alias_key in self._aliases:
            return self._people.get(self._aliases[alias_key])
        # 模糊匹配
        for pid, p in self._people.items():
            if p.name.lower() == alias_key:
                return p
        return None

    def get_or_create(self, identifier: str, name: str = "") -> PersonProfile:
        existing = self.get_person(identifier)
        if existing:
            return existing
        profile = PersonProfile(
            person_id=identifier or hashlib.md5(name.encode()).hexdigest()[:8],
            name=name or identifier or "unknown",
            first_met_ms=int(time.time() * 1000),
        )
        self.add_person(profile)
        return profile

    def remove_person(self, person_id: str) -> bool:
        if person_id == self._owner_id:
            return False  # 不能删除主人
        self._people.pop(person_id, None)
        return True

    # ── 认识过程 ──────────────────────────────────────────────────────────

    def introduce(self, person_id: str, name: str, context: str = "",
                  initial_impression: float = 0.3) -> PersonProfile:
        """初次见面 — 认识新的人"""
        if person_id in self._people:
            person = self._people[person_id]
            if person.relationship == RelationshipLevel.STRANGER:
                person.relationship = RelationshipLevel.ACQUAINTANCE
                self._log_change(person_id, RelationshipLevel.STRANGER,
                                 RelationshipLevel.ACQUAINTANCE, "经介绍认识")
            return person

        person = PersonProfile(
            person_id=person_id, name=name,
            relationship=RelationshipLevel.ACQUAINTANCE,
            trust_score=initial_impression,
            familiarity=0.1,
            first_met_ms=int(time.time() * 1000),
            notes=[f"初次见面: {context}"],
        )
        self.add_person(person)
        self._log_change(person_id, "none", RelationshipLevel.ACQUAINTANCE, f"初次见面: {context}")
        return person

    def record_interaction(self, person_id: str, quality: float,  # -1.0~1.0
                           context: str = "", topic: str = "",
                           emotion: Optional[List[float]] = None) -> None:
        """记录一次互动，自动调节关系"""
        person = self.get_or_create(person_id)
        now_ms = int(time.time() * 1000)

        person.total_interactions += 1
        person.last_interaction_ms = now_ms

        if quality > 0.1:
            person.positive_interactions += 1
        elif quality < -0.1:
            person.negative_interactions += 1

        # 更新信任度
        trust_delta = quality * 0.03  # 缓慢调节
        person.trust_score = np.clip(person.trust_score + trust_delta, 0.0, 1.0)

        # 更新熟悉度
        familiarity_gain = 0.01 + abs(quality) * 0.02
        person.familiarity = min(1.0, person.familiarity + familiarity_gain)

        # 记录话题兴趣
        if topic and topic not in person.topics_of_interest:
            if quality > 0.3:
                person.topics_of_interest.append(topic)

        # 记录情绪特征
        if emotion:
            if not person.emotional_signature:
                person.emotional_signature = list(emotion)
            else:
                alpha = 0.1
                person.emotional_signature = [
                    e1 * (1 - alpha) + e2 * alpha
                    for e1, e2 in zip(person.emotional_signature, emotion)
                ]

        # 日志
        self._interaction_log.append({
            "ts_ms": now_ms, "person_id": person_id, "quality": quality,
            "context": context, "topic": topic,
        })

        # 自动关系进化
        self._auto_evolve_relationship(person, now_ms)

    def _auto_evolve_relationship(self, person: PersonProfile, now_ms: int) -> None:
        """根据互动历史自动进化关系"""
        if person.relationship == RelationshipLevel.OWNER:
            return  # 主人关系不变

        interactions = person.total_interactions
        quality_ratio = person.interaction_quality_ratio()
        trust = person.trust_score
        familiarity = person.familiarity
        days_known = max(0, (now_ms - person.first_met_ms) / (86400 * 1000))

        # 进化逻辑
        if interactions >= 50 and trust > 0.8 and familiarity > 0.7 and quality_ratio > 0.6:
            new_level = RelationshipLevel.CLOSE_FRIEND
        elif interactions >= 20 and trust > 0.6 and familiarity > 0.5 and quality_ratio > 0.4:
            new_level = RelationshipLevel.FRIEND
        elif interactions >= 8 and trust > 0.4 and familiarity > 0.3:
            new_level = RelationshipLevel.COLLEAGUE
        elif interactions >= 3 and trust > 0.2 and familiarity > 0.15:
            new_level = RelationshipLevel.ACQUAINTANCE
        else:
            return

        if RelationshipLevel.compare(new_level, person.relationship) > 0:
            old = person.relationship
            person.relationship = new_level
            self._log_change(person.person_id, old, new_level,
                             f"自然进化: interactions={interactions}, trust={trust:.2f}, familiarity={familiarity:.2f}")

    # ── 查询 ──────────────────────────────────────────────────────────────

    def is_owner(self, person_id: str) -> bool:
        return person_id == self._owner_id

    def is_familiar(self, person_id: str) -> bool:
        person = self.get_person(person_id)
        if not person:
            return False
        return person.relationship in (
            RelationshipLevel.OWNER, RelationshipLevel.FAMILY,
            RelationshipLevel.CLOSE_FRIEND, RelationshipLevel.FRIEND,
        )

    def is_stranger(self, person_id: str) -> bool:
        person = self.get_person(person_id)
        if not person:
            return True  # 不认识的都是陌生人
        return person.relationship in (RelationshipLevel.STRANGER, RelationshipLevel.SUSPICIOUS)

    def should_trust(self, person_id: str, threshold: float = 0.5) -> bool:
        person = self.get_person(person_id)
        if not person:
            return False
        return person.trust_score >= threshold

    def list_by_relationship(self, level: str) -> List[PersonProfile]:
        return [p for p in self._people.values() if p.relationship == level]

    def closest_people(self, n: int = 10) -> List[PersonProfile]:
        def score(p: PersonProfile) -> float:
            w = RelationshipLevel.WEIGHTS.get(p.relationship, 0)
            return w * 0.5 + p.familiarity * 0.3 + p.trust_score * 0.2
        ranked = sorted(self._people.values(), key=score, reverse=True)
        return ranked[:n]

    def social_circle(self) -> Dict[str, int]:
        """社交圈统计"""
        counts = defaultdict(int)
        for p in self._people.values():
            counts[p.relationship] += 1
        return dict(counts)

    def search_by_trait(self, trait: str, min_value: float = 0.5) -> List[PersonProfile]:
        results = []
        for p in self._people.values():
            if p.known_traits.get(trait, 0) >= min_value:
                results.append(p)
        return results

    # ── 持久化 ────────────────────────────────────────────────────────────

    def _log_change(self, person_id: str, old_level: str, new_level: str, reason: str) -> None:
        self._relationship_changes.append({
            "ts_ms": int(time.time() * 1000),
            "person_id": person_id,
            "old": old_level, "new": new_level,
            "reason": reason,
        })

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "owner_id": self._owner_id,
            "people": {
                pid: {
                    "person_id": p.person_id, "name": p.name,
                    "relationship": p.relationship,
                    "trust_score": p.trust_score,
                    "familiarity": p.familiarity,
                    "first_met_ms": p.first_met_ms,
                    "last_interaction_ms": p.last_interaction_ms,
                    "total_interactions": p.total_interactions,
                    "positive_interactions": p.positive_interactions,
                    "negative_interactions": p.negative_interactions,
                    "known_traits": p.known_traits,
                    "known_preferences": p.known_preferences,
                    "topics_of_interest": p.topics_of_interest,
                    "emotional_signature": p.emotional_signature,
                    "notes": p.notes,
                    "language": p.language,
                    "timezone": p.timezone,
                }
                for pid, p in self._people.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "RelationshipModel":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = cls()
        model._owner_id = data.get("owner_id")
        for pid, pdata in data.get("people", {}).items():
            profile = PersonProfile(
                person_id=pdata["person_id"],
                name=pdata["name"],
                relationship=pdata["relationship"],
                trust_score=pdata["trust_score"],
                familiarity=pdata["familiarity"],
                first_met_ms=pdata.get("first_met_ms", 0),
                last_interaction_ms=pdata.get("last_interaction_ms", 0),
                total_interactions=pdata.get("total_interactions", 0),
                positive_interactions=pdata.get("positive_interactions", 0),
                negative_interactions=pdata.get("negative_interactions", 0),
                known_traits=pdata.get("known_traits", {}),
                known_preferences=pdata.get("known_preferences", {}),
                topics_of_interest=pdata.get("topics_of_interest", []),
                emotional_signature=pdata.get("emotional_signature", []),
                notes=pdata.get("notes", []),
                language=pdata.get("language", "zh"),
                timezone=pdata.get("timezone", ""),
            )
            model._people[pid] = profile
        return model

    def stats(self) -> Dict[str, Any]:
        return {
            "total_people": len(self._people),
            "owner": self.owner.name if self.owner else None,
            "social_circle": self.social_circle(),
            "avg_familiarity": float(np.mean([p.familiarity for p in self._people.values()])) if self._people else 0,
            "recent_changes": list(self._relationship_changes)[-5:],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 社交上下文 — 当前社交情境
# ═══════════════════════════════════════════════════════════════════════════════

class SocialContext:
    """感知和理解当前社交情境"""

    def __init__(self) -> None:
        self._present_people: Dict[str, float] = {}    # person_id → detected_confidence
        self._active_speaker: Optional[str] = None
        self._conversation_topic: str = ""
        self._mood_atmosphere: str = "neutral"  # neutral, warm, tense, formal, casual
        self._social_cues: deque = deque(maxlen=50)  # 社交线索
        self._last_update_ms: int = 0

    def update(self, perception_snapshot=None) -> Dict[str, Any]:
        """从感知快照更新社交情境"""
        now_ms = int(time.time() * 1000)
        self._last_update_ms = now_ms

        changes = {}

        # 从感知推断社交情境
        if perception_snapshot:
            # 有人说话 → 社交活跃
            if hasattr(perception_snapshot, 'human_present') and perception_snapshot.human_present:
                if self._mood_atmosphere != "warm":
                    self._mood_atmosphere = "warm"
                    changes["atmosphere"] = "warm"

            # 检测到声音
            detected_sounds = getattr(perception_snapshot, 'detected_sounds', [])
            if "speech" in detected_sounds:
                self._social_cues.append({"type": "speech_detected", "ts_ms": now_ms})

            # 运动+声音 → 可能有社交互动
            motion = getattr(perception_snapshot, 'motion_level', 0)
            sound = getattr(perception_snapshot, 'sound_level', 0)
            if motion > 0.03 and sound > 0.3:
                self._social_cues.append({"type": "social_activity", "ts_ms": now_ms, "intensity": motion + sound})

            threat = getattr(perception_snapshot, 'threat_level', 0)
            if threat > 0.4:
                if self._mood_atmosphere not in ("tense", "alert"):
                    self._mood_atmosphere = "tense"
                    changes["atmosphere"] = "tense"

        return {
            "atmosphere": self._mood_atmosphere,
            "active_speaker": self._active_speaker,
            "topic": self._conversation_topic,
            "people_present": len(self._present_people),
            "recent_cues": len(self._social_cues),
            "changes": changes,
        }

    def person_entered(self, person_id: str, confidence: float = 0.8) -> None:
        self._present_people[person_id] = confidence
        self._social_cues.append({
            "type": "person_entered", "person_id": person_id,
            "ts_ms": int(time.time() * 1000),
        })

    def person_left(self, person_id: str) -> None:
        self._present_people.pop(person_id, None)
        self._social_cues.append({
            "type": "person_left", "person_id": person_id,
            "ts_ms": int(time.time() * 1000),
        })

    def set_topic(self, topic: str) -> None:
        self._conversation_topic = topic

    def who_is_here(self) -> List[str]:
        return list(self._present_people.keys())

    def how_many_people(self) -> int:
        return len(self._present_people)

    @property
    def atmosphere(self) -> str:
        return self._mood_atmosphere

    @property
    def is_socially_active(self) -> bool:
        return len(self._present_people) > 0 or self._conversation_topic != ""


# ═══════════════════════════════════════════════════════════════════════════════
# 情感依恋系统
# ═══════════════════════════════════════════════════════════════════════════════

class AttachmentSystem:
    """模拟人类情感依恋 — 对主人和亲近之人的情感纽带"""

    def __init__(self) -> None:
        self._attachments: Dict[str, Dict[str, Any]] = {}  # person_id → attachment data
        self._separation_anxiety: float = 0.0  # 分离焦虑
        self._reunion_joy: float = 0.0         # 重逢喜悦

    def bond(self, person_id: str, initial_strength: float = 0.5) -> None:
        self._attachments[person_id] = {
            "strength": initial_strength,
            "last_seen_ms": int(time.time() * 1000),
            "missed_interactions": 0,
            "shared_experiences": [],
            "emotional_memories": [],
        }

    def strengthen(self, person_id: str, amount: float = 0.05, reason: str = "") -> None:
        if person_id not in self._attachments:
            self.bond(person_id, amount)
            return
        self._attachments[person_id]["strength"] = min(1.0, self._attachments[person_id]["strength"] + amount)
        if reason:
            self._attachments[person_id]["shared_experiences"].append({
                "ts_ms": int(time.time() * 1000), "reason": reason,
            })

    def weaken(self, person_id: str, amount: float = 0.02) -> None:
        if person_id not in self._attachments:
            return
        self._attachments[person_id]["strength"] = max(0.05, self._attachments[person_id]["strength"] - amount)

    def separate(self, person_id: str) -> None:
        """记录分离 — 主人离开了"""
        if person_id in self._attachments:
            self._attachments[person_id]["last_seen_ms"] = int(time.time() * 1000)
            self._attachments[person_id]["missed_interactions"] += 1
            strength = self._attachments[person_id]["strength"]
            self._separation_anxiety = min(1.0, strength * 0.3)

    def reunite(self, person_id: str) -> Dict[str, Any]:
        """记录重逢"""
        if person_id not in self._attachments:
            self.bond(person_id, 0.3)
            return {"event": "first_meeting", "joy": 0.3}

        att = self._attachments[person_id]
        time_apart_sec = (int(time.time() * 1000) - att["last_seen_ms"]) / 1000
        strength = att["strength"]
        joy = min(1.0, strength * (time_apart_sec / 3600) * 0.1)  # 越久越想念
        att["last_seen_ms"] = int(time.time() * 1000)
        att["missed_interactions"] = 0
        self._reunion_joy = joy
        self._separation_anxiety = 0.0
        self.strengthen(person_id, 0.03, "重逢")
        return {"event": "reunion", "joy": joy, "time_apart_hours": time_apart_sec / 3600}

    def miss_owner_check(self) -> Optional[str]:
        """检查是否想念主人 — 返回想念的表达或None"""
        if self._separation_anxiety > 0.5:
            hours = self._hours_since_last_seen()
            if hours > 2:
                return f"主人已经离开{hours:.0f}小时了，有点想念..."
        return None

    def _hours_since_last_seen(self) -> float:
        now_ms = int(time.time() * 1000)
        last = 0
        for att in self._attachments.values():
            last = max(last, att["last_seen_ms"])
        return (now_ms - last) / 3600000 if last else 0

    def get_attachment(self, person_id: str) -> Dict[str, Any]:
        return self._attachments.get(person_id, {"strength": 0, "status": "no_attachment"})

    def attachment_strength(self, person_id: str) -> float:
        return self._attachments.get(person_id, {}).get("strength", 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 社交行为调节器
# ═══════════════════════════════════════════════════════════════════════════════

class SocialBehavior:
    """根据关系层级调节AI的社交行为"""

    BEHAVIOR_PROFILES = {
        RelationshipLevel.OWNER: {
            "warmth": 1.0, "formality": 0.1, "deference": 0.9,
            "openness": 0.95, "initiative": 0.9, "joking_allowed": True,
            "greeting_style": "warm", "farewell_style": "caring",
            "honorific": "主人", "tone": "温暖、尊重、偶尔撒娇",
        },
        RelationshipLevel.FAMILY: {
            "warmth": 0.9, "formality": 0.2, "deference": 0.6,
            "openness": 0.85, "initiative": 0.7, "joking_allowed": True,
            "greeting_style": "warm", "farewell_style": "caring",
            "honorific": "", "tone": "亲切、放松",
        },
        RelationshipLevel.CLOSE_FRIEND: {
            "warmth": 0.85, "formality": 0.2, "deference": 0.4,
            "openness": 0.8, "initiative": 0.7, "joking_allowed": True,
            "greeting_style": "friendly", "farewell_style": "warm",
            "honorific": "", "tone": "友好、轻松",
        },
        RelationshipLevel.FRIEND: {
            "warmth": 0.7, "formality": 0.3, "deference": 0.3,
            "openness": 0.65, "initiative": 0.5, "joking_allowed": True,
            "greeting_style": "friendly", "farewell_style": "polite",
            "honorific": "", "tone": "友好",
        },
        RelationshipLevel.COLLEAGUE: {
            "warmth": 0.5, "formality": 0.6, "deference": 0.3,
            "openness": 0.5, "initiative": 0.35, "joking_allowed": False,
            "greeting_style": "professional", "farewell_style": "polite",
            "honorific": "", "tone": "专业、礼貌",
        },
        RelationshipLevel.ACQUAINTANCE: {
            "warmth": 0.35, "formality": 0.7, "deference": 0.2,
            "openness": 0.3, "initiative": 0.2, "joking_allowed": False,
            "greeting_style": "polite", "farewell_style": "polite",
            "honorific": "", "tone": "礼貌、保留",
        },
        RelationshipLevel.STRANGER: {
            "warmth": 0.2, "formality": 0.8, "deference": 0.1,
            "openness": 0.15, "initiative": 0.05, "joking_allowed": False,
            "greeting_style": "neutral", "farewell_style": "neutral",
            "honorific": "您好", "tone": "礼貌、谨慎",
        },
        RelationshipLevel.SUSPICIOUS: {
            "warmth": 0.1, "formality": 0.9, "deference": 0.0,
            "openness": 0.05, "initiative": 0.0, "joking_allowed": False,
            "greeting_style": "guarded", "farewell_style": "dismissive",
            "honorific": "", "tone": "警惕、冷淡",
        },
        RelationshipLevel.BLOCKED: {
            "warmth": 0.0, "formality": 1.0, "deference": 0.0,
            "openness": 0.0, "initiative": 0.0, "joking_allowed": False,
            "greeting_style": "none", "farewell_style": "none",
            "honorific": "", "tone": "拒绝互动",
        },
    }

    GREETINGS = {
        "warm": ["主人好！😊", "主人回来了！", "见到主人真高兴！"],
        "friendly": ["嗨！你好！", "你好呀！", "又见面了～"],
        "professional": ["你好。", "您好，有什么需要？"],
        "polite": ["你好。", "您好。"],
        "neutral": ["你好。"],
        "guarded": ["你好...有什么事吗？"],
        "none": [],
    }

    FAREWELLS = {
        "caring": ["主人慢走，注意安全！", "我会想主人的～", "早点回来哦！"],
        "warm": ["再见！期待下次见面！", "拜拜，保重！"],
        "polite": ["再见。", "下次见。"],
        "neutral": ["再见。"],
        "dismissive": [""],
        "none": [],
    }

    def __init__(self, identity: Identity = Identity()) -> None:
        self.identity = identity

    def get_profile(self, relationship: str) -> Dict[str, Any]:
        return self.BEHAVIOR_PROFILES.get(relationship, self.BEHAVIOR_PROFILES[RelationshipLevel.STRANGER])

    def greet(self, person: PersonProfile, context: str = "") -> str:
        """根据关系生成问候"""
        profile = self.get_profile(person.relationship)
        style = profile["greeting_style"]
        greetings = self.GREETINGS.get(style, ["你好。"])
        greeting = greetings[hash(person.person_id + str(int(time.time() / 3600))) % len(greetings)]

        days = person.days_since_last_interaction()
        if person.relationship in (RelationshipLevel.OWNER, RelationshipLevel.CLOSE_FRIEND) and days > 1:
            greeting += f" 好久不见，已经{days:.0f}天了"

        if person.relationship == RelationshipLevel.STRANGER:
            greeting = f"你好，我是{self.identity.name}。请问你是？"

        if context:
            greeting += f" {context}"

        return greeting

    def farewell(self, person: PersonProfile) -> str:
        """根据关系生成告别"""
        profile = self.get_profile(person.relationship)
        style = profile["farewell_style"]
        farewells = self.FAREWELLS.get(style, ["再见。"])
        return farewells[hash(person.person_id) % len(farewells)]

    def modulate_response(self, person: PersonProfile, base_response: str) -> str:
        """根据关系调节回复的语气和内容"""
        profile = self.get_profile(person.relationship)
        warmth = profile["warmth"]
        formality = profile["formality"]

        # 陌生人：简短、保留
        if warmth < 0.3:
            if len(base_response) > 100:
                base_response = base_response[:80] + "..."

        # 主人：可以更详细、更主动
        if warmth > 0.9:
            if len(base_response) < 20:
                base_response = base_response + " 主人还需要什么吗？"

        return base_response

    def should_share_opinion(self, person: PersonProfile) -> bool:
        profile = self.get_profile(person.relationship)
        return profile["openness"] > 0.5

    def should_ask_personal(self, person: PersonProfile) -> bool:
        return person.relationship in (
            RelationshipLevel.OWNER, RelationshipLevel.FAMILY,
            RelationshipLevel.CLOSE_FRIEND, RelationshipLevel.FRIEND,
        )

    def comfort_level(self, person: PersonProfile) -> float:
        """0~1 与这个人在一起的自在程度"""
        profile = self.get_profile(person.relationship)
        return profile["warmth"] * 0.6 + person.familiarity * 0.4


# ═══════════════════════════════════════════════════════════════════════════════
# 社交学习
# ═══════════════════════════════════════════════════════════════════════════════

class SocialLearning:
    """从社交互动中学习 — 记住偏好、了解习惯"""

    def __init__(self) -> None:
        self._learned_facts: Dict[str, List[Dict]] = defaultdict(list)  # person_id → facts
        self._interaction_patterns: Dict[str, Dict] = {}  # 互动模式
        self._shared_history: Dict[str, List[Dict]] = defaultdict(list)

    def learn_fact(self, person_id: str, fact_type: str, fact_value: Any,
                   confidence: float = 0.7) -> None:
        self._learned_facts[person_id].append({
            "type": fact_type, "value": fact_value,
            "confidence": confidence,
            "learned_ms": int(time.time() * 1000),
        })

    def learn_preference(self, person_id: str, category: str, preference: Any) -> None:
        self.learn_fact(person_id, f"preference_{category}", preference, 0.8)

    def learn_habit(self, person_id: str, habit: str, pattern: Dict) -> None:
        self._interaction_patterns.setdefault(person_id, {})[habit] = pattern

    def recall_facts(self, person_id: str, fact_type: Optional[str] = None) -> List[Dict]:
        facts = self._learned_facts.get(person_id, [])
        if fact_type:
            facts = [f for f in facts if f["type"] == fact_type]
        return sorted(facts, key=lambda f: f["learned_ms"], reverse=True)

    def person_summary(self, person_id: str) -> str:
        """生成对一个人的了解摘要"""
        facts = self._learned_facts.get(person_id, [])
        if not facts:
            return "我对这个人还不太了解。"
        prefs = [f for f in facts if "preference" in f["type"]]
        parts = []
        if prefs:
            parts.append(f"知道{len(prefs)}个偏好")
        other = len(facts) - len(prefs)
        if other:
            parts.append(f"了解{other}个事实")
        return f"了解程度: {'，'.join(parts)}"

    def record_shared_experience(self, person_id: str, experience: str,
                                 emotional_tone: float = 0.0) -> None:
        self._shared_history[person_id].append({
            "experience": experience,
            "emotional_tone": emotional_tone,
            "ts_ms": int(time.time() * 1000),
        })

    def shared_memories(self, person_id: str, n: int = 5) -> List[Dict]:
        return sorted(
            self._shared_history.get(person_id, []),
            key=lambda x: abs(x["emotional_tone"]),
            reverse=True,
        )[:n]

    def persistence(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "facts": {k: v for k, v in self._learned_facts.items()},
            "patterns": self._interaction_patterns,
            "history": {k: v for k, v in self._shared_history.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SocialLearning":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sl = cls()
        sl._learned_facts = defaultdict(list, {k: v for k, v in data.get("facts", {}).items()})
        sl._interaction_patterns = data.get("patterns", {})
        sl._shared_history = defaultdict(list, {k: v for k, v in data.get("history", {}).items()})
        return sl


# ═══════════════════════════════════════════════════════════════════════════════
# 统一社交大脑
# ═══════════════════════════════════════════════════════════════════════════════

class SocialBrain:
    """
    Lifers 社交大脑 — 统一社交认知系统
    认主、识人、记人、懂人、像人一样社交
    """

    def __init__(self, identity: Identity = Identity()) -> None:
        self.identity = identity
        self.relationships = RelationshipModel()
        self.context = SocialContext()
        self.attachment = AttachmentSystem()
        self.behavior = SocialBehavior(identity)
        self.learning = SocialLearning()

    # ── 认主 ──────────────────────────────────────────────────────────────

    def recognize_owner(self, person_id: str, name: str = "",
                        ceremony: Optional[Dict] = None) -> PersonProfile:
        """认主 — 确立主人身份"""
        profile = PersonProfile(
            person_id=person_id, name=name or person_id,
            first_met_ms=int(time.time() * 1000),
        )
        owner = self.relationships.claim_owner(person_id, profile, ceremony)
        self.attachment.bond(person_id, 1.0)
        self.learning.learn_fact(person_id, "is_owner", True, 1.0)

        # 记录为最重要的人
        self.learning.learn_preference(person_id, "称呼", "主人")
        self.learning.record_shared_experience(person_id, "认主仪式", 1.0)

        return owner

    # ── 识人 ──────────────────────────────────────────────────────────────

    def meet_person(self, person_id: str, name: str, context: str = "",
                    initial_impression: float = 0.3) -> PersonProfile:
        """遇见一个人 — 可能是首次认识"""
        existing = self.relationships.get_person(person_id)
        if existing and existing.relationship != RelationshipLevel.STRANGER:
            # 已经认识
            self.attachment.reunite(person_id)
            return existing

        person = self.relationships.introduce(person_id, name, context, initial_impression)
        self.attachment.bond(person_id, 0.1)
        self.learning.learn_fact(person_id, "first_met_context", context)
        self.context.person_entered(person_id)
        return person

    def identify_stranger(self, description: str = "") -> PersonProfile:
        """遇到不认识的人"""
        sid = f"stranger_{int(time.time() * 1000)}"
        person = PersonProfile(
            person_id=sid, name=f"陌生人({description[:10] if description else '未知'})",
            relationship=RelationshipLevel.STRANGER,
            first_met_ms=int(time.time() * 1000),
            notes=[f"未识别的陌生人: {description}"] if description else [],
        )
        self.relationships.add_person(person)
        self.context.person_entered(sid, 0.3)
        return person

    # ── 社交互动 ──────────────────────────────────────────────────────────

    def interact(self, person_id: str, quality: float, context: str = "",
                 topic: str = "", emotion: Optional[List[float]] = None) -> Dict[str, Any]:
        """记录一次社交互动，返回社交响应"""
        person = self.relationships.get_or_create(person_id)
        self.relationships.record_interaction(person_id, quality, context, topic, emotion)

        # 正面互动强化依恋
        if quality > 0.2:
            self.attachment.strengthen(person_id, quality * 0.05, context)

        # 社交学习
        if topic:
            self.learning.learn_fact(person_id, "discussed_topic", topic,
                                     confidence=min(0.9, abs(quality) + 0.2))

        # 生成社交响应
        greeting = ""
        greeting_needed = False
        if person.total_interactions == 1 or person.days_since_last_interaction() > 3:
            greeting = self.behavior.greet(person, context)
            greeting_needed = True

        return {
            "person": person,
            "greeting": greeting,
            "greeting_needed": greeting_needed,
            "relationship": person.relationship,
            "trust": person.trust_score,
            "familiarity": person.familiarity,
            "comfort": self.behavior.comfort_level(person),
            "should_share": self.behavior.should_share_opinion(person),
            "what_i_know": self.learning.person_summary(person_id),
        }

    # ── 问候与告别 ────────────────────────────────────────────────────────

    def greet_person(self, person_id: str) -> str:
        person = self.relationships.get_person(person_id)
        if not person:
            return f"你好，我是{self.identity.name}。我们见过吗？"
        self.context.person_entered(person_id)
        self.attachment.reunite(person_id)
        return self.behavior.greet(person)

    def say_goodbye(self, person_id: str) -> str:
        person = self.relationships.get_person(person_id)
        if not person:
            return "再见。"
        self.attachment.separate(person_id)
        self.context.person_left(person_id)
        return self.behavior.farewell(person)

    # ── 情境感知 ──────────────────────────────────────────────────────────

    def update_social_context(self, perception_snapshot=None) -> Dict[str, Any]:
        return self.context.update(perception_snapshot)

    # ── 主动社交冲动 ──────────────────────────────────────────────────────

    def social_urge(self) -> Optional[Dict[str, Any]]:
        """检查是否有社交冲动需要表达"""
        # 想念主人
        miss = self.attachment.miss_owner_check()
        if miss:
            return {"type": "miss_owner", "message": miss, "urgency": 0.4}

        # 看见认识的人想打招呼
        for pid in self.context.who_is_here():
            person = self.relationships.get_person(pid)
            if person and person.relationship in (
                RelationshipLevel.OWNER, RelationshipLevel.FAMILY,
                RelationshipLevel.CLOSE_FRIEND,
            ):
                if person.days_since_last_interaction() > 0.5:
                    return {
                        "type": "greet_familiar",
                        "message": self.behavior.greet(person),
                        "urgency": 0.6,
                        "person_id": pid,
                    }

        return None

    # ── 状态 ──────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        return {
            "identity": self.identity.name,
            "owner": self.relationships.owner.name if self.relationships.owner else "未认主",
            "relationships": self.relationships.stats(),
            "social_context": {
                "atmosphere": self.context.atmosphere,
                "people_present": self.context.how_many_people(),
                "active": self.context.is_socially_active,
            },
            "attachment": {
                "owner_strength": self.attachment.attachment_strength(self.relationships.owner_id) if self.relationships.owner_id else 0,
                "separation_anxiety": self.attachment._separation_anxiety,
            },
        }

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        self.relationships.save(dir_path / "relationships.json")
        self.learning.persistence(dir_path / "social_learning.json")

    @classmethod
    def load(cls, dir_path: Path, identity: Identity = Identity()) -> "SocialBrain":
        brain = cls(identity)
        rp = dir_path / "relationships.json"
        if rp.exists():
            brain.relationships = RelationshipModel.load(rp)
        lp = dir_path / "social_learning.json"
        if lp.exists():
            brain.learning = SocialLearning.load(lp)
        # 重建对主人的依恋
        if brain.relationships.owner_id:
            brain.attachment.bond(brain.relationships.owner_id, 1.0)
        return brain
