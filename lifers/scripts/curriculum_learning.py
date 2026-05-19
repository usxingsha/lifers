"""
Lifers 课程学习系统 — 人类成长模式训练
从基础到专家，渐进式学习，动态适应

成长阶段：
  Stage 0: 婴儿期 (0-20%) — 基础概念，高LR，小数据
  Stage 1: 儿童期 (20-40%) — 模式识别，中等数据
  Stage 2: 少年期 (40-60%) — 复杂推理，降低LR
  Stage 3: 青年期 (60-80%) — 专业知识，大数据
  Stage 4: 成年期 (80-95%) — 专家级，微调LR
  Stage 5: 大师期 (95-100%) — 持续学习，极小LR
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

GROWTH_STAGES = {
    0: {"name": "婴儿期", "pct_range": (0, 20), "lr_mult": 2.0, "data_mult": 0.1, "epoch_mult": 0.3},
    1: {"name": "儿童期", "pct_range": (20, 40), "lr_mult": 1.5, "data_mult": 0.3, "epoch_mult": 0.5},
    2: {"name": "少年期", "pct_range": (40, 60), "lr_mult": 1.0, "data_mult": 0.6, "epoch_mult": 0.8},
    3: {"name": "青年期", "pct_range": (60, 80), "lr_mult": 0.7, "data_mult": 0.8, "epoch_mult": 1.0},
    4: {"name": "成年期", "pct_range": (80, 95), "lr_mult": 0.3, "data_mult": 1.0, "epoch_mult": 1.2},
    5: {"name": "大师期", "pct_range": (95, 100), "lr_mult": 0.1, "data_mult": 1.5, "epoch_mult": 1.5},
}


@dataclass
class GrowthMetrics:
    """单个支柱的成长指标"""
    pillar: str
    current_accuracy: float = 0.0
    best_accuracy: float = 0.0
    current_loss: float = float("inf")
    epochs_trained: int = 0
    total_epochs_target: int = 100
    samples_used: int = 0
    stage: int = 0
    stage_name: str = "婴儿期"
    lr_current: float = 0.01
    lr_base: float = 0.01
    plateau_count: int = 0  # 连续无提升计数
    last_improvement_epoch: int = 0
    growth_rate: float = 0.0  # 每epoch准确率提升
    history: List[Dict] = field(default_factory=list)

    @property
    def progress_pct(self) -> float:
        return self.epochs_trained / max(self.total_epochs_target, 1) * 100

    @property
    def is_plateau(self) -> bool:
        return self.plateau_count >= 10

    @property
    def maturity(self) -> float:
        """成熟度 0-1"""
        return min(1.0, self.best_accuracy / 0.95)


class CurriculumTrainer:
    """课程学习训练器 — 人类成长模式"""

    def __init__(self):
        self.metrics: Dict[str, GrowthMetrics] = {}
        self.global_stage = 0
        self.global_epoch = 0
        self._history_path = ROOT / "weights" / "lifers_growth_history.json"

    def register_pillar(self, pillar: str, base_lr: float, target_epochs: int, target_samples: int):
        """注册训练支柱"""
        self.metrics[pillar] = GrowthMetrics(
            pillar=pillar,
            lr_current=base_lr,
            lr_base=base_lr,
            total_epochs_target=target_epochs,
            samples_used=target_samples,
        )

    def get_stage_config(self, pillar: str) -> Dict[str, Any]:
        """根据当前成长阶段获取训练配置"""
        gm = self.metrics.get(pillar)
        if gm is None:
            return GROWTH_STAGES[0]

        if gm.progress_pct <= 20:
            return GROWTH_STAGES[0]
        elif gm.progress_pct <= 40:
            return GROWTH_STAGES[1]
        elif gm.progress_pct <= 60:
            return GROWTH_STAGES[2]
        elif gm.progress_pct <= 80:
            return GROWTH_STAGES[3]
        elif gm.progress_pct <= 95:
            return GROWTH_STAGES[4]
        else:
            return GROWTH_STAGES[5]

    def update(self, pillar: str, loss: float, accuracy: float):
        """每个epoch更新成长指标"""
        gm = self.metrics.get(pillar)
        if gm is None:
            return

        gm.epochs_trained += 1
        gm.current_loss = loss
        gm.current_accuracy = accuracy

        # 检测提升
        improved = accuracy > gm.best_accuracy + 0.001
        if improved:
            gm.best_accuracy = accuracy
            gm.plateau_count = 0
            gm.last_improvement_epoch = gm.epochs_trained
            gm.growth_rate = (accuracy - gm.best_accuracy) / max(gm.epochs_trained, 1)
        else:
            gm.plateau_count += 1

        # 阶段转换
        old_stage = gm.stage
        stage_cfg = self.get_stage_config(pillar)
        for s_id in sorted(GROWTH_STAGES.keys()):
            lo, hi = GROWTH_STAGES[s_id]["pct_range"]
            if lo <= gm.progress_pct < hi:
                gm.stage = s_id
                gm.stage_name = GROWTH_STAGES[s_id]["name"]
                break

        # 阶段变化时调整学习率
        if gm.stage != old_stage:
            gm.lr_current = gm.lr_base * stage_cfg["lr_mult"]
            gm.plateau_count = 0  # 重置

        # 平台期自动调整
        if gm.is_plateau and gm.epochs_trained > 20:
            gm.lr_current *= 0.5
            gm.plateau_count = 0

        # 记录历史
        gm.history.append({
            "epoch": gm.epochs_trained,
            "loss": loss,
            "accuracy": accuracy,
            "stage": gm.stage,
            "lr": gm.lr_current,
        })

    def get_lr(self, pillar: str) -> float:
        gm = self.metrics.get(pillar)
        return gm.lr_current if gm else 0.01

    def get_epoch_boost(self, pillar: str) -> float:
        """根据阶段返回epoch倍数"""
        cfg = self.get_stage_config(pillar)
        return cfg["epoch_mult"]

    def get_data_fraction(self, pillar: str) -> float:
        """根据阶段返回数据使用比例"""
        cfg = self.get_stage_config(pillar)
        return cfg["data_mult"]

    def should_skip(self, pillar: str) -> bool:
        """如果已经收敛(大师期+高准确率)则可跳过"""
        gm = self.metrics.get(pillar)
        if gm is None:
            return False
        return gm.stage >= 5 and gm.best_accuracy >= 0.98

    def summary(self) -> str:
        """成长摘要"""
        lines = ["", "=" * 70, "  Lifers 人类成长训练报告", "=" * 70]
        total_maturity = 0.0
        for gm in sorted(self.metrics.values(), key=lambda g: g.maturity, reverse=True):
            bar = "#" * int(gm.maturity * 20) + "-" * (20 - int(gm.maturity * 20))
            lines.append(
                f"  [{gm.pillar:<16}] {gm.stage_name:<6} [{bar}] "
                f"acc={gm.best_accuracy:.3f} epoch={gm.epochs_trained}/{gm.total_epochs_target} "
                f"lr={gm.lr_current:.5f}"
            )
            total_maturity += gm.maturity
        avg_maturity = total_maturity / max(len(self.metrics), 1)
        lines.append(f"  全局成熟度: {avg_maturity:.2%}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def save(self):
        """保存成长历史"""
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "brand": "Lifers Growth History",
            "global_stage": self.global_stage,
            "global_epoch": self.global_epoch,
            "pillars": {
                p: {
                    "current_accuracy": gm.current_accuracy,
                    "best_accuracy": gm.best_accuracy,
                    "epochs_trained": gm.epochs_trained,
                    "total_epochs_target": gm.total_epochs_target,
                    "stage": gm.stage,
                    "stage_name": gm.stage_name,
                    "lr_current": gm.lr_current,
                    "samples_used": gm.samples_used,
                    "plateau_count": gm.plateau_count,
                    "maturity": gm.maturity,
                    "history": gm.history[-50:],  # 只保存最近50条
                }
                for p, gm in self.metrics.items()
            },
        }
        with open(self._history_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        """加载成长历史"""
        if not self._history_path.exists():
            return False
        with open(self._history_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.global_stage = data.get("global_stage", 0)
        self.global_epoch = data.get("global_epoch", 0)
        for pillar, pd in data.get("pillars", {}).items():
            gm = GrowthMetrics(
                pillar=pillar,
                current_accuracy=pd.get("current_accuracy", 0),
                best_accuracy=pd.get("best_accuracy", 0),
                epochs_trained=pd.get("epochs_trained", 0),
                total_epochs_target=pd.get("total_epochs_target", 100),
                stage=pd.get("stage", 0),
                stage_name=pd.get("stage_name", "婴儿期"),
                lr_current=pd.get("lr_current", 0.01),
                samples_used=pd.get("samples_used", 0),
                plateau_count=pd.get("plateau_count", 0),
            )
            self.metrics[pillar] = gm
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# 自动成长数据生成器 — 根据训练结果自动扩展数据
# ═══════════════════════════════════════════════════════════════════════════════

class AutoGrowthDataGenerator:
    """根据训练情况自动生成新数据 — 数据库随时间自动成长"""

    def __init__(self):
        self._rng = np.random.RandomState(int(time.time()))

    def generate_for_pillar(self, pillar: str, current_accuracy: float,
                            existing_samples: int = 0) -> List[Dict]:
        """根据当前准确率生成更多/更难数据"""
        from lifers.scripts.generate_training_data import (
            _gen_safety_samples, _gen_social_samples,
            _gen_proactive_samples, _gen_perception_samples,
        )

        # 低准确率 → 更多基础数据；高准确率 → 进阶数据
        if current_accuracy < 0.7:
            batch_size = 10000  # 大量基础数据
        elif current_accuracy < 0.85:
            batch_size = 5000   # 中等数据
        else:
            batch_size = 2000   # 精细化数据

        if pillar == "safety":
            safe, unsafe = _gen_safety_samples()
            safe = safe[:batch_size]
            unsafe = unsafe[:batch_size]
            return [
                {"text": s, "label": 0} for s in safe
            ] + [
                {"text": u, "label": 1} for u in unsafe
            ]
        elif pillar == "social":
            return _gen_social_samples()[:batch_size]
        elif pillar == "proactive":
            return _gen_proactive_samples()[:batch_size]
        elif pillar == "perception":
            return _gen_perception_samples()[:batch_size]

        return []

    def expand_data(self, pillar: str, accuracy: float,
                    data_dir: Optional[Path] = None) -> int:
        """扩展现有数据，返回新增数量"""
        if data_dir is None:
            data_dir = ROOT / "data"

        data_dir.mkdir(parents=True, exist_ok=True)

        # 检查现有数据量
        existing = 0
        mapping = {
            "safety": ("safety_safe.jsonl", "safety_unsafe.jsonl"),
            "social": ("social_samples.jsonl",),
            "proactive": ("proactive_samples.jsonl",),
            "perception": ("perception_samples.jsonl",),
        }

        files = mapping.get(pillar, ())
        for f in files:
            path = data_dir / f
            if path.exists():
                with open(path, encoding="utf-8") as fp:
                    existing += sum(1 for _ in fp)

        # 数据不足或准确率低时生成更多
        if existing < 50000 or accuracy < 0.85:
            new_samples = self.generate_for_pillar(pillar, accuracy, existing)
            if new_samples:
                # 追加到现有文件
                for f in files:
                    path = data_dir / f
                    label = 0 if "safe" in f else (1 if "unsafe" in f else None)
                    mode = "a" if path.exists() else "w"
                    with open(path, mode, encoding="utf-8") as fp:
                        for s in new_samples:
                            if label is not None and s.get("label") == label:
                                fp.write(json.dumps(s, ensure_ascii=False) + "\n")
                            elif label is None:
                                fp.write(json.dumps(s, ensure_ascii=False) + "\n")
                return len(new_samples)
        return 0


curriculum = CurriculumTrainer()
auto_growth = AutoGrowthDataGenerator()
