"""
Lifers Dashboard 仪表盘配置训练 — 指标聚合与可视化配置优化
品牌化权重: weights/lifers_dashboard_config.json
纯numpy 指标重要性排序 + 自适应刷新率
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

# 仪表盘指标: (name, importance, refresh_interval_s, alert_threshold, category)
_DASHBOARD_METRICS = [
    ("系统健康分", 0.95, 10, 0.7, "health"),
    ("CPU使用率", 0.85, 5, 90.0, "resource"),
    ("内存使用率", 0.85, 5, 85.0, "resource"),
    ("磁盘使用率", 0.75, 30, 90.0, "resource"),
    ("网络带宽", 0.70, 10, 0.0, "resource"),
    ("训练进度", 0.90, 15, 0.0, "training"),
    ("训练损失", 0.90, 15, 0.0, "training"),
    ("推理延迟", 0.80, 5, 500.0, "performance"),
    ("对话响应率", 0.80, 10, 0.95, "interaction"),
    ("安全告警", 0.95, 5, 1.0, "safety"),
    ("主动行为触发", 0.70, 30, 0.0, "behavior"),
    ("知识图谱更新", 0.65, 60, 0.0, "knowledge"),
    ("语料库大小", 0.55, 120, 0.0, "storage"),
    ("模型版本", 0.60, 300, 0.0, "config"),
    ("日志错误率", 0.90, 10, 0.01, "health"),
]


class LifersDashboardConfig:
    """仪表盘配置优化器"""

    def __init__(self):
        self.metrics = []
        self.importance_threshold = 0.6
        self.default_refresh = 30

    def fit(self, metric_data: List[Tuple[str, float, int, float, str]]):
        self.metrics = []
        for name, imp, refresh, alert, cat in metric_data:
            self.metrics.append({
                "name": name,
                "importance": imp,
                "refresh_interval": refresh,
                "alert_threshold": alert,
                "category": cat,
            })

    def optimize(self) -> Dict:
        """优化仪表盘配置"""
        # 按重要性排序
        sorted_metrics = sorted(self.metrics, key=lambda m: m["importance"], reverse=True)

        # 分类聚合
        categories = {}
        for m in sorted_metrics:
            cat = m["category"]
            if cat not in categories:
                categories[cat] = {"count": 0, "avg_importance": [], "min_refresh": float("inf")}
            categories[cat]["count"] += 1
            categories[cat]["avg_importance"].append(m["importance"])
            categories[cat]["min_refresh"] = min(categories[cat]["min_refresh"], m["refresh_interval"])

        for cat in categories:
            categories[cat]["avg_importance"] = float(np.mean(categories[cat]["avg_importance"]))

        # 关键指标 (重要性 > 阈值)
        critical = [m for m in sorted_metrics if m["importance"] >= self.importance_threshold]

        # 自适应刷新率
        avg_importance = np.mean([m["importance"] for m in sorted_metrics])
        adaptive_refresh = int(self.default_refresh / max(avg_importance, 0.1))

        return {
            "critical_metrics": [m["name"] for m in critical],
            "categories": {k: {"count": v["count"], "avg_importance": round(v["avg_importance"], 3),
                               "min_refresh": int(v["min_refresh"])}
                          for k, v in categories.items()},
            "adaptive_refresh_s": adaptive_refresh,
            "importance_threshold": self.importance_threshold,
            "total_metrics": len(self.metrics),
            "sorted_by_importance": [(m["name"], m["importance"]) for m in sorted_metrics[:5]],
        }


def train_dashboard_config(
    save_path: Optional[Path] = None, verbose=True,
) -> LifersDashboardConfig:
    if save_path is None:
        save_path = ROOT / "weights" / "lifers_dashboard_config.json"

    config = LifersDashboardConfig()
    config.fit(_DASHBOARD_METRICS)
    optimized = config.optimize()

    _save_dashboard_config(config, optimized, save_path)

    if verbose:
        print(f"[Lifers-Dashboard] 配置优化完成")
        print(f"  关键指标: {len(optimized['critical_metrics'])}")
        print(f"  分类: {list(optimized['categories'].keys())}")
        print(f"  自适应刷新: {optimized['adaptive_refresh_s']}s")
        print(f"  -> {save_path}")

    return config


def _save_dashboard_config(config: LifersDashboardConfig, optimized: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "brand": "Lifers Dashboard Config",
        "version": 1,
        "metrics": config.metrics,
        "optimized": optimized,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    out = ROOT / "weights" / "lifers_dashboard_config.json"
    print(f"[Lifers-Dashboard] 品牌化仪表盘配置训练")
    t0 = time.time()
    train_dashboard_config(save_path=out, verbose=True)
    print(f"[Lifers-Dashboard] 完成 耗时={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
