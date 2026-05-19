"""
Kali 全支柱训练 — 安全+感知+主动+社交 并行训练
Kali 4vCPU 轻量训练，Windows 负责 Deep Escalate
权重输出: $LIFERS_KALI_HOME/lifers/weights/lifers_*.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
WEIGHTS.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT.parent / "data"

# Kali 4vCPU → 保守线程配置
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

STATUS_FILE = WEIGHTS / ".kali_train_status.json"


def save_status(data: Dict):
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def train_safety() -> Dict:
    from lifers.scripts.train_lifers_safety import train_safety_classifier
    t0 = time.time()
    try:
        result = train_safety_classifier(n_epochs=200, lr=0.01, batch_size=64, verbose=True)
        elapsed = time.time() - t0
        return {"status": "ok", "elapsed": round(elapsed, 1)}
    except Exception as e:
        return {"status": "failed", "error": str(e), "elapsed": round(time.time() - t0, 1)}


def train_perception() -> Dict:
    from lifers.scripts.train_lifers_perception import train_perception_classifier
    t0 = time.time()
    try:
        result = train_perception_classifier(n_epochs=300, verbose=True)
        elapsed = time.time() - t0
        return {"status": "ok", "elapsed": round(elapsed, 1)}
    except Exception as e:
        return {"status": "failed", "error": str(e), "elapsed": round(time.time() - t0, 1)}


def train_proactive() -> Dict:
    from lifers.scripts.train_lifers_proactive import train_proactive_predictor
    t0 = time.time()
    try:
        result = train_proactive_predictor(n_epochs=200, verbose=True)
        elapsed = time.time() - t0
        return {"status": "ok", "elapsed": round(elapsed, 1)}
    except Exception as e:
        return {"status": "failed", "error": str(e), "elapsed": round(time.time() - t0, 1)}


def train_social() -> Dict:
    from lifers.scripts.train_lifers_social import train_social_classifier
    t0 = time.time()
    try:
        result = train_social_classifier(n_epochs=200, verbose=True)
        elapsed = time.time() - t0
        return {"status": "ok", "elapsed": round(elapsed, 1)}
    except Exception as e:
        return {"status": "failed", "error": str(e), "elapsed": round(time.time() - t0, 1)}


PILLARS = {
    "safety": train_safety,
    "perception": train_perception,
    "proactive": train_proactive,
    "social": train_social,
}


def train_all_sequential():
    print("=" * 60)
    print(f"  Kali Lifers 全支柱训练 ({len(PILLARS)}支柱)")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  权重输出: {WEIGHTS}")
    print(f"  线程配置: OMP={os.environ.get('OMP_NUM_THREADS', '?')}")
    print("=" * 60)

    all_stats = {}
    t0 = time.time()

    for i, (name, fn) in enumerate(PILLARS.items()):
        print(f"\n[{i+1}/{len(PILLARS)}] {name} 训练开始...")
        save_status({"phase": f"training_{name}", "pillar": name, "progress": f"{i+1}/{len(PILLARS)}"})

        stats = fn()
        all_stats[name] = stats
        print(f"  [{name}] {stats['status']} ({stats['elapsed']}s)")

        save_status({
            "phase": "training",
            "current_pillar": name,
            "progress": f"{i+1}/{len(PILLARS)}",
            "current_result": stats,
            "all_results": all_stats,
        })

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  Kali 训练完成 总耗时: {elapsed:.1f}s")
    for name, stats in all_stats.items():
        print(f"    [{name}] {stats['status']} {stats['elapsed']}s")
    print(f"{'='*60}")

    save_status({
        "phase": "completed",
        "total_elapsed": round(elapsed, 1),
        "results": all_stats,
    })

    return all_stats


def main():
    if len(sys.argv) > 1 and sys.argv[1] in PILLARS:
        pillar = sys.argv[1]
        fn = PILLARS[pillar]
        print(f"[Kali] 单支柱训练: {pillar}")
        result = fn()
        print(f"  结果: {result}")
    else:
        train_all_sequential()


if __name__ == "__main__":
    main()
