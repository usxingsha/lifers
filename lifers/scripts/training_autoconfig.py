"""
Lifers 训练自动配置 — 读取硬件配置，自动匹配最优训练参数
用法: from lifers.scripts.training_autoconfig import get_training_config
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_CONFIG = {
    "rl": {"episodes": 500, "lr": 1e-3, "batch_size": 32, "hidden_dim": 128},
    "voice": {"epochs": 30, "lr": 1e-2, "n_samples": 500, "hidden_dim": 128},
    "kg": {"epochs": 50, "lr": 1e-2, "n_triplets": 200, "embedding_dim": 64, "hard_negatives": 8},
    "safety": {"epochs": 30, "lr": 1e-2, "batch_size": 256, "max_len": 128},
    "perception": {"epochs": 80, "lr": 2e-3, "hidden_dim": 64},
    "social": {"epochs": 60, "lr": 2e-3, "hidden_dim": 64},
    "proactive": {"epochs": 80, "lr": 1e-3, "input_dim": 64},
    "robot_hal": {"episodes": 300, "lr": 3e-3, "hidden_dim": 96},
    "swarm": {"episodes": 600, "lr": 1e-2},
    "simulation": {"epochs": 60, "lr": 1e-2, "hidden_dim": 64},
    "telemetry": {"epochs": 50, "lr": 1e-3, "hidden_dim": 32},
    "dashboard": {"epochs": 1, "lr": 0},
}


def load_hardware_profile(path: Optional[Path] = None) -> Dict[str, Any]:
    if path is None:
        path = ROOT / "weights" / "lifers_hardware_profile.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_training_config(pillar: str, hw_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """根据硬件配置返回指定支柱的最优训练参数"""
    if hw_profile is None:
        hw_profile = load_hardware_profile()

    base = dict(_DEFAULT_CONFIG.get(pillar, {}))
    if not hw_profile:
        return base

    training = hw_profile.get("training", {})
    ram = hw_profile.get("ram", {})
    cpu = hw_profile.get("cpu", {})
    gpu = hw_profile.get("gpu", {})

    tier = training.get("memory_tier", "low")
    workers = training.get("max_parallel_workers", 1)
    batch = training.get("recommended_batch_size", 32)
    device = training.get("preferred_device", "cpu")
    epoch_mult = training.get("recommended_epochs_multiplier", 1.0)
    has_cuda = gpu.get("cuda_available", False)
    vram_mb = gpu.get("vram_mb", 0)
    ram_mb = ram.get("total_mb", 4096)
    cores = cpu.get("cores_physical", 1)

    # 通用参数
    base["max_parallel_workers"] = workers
    base["preferred_device"] = device
    base["memory_tier"] = tier

    # 按支柱微调
    if pillar == "rl":
        base["lr"] = 2e-3 if tier == "high" else 1e-3
        base["episodes"] = int(500 * epoch_mult)
        base["batch_size"] = min(batch, 64)
        base["hidden_dim"] = 256 if tier == "high" else 128
        if has_cuda and vram_mb >= 4096:
            base["hidden_dim"] = 512

    elif pillar == "voice":
        base["lr"] = 2e-2 if tier == "high" else 1e-2
        base["epochs"] = int(30 * epoch_mult)
        base["n_samples"] = int(500 * epoch_mult)
        base["hidden_dim"] = 256 if tier == "high" else 128
        if tier == "low":
            base["n_samples"] = 200
            base["epochs"] = 20

    elif pillar == "kg":
        base["lr"] = 2e-2 if tier == "high" else 1e-2
        base["epochs"] = int(50 * epoch_mult)
        base["n_triplets"] = 200 if tier == "low" else (400 if tier == "medium" else 800)
        base["embedding_dim"] = 128 if tier == "high" else 64
        base["hard_negatives"] = 16 if tier == "high" else 8

    elif pillar == "safety":
        base["lr"] = 2e-2 if tier == "high" else 1e-2
        base["epochs"] = min(int(base["epochs"] * epoch_mult), 80)

    elif pillar == "perception":
        base["lr"] = 4e-3 if tier == "high" else 2e-3
        base["epochs"] = min(int(base["epochs"] * epoch_mult), 150)
        base["hidden_dim"] = 128 if tier == "high" else 64

    elif pillar == "social":
        base["lr"] = 4e-3 if tier == "high" else 2e-3
        base["epochs"] = min(int(base["epochs"] * epoch_mult), 120)
        base["hidden_dim"] = 128 if tier == "high" else 64

    elif pillar == "proactive":
        base["lr"] = 2e-3 if tier == "high" else 1e-3
        base["epochs"] = min(int(base["epochs"] * epoch_mult), 150)

    elif pillar == "robot_hal":
        base["lr"] = 3e-3
        base["episodes"] = min(int(base["episodes"] * epoch_mult), 800)
        base["hidden_dim"] = 96

    elif pillar == "swarm":
        base["episodes"] = min(int(base["episodes"] * epoch_mult), 1200)

    elif pillar == "simulation":
        base["lr"] = 2e-2 if tier == "high" else 1e-2
        base["epochs"] = min(int(base["epochs"] * epoch_mult), 120)
        base["hidden_dim"] = 128 if tier == "high" else 64

    elif pillar in ("telemetry", "dashboard"):
        pass  # 无epochs参数

    return base


def get_all_configs(hw_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """获取所有支柱的自动配置"""
    if hw_profile is None:
        hw_profile = load_hardware_profile()
    return {p: get_training_config(p, hw_profile) for p in _DEFAULT_CONFIG}


def get_split_plan(hw_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """根据硬件决定是否拆分训练以及如何合并"""
    if hw_profile is None:
        hw_profile = load_hardware_profile()

    training = hw_profile.get("training", {})
    ram = hw_profile.get("ram", {})
    cpu = hw_profile.get("cpu", {})
    gpu = hw_profile.get("gpu", {})

    tier = training.get("memory_tier", "low")
    ram_mb = ram.get("total_mb", 4096)
    cores = cpu.get("cores_physical", 1)
    has_cuda = gpu.get("cuda_available", False)
    vram_mb = gpu.get("vram_mb", 0)

    plan = {
        "can_train_all_at_once": ram_mb >= 8192 and cores >= 4,
        "split_training_recommended": ram_mb < 4096 or cores < 2,
        "parallel_groups": [],
        "sequential_order": ["corpus", "kg", "voice", "rl", "safety", "perception",
                           "social", "proactive", "robot_hal", "swarm",
                           "simulation", "telemetry", "dashboard"],
        "merge_strategy": "weight_concat",
        "estimated_total_time_s": 0,
    }

    # 低配：分组顺序执行
    if tier == "low":
        plan["parallel_groups"] = [
            ["corpus"], ["kg"], ["voice"], ["rl"],
            ["safety", "social"], ["perception", "proactive"],
            ["robot_hal"], ["swarm"], ["simulation", "telemetry", "dashboard"],
        ]
        plan["estimated_total_time_s"] = 600
    # 中配：3组并行
    elif tier == "medium":
        plan["parallel_groups"] = [
            ["corpus", "kg", "voice", "rl"],
            ["safety", "perception", "social", "proactive"],
            ["robot_hal", "swarm", "simulation", "telemetry", "dashboard"],
        ]
        plan["estimated_total_time_s"] = 250
    # 高配：2组并行
    else:
        plan["parallel_groups"] = [
            ["corpus", "kg", "voice", "rl", "safety", "perception", "social", "proactive"],
            ["robot_hal", "swarm", "simulation", "telemetry", "dashboard"],
        ]
        plan["estimated_total_time_s"] = 60

    if has_cuda and vram_mb >= 4096:
        plan["can_use_gpu_acceleration"] = True
        plan["estimated_total_time_s"] = max(30, plan["estimated_total_time_s"] // 2)

    return plan


def print_config_summary(configs: Dict[str, Dict[str, Any]], plan: Dict[str, Any]):
    """打印自动配置摘要"""
    print("\n" + "=" * 60)
    print("  Lifers 训练自动配置")
    print("=" * 60)
    print(f"  内存级别: {configs.get('rl', {}).get('memory_tier', 'unknown')}")
    print(f"  计算设备: {configs.get('rl', {}).get('preferred_device', 'cpu')}")
    print(f"  全部同时训练: {'是' if plan.get('can_train_all_at_once') else '否'}")
    print(f"  预计总时间: ~{plan.get('estimated_total_time_s', '?')}s")
    print(f"  合并策略: {plan.get('merge_strategy', 'none')}")
    print()
    for p, cfg in configs.items():
        print(f"  [{p}] epochs={cfg.get('epochs','?')}  lr={cfg.get('lr','?'):.0e}  "
              f"batch={cfg.get('batch_size', cfg.get('n_samples', '?'))}  "
              f"workers={cfg.get('max_parallel_workers','?')}")
    print("=" * 60)


def main():
    from lifers.scripts.hardware_probe import probe_hardware

    hw = probe_hardware(verbose=False)
    configs = get_all_configs(hw)
    plan = get_split_plan(hw)
    print_config_summary(configs, plan)

    # 保存配置
    out = ROOT / "weights" / "lifers_training_config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"configs": configs, "split_plan": plan}, f, ensure_ascii=False, indent=2)
    print(f"\n配置已保存 → {out}")


if __name__ == "__main__":
    main()
