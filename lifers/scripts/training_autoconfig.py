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
    """
    全自适应训练配置 — 根据实时硬件探测动态计算最优参数。
    不再使用硬编码 tier 分支，改为连续缩放。
    """
    # 自适应硬件探测（优先使用传入的，否则实时探测）
    if hw_profile is None:
        try:
            from lifers.core.hardware_profile import get_profile
            hw = get_profile()
            hw_dict = hw.to_dict()
        except Exception:
            hw_dict = load_hardware_profile()
    else:
        hw_dict = hw_profile

    base = dict(_DEFAULT_CONFIG.get(pillar, {}))
    if not hw_dict:
        return base

    training = hw_dict.get("training", {})
    ram = hw_dict.get("ram", {})
    cpu = hw_dict.get("cpu", {})
    gpu = hw_dict.get("gpu", {})

    # 连续缩放因子（基于实际硬件，非离散tier）
    ram_total_mb = ram.get("total_mb", 4096)
    cores = cpu.get("cores_physical", 1)
    gpu_ok = gpu.get("available", False) or gpu.get("cuda_available", False)
    vram_mb = gpu.get("vram_mb", 0)

    # 硬件质量因子: RAM越多、核心越多 → 因子越高
    ram_factor = max(0.25, min(3.0, ram_total_mb / 8192))  # 8GB=1.0x
    core_factor = max(0.25, min(3.0, cores / 4))             # 4核=1.0x
    gpu_factor = 2.0 if (gpu_ok and vram_mb >= 4096) else (1.5 if gpu_ok else 1.0)

    # epoch倍率: 硬件越好 epoch 越多
    epoch_mult = training.get("epoch_multiplier", 1.0)

    # 获取自适应线程和batch
    try:
        from lifers.core.hardware_profile import get_profile
        hw_live = get_profile()
        threads = hw_live.threads_for(pillar)
        batch = hw_live.batch_size_for(pillar)
    except Exception:
        threads = training.get("global_threads", cores)
        batch = training.get("recommended_batch_size", 32)

    base["max_parallel_workers"] = max(1, int(cores * 0.75))
    base["preferred_device"] = "cuda" if gpu_ok else "cpu"
    base["memory_tier"] = training.get("memory_tier", "medium")
    base["optimal_threads"] = threads
    base["batch_size"] = batch

    # 动态计算每支柱参数（基于硬件因子的连续缩放）
    if pillar == "rl":
        base["lr"] = 1e-3 * ram_factor
        base["episodes"] = int(500 * epoch_mult)
        base["hidden_dim"] = int(128 * ram_factor * gpu_factor)

    elif pillar == "voice":
        base["lr"] = 1e-2 * ram_factor
        base["epochs"] = int(30 * epoch_mult)
        base["n_samples"] = int(500 * epoch_mult)
        base["hidden_dim"] = int(128 * ram_factor * gpu_factor)

    elif pillar == "kg":
        base["lr"] = 1e-2 * ram_factor
        base["epochs"] = int(50 * epoch_mult)
        base["n_triplets"] = int(400 * ram_factor * core_factor)
        base["embedding_dim"] = int(64 * ram_factor * gpu_factor)

    elif pillar == "safety":
        base["lr"] = 1e-2 * ram_factor
        base["epochs"] = int(30 * epoch_mult)

    elif pillar == "perception":
        base["lr"] = 2e-3 * ram_factor
        base["epochs"] = int(80 * epoch_mult)
        base["hidden_dim"] = int(64 * ram_factor * gpu_factor)

    elif pillar == "social":
        base["lr"] = 2e-3 * ram_factor
        base["epochs"] = int(60 * epoch_mult)
        base["hidden_dim"] = int(64 * ram_factor * gpu_factor)

    elif pillar == "proactive":
        base["lr"] = 1e-3 * ram_factor
        base["epochs"] = int(80 * epoch_mult)

    elif pillar == "robot_hal":
        base["lr"] = 3e-3 * ram_factor
        base["episodes"] = int(300 * epoch_mult)
        base["hidden_dim"] = int(96 * ram_factor)

    elif pillar == "swarm":
        base["episodes"] = int(600 * epoch_mult)

    elif pillar == "simulation":
        base["lr"] = 1e-2 * ram_factor
        base["epochs"] = int(60 * epoch_mult)
        base["hidden_dim"] = int(64 * ram_factor * gpu_factor)

    elif pillar in ("telemetry", "dashboard"):
        pass

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
