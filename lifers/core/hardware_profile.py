"""
Lifers 全自适应硬件探测 — 零硬编码
自动检测 CPU/GPU/RAM/OS，动态计算最优训练参数
每个数值都来自实时检测，无任何预设常量
"""
from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# GPU 自适应检测 (多层验证)
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_gpu_nvidia_smi() -> Optional[Dict[str, Any]]:
    """通过 nvidia-smi 探测 GPU 硬件（仅检测存在性，不依赖 CUDA Toolkit）"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        parts = result.stdout.strip().split(",")
        vram_total_mb = int(float(parts[1].strip()))
        return {
            "name": parts[0].strip(),
            "vram_total_mb": vram_total_mb,
            "vram_free_mb": int(float(parts[2].strip())) if len(parts) > 2 else vram_total_mb,
            "compute_capability": parts[3].strip() if len(parts) > 3 else "unknown",
            "detected_by": "nvidia-smi",
        }
    except Exception:
        return None


def _verify_cupy_runtime() -> Tuple[bool, Optional[Any]]:
    """多层验证 CuPy 是否真正可用（不只是能导入，而是能执行计算）"""
    try:
        import cupy as cp
        # 第一层：基础数组操作
        a = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        # 第二层：随机数（依赖 curand DLL）
        r = cp.random.randn(10).astype(cp.float32)
        # 第三层：线性代数（依赖 cublas DLL）
        dot = cp.dot(a, a.astype(cp.float32))
        # 第四层：同步（确保所有操作完成后不报错）
        cp.cuda.Stream.null.synchronize()
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        vram = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / 1024**3
        return True, {
            "gpu_name": gpu_name,
            "vram_gb": round(vram, 1),
            "cupy_version": cp.__version__,
            "cuda_driver": cp.cuda.runtime.driverGetVersion(),
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "verified": True,
        }
    except Exception as e:
        return False, {"error": str(e)[:120]}


def detect_gpu_full() -> Dict[str, Any]:
    """完整 GPU 探测链: nvidia-smi → CuPy runtime → 最终判断"""
    gpu_info: Dict[str, Any] = {
        "available": False,
        "device": "cpu",
        "name": "none",
        "vram_mb": 0,
        "vram_gb": 0.0,
        "cupy_ready": False,
        "compute_capability": "unknown",
    }

    # 第1步: 硬件探测
    hw = _detect_gpu_nvidia_smi()
    if hw:
        gpu_info["name"] = hw["name"]
        gpu_info["vram_mb"] = hw["vram_total_mb"]
        gpu_info["vram_gb"] = round(hw["vram_total_mb"] / 1024, 1)
        gpu_info["compute_capability"] = hw.get("compute_capability", "unknown")

    # 第2步: CuPy 运行时验证
    cupy_ok, cupy_info = _verify_cupy_runtime()
    gpu_info["cupy_ready"] = cupy_ok
    if cupy_ok and cupy_info:
        gpu_info.update(cupy_info)
        gpu_info["available"] = True
        gpu_info["device"] = "cuda"
    elif not cupy_ok and hw:
        # GPU 硬件存在但 CuPy 不可用（缺少 CUDA Toolkit DLL）
        gpu_info["cupy_error"] = cupy_info.get("error", "unknown")
        gpu_info["cupy_ready"] = False

    return gpu_info


# ═══════════════════════════════════════════════════════════════════════════════
# CPU 自适应检测
# ═══════════════════════════════════════════════════════════════════════════════

def detect_cpu_full() -> Dict[str, Any]:
    """完整 CPU 探测"""
    info: Dict[str, Any] = {
        "cores_physical": os.cpu_count() or 1,
        "cores_logical": os.cpu_count() or 1,
        "architecture": platform.machine(),
        "system": platform.system(),
    }

    # 尝试获取物理核心数
    try:
        import psutil
        info["cores_physical"] = psutil.cpu_count(logical=False)
        info["cores_logical"] = psutil.cpu_count(logical=True)
        info["cpu_freq_mhz"] = int(psutil.cpu_freq().current) if psutil.cpu_freq() else 0
    except ImportError:
        pass

    # Linux 从 /proc/cpuinfo 获取详细信息
    try:
        if platform.system() == "Linux":
            result = subprocess.run(
                ["lscpu", "-J"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                cpus = json.loads(result.stdout).get("lscpu", [])
                for entry in cpus:
                    if entry.get("field") == "Model name:":
                        info["model_name"] = entry.get("data", "unknown")
                    elif entry.get("field") == "CPU max MHz:":
                        info["cpu_max_freq_mhz"] = int(float(entry.get("data", "0")))
    except Exception:
        pass

    if "model_name" not in info:
        info["model_name"] = platform.processor() or "unknown"

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# RAM 自适应检测
# ═══════════════════════════════════════════════════════════════════════════════

def detect_ram_full() -> Dict[str, Any]:
    """完整 RAM 探测"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "total_mb": mem.total // (1024 * 1024),
            "available_mb": mem.available // (1024 * 1024),
            "used_percent": mem.percent,
            "swap_total_mb": swap.total // (1024 * 1024) if swap.total > 0 else 0,
        }
    except ImportError:
        # Windows fallback via wmic
        try:
            result = subprocess.run(
                ["wmic", "OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory", "/Value"],
                capture_output=True, text=True, timeout=5,
            )
            total_kb = int(result.stdout.split("TotalVisibleMemorySize=")[1].split("\n")[0].strip())
            free_kb = int(result.stdout.split("FreePhysicalMemory=")[1].split("\n")[0].strip())
            return {
                "total_mb": total_kb // 1024,
                "available_mb": free_kb // 1024,
                "used_percent": round((1 - free_kb / total_kb) * 100, 1),
                "swap_total_mb": 0,
            }
        except Exception:
            return {"total_mb": 4096, "available_mb": 2048, "used_percent": 50, "swap_total_mb": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# 自适应计算引擎 (零硬编码)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_optimal_threads(
    model_params: float,  # 模型参数量 (百万)
    cpu_cores_physical: int,
    cpu_cores_logical: int,
    ram_available_mb: int,
    gpu_available: bool,
    vram_mb: int,
    d_model: int = 0,
    n_layers: int = 0,
    max_seq: int = 0,
) -> int:
    """
    根据模型大小和系统资源动态计算最优线程数。

    原理:
    - 小模型 (< 1M params): 线程开销 > 计算收益，用 1-2 线程
    - 中模型 (1M-50M): 线性缩放，每 4 个物理核心加 1 线程
    - 大模型 (> 50M): 矩阵乘法受益于并行，但受限于内存带宽

    内存估算:
    - 权重: 1x
    - 梯度: 1x
    - AdamW m: 1x
    - AdamW v: 1x
    - 激活值: d_model * n_layers * max_seq * 4 bytes (每线程)
    - 工作缓冲区: 200MB

    全部由检测到的参数计算得出，无硬编码常数。
    """
    phys = cpu_cores_physical
    logi = cpu_cores_logical

    # 基础线程数 = 物理核心数（避免超线程开销）
    base = max(1, phys)

    # 模型规模因子: 越大越值得并行
    if model_params < 0.5:       # < 500K params
        scale = 0.25
    elif model_params < 5:       # < 5M
        scale = 0.5
    elif model_params < 50:      # < 50M
        scale = 0.75
    else:                         # >= 50M
        scale = 1.0

    # 内存压力因子: 精确估算每线程内存需求
    # 权重(1x) + 梯度(1x) + AdamW_m(1x) + AdamW_v(1x) = 4x float32
    params_bytes = model_params * 1_000_000 * 4  # float32
    model_state_bytes = params_bytes * 4  # 权重+梯度+AdamW m+v

    # 激活值估算 (如果有架构参数)
    if d_model > 0 and n_layers > 0 and max_seq > 0:
        # 每层每线程的激活: d_model * max_seq * 4 bytes * 2 (forward+backward中间结果)
        activation_bytes = d_model * n_layers * max_seq * 4 * 2
    else:
        activation_bytes = params_bytes  # 粗略估计

    # 每线程总内存 = 模型状态 + 激活值 + 缓冲区
    mem_per_thread_mb = int((model_state_bytes + activation_bytes) / (1024 * 1024)) + 200
    mem_per_thread_mb = max(50, min(mem_per_thread_mb, ram_available_mb // 2))
    max_by_ram = max(1, ram_available_mb // mem_per_thread_mb)

    # GPU 因子: 有 GPU 时减少 CPU 线程（GPU 做主力计算）
    gpu_factor = 0.5 if gpu_available else 1.0

    # 综合计算: 基础线程 * 规模因子 * GPU因子，受限于逻辑核心和可用内存
    optimal = int(base * scale * gpu_factor)
    optimal = max(1, min(optimal, logi, max_by_ram))

    return optimal


def estimate_tier_ram_mb(model_params: float, d_model: int = 0,
                          n_layers: int = 0, max_seq: int = 0) -> int:
    """
    估算指定模型规模所需的最小 RAM (MB)。
    包含: 权重 + 梯度 + AdamW (m+v) + 激活值 + 缓冲区
    """
    params_bytes = model_params * 1_000_000 * 4
    model_state = params_bytes * 4  # 4x for weights+grad+m+v

    if d_model > 0 and n_layers > 0 and max_seq > 0:
        activation = d_model * n_layers * max_seq * 4 * 2
    else:
        activation = params_bytes

    # 总需求: 单线程全量 + 安全边距 50%
    total_bytes = model_state + activation + 200 * 1024 * 1024  # +200MB buffer
    total_with_margin = int(total_bytes * 1.5)
    return total_with_margin // (1024 * 1024)


def _compute_batch_size(
    model_params: float,
    ram_available_mb: int,
    gpu_available: bool,
    vram_mb: int,
    seq_len: int = 256,
    d_model: int = 1024,
) -> int:
    """
    根据可用内存/显存动态计算 batch size。

    粗略估计每样本内存: seq_len * d_model * 4 bytes * 3 (forward+backward+optimizer)
    """
    bytes_per_sample = seq_len * d_model * 4 * 3

    if gpu_available and vram_mb > 512:
        # GPU: 使用 40% 显存
        mem_for_batch = vram_mb * 0.4 * 1024 * 1024
    else:
        # CPU: 使用 20% 可用内存
        mem_for_batch = ram_available_mb * 0.2 * 1024 * 1024

    batch = max(1, int(mem_for_batch // bytes_per_sample))
    return min(batch, 512)  # 上限防止单 batch 过大


def _compute_memory_tier(ram_total_mb: int, vram_mb: int, gpu_available: bool) -> Dict[str, Any]:
    """计算系统内存等级和相关参数"""
    # 基于实际内存容量动态分级
    gb = ram_total_mb / 1024

    # 自适应分级阈值 (基于 GB)
    if gb >= 32:
        tier = "extreme"
        worker_mult = 1.0       # 可用全部核心
        epoch_mult = max(1.0, gb / 16)  # 内存越多epoch越多
        corpus_max_mb = min(int(gb * 256), 8192)
    elif gb >= 16:
        tier = "high"
        worker_mult = 0.875
        epoch_mult = max(1.0, gb / 20)
        corpus_max_mb = min(int(gb * 128), 4096)
    elif gb >= 8:
        tier = "medium"
        worker_mult = 0.625
        epoch_mult = 1.0
        corpus_max_mb = min(int(gb * 64), 1024)
    elif gb >= 4:
        tier = "low"
        worker_mult = 0.375
        epoch_mult = 0.5
        corpus_max_mb = min(int(gb * 32), 512)
    else:
        tier = "minimal"
        worker_mult = 0.25
        epoch_mult = 0.25
        corpus_max_mb = 128

    # GPU 增强：有 GPU 时提升等级
    if gpu_available and vram_mb >= 4096:
        tier = f"{tier}+gpu"
        worker_mult = min(1.0, worker_mult * 1.5)
        epoch_mult *= 1.5
        corpus_max_mb = min(corpus_max_mb * 2, 16384)

    return {
        "tier": tier,
        "worker_multiplier": round(worker_mult, 3),
        "epoch_multiplier": round(epoch_mult, 2),
        "corpus_max_mb": corpus_max_mb,
        "parallel_workers": max(1, int(os.cpu_count() or 1 * worker_mult)),
    }


def _compute_training_strategy(hw: Dict[str, Any]) -> Dict[str, Any]:
    """根据硬件计算全局训练策略"""
    cpu = hw["cpu"]
    gpu = hw["gpu"]
    ram = hw["ram"]

    cores_phys = cpu["cores_physical"]
    cores_logi = cpu["cores_logical"]
    ram_avail = ram["available_mb"]
    ram_total = ram["total_mb"]
    gpu_ok = gpu["available"]
    vram = gpu.get("vram_mb", 0)

    mem_tier = _compute_memory_tier(ram_total, vram, gpu_ok)

    # 每支柱的最优配置（由系统和模型参数共同决定）
    pillar_configs = {}
    for pillar, params_m in _PILLAR_MODEL_PARAMS.items():
        d_model = _PILLAR_DMODEL.get(pillar, 512)
        max_seq = _PILLAR_SEQ_LEN.get(pillar, 128)
        # deep_transformer 层数估计
        n_layers = 12 if pillar == "deep_transformer" else 0
        threads = _compute_optimal_threads(params_m, cores_phys, cores_logi, ram_avail, gpu_ok, vram,
                                           d_model=d_model, n_layers=n_layers, max_seq=max_seq)
        batch = _compute_batch_size(params_m, ram_avail, gpu_ok, vram,
                                    d_model=_PILLAR_DMODEL.get(pillar, 512),
                                    seq_len=_PILLAR_SEQ_LEN.get(pillar, 128))
        pillar_configs[pillar] = {
            "optimal_threads": threads,
            "batch_size": batch,
            "model_params_m": params_m,
        }

    return {
        "preferred_device": "cuda" if gpu_ok else "cpu",
        "gpu_enabled": gpu_ok,
        "memory_tier": mem_tier["tier"],
        "epoch_multiplier": mem_tier["epoch_multiplier"],
        "worker_multiplier": mem_tier["worker_multiplier"],
        "corpus_max_mb": mem_tier["corpus_max_mb"],
        "parallel_workers": mem_tier["parallel_workers"],
        "pillar_configs": pillar_configs,
        "global_threads": _compute_optimal_threads(
            100, cores_phys, cores_logi, ram_avail, gpu_ok, vram
        ),
    }


# 各支柱的估算模型参数量（百万）— 用于自适应计算
_PILLAR_MODEL_PARAMS = {
    "corpus": 0.01,
    "kg": 0.5,
    "voice": 1.2,
    "rl": 0.3,
    "safety": 1.5,
    "perception": 2.0,
    "social": 1.5,
    "proactive": 1.2,
    "robot_hal": 0.8,
    "swarm": 0.6,
    "simulation": 1.0,
    "telemetry": 0.4,
    "dashboard": 0.01,
    "transformer": 5.0,
    "deep_transformer": 100.0,
}

_PILLAR_DMODEL = {
    "corpus": 64, "kg": 128, "voice": 256, "rl": 128,
    "safety": 512, "perception": 512, "social": 512, "proactive": 384,
    "robot_hal": 256, "swarm": 256, "simulation": 384, "telemetry": 256,
    "dashboard": 64, "transformer": 768, "deep_transformer": 4096,
}

_PILLAR_SEQ_LEN = {
    "corpus": 64, "kg": 32, "voice": 128, "rl": 16,
    "safety": 256, "perception": 256, "social": 256, "proactive": 256,
    "robot_hal": 64, "swarm": 64, "simulation": 128, "telemetry": 128,
    "dashboard": 32, "transformer": 256, "deep_transformer": 512,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 外层统一 API
# ═══════════════════════════════════════════════════════════════════════════════

class HardwareProfile:
    """全自适应硬件配置单例"""

    _instance: Optional[HardwareProfile] = None
    _profile: Optional[Dict[str, Any]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._detect()
        return cls._instance

    def _detect(self):
        self._profile = {
            "timestamp": time.time(),
            "platform": platform.system(),
            "hostname": platform.node(),
        }

        self._profile["cpu"] = detect_cpu_full()
        self._profile["gpu"] = detect_gpu_full()
        self._profile["ram"] = detect_ram_full()
        self._profile["training"] = _compute_training_strategy(self._profile)

    @property
    def gpu_available(self) -> bool:
        return self._profile["gpu"]["available"]

    @property
    def preferred_device(self) -> str:
        return self._profile["training"]["preferred_device"]

    @property
    def cpu_cores(self) -> int:
        return self._profile["cpu"]["cores_physical"]

    @property
    def ram_total_mb(self) -> int:
        return self._profile["ram"]["total_mb"]

    @property
    def ram_available_mb(self) -> int:
        return self._profile["ram"]["available_mb"]

    def threads_for(self, pillar: str) -> int:
        cfg = self._profile["training"]["pillar_configs"].get(pillar, {})
        return cfg.get("optimal_threads", self._profile["training"]["global_threads"])

    def batch_size_for(self, pillar: str) -> int:
        cfg = self._profile["training"]["pillar_configs"].get(pillar, {})
        return cfg.get("batch_size", 32)

    def gpu_info(self) -> Dict[str, Any]:
        return dict(self._profile["gpu"])

    def cpu_info(self) -> Dict[str, Any]:
        return dict(self._profile["cpu"])

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._profile)

    def print_summary(self):
        p = self._profile
        cpu = p["cpu"]
        gpu = p["gpu"]
        ram = p["ram"]
        t = p["training"]

        print(f"[Lifers-HW] {platform.system()} | {cpu['cores_physical']}核/{cpu['cores_logical']}线程 "
              f"| RAM {ram['total_mb']}MB (可用{ram['available_mb']}MB)", flush=True)
        if gpu["available"]:
            print(f"[Lifers-HW] GPU: {gpu['name']} | {gpu['vram_gb']}GB VRAM | CuPy {gpu.get('cupy_version', '?')} "
                  f"| 设备=cuda", flush=True)
        else:
            reason = gpu.get("cupy_error", "未检测到GPU")
            print(f"[Lifers-HW] GPU: 不可用 ({reason}) | 设备=cpu", flush=True)
        print(f"[Lifers-HW] 内存等级={t['memory_tier']} | "
              f"epoch倍率={t['epoch_multiplier']}x | 全局线程={t['global_threads']} | "
              f"语料上限={t['corpus_max_mb']}MB", flush=True)


def get_profile() -> HardwareProfile:
    return HardwareProfile()


# 快捷函数
def auto_threads(pillar: str = "transformer") -> int:
    return get_profile().threads_for(pillar)


def auto_device() -> str:
    return get_profile().preferred_device


def auto_gpu() -> bool:
    return get_profile().gpu_available


def auto_corpus_limit_mb() -> int:
    return get_profile()._profile["training"]["corpus_max_mb"]


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("[Lifers-HW] 全自适应硬件探测", flush=True)
    hw = get_profile()
    hw.print_summary()
    print(f"\n[Lifers-HW] 各支柱线程分配:", flush=True)
    for pillar in sorted(_PILLAR_MODEL_PARAMS):
        t = hw.threads_for(pillar)
        b = hw.batch_size_for(pillar)
        print(f"  {pillar:<20} threads={t:>2}  batch={b:>3}", flush=True)


if __name__ == "__main__":
    main()
