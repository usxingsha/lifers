"""
Lifers 硬件与软件环境探测
自动检测 CPU/GPU/RAM/Disk/OS/Python 环境
输出标准化能力配置文件 → weights/lifers_hardware_profile.json
"""

from __future__ import annotations

import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

ROOT = Path(__file__).resolve().parent.parent.parent


def _cpu_info() -> Dict[str, Any]:
    """CPU 探测"""
    info = {
        "cores_physical": multiprocessing.cpu_count(),
        "cores_logical": os.cpu_count() or 1,
        "architecture": platform.machine(),
        "processor": platform.processor() or "unknown",
    }
    # 尝试获取 CPU 频率 (Windows)
    try:
        import ctypes
        freq = ctypes.c_uint64(0)
        ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(freq))
        info["freq_mhz_approx"] = freq.value // 1_000_000
    except Exception:
        info["freq_mhz_approx"] = 0
    return info


def _gpu_info() -> Dict[str, Any]:
    """GPU 探测 — CUDA / OpenCL / Vulkan"""
    info: Dict[str, Any] = {
        "available": False,
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_name": "none",
        "vram_mb": 0,
    }
    # 尝试 NVIDIA CUDA
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            info["gpu_name"] = parts[0].strip()
            info["vram_mb"] = int(float(parts[1].strip()))
            info["cuda_available"] = True
            info["available"] = True
            # 统计 GPU 数量
            count = len([l for l in result.stdout.strip().split("\n") if l.strip()])
            info["gpu_count"] = max(1, count)
    except Exception:
        pass

    if not info["cuda_available"]:
        # Windows 尝试 DXGI / WMI
        try:
            result = subprocess.run(
                ["wmic", "path", "Win32_VideoController", "get", "name,AdapterRAM"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                lines = [l.strip() for l in result.stdout.split("\n") if l.strip() and "AdapterRAM" not in l]
                if lines:
                    info["gpu_count"] = len(lines)
                    info["gpu_name"] = lines[0].rsplit("  ", 1)[0].strip() if "  " in lines[0] else lines[0]
                    try:
                        vram_bytes = int(lines[0].rsplit("  ", 1)[-1].strip())
                        info["vram_mb"] = vram_bytes // (1024 * 1024)
                    except ValueError:
                        pass
        except Exception:
            pass

    return info


def _ram_info() -> Dict[str, Any]:
    """内存探测"""
    import psutil
    mem = psutil.virtual_memory()
    return {
        "total_mb": mem.total // (1024 * 1024),
        "available_mb": mem.available // (1024 * 1024),
        "used_percent": mem.percent,
    }


def _disk_info() -> Dict[str, Any]:
    """磁盘探测"""
    usage = shutil.disk_usage(ROOT)
    return {
        "total_gb": usage.total // (1024 ** 3),
        "free_gb": usage.free // (1024 ** 3),
        "used_gb": usage.used // (1024 ** 3),
        "root_path": str(ROOT),
    }


def _os_info() -> Dict[str, Any]:
    """操作系统探测"""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "hostname": platform.node(),
        "python_version": sys.version,
        "python_impl": platform.python_implementation(),
    }


def _python_packages() -> Dict[str, Any]:
    """Python 包检测 — 自动发现新包"""
    pkgs: Dict[str, Any] = {}
    # 核心包
    for pkg in ["numpy", "scipy", "psutil", "json", "pathlib", "re", "hashlib",
                "ssl", "socket", "http", "urllib", "concurrent.futures",
                "asyncio", "threading", "multiprocessing"]:
        try:
            __import__(pkg)
            pkgs[pkg] = True
        except ImportError:
            pkgs[pkg] = False

    # 可选加速包
    for pkg in ["torch", "tensorflow", "jax", "cupy", "numba", "pandas",
                "sklearn", "matplotlib", "scipy.ndimage", "cv2"]:
        try:
            __import__(pkg.split(".")[0])
            pkgs[pkg] = True
        except ImportError:
            pkgs[pkg] = False

    # numpy 版本
    try:
        import numpy as np
        pkgs["numpy_version"] = np.__version__
    except Exception:
        pass

    # pip可用包列表（自动发现新的AI/ML包）
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            installed = json.loads(result.stdout)
            ml_packages = [p["name"] for p in installed
                          if any(kw in p["name"].lower() for kw in
                                 ["torch", "tensorflow", "jax", "cuda", "onnx",
                                  "transformers", "sentence", "tokenizers",
                                  "diffusers", "accelerate", "triton", "vllm",
                                  "langchain", "llama", "openai", "anthropic"])]
            pkgs["ml_packages_available"] = ml_packages
    except Exception:
        pkgs["ml_packages_available"] = []

    return pkgs


def _network_info() -> Dict[str, Any]:
    """网络能力探测"""
    import socket
    info = {
        "hostname": socket.gethostname(),
        "has_ipv4": False,
        "has_ipv6": False,
    }
    try:
        socket.getaddrinfo("www.baidu.com", 80, socket.AF_INET)
        info["has_ipv4"] = True
    except Exception:
        pass
    try:
        socket.getaddrinfo("www.baidu.com", 80, socket.AF_INET6)
        info["has_ipv6"] = True
    except Exception:
        pass
    return info


def _new_hardware_check(prev_profile: Optional[Dict] = None) -> Dict[str, Any]:
    """检测自上次探测以来的硬件变化 (自动发现新硬件)"""
    if prev_profile is None:
        prev_path = ROOT / "weights" / "lifers_hardware_profile.json"
        if prev_path.exists():
            with open(prev_path, "r", encoding="utf-8") as f:
                prev_profile = json.load(f)
        else:
            prev_profile = {}

    changes = {"new_hardware": [], "removed_hardware": [], "changed_hardware": []}

    # 比较GPU
    prev_gpu = prev_profile.get("gpu", {})
    if prev_gpu:
        prev_name = prev_gpu.get("gpu_name", "")
        prev_count = prev_gpu.get("gpu_count", 0)
    else:
        prev_name = ""
        prev_count = 0

    curr_gpu = _gpu_info()
    if curr_gpu.get("gpu_name", "") != prev_name:
        changes["changed_hardware"].append(f"GPU: {prev_name} -> {curr_gpu.get('gpu_name')}")
    if curr_gpu.get("gpu_count", 0) != prev_count:
        if curr_gpu.get("gpu_count", 0) > prev_count:
            changes["new_hardware"].append("新增GPU")
        else:
            changes["removed_hardware"].append("GPU减少")

    # 比较RAM
    prev_ram = prev_profile.get("ram", {}).get("total_mb", 0)
    curr_ram = _ram_info().get("total_mb", 0)
    if abs(curr_ram - prev_ram) > 1024:  # 超过1GB变化
        changes["changed_hardware"].append(f"RAM: {prev_ram}MB -> {curr_ram}MB")

    # 比较Python包
    prev_pkgs = set(prev_profile.get("packages", {}).keys())
    curr_pkgs = set(_python_packages().keys())
    new_pkgs = curr_pkgs - prev_pkgs
    if new_pkgs:
        changes["new_hardware"].append(f"新增Python包: {', '.join(sorted(new_pkgs)[:5])}")

    return changes


def _training_capability(hw: Dict[str, Any]) -> Dict[str, Any]:
    """根据硬件计算训练能力"""
    cpu = hw.get("cpu", {})
    gpu = hw.get("gpu", {})
    ram = hw.get("ram", {})

    cores = cpu.get("cores_physical", 1)
    ram_mb = ram.get("total_mb", 4096)
    vram_mb = gpu.get("vram_mb", 0)
    has_cuda = gpu.get("cuda_available", False)

    cap = {
        "can_train_all": ram_mb >= 2048,
        "max_parallel_workers": max(1, cores - 1),
        "recommended_batch_size": 32,
        "recommended_epochs_multiplier": 1.0,
        "can_use_gpu": has_cuda and vram_mb >= 512,
        "preferred_device": "cpu",
        "memory_tier": "low",
    }

    # 根据内存分级
    if ram_mb >= 16384:
        cap["memory_tier"] = "high"
        cap["recommended_batch_size"] = 256
        cap["recommended_epochs_multiplier"] = 2.0
        cap["max_parallel_workers"] = cores
    elif ram_mb >= 8192:
        cap["memory_tier"] = "medium"
        cap["recommended_batch_size"] = 128
        cap["recommended_epochs_multiplier"] = 1.5
        cap["max_parallel_workers"] = max(2, cores - 2)

    # GPU 优先
    if has_cuda and vram_mb >= 2048:
        cap["preferred_device"] = "cuda"
        cap["recommended_batch_size"] = min(cap["recommended_batch_size"] * 2, 512)

    # 低配限制
    if not cap["can_train_all"]:
        cap["recommended_batch_size"] = 8
        cap["max_parallel_workers"] = 1

    return cap


def probe_hardware(output_path: Optional[Path] = None, verbose: bool = True) -> Dict[str, Any]:
    """执行完整硬件探测，返回能力配置"""
    if output_path is None:
        output_path = ROOT / "weights" / "lifers_hardware_profile.json"

    hw = {}
    if verbose:
        print("[Lifers-Probe] 探测 CPU...")
    hw["cpu"] = _cpu_info()

    if verbose:
        print("[Lifers-Probe] 探测 GPU...")
    hw["gpu"] = _gpu_info()

    if verbose:
        print("[Lifers-Probe] 探测 RAM...")
    hw["ram"] = _ram_info()

    if verbose:
        print("[Lifers-Probe] 探测 Disk...")
    hw["disk"] = _disk_info()

    if verbose:
        print("[Lifers-Probe] 探测 Network...")
    hw["network"] = _network_info()

    if verbose:
        print("[Lifers-Probe] 探测 OS...")
    hw["os"] = _os_info()

    if verbose:
        print("[Lifers-Probe] 探测 Python 环境...")
    hw["packages"] = _python_packages()

    if verbose:
        print("[Lifers-Probe] 检测硬件变化...")
    hw["hardware_changes"] = _new_hardware_check()

    if verbose:
        print("[Lifers-Probe] 计算训练能力...")
    hw["training"] = _training_capability(hw)

    hw["brand"] = "Lifers Hardware Profile"
    hw["version"] = 1
    hw["timestamp"] = time.time()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(hw, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"[Lifers-Probe] 配置已保存 → {output_path}")
        _print_summary(hw)

    return hw


def load_hardware_profile(path: Optional[Path] = None) -> Dict[str, Any]:
    """加载已保存的硬件配置"""
    if path is None:
        path = ROOT / "weights" / "lifers_hardware_profile.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _print_summary(hw: Dict[str, Any]):
    t = hw.get("training", {})
    cpu = hw.get("cpu", {})
    ram = hw.get("ram", {})
    gpu = hw.get("gpu", {})

    print(f"  CPU: {cpu.get('cores_physical')}核/{cpu.get('cores_logical')}线程  {cpu.get('architecture')}")
    print(f"  GPU: {gpu.get('gpu_name')}  VRAM={gpu.get('vram_mb')}MB  CUDA={'yes' if gpu.get('cuda_available') else 'no'}")
    print(f"  RAM: {ram.get('total_mb')}MB (可用{ram.get('available_mb')}MB)")
    print(f"  训练能力: device={t.get('preferred_device')}  tier={t.get('memory_tier')}  "
          f"batch={t.get('recommended_batch_size')}  workers={t.get('max_parallel_workers')}")


def main():
    print("[Lifers-Probe] 硬件环境探测")
    probe_hardware(verbose=True)


if __name__ == "__main__":
    main()
