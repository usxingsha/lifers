"""
Lifers 统一计算后端 — 零硬编码自动检测 GPU/CPU
Windows → 优先 CuPy GPU / Kali → 自动 NumPy CPU
单点导入，消除 12 个训练脚本中的重复检测逻辑
"""
from __future__ import annotations

import os
import sys
import platform
from typing import Any, Optional, Tuple

_CPU_NP: Any = None
_BACKEND: Any = None
_DEVICE: str = "cpu"
_GPU_INFO: Optional[dict] = None
_INITIALIZED: bool = False
_DEFAULT_FLOAT: Any = None
_DEFAULT_INT: Any = None


def _init_cpu_numpy():
    global _CPU_NP
    if _CPU_NP is None:
        import numpy as _cpu_np
        _CPU_NP = _cpu_np


def get_cpu_numpy():
    """始终返回 CPU NumPy（用于文件 I/O 等操作）"""
    _init_cpu_numpy()
    return _CPU_NP


def get_compute_backend(prefer_gpu: Optional[bool] = None) -> Tuple[Any, str, Optional[dict]]:
    """
    返回 (计算模块, 设备名, GPU信息)
    - Windows: 自动尝试 CuPy GPU，失败回退 NumPy CPU
    - Kali/Linux: 检测是否有 GPU，有则用 CuPy，无则用 NumPy
    - prefer_gpu=None → 自动检测
    - prefer_gpu=True → 强制 GPU，不可用时抛出异常
    - prefer_gpu=False → 强制 CPU NumPy
    """
    global _BACKEND, _DEVICE, _GPU_INFO, _INITIALIZED, _DEFAULT_FLOAT, _DEFAULT_INT

    if _INITIALIZED and prefer_gpu is None:
        return _BACKEND, _DEVICE, _GPU_INFO

    _init_cpu_numpy()

    # 确定是否尝试 GPU
    if prefer_gpu is None:
        try_gpu = os.environ.get("LIFERS_PREFER_GPU", "").strip()
        if try_gpu.lower() in ("0", "false", "no", "off"):
            try_gpu_flag = False
        elif try_gpu.lower() in ("1", "true", "yes", "on"):
            try_gpu_flag = True
        else:
            try_gpu_flag = True  # 默认尝试 GPU
    else:
        try_gpu_flag = prefer_gpu

    if try_gpu_flag:
        gpu_ok, backend, gpu_info = _try_init_gpu()
        if gpu_ok:
            _BACKEND = backend
            _DEVICE = "cuda"
            _GPU_INFO = gpu_info
            _INITIALIZED = True
            _DEFAULT_FLOAT = backend.float32
            _DEFAULT_INT = backend.intp
            return _BACKEND, _DEVICE, _GPU_INFO
        elif prefer_gpu is True:
            raise RuntimeError(f"GPU 不可用: {gpu_info.get('error', 'unknown')}")

    # CPU 回退
    _BACKEND = _CPU_NP
    _DEVICE = "cpu"
    _GPU_INFO = {"available": False, "device": "cpu"}
    _INITIALIZED = True
    _DEFAULT_FLOAT = _CPU_NP.float64
    _DEFAULT_INT = _CPU_NP.intp
    return _BACKEND, _DEVICE, _GPU_INFO


def _try_init_gpu() -> Tuple[bool, Any, dict]:
    """多层验证 CuPy GPU 是否可用"""
    gpu_info: dict = {
        "available": False,
        "device": "cpu",
        "name": "unknown",
        "vram_mb": 0,
        "vram_gb": 0.0,
        "cupy_ready": False,
    }

    # 第1步: 硬件探测 nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            gpu_info["name"] = parts[0].strip()
            gpu_info["vram_mb"] = int(float(parts[1].strip()))
            gpu_info["vram_gb"] = round(gpu_info["vram_mb"] / 1024, 1)
    except Exception:
        pass

    # 第2步: CuPy 运行时验证 (4层测试)
    try:
        import cupy as cp
        a = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        r = cp.random.randn(10).astype(cp.float32)
        dot = cp.dot(a, a.astype(cp.float32))
        cp.cuda.Stream.null.synchronize()

        gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        vram = cp.cuda.runtime.getDeviceProperties(0)["totalGlobalMem"] / 1024**3

        gpu_info["available"] = True
        gpu_info["device"] = "cuda"
        gpu_info["cupy_ready"] = True
        gpu_info["name"] = gpu_name
        gpu_info["vram_gb"] = round(vram, 1)
        gpu_info["vram_mb"] = int(vram * 1024)
        gpu_info["cupy_version"] = cp.__version__
        gpu_info["verified"] = True

        return True, cp, gpu_info
    except Exception as e:
        gpu_info["error"] = str(e)[:200]
        gpu_info["cupy_ready"] = False
        return False, None, gpu_info


def get_device() -> str:
    get_compute_backend()
    return _DEVICE


def is_gpu() -> bool:
    return get_device() == "cuda"


def get_default_float():
    get_compute_backend()
    return _DEFAULT_FLOAT


def get_default_int():
    get_compute_backend()
    return _DEFAULT_INT


def to_cpu(arr: Any) -> Any:
    """将 GPU 数组转换为 CPU NumPy 数组（用于文件 I/O）"""
    if hasattr(arr, "get"):
        return arr.get()
    return arr


def to_device(arr: Any, np_module: Any = None) -> Any:
    """将 CPU 数组传输到当前计算设备"""
    if np_module is None:
        np_module = _BACKEND if _INITIALIZED else None
        if np_module is None:
            _, _, _ = get_compute_backend()
            np_module = _BACKEND
    if hasattr(np_module, "asarray"):
        return np_module.asarray(arr)
    return arr


def backend_summary() -> str:
    np_mod, device, gpu_info = get_compute_backend()
    if device == "cuda" and gpu_info:
        return (f"GPU: {gpu_info.get('name', '?')} | "
                f"{gpu_info.get('vram_gb', 0)}GB VRAM | "
                f"CuPy {gpu_info.get('cupy_version', '?')}")
    return f"CPU: {platform.processor() or platform.machine()} | NumPy"


def print_backend_info():
    print(f"[compute-backend] 设备: {get_device().upper()} | {backend_summary()}", flush=True)


def _reset_cache():
    """测试用：重置缓存"""
    global _BACKEND, _DEVICE, _GPU_INFO, _INITIALIZED, _DEFAULT_FLOAT, _DEFAULT_INT
    _BACKEND = None
    _DEVICE = "cpu"
    _GPU_INFO = None
    _INITIALIZED = False
    _DEFAULT_FLOAT = None
    _DEFAULT_INT = None
