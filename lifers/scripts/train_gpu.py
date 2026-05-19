#!/usr/bin/env python3
"""
Lifers 全自适应训练启动器
自动检测 GPU/CPU/RAM → 动态配置最优训练参数 → 零硬编码
Windows: 优先 GPU (CuPy) → 回退 CPU NumPy
Kali: 自动检测 CPU 最优配置
"""
import sys, os, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════════
# 全自适应硬件探测
# ═══════════════════════════════════════════════════════════════════════════════

from lifers.core.hardware_profile import get_profile, auto_gpu, auto_device
from lifers.core.hardware_profile import auto_threads, auto_corpus_limit_mb
from lifers.core.compute_backend import get_compute_backend, print_backend_info

hw = get_profile()
hw.print_summary()

# ═══════════════════════════════════════════════════════════════════════════════
# 自适应环境配置
# ═══════════════════════════════════════════════════════════════════════════════

# 全局线程数
global_threads = hw._profile["training"]["global_threads"]
os.environ.setdefault("OMP_NUM_THREADS", str(global_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(global_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(global_threads))

# 语料上限（自适应）
corpus_mb = auto_corpus_limit_mb()
if "LIFERS_CORPUS_MAX_MB" not in os.environ:
    os.environ["LIFERS_CORPUS_MAX_MB"] = str(corpus_mb)

# Epoch 倍率（根据硬件等级自动调整）
epoch_mult = hw._profile["training"]["epoch_multiplier"]
if "LIFERS_EPOCH_MULT" not in os.environ:
    os.environ["LIFERS_EPOCH_MULT"] = str(epoch_mult)

# 训练设备
device = auto_device()
print_backend_info()
if auto_gpu():
    os.environ["LIFERS_PREFER_GPU"] = "1"
print(f"\n[Lifers] 训练设备: {device.upper()}")
print(f"[Lifers] 全局线程: {global_threads} | 语料上限: {corpus_mb}MB | Epoch倍率: {epoch_mult}x")
print(f"[Lifers] GPU可用: {auto_gpu()} | 内存等级: {hw._profile['training']['memory_tier']}")

# ═══════════════════════════════════════════════════════════════════════════════
# 启动训练
# ═══════════════════════════════════════════════════════════════════════════════

from lifers.scripts.train_lifers_all import main as train_main

parser = argparse.ArgumentParser()
parser.add_argument('--pillar', default='all')
parser.add_argument('--epochs', type=int, default=0)
parser.add_argument('--no-legacy', action='store_true', help='Use guardian mode (early stopping)')
args, remaining = parser.parse_known_args()

sys.argv = ['train_gpu.py', '--pillar', args.pillar]
if not args.no_legacy:
    sys.argv.append('--legacy')

if args.epochs > 0:
    os.environ['LIFERS_FORCE_EPOCHS'] = str(args.epochs)

print(f'\n[Lifers] 启动训练: {args.pillar} pillar(s) | mode={"legacy" if not args.no_legacy else "guardian"}')
print(f'[Lifers] 各支柱自适应线程:')
for pillar in sorted(hw._profile["training"]["pillar_configs"]):
    t = hw.threads_for(pillar)
    b = hw.batch_size_for(pillar)
    print(f'  {pillar:<20} t={t:>2} batch={b:>3}')

train_main()
