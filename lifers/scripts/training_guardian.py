"""
Lifers 训练守护者 — 故障模拟 + 训练监控 + 断点恢复
模拟: 网络中断、断电、延迟、无限循环、死锁、超时
功能: 心跳检测、超时熔断、自动checkpoint、崩溃恢复
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent

# ═══════════════════════════════════════════════════════════════════════════════
# 故障类型定义
# ═══════════════════════════════════════════════════════════════════════════════

class FailureType:
    NETWORK_OUTAGE = "network_outage"       # 网络中断
    POWER_LOSS = "power_loss"               # 断电
    HIGH_LATENCY = "high_latency"           # 高延迟
    INFINITE_LOOP = "infinite_loop"         # 无限循环
    DEADLOCK = "deadlock"                   # 死锁
    TIMEOUT = "timeout"                     # 超时
    MEMORY_OOM = "memory_oom"               # 内存溢出
    DISK_FULL = "disk_full"                 # 磁盘满
    GRADIENT_EXPLOSION = "gradient_explosion"  # 梯度爆炸
    STUCK_PROGRESS = "stuck_progress"       # 进度卡死


# ═══════════════════════════════════════════════════════════════════════════════
# 故障模拟器
# ═══════════════════════════════════════════════════════════════════════════════

class FailureSimulator:
    """故障注入 — 模拟各类训练中断"""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.active_failures: Dict[str, bool] = {}
        self.failure_stats: Dict[str, int] = {}
        for name in dir(FailureType):
            if not name.startswith("_"):
                self.failure_stats[getattr(FailureType, name)] = 0

    def inject(self, failure_type: str) -> bool:
        """注入指定故障，返回是否触发"""
        self.failure_stats[failure_type] = self.failure_stats.get(failure_type, 0) + 1
        self.active_failures[failure_type] = True

        if failure_type == FailureType.NETWORK_OUTAGE:
            print("[Lifers-Guardian] [!] 模拟: 网络中断")
            time.sleep(2.0)
            self.active_failures[failure_type] = False
            print("[Lifers-Guardian] [OK] 网络恢复")
            return True

        elif failure_type == FailureType.POWER_LOSS:
            print("[Lifers-Guardian] [!] 模拟: 断电! 保存checkpoint后退出...")
            time.sleep(0.5)
            self.active_failures[failure_type] = False
            return True

        elif failure_type == FailureType.HIGH_LATENCY:
            delay = self.rng.uniform(1.0, 5.0)
            print(f"[Lifers-Guardian] [!] 模拟: 高延迟 {delay:.1f}s")
            time.sleep(delay)
            self.active_failures[failure_type] = False
            return True

        elif failure_type == FailureType.INFINITE_LOOP:
            print("[Lifers-Guardian] [!] 模拟: 无限循环 — 检测并中断")
            time.sleep(0.3)
            self.active_failures[failure_type] = False
            raise RuntimeError("[Lifers-Guardian] 检测到无限循环，已中断")

        elif failure_type == FailureType.DEADLOCK:
            print("[Lifers-Guardian] [!] 模拟: 死锁 — 检测并中断")
            time.sleep(0.3)
            self.active_failures[failure_type] = False
            raise RuntimeError("[Lifers-Guardian] 检测到死锁，已中断")

        elif failure_type == FailureType.TIMEOUT:
            print("[Lifers-Guardian] [!] 模拟: 操作超时")
            self.active_failures[failure_type] = False
            raise TimeoutError("[Lifers-Guardian] 操作超时")

        elif failure_type == FailureType.MEMORY_OOM:
            print("[Lifers-Guardian] [!] 模拟: 内存溢出")
            self.active_failures[failure_type] = False
            raise MemoryError("[Lifers-Guardian] 内存不足")

        elif failure_type == FailureType.DISK_FULL:
            print("[Lifers-Guardian] [!] 模拟: 磁盘满")
            self.active_failures[failure_type] = False
            raise OSError("[Lifers-Guardian] 磁盘空间不足")

        elif failure_type == FailureType.GRADIENT_EXPLOSION:
            print("[Lifers-Guardian] [!] 模拟: 梯度爆炸 — 自动裁剪")
            self.active_failures[failure_type] = False
            return True  # 梯度爆炸可自动处理

        elif failure_type == FailureType.STUCK_PROGRESS:
            print("[Lifers-Guardian] [!] 模拟: 训练进度卡死 — 触发早停")
            self.active_failures[failure_type] = False
            return True

        return False

    def random_inject(self, probability: float = 0.1) -> Optional[str]:
        """随机注入故障"""
        if self.rng.random() < probability:
            failures = [getattr(FailureType, n) for n in dir(FailureType) if not n.startswith("_")]
            ft = self.rng.choice(failures)
            self.inject(ft)
            return ft
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 训练守护者
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingGuardian:
    """训练守护 — 心跳监控 + 超时熔断 + checkpoint管理"""

    def __init__(
        self,
        pillar_name: str = "unknown",
        timeout_s: float = 300.0,
        heartbeat_interval_s: float = 10.0,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.pillar_name = pillar_name
        self.timeout_s = timeout_s
        self.heartbeat_interval_s = heartbeat_interval_s
        self.checkpoint_dir = checkpoint_dir or (ROOT / "weights" / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._last_heartbeat = time.time()
        self._start_time = time.time()
        self._running = False
        self._paused = False
        self._stop_requested = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._checkpoints: List[Path] = []
        self._failure_simulator = FailureSimulator()

        # 统计
        self.stats = {
            "elapsed_s": 0.0,
            "heartbeat_count": 0,
            "recoveries": 0,
            "timeouts": 0,
            "failures_handled": 0,
            "checkpoints_saved": 0,
        }

    @property
    def simulator(self) -> FailureSimulator:
        return self._failure_simulator

    def heartbeat(self):
        """发送心跳信号"""
        with self._lock:
            self._last_heartbeat = time.time()
            self.stats["heartbeat_count"] += 1

    def checkpoint(self, data: Dict[str, Any], name: Optional[str] = None):
        """保存检查点"""
        if name is None:
            name = f"{self.pillar_name}_ckpt_{int(time.time())}"
        path = self.checkpoint_dir / f"{name}.json"
        payload = {
            "pillar": self.pillar_name,
            "timestamp": time.time(),
            "stats": dict(self.stats),
            "data": data,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with self._lock:
            self._checkpoints.append(path)
            self.stats["checkpoints_saved"] += 1
            # 保留最近5个检查点
            while len(self._checkpoints) > 5:
                old = self._checkpoints.pop(0)
                try:
                    old.unlink(missing_ok=True)
                except Exception:
                    pass
        return path

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """加载最近的检查点"""
        ckpts = sorted(self.checkpoint_dir.glob(f"{self.pillar_name}_ckpt_*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not ckpts:
            return None
        with open(ckpts[0], "r", encoding="utf-8") as f:
            return json.load(f)

    def is_stale(self) -> bool:
        """检查心跳是否过期（可能卡死）"""
        return time.time() - self._last_heartbeat > self.timeout_s

    def request_stop(self):
        """请求停止训练"""
        with self._lock:
            self._stop_requested = True

    def should_stop(self) -> bool:
        return self._stop_requested

    def _monitor_loop(self):
        """后台监控线程"""
        while self._running:
            time.sleep(self.heartbeat_interval_s)
            if not self._running:
                break
            if self._paused:
                continue

            # 超时检测
            if self.is_stale():
                self.stats["timeouts"] += 1
                print(f"[Lifers-Guardian] [WARN] {self.pillar_name} 训练超时 ({self.timeout_s}s无心跳)")
                self.request_stop()

            # 随机故障注入 (用于测试)
            fault = self._failure_simulator.random_inject(probability=0.0)  # 默认关闭，测试时提高
            if fault:
                self.stats["failures_handled"] += 1

    def start(self):
        """启动守护"""
        self._running = True
        self._start_time = time.time()
        self._last_heartbeat = time.time()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self):
        """停止守护"""
        self._running = False
        self.stats["elapsed_s"] = time.time() - self._start_time
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False
        self._last_heartbeat = time.time()


# ═══════════════════════════════════════════════════════════════════════════════
# 安全训练上下文
# ═══════════════════════════════════════════════════════════════════════════════

@contextmanager
def safe_training(pillar_name: str, timeout_s: float = 300.0, enable_failure_sim: bool = False):
    """安全训练上下文管理器 — 自动处理故障和恢复"""
    guardian = TrainingGuardian(pillar_name=pillar_name, timeout_s=timeout_s)
    guardian.start()

    try:
        yield guardian
    except (RuntimeError, TimeoutError, MemoryError, OSError) as e:
        print(f"[Lifers-Guardian] [FAIL] {pillar_name} 训练异常: {e}")
        guardian.stats["failures_handled"] += 1

        # 尝试从checkpoint恢复
        ckpt = guardian.load_latest_checkpoint()
        if ckpt:
            print(f"[Lifers-Guardian] [RECOVER] 从checkpoint恢复: {ckpt.get('timestamp', '?')}")
            guardian.stats["recoveries"] += 1
        traceback.print_exc()

    except KeyboardInterrupt:
        print(f"\n[Lifers-Guardian] [WARN] {pillar_name} 训练被用户中断")
        guardian.stats["failures_handled"] += 1

    finally:
        guardian.stop()
        print(f"[Lifers-Guardian] {pillar_name} 完成 "
              f"耗时={guardian.stats['elapsed_s']:.1f}s "
              f"恢复={guardian.stats['recoveries']} "
              f"失败处理={guardian.stats['failures_handled']}")


# ═══════════════════════════════════════════════════════════════════════════════
# 训练拆分与合并
# ═══════════════════════════════════════════════════════════════════════════════

def split_training_weights(pillar: str, weights: Dict[str, Any], n_splits: int = 2) -> List[Dict[str, Any]]:
    """将训练权重拆分为多份（用于分布式训练）"""
    splits = [{} for _ in range(n_splits)]

    def _chunk(arr, n):
        """均匀分块，余数分配给最后一块"""
        base = len(arr) // n
        remainder = len(arr) % n
        chunks = []
        start = 0
        for i in range(n):
            size = base + (1 if i < remainder else 0)
            chunks.append(arr[start:start + size])
            start += size
        return chunks

    # 按参数维度拆分
    for key in ["W1", "W2", "Wy"]:
        if key in weights and isinstance(weights[key], list):
            arr = weights[key]
            if isinstance(arr[0], list):  # 2D
                for i, chunk in enumerate(_chunk(arr, n_splits)):
                    splits[i][key] = chunk
            else:  # 1D
                for i, chunk in enumerate(_chunk(arr, n_splits)):
                    splits[i][key] = chunk

    # 标量参数复制
    for key in ["b", "b1", "b2", "by", "version", "input_dim", "hidden_dim", "n_classes", "vocab", "action_dim", "state_dim", "embedding_dim", "n_entities", "n_relations"]:
        if key in weights:
            for s in splits:
                s[key] = weights[key]

    # 元数据
    for i, s in enumerate(splits):
        s["brand"] = f"{weights.get('brand', 'Unknown')} [split {i+1}/{n_splits}]"
        s["split_index"] = i
        s["split_total"] = n_splits

    return splits


def merge_training_weights(splits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """合并分布式训练的权重"""
    if not splits:
        return {}
    if len(splits) == 1:
        return dict(splits[0])

    merged = {}
    for key in splits[0]:
        if key in ("split_index", "split_total", "brand"):
            continue

        vals = [s[key] for s in splits if key in s]
        if not vals:
            continue

        if isinstance(vals[0], list):
            if isinstance(vals[0][0], list):  # 2D — 行拼接
                merged[key] = []
                for v in vals:
                    merged[key].extend(v)
            else:  # 1D — 拼接
                merged[key] = []
                for v in vals:
                    merged[key].extend(v)
        else:
            merged[key] = vals[0]  # 标量取第一个

    merged["brand"] = splits[0].get("brand", "Unknown").replace(" [split 1/", "")
    merged["merged_from"] = len(splits)
    return merged


def save_split_weights(splits: List[Dict[str, Any]], pillar: str):
    """保存拆分后的权重"""
    out_dir = ROOT / "weights" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, s in enumerate(splits):
        p = out_dir / f"lifers_{pillar}_split{i+1}of{len(splits)}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
        paths.append(p)
    return paths


def load_and_merge_splits(pillar: str) -> Optional[Dict[str, Any]]:
    """加载拆分权重并合并"""
    split_dir = ROOT / "weights" / "splits"
    if not split_dir.exists():
        return None
    files = sorted(split_dir.glob(f"lifers_{pillar}_split*.json"))
    if not files:
        return None
    splits = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            splits.append(json.load(fh))
    return merge_training_weights(splits)


# ═══════════════════════════════════════════════════════════════════════════════
# 训练稳定性检查
# ═══════════════════════════════════════════════════════════════════════════════

def check_training_stability(loss_history: List[float], window: int = 20, threshold: float = 0.001) -> Dict[str, Any]:
    """检查训练稳定性"""
    if len(loss_history) < window * 2:
        return {"stable": True, "issue": None, "suggestion": None}

    recent = loss_history[-window:]
    earlier = loss_history[-window * 2:-window]
    avg_recent = sum(recent) / len(recent)
    avg_earlier = sum(earlier) / len(earlier)

    result = {"stable": True, "issue": None, "suggestion": None}

    # 检测梯度爆炸
    if avg_recent > avg_earlier * 5:
        result["stable"] = False
        result["issue"] = "gradient_explosion"
        result["suggestion"] = "降低学习率或应用梯度裁剪"
    # 检测进度卡死
    elif abs(avg_recent - avg_earlier) / max(abs(avg_earlier), 1e-8) < threshold:
        result["stable"] = False
        result["issue"] = "stuck_progress"
        result["suggestion"] = "调整学习率或增加数据多样性"
    # 检测过拟合趋势
    elif avg_recent < avg_earlier * 0.01:
        result["stable"] = False
        result["issue"] = "overfitting_risk"
        result["suggestion"] = "考虑早停或增加正则化"

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

def run_failure_drill():
    """故障演练 — 测试所有故障类型的处理"""
    print("=" * 60)
    print("  Lifers 故障演练")
    print("=" * 60)

    sim = FailureSimulator()
    failure_types = [getattr(FailureType, n) for n in dir(FailureType) if not n.startswith("_")]

    results = []
    for ft in failure_types:
        print(f"\n--- 测试: {ft} ---")
        try:
            sim.inject(ft)
            results.append((ft, "handled"))
        except Exception as e:
            results.append((ft, f"caught: {e}"))

    print("\n" + "=" * 60)
    print("  演练结果:")
    for ft, result in results:
        status = "[OK]" if result == "handled" else "[WARN]"
        print(f"  {status} {ft}: {result}")
    print(f"  统计: {sim.failure_stats}")
    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Lifers 训练守护者")
    parser.add_argument("--drill", action="store_true", help="运行故障演练")
    parser.add_argument("--pillar", type=str, default="all", help="监控的支柱名称")
    parser.add_argument("--timeout", type=int, default=300, help="超时时间(秒)")
    args = parser.parse_args()

    if args.drill:
        run_failure_drill()
    else:
        print(f"[Lifers-Guardian] 守护者就绪 pillar={args.pillar} timeout={args.timeout}s")


if __name__ == "__main__":
    main()
