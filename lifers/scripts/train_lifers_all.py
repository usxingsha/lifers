"""
Lifers 全支柱统一训练入口
用法: python -m lifers.scripts.train_lifers_all [--pillar rl|voice|kg|corpus|all]
品牌化训练权重输出: weights/lifers_*.json
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def train_corpus():
    """扩展语料库"""
    from lifers.scripts.expand_lifers_corpus import expand_corpus
    expand_corpus(ROOT)


def train_rl():
    """训练RL策略网络"""
    from lifers.scripts.train_lifers_rl import train_lifers_rl
    episodes = int(os.environ.get("LIFERS_RL_EPISODES", "500"))
    train_lifers_rl(total_episodes=episodes, verbose=True)


def train_voice():
    """训练语音声学模型"""
    from lifers.scripts.train_lifers_voice import train_lifers_voice
    epochs = int(os.environ.get("LIFERS_VOICE_EPOCHS", "30"))
    samples = int(os.environ.get("LIFERS_VOICE_SAMPLES", "500"))
    train_lifers_voice(n_epochs=epochs, n_samples=samples, verbose=True)


def train_kg():
    """训练知识图谱嵌入"""
    from lifers.scripts.train_lifers_kg import train_lifers_kg
    epochs = int(os.environ.get("LIFERS_KG_EPOCHS", "50"))
    train_lifers_kg(n_epochs=epochs, verbose=True)


def train_all():
    """运行全部训练"""
    print("=" * 60)
    print("  Lifers 全支柱品牌化训练")
    print("=" * 60)

    t0 = time.time()

    print("\n[1/4] 语料库扩展...")
    train_corpus()

    print("\n[2/4] 知识图谱嵌入训练...")
    train_kg()

    print("\n[3/4] 语音声学模型训练...")
    train_voice()

    print("\n[4/4] RL策略网络训练...")
    train_rl()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Lifers 全支柱训练完成 总耗时: {elapsed:.1f}s")
    print(f"  权重输出目录: {ROOT / 'weights'}")
    print(f"{'=' * 60}")


PILLARS = {
    "corpus": train_corpus,
    "rl": train_rl,
    "voice": train_voice,
    "kg": train_kg,
    "all": train_all,
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Lifers 全支柱训练")
    parser.add_argument("--pillar", choices=list(PILLARS.keys()), default="all",
                        help="要训练的支柱 (默认: all)")
    args = parser.parse_args()

    fn = PILLARS.get(args.pillar, train_all)
    fn()


if __name__ == "__main__":
    main()
