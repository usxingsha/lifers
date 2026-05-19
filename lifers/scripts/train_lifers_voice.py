"""
Lifers Voice 训练 — LSTM声学模型 + TTS音色学习
品牌化权重: weights/lifers_voice_acoustic.json
纯numpy DSP + 合成训练数据
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as cpu_np
from lifers.core.compute_backend import get_compute_backend
np, _DEVICE, _GPU_INFO = get_compute_backend()

ROOT = Path(__file__).resolve().parent.parent.parent


# ═══════════════════════════════════════════════════════════════════════════════
# Lifers 合成音频数据生成器
# ═══════════════════════════════════════════════════════════════════════════════

_PHONEMES = [
    "aa", "ae", "ah", "ao", "aw", "ay", "b", "ch", "d", "dh", "eh", "er",
    "ey", "f", "g", "hh", "ih", "iy", "jh", "k", "l", "m", "n", "ng",
    "ow", "oy", "p", "r", "s", "sh", "t", "th", "uh", "uw", "v", "w",
    "y", "z", "zh", "sil",
]
_P2I = {p: i for i, p in enumerate(_PHONEMES)}
N_PHONEMES = len(_PHONEMES)

# 共振峰频率参考 (F1, F2, F3) — 元音声学特征
_FORMANT_MAP = {
    "iy": (270, 2290, 3010), "ih": (390, 1990, 2550), "ey": (530, 1840, 2480),
    "eh": (530, 1840, 2480), "ae": (660, 1720, 2410), "aa": (730, 1090, 2440),
    "ao": (570, 840, 2410), "ow": (440, 1020, 2240), "uh": (440, 1020, 2240),
    "uw": (300, 870, 2240), "ah": (640, 1190, 2390), "er": (490, 1350, 1690),
    "aw": (660, 1100, 2300), "ay": (640, 1400, 2300), "oy": (440, 1100, 2300),
}


def _generate_synthetic_phoneme_audio(phoneme: str, sr: int = 16000, duration_ms: int = 200) -> np.ndarray:
    """共振峰合成音素音频 — 更长时长、真实滤波、辅音区分"""
    n_samples = int(sr * duration_ms / 1000)
    t = np.arange(n_samples) / sr
    f1, f2, f3 = _FORMANT_MAP.get(phoneme, (500, 1500, 2500))
    rng = np.random.RandomState(hash(phoneme) % 2**31)

    # 静音
    if phoneme in ("sil",):
        return (rng.randn(n_samples) * 0.005).astype(np.float32)

    # 基频 F0
    if phoneme in ("b", "d", "g", "p", "t", "k"):
        f0 = rng.uniform(100, 140)  # 塞音低基频
    elif phoneme in _VOWELS:
        f0 = rng.uniform(140, 260)
    elif phoneme in ("s", "sh", "f", "th", "ch", "z", "zh", "v", "hh"):
        f0 = 0  # 清擦音无基频
    elif phoneme in ("m", "n", "ng", "l", "r", "w", "y"):
        f0 = rng.uniform(120, 200)  # 鼻音/流音
    else:
        f0 = 160

    # 声门脉冲序列
    if f0 > 0:
        pulse_train = np.zeros(n_samples, dtype=np.float32)
        period = int(sr / f0)
        for i in range(0, n_samples, period):
            end = min(i + 10, n_samples)
            pulse_train[i:end] = np.hanning(end - i)
        source = pulse_train
    else:
        # 清音: 噪声源
        source = rng.randn(n_samples).astype(np.float32) * 0.3
        # 高频加重 (擦音特征)
        source = np.convolve(source, [1, -0.95], mode='same')

    # 共振峰 IIR 滤波器 (简化: 每共振峰用带通)
    bw1, bw2, bw3 = 60, 90, 120  # 共振峰带宽
    audio = np.zeros(n_samples, dtype=np.float32)
    for fi, bw, amp in [(f1, bw1, 0.5), (f2, bw2, 0.35), (f3, bw3, 0.2)]:
        # 用正弦近似带通 (实际IIR太重, 这里用调幅正弦模拟共振峰响应)
        envelope = np.exp(-t * bw * 3)  # 衰减包络
        carrier = np.sin(2 * np.pi * fi * t + rng.uniform(0, np.pi * 2))
        audio += amp * carrier * (0.5 + 0.5 * source)

    # 辅音附加特征
    if phoneme in ("p", "t", "k"):  # 塞音 — 爆破开始
        burst = np.exp(-t * 500) * rng.randn(n_samples).astype(np.float32)
        audio[:n_samples//4] += burst[:n_samples//4] * 0.5
    elif phoneme in ("b", "d", "g"):  # 浊塞音 — 弱爆破
        burst = np.exp(-t * 400) * rng.randn(n_samples).astype(np.float32)
        audio[:n_samples//3] += burst[:n_samples//3] * 0.3
    elif phoneme in ("s", "sh", "ch", "z", "zh"):  # 咝音 — 高频噪声
        noise_hi = np.convolve(rng.randn(n_samples), [1, -0.97], mode='same')
        audio += noise_hi.astype(np.float32) * 0.15 * (0.5 + 0.5 * source)
    elif phoneme in ("m", "n", "ng"):  # 鼻音 — 低频增强
        audio += np.sin(2 * np.pi * 250 * t) * source * 0.2

    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8
    return audio.astype(np.float32)


_VOWELS = {"aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"}
_CONSONANTS = set(_PHONEMES) - _VOWELS - {"sil"}


def _mfcc(audio: np.ndarray, sr: int = 16000, n_mfcc: int = 13, n_fft: int = 512, hop: int = 160) -> np.ndarray:
    """Mel频率倒谱系数 — 纯numpy实现"""
    n_frames = max(1, (len(audio) - n_fft) // hop + 1)
    mfccs = np.zeros((n_frames, n_mfcc), dtype=np.float32)

    # Mel滤波器组
    n_mels = 26
    mel_freqs = 2595 * np.log10(1 + np.linspace(0, sr / 2, n_mels + 2) / 700)
    mel_bins = np.floor((n_fft + 1) * mel_freqs / (sr / 2)).astype(int)
    mel_filter = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        mel_filter[m, mel_bins[m]:mel_bins[m + 2]] = np.hanning(mel_bins[m + 2] - mel_bins[m])

    # DCT矩阵 (Type-II, orthonormal)
    _dct_m = np.zeros((n_mels, n_mfcc), dtype=np.float32)
    for k in range(n_mfcc):
        for n in range(n_mels):
            _dct_m[n, k] = np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))
    _dct_m[:, 0] *= np.sqrt(1.0 / n_mels)
    _dct_m[:, 1:] *= np.sqrt(2.0 / n_mels)

    window = np.hanning(n_fft)
    for frame in range(n_frames):
        segment = audio[frame * hop:frame * hop + n_fft]
        if len(segment) < n_fft:
            segment = np.pad(segment, (0, n_fft - len(segment)))
        spec = np.abs(np.fft.rfft(segment * window))
        mel_spec = mel_filter @ spec
        mel_spec = np.log(mel_spec + 1e-8)
        mfccs[frame] = mel_spec @ _dct_m

    return mfccs


# ═══════════════════════════════════════════════════════════════════════════════
# Lifers 声学模型
# ═══════════════════════════════════════════════════════════════════════════════

class LifersAcousticModel:
    """Lifers 品牌化MLP声学模型 — MFCC → 音素分类 (2层+ReLU)"""

    def __init__(self, input_dim: int = 13, hidden_dim: int = 128, n_phonemes: int = N_PHONEMES):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_phonemes = n_phonemes
        rng = np.random.RandomState(42)
        scale1 = math.sqrt(2.0 / input_dim) * 0.1
        self.W1 = rng.randn(input_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        scale2 = math.sqrt(2.0 / hidden_dim) * 0.1
        self.W2 = rng.randn(hidden_dim, hidden_dim // 2).astype(np.float32) * scale2
        self.b2 = np.zeros(hidden_dim // 2, dtype=np.float32)
        self.Wy = rng.randn(hidden_dim // 2, n_phonemes).astype(np.float32) * 0.01
        self.by = np.zeros(n_phonemes, dtype=np.float32)

    def forward(self, x: np.ndarray, state=None):
        h1 = np.maximum(0, x.reshape(1, -1) @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        logits = h2 @ self.Wy + self.by
        return logits, (h1, h2)

    def get_params(self) -> Dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2, "Wy": self.Wy, "by": self.by}

    def set_params(self, params: Dict[str, np.ndarray]):
        for k, v in params.items():
            setattr(self, k, v.copy())


# ═══════════════════════════════════════════════════════════════════════════════
# Lifers Voice 训练数据生成
# ═══════════════════════════════════════════════════════════════════════════════

def generate_voice_training_data(n_samples: int = 500) -> List[Tuple[np.ndarray, int]]:
    """生成合成训练数据: (MFCC特征, 音素标签) — 全局归一化"""
    data = []
    rng = np.random.RandomState(123)

    for _ in range(n_samples):
        phoneme = _PHONEMES[rng.randint(0, len(_PHONEMES))]
        audio = _generate_synthetic_phoneme_audio(phoneme)
        mfcc = _mfcc(audio)
        label = _P2I[phoneme]
        for frame in range(mfcc.shape[0]):
            data.append((mfcc[frame], label))

    # 全局 z-score 归一化
    all_features = np.stack([d[0] for d in data])
    mean = all_features.mean(axis=0, keepdims=True)
    std = all_features.std(axis=0, keepdims=True) + 1e-8
    data = [((d[0] - mean[0]) / std[0], d[1]) for d in data]

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Lifers Voice Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class LifersVoiceTrainer:
    """Lifers 品牌化语音训练器"""

    def __init__(self, lr: float = 1e-2, hidden_dim: int = 128):
        self.model = LifersAcousticModel(hidden_dim=hidden_dim)
        self.lr = lr
        self._loss_history: List[float] = []

    def train_epoch(self, data: List[Tuple[np.ndarray, int]], batch_size: int = 32) -> Dict[str, float]:
        rng = cpu_np.random.RandomState()
        indices = cpu_np.arange(len(data), dtype=cpu_np.int32)
        rng.shuffle(indices)
        total_loss = 0.0
        correct = 0
        n_batches = 0

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            bs = len(batch_idx)

            # GPU批量化: 堆叠所有帧为矩阵 (bs, input_dim)
            X_batch = np.array([data[int(i)][0] for i in batch_idx], dtype=np.float32)
            y_batch = np.array([data[int(i)][1] for i in batch_idx], dtype=np.int32)

            # 单次前向传播 (全batch)
            H1_pre = X_batch @ self.model.W1 + self.model.b1
            H1 = np.maximum(0, H1_pre)
            H2_pre = H1 @ self.model.W2 + self.model.b2
            H2 = np.maximum(0, H2_pre)
            logits = H2 @ self.model.Wy + self.model.by  # (bs, n_phonemes)

            # 批量softmax
            logits_max = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-8)  # (bs, n_phonemes)

            # 批量交叉熵loss
            bs_range = np.arange(bs, dtype=np.int32)
            losses = -np.log(probs[bs_range, y_batch] + 1e-8)
            batch_loss = np.sum(losses)
            batch_correct = int(np.sum(np.argmax(probs, axis=1) == y_batch))

            # 批量反向传播
            dlogits = probs.copy()
            dlogits[bs_range, y_batch] -= 1.0  # (bs, n_phonemes)

            # 第3层梯度
            gWy = H2.T @ dlogits / bs
            gby = np.sum(dlogits, axis=0) / bs

            dh2 = dlogits @ self.model.Wy.T
            dh2[H2_pre <= 0] = 0
            gW2 = H1.T @ dh2 / bs
            gb2 = np.sum(dh2, axis=0) / bs

            dh1 = dh2 @ self.model.W2.T
            dh1[H1_pre <= 0] = 0
            gW1 = X_batch.T @ dh1 / bs
            gb1 = np.sum(dh1, axis=0) / bs

            # 梯度更新
            self.model.W1 -= self.lr * gW1
            self.model.b1 -= self.lr * gb1
            self.model.W2 -= self.lr * gW2
            self.model.b2 -= self.lr * gb2
            self.model.Wy -= self.lr * gWy
            self.model.by -= self.lr * gby

            total_loss += float(batch_loss) / bs
            correct += batch_correct
            n_batches += 1

        accuracy = correct / len(indices) if len(indices) > 0 else 0
        avg_loss = total_loss / max(n_batches, 1)
        self._loss_history.append(avg_loss)
        return {"loss": avg_loss, "accuracy": accuracy}

    def _backward_mlp(self, x, dlogits, grads):
        """2层MLP完整反向传播"""
        x_2d = x.reshape(1, -1)
        h1_pre = x_2d @ self.model.W1 + self.model.b1
        h1 = np.maximum(0, h1_pre)
        h2_pre = h1 @ self.model.W2 + self.model.b2
        h2 = np.maximum(0, h2_pre)

        grads["Wy"] += h2.T @ dlogits
        grads["by"] += dlogits[0]

        dh2 = dlogits @ self.model.Wy.T
        dh2[h2_pre <= 0] = 0
        grads["W2"] += h1.T @ dh2
        grads["b2"] += dh2[0]

        dh1 = dh2 @ self.model.W2.T
        dh1[h1_pre <= 0] = 0
        grads["W1"] += x_2d.T @ dh1
        grads["b1"] += dh1[0]


def train_lifers_voice(
    n_epochs: int = 50,
    n_samples: int = 500,
    save_path: Optional[Path] = None,
    verbose: bool = True,
) -> LifersVoiceTrainer:
    """Lifers Voice 品牌化训练"""
    if save_path is None:
        save_path = ROOT / "weights" / "lifers_voice_acoustic.json"

    trainer = LifersVoiceTrainer()
    data = generate_voice_training_data(n_samples)

    if verbose:
        print(f"[Lifers-Voice] 训练数据: {len(data)} 帧 ({n_samples} 音素样本)")

    best_acc = 0.0
    for epoch in range(n_epochs):
        metrics = trainer.train_epoch(data)
        if (epoch + 1) % 10 == 0 and verbose:
            print(f"[Lifers-Voice] epoch {epoch + 1}/{n_epochs}  "
                  f"loss={metrics['loss']:.4f}  acc={metrics['accuracy']:.3f}")

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            _save_acoustic_model(trainer.model, save_path)

    _save_acoustic_model(trainer.model, save_path)
    if verbose:
        print(f"[Lifers-Voice] 训练完成 best_acc={best_acc:.3f} → {save_path}")
    return trainer


def _save_acoustic_model(model: LifersAcousticModel, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "brand": "Lifers Voice Acoustic",
        "version": 2,
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "n_phonemes": model.n_phonemes,
        "phonemes": _PHONEMES,
        "W1": model.W1.tolist(),
        "b1": model.b1.tolist(),
        "W2": model.W2.tolist(),
        "b2": model.b2.tolist(),
        "Wy": model.Wy.tolist(),
        "by": model.by.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_acoustic_model(path: Path) -> LifersAcousticModel:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    model = LifersAcousticModel(
        input_dim=data["input_dim"],
        hidden_dim=data["hidden_dim"],
        n_phonemes=data["n_phonemes"],
    )
    for k in ["W1", "b1", "W2", "b2", "Wy", "by"]:
        setattr(model, k, np.array(data[k], dtype=np.float32))
    return model


def main():
    epochs = int(os.environ.get("LIFERS_VOICE_EPOCHS", "50"))
    samples = int(os.environ.get("LIFERS_VOICE_SAMPLES", "500"))
    out = ROOT / "weights" / "lifers_voice_acoustic.json"

    print(f"[Lifers-Voice] 品牌化语音训练 epochs={epochs} samples={samples}")
    t0 = time.time()
    train_lifers_voice(n_epochs=epochs, n_samples=samples, save_path=out, verbose=True)
    print(f"[Lifers-Voice] 完成 耗时={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
