"""
Lifers Voice — 语音闭环 (ASR + TTS + 唤醒词)
纯 NumPy DSP 实现，自给自足
"""

from __future__ import annotations

import math
import struct
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# DSP utilities
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 16000
FRAME_MS = 25
HOP_MS = 10
N_FFT = 512
N_MELS = 40


def _hann_window(n: int) -> np.ndarray:
    return 0.5 * (1 - np.cos(2 * math.pi * np.arange(n) / (n - 1)))


def _mel_filterbank(n_fft: int, sr: int, n_mels: int) -> np.ndarray:
    mel_min, mel_max = 0, 2595 * math.log10(1 + sr / 2 / 700)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    freq_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bins = np.floor((n_fft + 1) * freq_pts / sr).astype(int)
    filt = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        for j in range(bins[i], bins[i + 1]):
            filt[i, j] = (j - bins[i]) / max(bins[i + 1] - bins[i], 1)
        for j in range(bins[i + 1], bins[i + 2] + 1):
            if j < filt.shape[1]:
                filt[i, j] = (bins[i + 2] - j) / max(bins[i + 2] - bins[i + 1], 1)
    return filt


def _stft(signal: np.ndarray, frame_len: int, hop: int, window: np.ndarray) -> np.ndarray:
    n_frames = max(1, (len(signal) - frame_len) // hop + 1)
    spec = np.zeros((n_frames, N_FFT // 2 + 1), dtype=np.complex64)
    for i in range(n_frames):
        start = i * hop
        segment = signal[start:start + frame_len].astype(np.float64)
        if len(segment) < frame_len:
            segment = np.pad(segment, (0, frame_len - len(segment)))
        padded = np.zeros(N_FFT)
        padded[:frame_len] = segment * window
        spec[i] = np.fft.rfft(padded)
    return spec


def _mel_spectrogram(signal: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    frame_len = int(sr * FRAME_MS / 1000)
    hop = int(sr * HOP_MS / 1000)
    window = _hann_window(frame_len)
    spec = _stft(signal, frame_len, hop, window)
    mel_filt = _mel_filterbank(N_FFT, sr, N_MELS)
    power = np.abs(spec) ** 2
    mel = power @ mel_filt.T
    mel = np.log(mel + 1e-6)
    return mel.astype(np.float32)


def _mfcc(mel_spec: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    log_mel = np.log(mel_spec + 1e-6)
    mfcc = np.fft.dct(log_mel, axis=1, type=2, norm="ortho")[:, :n_mfcc]
    return mfcc.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Wake Word Detection
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WakeWordConfig:
    word: str = "lifers"
    threshold: float = 0.7
    cooldown_ms: int = 2000


class WakeWordDetector:
    """Energy + simple pattern-based wake word detection."""

    def __init__(self, config: WakeWordConfig = WakeWordConfig()) -> None:
        self.config = config
        self._last_trigger_ms = 0
        self._energy_history = deque(maxlen=30)
        self._rng = np.random.RandomState(42)
        # Simple phoneme templates for "li-fers" approximated as energy patterns
        self._template = self._make_template(config.word)

    def _make_template(self, word: str) -> np.ndarray:
        """Create a fake energy template for a given word (simple 2-syllable pattern)."""
        t = np.linspace(0, math.pi * 4, 60)
        return 0.5 * (np.sin(t) + np.sin(2 * t) * 0.5 + 0.3 * np.sin(3 * t))

    def detect(self, audio_chunk: np.ndarray, ts_ms: int) -> bool:
        if ts_ms - self._last_trigger_ms < self.config.cooldown_ms:
            return False
        energy = float(np.sqrt(np.mean(audio_chunk ** 2)))
        self._energy_history.append(energy)
        if len(self._energy_history) < 30:
            return False
        # Compare energy envelope with template
        env = np.array(self._energy_history)
        env_norm = (env - np.mean(env)) / (np.std(env) + 1e-6)
        template_norm = (self._template - np.mean(self._template)) / (np.std(self._template) + 1e-6)
        if len(template_norm) != len(env_norm):
            # Resample template to match
            indices = np.linspace(0, len(template_norm) - 1, len(env_norm))
            template_resampled = np.interp(indices, np.arange(len(template_norm)), template_norm)
            corr = np.corrcoef(env_norm, template_resampled)[0, 1]
        else:
            corr = np.corrcoef(env_norm, template_norm)[0, 1]
        if corr > self.config.threshold and energy > 0.01:
            self._last_trigger_ms = ts_ms
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Speech-to-Text (ASR) — phoneme-level
# ═══════════════════════════════════════════════════════════════════════════════

# Simple English phoneme inventory
_PHONEMES = [
    "aa", "ae", "ah", "ao", "aw", "ay", "b", "ch", "d", "dh", "eh", "er",
    "ey", "f", "g", "hh", "ih", "iy", "jh", "k", "l", "m", "n", "ng",
    "ow", "oy", "p", "r", "s", "sh", "t", "th", "uh", "uw", "v", "w",
    "y", "z", "zh", "sil",
]
_P2I = {p: i for i, p in enumerate(_PHONEMES)}


class LSTMAcousticModel:
    """Tiny LSTM-based acoustic model for phoneme classification. Pure numpy."""

    def __init__(self, input_dim: int = 13, hidden_dim: int = 64, n_phonemes: int = 38) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_phonemes = n_phonemes
        rng = np.random.RandomState(42)
        scale = math.sqrt(2.0 / hidden_dim)
        self.Wf = rng.randn(input_dim + hidden_dim, hidden_dim).astype(np.float32) * scale * 0.1
        self.Wi = rng.randn(input_dim + hidden_dim, hidden_dim).astype(np.float32) * scale * 0.1
        self.Wc = rng.randn(input_dim + hidden_dim, hidden_dim).astype(np.float32) * scale * 0.1
        self.Wo = rng.randn(input_dim + hidden_dim, hidden_dim).astype(np.float32) * scale * 0.1
        self.Wy = rng.randn(hidden_dim, n_phonemes).astype(np.float32) * 0.01
        self.by = np.zeros(n_phonemes, dtype=np.float32)

    def forward(self, x: np.ndarray, state: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """x: [1, input_dim], return logits + (h, c)."""
        if state is None:
            h = np.zeros((1, self.hidden_dim), dtype=np.float32)
            c = np.zeros((1, self.hidden_dim), dtype=np.float32)
        else:
            h, c = state
        xh = np.concatenate([x.reshape(1, -1), h], axis=1)
        f = 1.0 / (1.0 + np.exp(-xh @ self.Wf))
        i = 1.0 / (1.0 + np.exp(-xh @ self.Wi))
        c_tilde = np.tanh(xh @ self.Wc)
        c = f * c + i * c_tilde
        o = 1.0 / (1.0 + np.exp(-xh @ self.Wo))
        h = o * np.tanh(c)
        logits = h @ self.Wy + self.by
        return logits, (h, c)


class LifersASR:
    """Lifers Speech-to-Text: MFCC → LSTM → phoneme → text (simple)."""

    def __init__(self) -> None:
        self.acoustic = LSTMAcousticModel()
        self._phoneme_map: Dict[str, str] = {
            "aa": "a", "ae": "a", "ah": "a", "ao": "o", "aw": "ow",
            "ay": "i", "b": "b", "ch": "ch", "d": "d", "dh": "th",
            "eh": "e", "er": "er", "ey": "ay", "f": "f", "g": "g",
            "hh": "h", "ih": "i", "iy": "ee", "jh": "j", "k": "k",
            "l": "l", "m": "m", "n": "n", "ng": "ng",
            "ow": "o", "oy": "oy", "p": "p", "r": "r", "s": "s",
            "sh": "sh", "t": "t", "th": "th", "uh": "u", "uw": "oo",
            "v": "v", "w": "w", "y": "y", "z": "z", "zh": "zh", "sil": "",
        }

    def transcribe(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> Tuple[str, float]:
        mel = _mel_spectrogram(audio, sr)
        mfcc = _mfcc(mel, n_mfcc=13)
        state = None
        phoneme_ids = []
        for t in range(mfcc.shape[0]):
            logits, state = self.acoustic.forward(mfcc[t:t+1], state)
            pid = int(np.argmax(logits[0]))
            phoneme_ids.append(pid)
        # Merge consecutive duplicates
        merged = []
        for pid in phoneme_ids:
            if not merged or pid != merged[-1]:
                merged.append(pid)
        # Remove sil
        merged = [p for p in merged if _PHONEMES[p] != "sil"]
        # Phonemes → rough text
        text_parts = [self._phoneme_map.get(_PHONEMES[p], "") for p in merged]
        text = "".join(text_parts)
        # Crude confidence: average softmax max
        confidence = 0.5  # placeholder
        return text.strip(), confidence

    @staticmethod
    def from_file(path: str) -> Tuple[np.ndarray, int]:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            data = wf.readframes(n_frames)
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, sr


# ═══════════════════════════════════════════════════════════════════════════════
# Text-to-Speech (TTS) — phoneme synthesis
# ═══════════════════════════════════════════════════════════════════════════════

class LifersTTS:
    """Lifers Text-to-Speech: text → phonemes → formant synthesis.  Pure numpy."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sr = sample_rate
        self._formants: Dict[str, Tuple[float, float, float]] = {
            "aa": (730, 1090, 2440), "iy": (270, 2290, 3010),
            "uw": (300, 870, 2240), "ae": (660, 1720, 2410),
            "ah": (640, 1190, 2390), "eh": (530, 1840, 2480),
            "ih": (390, 1990, 2550), "ow": (480, 920, 2200),
            "uh": (440, 1020, 2240), "er": (490, 1350, 1690),
            "ao": (570, 840, 2410), "ay": (700, 1400, 2400),
            "aw": (700, 1200, 2300), "ey": (500, 2000, 2600),
            "oy": (400, 800, 2500), "ax": (500, 1500, 2500),
        }

    def synthesize(self, text: str, duration_sec: float = 2.0) -> np.ndarray:
        phones = self._text_to_phonemes(text.lower())
        if not phones:
            phones = ["ax"]
        n_samples = int(self.sr * duration_sec)
        samples_per_phone = n_samples // len(phones)
        t = np.arange(samples_per_phone, dtype=np.float32) / self.sr
        output = np.zeros(n_samples, dtype=np.float32)
        for i, ph in enumerate(phones):
            start = i * samples_per_phone
            end = start + samples_per_phone
            f1, f2, f3 = self._formants.get(ph, (500, 1500, 2500))
            # Simple formant synthesis: sum of sine waves + noise
            glottal = np.sin(2 * math.pi * 120 * t) * 0.3  # fundamental
            glottal += np.sin(2 * math.pi * f1 * t) * 0.15
            glottal += np.sin(2 * math.pi * f2 * t) * 0.10
            glottal += np.sin(2 * math.pi * f3 * t) * 0.05
            # Amplitude envelope
            env = np.exp(-3.0 * t / (t[-1] + 1e-6))
            segment = glottal * env
            actual_len = min(len(segment), n_samples - start)
            output[start:start + actual_len] = segment[:actual_len]
        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output /= max_val * 1.2
        return output

    def _text_to_phonemes(self, text: str) -> List[str]:
        """Very simple text→phoneme mapping (placeholder for CMUdict)."""
        # Simple grapheme-to-phoneme rules
        mapping = {
            "hello": ["hh", "eh", "l", "ow"],
            "world": ["w", "er", "l", "d"],
            "lifers": ["l", "ay", "f", "er", "z"],
            "yes": ["y", "eh", "s"],
            "no": ["n", "ow"],
            "go": ["g", "ow"],
            "stop": ["s", "t", "ao", "p"],
            "robot": ["r", "ow", "b", "ao", "t"],
            "human": ["hh", "y", "uw", "m", "ah", "n"],
            "ai": ["ey", "ay"],
            "think": ["th", "ih", "ng", "k"],
            "plan": ["p", "l", "ae", "n"],
            "good": ["g", "uh", "d"],
            "bad": ["b", "ae", "d"],
            "move": ["m", "uw", "v"],
            "sense": ["s", "eh", "n", "s"],
            "speak": ["s", "p", "iy", "k"],
            "listen": ["l", "ih", "s", "ah", "n"],
            "learn": ["l", "er", "n"],
            "remember": ["r", "ih", "m", "eh", "m", "b", "er"],
            "safe": ["s", "ey", "f"],
            "help": ["hh", "eh", "l", "p"],
            "error": ["eh", "r", "er"],
            "done": ["d", "ah", "n"],
        }
        words = text.split()
        result = []
        for w in words:
            clean = "".join(c for c in w if c.isalpha())
            if clean in mapping:
                result.extend(mapping[clean])
            else:
                # Fallback: letter-by-letter
                for c in clean:
                    if c in "aeiou":
                        result.append({"a": "aa", "e": "eh", "i": "ih", "o": "ow", "u": "uh"}.get(c, "ax"))
        return result

    def save_wav(self, audio: np.ndarray, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sr)
            wf.writeframes(audio_int16.tobytes())


# ═══════════════════════════════════════════════════════════════════════════════
# Voice Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class VoicePipeline:
    """Complete voice loop: listen → transcribe → (process) → synthesize → speak."""

    def __init__(self) -> None:
        self.asr = LifersASR()
        self.tts = LifersTTS()
        self.wakeword = WakeWordDetector(WakeWordConfig(word="lifers", threshold=0.6))
        self.is_listening = False

    def process_audio_chunk(self, audio: np.ndarray, ts_ms: int) -> Optional[Dict[str, Any]]:
        """Process one audio chunk. Returns transcription if wake word triggered and speech detected."""
        if not self.is_listening:
            if self.wakeword.detect(audio, ts_ms):
                self.is_listening = True
                return {"event": "wakeword", "word": "lifers"}
            return None
        # Check for silence to stop listening
        energy = float(np.sqrt(np.mean(audio ** 2)))
        if energy < 0.005:
            self.is_listening = False
            return None
        text, conf = self.asr.transcribe(audio)
        if text:
            return {"event": "speech", "text": text, "confidence": conf}
        return None

    def speak(self, text: str) -> np.ndarray:
        dur = max(0.8, len(text.split()) * 0.3)
        return self.tts.synthesize(text, duration_sec=dur)
