from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class MarkovWeights:
    """
    A tiny self-made language model:
    - char-level bigram counts with Laplace smoothing
    - saved as JSON "weights"

    This is intentionally dependency-free and fully local.
    """

    vocab: List[str]
    counts: Dict[str, Dict[str, int]]
    total: Dict[str, int]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"vocab": self.vocab, "counts": self.counts, "total": self.total}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "MarkovWeights":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return MarkovWeights(vocab=list(obj["vocab"]), counts=obj["counts"], total=obj["total"])


def train_from_text(text: str, min_char_freq: int = 1) -> MarkovWeights:
    # Basic char vocab.
    freq: Dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    vocab = [ch for ch, c in freq.items() if c >= min_char_freq]
    # Ensure we can always end.
    if "\n" not in vocab:
        vocab.append("\n")

    counts: Dict[str, Dict[str, int]] = {}
    total: Dict[str, int] = {}
    prev = "\n"
    for ch in text:
        if ch not in freq:
            continue
        if prev not in counts:
            counts[prev] = {}
            total[prev] = 0
        counts[prev][ch] = counts[prev].get(ch, 0) + 1
        total[prev] += 1
        prev = ch
    if prev not in counts:
        counts[prev] = {}
        total[prev] = 0
    return MarkovWeights(vocab=vocab, counts=counts, total=total)


def _sample_next(
    w: MarkovWeights,
    prev: str,
    rng: random.Random,
    temperature: float = 0.8,
    top_k: int = 50,
) -> str:
    # Laplace smoothing.
    vocab = w.vocab
    row = w.counts.get(prev, {})
    denom = w.total.get(prev, 0) + len(vocab)

    # Build probabilities.
    scored: List[Tuple[str, float]] = []
    for ch in vocab:
        p = (row.get(ch, 0) + 1) / denom
        # temperature in log-space
        lp = math.log(p) / max(1e-6, temperature)
        scored.append((ch, lp))
    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[: max(1, min(top_k, len(scored)))]

    # Softmax
    m = scored[0][1]
    exps = [math.exp(s - m) for _, s in scored]
    ssum = sum(exps)
    r = rng.random() * ssum
    acc = 0.0
    for (ch, _), e in zip(scored, exps):
        acc += e
        if acc >= r:
            return ch
    return scored[-1][0]


def generate(
    w: MarkovWeights,
    prompt: str,
    max_chars: int = 200,
    seed: int = 1,
    temperature: float = 0.8,
    top_k: int = 80,
) -> str:
    rng = random.Random(seed)
    # Extract user message from full agent prompt (SYSTEM...USER:...ASSISTANT:)
    user_line = prompt
    for marker in ("\nUSER:\n", "\nUSER:", "USER:\n", "USER:"):
        if marker in prompt:
            _, _, after = prompt.partition(marker)
            # Take the user message line
            user_line = after.split("\n")[0].strip()
            break
    # Also try EXECUTION_REQUEST for legacy prompts
    if user_line == prompt:
        for marker in ("\nEXECUTION_REQUEST:\n", "EXECUTION_REQUEST:"):
            if marker in prompt:
                _, _, after = prompt.partition(marker)
                user_line = after.split("\n")[0].strip()
                break
    out: list[str] = []
    if user_line.strip():
        # Use the user's actual message as prompt start
        out.append(user_line.strip())
        out.append("：")
        prev = user_line.strip()[-1] if user_line.strip() else "\n"
    elif prompt.strip():
        prev = prompt.strip()[-1]
    else:
        prev = "\n"
    newline_count = 0
    for _ in range(max_chars):
        ch = _sample_next(w, prev=prev, rng=rng, temperature=temperature, top_k=top_k)
        out.append(ch)
        prev = ch
        if ch == "\n":
            newline_count += 1
            if newline_count >= 2 and len(out) > 20:
                break
        elif ch in ("。", "！", "？"):
            if len(out) > 30:
                break
    return "".join(out)

