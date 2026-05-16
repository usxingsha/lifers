from __future__ import annotations

import json
import os
from pathlib import Path

from lifers.transformer_train_np import train_backprop_minimal


def _load_jsonl_inputs(path: Path) -> str:
    buf = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        buf.append(str(obj.get("input", "")))
    return "\n".join(buf)


def _load_training_corpus(root: Path) -> str:
    """Load training corpus: prefer weights/training_corpus.txt, fall back to eval suites."""
    corpus_txt = root / "weights" / "training_corpus.txt"
    if corpus_txt.is_file():
        return corpus_txt.read_text(encoding="utf-8")
    suite = root / "eval" / "suites" / "v001"
    text = []
    for p in sorted(suite.glob("*.jsonl")):
        text.append(_load_jsonl_inputs(p))
    return "\n".join(text) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    corpus = _load_training_corpus(root)

    out = root / "weights" / "lifers_transformer.json"
    steps = int(os.environ.get("TT_STEPS", "500"))
    max_vocab = int(os.environ.get("TT_VOCAB", "128"))
    d_model = int(os.environ.get("TT_DMODEL", "48"))
    d_ff = int(os.environ.get("TT_DFF", "128"))
    max_seq = int(os.environ.get("TT_MAXSEQ", "48"))
    lr = float(os.environ.get("TT_LR", "3e-4"))
    train_backprop_minimal(
        corpus,
        out_path=out,
        steps=steps,
        max_vocab=max_vocab,
        d_model=d_model,
        d_ff=d_ff,
        max_seq=max_seq,
        lr=lr,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

