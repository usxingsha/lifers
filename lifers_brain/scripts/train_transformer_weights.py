from __future__ import annotations

import json
import os
from pathlib import Path

from lifers_brain.transformer_lm import train_sgd_minimal


def _load_jsonl_inputs(path: Path) -> str:
    buf = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        buf.append(str(obj.get("input", "")))
    return "\n".join(buf)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    suite = root / "eval" / "suites" / "v001"
    text = []
    for p in sorted(suite.glob("*.jsonl")):
        text.append(_load_jsonl_inputs(p))
    corpus = "\n".join(text) + "\n"

    out = root / "weights" / "lifers_transformer.json"
    # Keep it light; you can scale steps/dims later.
    steps = int(os.environ.get("TT_STEPS", "2"))
    max_vocab = int(os.environ.get("TT_VOCAB", "96"))
    d_model = int(os.environ.get("TT_DMODEL", "24"))
    d_ff = int(os.environ.get("TT_DFF", "48"))
    max_seq = int(os.environ.get("TT_MAXSEQ", "32"))
    train_sgd_minimal(
        corpus,
        out_path=out,
        steps=steps,
        max_vocab=max_vocab,
        d_model=d_model,
        d_ff=d_ff,
        max_seq=max_seq,
        lr=1e-2,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

