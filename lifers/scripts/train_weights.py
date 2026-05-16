from __future__ import annotations

import json
from pathlib import Path

from lifers.markov_lm import MarkovWeights, train_from_text


def _load_jsonl(path: Path) -> str:
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
        text.append(_load_jsonl(p))
    corpus = "\n".join(text) + "\n"
    w = train_from_text(corpus)
    out = root / "weights" / "lifers_markov.json"
    w.save(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

