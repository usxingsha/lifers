from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lifers.markov_lm import MarkovWeights, generate
from lifers.model_names import canonical_brain_model, resolve_existing_weight_file
from lifers.transformer_lm import TinyTransformerWeights, generate_text


def _respond(prompt: str) -> str:
    root = Path(__file__).resolve().parent.parent
    model = canonical_brain_model(os.environ.get("MODEL", "markov"))
    if model == "transformer":
        weights_path = resolve_existing_weight_file(root, "transformer")
        if not weights_path:
            return f"(missing transformer weights) {prompt[:80]}"
        w = TinyTransformerWeights.load(weights_path)
        return generate_text(w, prompt=prompt + "\n", max_chars=80, seed=2, temperature=1.0, top_k=40).strip()
    weights_path = resolve_existing_weight_file(root, "markov")
    if not weights_path:
        return f"(missing markov weights) {prompt[:80]}"
    w = MarkovWeights.load(weights_path)
    return generate(w, prompt=prompt + "\n", max_chars=120, seed=2, temperature=0.9, top_k=80).strip()


def run_smoke(prompts: List[str]) -> Dict[str, Any]:
    t0 = time.time()
    outputs = []
    empty = 0
    for p in prompts:
        out = _respond(p)
        if not out.strip():
            empty += 1
        outputs.append({"prompt": p, "output": out})
    elapsed = time.time() - t0
    return {
        "count": len(prompts),
        "empty_rate": empty / max(1, len(prompts)),
        "elapsed_s": elapsed,
        "avg_s": elapsed / max(1, len(prompts)),
        "sandbox": os.environ.get("SANDBOX", "0") == "1",
        "samples": outputs[:3],
    }


def main() -> None:
    prompts = [
        "Say hello in Chinese.",
        "Summarize: offline local LLM with short/long memory.",
        "Explain Q4 vs Q5 in one sentence.",
    ]
    report = run_smoke(prompts)
    out_path = Path(__file__).resolve().parent.parent / "exp_smoke.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

