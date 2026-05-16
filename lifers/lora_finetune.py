"""LoRA 风格增量：低秩 Δ 合并到 TinyTransformerWeights.Wlm（可选 sidecar JSON）。

训练脚本可写入 `weights/lifers_lora.json`；推理时由 `LocalBrain` 在加载 base 后合并。
JSON 格式::

    { "rank": 4, "alpha": 1.0, "A": [[... d_model x r ]], "B": [[... r x V ]] }

或旧式全矩阵（不推荐，体积大）::

    { "Wlm_delta": [[... d_model x V ]] }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from lifers.transformer_lm import TinyTransformerWeights


def _matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    m, k = len(A), len(A[0])
    n = len(B[0])
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for kk in range(k):
            aik = A[i][kk]
            for j in range(n):
                out[i][j] += aik * B[kk][j]
    return out


def _add_matrix(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def merge_lora_into_weights(w: TinyTransformerWeights, lora_path: Path, *, alpha: float = 1.0) -> TinyTransformerWeights:
    """若文件不存在或解析失败，返回原对象引用。"""
    if not lora_path.is_file():
        return w
    try:
        obj: dict[str, Any] = json.loads(lora_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return w
    alpha = float(obj.get("alpha", alpha) or 1.0)
    new_wlm = list(w.Wlm)
    if "Wlm_delta" in obj and isinstance(obj["Wlm_delta"], list):
        delta = obj["Wlm_delta"]
        if len(delta) == len(new_wlm) and all(isinstance(row, list) and len(row) == len(new_wlm[0]) for row in delta):
            new_wlm = _add_matrix(new_wlm, [[alpha * float(x) for x in row] for row in delta])
    elif "A" in obj and "B" in obj:
        A, B = obj["A"], obj["B"]
        if isinstance(A, list) and isinstance(B, list):
            delta = _matmul(A, B)
            if len(delta) == len(new_wlm) and len(delta[0]) == len(new_wlm[0]):
                new_wlm = _add_matrix(new_wlm, [[alpha * float(x) for x in row] for row in delta])
    return TinyTransformerWeights(
        vocab=list(w.vocab),
        d_model=w.d_model,
        d_ff=w.d_ff,
        n_heads=w.n_heads,
        max_seq=w.max_seq,
        tok_emb=w.tok_emb,
        pos_emb=w.pos_emb,
        Wq=w.Wq,
        Wk=w.Wk,
        Wv=w.Wv,
        Wo=w.Wo,
        W1=w.W1,
        W2=w.W2,
        Wlm=new_wlm,
    )


def lora_sidecar_path(root: Path, stack: dict) -> Path | None:
    brain = stack.get("brain") or {}
    lcfg = brain.get("lora") if isinstance(brain.get("lora"), dict) else {}
    if not (lcfg.get("enabled") is True or str(lcfg.get("enabled")).lower() == "true"):
        return None
    rel = str(lcfg.get("weights_relpath") or "weights/lifers_lora.json").strip()
    if not rel:
        return None
    p = (root / rel).resolve()
    return p
