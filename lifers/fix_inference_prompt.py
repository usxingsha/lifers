"""TinyTransformer 长 prompt 在进入 `generate_text` / 流式前裁剪。

根因：`transformer_lm.generate_text` 每步 `forward(w, ids[-w.max_seq:])`，若 prompt 经
`encode` 后长度远大于 `max_seq`，有效因果窗只剩**末尾**若干 token，易变成仅
`ASSISTANT:\\n` 等高频片段，指令与 SYSTEM 丢失。

本模块在 **调用 `generate_text` / 流式编码前** 将字符串裁到「SYSTEM 头 + USER/ASSISTANT
尾」结构，使总长度不超过 `max_seq`（本仓 `encode` 为每字符一 id，按字符预算保守处理）。
"""

from __future__ import annotations

_ELLIPSIS_BLOCK = "\n…【上下文已截断以适配 max_seq】\n"


def clip_prompt_for_transformer(prompt: str, max_seq: int) -> str:
    """
    将超长 prompt 裁到不超过 ``max_seq`` 字符，并尽量保留 ``USER:`` / ``ASSISTANT:`` 尾段。

    ``max_seq`` 与权重 JSON 中 TinyTransformer 的 ``max_seq`` 一致；与 ``forward`` 实际
    可见窗长度对齐（字符数≈token 数）。
    """
    if max_seq < 16:
        max_seq = 16
    budget = max_seq
    if len(prompt) <= budget:
        return prompt

    user_markers = ("\nUSER:\n", "\nUSER:", "USER:\n")
    asst_markers = ("\nASSISTANT:\n", "\nASSISTANT:", "ASSISTANT:\n")

    ui = -1
    for m in user_markers:
        j = prompt.rfind(m)
        if j > ui:
            ui = j
    if ui < 0:
        return prompt[-budget:]

    ai = -1
    for m in asst_markers:
        j = prompt.rfind(m)
        if j > ai:
            ai = j
    if ai < ui:
        return prompt[-budget:]

    tail = prompt[ui:]
    if len(tail) > budget:
        return tail[-budget:]

    head_cap = budget - len(tail) - len(_ELLIPSIS_BLOCK)
    if head_cap < 24:
        stub = "SYSTEM:\n…【上文过长已截断】…\n"
        combo = stub + tail
        return combo[-budget:] if len(combo) > budget else combo

    head = prompt[:ui]
    if len(head) <= head_cap:
        return head + tail

    head2 = head[:head_cap]
    cut = head2.rfind("\n")
    if cut > head_cap // 3:
        head2 = head2[: cut + 1]

    out = head2 + _ELLIPSIS_BLOCK + tail
    if len(out) <= budget:
        return out
    return out[:budget]


def patch_agent_generate_call(prompt: str, w_max_seq: int) -> str:
    """供 `LocalBrain.generate` 在 ``tt_generate`` 之前调用（与 `clip_prompt_for_transformer` 等价）。"""
    return clip_prompt_for_transformer(prompt, w_max_seq)
