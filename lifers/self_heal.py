"""
自修复（配置侧）：维护 stack.json 可读与缺省键；不替代 lifers_workspace_write 等显式改码。

- stack.json 损坏：备份后从包内模板恢复（若存在）。
- 缺失键：递归合并默认值（不覆盖用户已有值）。
- 环境开关：LIFERS_SELF_HEAL=0 关闭。
"""

from __future__ import annotations

import json
import os
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

# 仅补全缺失路径；与 config/stack.json 保持语义一致即可。
_DEFAULT_STACK_KEYS: Dict[str, Any] = {
    "embodied_world": {
        "dynamic_npc": [
            {
                "name": "示例NPC",
                "persona": "友善的居民（示例；按需增删）",
                "backstory": "",
                "voice": "",
                "greeting": "你好，我是这里的居民。",
                "portrait_emoji": "👤",
            }
        ],
    },
    "brain": {
        "self_code": {
            "enabled": True,
            "auto_consume_queue": True,
            "queue_dir": "state/self_code_queue",
            "done_dir": "state/self_code_done",
            "error_dir": "state/self_code_error",
            "max_file_bytes": 800000,
            "allow_rel_prefixes": None,
        },
        "deep_steward": {
            "enabled": True,
            "prune_taskflow_older_than_days": 14,
            "prune_taskflow_max_delete": 800,
            "global_forget": {
                "enabled": True,
                "min_importance": 0.14,
                "older_than_days": 48,
                "limit": 260,
                "auto_threshold": {
                    "enabled": True,
                    "min_importance_floor": 0.055,
                    "min_importance_ceiling": 0.28,
                    "rows_soft_cap": 12000,
                    "limit_boost_per_1k_rows": 48,
                    "limit_max": 1400,
                },
            },
            "note_zh": "任务流学习写入 taskflow 后 steward 按天删旧；global_forget 可配 auto_threshold；自改码见 self_code。",
        },
    }
}

# 启动时若 stack 缺顶层节，则补全（不覆盖已有节）。
_DEFAULT_STACK_TOP: Dict[str, Any] = {
    "remote_infer": {
        "enabled": False,
        "provider": "nvidia_integrate",
        "chat_url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "model": "meta/llama-3.1-8b-instruct",
        "max_tokens": 1024,
        "api_key_env": "NVIDIA_API_KEY",
    },
    "local_llm": {
        "enabled": False,
        "chat_url": "http://localhost:11434/v1/chat/completions",
        "model": "qwen2.5:7b",
        "max_tokens": 1024,
        "timeout_sec": 120,
        "api_key_env": "",
    },
}

_MINIMAL_STACK: Dict[str, Any] = {
    "version": 1,
    "runtime": {"role": "auto"},
    "embodied_world": {
        "dynamic_npc": [],
    },
    "brain": {
        "model": "lifers",
        "sandbox": False,
        "max_tool_steps": 6,
        "session_max_turns": 8,
        "llm_identity_short": "Lifers",
        "llm_product_name": "Lifers-20B",
        "memory_db": "memory/longterm.sqlite3",
        "audit_log": "logs/audit.jsonl",
        "weights": {
            "markov": "weights/lifers_markov.json",
            "transformer": "weights/lifers_transformer.json",
        },
        "self_code": {
            "enabled": True,
            "auto_consume_queue": True,
            "queue_dir": "state/self_code_queue",
            "done_dir": "state/self_code_done",
            "error_dir": "state/self_code_error",
            "max_file_bytes": 800000,
            "allow_rel_prefixes": None,
        },
        "deep_steward": {
            "enabled": True,
            "prune_taskflow_older_than_days": 14,
            "prune_taskflow_max_delete": 800,
            "global_forget": {
                "enabled": True,
                "min_importance": 0.14,
                "older_than_days": 48,
                "limit": 260,
                "auto_threshold": {
                    "enabled": True,
                    "min_importance_floor": 0.055,
                    "min_importance_ceiling": 0.28,
                    "rows_soft_cap": 12000,
                    "limit_boost_per_1k_rows": 48,
                    "limit_max": 1400,
                },
            },
        },
    },
    "human_sim": {
        "enabled": True,
        "persona_name": "Lifers",
        "system_prompt_extra": "",
        "local_lm_max_chars": 200,
    },
    "robot": {
        "sim_exec_cmd": "",
        "sense_exec_cmd": "",
        "act_exec_cmd": "",
        "stub_when_no_cmd": True,
        "default_sim_runs": 10,
        "tasks_file": "sim/tasks/tasks_v001.jsonl",
    },
}


def _package_template_stack() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "stack.json"


def _merge_missing(dst: Any, src: Any) -> bool:
    """把 src 里 dst 没有的键补进 dst；均为 dict 时递归。返回是否改过。"""
    changed = False
    if not isinstance(dst, dict) or not isinstance(src, dict):
        return False
    for k, v in src.items():
        if k not in dst:
            dst[k] = deepcopy(v)
            changed = True
        elif isinstance(dst[k], dict) and isinstance(v, dict):
            if _merge_missing(dst[k], v):
                changed = True
    return changed


def heal_stack_at_startup(root: Path) -> Dict[str, Any]:
    if os.environ.get("LIFERS_SELF_HEAL", "1").strip().lower() in ("0", "false", "no", "off"):
        return {"skipped": True}

    p = root / "config" / "stack.json"
    tpl = _package_template_stack()
    report: Dict[str, Any] = {"path": str(p)}

    if not p.parent.is_dir():
        p.parent.mkdir(parents=True, exist_ok=True)

    if not p.is_file():
        if tpl.is_file():
            shutil.copy2(tpl, p)
            report["created_from_template"] = True
        else:
            p.write_text(json.dumps(_MINIMAL_STACK, ensure_ascii=False, indent=2), encoding="utf-8")
            report["created_minimal"] = True

    data, err = _load_stack_data(p)
    if err:
        try:
            bad = p.with_suffix(f".json.corrupt-{int(time.time())}")
            p.rename(bad)
            report["corrupt_renamed"] = str(bad)
        except OSError:
            try:
                p.unlink(missing_ok=True)  # type: ignore[arg-type]
            except OSError:
                pass
        if tpl.is_file():
            shutil.copy2(tpl, p)
            report["restored_from_template"] = True
        else:
            p.write_text(json.dumps(_MINIMAL_STACK, ensure_ascii=False, indent=2), encoding="utf-8")
            report["restored_minimal"] = True
        data, err2 = _load_stack_data(p)
        if err2:
            report["error"] = str(err2)
            return report

    if _merge_missing(data, _DEFAULT_STACK_KEYS):
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        report["merged_default_keys"] = True

    if _merge_missing(data, _DEFAULT_STACK_TOP):
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        report["merged_default_top_keys"] = True

    return report


def _load_stack_data(p: Path) -> Tuple[Dict[str, Any], Exception | None]:
    try:
        raw = p.read_text(encoding="utf-8")
        return json.loads(raw), None
    except Exception as e:  # noqa: BLE001
        return {}, e
