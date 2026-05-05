"""
深层保障（Deep steward）：在任务流学习写入之后，做轻量自维护。

- 不绑定特定检索口令或浏览器；出站 HTTP 由 tools 层统一走系统/环境代理。
- 自动学习：见 taskflow.learn → longterm type=taskflow。
- 自动减重：按配置删除过旧的 taskflow 记录，避免 SQLite 无限膨胀。
- 可选 global_forget：对任意类型里「低重要性且过旧」的记忆做类人遗忘；可开 auto_threshold 按库行数调节阈值。

配置：stack.json → brain.deep_steward（见默认字段）。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from lifers_brain.agent import LifersAgent


def _deep_cfg(root: Any) -> Dict[str, Any]:
    try:
        from lifers_brain.stack_env import load_stack

        st = load_stack(root)
        return (st.get("brain") or {}).get("deep_steward") or {}
    except Exception:
        return {}


def _resolve_global_forget_params(
    gf: Dict[str, Any],
    agent: "LifersAgent",
) -> Tuple[float, int, int, Dict[str, Any]]:
    """
    返回 (min_importance, older_than_days, limit, debug_dict)。
    auto_threshold.enabled 时随 memories 行数抬高删除力度（更旧/更多条）。
    """
    base_mi = float(gf.get("min_importance", 0.12))
    base_days = int(gf.get("older_than_days", 60) or 60)
    base_lim = int(gf.get("limit", 200) or 200)
    debug: Dict[str, Any] = {"base_min_importance": base_mi, "base_older_than_days": base_days, "base_limit": base_lim}

    at = gf.get("auto_threshold") or {}
    if not isinstance(at, dict) or not at.get("enabled"):
        return base_mi, base_days, base_lim, debug

    try:
        n = int(agent.longterm.count_all())
    except Exception:
        n = 0
    debug["memory_rows"] = n

    floor = float(at.get("min_importance_floor", 0.06))
    ceil = float(at.get("min_importance_ceiling", 0.26))
    soft = max(800, int(at.get("rows_soft_cap", 12000) or 12000))
    scale = min(1.0, max(0.0, (float(n) - float(soft)) / float(soft)))
    # 行数超过 soft：略降低 min_importance（更易删低权重旧记忆）
    adj_mi = base_mi - scale * max(0.0, (base_mi - floor) * 0.9)
    mi = max(floor, min(ceil, adj_mi))

    days = max(10, base_days - int(scale * 35))
    boost = int(at.get("limit_boost_per_1k_rows", 45) or 45)
    lim_cap = int(at.get("limit_max", 1200) or 1200)
    lim = base_lim + int((n // 1000) * boost)
    lim = max(base_lim, min(lim_cap, lim))

    debug.update(
        {
            "auto_scale": round(scale, 4),
            "resolved_min_importance": mi,
            "resolved_older_than_days": days,
            "resolved_limit": lim,
        }
    )
    return mi, days, lim, debug


def steward_after_learn(agent: "LifersAgent") -> Dict[str, Any]:
    """每轮 taskflow 学习后调用：按策略修剪 taskflow 记忆。"""
    if os.environ.get("LIFERS_STEWARD", "1").strip().lower() in ("0", "false", "no", "off"):
        return {"skipped": True}
    cfg = _deep_cfg(agent.cfg.root_dir)
    if not cfg.get("enabled", True):
        return {"skipped": True, "reason": "deep_steward.enabled=false"}
    days = int(cfg.get("prune_taskflow_older_than_days", 14) or 14)
    limit = int(cfg.get("prune_taskflow_max_delete", 800) or 800)
    out: Dict[str, Any]
    try:
        out = agent.longterm.prune_type_older_than("taskflow", older_than_days=days, limit=limit)
    except Exception as e:
        return {"error": str(e)}

    gf = (cfg.get("global_forget") or {}) if isinstance(cfg, dict) else {}
    if isinstance(gf, dict) and gf.get("enabled"):
        try:
            mi, od, lim, dbg = _resolve_global_forget_params(gf, agent)
            out["global_forget_params"] = dbg
            out["global_forget"] = agent.longterm.prune(
                min_importance=float(mi),
                older_than_days=int(od),
                limit=int(lim),
            )
        except Exception as e:
            out["global_forget_error"] = str(e)
    return out
