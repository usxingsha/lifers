from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _run_py(module_path: Path) -> None:
    # Run by importing/execing to avoid depending on subprocess (still stdlib, but simpler).
    ns = {"__name__": "__main__", "__file__": str(module_path)}
    code = module_path.read_text(encoding="utf-8")
    exec(compile(code, str(module_path), "exec"), ns, ns)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    # Allow running without installing the package.
    sys.path.insert(0, str(root))
    from lifers.stack_env import apply_stack_env

    apply_stack_env(root)
    os.environ.setdefault("SANDBOX", "1")
    from lifers.model_names import canonical_brain_model

    model = canonical_brain_model(os.environ.get("MODEL", "markov"))

    try:
        from lifers.llm_ops_context import format_llm_ops_context
        from lifers.stack_env import load_stack

        if format_llm_ops_context(load_stack(root), root).strip():
            print("[pipeline] stack.llm_ops 已配置：日常说明与权重流水线并列（config/stack.json#llm_ops）")
    except Exception:
        pass

    print(f"[pipeline] root={root}")
    print(f"[pipeline] SANDBOX={os.environ.get('SANDBOX')}")
    print(f"[pipeline] MODEL={model}")

    t0 = time.time()
    if model == "transformer":
        _run_py(root / "scripts" / "train_transformer_weights.py")
    else:
        os.environ["MODEL"] = "markov"
        _run_py(root / "scripts" / "train_weights.py")
    _run_py(root / "eval" / "run_eval.py")
    _run_py(root / "eval" / "smoke.py")
    _run_py(root / "sim" / "sim_eval.py")

    # Minimal “promote” gate: require eval pass_rate >= 0.8 and smoke empty_rate <= 0.05
    eval_report = json.loads((root / "exp_eval_report.json").read_text(encoding="utf-8"))
    smoke_report = json.loads((root / "exp_smoke.json").read_text(encoding="utf-8"))
    sim_report = json.loads((root / "exp_sim_report.json").read_text(encoding="utf-8"))

    ok = True
    if eval_report.get("pass_rate", 0) < 0.8:
        ok = False
        print("[gate] eval pass_rate too low")
    if smoke_report.get("empty_rate", 1) > 0.05:
        ok = False
        print("[gate] smoke empty_rate too high")
    if sim_report.get("overall_success_rate", 0) < 0.7:
        ok = False
        print("[gate] sim success too low")

    status = {
        "ok": ok,
        "eval_pass_rate": eval_report.get("pass_rate"),
        "smoke_empty_rate": smoke_report.get("empty_rate"),
        "sim_success": sim_report.get("overall_success_rate"),
        "elapsed_s": time.time() - t0,
    }
    (root / "exp_pipeline_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[pipeline] wrote exp_pipeline_status.json ok={ok}")
    print("[pipeline] 下一步：python scripts/check_lifers_llm_ready.py（权重+vendor 树）；对话闭环说明见 config/lifers_llm_bootstrap.json")


if __name__ == "__main__":
    main()

