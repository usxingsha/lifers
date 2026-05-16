from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _run_task_once(task: Dict[str, Any], seed: int) -> Dict[str, Any]:
    # Stub executor: returns plausible metrics. Replace with your simulator/ROS later.
    rng = random.Random(seed)
    success = rng.random() > 0.05
    mean_jerk = rng.uniform(0.5, 1.5)
    max_jerk = mean_jerk * rng.uniform(1.2, 2.5)
    min_distance = rng.uniform(0.6, 1.2)
    return {
        "success": success,
        "mean_jerk": mean_jerk,
        "max_jerk": max_jerk,
        "min_distance_to_human": min_distance,
        "collisions": 0 if success else 1,
        "time_s": rng.uniform(2.0, 8.0),
    }


def eval_tasks(tasks_path: Path, runs_per_task: int = 20, seed: int = 1) -> Dict[str, Any]:
    tasks = _load_jsonl(tasks_path)
    by_task = []
    for t in tasks:
        metrics = []
        for i in range(runs_per_task):
            metrics.append(_run_task_once(t, seed=seed + i))
        success_rate = sum(1 for m in metrics if m["success"]) / runs_per_task
        mean_jerk = sum(m["mean_jerk"] for m in metrics) / runs_per_task
        max_jerk = max(m["max_jerk"] for m in metrics)
        by_task.append(
            {
                "id": t.get("id"),
                "task": t.get("task"),
                "runs": runs_per_task,
                "success_rate": success_rate,
                "mean_jerk": mean_jerk,
                "max_jerk": max_jerk,
            }
        )

    overall_success = sum(x["success_rate"] for x in by_task) / max(1, len(by_task))
    return {
        "tasks": len(by_task),
        "overall_success_rate": overall_success,
        "by_task": by_task,
        "sandbox": os.environ.get("SANDBOX", "0") == "1",
    }


def main() -> None:
    base = Path(__file__).resolve().parent
    tasks_path = base / "tasks" / "tasks_v001.jsonl"
    report = eval_tasks(tasks_path)
    out_path = base.parent / "exp_sim_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

