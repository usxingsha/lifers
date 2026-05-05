from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class SimResult:
    task_id: str
    runs: int
    success_rate: float
    metrics: Dict[str, Any]


class SimExecutor:
    def run(self, task: Dict[str, Any], runs: int) -> SimResult:  # pragma: no cover
        raise NotImplementedError


class StubSimExecutor(SimExecutor):
    def run(self, task: Dict[str, Any], runs: int) -> SimResult:
        rng = random.Random(1)
        ok = max(0, runs - 1)
        metrics = {
            "mean_jerk": rng.uniform(0.5, 1.5),
            "max_jerk": rng.uniform(1.0, 3.0),
        }
        return SimResult(task_id=str(task.get("id")), runs=runs, success_rate=ok / max(1, runs), metrics=metrics)


class ExternalCmdSimExecutor(SimExecutor):
    """
    Adapter for Linux/ROS2/sim without heavy deps:
    - Provide SIM_EXEC_CMD which points to an executable/script.
    - We write task JSON to a temp file and expect JSON result in another file.
    - Command will be invoked as: <SIM_EXEC_CMD> <task_json_path> <out_json_path>
    Result JSON must include: {success_rate: float, metrics: object}
    """

    def __init__(self, cmd: str) -> None:
        self.cmd = cmd

    def run(self, task: Dict[str, Any], runs: int) -> SimResult:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            task_path = td_path / "task.json"
            out_path = td_path / "out.json"
            payload = {"task": task, "runs": runs}
            task_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            p = subprocess.run(
                [self.cmd, str(task_path), str(out_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if p.returncode != 0:
                raise RuntimeError(f"SIM_EXEC_CMD failed: rc={p.returncode} stderr={p.stderr[:500]}")
            if not out_path.exists():
                raise RuntimeError("SIM_EXEC_CMD did not produce out.json")
            out = json.loads(out_path.read_text(encoding="utf-8"))
            sr = float(out.get("success_rate", 0.0))
            metrics = out.get("metrics", {})
            return SimResult(task_id=str(task.get("id")), runs=runs, success_rate=sr, metrics=metrics)


def load_tasks(root: Path) -> List[Dict[str, Any]]:
    from .stack_env import load_stack

    rel_default = "sim/tasks/tasks_v001.jsonl"
    rel = str((load_stack(root).get("robot") or {}).get("tasks_file", rel_default)).strip() or rel_default
    path = Path(rel) if Path(rel).is_absolute() else (root / rel)
    if not path.exists():
        return []
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        tasks.append(json.loads(line))
    return tasks


def default_executor() -> SimExecutor:
    cmd = os.environ.get("SIM_EXEC_CMD", "").strip()
    if cmd:
        return ExternalCmdSimExecutor(cmd=cmd)
    return StubSimExecutor()

