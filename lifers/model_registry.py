"""
Lifers Model Registry v1 — 模型版本管理
版本注册、A/B 切换、回滚、校验和、元数据索引
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from threading import Lock

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DIR = ROOT / "weights" / ".registry"
VERSIONS_FILE = REGISTRY_DIR / "versions.json"


@dataclass
class ModelVersion:
    name: str
    version: int
    file: str
    checksum: str
    created_at: float
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    parent_version: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """模型版本注册表"""

    def __init__(self):
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._index: Dict[str, List[ModelVersion]] = {}
        self._load()

    def _load(self):
        if VERSIONS_FILE.exists():
            with open(VERSIONS_FILE, "r") as f:
                data = json.load(f)
                for name, versions in data.items():
                    self._index[name] = [ModelVersion(**v) for v in versions]

    def _save(self):
        data = {}
        for name, versions in self._index.items():
            data[name] = [{
                "name": v.name, "version": v.version, "file": v.file,
                "checksum": v.checksum, "created_at": v.created_at,
                "metrics": v.metrics, "tags": v.tags,
                "parent_version": v.parent_version, "metadata": v.metadata,
            } for v in versions]
        with open(VERSIONS_FILE, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def checksum_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def register(self, name: str, file_path: Path, metrics: Dict[str, float] = None,
                 tags: List[str] = None, metadata: Dict[str, Any] = None) -> ModelVersion:
        with self._lock:
            versions = self._index.get(name, [])
            next_version = max([v.version for v in versions], default=0) + 1

            # 存档
            archive_dir = REGISTRY_DIR / name
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / f"v{next_version:04d}_{file_path.name}"
            shutil.copy2(file_path, archive_path)

            mv = ModelVersion(
                name=name,
                version=next_version,
                file=str(archive_path.relative_to(ROOT)),
                checksum=self.checksum_file(file_path),
                created_at=time.time(),
                metrics=metrics or {},
                tags=tags or [],
                parent_version=max([v.version for v in versions], default=0) if versions else None,
                metadata=metadata or {},
            )

            if name not in self._index:
                self._index[name] = []
            self._index[name].append(mv)
            self._save()

            # 更新 latest 符号链接
            latest_link = archive_dir / "latest.json"
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(archive_path.name)

            print(f"[Registry] {name} v{next_version} registered "
                  f"(checksum={mv.checksum[:12]}... metrics={mv.metrics})")
            return mv

    def get_latest(self, name: str) -> Optional[ModelVersion]:
        versions = self._index.get(name, [])
        if not versions:
            return None
        return versions[-1]

    def get_version(self, name: str, version: int) -> Optional[ModelVersion]:
        versions = self._index.get(name, [])
        for v in versions:
            if v.version == version:
                return v
        return None

    def get_by_tag(self, name: str, tag: str) -> Optional[ModelVersion]:
        versions = self._index.get(name, [])
        for v in reversed(versions):
            if tag in v.tags:
                return v
        return None

    def list_versions(self, name: str) -> List[ModelVersion]:
        return self._index.get(name, [])

    def list_models(self) -> List[str]:
        return list(self._index.keys())

    def get_best(self, name: str, metric: str, higher_is_better: bool = True) -> Optional[ModelVersion]:
        versions = self._index.get(name, [])
        if not versions:
            return None
        scored = [(v, v.metrics.get(metric, -float("inf"))) for v in versions]
        if higher_is_better:
            return max(scored, key=lambda x: x[1])[0]
        return min(scored, key=lambda x: x[1])[0]

    def rollback(self, name: str, version: int) -> bool:
        mv = self.get_version(name, version)
        if not mv:
            return False
        source = ROOT / mv.file
        target = ROOT / "weights" / Path(mv.file).name.replace(f"v{version:04d}_", "")
        shutil.copy2(source, target)
        print(f"[Registry] Rollback {name} to v{version} -> {target}")
        return True

    def compare(self, name: str, v1: int, v2: int) -> dict:
        mv1 = self.get_version(name, v1)
        mv2 = self.get_version(name, v2)
        if not mv1 or not mv2:
            return {"error": "version not found"}
        return {
            "v1": {"version": mv1.version, "metrics": mv1.metrics, "created": mv1.created_at},
            "v2": {"version": mv2.version, "metrics": mv2.metrics, "created": mv2.created_at},
            "metrics_diff": {k: mv2.metrics.get(k, 0) - mv1.metrics.get(k, 0)
                             for k in set(mv1.metrics) | set(mv2.metrics)},
            "size_diff_bytes": (Path(ROOT / mv2.file).stat().st_size -
                               Path(ROOT / mv1.file).stat().st_size),
        }

    def cleanup_old(self, name: str, keep_last: int = 10):
        versions = self._index.get(name, [])
        if len(versions) <= keep_last:
            return
        with self._lock:
            to_remove = versions[:-keep_last]
            for mv in to_remove:
                archive = ROOT / mv.file
                if archive.exists():
                    archive.unlink()
                self._index[name].remove(mv)
            self._save()

    def autoregister_checkpoint(self, pillar: str, metrics: Dict[str, float] = None):
        """自动注册训练检查点"""
        weights_map = {
            "safety": "lifers_safety_classifier.json",
            "social": "lifers_social_classifier.json",
            "perception": "lifers_perception_classifier.json",
            "proactive": "lifers_proactive_predictor.json",
            "voice": "lifers_voice_acoustic.json",
            "kg": "lifers_kg_embeddings.json",
            "rl": "lifers_rl_policy.json",
            "robot_hal": "lifers_robot_hal_policy.json",
            "swarm": "lifers_swarm_policy.json",
            "simulation": "lifers_simulation_evaluator.json",
            "telemetry": "lifers_telemetry_detector.json",
            "dashboard": "lifers_dashboard_config.json",
        }
        weight_file = weights_map.get(pillar)
        if not weight_file:
            return
        wpath = ROOT / "weights" / weight_file
        if wpath.exists():
            self.register(pillar, wpath, metrics or {},
                         tags=[] if metrics else [],
                         metadata={"source": "autoregister"})


# ============================================================================
# A/B 模型切换器
# ============================================================================

class ABRouter:
    """A/B 模型路由 — 金丝雀/蓝绿部署"""

    def __init__(self):
        self._registry = ModelRegistry()
        self._active: Dict[str, int] = {}  # model_name -> active_version
        self._staged: Dict[str, int] = {}  # model_name -> staged_version
        self._traffic_split: Dict[str, float] = {}  # model_name -> canary_pct
        self._load_config()

    def _load_config(self):
        config_file = REGISTRY_DIR / "ab_config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                data = json.load(f)
                self._active = data.get("active", {})
                self._staged = data.get("staged", {})
                self._traffic_split = data.get("traffic_split", {})

    def _save_config(self):
        config_file = REGISTRY_DIR / "ab_config.json"
        with open(config_file, "w") as f:
            json.dump({
                "active": self._active, "staged": self._staged,
                "traffic_split": self._traffic_split,
            }, f, ensure_ascii=False, indent=2)

    def promote(self, name: str, version: int, canary_pct: float = 0.0):
        if canary_pct > 0:
            self._staged[name] = version
            self._traffic_split[name] = canary_pct
        else:
            self._active[name] = version
            self._staged.pop(name, None)
            self._traffic_split.pop(name, None)
        self._save_config()

    def rollback_active(self, name: str):
        versions = self._registry.list_versions(name)
        if len(versions) < 2:
            return
        current = self._active.get(name)
        prev = None
        for v in versions:
            if v.version == current:
                break
            prev = v.version
        if prev:
            self._active[name] = prev
            self._registry.rollback(name, prev)
            self._save_config()

    def get_active_version(self, name: str) -> Optional[int]:
        return self._active.get(name)

    def resolve_route(self, name: str, request_id: str = "") -> int:
        """金丝雀路由决定"""
        active = self._active.get(name)
        staged = self._staged.get(name)
        if staged is None:
            return active

        pct = self._traffic_split.get(name, 0)
        if pct <= 0:
            return active

        h = hashlib.md5(request_id.encode()).hexdigest() if request_id else os.urandom(8).hex()
        bucket = int(h[:8], 16) % 100
        return staged if bucket < pct else active

    def compare_active_staged(self, name: str) -> dict:
        active_v = self._active.get(name)
        staged_v = self._staged.get(name)
        if not active_v or not staged_v:
            return {"error": "need both active and staged"}
        return self._registry.compare(name, active_v, staged_v)


# ============================================================================
# 全局实例
# ============================================================================

_registry_instance: Optional[ModelRegistry] = None
_ab_router: Optional[ABRouter] = None


def get_registry() -> ModelRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance


def get_ab_router() -> ABRouter:
    global _ab_router
    if _ab_router is None:
        _ab_router = ABRouter()
    return _ab_router
