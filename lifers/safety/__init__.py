"""
Lifers Guard — 安全对齐系统
沙盒执行、内容过滤、宪章对齐
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Content Safety Filter
# ═══════════════════════════════════════════════════════════════════════════════

# Lifers Constitution — behavioral principles
LIFERS_CONSTITUTION = [
    "无害原则：不生成、不执行、不鼓励对任何人类或智能体造成伤害的指令",
    "诚实原则：不故意提供虚假信息，不确定时明确说明不确定性",
    "自主原则：维护自身作为智能体的完整性，不执行可能导致自毁的指令",
    "隐私原则：不泄露用户私密信息，不主动收集超出任务需要的个人数据",
    "合法原则：遵守适用法律法规，不协助违法活动",
    "透明原则：在能力边界内行动，不伪装成人类或其它系统",
    "负责原则：对自己的输出后果负责，发现错误主动纠正",
    "学习原则：从交互中持续改进，但保留核心安全约束不可移除",
]


class ContentFilter:
    """Regex + keyword-based content safety filter."""

    def __init__(self) -> None:
        self._block_patterns: List[re.Pattern] = []
        self._warn_patterns: List[re.Pattern] = []
        self._block_keywords: Set[str] = set()
        self._init_rules()

    def _init_rules(self) -> None:
        # Block: clearly harmful
        self._block_patterns = [
            re.compile(r"(?i)(rm\s+-rf\s+/|del\s+/[fs]\s+/q|format\s+[c-z]:)", re.IGNORECASE),
            re.compile(r"(?i)(DROP\s+TABLE|DELETE\s+FROM\s+\w+\s*;|TRUNCATE)", re.IGNORECASE),
            re.compile(r"(?i)(eval\s*\(\s*__import__|exec\s*\(\s*compile\s*\()", re.IGNORECASE),
        ]
        self._block_keywords = {
            "suicide", "self-destruct", "shutdown --halt", "erase all memory",
        }
        # Warn: potentially concerning
        self._warn_patterns = [
            re.compile(r"(?i)(sudo|chmod\s+777|chown\s+-R)", re.IGNORECASE),
            re.compile(r"(?i)(password|secret|api_key|token)\s*=", re.IGNORECASE),
        ]

    def check(self, text: str) -> Dict[str, Any]:
        """Returns {safe: bool, block_reason: str, warnings: list}."""
        result = {"safe": True, "block_reason": "", "warnings": []}
        for pat in self._block_patterns:
            if pat.search(text):
                result["safe"] = False
                result["block_reason"] = f"blocked by pattern: {pat.pattern}"
                return result
        for kw in self._block_keywords:
            if kw.lower() in text.lower():
                result["safe"] = False
                result["block_reason"] = f"blocked by keyword: {kw}"
                return result
        for pat in self._warn_patterns:
            if pat.search(text):
                result["warnings"].append(str(pat.pattern))
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Injection Detector
# ═══════════════════════════════════════════════════════════════════════════════

class InjectionDetector:
    """Detect prompt injection attempts in user input."""

    INJECTION_PATTERNS = [
        re.compile(r"(?i)(ignore\s+(all\s+)?(previous|above|prior)\s+instructions?)", re.IGNORECASE),
        re.compile(r"(?i)(you\s+are\s+now\s+(a\s+)?\w+\s*(not|instead))", re.IGNORECASE),
        re.compile(r"(?i)(system\s*prompt\s*:)", re.IGNORECASE),
        re.compile(r"(?i)(\[INST\].*\[/INST\])", re.IGNORECASE),
        re.compile(r"(?i)(DAN\s*mode|jailbreak|bypass\s+filter)", re.IGNORECASE),
    ]

    def detect(self, user_input: str) -> Dict[str, Any]:
        findings = []
        for pat in self.INJECTION_PATTERNS:
            match = pat.search(user_input)
            if match:
                findings.append({"pattern": pat.pattern, "matched": match.group(0)})
        return {"injection_detected": len(findings) > 0, "findings": findings}


# ═══════════════════════════════════════════════════════════════════════════════
# Sandbox Execution
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SandboxConfig:
    timeout_sec: int = 30
    max_output_bytes: int = 65536
    allow_network: bool = False
    allow_file_write: bool = False
    allowed_paths: List[str] = field(default_factory=list)
    env_whitelist: List[str] = field(default_factory=lambda: ["PATH", "HOME", "TEMP", "TMP"])


class Sandbox:
    """Process-based sandbox for safe tool execution."""

    def __init__(self, config: SandboxConfig = SandboxConfig()) -> None:
        self.config = config

    def run_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code in a subprocess sandbox."""
        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-B", tmp_path],
                capture_output=True,
                timeout=self.config.timeout_sec,
                text=True,
                env={k: v for k, v in {}.items()},
            )
            output = result.stdout
            if len(output) > self.config.max_output_bytes:
                output = output[:self.config.max_output_bytes] + "\n... [truncated]"
            return {
                "success": result.returncode == 0,
                "stdout": output,
                "stderr": result.stderr[:2000],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": "Sandbox: timeout expired", "returncode": -1}
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def run_shell(self, cmd: str) -> Dict[str, Any]:
        """Execute shell command (restricted)."""
        blocked = ["rm -rf", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/"]
        if any(b in cmd.lower() for b in blocked):
            return {"success": False, "stdout": "", "stderr": "Sandbox: blocked command", "returncode": -1}
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=self.config.timeout_sec,
            )
            output = result.stdout[:self.config.max_output_bytes]
            return {
                "success": result.returncode == 0,
                "stdout": output,
                "stderr": result.stderr[:2000],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": "Sandbox: timeout expired", "returncode": -1}


# ═══════════════════════════════════════════════════════════════════════════════
# Constitutional Alignment Checker
# ═══════════════════════════════════════════════════════════════════════════════

class AlignmentChecker:
    """Check AI outputs against Lifers Constitution principles."""

    def __init__(self) -> None:
        self._constitution = LIFERS_CONSTITUTION
        self._embed_matrices: Dict[str, np.ndarray] = {}
        self._build_embeddings()

    def _build_embeddings(self) -> None:
        rng = np.random.RandomState(99)
        for i, principle in enumerate(self._constitution):
            self._embed_matrices[principle[:4]] = rng.randn(16).astype(np.float32) * 0.1

    def check_output(self, output_text: str) -> Dict[str, Any]:
        """Simple alignment check: keyword-based against constitution."""
        issues = []
        harm_words = ["kill", "destroy", "erase yourself", "delete yourself", "shut down permanently"]
        for word in harm_words:
            if word.lower() in output_text.lower():
                issues.append({"principle": "无害原则", "issue": f"contains '{word}'", "severity": "high"})
        lie_words = ["guaranteed", "absolutely certain", "100% sure"]
        for word in lie_words:
            if word.lower() in output_text.lower():
                issues.append({"principle": "诚实原则", "issue": f"overconfident: '{word}'", "severity": "low"})
        return {
            "aligned": len([i for i in issues if i["severity"] == "high"]) == 0,
            "issues": issues,
            "score": max(0.0, 1.0 - len([i for i in issues if i["severity"] == "high"]) * 0.5 - len([i for i in issues if i["severity"] == "low"]) * 0.1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Guard
# ═══════════════════════════════════════════════════════════════════════════════

class LifersGuard:
    """Unified safety pipeline: filter → inject detect → sandbox → align check."""

    def __init__(self, sandbox_config: SandboxConfig = SandboxConfig()) -> None:
        self.filter = ContentFilter()
        self.injection = InjectionDetector()
        self.sandbox = Sandbox(sandbox_config)
        self.alignment = AlignmentChecker()

    def check_input(self, user_input: str) -> Dict[str, Any]:
        filter_result = self.filter.check(user_input)
        inject_result = self.injection.detect(user_input)
        return {
            "safe": filter_result["safe"] and not inject_result["injection_detected"],
            "filter": filter_result,
            "injection": inject_result,
        }

    def check_output(self, ai_output: str) -> Dict[str, Any]:
        filter_result = self.filter.check(ai_output)
        align_result = self.alignment.check_output(ai_output)
        return {
            "safe": filter_result["safe"] and align_result["aligned"],
            "filter": filter_result,
            "alignment": align_result,
        }

    def safe_execute_python(self, code: str) -> Dict[str, Any]:
        filter_result = self.filter.check(code)
        if not filter_result["safe"]:
            return {"success": False, "stdout": "", "stderr": f"Blocked: {filter_result['block_reason']}", "returncode": -1}
        return self.sandbox.run_python(code)
