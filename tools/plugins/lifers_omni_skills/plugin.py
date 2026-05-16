"""
全能AI技能插件 — lifers_omni_skills

为Lifers添加实用工具技能：
  calculator  — 安全数学计算
  grep_search — 递归文件内容搜索
  sys_monitor — 系统资源监控
  csv_process — CSV读写处理
  json_query  — JSON数据查询
  file_download — 二进制文件下载
  git_helper  — Git基础操作

启用方式：
  stack.json: {"plugins": {"enabled": true}}
  或环境变量: LIFERS_PLUGINS=1
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional

# Tool base classes (imported at registration time to avoid circular imports)


def register_plugin_tools(registry, root: Path) -> None:
    """注册全能AI技能工具到ToolRegistry。"""
    from lifers.tools import Tool, ToolCall, ToolResult, ToolSpec

    # =========================================================================
    # 1. 安全计算器
    # =========================================================================
    class CalculatorTool(Tool):
        """安全地计算数学表达式，支持基本运算、三角函数、常量。"""

        spec = ToolSpec(
            name="calculator",
            args_schema={"expression": "string, math expression to evaluate"},
            permissions=["cpu:compute"],
            supports_modes=("dry_run", "execute", "verify", "rollback"),
            risk_level="low",
        )

        _SAFE_BUILTINS = {
            "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
            "int": int, "float": float, "str": str, "bool": bool,
            "len": len, "range": range, "list": list, "tuple": tuple,
        }
        _SAFE_MATH = {
            k: getattr(math, k)
            for k in [
                "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
                "sinh", "cosh", "tanh", "exp", "log", "log2", "log10",
                "pow", "ceil", "floor", "fabs", "fmod", "degrees", "radians",
                "pi", "e", "tau", "inf", "nan", "isinf", "isnan",
            ]
        }
        _SAFE_NAMES = {**_SAFE_BUILTINS, **_SAFE_MATH}

        def dry_run(self, call: ToolCall) -> ToolResult:
            expr = str(call.args.get("expression", "")).strip()
            return ToolResult(ok=True, data={"would_evaluate": expr[:120]})

        def execute(self, call: ToolCall) -> ToolResult:
            expr = str(call.args.get("expression", "")).strip()
            if not expr:
                return ToolResult(ok=False, error="expression为空")
            if len(expr) > 2000:
                return ToolResult(ok=False, error="表达式过长（限2000字符）")
            # 安全限制：只允许安全字符
            if re.search(r"[^0-9+\-*/().,%\s\w\[\]:'\"<>=!&|^~@]", expr):
                return ToolResult(ok=False, error="表达式包含不安全字符")
            # 禁止危险关键字
            forbidden = ["__", "import", "exec", "eval", "compile", "open", "write",
                        "delete", "remove", "system", "popen", "subprocess"]
            for kw in forbidden:
                if kw in expr.lower():
                    return ToolResult(ok=False, error=f"禁止使用关键字: {kw}")
            try:
                result = eval(expr, {"__builtins__": {}}, self._SAFE_NAMES)
                return ToolResult(ok=True, data={"expression": expr, "result": result})
            except Exception as e:
                return ToolResult(ok=False, error=f"计算失败: {e}")

        def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
            return ToolResult(ok=True, data={"verified": prior.data if prior else {}})

        def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
            return ToolResult(ok=True, data={"rolled_back": True})

    # =========================================================================
    # 2. 文件内容搜索
    # =========================================================================
    class GrepSearchTool(Tool):
        """递归搜索文件内容（类似 grep -r），支持正则表达式和文件过滤。"""

        spec = ToolSpec(
            name="grep_search",
            args_schema={
                "pattern": "string, regex or literal text to search",
                "directory": "string, starting directory (default: LIFERS_ROOT)",
                "glob_filter": "string, e.g. *.py or *.md (optional)",
                "max_results": "int, max matches to return (default: 50)",
                "case_sensitive": "bool (default: false)",
                "regex": "bool, treat pattern as regex (default: true)",
            },
            permissions=["fs:read"],
            supports_modes=("dry_run", "execute", "verify", "rollback"),
            risk_level="low",
        )

        def dry_run(self, call: ToolCall) -> ToolResult:
            pat = str(call.args.get("pattern", ""))
            return ToolResult(ok=True, data={"would_search": pat})

        def execute(self, call: ToolCall) -> ToolResult:
            pattern = str(call.args.get("pattern", "")).strip()
            if not pattern:
                return ToolResult(ok=False, error="pattern为空")
            if len(pattern) > 500:
                return ToolResult(ok=False, error="搜索模式过长")

            start_dir = Path(str(call.args.get("directory", str(root))))
            if not start_dir.is_absolute():
                start_dir = root / start_dir
            if not start_dir.is_dir():
                return ToolResult(ok=False, error=f"目录不存在: {start_dir}")

            glob_filter = str(call.args.get("glob_filter", "*")).strip() or "*"
            max_results = int(call.args.get("max_results", 50))
            case_sensitive = bool(call.args.get("case_sensitive", False))
            use_regex = bool(call.args.get("regex", True))

            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                if use_regex:
                    compiled = re.compile(pattern, flags)
                else:
                    compiled = re.compile(re.escape(pattern), flags)
            except re.error as e:
                return ToolResult(ok=False, error=f"正则错误: {e}")

            results: List[dict] = []
            try:
                for fpath in start_dir.rglob(glob_filter):
                    if fpath.is_dir():
                        continue
                    # 跳过二进制和隐藏目录
                    if any(p.startswith(".") for p in fpath.parts):
                        continue
                    if fpath.suffix in (".npz", ".pyc", ".sqlite3", ".sqlite3-wal", ".sqlite3-shm",
                                        ".png", ".jpg", ".jpeg", ".gif", ".ico", ".exe", ".dll",
                                        ".pem", ".zip", ".tar", ".gz"):
                        continue
                    try:
                        text = fpath.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        continue
                    for lineno, line in enumerate(text.splitlines(), 1):
                        if compiled.search(line):
                            rel = str(fpath.relative_to(start_dir))
                            results.append({
                                "file": rel,
                                "line": lineno,
                                "text": line.strip()[:200],
                            })
                            if len(results) >= max_results:
                                break
                    if len(results) >= max_results:
                        break
            except (OSError, PermissionError) as e:
                return ToolResult(ok=False, error=f"搜索错误: {e}")

            return ToolResult(ok=True, data={
                "pattern": pattern,
                "directory": str(start_dir),
                "matches": len(results),
                "truncated": len(results) >= max_results,
                "results": results,
            })

        def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
            return ToolResult(ok=True, data={"verified": True})

        def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
            return ToolResult(ok=True, data={"rolled_back": True})

    # =========================================================================
    # 3. 系统监控
    # =========================================================================
    class SysMonitorTool(Tool):
        """获取系统资源信息：CPU/内存/磁盘。"""

        spec = ToolSpec(
            name="sys_monitor",
            args_schema={"metric": "cpu|memory|disk|all (default: all)"},
            permissions=["sys:read"],
            supports_modes=("dry_run", "execute", "verify", "rollback"),
            risk_level="low",
        )

        def dry_run(self, call: ToolCall) -> ToolResult:
            return ToolResult(ok=True, data={"would_check": str(call.args.get("metric", "all"))})

        def execute(self, call: ToolCall) -> ToolResult:
            metric = str(call.args.get("metric", "all")).strip().lower()
            data: dict = {}

            try:
                import platform
                data["platform"] = platform.platform()
                data["python"] = sys.version.split()[0]
            except Exception:
                pass

            if metric in ("cpu", "all"):
                try:
                    import psutil
                    data["cpu_percent"] = psutil.cpu_percent(interval=0.3)
                    data["cpu_count"] = psutil.cpu_count()
                except ImportError:
                    try:
                        # Fallback: read /proc/stat on Linux
                        with open("/proc/stat") as f:
                            data["cpu_info"] = f.readline().strip()[:120]
                    except Exception:
                        data["cpu_info"] = "unavailable"

            if metric in ("memory", "all"):
                try:
                    import psutil
                    mem = psutil.virtual_memory()
                    data["memory_total_gb"] = round(mem.total / (1024**3), 1)
                    data["memory_used_gb"] = round(mem.used / (1024**3), 1)
                    data["memory_percent"] = mem.percent
                except ImportError:
                    data["memory_info"] = "psutil not installed"

            if metric in ("disk", "all"):
                try:
                    import psutil
                    disk = psutil.disk_usage(str(root))
                    data["disk_total_gb"] = round(disk.total / (1024**3), 1)
                    data["disk_used_gb"] = round(disk.used / (1024**3), 1)
                    data["disk_percent"] = disk.percent
                except ImportError:
                    try:
                        import shutil
                        usage = shutil.disk_usage(str(root))
                        data["disk_total_gb"] = round(usage.total / (1024**3), 1)
                        data["disk_used_gb"] = round(usage.used / (1024**3), 1)
                    except Exception:
                        data["disk_info"] = "unavailable"

            return ToolResult(ok=True, data=data)

        def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
            return ToolResult(ok=True, data={"verified": True})

        def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
            return ToolResult(ok=True, data={"rolled_back": True})

    # =========================================================================
    # 4. CSV 处理
    # =========================================================================
    class CsvProcessTool(Tool):
        """读写CSV文件，支持查询、统计、导出。"""

        spec = ToolSpec(
            name="csv_process",
            args_schema={
                "action": "read|write|stats",
                "file": "string, path relative to LIFERS_ROOT or absolute",
                "data": "list of rows for write action (optional)",
                "query": "dict with column filters (optional)",
                "limit": "int, max rows for read (default: 100)",
            },
            permissions=["fs:read", "fs:write"],
            supports_modes=("dry_run", "execute", "verify", "rollback"),
            risk_level="medium",
        )

        def dry_run(self, call: ToolCall) -> ToolResult:
            return ToolResult(ok=True, data={"would": str(call.args.get("action", "read"))})

        def execute(self, call: ToolCall) -> ToolResult:
            action = str(call.args.get("action", "read")).strip().lower()
            file_path = Path(str(call.args.get("file", "")))
            if not file_path.is_absolute():
                file_path = root / file_path

            if action == "read":
                if not file_path.is_file():
                    return ToolResult(ok=False, error=f"文件不存在: {file_path}")
                limit = int(call.args.get("limit", 100))
                try:
                    text = file_path.read_text(encoding="utf-8", errors="replace")
                    reader = csv.DictReader(io.StringIO(text))
                    rows = []
                    columns = reader.fieldnames or []
                    for i, row in enumerate(reader):
                        if i >= limit:
                            break
                        rows.append(row)
                    return ToolResult(ok=True, data={
                        "file": str(file_path),
                        "columns": columns,
                        "row_count": len(rows),
                        "rows": rows,
                    })
                except Exception as e:
                    return ToolResult(ok=False, error=f"读取CSV失败: {e}")

            elif action == "write":
                data_rows = call.args.get("data", [])
                if not data_rows:
                    return ToolResult(ok=False, error="data为空")
                try:
                    if not data_rows:
                        return ToolResult(ok=False, error="无数据行")
                    fieldnames = list(data_rows[0].keys())
                    buf = io.StringIO()
                    writer = csv.DictWriter(buf, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data_rows)
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(buf.getvalue(), encoding="utf-8")
                    return ToolResult(ok=True, data={
                        "file": str(file_path),
                        "rows_written": len(data_rows),
                        "columns": fieldnames,
                    })
                except Exception as e:
                    return ToolResult(ok=False, error=f"写入CSV失败: {e}")

            elif action == "stats":
                if not file_path.is_file():
                    return ToolResult(ok=False, error=f"文件不存在: {file_path}")
                try:
                    text = file_path.read_text(encoding="utf-8", errors="replace")
                    reader = csv.DictReader(io.StringIO(text))
                    rows = list(reader)
                    columns = reader.fieldnames or []
                    stats = {
                        "file": str(file_path),
                        "total_rows": len(rows),
                        "columns": columns,
                        "column_count": len(columns),
                    }
                    # Per-column stats for numeric columns
                    col_stats = {}
                    for col in columns:
                        vals = []
                        for row in rows:
                            try:
                                vals.append(float(row.get(col, "")))
                            except (ValueError, TypeError):
                                pass
                        if vals:
                            col_stats[col] = {
                                "count": len(vals),
                                "min": min(vals),
                                "max": max(vals),
                                "mean": round(sum(vals) / len(vals), 4),
                            }
                    stats["numeric_columns"] = col_stats
                    return ToolResult(ok=True, data=stats)
                except Exception as e:
                    return ToolResult(ok=False, error=f"统计失败: {e}")

            return ToolResult(ok=False, error=f"未知操作: {action}")

        def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
            return ToolResult(ok=True, data={"verified": prior.data if prior else {}})

        def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
            return ToolResult(ok=True, data={"rolled_back": True})

    # =========================================================================
    # 5. JSON 查询
    # =========================================================================
    class JsonQueryTool(Tool):
        """读取JSON文件并支持路径查询（a.b[0].c 语法）。"""

        spec = ToolSpec(
            name="json_query",
            args_schema={
                "file": "string, path to JSON file",
                "query_path": "string, dot-path like 'key.subkey[0].field' (optional, returns all)",
                "action": "read|keys|count (default: read)",
            },
            permissions=["fs:read"],
            supports_modes=("dry_run", "execute", "verify", "rollback"),
            risk_level="low",
        )

        def dry_run(self, call: ToolCall) -> ToolResult:
            return ToolResult(ok=True, data={"would_query": str(call.args.get("file", ""))})

        def execute(self, call: ToolCall) -> ToolResult:
            file_path = Path(str(call.args.get("file", "")))
            if not file_path.is_absolute():
                file_path = root / file_path
            if not file_path.is_file():
                return ToolResult(ok=False, error=f"文件不存在: {file_path}")

            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception as e:
                return ToolResult(ok=False, error=f"JSON解析失败: {e}")

            action = str(call.args.get("action", "read")).strip().lower()
            query_path = str(call.args.get("query_path", "")).strip()

            if action == "keys":
                if isinstance(data, dict):
                    return ToolResult(ok=True, data={"keys": list(data.keys()), "type": "object"})
                elif isinstance(data, list):
                    return ToolResult(ok=True, data={"length": len(data), "type": "array"})
                return ToolResult(ok=True, data={"type": type(data).__name__})

            if action == "count":
                if isinstance(data, list):
                    return ToolResult(ok=True, data={"count": len(data)})
                elif isinstance(data, dict):
                    return ToolResult(ok=True, data={"count": len(data)})
                return ToolResult(ok=True, data={"count": 1})

            # Navigate path: "a.b[0].c"
            if query_path:
                try:
                    current = data
                    tokens = re.split(r"\.|(?=\[)", query_path)
                    for token in tokens:
                        token = token.strip()
                        if not token:
                            continue
                        if token.startswith("[") and token.endswith("]"):
                            idx = int(token[1:-1])
                            current = current[idx]
                        else:
                            current = current[token]
                    return ToolResult(ok=True, data={"result": current, "path": query_path})
                except (KeyError, IndexError, TypeError, ValueError) as e:
                    return ToolResult(ok=False, error=f"路径查询失败: {e} ({query_path})")

            return ToolResult(ok=True, data={"result": data})

        def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
            return ToolResult(ok=True, data={"verified": True})

        def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
            return ToolResult(ok=True, data={"rolled_back": True})

    # =========================================================================
    # 6. 文件下载
    # =========================================================================
    class FileDownloadTool(Tool):
        """下载文件（支持文本和二进制），保存到指定路径。"""

        spec = ToolSpec(
            name="file_download",
            args_schema={
                "url": "string, URL to download",
                "save_path": "string, relative to LIFERS_ROOT or absolute",
                "binary": "bool (default: auto-detect from extension)",
                "timeout_sec": "float (default: 30)",
            },
            permissions=["net:read", "fs:write"],
            supports_modes=("dry_run", "execute", "verify", "rollback"),
            risk_level="medium",
        )

        def dry_run(self, call: ToolCall) -> ToolResult:
            return ToolResult(ok=True, data={"would_download": str(call.args.get("url", ""))})

        def execute(self, call: ToolCall) -> ToolResult:
            url = str(call.args.get("url", "")).strip()
            if not url:
                return ToolResult(ok=False, error="url为空")
            if not url.startswith(("http://", "https://")):
                return ToolResult(ok=False, error="仅支持http/https")

            save_path = Path(str(call.args.get("save_path", "")))
            if not save_path.is_absolute():
                save_path = root / save_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            timeout = float(call.args.get("timeout_sec", 30))
            binary = call.args.get("binary", None)
            if binary is None:
                binary = save_path.suffix.lower() in (
                    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip",
                    ".tar", ".gz", ".exe", ".dll", ".npz", ".mp3", ".mp4",
                    ".wav", ".webp", ".svg", ".ttf", ".woff", ".woff2",
                )

            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Lifers/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    content = resp.read()
                    if binary:
                        save_path.write_bytes(content)
                    else:
                        save_path.write_text(content.decode("utf-8", errors="replace"), encoding="utf-8")
                size_kb = len(content) / 1024
                return ToolResult(ok=True, data={
                    "url": url,
                    "saved_to": str(save_path),
                    "size_kb": round(size_kb, 1),
                    "binary": binary,
                })
            except urllib.error.URLError as e:
                return ToolResult(ok=False, error=f"下载失败: {e}")
            except Exception as e:
                return ToolResult(ok=False, error=f"保存失败: {e}")

        def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
            if prior and prior.ok:
                sp = prior.data.get("saved_to", "")
                if sp and Path(sp).is_file():
                    return ToolResult(ok=True, data={"verified": True, "file_exists": True})
            return ToolResult(ok=True, data={"verified": False})

        def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
            if prior and prior.ok:
                sp = prior.data.get("saved_to", "")
                try:
                    Path(sp).unlink(missing_ok=True)
                    return ToolResult(ok=True, data={"removed": sp})
                except Exception as e:
                    return ToolResult(ok=False, error=str(e))
            return ToolResult(ok=True, data={"nothing_to_rollback": True})

    # =========================================================================
    # 7. Git 操作助手
    # =========================================================================
    class GitHelperTool(Tool):
        """执行基础Git操作：status/diff/log/branch。"""

        spec = ToolSpec(
            name="git_helper",
            args_schema={
                "action": "status|diff|log|branch|pull",
                "path": "string, repo path (default: LIFERS_ROOT)",
                "max_entries": "int, max log entries (default: 10)",
            },
            permissions=["sys:exec"],
            supports_modes=("dry_run", "execute", "verify", "rollback"),
            risk_level="medium",
        )

        def _git_dir(self, call: ToolCall) -> Path:
            p = Path(str(call.args.get("path", str(root))))
            if not p.is_absolute():
                p = root / p
            return p

        def dry_run(self, call: ToolCall) -> ToolResult:
            return ToolResult(ok=True, data={"would": str(call.args.get("action", "status"))})

        def execute(self, call: ToolCall) -> ToolResult:
            action = str(call.args.get("action", "status")).strip().lower()
            repo = self._git_dir(call)
            max_entries = int(call.args.get("max_entries", 10))

            safe_actions = {
                "status": ["git", "status", "--short"],
                "diff": ["git", "diff", "--stat"],
                "log": ["git", "log", f"--oneline", f"-n{max_entries}"],
                "branch": ["git", "branch", "--list"],
            }

            if action not in safe_actions:
                return ToolResult(ok=False, error=f"不支持的操作: {action}。支持: {list(safe_actions)}")

            cmd = safe_actions[action]
            try:
                result = subprocess.run(
                    cmd, cwd=str(repo), capture_output=True, text=True, timeout=15,
                )
                return ToolResult(ok=True, data={
                    "action": action,
                    "repo": str(repo),
                    "output": result.stdout.strip() or "(empty)",
                    "stderr": result.stderr.strip()[:500],
                    "exit_code": result.returncode,
                })
            except subprocess.TimeoutExpired:
                return ToolResult(ok=False, error="Git命令超时")
            except FileNotFoundError:
                return ToolResult(ok=False, error="Git未安装或不在PATH中")
            except Exception as e:
                return ToolResult(ok=False, error=str(e))

        def verify(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
            return ToolResult(ok=True, data={"verified": True})

        def rollback(self, call: ToolCall, prior: Optional[ToolResult] = None) -> ToolResult:
            return ToolResult(ok=True, data={"rolled_back": True})

    # =========================================================================
    # Register all tools
    # =========================================================================
    registry.register(CalculatorTool())
    registry.register(GrepSearchTool())
    registry.register(SysMonitorTool())
    registry.register(CsvProcessTool())
    registry.register(JsonQueryTool())
    registry.register(FileDownloadTool())
    registry.register(GitHelperTool())
