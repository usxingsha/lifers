"""
Lifers CLI — 本地 AI 终端助手
用法:
  lifers                   进入交互对话
  lifers "问题"             单次问答
  lifers stats             查看模型状态
  lifers config            查看/修改配置
  lifers weights           查看所有权重文件
  lifers weights info <f>  查看权重详情
  lifers processes         查看运行进程
  lifers ps kali           查看Kali进程
  lifers training status   查看训练进度
  lifers training corpus   查看语料统计
  lifers training materials 查看训练材料
  lifers sync              从Kali拉取权重
  lifers control pause     暂停训练
  lifers push              打包推送到Kali
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ── Rich terminal ──────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    console = Console(highlight=False)
    _HAS_RICH = True
except ImportError:
    console = None
    _HAS_RICH = False

# ── Config ─────────────────────────────────────────────────────
LIFERS_HOME = Path(os.environ.get("LIFERS_HOME", Path.home() / ".lifers"))
LIFERS_ROOT = Path(os.environ.get("LIFERS_ROOT", Path(__file__).resolve().parent.parent))
WEIGHTS_DIR = LIFERS_ROOT / "weights"


def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("lifers")
    except Exception:
        pass
    pt = LIFERS_ROOT / "pyproject.toml"
    if pt.is_file():
        try:
            import tomllib
            return tomllib.loads(pt.read_text("utf-8"))["project"]["version"]
        except ImportError:
            pass
        try:
            import tomli
            return tomli.loads(pt.read_text("utf-8"))["project"]["version"]
        except ImportError:
            pass
        # Minimal parser for version line
        try:
            for line in pt.read_text("utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("version"):
                    return stripped.split("=")[1].strip().strip('"')
        except Exception:
            pass
    return "0.2.0"


def _wprint(*args, **kw):
    """Print to stderr (for info/debug, never pollutes stdout)."""
    if console:
        try:
            console.print(*args, **kw)
        except UnicodeEncodeError:
            # Fallback for Windows GBK terminals — strip Rich markup
            import re
            plain = re.sub(r'\[/?\w+(?:[ #]\w+)?\]', '', " ".join(str(a) for a in args))
            print(plain.encode(sys.stderr.encoding or 'utf-8', errors='replace').decode(sys.stderr.encoding or 'utf-8', errors='replace'), file=sys.stderr)
    else:
        print(*args, file=sys.stderr, **kw)


# ── Config management ──────────────────────────────────────────

def _load_config() -> dict:
    LIFERS_HOME.mkdir(parents=True, exist_ok=True)
    cfg_path = LIFERS_HOME / "config.json"
    if cfg_path.is_file():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_config(cfg: dict) -> None:
    LIFERS_HOME.mkdir(parents=True, exist_ok=True)
    (LIFERS_HOME / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _get_config(key: str, default=None):
    env_key = f"LIFERS_{key.upper()}"
    if env_key in os.environ:
        return type(default)(os.environ[env_key]) if default is not None else os.environ[env_key]
    return _load_config().get(key, default)


# ── Remote fallback ────────────────────────────────────────────

def _try_remote(args: list) -> bool:
    """如果配置了 remote，SSH 到远程运行 lifers。返回 True 表示已处理。"""
    remote = _get_config("remote", os.environ.get("LIFERS_REMOTE", ""))
    if not remote:
        return False
    import subprocess
    cmd = ["ssh", "-o", "ControlMaster=no", "-o", "StrictHostKeyChecking=no",
           remote, "lifers"] + args
    subprocess.run(cmd)
    return True


def _no_model_help() -> None:
    """显示未找到模型时的帮助信息。"""
    _wprint("[red]未找到权重文件[/red]")
    if _HAS_RICH:
        _wprint("\n[bold]解决方案:[/bold]")
        _wprint("  1. 训练模型: [dim]python scripts/train_deep_escalate.py[/dim]")
        _wprint(f"  2. 配置远程: [dim]lifers config remote=user@host[/dim]")
        _wprint(f"  3. 设置环境变量: [dim]export LIFERS_REMOTE=user@host[/dim]")
    else:
        _wprint("  请先训练: python scripts/train_deep_escalate.py")
        _wprint("  或配置远程: lifers config remote=user@host")


# ── Model discovery ────────────────────────────────────────────

def _find_model() -> Path | None:
    """自动发现最新权重。优先 deep transformer，回退 tiny/markov。"""
    deep_json = WEIGHTS_DIR / "lifers_deep_transformer.json"
    if deep_json.is_file() and deep_json.stat().st_size > 1024:
        return deep_json
    tiny_json = WEIGHTS_DIR / "lifers_transformer.json"
    if tiny_json.is_file() and tiny_json.stat().st_size > 1024:
        return tiny_json
    # Also check LIFERS_HOME
    alt = LIFERS_HOME / "weights" / "lifers_deep_transformer.json"
    if alt.is_file() and alt.stat().st_size > 1024:
        return alt
    return None


def _load_model(json_path: Path) -> dict:
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    npz_rel = meta.get("_npz", json_path.with_suffix(".npz").name)
    npz_path = json_path.parent / npz_rel
    if not npz_path.is_file():
        raise FileNotFoundError(f"权重文件不存在: {npz_path}")
    import numpy as np
    arrs = np.load(npz_path)
    return {
        "tok_emb": arrs["tok_emb"], "pos_emb": arrs["pos_emb"],
        "Wq": arrs["Wq"], "Wk": arrs["Wk"], "Wv": arrs["Wv"],
        "Wo": arrs["Wo"], "W1": arrs["W1"], "W2": arrs["W2"],
        "Wlm": arrs["Wlm"], "vocab": meta["vocab"],
        "n_heads": int(meta.get("n_heads", 4)),
        "n_layers": int(meta.get("n_layers", 2)),
        "max_seq": int(meta.get("max_seq", 64)),
        "d_model": int(arrs["tok_emb"].shape[1]),
    }


# ── Forward pass ───────────────────────────────────────────────

def _forward_deep(ids, tok_emb, pos_emb, Wq, Wk, Wv, Wo, W1, W2, Wlm,
                  n_heads, n_layers, max_seq):
    import numpy as np
    T = min(len(ids), max_seq)
    ids = ids[-T:]
    d_model = tok_emb.shape[1]
    head_dim = d_model // n_heads

    def _ln(x):
        return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + 1e-5)

    x = tok_emb[np.asarray(ids, dtype=np.intp)] + pos_emb[:T]

    # Pre-compute RoPE frequencies once (same for all layers with weight sharing)
    d2 = head_dim // 2
    pos = np.arange(T, dtype=np.float64).reshape(T, 1)
    dim_arr = np.arange(d2, dtype=np.float64).reshape(1, d2)
    theta = 1.0 / (10000.0 ** (2.0 * dim_arr / head_dim))
    freqs = pos @ theta
    cos = np.cos(freqs).reshape(1, T, d2)
    sin = np.sin(freqs).reshape(1, T, d2)

    for _ in range(n_layers):
        # Pre-LN + Multi-head Attention + Residual
        xn = _ln(x)
        q = xn @ Wq
        k = xn @ Wk
        v = xn @ Wv
        q = q.reshape(T, n_heads, head_dim).transpose(1, 0, 2)
        k = k.reshape(T, n_heads, head_dim).transpose(1, 0, 2)
        v = v.reshape(T, n_heads, head_dim).transpose(1, 0, 2)
        # RoPE
        q_even, q_odd = q[:, :, 0::2], q[:, :, 1::2]
        k_even, k_odd = k[:, :, 0::2], k[:, :, 1::2]
        qr = np.empty_like(q)
        kr = np.empty_like(k)
        qr[:, :, 0::2] = cos * q_even - sin * q_odd
        qr[:, :, 1::2] = sin * q_even + cos * q_odd
        kr[:, :, 0::2] = cos * k_even - sin * k_odd
        kr[:, :, 1::2] = sin * k_even + cos * k_odd
        # Attention
        attn = qr @ kr.transpose(0, 2, 1) / math.sqrt(head_dim)
        mask = np.triu(np.full((T, T), -1e10, dtype=np.float64), k=1)
        attn = attn + mask
        attn_w = np.exp(attn - attn.max(axis=-1, keepdims=True))
        attn_w = attn_w / attn_w.sum(axis=-1, keepdims=True)
        ctx = attn_w @ v
        ctx = ctx.transpose(1, 0, 2).reshape(T, d_model)
        x = x + ctx @ Wo  # Residual: add to original x

        # Pre-LN + FFN (GELU) + Residual
        xn = _ln(x)
        ffn = xn @ W1
        ffn = 0.5 * ffn * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (ffn + 0.044715 * ffn**3)))
        x = x + ffn @ W2  # Residual: add to original x (not xn!)

    y = _ln(x)
    return y @ Wlm


def _generate(model: dict, prompt: str, max_tokens: int = 50,
              temperature: float = 0.8):
    """流式生成器：逐 token yield 字符串。"""
    import numpy as np
    import random as _random
    seed = int.from_bytes(os.urandom(4), "big")
    rng = _random.Random(seed)
    vocab = model["vocab"]
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    ids = [stoi.get(c, 0) for c in prompt]
    if not ids:
        ids = [0]
    max_seq = model["max_seq"]

    for _ in range(max_tokens):
        logits = _forward_deep(
            ids[-max_seq:] if len(ids) > max_seq else ids,
            model["tok_emb"], model["pos_emb"],
            model["Wq"], model["Wk"], model["Wv"], model["Wo"],
            model["W1"], model["W2"], model["Wlm"],
            model["n_heads"], model["n_layers"], model["max_seq"],
        )
        last = np.asarray(logits[-1], dtype=np.float64)
        if temperature > 0:
            last = last - last.max()
            last = np.exp(last / max(temperature, 0.01))
            last = last / last.sum()
            token = rng.choices(range(len(last)), weights=last.tolist(), k=1)[0]
        else:
            token = int(np.argmax(last))
        ids.append(token)
        ch = itos.get(token, "?")
        yield ch
        if ch == "\n" or ch in (".", "!", "?", "。", "！", "？", "\n"):
            pass  # 自然停顿点，但继续生成


# ── Banner (Claude-Code style) ─────────────────────────────────

def _banner(model_info: dict):
    """Claude-Code 风格启动横幅。"""
    D = model_info.get("d_model", "?")
    L = model_info.get("n_layers", "?")
    H = model_info.get("n_heads", "?")
    V = model_info.get("vocab_size", "?")
    params_m = model_info.get("params_m", 0)
    loss = model_info.get("loss", 0)

    if _HAS_RICH:
        title = Text("Lifers", style="bold #FF6B6B")
        tagline = Text("  local AI terminal assistant", style="dim italic")
        header = Text.assemble(title, tagline)

        meta_items = [
            f"D={D}  L={L}  H={H}  V={V}",
        ]
        if params_m:
            meta_items.append(f"{params_m:.1f}M params")
        if loss:
            meta_items.append(f"loss={loss:.4f}")
        meta = Text("  " + "  |  ".join(meta_items), style="dim")

        tip = Text("\n  Commands: "
                   "/stats /config /clear /help /quit  —  Ctrl+C to exit",
                   style="dim italic")

        console.print(Panel(Text.assemble(header, meta, tip),
                            box=box.ROUNDED, border_style="#FF6B6B"))
    else:
        print(f"Lifers — local AI terminal assistant", file=sys.stderr)
        print(f"D={D} L={L} H={H} V={V} | {params_m:.1f}M params | loss={loss:.4f}",
              file=sys.stderr)


# ── Stats (Claude-Code style) ──────────────────────────────────

def cmd_stats() -> int:
    json_path = _find_model()
    if json_path is None:
        _wprint("[red]未找到权重文件[/red]")
        _wprint(f"  请先训练: [dim]python scripts/train_deep_escalate.py[/dim]")
        return 1

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    npz_rel = meta.get("_npz", "")
    npz_path = json_path.parent / npz_rel if npz_rel else None
    mb = npz_path.stat().st_size / 1e6 if npz_path and npz_path.is_file() else 0
    D = meta.get("d_model", "?")
    L = meta.get("n_layers", "?")
    H = meta.get("n_heads", "?")
    V = len(meta.get("vocab", []))
    S = meta.get("max_seq", "?")

    if _HAS_RICH:
        table = Table(title="Lifers Model Status", box=box.ROUNDED, border_style="#FF6B6B")
        table.add_column("Property", style="bold", justify="right")
        table.add_column("Value", style="dim")

        table.add_row("Architecture", f"D={D}  L={L}  H={H}  V={V}  seq={S}")
        if isinstance(D, int) and isinstance(L, int):
            d_ff = meta.get("d_ff", D * 4)
            params = (V * D + 256 * D + 4 * D * D + 2 * D * d_ff + D * V)
            params_m = params / 1e6
            table.add_row("Parameters", f"~{params_m:.1f}M")
        if mb:
            table.add_row("Weights file", f"{npz_path.name} ({mb:.0f}MB)")
        table.add_row("Vocabulary", f"{V} tokens")
        table.add_row("Max sequence", str(S))

        console.print(table)
    else:
        print(f"模型: D={D} L={L} H={H} V={V} seq={S}")
        if mb:
            print(f"权重: {mb:.0f}MB")

    # Training heartbeat
    hb = json_path.parent / ".train_heartbeat.json"
    if hb.is_file():
        try:
            d = json.loads(hb.read_text(encoding="utf-8"))
            loss = d.get("loss", 0)
            tier = d.get("ramp_iter", "?")
            tier_max = d.get("ramp_max", "?")
            step = d.get("sgd_step", "?")
            step_total = d.get("sgd_total", "?")

            if _HAS_RICH:
                from rich.progress import BarColumn, Progress, TextColumn
                pct = step / max(step_total, 1) if isinstance(step, int) and isinstance(step_total, int) else 0
                progress = Progress(TextColumn("[bold]{task.description}"), BarColumn(), TextColumn("{task.percentage:.0f}%"),
                                    console=console)
                task = progress.add_task(f"Training tier {tier}/{tier_max}", total=step_total, completed=step)
                console.print(progress)
                console.print(f"  [dim]loss={loss:.4f}  step={step}/{step_total}[/dim]")
            else:
                print(f"训练: tier {tier}/{tier_max} step {step}/{step_total} loss={loss:.4f}")
        except Exception:
            pass

    return 0


# ── Config command ─────────────────────────────────────────────

def cmd_config(args: list) -> int:
    cfg = _load_config()

    if not args:
        # Show config
        if _HAS_RICH:
            table = Table(title="Lifers Configuration", box=box.ROUNDED, border_style="#FF6B6B")
            table.add_column("Key", style="bold")
            table.add_column("Value")
            table.add_column("Default", style="dim")
            defaults = {"temperature": "0.7", "max_tokens": "80", "lifers_root": str(LIFERS_ROOT)}
            all_keys = set(list(cfg.keys()) + list(defaults.keys()))
            for k in sorted(all_keys):
                val = cfg.get(k, "-")
                defv = defaults.get(k, "-")
                table.add_row(k, str(val), str(defv))
            console.print(table)
            console.print("\n[dim]Config file:[/dim] " + str(LIFERS_HOME / "config.json"))
        else:
            for k, v in cfg.items():
                print(f"  {k} = {v}")
        return 0

    # Set config: lifers config key=value
    for arg in args:
        if "=" in arg:
            k, v = arg.split("=", 1)
            k = k.strip()
            v = v.strip()
            # Auto-convert types
            if v.lower() in ("true", "yes"): v = True
            elif v.lower() in ("false", "no"): v = False
            else:
                try: v = float(v) if "." in v else int(v)
                except ValueError: pass
            cfg[k] = v
            _wprint(f"[green][OK][/green] {k} = {v}")
        elif arg == "--path":
            _wprint(str(LIFERS_HOME / "config.json"))
            return 0

    _save_config(cfg)
    return 0


# ── Single-shot ────────────────────────────────────────────────

# ── 运维命令：权重、进程、训练 ──────────────────────────────────

def _kali_config():
    """读取部署配置，返回 (ssh_target, remote_root)。"""
    host = os.environ.get("LIFERS_KALI_HOST", "192.168.234.152")
    user = os.environ.get("LIFERS_KALI_USER", "kali")
    ssh = f"{user}@{host}"
    root = os.environ.get("LIFERS_KALI_HOME", "/home/kali/lifers")
    return ssh, root


KALI_SSH, KALI_ROOT = _kali_config()


def _kali_ssh(cmd: str, timeout: int = 10) -> str:
    """执行 Kali SSH 命令，返回输出或错误信息。"""
    import subprocess
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             KALI_SSH, cmd],
            capture_output=True, timeout=timeout,
        )
        out = r.stdout.decode("utf-8", errors="replace").strip()
        if out:
            return out
        return r.stderr.decode("utf-8", errors="replace").strip() or "(empty)"
    except Exception as e:
        return f"(Kali 不可达: {e})"


def cmd_weights(args: list) -> int:
    """查看权��文件状态。用法: lifers weights [list|info <file>]"""
    sub = args[1].lower() if len(args) > 1 else "list"

    if sub == "list" or sub == "ls":
        # 正确路径: 外层 weights/ = 项目根/weights, 内层 = lifers包/weights
        PROJECT_ROOT = LIFERS_ROOT.parent if LIFERS_ROOT.name == "lifers" else LIFERS_ROOT
        weight_dirs = [
            (PROJECT_ROOT / "weights", "weights/"),
            (LIFERS_ROOT / "weights", "lifers/weights/"),
        ]
        all_files = []
        for wdir, label in weight_dirs:
            if not wdir.is_dir():
                continue
            for f in sorted(wdir.glob("*.json"), key=lambda p: p.stat().st_size, reverse=True):
                sz = f.stat().st_size
                mt = time.strftime("%m-%d %H:%M", time.localtime(f.stat().st_mtime))
                all_files.append((label, f.name, sz, mt))
            for f in sorted(wdir.glob("*.npz"), key=lambda p: p.stat().st_size, reverse=True):
                sz = f.stat().st_size
                mt = time.strftime("%m-%d %H:%M", time.localtime(f.stat().st_mtime))
                all_files.append((label, f.name, sz, mt))

        if _HAS_RICH:
            t = Table(title="Lifers 权重文件", box=box.ROUNDED)
            t.add_column("位置", style="dim")
            t.add_column("文件名")
            t.add_column("大小", justify="right")
            t.add_column("更新时间")
            for label, name, sz, mt in all_files:
                sizestr = f"{sz/1024:.0f}KB" if sz < 1024*1024 else f"{sz/1024/1024:.1f}MB"
                t.add_row(label, name, sizestr, mt)
            console.print(t)

            # 检查重复
            inner_names = {f[1] for f in all_files if f[0] == "lifers/weights/"}
            outer_names = {f[1] for f in all_files if f[0] == "weights/"}
            dups = inner_names & outer_names
            if dups:
                console.print(f"\n[yellow]⚠ 重复文件 ({len(dups)}): {', '.join(sorted(dups))}[/yellow]")
        else:
            for label, name, sz, mt in all_files:
                sizestr = f"{sz/1024:.0f}KB" if sz < 1024*1024 else f"{sz/1024/1024:.1f}MB"
                print(f"  {label:20s} {name:45s} {sizestr:>10s}  {mt}")
        return 0

    if sub == "info" and len(args) > 2:
        fname = args[2]
        PROJECT_ROOT = LIFERS_ROOT.parent if LIFERS_ROOT.name == "lifers" else LIFERS_ROOT
        found = None
        for wdir in [PROJECT_ROOT / "weights", LIFERS_ROOT / "weights"]:
            cand = wdir / fname
            if cand.is_file():
                found = cand
                break
        if not found:
            # 支持部分名称匹配
            for wdir in [LIFERS_ROOT / "weights", LIFERS_ROOT / "lifers" / "weights"]:
                for f in wdir.glob(f"*{fname}*"):
                    found = f
                    break
        if not found:
            print(f"未找到权重文件: {fname}")
            return 1
        print(f"文件: {found}")
        print(f"路径: {found.resolve()}")
        print(f"大小: {found.stat().st_size/1024:.0f}KB")
        print(f"修改: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(found.stat().st_mtime))}")
        if found.suffix == ".json":
            try:
                data = json.loads(found.read_text("utf-8"))
                print(f"\n架构:")
                for k in ["d_model", "n_heads", "n_layers", "max_seq"]:
                    if k in data:
                        print(f"  {k}: {data[k]}")
                if "vocab" in data:
                    print(f"  vocab: {len(data['vocab'])} tokens")
                if "_npz" in data:
                    print(f"  权重文件: {data['_npz']}")
                    npz = found.parent / data["_npz"]
                    if npz.is_file():
                        print(f"  NPZ大小: {npz.stat().st_size/1024/1024:.1f}MB")
            except Exception:
                pass
        return 0

    print("用法: lifers weights [list|info <file>]")
    return 1


def cmd_processes(args: list) -> int:
    """查看 Lifers 进程。用法: lifers processes [local|kali|all]"""
    sub = args[1].lower() if len(args) > 1 else "local"

    if sub in ("local", "all"):
        import subprocess
        if _HAS_RICH:
            console.print("[bold]Windows 本地进程:[/bold]")
        else:
            print("=== Windows 本地进程 ===")
        try:
            r = subprocess.run(
                ["powershell", "-Command",
                 "Get-CimInstance Win32_Process | Where-Object { "
                 "$_.CommandLine -like '*lifers*' -or $_.CommandLine -like '*train_*' "
                 "} | Select-Object ProcessId,WorkingSetSize,CommandLine | "
                 "ForEach-Object { '{0},{1},{2}' -f $_.ProcessId,$_.WorkingSetSize,$_.CommandLine }"],
                capture_output=True, text=True, timeout=15, encoding="utf-8",
            )
            lines = [l.strip() for l in r.stdout.split("\n") if l.strip()]
            if lines:
                for l in lines:
                    parts = l.split(",", 2)
                    if len(parts) >= 3:
                        pid = parts[0].strip()
                        try:
                            mem = int(parts[1].strip()) // 1024 // 1024
                            memstr = f"{mem}MB" if mem > 0 else "?"
                        except ValueError:
                            memstr = "?"
                        # 提取关键文件名
                        cmd = parts[2].strip()
                        for keyword in ["train_", "lifers", "cli.py", "gui_host", "monitor", "gate"]:
                            idx = cmd.find(keyword)
                            if idx >= 0:
                                cmd = cmd[idx:][:80]
                                break
                        else:
                            cmd = cmd[:80]
                        print(f"  PID:{pid:>8s}  MEM:{memstr:>8s}  {cmd}")
            else:
                print("  (未找到Lifers进程)")
        except Exception as e:
            print(f"  (进程查询失败: {e})")

    if sub in ("kali", "all"):
        if _HAS_RICH:
            console.print("\n[bold]Kali 进程:[/bold]")
        else:
            print("\n=== Kali 进程 ===")
        out = _kali_ssh(
            "ps aux --sort=-%mem | grep python | grep -v 'grep\\|applet\\|blueman' | "
            "awk '{printf \"  PID:%-8s CPU:%-6s MEM:%-6s \", $2, $3, $4; "
            "for(i=11;i<=NF;i++) printf \"%s \", $i; print \"\"}'",
        )
        print(out if out else "  (无Kali进程)")

    return 0


def cmd_training(args: list) -> int:
    """查看训练状态和材料。用法: lifers training [status|corpus|materials]"""
    sub = args[1].lower() if len(args) > 1 else "status"

    if sub == "status":
        if _HAS_RICH:
            console.print("[bold]Windows 训练状态:[/bold]")
        else:
            print("=== Windows 训练状态 ===")
        PROJECT_ROOT = LIFERS_ROOT.parent if LIFERS_ROOT.name == "lifers" else LIFERS_ROOT
        # Windows
        for wdir in [LIFERS_ROOT / "weights", PROJECT_ROOT / "weights"]:
            sf = wdir / ".train_status.json"
            if sf.is_file():
                try:
                    d = json.loads(sf.read_text("utf-8"))
                    print(f"  模型: D={d['architecture']['d_model']} "
                          f"tier {d['ramp']['iter']}/{d['ramp']['max']} "
                          f"step {d['sgd']['step']}/{d['sgd']['total_steps']} "
                          f"({d['overall_pct_approx']}%)")
                    print(f"  词表: {d['sgd']['vocab_size']} tokens  "
                          f"进程: PID {d.get('pid', '?')}  "
                          f"阶段: {d.get('phase', '?')}")
                except Exception:
                    pass

        # Kali
        if _HAS_RICH:
            console.print("\n[bold]Kali 训练状态:[/bold]")
        else:
            print("\n=== Kali 训练状态 ===")
        out = _kali_ssh(
            f"cat {KALI_ROOT}/lifers/weights/.train_status.json 2>/dev/null | "
            "python3 -c \"import sys,json; d=json.load(sys.stdin); "
            "print(f'  模型: D={d[\\\"architecture\\\"][\\\"d_model\\\"]} "
            "tier {d[\\\"ramp\\\"][\\\"iter\\\"]}/{d[\\\"ramp\\\"][\\\"max\\\"]} "
            "step {d[\\\"sgd\\\"][\\\"step\\\"]}/{d[\\\"sgd\\\"][\\\"total_steps\\\"]} "
            "({d[\\\"overall_pct_approx\\\"]}%)'); "
            "print(f'  进程: PID {d.get(\\\"pid\\\",\\\"?\\\")}  "
            "阶段: {d.get(\\\"phase\\\",\\\"?\\\")}')\" 2>/dev/null"
        )
        print(out if out else "  (Kali 状态不可用)")

        # Kali 心跳
        hb = _kali_ssh(f"cat {KALI_ROOT}/lifers/weights/.train_heartbeat.json 2>/dev/null")
        if hb and "ts" in hb:
            try:
                hd = json.loads(hb)
                print(f"  Kali 心跳: {hd.get('ts','?')}  loss={hd.get('loss','?')}")
            except Exception:
                pass
        return 0

    if sub == "corpus":
        if _HAS_RICH:
            console.print("[bold]训练语料统计:[/bold]")
        else:
            print("=== 训练语料统计 ===")
        PROJECT_ROOT = LIFERS_ROOT.parent if LIFERS_ROOT.name == "lifers" else LIFERS_ROOT
        corpus_paths = [
            PROJECT_ROOT / "weights" / "training_corpus.txt",
            LIFERS_ROOT / "weights" / "training_corpus.txt",
        ]
        for cp in corpus_paths:
            if cp.is_file():
                sz = cp.stat().st_size
                print(f"  {cp.parent.name}/{cp.name}:")
                print(f"    大小: {sz/1024/1024:.1f}MB")
                try:
                    # 只读前 10MB 用于统计
                    with open(cp, "r", encoding="utf-8", errors="ignore") as f:
                        sample = f.read(10_000_000)
                    lines_est = int(sz / max(1, len(sample)) * sample.count("\n"))
                    cn = sum(1 for c in sample if '一' <= c <= '鿿')
                    en = sum(1 for c in sample if c.isascii() and c.isalpha())
                    cn_ratio = cn / len(sample) * 100 if sample else 0
                    en_ratio = en / len(sample) * 100 if sample else 0
                    print(f"    估算行数: {lines_est:,}")
                    print(f"    中文占比: {cn_ratio:.1f}%  英文占比: {en_ratio:.1f}%")
                except Exception as e:
                    print(f"    (统计失败: {e})")
        return 0

    if sub == "materials":
        if _HAS_RICH:
            console.print("[bold]训练材料文件:[/bold]")
        else:
            print("=== 训练材料文件 ===")
        PROJECT_ROOT = LIFERS_ROOT.parent if LIFERS_ROOT.name == "lifers" else LIFERS_ROOT
        # 语料生成脚本（在项目根目录）
        gen_scripts = sorted(PROJECT_ROOT.glob("gen_*.py")) + sorted(PROJECT_ROOT.glob("build_*.py"))
        scrape_scripts = sorted(PROJECT_ROOT.glob("scrape_*.py"))
        expand_scripts = sorted(PROJECT_ROOT.glob("expand_*.py"))
        all_gen = gen_scripts + scrape_scripts + expand_scripts
        print(f"\n语料生成脚本 ({len(all_gen)}):")
        for s in all_gen:
            print(f"  {s.name} ({s.stat().st_size/1024:.0f}KB)")

        # 训练脚本（在 lifers/scripts/ 下）
        train_scripts = sorted((LIFERS_ROOT / "scripts").glob("train_*.py"))
        print(f"\n训练脚本 ({len(train_scripts)}):")
        for s in train_scripts:
            print(f"  {s.name} ({s.stat().st_size/1024:.0f}KB)")

        # 数据目录（项目根下）
        data_dir = PROJECT_ROOT / "data"
        if data_dir.is_dir():
            total = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
            files = len(list(data_dir.rglob("*")))
            print(f"\ndata/ 目录: {files} 文件, {total/1024/1024:.1f}MB")

        # 配置文件（lifers/config/ 下）
        config_dir = LIFERS_ROOT / "config"
        if config_dir.is_dir():
            cfgs = sorted(config_dir.glob("*.json"))
            print(f"\n配置文件 ({len(cfgs)}):")
            for c in cfgs:
                print(f"  {c.name} ({c.stat().st_size/1024:.0f}KB)")
        return 0

    print("用法: lifers training [status|corpus|materials]")
    return 1


# ── Sync: 从 Kali 拉取权重 ───────────────────────────────────────

def cmd_sync(args: list) -> int:
    """从 Kali 同步权重文件到本地。用法: lifers sync [--force] [--watch [N]]"""
    import subprocess

    force = "--force" in args or "-f" in args
    watch = False
    interval = 0
    for i, a in enumerate(args):
        if a in ("--watch", "-w"):
            watch = True
            try:
                interval = int(args[i + 1]) if i + 1 < len(args) else 120
            except (ValueError, IndexError):
                interval = 120

    PROJECT_ROOT = LIFERS_ROOT.parent if LIFERS_ROOT.name == "lifers" else LIFERS_ROOT
    weights_dir = PROJECT_ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    def _do_sync():
        ssh, remote_root = _kali_config()
        rw = f"{remote_root}/lifers/weights"
        if _HAS_RICH:
            console.print(f"[bold]同步权重:[/bold] {ssh}:{rw}")
        else:
            print(f"=== 同步权重: {ssh}:{rw} ===")

        # Small files always sync
        small = [".train_control", "lifers_markov.json", ".lifers_train_state.json"]
        for fname in small:
            try:
                r = subprocess.run(
                    ["scp", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
                     f"{ssh}:{rw}/{fname}", str(weights_dir)],
                    capture_output=True, timeout=30,
                )
                if r.returncode == 0:
                    _wprint(f"  [green]✓[/green] {fname}")
                else:
                    _wprint(f"  [dim]- {fname} (not on remote)[/dim]")
            except Exception as e:
                _wprint(f"  [yellow]! {fname}: {e}[/yellow]")

        # Large transformer - check mtime
        tf = "lifers_transformer.json"
        local_tf = weights_dir / tf
        try:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
                 ssh, f"stat -c '%Y' {rw}/{tf} 2>/dev/null || echo 0"],
                capture_output=True, timeout=15,
            )
            remote_mtime = int(r.stdout.decode("utf-8", errors="replace").strip() or "0")
        except Exception:
            remote_mtime = 0

        local_mtime = 0
        if local_tf.is_file():
            local_mtime = int(local_tf.stat().st_mtime)

        if force or remote_mtime > local_mtime or not local_tf.is_file():
            reason = "force" if force else ("newer" if remote_mtime > local_mtime else "missing")
            _wprint(f"  [bold]拉取 {tf}[/bold] (reason={reason})")
            try:
                subprocess.run(
                    ["scp", "-o", "ConnectTimeout=60", "-o", "BatchMode=yes",
                     f"{ssh}:{rw}/{tf}", str(local_tf)],
                    check=True, timeout=120,
                )
                _wprint(f"  [green]✓[/green] {tf}")
            except Exception as e:
                _wprint(f"  [red]✗ {tf}: {e}[/red]")
        else:
            _wprint(f"  [dim]跳过 {tf} (已是最新)[/dim]")

        # Deep transformer
        dtf = "lifers_deep_transformer.json"
        local_dtf = weights_dir / dtf
        try:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
                 ssh, f"stat -c '%Y' {rw}/{dtf} 2>/dev/null || echo 0"],
                capture_output=True, timeout=15,
            )
            remote_mtime = int(r.stdout.decode("utf-8", errors="replace").strip() or "0")
        except Exception:
            remote_mtime = 0

        local_mtime = 0
        if local_dtf.is_file():
            local_mtime = int(local_dtf.stat().st_mtime)

        if force or remote_mtime > local_mtime or not local_dtf.is_file():
            reason = "force" if force else ("newer" if remote_mtime > local_mtime else "missing")
            _wprint(f"  [bold]拉取 {dtf}[/bold] (reason={reason})")
            try:
                subprocess.run(
                    ["scp", "-o", "ConnectTimeout=60", "-o", "BatchMode=yes",
                     f"{ssh}:{rw}/{dtf}", str(local_dtf)],
                    check=True, timeout=300,
                )
                _wprint(f"  [green]✓[/green] {dtf}")
            except Exception as e:
                _wprint(f"  [red]✗ {dtf}: {e}[/red]")
        else:
            _wprint(f"  [dim]跳过 {dtf} (已是最新)[/dim]")

        _wprint("[green]同步完成[/green]")

    if watch:
        _wprint(f"[dim]轮询模式: {interval}s 间隔 (Ctrl+C 停止)[/dim]")
        while True:
            _do_sync()
            time.sleep(interval)
    else:
        _do_sync()

    return 0


# ── Control: 训练控制 (本地/远端) ───────────────────────────────

def cmd_control(args: list) -> int:
    """控制训练: lifers control [pause|resume|stop|status] [local|kali|all]"""
    sub = args[1].lower() if len(args) > 1 else "status"
    target = args[2].lower() if len(args) > 2 else "all"

    if sub not in ("pause", "resume", "stop", "status"):
        print("用法: lifers control [pause|resume|stop|status] [local|kali|all]")
        return 1

    cmd_map = {"pause": "pause", "resume": "run", "stop": "stop"}

    PROJECT_ROOT = LIFERS_ROOT.parent if LIFERS_ROOT.name == "lifers" else LIFERS_ROOT

    def _local_control(action: str):
        ctl_file = PROJECT_ROOT / "weights" / ".train_control"
        ctl_file.parent.mkdir(parents=True, exist_ok=True)
        old = ctl_file.read_text().strip() if ctl_file.is_file() else "(none)"
        ctl_file.write_text(f"{action}\n")
        _wprint(f"  本地: {old} → [bold]{action}[/bold]  ({ctl_file})")

    def _kali_control(action: str):
        ssh, remote_root = _kali_config()
        wf = f"{remote_root}/lifers/weights"
        out = _kali_ssh(
            f"echo {action} > {wf}/.train_control && "
            f"echo 'control='$(cat {wf}/.train_control)"
        )
        _wprint(f"  Kali: {out}")

    def _local_status():
        for wdir in [PROJECT_ROOT / "weights", LIFERS_ROOT / "weights"]:
            ctl = wdir / ".train_control"
            if ctl.is_file():
                print(f"  本地控制: {ctl.read_text().strip()}  ({ctl})")
                break
        else:
            print("  本地控制: (none)")

    def _kali_status():
        ssh, remote_root = _kali_config()
        out = _kali_ssh(f"cat {remote_root}/lifers/weights/.train_control 2>/dev/null || echo '(none)'")
        print(f"  Kali 控制: {out}")
        # Also check processes
        procs = _kali_ssh("pgrep -af train_lifers_escalate 2>/dev/null || echo '(none)'")
        print(f"  Kali 训练进程: {procs}")

    action = cmd_map.get(sub)
    if action:  # pause/resume/stop
        if target in ("local", "all"):
            _local_control(action)
        if target in ("kali", "all"):
            _kali_control(action)
    else:  # status
        if target in ("local", "all"):
            _local_status()
        if target in ("kali", "all"):
            _kali_status()

    return 0


# ── Push: 打包推送到 Kali ────────────────────────────────────────

def cmd_push(args: list) -> int:
    """打包推送到 Kali。用法: lifers push [--weights] [--data] [--skip-bootstrap]"""
    import subprocess

    include_weights = "--weights" in args or "-w" in args
    include_data = "--data" in args
    skip_bootstrap = "--skip-bootstrap" in args or "--no-boot" in args

    PROJECT_ROOT = LIFERS_ROOT.parent if LIFERS_ROOT.name == "lifers" else LIFERS_ROOT
    dist_dir = LIFERS_ROOT / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dist_dir / "lifers_kali.tar.gz"

    ssh, _ = _kali_config()

    # Step 1: Package
    _wprint("[bold]打包项目...[/bold]")
    excludes = [
        "--exclude=.git",
        "--exclude=third_party/openclaw/.git",
        "--exclude=third_party/openclaw/node_modules",
        "--exclude=third_party/claw_code_rust/target",
        "--exclude=third_party/_refs",
        "--exclude=lifers/.venv",
        "--exclude=lifers/dist",
        "--exclude=__pycache__",
        "--exclude=**/node_modules",
        "--exclude=shell",
        "--exclude=.cursor",
    ]
    if not include_data:
        excludes.append("--exclude=data")
    if not include_weights:
        excludes.append("--exclude=lifers/weights")

    try:
        subprocess.run(
            ["tar", "-czf", str(tar_path)] + excludes + ["."],
            cwd=str(PROJECT_ROOT), check=True, timeout=120,
        )
        sz_mb = tar_path.stat().st_size / 1024 / 1024
        _wprint(f"  [green]✓[/green] {tar_path.name} ({sz_mb:.1f}MB)")
    except subprocess.CalledProcessError as e:
        _wprint(f"[red]打包失败: {e}[/red]")
        return 1

    # Step 2: Pause remote training
    _wprint("[bold]暂停远程训练...[/bold]")
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
             ssh, f"echo pause > {KALI_ROOT}/lifers/weights/.train_control"],
            capture_output=True, timeout=15,
        )
        _wprint("  [green]✓[/green] 远程训练已暂停")
    except Exception as e:
        _wprint(f"  [yellow]! 暂停失败 (继续推送): {e}[/yellow]")

    # Step 3: SCP to Kali
    remote_tar = "/tmp/lifers_kali_push.tar.gz"
    _wprint(f"[bold]推送到 {ssh}...[/bold]")
    try:
        subprocess.run(
            ["scp", "-o", "ConnectTimeout=60", "-o", "BatchMode=yes",
             str(tar_path), f"{ssh}:{remote_tar}"],
            check=True, timeout=300,
        )
        _wprint("  [green]✓[/green] 推送完成")
    except subprocess.CalledProcessError as e:
        _wprint(f"[red]推送失败: {e}[/red]")
        return 1

    # Step 4: Extract and bootstrap on Kali
    remote_root = KALI_ROOT
    extract_cmd = (
        f"mkdir -p {remote_root} && "
        f"tar -xzf {remote_tar} -C {remote_root} && "
        f"rm -f {remote_tar} && "
        f"chmod +x {remote_root}/lifers/scripts/remote_kali_bootstrap_train_loop.sh"
    )

    if skip_bootstrap:
        extract_cmd += " && echo 'extracted only (skip bootstrap)'"
    else:
        extract_cmd += f" && exec bash {remote_root}/lifers/scripts/remote_kali_bootstrap_train_loop.sh"

    _wprint(f"[bold]{'解压' if skip_bootstrap else '解压+启动训练循环'}...[/bold]")
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=30", "-o", "BatchMode=yes", ssh, extract_cmd],
            capture_output=True, timeout=120,
        )
        for line in r.stdout.decode("utf-8", errors="replace").split("\n"):
            if line.strip():
                _wprint(f"  {line}")
        if r.returncode != 0:
            _wprint(f"[yellow]远程执行警告 (exit={r.returncode})[/yellow]")
    except Exception as e:
        _wprint(f"[red]远程执行失败: {e}[/red]")
        _wprint(f"[dim]手动执行: ssh {ssh} 'bash {remote_root}/lifers/scripts/remote_kali_bootstrap_train_loop.sh'[/dim]")
        return 1

    _wprint("[green]推送完成[/green]")
    if not skip_bootstrap:
        _wprint(f"[dim]Kali 接入: ssh {ssh} -t tmux attach -t lifers-stack[/dim]")
    return 0


def cmd_ask(prompt: str, max_tokens: int = 80, temperature: float = 0.7) -> int:
    json_path = _find_model()
    if json_path is None:
        _wprint("[red]错误:[/red] 未找到权重文件")
        _wprint(f"  请先训练: [dim]python scripts/train_deep_escalate.py[/dim]")
        return 1

    model = _load_model(json_path)
    D, L, H, V = model["d_model"], model["n_layers"], model["n_heads"], len(model["vocab"])
    _wprint(f"[dim]Model: D={D} L={L} H={H} V={V}  |  {json_path.name}[/dim]")

    t0 = time.time()
    count = 0
    for ch in _generate(model, prompt, max_tokens=max_tokens, temperature=temperature):
        sys.stdout.write(ch)
        sys.stdout.flush()
        count += 1
    elapsed = time.time() - t0

    print()
    _wprint(f"[dim]{count} tokens / {elapsed:.1f}s[/dim]")
    return 0


# ── Tab completion ─────────────────────────────────────────────

_SLASH_COMMANDS = ["stats", "config", "clear", "help", "quit", "exit", "q"]

def _setup_readline():
    try:
        import readline
    except ImportError:
        try:
            import pyreadline3 as readline
        except ImportError:
            return

    def completer(text: str, state: int) -> str | None:
        if text.startswith("/"):
            cmd = text[1:]
            matches = [f"/{c}" for c in _SLASH_COMMANDS if c.startswith(cmd)]
            if state < len(matches):
                return matches[state]
            return None
        # File path completion
        import glob as _glob
        expanded = _glob.glob(text + "*")
        dirs = [p + "/" for p in expanded if Path(p).is_dir()]
        files = [p for p in expanded if not Path(p).is_dir()]
        options = sorted(dirs) + sorted(files)
        if state < len(options):
            return options[state]
        return None

    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")
    readline.set_completer_delims(" \t\n")


# ── Interactive chat (Claude-Code style) ───────────────────────

def cmd_chat(max_tokens: int = 80, temperature: float = 0.7) -> int:
    json_path = _find_model()
    if json_path is None:
        _wprint("[red]错误:[/red] 未找到权重文件")
        _wprint(f"  请先训练: [dim]python scripts/train_deep_escalate.py[/dim]")
        return 1

    model = _load_model(json_path)
    D, L, H, V = model["d_model"], model["n_layers"], model["n_heads"], len(model["vocab"])

    # Setup tab completion
    _setup_readline()

    # Show banner
    hb_data = {}
    hb = json_path.parent / ".train_heartbeat.json"
    if hb.is_file():
        try:
            hb_data = json.loads(hb.read_text(encoding="utf-8"))
        except Exception:
            pass

    _banner({
        "d_model": D, "n_layers": L, "n_heads": H, "vocab_size": V,
        "params_m": (D * D * 4 + D * V + V * D) / 1e6,  # attn+ffn+emb+head
        "loss": hb_data.get("loss", 0),
    })

    _wprint("")

    history: list[tuple[str, str]] = []  # (user, assistant) pairs, last N rounds
    max_history = 5

    while True:
        try:
            user_input = console.input("[bold #FF6B6B]❯[/bold #FF6B6B] ").strip() if _HAS_RICH else input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            _wprint("\n[dim]Bye[/dim]")
            break

        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            parts = user_input[1:].strip().lower().split()
            cmd = parts[0] if parts else ""

            if cmd in ("quit", "exit", "q"):
                _wprint("[dim]Bye[/dim]")
                break
            elif cmd == "stats":
                cmd_stats()
                continue
            elif cmd == "config":
                cmd_config(parts[1:])
                continue
            elif cmd in ("weights", "w"):
                cmd_weights(parts)
                continue
            elif cmd in ("processes", "ps", "procs"):
                cmd_processes(parts)
                continue
            elif cmd in ("training", "train", "tr"):
                cmd_training(parts)
                continue
            elif cmd in ("sync", "pull"):
                cmd_sync(parts)
                continue
            elif cmd in ("control", "ctl"):
                cmd_control(parts)
                continue
            elif cmd in ("push",):
                cmd_push(parts)
                continue
            elif cmd == "clear":
                os.system("cls" if sys.platform == "win32" else "clear")
                continue
            elif cmd == "help":
                _help_text()
                continue
            else:
                _wprint(f"[yellow]未知命令:[/yellow] /{cmd}  输入 /help 查看帮助")
                continue

        # Build prompt with conversation history
        prompt_parts: list[str] = []
        for u, a in history[-max_history:]:
            prompt_parts.append(u)
            prompt_parts.append(a)
        prompt_parts.append(user_input)
        full_prompt = "\n".join(prompt_parts)

        # Generate (streaming)
        try:
            t0 = time.time()
            count = 0
            result_chars: list[str] = []

            if _HAS_RICH:
                console.print()  # newline before response
            for ch in _generate(model, full_prompt, max_tokens=max_tokens, temperature=temperature):
                sys.stdout.write(ch)
                sys.stdout.flush()
                result_chars.append(ch)
                count += 1
            elapsed = time.time() - t0

            full = "".join(result_chars)
            history.append((user_input, full))
            if len(history) > max_history:
                history = history[-max_history:]

            if _HAS_RICH:
                console.print(f"\n  [dim]{count} tokens / {elapsed:.1f}s[/dim]")
            else:
                print()
        except Exception as e:
            _wprint(f"[red]错误:[/red] {e}")

        _wprint("")

    return 0


def _help_text():
    if _HAS_RICH:
        console.print(Panel("""
[bold]Lifers — 本地 AI 终端助手[/bold]

[bold]启动方式:[/bold]
  lifers              交互对话
  lifers "问题"        单次问答
  lifers stats         查看模型状态
  lifers config        查看/修改配置
  lifers --help        显示帮助

[bold]对话命令:[/bold]
  /stats, /tr status   查看模型和训练状态
  /weights, /w         查看所有权重文件
  /ps, /procs          查看运行进程
  /ps kali             查看Kali进程
  /training, /tr       训练状态/语料/材料
  /config [key=value]  查看/设置配置项
  /clear               清屏
  /help                显示此帮助
  /quit, /exit         退出

[bold]配置文件:[/bold]
  {0}/config.json

[bold]环境变量:[/bold]
  LIFERS_ROOT          项目根目录
  LIFERS_HOME          配置文件目录
  LIFERS_TEMPERATURE   默认温度
  LIFERS_MAX_TOKENS    默认最大 tokens
""".strip().format(LIFERS_HOME), box=box.ROUNDED, border_style="#FF6B6B"))
    else:
        print("""
Lifers — 本地 AI 终端助手
  lifers              交互对话
  lifers "问题"        单次问答
  lifers stats         查看模型状态
  lifers weights       查看权重文件
  lifers ps            查看进程
  lifers training      训练状态/语料/材料
  lifers sync          从Kali拉取权重
  lifers control       训练控制(pause/resume/stop)
  lifers push          打包推送到Kali
  lifers config        查看配置
  对话命令: /stats /weights /ps /training /sync /control /push /config /clear /help /quit
""".strip())


# ── Main ───────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="lifers",
        description="Lifers — 本地 AI 终端助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  lifers                    交互对话
  lifers "你好"             单次问答
  lifers stats              查看模型状态
  lifers weights            查看所有权重文件
  lifers weights info deep  查看Deep权重详情
  lifers ps                 查看本地进程
  lifers ps kali            查看Kali进程
  lifers training status    查看训练进度
  lifers training corpus    查看语料统计
  lifers training materials 查看训练材料
  lifers sync               从Kali拉取权重
  lifers sync --force       强制同步
  lifers sync --watch 300   每5分钟轮询
  lifers control pause      暂停本地训练
  lifers control resume kali  恢复Kali训练
  lifers control status all   查看所有控制状态
  lifers push               打包推送到Kali
  lifers push --weights     含权重推送
  lifers push --skip-bootstrap 只推送不解压启动
  lifers config             查看配置
  lifers config temperature=0.5  修改配置
        """.strip(),
    )
    parser.add_argument("prompt", nargs="*", help="直接提问（留空进入交互模式）")
    parser.add_argument("--max-tokens", type=int, help="最大输出 tokens")
    parser.add_argument("--temperature", type=float, help="采样温度 (0=贪婪)")
    parser.add_argument("--version", action="store_true", help="显示版本")
    parser.add_argument("--path", action="store_true", help="显示配置文件路径")
    parser.add_argument("--plain", action="store_true", help="纯文本输出（禁用 Rich 格式）")
    args = parser.parse_args()

    if args.plain:
        global console, _HAS_RICH
        console = None
        _HAS_RICH = False

    # Config defaults
    cfg = _load_config()
    max_tokens = args.max_tokens or int(cfg.get("max_tokens", 80))
    temperature = args.temperature or float(cfg.get("temperature", 0.7))

    if args.version:
        ver = _get_version()
        if _HAS_RICH:
            console.print(f"[bold #FF6B6B]lifers[/bold #FF6B6B] v{ver}")
        else:
            print(f"lifers v{ver}")
        return 0

    if args.path:
        print(str(LIFERS_HOME / "config.json"))
        return 0

    if args.prompt:
        cmd = args.prompt[0].lower()
        if cmd == "stats":
            if _find_model() is None:
                if _try_remote(["stats"]):
                    return 0
                _no_model_help()
                return 1
            return cmd_stats()
        if cmd == "config":
            return cmd_config(args.prompt[1:])
        if cmd in ("weights", "w"):
            return cmd_weights(args.prompt)
        if cmd in ("processes", "ps", "procs"):
            return cmd_processes(args.prompt)
        if cmd in ("training", "train", "tr"):
            return cmd_training(args.prompt)
        if cmd in ("sync", "pull"):
            return cmd_sync(args.prompt)
        if cmd in ("control", "ctl"):
            return cmd_control(args.prompt)
        if cmd in ("push",):
            return cmd_push(args.prompt)
        if _find_model() is None:
            if _try_remote(args.prompt):
                return 0
            _no_model_help()
            return 1
        return cmd_ask(" ".join(args.prompt), max_tokens, temperature)

    if _find_model() is None:
        if _try_remote([]):
            return 0
        _no_model_help()
        return 1
    # Pipe detection: if stdin is not a TTY, read it as a single prompt
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return cmd_ask(piped, max_tokens, temperature)
    return cmd_chat(max_tokens, temperature)


if __name__ == "__main__":
    raise SystemExit(main())
