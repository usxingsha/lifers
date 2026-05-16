"""
Lifers CLI — 本地 AI 终端助手
用法:
  lifers               进入交互对话
  lifers "问题"         单次问答
  lifers stats          查看模型状态
  lifers config         查看/修改配置
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

    # Config
    cfg = _load_config()
    if cfg:
        if _HAS_RICH:
            console.print("\n[bold]Configuration:[/bold]")
            for k, v in cfg.items():
                console.print(f"  [dim]{k}[/dim] = {v}")
        else:
            print("\n配置:", cfg)

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
  /stats               查看模型和训练状态
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
  lifers config        查看配置
  对话命令: /stats /config /clear /help /quit
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
