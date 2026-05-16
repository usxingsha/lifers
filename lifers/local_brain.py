"""
Local brain: self-contained local language model backends (Markov / TinyTransformer / DeepTransformer).

Extracted from agent.py — owns:
- ``AgentConfig`` (shared config dataclass)
- ``LocalBrain`` (hot-reloadable LM wrapper with wallclock timeout)
- Helper functions for sampling, timeouts, and fast-path env knobs
"""

from __future__ import annotations

import os
import random
import signal
import sys
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Set

from lifers.inference_pipeline import log_inference
from lifers.markov_lm import MarkovWeights, generate as markov_generate
from lifers.model_names import canonical_brain_model, default_weight_paths
from lifers.speed_env import local_lm_max_chars as speed_local_lm_max_chars
from lifers.stack_env import load_stack
from lifers.transformer_lm import TinyTransformerWeights, generate_text as tt_generate


@dataclass
class AgentConfig:
    """Shared configuration for LifersAgent and LocalBrain."""
    root_dir: Path
    model: str = "lifers"  # lifers|transformer|markov
    sandbox: bool = True
    max_tool_steps: int = 6
    local_lm_max_chars: int = 200


def _local_lm_sampling_from_stack(root: Path, backend: str) -> tuple[float, int]:
    """Read stack.json brain.local_lm_sampling (lifers|transformer|markov) temperature / top_k."""
    defaults = {"lifers": (0.8, 80), "transformer": (1.1, 80), "markov": (0.9, 80)}
    d_temp, d_top = defaults.get(backend, (1.1, 80))
    try:
        stack = load_stack(root)
        brain = stack.get("brain") or {}
        block = (brain.get("local_lm_sampling") or {}).get(backend) or {}
        if not isinstance(block, dict):
            return d_temp, d_top
        t = float(block.get("temperature", d_temp))
        k = int(block.get("top_k", d_top))
        return max(0.01, min(t, 4.0)), max(1, min(k, 256))
    except (TypeError, ValueError):
        return d_temp, d_top


def _quick_fast_enabled() -> bool:
    return os.environ.get("LIFERS_QUICK_FAST", "0").strip().lower() in ("1", "true", "yes", "on")


def _quick_skip_kb() -> bool:
    return os.environ.get("LIFERS_QUICK_SKIP_KB", "0").strip().lower() in ("1", "true", "yes", "on")


def _quick_generate_wallclock_sec() -> float:
    """
    CHAT_QUICK LocalBrain.generate wallclock timeout in seconds.

    - LIFERS_QUICK_GENERATE_TIMEOUT_SEC: explicit value >=0.5 enables; 0/false/off disables.
    - Unset with Agents UI Bridge: defaults to 120s.
    - Unset on POSIX (non-bridge): 120s.
    - Otherwise: 0.0 (disabled).
    """
    raw = os.environ.get("LIFERS_QUICK_GENERATE_TIMEOUT_SEC", "").strip()
    if raw.lower() in ("0", "false", "no", "off"):
        return 0.0
    if raw:
        try:
            return max(0.5, float(raw))
        except ValueError:
            pass
    bridge = os.environ.get("LIFERS_AGENTS_UI_BRIDGE", "").strip().lower() in ("1", "true", "yes", "on")
    if bridge:
        return 120.0
    if os.name == "posix":
        return 120.0
    return 0.0


def _generate_timeout_reply_message(sec: float) -> str:
    return (
        "本地推理已超时（常见于 **训练占满 CPU**、**prompt 过长**或首次冷加载大权重）。建议：\n"
        "- 先 **`bash scripts/lifers_train_ctl.sh pause`** 或停掉占满 CPU 的进程后再试；或\n"
        "- 缩短上下文：调小 **`LIFERS_QUICK_SESSION_CONTEXT_CHARS`** / **`LIFERS_QUICK_PACK_MAX_CHARS`**，或新开会话；\n"
        "- **扩展**：`lifers.edgeGenerateTimeoutSec` → 注入 **LIFERS_QUICK_GENERATE_TIMEOUT_SEC**（秒；**0** 关闭墙钟保护，可能长时间无回复）。\n"
        f"- 当前墙钟上限约 **{int(sec)}s**。"
    )


class _QuickGenerateTimeout(Exception):
    """Raised when wallclock time is exceeded during local LM generation."""
    pass


# Module-level set to track orphaned futures from Windows timeout path.
# Completed futures are cleaned on each new call to prevent unbounded growth.
_ORPHANED_FUTURES: Set[Future] = set()


def _reap_orphaned() -> None:
    done = {f for f in _ORPHANED_FUTURES if f.done()}
    _ORPHANED_FUTURES.difference_update(done)


def _generate_with_wallclock_timeout(brain: "LocalBrain", prompt: str, max_out_chars: Optional[int]) -> str:
    """Run brain.generate() with a wallclock timeout."""
    sec = _quick_generate_wallclock_sec()
    if sec <= 0:
        return brain.generate(prompt, max_out_chars=max_out_chars).strip()

    if os.name == "posix":
        def _h(_sig: int, _frame: Any) -> None:
            raise _QuickGenerateTimeout()

        old = signal.signal(signal.SIGALRM, _h)
        try:
            signal.setitimer(signal.ITIMER_REAL, sec, 0.0)
            try:
                return brain.generate(prompt, max_out_chars=max_out_chars).strip()
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)
        except _QuickGenerateTimeout:
            log_inference("generate_local", timeout_sec=sec, model=brain.model, prompt_chars=len(prompt))
            return _generate_timeout_reply_message(sec)
        finally:
            signal.signal(signal.SIGALRM, old)

    # Windows path: ThreadPoolExecutor with timeout.
    # Python threads cannot be killed, so a timed-out thread continues running
    # in the background until brain.generate() returns.  We track orphaned
    # futures and reap completed ones on each call so they don't accumulate
    # unboundedly.  Daemon threads are automatically cleaned up at process exit.
    _reap_orphaned()
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(lambda: brain.generate(prompt, max_out_chars=max_out_chars).strip())
        try:
            return fut.result(timeout=sec)
        except FuturesTimeout:
            log_inference("generate_local", timeout_sec=sec, model=brain.model, prompt_chars=len(prompt), warning="thread_orphaned")
            _ORPHANED_FUTURES.add(fut)
            return _generate_timeout_reply_message(sec)
    finally:
        ex.shutdown(wait=False)


class LocalBrain:
    """Hot-reloadable wrapper around DeepTransformer, Markov, or TinyTransformer LM."""

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.model = canonical_brain_model(cfg.model)
        self._dw_sig: tuple[str, float] | None = None
        self._dw_loaded: object | None = None
        self._tw_sig: tuple[str, float] | None = None
        self._tw_loaded: object | None = None
        self._mw_sig: tuple[str, float] | None = None
        self._mw_loaded: object | None = None

    def _weights_path(self) -> Path:
        stack = load_stack(self.cfg.root_dir)
        brain_s = stack.get("brain") or {}
        wmap = brain_s.get("weights") or {}
        rel = wmap.get(self.model)
        if isinstance(rel, str) and rel.strip():
            p = (self.cfg.root_dir / rel.strip()).resolve()
            if p.is_file():
                return p
        root = self.cfg.root_dir
        for rel in default_weight_paths(self.model):
            cand = (root / rel).resolve()
            if cand.is_file():
                return cand
        # Also search lifers/weights/ (common layout where weights live inside the package)
        for rel in default_weight_paths(self.model):
            cand = (root / "lifers" / rel).resolve()
            if cand.is_file():
                return cand
        return (root / default_weight_paths(self.model)[0]).resolve()

    def _transformer_weights(self, wp: Path) -> TinyTransformerWeights:
        """Hot-reload by mtime; LIFERS_FORCE_WEIGHT_RELOAD=1 skips cache."""
        force = os.environ.get("LIFERS_FORCE_WEIGHT_RELOAD", "").strip().lower() in ("1", "true", "yes", "on")
        try:
            mt = wp.stat().st_mtime
        except OSError:
            mt = 0.0
        key = str(wp.resolve())
        sig = (key, mt)
        if (
            not force
            and self._tw_sig == sig
            and self._tw_loaded is not None
            and isinstance(self._tw_loaded, TinyTransformerWeights)
        ):
            return self._tw_loaded
        w = TinyTransformerWeights.load(wp)
        try:
            from lifers.lora_finetune import lora_sidecar_path, merge_lora_into_weights

            lp = lora_sidecar_path(self.cfg.root_dir, load_stack(self.cfg.root_dir))
            if lp is not None:
                w = merge_lora_into_weights(w, lp)
        except Exception as exc:
            sys.stderr.write(f"LIFERS_PROGRESS lora_merge_skipped {exc}\n")
            sys.stderr.flush()
        self._tw_sig = sig
        self._tw_loaded = w
        return w

    def _deep_weights(self, wp: Path):
        """Hot-reload deep transformer weights by mtime."""
        from lifers.deep_transformer import DeepTransformerWeights

        force = os.environ.get("LIFERS_FORCE_WEIGHT_RELOAD", "").strip().lower() in ("1", "true", "yes", "on")
        try:
            mt = wp.stat().st_mtime
        except OSError:
            mt = 0.0
        key = str(wp.resolve())
        sig = (key, mt)
        if (
            not force
            and self._dw_sig == sig
            and self._dw_loaded is not None
            and isinstance(self._dw_loaded, DeepTransformerWeights)
        ):
            return self._dw_loaded
        w = DeepTransformerWeights.load(wp)
        self._dw_sig = sig
        self._dw_loaded = w
        return w

    def _markov_weights(self, wp: Path) -> MarkovWeights:
        force = os.environ.get("LIFERS_FORCE_WEIGHT_RELOAD", "").strip().lower() in ("1", "true", "yes", "on")
        try:
            mt = wp.stat().st_mtime
        except OSError:
            mt = 0.0
        key = str(wp.resolve())
        sig = (key, mt)
        if (
            not force
            and self._mw_sig == sig
            and self._mw_loaded is not None
            and isinstance(self._mw_loaded, MarkovWeights)
        ):
            return self._mw_loaded
        w = MarkovWeights.load(wp)
        self._mw_sig = sig
        self._mw_loaded = w
        return w

    def _generate_markov(self, prompt: str, mc: int) -> str:
        """Generate with Markov backend, with size guard."""
        wp = self._weights_path()
        try:
            raw_cap = os.environ.get("LIFERS_MARKOV_JSON_MAX_BYTES", str(128 * 1024 * 1024)).strip()
            mx = int(raw_cap) if raw_cap else 0
        except ValueError:
            mx = 128 * 1024 * 1024
        if mx > 0:
            try:
                sz = wp.stat().st_size
            except OSError:
                sz = 0
            if sz > mx:
                mb = sz // (1024 * 1024)
                cap_mb = mx // (1024 * 1024)
                return (
                    f"（Markov 权重文件约 **{mb}MB**，超过快路径安全上限 **{cap_mb}MB**，未加载以免 Bridge 卡死。）\n"
                    f"文件：`{wp.name}`。可选：**缩小** `weights/lifers_markov.json`、改用 **transformer** 权重、"
                    f"或在本机/扩展环境设 **`LIFERS_MARKOV_JSON_MAX_BYTES`**（字节）提高上限后再 Reload。"
                )
        w = self._markov_weights(wp)
        m_temp, m_top = _local_lm_sampling_from_stack(self.cfg.root_dir, "markov")
        return markov_generate(w, prompt=prompt, max_chars=max(mc, 120), seed=3, temperature=m_temp, top_k=m_top).strip()

    def _generate_transformer(self, prompt: str, mc: int) -> str:
        """Generate with transformer backend."""
        wp = self._weights_path()
        try:
            sz = int(wp.stat().st_size)
        except OSError:
            sz = 0
        log_inference(
            "transformer_generate_begin",
            weight_mb=round(sz / 1_000_000, 2),
            prompt_chars=len(prompt),
            max_out_chars=mc,
        )
        w = self._transformer_weights(wp)
        t_temp, t_top = _local_lm_sampling_from_stack(self.cfg.root_dir, "transformer")
        out = tt_generate(w, prompt=prompt, max_chars=mc, seed=3, temperature=t_temp, top_k=t_top).strip()
        log_inference("transformer_generate_end", reply_chars=len(out))
        return out

    def _generate_deep(self, prompt: str, mc: int) -> str:
        """Generate with the deep multi-layer transformer backend."""
        from lifers.deep_transformer import generate_text

        wp = self._weights_path()
        try:
            sz = int(wp.stat().st_size)
        except OSError:
            sz = 0
        log_inference(
            "lifers_generate_begin",
            weight_mb=round(sz / 1_000_000, 2),
            prompt_chars=len(prompt),
            max_out_chars=mc,
        )
        w = self._deep_weights(wp)
        d_temp, d_top = _local_lm_sampling_from_stack(self.cfg.root_dir, "lifers")
        max_tokens = max(32, min(mc, 2000))
        out = generate_text(
            w, prompt,
            max_new_tokens=max_tokens,
            temperature=d_temp,
            seed=random.randint(0, 2**31 - 1),
        ).strip()
        log_inference("lifers_generate_end", reply_chars=len(out))
        return out

    def generate(self, prompt: str, max_out_chars: Optional[int] = None) -> str:
        wp = self._weights_path()
        mc = speed_local_lm_max_chars(int(getattr(self.cfg, "local_lm_max_chars", 200) or 200))
        if max_out_chars is not None:
            mc = max(32, min(mc, int(max_out_chars)))
        if not wp.exists():
            # Graceful degradation: try lower-capability backends
            fb_order = ("transformer", "markov") if self.model == "lifers" else (
                ("markov",) if self.model == "transformer" else ()
            )
            for fb_model in fb_order:
                fallback = default_weight_paths(fb_model)
                for rel in fallback:
                    cand = (self.cfg.root_dir / rel).resolve()
                    if cand.is_file():
                        sys.stderr.write(f"LIFERS_PROGRESS {self.model} weights missing — falling back to {fb_model}\n")
                        sys.stderr.flush()
                        self.model = fb_model
                        if fb_model == "transformer":
                            return self._generate_transformer(prompt, mc)
                        return self._generate_markov(prompt, mc)
            return (
                f"（未找到权重文件。请运行训练流程或将权重放置于 `weights/` 目录。）\n"
                f"期望路径：`{wp}`"
            )

        if self.model == "markov":
            return self._generate_markov(prompt, mc)
        if self.model == "lifers":
            try:
                return self._generate_deep(prompt, mc)
            except Exception as exc:
                log_inference("lifers_generate_error", error=str(exc)[:200])
                # Fallback: try transformer or markov
                for fb_model in ("transformer", "markov"):
                    fb_paths = default_weight_paths(fb_model)
                    for rel in fb_paths:
                        cand = (self.cfg.root_dir / rel).resolve()
                        if cand.is_file():
                            sys.stderr.write(f"LIFERS_PROGRESS lifers failed ({exc}) — falling back to {fb_model}\n")
                            sys.stderr.flush()
                            self.model = fb_model
                            if fb_model == "transformer":
                                return self._generate_transformer(prompt, mc)
                            return self._generate_markov(prompt, mc)
                return (
                    f"（Deep Transformer 生成失败：{exc}。且无备用权重。）\n"
                    f"建议：检查权重文件是否完整，或重新训练。"
                )
        # transformer path with fallback
        try:
            return self._generate_transformer(prompt, mc)
        except Exception as exc:
            log_inference("transformer_generate_error", error=str(exc)[:200])
            # Fallback: if transformer fails, try lifers or markov
            for fb_model in ("lifers", "markov"):
                fb_paths = default_weight_paths(fb_model)
                for rel in fb_paths:
                    cand = (self.cfg.root_dir / rel).resolve()
                    if cand.is_file():
                        sys.stderr.write(f"LIFERS_PROGRESS transformer failed ({exc}) — falling back to {fb_model}\n")
                        sys.stderr.flush()
                        self.model = fb_model
                        if fb_model == "lifers":
                            return self._generate_deep(prompt, mc)
                        return self._generate_markov(prompt, mc)
            return (
                f"（Transformer 生成失败：{exc}。且无备用权重。）\n"
                f"建议：检查权重文件是否完整，或重新训练。"
            )
