"""
lifers/core/output_formatter.py
──────────────────────────────────────
Output optimization layer:
  - Token streaming via generator
  - Markdown / plain-text / JSON normalization
  - Hallucination / low-confidence filter
  - Response length guard
"""
from __future__ import annotations
import re, json, logging
from dataclasses import dataclass
from enum import Enum
from typing import Generator, Iterable, Optional

log = logging.getLogger(__name__)


class OutputFormat(str, Enum):
    MARKDOWN = "markdown"
    PLAIN    = "plain"
    JSON     = "json"


_HEDGE_RE      = re.compile(r"\b(I (don't|do not) (know|have access)|As an AI|As a language model"
                             r"|我不确定|可能是|据我所知但不确定)\b", re.I)
_REPETITION_RE = re.compile(r"(\b\w{4,}\b)(\s+\1){3,}")
_TRUNCATED_RE  = re.compile(r"\.{3,}$|…$")
_SPECIAL_TOKENS = re.compile(r"<\|[^|]+\|>")


@dataclass
class FormattedOutput:
    text:               str
    format:             OutputFormat
    token_count:        int
    hallucination_flag: bool
    confidence:         float
    warnings:           list[str]


class OutputFormatter:
    _CPT = 3.5  # chars per token estimate (Chinese/English mix)

    def __init__(
        self,
        fmt:                  OutputFormat = OutputFormat.MARKDOWN,
        max_response_tokens:  int   = 1024,
        hallucination_filter: bool  = True,
        confidence_threshold: float = 0.4,
        stream:               bool  = True,
    ) -> None:
        self.fmt                  = fmt
        self.max_chars            = int(max_response_tokens * self._CPT)
        self.hallucination_filter = hallucination_filter
        self.confidence_threshold = confidence_threshold
        self.stream               = stream

    # ── Streaming ─────────────────────────────────────────────────────────────

    def stream_tokens(self, chunks: Iterable[str]) -> Generator[str, None, FormattedOutput]:
        buf = []
        for chunk in chunks:
            c = _SPECIAL_TOKENS.sub("", chunk)
            if c:
                buf.append(c)
                yield c
        return self._analyze("".join(buf))

    # ── Single-shot ───────────────────────────────────────────────────────────

    def format(self, raw: str) -> FormattedOutput:
        return self._analyze(self._clean(raw))

    # ── Private ───────────────────────────────────────────────────────────────

    def _clean(self, raw: str) -> str:
        text = _SPECIAL_TOKENS.sub("", raw)
        if self.fmt == OutputFormat.PLAIN:
            text = re.sub(r"[*_#`~]+", "", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
        elif self.fmt == OutputFormat.JSON:
            m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
            if m:
                text = m.group(1).strip()
        if len(text) > self.max_chars:
            text = text[: self.max_chars].rstrip() + "…"
            log.warning("Response truncated to %d chars", self.max_chars)
        return text.strip()

    def _analyze(self, text: str) -> FormattedOutput:
        warnings: list[str] = []
        confidence = 1.0

        if not text:
            return FormattedOutput("", self.fmt, 0, True, 0.0,
                                   ["Empty response from model"])

        if self.hallucination_filter:
            hedges = len(_HEDGE_RE.findall(text))
            if hedges:
                confidence -= hedges * 0.15
                warnings.append(f"Hedge phrases ({hedges})")
            if _REPETITION_RE.search(text):
                confidence -= 0.3
                warnings.append("Repetition loop")
            if _TRUNCATED_RE.search(text):
                confidence -= 0.1
                warnings.append("Truncated response")
            confidence = max(0.0, min(1.0, confidence))

        flag = self.hallucination_filter and confidence < self.confidence_threshold
        if flag:
            warnings.append(f"Low confidence: {confidence:.2f}")
            log.warning("Low-confidence response flagged")

        return FormattedOutput(
            text               = text,
            format             = self.fmt,
            token_count        = int(len(text) / self._CPT),
            hallucination_flag = flag,
            confidence         = round(confidence, 3),
            warnings           = warnings,
        )


def from_stack(stack_cfg: dict) -> OutputFormatter:
    out = stack_cfg.get("output", {})
    try:
        fmt = OutputFormat(out.get("format", "markdown"))
    except ValueError:
        fmt = OutputFormat.MARKDOWN
    return OutputFormatter(
        fmt                  = fmt,
        max_response_tokens  = out.get("max_response_tokens", 1024),
        hallucination_filter = out.get("hallucination_filter", True),
        confidence_threshold = out.get("confidence_threshold", 0.4),
        stream               = out.get("stream", True),
    )
