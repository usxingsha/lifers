"""
lifers/core/input_validator.py
─────────────────────────────────────
Input processing layer: sanitization, validation, length control.
Called first — before routing or inference.
"""
from __future__ import annotations
import re, unicodedata, logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_PROMPT_INJECTION = re.compile(
    r"(ignore\s+previous|disregard\s+(all|above)|you\s+are\s+now\s+a"
    r"|forget\s+(everything|all)|system\s*:\s*you|<\|system\|>)",
    re.IGNORECASE,
)
_CONTROL_CHARS  = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_REPEATED_CHARS = re.compile(r"(.)\1{80,}")
_URL_FLOOD      = re.compile(r"(https?://\S+\s*){5,}")


@dataclass
class ValidationResult:
    ok:       bool
    text:     str
    warnings: list[str] = field(default_factory=list)
    errors:   list[str] = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return not self.ok


class InputValidator:
    """
    Sanitize and validate raw user input before it enters the pipeline.

    Usage
    -----
    v = InputValidator()
    result = v.validate(raw_text)
    if result.rejected:
        return result.errors[0]
    clean = result.text
    """

    def __init__(self, max_chars: int = 4096, strict: bool = False) -> None:
        self.max_chars = max_chars
        self.strict    = strict

    def validate(self, raw: str) -> ValidationResult:
        warnings, errors = [], []

        if not raw or not raw.strip():
            return ValidationResult(ok=False, text=raw, errors=["Empty input"])

        if len(raw) > self.max_chars:
            raw = raw[: self.max_chars]
            warnings.append(f"Input truncated to {self.max_chars} chars")

        # Remove control characters
        clean = _CONTROL_CHARS.sub("", raw)
        if clean != raw:
            warnings.append("Removed control characters")
        raw = clean

        # Unicode normalize (consistent CJK/emoji)
        raw = unicodedata.normalize("NFC", raw)

        # Prompt injection
        if _PROMPT_INJECTION.search(raw):
            msg = "Prompt-injection pattern detected"
            warnings.append(msg)
            log.warning("InputValidator: %s", msg)

        if _REPEATED_CHARS.search(raw):
            warnings.append("Repeated-character flood")

        if _URL_FLOOD.search(raw):
            warnings.append("URL flood detected")

        if self.strict and warnings:
            errors.extend(warnings)

        return ValidationResult(ok=len(errors) == 0, text=raw,
                                warnings=warnings, errors=errors)


_default = InputValidator()


def validate_input(raw: str, **kw) -> ValidationResult:
    return InputValidator(**kw).validate(raw) if kw else _default.validate(raw)
