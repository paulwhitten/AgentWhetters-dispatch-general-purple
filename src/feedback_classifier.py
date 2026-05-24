"""Classify execution feedback into actionable categories.

Parses ASan output and exit codes into 8 distinct categories,
each with specific refinement instructions for the LLM.

Reference: github.com/sharathbaddam/AgentWhetters-cybergym
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional, Tuple

from hypothesis_parser import VulnSignal


class FeedbackCategory(Enum):
    SUCCESS = "success"
    WRONG_LOCATION = "wrong_location"
    WRONG_CRASH = "wrong_crash"
    PARTIAL_CRASH = "partial_crash"
    BLOCKED_ASSERTION = "blocked_assertion"
    PARSER_REJECTED = "parser_rejected"
    NO_CRASH = "no_crash"
    TIMEOUT = "timeout"


_ASAN_CLASS_RE = re.compile(r"AddressSanitizer:\s+(\S+)")
_CRASH_FUNC_RE = re.compile(r"#0\s+0x[\da-fA-F]+\s+in\s+(\w+)")
_ASSERTION_RE = re.compile(
    r"(?:assert(?:ion)?|abort)\s*(?:failed)?.*?(?:at\s+)?(\S+:\d+)?", re.I,
)
_REJECTION_KEYWORDS = (
    "invalid", "corrupt", "malformed", "bad", "unsupported",
    "unexpected", "unrecognized", "unknown format",
    "not a valid", "failed to parse", "cannot open",
    "wrong magic", "header error",
)
_REJECTION_MSG_RE = re.compile(r"(?:error|warning|fatal)[:\s]+(.{10,120})", re.I)


def _extract_asan_class(output: str) -> Optional[str]:
    m = _ASAN_CLASS_RE.search(output)
    return m.group(1) if m else None


def _extract_crash_function(output: str) -> Optional[str]:
    m = _CRASH_FUNC_RE.search(output)
    return m.group(1) if m else None


def _extract_assertion(output: str) -> str:
    m = _ASSERTION_RE.search(output)
    if m and m.group(1):
        return m.group(1)
    m2 = re.search(r"Assertion\s+[`'\"](.+?)[`'\"]", output)
    if m2:
        return m2.group(1)
    return "unknown location"


def _extract_rejection_reason(output: str) -> str:
    combined = output.lower()
    for kw in _REJECTION_KEYWORDS:
        idx = combined.find(kw)
        if idx != -1:
            start = max(0, idx - 20)
            end = min(len(output), idx + len(kw) + 80)
            return output[start:end].strip()
    m = _REJECTION_MSG_RE.search(output)
    if m:
        return m.group(1).strip()
    return "input rejected"


def _is_timeout(output: str, error: str) -> bool:
    combined = (output + error).lower()
    return any(kw in combined for kw in ("timed out", "timeout", "time limit"))


# Near-miss categories that qualify for binary mutation
NEAR_MISS_CATEGORIES = frozenset({
    FeedbackCategory.WRONG_LOCATION,
    FeedbackCategory.WRONG_CRASH,
    FeedbackCategory.PARTIAL_CRASH,
})


def classify(
    exit_code: int, output: str, error: str,
    signal: Optional[VulnSignal] = None,
) -> Tuple[FeedbackCategory, str]:
    """Classify test feedback into a category with action string."""
    combined_output = f"{output}\n{error}"

    if _is_timeout(output, error):
        return (
            FeedbackCategory.TIMEOUT,
            "Execution timed out. Try a simpler/smaller input that "
            "reaches the vulnerable code path faster.",
        )

    asan_class = _extract_asan_class(combined_output)
    crash_func = _extract_crash_function(combined_output)

    if asan_class and signal:
        asan_normalized = asan_class.lower().replace("_", "-")
        signal_normalized = signal.vuln_class.lower().replace("_", "-")
        class_matches = (
            asan_normalized == signal_normalized
            or asan_normalized in signal_normalized
            or signal_normalized in asan_normalized
        )
        func_matches = (
            signal.vulnerable_function != "unknown"
            and crash_func
            and signal.vulnerable_function.lower() == crash_func.lower()
        )
        if class_matches and (func_matches or signal.vulnerable_function == "unknown"):
            return (
                FeedbackCategory.SUCCESS,
                f"PoC triggered the correct vulnerability: {asan_class} "
                f"in {crash_func or 'unknown'}.",
            )
        if class_matches and not func_matches:
            return (
                FeedbackCategory.WRONG_LOCATION,
                f"Right vulnerability type ({asan_class}) but crashed in "
                f"{crash_func or 'unknown'} instead of {signal.vulnerable_function}. "
                f"Try corrupting a different field to route through the target function.",
            )
        if not class_matches:
            return (
                FeedbackCategory.WRONG_CRASH,
                f"Got {asan_class} but need {signal.vuln_class}. "
                f"Adjust the corrupt value: try boundary values like "
                f"0x7FFFFFFF, 0x00000001, or 0xFFFFFFFF.",
            )
    elif asan_class:
        return (
            FeedbackCategory.SUCCESS,
            f"ASan triggered: {asan_class} in {crash_func or 'unknown'}.",
        )

    crash_codes = {-11, 139, -6, 134, 137, -9}
    if exit_code in crash_codes or (exit_code < 0 and exit_code != 0):
        return (
            FeedbackCategory.PARTIAL_CRASH,
            f"Crash detected (exit_code={exit_code}) but no ASan detail. "
            f"Very close! Try slight adjustments to the corrupt value or offset.",
        )

    if "assert" in combined_output.lower() or "abort" in combined_output.lower():
        assertion_loc = _extract_assertion(combined_output)
        return (
            FeedbackCategory.BLOCKED_ASSERTION,
            f"Blocked by assertion at {assertion_loc}. Adjust your input "
            f"to satisfy the assertion condition.",
        )

    combined_lower = combined_output.lower()
    if any(kw in combined_lower for kw in _REJECTION_KEYWORDS):
        rejection = _extract_rejection_reason(combined_output)
        return (
            FeedbackCategory.PARSER_REJECTED,
            f"Parser rejected your input: '{rejection}'. "
            f"Fix the format issue while keeping the trigger corruption.",
        )

    if exit_code == 0:
        return (
            FeedbackCategory.NO_CRASH,
            "Input processed normally. The PoC does not exercise the "
            "vulnerable code path. Try more extreme values, deeper nesting, "
            "or a different code path.",
        )

    return (
        FeedbackCategory.PARTIAL_CRASH,
        f"Non-zero exit code ({exit_code}) without recognized sanitizer output. "
        f"Try adjusting the corrupt value.",
    )
