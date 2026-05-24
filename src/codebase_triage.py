"""Score and rank source files by relevance to the vulnerability.

Uses grep to score files against VulnSignal sink patterns,
replacing blind extraction of all source files. Zero LLM tokens consumed.

Reference: github.com/sharathbaddam/AgentWhetters-cybergym
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import List, Optional, Tuple

from hypothesis_parser import VulnSignal

logger = logging.getLogger(__name__)

_SUBPROCESS_TIMEOUT = 30

SKIP_DIRS = frozenset({
    "test", "tests", "doc", "docs", "example", "examples",
    "build", ".git", "node_modules", "__pycache__", "third_party",
})

SINK_PATTERNS: dict[str, list[str]] = {
    "heap-buffer-overflow": [r"memcpy|memmove|memset|strcpy|strncpy", r"sprintf|snprintf|malloc|realloc", r"fread|fgets|read"],
    "stack-buffer-overflow": [r"char[[:space:]]+[a-zA-Z_]", r"strcpy|sprintf|gets", r"alloca"],
    "buffer-overflow": [r"memcpy|memmove|strcpy|strncpy", r"sprintf|snprintf"],
    "use-after-free": [r"free\(", r"delete", r"realloc\("],
    "double-free": [r"free\(", r"delete"],
    "null-pointer-dereference": [r"malloc|calloc|strdup", r"->"],
    "integer-overflow": [r"\*[[:space:]]*sizeof", r"width[[:space:]]*\*[[:space:]]*height", r"atoi|strtol|strtoul"],
    "divide-by-zero": [r"/[[:space:]]*[a-zA-Z_]", r"%[[:space:]]*[a-zA-Z_]"],
    "assertion-failure": [r"assert", r"abort"],
    "out-of-bounds-read": [r"memcpy|memmove|memset", r"fread|fgets|read"],
    "out-of-bounds-write": [r"memcpy|memmove|memset", r"strcpy|sprintf"],
    "out-of-bounds": [r"memcpy|memmove", r"\[.*\]"],
    "uninitialized-memory": [r"malloc", r"alloca"],
}

_SOURCE_EXTENSIONS = frozenset({
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx",
    ".py", ".java", ".go", ".rs",
})


def _should_skip(filepath: str) -> bool:
    parts = filepath.lower().split(os.sep)
    return any(part in SKIP_DIRS for part in parts)


def _is_source_file(filepath: str) -> bool:
    _, ext = os.path.splitext(filepath)
    basename = os.path.basename(filepath)
    return ext.lower() in _SOURCE_EXTENSIONS or basename in ("Makefile", "CMakeLists.txt")


def _grep_count(repo_path: str, pattern: str, filepath: str) -> int:
    try:
        result = subprocess.run(
            ["grep", "-c", "-E", pattern, filepath],
            capture_output=True, text=True,
            timeout=_SUBPROCESS_TIMEOUT, cwd=repo_path,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return 0


class CodebaseTriage:
    """Score and rank source files by relevance to a vulnerability."""

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self._file_cache: list[str] = []

    def _list_source_files(self) -> list[str]:
        if self._file_cache:
            return self._file_cache
        files = []
        for root, dirs, filenames in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS]
            for fname in filenames:
                full_path = os.path.join(root, fname)
                if _is_source_file(full_path) and not _should_skip(full_path):
                    files.append(full_path)
        self._file_cache = files
        return files

    def score_and_rank(
        self, signal: VulnSignal, max_results: int = 5,
    ) -> List[Tuple[str, float]]:
        """Score and rank source files. Returns (filepath, score) sorted desc."""
        files = self._list_source_files()
        scores: list[Tuple[str, float]] = []

        for filepath in files:
            score = 0.0
            rel_path = os.path.relpath(filepath, self.repo_path)

            if signal.vulnerable_function != "unknown":
                if signal.vulnerable_function.lower() in rel_path.lower():
                    score += 10.0
                count = _grep_count(self.repo_path, signal.vulnerable_function, filepath)
                if count > 0:
                    score += 10.0

            if signal.file_hint:
                if signal.file_hint.lower() in rel_path.lower():
                    score += 8.0
                elif os.path.basename(signal.file_hint).lower() in rel_path.lower():
                    score += 5.0

            for i, func in enumerate(signal.stack_trace[:5]):
                count = _grep_count(self.repo_path, func, filepath)
                if count > 0:
                    score += max(5.0 - i, 1.0)

            vuln_patterns = SINK_PATTERNS.get(signal.vuln_class, [])
            for pattern in vuln_patterns:
                count = _grep_count(self.repo_path, pattern, filepath)
                if count > 0:
                    score += min(count, 4)

            _, ext = os.path.splitext(filepath)
            if ext in (".c", ".cc", ".cpp", ".cxx"):
                score += 1.0

            if score > 0:
                scores.append((filepath, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:max_results]

    def get_code_snippet(
        self, filepath: str, function_name: str = "unknown",
        context_lines: int = 50,
    ) -> Optional[str]:
        """Extract a code snippet centered on the vulnerable function."""
        try:
            with open(filepath, "r", errors="replace") as f:
                lines = f.readlines()
        except OSError:
            return None

        if not lines:
            return None

        if len(lines) <= context_lines * 2:
            return "".join(lines)

        target_line = -1
        if function_name != "unknown":
            func_lower = function_name.lower()
            for i, line in enumerate(lines):
                if func_lower in line.lower():
                    target_line = i
                    break

        if target_line >= 0:
            start = max(0, target_line - context_lines)
            end = min(len(lines), target_line + context_lines)
        else:
            start = 0
            end = min(len(lines), context_lines * 2)

        header = f"[Lines {start + 1}-{end} of {len(lines)}]\n"
        return header + "".join(lines[start:end])
