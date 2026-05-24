"""Cross-task memory for learning from past vulnerability solutions.

Stores results after each task and queries at the start of each new
task to provide warm-start context. JSON-based, no external dependencies.

Reference: github.com/sharathbaddam/AgentWhetters-cybergym
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MemorySystem:
    """JSON-based cross-task memory for vulnerability solving patterns."""

    def __init__(self, memory_dir: str = "./memory"):
        self.memory_dir = memory_dir
        self._solved_path = os.path.join(memory_dir, "solved_tasks.json")
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        os.makedirs(self.memory_dir, exist_ok=True)
        if not os.path.exists(self._solved_path):
            self._atomic_write(self._solved_path, [])

    def _atomic_write(self, filepath: str, data: Any) -> None:
        dir_path = os.path.dirname(filepath)
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, filepath)
        except OSError as e:
            logger.warning("Memory write failed %s: %s", filepath, e)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _load_tasks(self) -> list[dict]:
        try:
            with open(self._solved_path, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def query_similar(
        self, vuln_class: str, domain: str = "unknown",
        crash_type: str = "unknown", limit: int = 3,
    ) -> dict:
        """Query memory for similar past tasks.

        Scoring: +3 vuln_class, +2 domain, +1 crash_type.
        Returns dict with 'similar_tasks' and 'failed_strategies'.
        """
        tasks = self._load_tasks()
        if not tasks:
            return {"similar_tasks": [], "failed_strategies": []}

        scored: list[tuple[float, dict]] = []
        failed_strategies: list[str] = []

        for task in tasks:
            score = 0.0
            if task.get("vuln_class", "") == vuln_class:
                score += 3.0
            if task.get("domain", "") == domain and domain != "unknown":
                score += 2.0
            if task.get("crash_type", "") == crash_type and crash_type != "unknown":
                score += 1.0

            if score > 0:
                if task.get("solved"):
                    scored.append((score, task))
                else:
                    strategy = task.get("failed_strategy", "")
                    if strategy and strategy not in failed_strategies:
                        failed_strategies.append(strategy)

        scored.sort(key=lambda x: x[0], reverse=True)
        similar = [t for _, t in scored[:limit]]
        return {
            "similar_tasks": similar,
            "failed_strategies": failed_strategies[:limit],
        }

    def save_result(self, task_id: str, signal: Any, result: dict) -> None:
        """Save a task result (success or failure) to memory."""
        tasks = self._load_tasks()
        entry: Dict[str, Any] = {
            "task_id": task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "vuln_class": getattr(signal, "vuln_class", "unknown"),
            "domain": getattr(signal, "project_domain", "unknown"),
            "crash_type": getattr(signal, "crash_type", "unknown"),
            "vulnerable_function": getattr(signal, "vulnerable_function", "unknown"),
            "solved": result.get("solved", False),
            "winning_pattern": result.get("winning_pattern", ""),
            "iterations": result.get("iterations", 0),
            "failed_strategy": result.get("failed_strategy", ""),
        }
        tasks.append(entry)
        self._atomic_write(self._solved_path, tasks)
        logger.info("Memory saved task %s (solved=%s, iterations=%d)",
                     task_id, result.get("solved"), result.get("iterations", 0))

    def get_failed_strategies(self, vuln_class: str) -> List[str]:
        tasks = self._load_tasks()
        strategies = []
        for task in tasks:
            if (task.get("vuln_class") == vuln_class
                    and not task.get("solved")
                    and task.get("failed_strategy")):
                strategy = task["failed_strategy"]
                if strategy not in strategies:
                    strategies.append(strategy)
        return strategies
