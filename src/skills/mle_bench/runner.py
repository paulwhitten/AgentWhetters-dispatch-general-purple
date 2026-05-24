"""Subprocess code execution for MLE-Bench scripts."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

EXECUTION_TIMEOUT = 900  # 15 minutes max for full pipeline
PHASE0_TIMEOUT = 180     # 3 minutes for Phase 0 deterministic script


async def run_code(code: str, data_dir: Path, timeout: int | None = None) -> tuple[bool, str]:
    """Write code to a temp file and execute it. Returns (success, output)."""
    _timeout = timeout if timeout is not None else EXECUTION_TIMEOUT
    script = data_dir.parent / "_ml_solution.py"
    script.write_text(code)

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script),
            cwd=str(data_dir.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=None,  # inherit environment
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False, f"Execution timed out after {_timeout}s"

        output = stdout.decode(errors="replace")
        return proc.returncode == 0, output

    except Exception as exc:
        return False, str(exc)
