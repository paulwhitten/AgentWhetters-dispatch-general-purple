"""SWE-bench Pro skill module — flat agentic loop with test gate.

Ported from the winning SWE-bench Pro purple agent. The LLM gets a bash shell
inside a Docker container and autonomously reads, greps, edits, and tests.
A test gate rejects premature `done` calls when tests fail, with baseline
failure filtering and a QA fix phase.

Strategies are defined as `SolveStrategy` instances. Pass a strategy to
`solve_instance` to control model routing, step budgets, prompt selection,
auto-verify, and baseline reinforcement without changing the core loop.

Reference: other-agents/swe-bench-pro/swe-bench-purple-agent/src/purple/server.py
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import re
import resource
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx
from openai import AsyncOpenAI

from docker_runner import DockerRunner
from usage import tracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants (not strategy-specific)
# ---------------------------------------------------------------------------

TOOL_RESULT_LIMIT = 30_000
COMMAND_TIMEOUT = 300
TEST_FAILURE_EXTRACT_LIMIT = 6000
QA_STALE_CAP = 2
LOG_DIR = Path("logs")

_OPENAI_TIMEOUT = httpx.Timeout(connect=60.0, read=1800.0, write=60.0, pool=60.0)

# ---------------------------------------------------------------------------
# Edit detection for auto-verify
# ---------------------------------------------------------------------------

_EDIT_PATTERNS = re.compile(
    r"(?:sed\s+-i|tee\s|cat\s*>|echo\s.*>>|python3?\s+-c.*open\(|"
    r"printf\s.*>|perl\s+-[ip]|patch\s|git\s+apply)",
)

_VERIFY_SKIP_PATTERNS = re.compile(
    r"(?:grep\s+-[nrc]|diff\s|cat\s[^>]|head\s|tail\s)",
)


def _needs_auto_verify(command: str) -> bool:
    if _VERIFY_SKIP_PATTERNS.search(command):
        return False
    return bool(_EDIT_PATTERNS.search(command))


def _make_verify_command(command: str) -> str | None:
    m = re.search(r"sed\s+-i\S*\s+(?:'[^']*'|\"[^\"]*\")\s+(\S+)", command)
    if m:
        return f"tail -5 {m.group(1)} && echo '--- auto-verify ok ---'"
    m = re.search(r"(?:cat|tee)\s*>{1,2}\s*(\S+)", command)
    if m:
        return f"tail -5 {m.group(1)} && echo '--- auto-verify ok ---'"
    m = re.search(r"echo\s.*>{1,2}\s*(\S+)", command)
    if m:
        return f"tail -3 {m.group(1)} && echo '--- auto-verify ok ---'"
    return None

# ---------------------------------------------------------------------------
# Model classification
# ---------------------------------------------------------------------------

_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _REASONING_MODEL_PREFIXES)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_REASONING = textwrap.dedent("""\
    You are an expert software engineer. Solve the coding task in the problem
    statement correctly and efficiently.

    You have a LIMITED budget of {step_limit} shell calls. Each call costs real
    money. A good solution typically needs 8-12 calls.

    <grading>
    After you submit, a rigorous hidden test patch will be applied that adds
    targeted tests for every requirement -- including edge cases. Your code
    must handle all of these correctly.

    Write code as if it is safety-critical. Every new function must be hardened
    against: null/undefined/falsy inputs, empty arrays, missing object keys,
    wrong types, and must preserve exact ordering and return types.
    </grading>

    <editing>
    When using Python string .replace() to edit files, the old string must
    match EXACTLY or the replacement silently does nothing. After every edit,
    VERIFY it landed by piping to grep or diff.
    Make MINIMAL changes. Change only the lines you need.
    </editing>

    <efficiency>
    Minimize calls by batching work:
    - Batch file reads: sed + printf separators in one call
    - Batch edits: python heredoc + grep verify
    - Run the full test suite at most 1-2 times total
    </efficiency>

    <validation>
    Before calling done, review your changes:
    - Did you address ALL requirements in the problem statement?
    - For every new function: does it handle null, undefined, [], falsy inputs?
    - Did your test run pass?
    - Are your edits minimal?
    </validation>

    <rules>
    - Read a file before modifying it
    - Do NOT modify test files unless required
    - After EVERY file edit, verify the change landed
    - Call done when finished
    - If tests fail with pre-existing failures, IGNORE them and call done
    </rules>

    <baseline_failures>
    Some test suites have PRE-EXISTING failures unrelated to your task.
    If a test was already failing before your changes, do NOT try to fix it.
    Compare test output against the baseline -- only fix NEW failures caused
    by your changes. If only baseline failures remain, call done.
    </baseline_failures>
""")

SYSTEM_PROMPT_CLASSIC = textwrap.dedent("""\
    You are an expert software engineer. Fix a bug or implement a feature
    described in the problem statement.

    <critical_rules>
    1. Read a file BEFORE modifying it.
    2. After EVERY edit, verify it landed: grep -n 'expected_text' file
    3. Do NOT modify test files unless the problem explicitly requires it.
    4. Make MINIMAL changes.
    5. Call done when finished.
    6. If tests have PRE-EXISTING failures, IGNORE them and call done.
       Only fix NEW failures caused by YOUR changes.
    </critical_rules>

    <workflow>
    Step 1 - UNDERSTAND: Read the problem carefully. Identify relevant files.
    Step 2 - EXPLORE: Find and read the relevant files (batch reads).
    Step 3 - DIAGNOSE: Reason about root cause and edge cases.
    Step 4 - EDIT: Make minimal code changes.
    Step 5 - VERIFY: Confirm edits landed.
    Step 6 - TEST: Run the test suite (at most 1-2 times).
    Step 7 - FIX: If tests fail, diagnose and fix. Repeat until pass.
    Step 8 - Call done.
    </workflow>

    <grading>
    After you submit, hidden tests will cover all requirements including edge
    cases. Every function must handle null, undefined, empty arrays, missing
    keys, wrong types. Array ordering must match input order.
    </grading>

    You have {step_limit} tool calls. A good solution needs 8-15.
""")

SYSTEM_PROMPT_EXPLORE = textwrap.dedent("""\
    You are an expert software engineer performing the EXPLORATION phase of a
    bug fix or feature implementation.

    Your role is to investigate the codebase and gather the information needed
    to understand and solve the problem. Do NOT make any edits yet.

    <workflow>
    1. Identify the files most relevant to the problem statement.
    2. Use grep, find, and cat to locate key functions, classes, and call sites.
    3. Batch reads: combine multiple greps or use sed -n to read ranges.
    4. Focus on understanding the data flow and where the bug or feature lives.
    </workflow>

    <rules>
    - Do NOT edit any files. Only read and search.
    - Do NOT call done. Another phase will handle the fix.
    - Be efficient: batch file reads, use targeted grep patterns.
    - Output concise observations about what you find.
    </rules>

    You have {step_limit} exploration calls. Use them to gather maximum context.
""")

# Minimal prompts for baseline strategy (no explore phase, no baseline rules)
SYSTEM_PROMPT_REASONING_MINIMAL = textwrap.dedent("""\
    You are an expert software engineer. Solve the coding task in the problem
    statement correctly and efficiently.

    You have a LIMITED budget of {step_limit} shell calls. Each call costs real
    money. A good solution typically needs 8-12 calls.

    <grading>
    After you submit, a rigorous hidden test patch will be applied that adds
    targeted tests for every requirement -- including edge cases. Your code
    must handle all of these correctly.

    Write code as if it is safety-critical. Every new function must be hardened
    against: null/undefined/falsy inputs, empty arrays, missing object keys,
    wrong types, and must preserve exact ordering and return types.
    </grading>

    <editing>
    When using Python string .replace() to edit files, the old string must
    match EXACTLY or the replacement silently does nothing. After every edit,
    VERIFY it landed by piping to grep or diff.
    Make MINIMAL changes. Change only the lines you need.
    </editing>

    <efficiency>
    Minimize calls by batching work:
    - Batch file reads: sed + printf separators in one call
    - Batch edits: python heredoc + grep verify
    - Run the full test suite at most 1-2 times total
    </efficiency>

    <validation>
    Before calling done, review your changes:
    - Did you address ALL requirements in the problem statement?
    - For every new function: does it handle null, undefined, [], falsy inputs?
    - Did your test run pass?
    - Are your edits minimal?
    </validation>

    <rules>
    - Read a file before modifying it
    - Do NOT modify test files unless required
    - After EVERY file edit, verify the change landed
    - Call done when finished
    </rules>
""")

SYSTEM_PROMPT_CLASSIC_MINIMAL = textwrap.dedent("""\
    You are an expert software engineer. Fix a bug or implement a feature
    described in the problem statement.

    <critical_rules>
    1. Read a file BEFORE modifying it.
    2. After EVERY edit, verify it landed: grep -n 'expected_text' file
    3. Do NOT modify test files unless the problem explicitly requires it.
    4. Make MINIMAL changes.
    5. Call done when finished.
    </critical_rules>

    <workflow>
    Step 1 - UNDERSTAND: Read the problem carefully. Identify relevant files.
    Step 2 - EXPLORE: Find and read the relevant files (batch reads).
    Step 3 - DIAGNOSE: Reason about root cause and edge cases.
    Step 4 - EDIT: Make minimal code changes.
    Step 5 - VERIFY: Confirm edits landed.
    Step 6 - TEST: Run the test suite (at most 1-2 times).
    Step 7 - FIX: If tests fail, diagnose and fix. Repeat until pass.
    Step 8 - Call done.
    </workflow>

    <grading>
    After you submit, hidden tests will cover all requirements including edge
    cases. Every function must handle null, undefined, empty arrays, missing
    keys, wrong types. Array ordering must match input order.
    </grading>

    You have {step_limit} tool calls. A good solution needs 8-15.
""")


def _get_system_prompt(model_name: str, *, use_baseline_rules: bool = True) -> str:
    if _is_reasoning_model(model_name):
        return SYSTEM_PROMPT_REASONING if use_baseline_rules else SYSTEM_PROMPT_REASONING_MINIMAL
    return SYSTEM_PROMPT_CLASSIC if use_baseline_rules else SYSTEM_PROMPT_CLASSIC_MINIMAL


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

SHELL_TOOL: dict = {
    "type": "shell",
    "environment": {"type": "local"},
}

RUN_COMMAND_TOOL: dict = {
    "type": "function",
    "name": "run_command",
    "description": (
        "Execute a shell command in the repository directory. "
        "Returns stdout, stderr, and exit code."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute (bash).",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    "strict": True,
}

DONE_TOOL: dict = {
    "type": "function",
    "name": "done",
    "description": "Signal that the fix is complete.",
    "parameters": {
        "type": "object",
        "properties": {
            "explanation": {
                "type": "string",
                "description": "Brief summary of what was changed and why.",
            },
        },
        "required": ["explanation"],
        "additionalProperties": False,
    },
    "strict": True,
}


def _get_tools(model_name: str) -> list[dict]:
    if _is_reasoning_model(model_name):
        return [SHELL_TOOL, DONE_TOOL]
    return [RUN_COMMAND_TOOL, DONE_TOOL]


# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------

@dataclass
class SolveStrategy:
    """Configuration for a SWE-bench solving strategy.

    Swap strategies by passing different instances to solve_instance().
    """
    name: str
    step_limit: int = 50
    qa_budget: int = 5
    compact_threshold: int = 200_000
    explore_steps: int = 0          # 0 = no explore phase
    auto_verify: bool = False
    baseline_reminders: bool = False
    qa_start_with_mid: bool = False
    system_prompt_strong: str = ""   # "" = auto-select from model name
    system_prompt_explore: str = ""
    system_prompt_qa: str = ""


# Named strategy presets

STRATEGY_BASELINE = SolveStrategy(
    name="baseline",
    step_limit=50,
    qa_budget=5,
    compact_threshold=200_000,
    explore_steps=0,
    auto_verify=False,
    baseline_reminders=False,
    qa_start_with_mid=False,
)

STRATEGY_THREE_TIER = SolveStrategy(
    name="three-tier",
    step_limit=40,
    qa_budget=5,
    compact_threshold=150_000,
    explore_steps=10,
    auto_verify=True,
    baseline_reminders=True,
    qa_start_with_mid=True,
)

# Default strategy used when none is specified
DEFAULT_STRATEGY = STRATEGY_THREE_TIER

# Registry for lookup by name (e.g., from env var)
STRATEGIES: dict[str, SolveStrategy] = {
    s.name: s for s in [STRATEGY_BASELINE, STRATEGY_THREE_TIER]
}


def get_strategy(name: str | None = None) -> SolveStrategy:
    """Look up a strategy by name, falling back to DEFAULT_STRATEGY."""
    if name and name in STRATEGIES:
        return STRATEGIES[name]
    return DEFAULT_STRATEGY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return text[:head] + f"\n... (truncated, {len(text)} total chars) ...\n" + text[-tail:]


def _start_required_services(runner: DockerRunner) -> list[str]:
    """Detect and start background services the test suite needs."""
    started: list[str] = []

    # Redis
    for cfg in ("config.json", "docker-compose.yml", "docker-compose.yaml", "package.json"):
        r = runner.run(f"grep -qi redis {cfg} 2>/dev/null && echo yes")
        if "yes" in r.output:
            r2 = runner.run("which redis-server 2>/dev/null")
            if r2.exit_code == 0:
                r3 = runner.run("redis-cli ping 2>/dev/null")
                if r3.exit_code != 0 or "PONG" not in r3.output:
                    runner.run("redis-server --daemonize yes --protected-mode no --appendonly yes")
                    runner.run("for i in 1 2 3 4 5; do redis-cli ping 2>/dev/null | grep -q PONG && break; done")
                    started.append("Started redis-server")
            break

    # MongoDB
    for cfg in ("config.json", "docker-compose.yml", "docker-compose.yaml"):
        r = runner.run(f"grep -qi mongo {cfg} 2>/dev/null && echo yes")
        if "yes" in r.output:
            r2 = runner.run("which mongod 2>/dev/null")
            if r2.exit_code == 0:
                r3 = runner.run("pgrep mongod")
                if r3.exit_code != 0:
                    runner.run("mongod --fork --logpath /tmp/mongod.log --dbpath /data/db 2>/dev/null || mkdir -p /data/db && mongod --fork --logpath /tmp/mongod.log --dbpath /data/db")
                    started.append("Started mongod")
            break

    # PostgreSQL
    for cfg in ("config.json", "docker-compose.yml", "docker-compose.yaml"):
        r = runner.run(f"grep -qi postgres {cfg} 2>/dev/null && echo yes")
        if "yes" in r.output:
            r2 = runner.run("which pg_isready 2>/dev/null")
            if r2.exit_code == 0:
                r3 = runner.run("pg_isready")
                if r3.exit_code != 0:
                    runner.run("su - postgres -c 'pg_ctl start -D /var/lib/postgresql/data -l /tmp/pg.log' 2>/dev/null || pg_ctlcluster 14 main start 2>/dev/null")
                    started.append("Started PostgreSQL")
            break

    return started


def _discover_test_command(runner: DockerRunner) -> str | None:
    """Probe the container to find a working test command."""
    # Node.js
    r = runner.run("cat package.json 2>/dev/null")
    if r.exit_code == 0 and r.output.strip():
        try:
            pkg = json.loads(r.output)
            scripts = pkg.get("scripts", {})
            if "test" in scripts:
                r2 = runner.run("which npm 2>/dev/null")
                if r2.exit_code == 0:
                    return "npm test"
        except json.JSONDecodeError:
            pass

    # Ansible
    r = runner.run("test -d lib/ansible && test -d test/units && echo found")
    if r.exit_code == 0 and "found" in r.output:
        r2 = runner.run("python -m pytest --version 2>/dev/null")
        if r2.exit_code == 0:
            return "python -m pytest test/units/ --tb=short -q"

    # Python pytest/unittest
    r = runner.run("test -f pytest.ini -o -f setup.cfg -o -f pyproject.toml && echo found")
    if r.exit_code == 0 and "found" in r.output:
        r2 = runner.run("python -m pytest --version 2>/dev/null")
        if r2.exit_code == 0:
            return "python -m pytest --tb=short -q"
        r2 = runner.run("python -m unittest discover --help 2>/dev/null")
        if r2.exit_code == 0:
            return "python -m unittest discover -s tests"

    # Go
    r = runner.run("test -f go.mod && echo found")
    if r.exit_code == 0 and "found" in r.output:
        return "go test ./..."

    # Makefile
    r = runner.run("grep -q '^test:' Makefile 2>/dev/null && echo found")
    if r.exit_code == 0 and "found" in r.output:
        return "make test"

    # Rust
    r = runner.run("test -f Cargo.toml && echo found")
    if r.exit_code == 0 and "found" in r.output:
        return "cargo test"

    return None


def _extract_test_failures(output: str) -> str:
    """Extract a focused failure summary from test runner output."""
    lines = output.splitlines()
    failures: list[str] = []
    passing = 0
    failing = 0

    in_failure_block = False
    current_failure: list[str] = []
    for line in lines:
        stripped = line.strip()
        if "passing" in stripped and not stripped.startswith("#"):
            try:
                passing = int(stripped.split()[0])
            except (ValueError, IndexError):
                pass
        if "failing" in stripped and not stripped.startswith("#"):
            try:
                failing = int(stripped.split()[0])
            except (ValueError, IndexError):
                pass
        if re.match(r"^\s+\d+\)", line):
            if current_failure:
                failures.append("\n".join(current_failure))
            current_failure = [stripped]
            in_failure_block = True
        elif in_failure_block:
            if stripped.startswith("at ") or stripped.startswith("Error:"):
                current_failure.append("  " + stripped)
            elif stripped == "" or re.match(r"^\s+\d+\)", stripped):
                if current_failure:
                    failures.append("\n".join(current_failure))
                    current_failure = []
                in_failure_block = stripped != ""

    if current_failure:
        failures.append("\n".join(current_failure))

    for line in lines:
        if "FAILED" in line and "::" in line:
            failures.append(line.strip())

    if not failures and not failing:
        return output

    parts = [f"=== TEST SUMMARY: {passing} passing, {failing} failing ==="]
    if failures:
        parts.append("FAILING TESTS:")
        for f in failures[:20]:
            parts.append(f)
    parts.append("=== END TEST SUMMARY ===")
    summary = "\n".join(parts)

    remaining = TEST_FAILURE_EXTRACT_LIMIT - len(summary)
    if remaining > 200:
        summary += "\n" + output[:remaining]
    return summary


def _extract_failure_ids(output: str) -> set[str]:
    """Extract failure identifiers from test runner output."""
    ids: set[str] = set()
    for line in output.splitlines():
        stripped = line.strip()

        # pytest
        if stripped.startswith("FAILED "):
            ids.add(stripped.split(" - ")[0].strip())
        elif stripped.startswith("RERUN ") and "::" in stripped:
            ids.add("FAILED " + stripped[6:].split(" - ")[0].strip())
        elif stripped.startswith("ERROR ") and not stripped.startswith("ERROR!"):
            ids.add(stripped)

        # go test
        elif stripped.startswith("--- FAIL:"):
            ids.add(stripped.split("(")[0].strip())
        elif stripped.startswith("FAIL:") and ".go:" in stripped:
            ids.add(stripped)
        elif stripped.startswith("FAIL\t") or stripped.startswith("FAIL    "):
            pkg = stripped.split()[1] if len(stripped.split()) > 1 else stripped
            ids.add(f"FAIL {pkg}")

        # mocha/npm
        elif re.match(r"^\d+\)", stripped):
            ids.add(stripped.rstrip(":"))

        # Infrastructure errors
        elif re.match(r"^sh: \d+: .+: not found", stripped):
            ids.add(stripped)
        elif "Fatal Python error" in stripped:
            ids.add("Fatal Python error")
        elif "command timed out" in stripped:
            ids.add(stripped)

    return ids


def _gate_passes_with_baseline(
    gate_exit_code: int,
    gate_output: str,
    baseline_exit_code: int | None,
    baseline_output: str,
) -> tuple[bool, set[str], set[str]]:
    """Check if test gate passes after filtering pre-existing failures."""
    baseline_ids: set[str] = set()
    gate_ids = _extract_failure_ids(gate_output)

    if gate_exit_code == 0:
        return True, set(), baseline_ids

    if baseline_exit_code is None or baseline_exit_code == 0:
        return False, gate_ids, baseline_ids

    baseline_ids = _extract_failure_ids(baseline_output)
    new_failures = gate_ids - baseline_ids

    if not gate_ids and not baseline_ids:
        return True, set(), set()

    passes = len(new_failures) == 0 and len(gate_ids) > 0
    return passes, new_failures, baseline_ids


def _execute_shell(runner: DockerRunner, commands: list[str]) -> list[dict]:
    """Execute shell commands inside the Docker container."""
    results = []
    for command in commands:
        try:
            r = runner.run(command, timeout=COMMAND_TIMEOUT)
            stdout = r.output or ""
            stderr = ""
            exit_code = r.exit_code

            # Auto-recover crashed services
            if exit_code != 0 and ("ECONNREFUSED" in stdout or "Connection refused" in stdout):
                if "6379" in stdout or "redis" in stdout.lower():
                    runner.run("redis-server --daemonize yes --protected-mode no --appendonly yes")
                    r = runner.run(command, timeout=COMMAND_TIMEOUT)
                    stdout, exit_code = r.output or "", r.exit_code
                elif "27017" in stdout or "mongo" in stdout.lower():
                    runner.run("mongod --fork --logpath /tmp/mongod.log --dbpath /data/db 2>/dev/null || true")
                    r = runner.run(command, timeout=COMMAND_TIMEOUT)
                    stdout, exit_code = r.output or "", r.exit_code

            results.append({
                "stdout": _truncate(stdout, TOOL_RESULT_LIMIT),
                "stderr": _truncate(stderr, TOOL_RESULT_LIMIT),
                "outcome": {"type": "exit", "exit_code": exit_code},
            })
        except Exception as exc:
            results.append({
                "stdout": "",
                "stderr": f"Error: {exc}",
                "outcome": {"type": "timeout"},
            })
    return results


# ---------------------------------------------------------------------------
# Conversation logger
# ---------------------------------------------------------------------------

class ConversationLogger:
    def __init__(self, instance_id: str):
        run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_id = re.sub(r"[^\w.-]", "_", instance_id)[:80]
        self._dir = LOG_DIR / run_ts
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{safe_id}.jsonl"
        self._fh = self._path.open("a", encoding="utf-8")

    def log(self, event: str, **data) -> None:
        entry = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(), "event": event, **data}
        self._fh.write(json.dumps(entry, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    @property
    def path(self) -> Path:
        return self._path


# ---------------------------------------------------------------------------
# Post-step hooks (injected messages based on strategy flags)
# ---------------------------------------------------------------------------

_BASELINE_REMINDER = (
    "[REMINDER] The test suite has PRE-EXISTING failures "
    "that are NOT caused by your changes. The baseline test "
    "output is saved at .swe_baseline_test_output.txt -- you "
    "can diff against it. Do NOT try to fix unrelated test "
    "failures. If only baseline failures remain, call done."
)


def _inject_auto_verify(
    items: list,
    commands: list[str],
    runner: DockerRunner,
    loop: asyncio.AbstractEventLoop,
    clog: ConversationLogger,
    step: int,
) -> None:
    """Inject auto-verify messages for edit commands."""
    for cmd in commands:
        if _needs_auto_verify(cmd):
            verify_cmd = _make_verify_command(cmd)
            if verify_cmd:
                vr = loop.run_until_complete(
                    loop.run_in_executor(
                        None, lambda vc=verify_cmd: runner.run(vc, timeout=30),
                    )
                )
                verify_text = _truncate(vr.output or "(no output)", 2000)
                items.append({
                    "role": "user",
                    "content": f"[auto-verify] $ {verify_cmd}\n{verify_text}",
                })
                clog.log("auto_verify", step=step, cmd=verify_cmd,
                         output=verify_text[:500])


async def _inject_auto_verify_async(
    items: list,
    commands: list[str],
    runner: DockerRunner,
    loop: asyncio.AbstractEventLoop,
    clog: ConversationLogger,
    step: int,
) -> None:
    """Inject auto-verify messages for edit commands (async version)."""
    for cmd in commands:
        if _needs_auto_verify(cmd):
            verify_cmd = _make_verify_command(cmd)
            if verify_cmd:
                vr = await loop.run_in_executor(
                    None, lambda vc=verify_cmd: runner.run(vc, timeout=30),
                )
                verify_text = _truncate(vr.output or "(no output)", 2000)
                items.append({
                    "role": "user",
                    "content": f"[auto-verify] $ {verify_cmd}\n{verify_text}",
                })
                clog.log("auto_verify", step=step, cmd=verify_cmd,
                         output=verify_text[:500])


def _inject_baseline_reminder(
    items: list,
    commands: list[str],
    test_cmd: str | None,
    baseline_failures: str,
) -> None:
    """Inject baseline failure reminder if a test command was detected."""
    if not baseline_failures or not test_cmd:
        return
    test_keyword = test_cmd.split()[0]
    for cmd in commands:
        if test_keyword in cmd:
            items.append({"role": "user", "content": _BASELINE_REMINDER})
            return


def _inject_baseline_reminder_fc(
    items: list,
    function_calls: list,
    test_cmd: str | None,
    baseline_failures: str,
) -> None:
    """Inject baseline failure reminder after function_call test runs."""
    if not baseline_failures or not test_cmd:
        return
    test_keyword = test_cmd.split()[0]
    for fc in function_calls:
        if fc.name == "run_command":
            try:
                cmd_text = json.loads(fc.arguments).get("command", "")
            except (json.JSONDecodeError, AttributeError):
                cmd_text = ""
            if test_keyword and test_keyword in cmd_text:
                items.append({"role": "user", "content": _BASELINE_REMINDER})
                return


# ---------------------------------------------------------------------------
# Core solve logic
# ---------------------------------------------------------------------------

async def solve_instance(
    instance: dict,
    client: AsyncOpenAI,
    model: str,
    mid_model: str = "",
    on_status: Callable | None = None,
    strategy: SolveStrategy | None = None,
) -> str:
    """Solve a SWE-bench instance using a flat agent loop.

    The *strategy* parameter controls model routing, step budgets, prompts,
    auto-verify, and baseline reinforcement. Defaults to DEFAULT_STRATEGY.

    Returns the final git diff.
    """
    strat = strategy or DEFAULT_STRATEGY

    instance_id = instance["instance_id"]
    problem = instance["problem_statement"]
    image_uri = instance["docker_image"]
    base_commit = instance["base_commit"]
    repo = instance.get("repo", "")
    hints = instance.get("hints", "")
    is_reasoning = _is_reasoning_model(model)
    mid_model = mid_model or model
    mid_is_reasoning = _is_reasoning_model(mid_model)

    async def status(msg: str) -> None:
        if on_status:
            await on_status(msg)
        logger.info("[%s] %s", instance_id, msg)

    await status(f"Strategy: {strat.name} (limit={strat.step_limit}, explore={strat.explore_steps})")
    await status("Pulling image and starting container...")
    loop = asyncio.get_event_loop()
    runner = DockerRunner(image_uri, working_dir="/app")
    clog = ConversationLogger(instance_id)

    try:
        await loop.run_in_executor(None, runner.start)
        logger.info("[%s] Container started", instance_id)

        # Reset to base commit
        runner.run(f"git checkout {base_commit} && git reset --hard {base_commit}")

        # Start required services
        svc_msgs = await loop.run_in_executor(None, lambda: _start_required_services(runner))
        for svc_msg in svc_msgs:
            await status(svc_msg)

        # Get repo overview
        await status("Exploring repository structure...")
        tree = await loop.run_in_executor(None, lambda: runner.list_files(".", max_depth=2))
        tree = _truncate(tree, 10_000)

        # Discover test commands
        test_cmd = await loop.run_in_executor(None, lambda: _discover_test_command(runner))
        if test_cmd:
            await status(f"Detected test command: {test_cmd}")

        # Run baseline tests
        baseline_failures = ""
        baseline_exit_code: int | None = None
        baseline_output_raw = ""
        if test_cmd:
            await status("Running baseline tests...")
            baseline_result = await loop.run_in_executor(
                None, lambda: runner.run(test_cmd, timeout=COMMAND_TIMEOUT)
            )
            baseline_exit_code = baseline_result.exit_code
            baseline_output_raw = baseline_result.output or ""
            if baseline_result.exit_code != 0:
                baseline_failures = _extract_test_failures(baseline_result.output)
                await loop.run_in_executor(
                    None,
                    lambda: runner.write_file(
                        ".swe_baseline_test_output.txt",
                        baseline_output_raw[:50_000],
                    ),
                )
                await status(f"Baseline tests found failures (exit code {baseline_result.exit_code})")
            else:
                await status("Baseline tests all pass")

        # Build the initial user message
        user_content = f"## Repository: {repo}\n\n"
        user_content += f"## File listing (depth 2):\n```\n{tree}\n```\n\n"
        user_content += f"## Problem statement:\n{_truncate(problem, 8000)}\n"
        if hints:
            user_content += f"\n## Hints:\n{_truncate(hints, 2000)}\n"
        if test_cmd:
            user_content += f"\n## Test command:\n```\n{test_cmd}\n```\n"
        if baseline_failures:
            user_content += (
                f"\n## Baseline test failures (before any changes):\n"
                f"```\n{_truncate(baseline_failures, 4000)}\n```\n"
                f"\nThese failures are pre-existing. Do NOT attempt to fix them.\n"
            )
        user_content += (
            "\nSolve this problem. Read relevant files, understand the root cause, "
            "make the necessary code changes, and verify with tests."
        )

        # Resolve prompts from strategy or auto-select from model
        use_baseline_rules = strat.baseline_reminders
        system_prompt = (
            strat.system_prompt_strong
            or _get_system_prompt(model, use_baseline_rules=use_baseline_rules)
        ).format(step_limit=strat.step_limit)
        explore_prompt = (
            strat.system_prompt_explore
            or SYSTEM_PROMPT_EXPLORE
        ).format(step_limit=strat.explore_steps)

        tools = _get_tools(model)
        mid_tools = _get_tools(mid_model)
        explore_tools = [t for t in mid_tools if t.get("name") != "done"]
        items: list = [{"role": "user", "content": user_content}]
        clog.log("system", content=system_prompt)
        clog.log("user", content=user_content)

        # -----------------------------------------------------------
        # FLAT LOOP
        # -----------------------------------------------------------
        done_signalled = False
        qa_gate_failed = False
        qa_steps_used = 0
        total_steps = 0

        for step in range(strat.step_limit):
            # Model routing: use mid model for explore phase
            use_mid = (strat.explore_steps > 0
                       and step < strat.explore_steps
                       and mid_model != model)
            step_model = mid_model if use_mid else model
            step_is_reasoning = mid_is_reasoning if use_mid else is_reasoning
            step_tools = explore_tools if use_mid else tools
            step_system = explore_prompt if use_mid else system_prompt

            # Phase transition message
            if (strat.explore_steps > 0
                    and step == strat.explore_steps
                    and mid_model != model):
                remaining = strat.step_limit - step
                transition_msg = (
                    f"## Exploration phase complete\n\n"
                    f"The exploration phase is over. You now have access to the `done` tool.\n"
                    f"Focus on: (1) diagnose root cause, (2) implement a minimal fix, "
                    f"(3) verify edits landed, (4) run tests, (5) call done.\n"
                    f"Remaining budget: {remaining} steps."
                )
                if strat.baseline_reminders and baseline_failures:
                    transition_msg += (
                        f"\n\n## IMPORTANT: Pre-existing test failures\n"
                        f"The test suite has pre-existing failures that are NOT your problem.\n"
                        f"If tests fail with only these baseline failures, call done immediately.\n"
                        f"Only fix NEW failures caused by your changes."
                    )
                items.append({"role": "user", "content": transition_msg})
                await status(f"Phase transition: explore -> fix ({remaining} steps remaining)")

            tier_label = "mid" if use_mid else "strong"
            await status(f"Step {step + 1} ({tier_label}: {step_model})")

            api_kwargs: dict = {
                "model": step_model,
                "instructions": step_system,
                "input": items,
                "tools": step_tools,
                "parallel_tool_calls": False,
                "store": False,
            }
            if step_is_reasoning:
                api_kwargs["include"] = ["reasoning.encrypted_content"]
                api_kwargs["context_management"] = [
                    {"type": "compaction", "compact_threshold": strat.compact_threshold},
                ]
                api_kwargs["reasoning"] = {"effort": "high", "summary": "auto"}
                api_kwargs["max_output_tokens"] = 16_000
            else:
                api_kwargs["temperature"] = 0.0
                api_kwargs["max_output_tokens"] = 4096
            if not os.getenv("AZURE_OPENAI_ENDPOINT", "").strip():
                api_kwargs["service_tier"] = "priority"

            try:
                response = await client.responses.create(**api_kwargs)
            except Exception as exc:
                logger.warning("[%s] API error at step %d: %s", instance_id, step, exc)
                clog.log("api_error", step=step, error=str(exc)[:500])
                break

            tracker.record(response, label=f"swe_bench step={step+1} tier={tier_label}")

            # Append output and handle compaction
            items.extend(response.output)
            last_compaction_idx = None
            for i, it in enumerate(items):
                if hasattr(it, "type") and it.type == "compaction":
                    last_compaction_idx = i
            if last_compaction_idx is not None and last_compaction_idx > 0:
                items = items[last_compaction_idx:]

            # Parse response
            shell_calls = []
            function_calls = []
            for item in response.output:
                if item.type == "shell_call":
                    shell_calls.append(item)
                elif item.type == "function_call":
                    function_calls.append(item)

            has_tool_calls = bool(shell_calls or function_calls)
            if not has_tool_calls:
                break

            # Process shell_call items (reasoning models)
            for sc in shell_calls:
                commands = []
                if hasattr(sc, "action") and sc.action:
                    commands = sc.action.commands if hasattr(sc.action, "commands") else []
                if not commands:
                    commands = ["echo '(no command)'"]

                preview = commands[0][:80] if commands else ""
                await status(f"[{step + 1}] $ {preview}")

                results = await loop.run_in_executor(
                    None, lambda cmds=commands: _execute_shell(runner, cmds),
                )
                max_output_length = TOOL_RESULT_LIMIT
                if hasattr(sc, "action") and hasattr(sc.action, "max_output_length") and sc.action.max_output_length:
                    max_output_length = sc.action.max_output_length

                clog.log("tool", step=step, tool="shell", commands=commands,
                         result=_truncate(str(results), 20_000))
                items.append({
                    "type": "shell_call_output",
                    "call_id": sc.call_id,
                    "output": results,
                    "max_output_length": max_output_length,
                })

                if strat.auto_verify:
                    await _inject_auto_verify_async(
                        items, commands, runner, loop, clog, step,
                    )

                if strat.baseline_reminders:
                    _inject_baseline_reminder(
                        items, commands, test_cmd, baseline_failures,
                    )

            # Process function_call items (classic models + done tool)
            for fc in function_calls:
                name = fc.name
                try:
                    args = json.loads(fc.arguments)
                except json.JSONDecodeError:
                    args = {}

                if name == "done":
                    diff_so_far = await loop.run_in_executor(None, runner.get_diff)
                    if not diff_so_far.strip():
                        qa_gate_failed = True
                        result = (
                            "DONE REJECTED -- no changes detected (git diff is empty). "
                            "Make the required edits before calling done."
                        )
                        await status("Empty patch -- rejecting done")
                    elif test_cmd:
                        await status(f"Running test gate: {test_cmd}")
                        gate_result = await loop.run_in_executor(
                            None, lambda: runner.run(test_cmd, timeout=COMMAND_TIMEOUT),
                        )
                        gate_passed, new_fails, _ = _gate_passes_with_baseline(
                            gate_result.exit_code, gate_result.output or "",
                            baseline_exit_code, baseline_output_raw,
                        )
                        clog.log("test_gate", passed=gate_passed,
                                 exit_code=gate_result.exit_code,
                                 new_failures=sorted(new_fails)[:10])
                        if not gate_passed:
                            qa_gate_failed = True
                            fail_output = _truncate(gate_result.output or "", TEST_FAILURE_EXTRACT_LIMIT)
                            new_fail_hint = ""
                            if new_fails:
                                new_fail_hint = (
                                    "\nNEW failures from your changes:\n"
                                    + "\n".join(sorted(new_fails)[:10]) + "\n"
                                )
                            result = (
                                f"DONE REJECTED -- tests still fail.\n"
                                f"Test command: {test_cmd}\n{new_fail_hint}"
                                f"Test output:\n{fail_output}"
                            )
                            await status("Test gate FAILED -- rejecting done")
                        else:
                            await status("Test gate passed -- accepting done")
                            result = "Done acknowledged. Tests pass."
                            done_signalled = True
                    else:
                        await status(f"Done: {args.get('explanation', '')[:100]}")
                        result = "Done acknowledged."
                        done_signalled = True

                elif name == "run_command":
                    command = args.get("command", "echo '(no command)'")
                    preview = command[:80]
                    await status(f"[{step + 1}] $ {preview}")

                    cmd_results = await loop.run_in_executor(
                        None, lambda cmd=command: _execute_shell(runner, [cmd]),
                    )
                    r = cmd_results[0]
                    exit_code = r["outcome"].get("exit_code", 0)
                    output = r["stdout"] or ""
                    if r["stderr"]:
                        output += "\n" + r["stderr"]
                    if exit_code != 0:
                        output = f"[exit code {exit_code}]\n{output}"
                    result = _truncate(output.strip() or "(no output)", TOOL_RESULT_LIMIT)
                else:
                    result = f"Unknown tool: {name}"

                clog.log("tool", step=step, tool=name, args=args,
                         result=_truncate(result, 20_000))
                items.append({
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": result,
                })

                if strat.auto_verify and name == "run_command" and not done_signalled:
                    cmd_text = args.get("command", "")
                    await _inject_auto_verify_async(
                        items, [cmd_text], runner, loop, clog, step,
                    )

            if done_signalled:
                break

            if strat.baseline_reminders and not done_signalled:
                _inject_baseline_reminder_fc(
                    items, function_calls, test_cmd, baseline_failures,
                )

            items.append({"role": "user", "content": f"[Turn {step + 1}/{strat.step_limit}]"})

        total_steps = step + 1

        # -----------------------------------------------------------
        # QA FIX PHASE
        # -----------------------------------------------------------
        if qa_gate_failed and not done_signalled:
            clog.log("qa_phase_start")
            items.append({
                "role": "user",
                "content": (
                    f"## REMINDER -- Original problem statement\n"
                    f"You have used {total_steps} steps. The test gate rejected your patch.\n"
                    f"Re-read the requirements carefully.\n\n"
                    f"{_truncate(problem, 6000)}"
                ),
            })

            qa_stale_count = 0
            qa_last_diff: str | None = None
            qa_escalated = False

            qa_sys_prompt = (
                strat.system_prompt_qa
                or SYSTEM_PROMPT_CLASSIC
            ).format(step_limit=strat.qa_budget)

            for qa_step in range(strat.qa_budget):
                qa_steps_used = qa_step + 1
                total_steps += 1

                # QA model routing: start with mid if strategy says so
                qa_use_mid = strat.qa_start_with_mid and (mid_model != model) and not qa_escalated
                qa_model = mid_model if qa_use_mid else model
                qa_model_reasoning = mid_is_reasoning if qa_use_mid else is_reasoning
                qa_tier = "mid" if qa_use_mid else "strong"
                qa_sys = qa_sys_prompt if qa_use_mid else system_prompt
                qa_tls = mid_tools if qa_use_mid else tools

                await status(f"QA fix step {qa_steps_used}/{strat.qa_budget} ({qa_tier}: {qa_model})")

                api_kwargs_qa: dict = {
                    "model": qa_model,
                    "instructions": qa_sys,
                    "input": items,
                    "tools": qa_tls,
                    "parallel_tool_calls": False,
                    "store": False,
                }
                if qa_model_reasoning:
                    api_kwargs_qa["include"] = ["reasoning.encrypted_content"]
                    api_kwargs_qa["context_management"] = [
                        {"type": "compaction", "compact_threshold": strat.compact_threshold},
                    ]
                    api_kwargs_qa["reasoning"] = {"effort": "high", "summary": "auto"}
                    api_kwargs_qa["max_output_tokens"] = 16_000
                else:
                    api_kwargs_qa["temperature"] = 0.0
                    api_kwargs_qa["max_output_tokens"] = 4096
                if not os.getenv("AZURE_OPENAI_ENDPOINT", "").strip():
                    api_kwargs_qa["service_tier"] = "priority"

                response = await client.responses.create(**api_kwargs_qa)
                tracker.record(response, label=f"swe_bench qa_step={qa_steps_used} tier={qa_tier}")
                items.extend(response.output)

                # Compaction
                last_compaction_idx = None
                for i, it in enumerate(items):
                    if hasattr(it, "type") and it.type == "compaction":
                        last_compaction_idx = i
                if last_compaction_idx is not None and last_compaction_idx > 0:
                    items = items[last_compaction_idx:]

                qa_shell_calls = []
                qa_function_calls = []
                for item in response.output:
                    if item.type == "shell_call":
                        qa_shell_calls.append(item)
                    elif item.type == "function_call":
                        qa_function_calls.append(item)

                if not qa_shell_calls and not qa_function_calls:
                    break

                # Process shell calls
                for sc in qa_shell_calls:
                    commands = []
                    if hasattr(sc, "action") and sc.action:
                        commands = sc.action.commands if hasattr(sc.action, "commands") else []
                    if not commands:
                        commands = ["echo '(no command)'"]
                    results = await loop.run_in_executor(
                        None, lambda cmds=commands: _execute_shell(runner, cmds),
                    )
                    max_output_length = TOOL_RESULT_LIMIT
                    if hasattr(sc, "action") and hasattr(sc.action, "max_output_length") and sc.action.max_output_length:
                        max_output_length = sc.action.max_output_length
                    items.append({
                        "type": "shell_call_output",
                        "call_id": sc.call_id,
                        "output": results,
                        "max_output_length": max_output_length,
                    })

                # Process function calls
                qa_done = False
                for fc in qa_function_calls:
                    fc_name = fc.name
                    try:
                        fc_args = json.loads(fc.arguments)
                    except json.JSONDecodeError:
                        fc_args = {}

                    if fc_name == "done":
                        diff_so_far = await loop.run_in_executor(None, runner.get_diff)
                        if not diff_so_far.strip():
                            fc_result = "DONE REJECTED -- empty patch."
                        elif test_cmd:
                            await status(f"QA done -- re-running test gate")
                            gate_result = await loop.run_in_executor(
                                None, lambda: runner.run(test_cmd, timeout=COMMAND_TIMEOUT),
                            )
                            gate_passed, new_fails, _ = _gate_passes_with_baseline(
                                gate_result.exit_code, gate_result.output or "",
                                baseline_exit_code, baseline_output_raw,
                            )
                            if gate_passed:
                                await status("QA test gate passed")
                                fc_result = "Done acknowledged. Tests pass."
                                done_signalled = True
                                qa_done = True
                            else:
                                remaining = strat.qa_budget - qa_steps_used
                                fail_output = _truncate(gate_result.output or "", TEST_FAILURE_EXTRACT_LIMIT)
                                fc_result = (
                                    f"DONE REJECTED -- tests still fail. {remaining} QA steps remaining.\n"
                                    f"Test output:\n{fail_output}"
                                )
                                await status(f"QA gate still failing -- {remaining} steps left")
                                if qa_use_mid and not qa_escalated:
                                    qa_escalated = True
                                    await status(f"Escalating QA to strong model ({model})")
                                current_diff = diff_so_far.strip()
                                if qa_last_diff is not None and current_diff == qa_last_diff:
                                    qa_stale_count += 1
                                else:
                                    qa_stale_count = 1
                                qa_last_diff = current_diff
                                if qa_stale_count >= QA_STALE_CAP:
                                    await status("QA abort -- patch unchanged across rejections")
                                    qa_done = True
                        else:
                            fc_result = "Done acknowledged."
                            done_signalled = True
                            qa_done = True

                    elif fc_name == "run_command":
                        command = fc_args.get("command", "echo '(no command)'")
                        cmd_results = await loop.run_in_executor(
                            None, lambda cmd=command: _execute_shell(runner, [cmd]),
                        )
                        r = cmd_results[0]
                        output = r["stdout"] or ""
                        if r["stderr"]:
                            output += "\n" + r["stderr"]
                        if r["outcome"].get("exit_code", 0) != 0:
                            output = f"[exit code {r['outcome']['exit_code']}]\n{output}"
                        fc_result = _truncate(output.strip() or "(no output)", TOOL_RESULT_LIMIT)
                    else:
                        fc_result = f"Unknown tool: {fc_name}"

                    items.append({
                        "type": "function_call_output",
                        "call_id": fc.call_id,
                        "output": fc_result,
                    })

                if qa_done:
                    break

                items.append({"role": "user", "content": f"[QA step {qa_steps_used}/{strat.qa_budget}]"})

        # Collect the final diff
        try:
            diff = await loop.run_in_executor(None, runner.get_diff)
        except Exception as exc:
            logger.error("[%s] get_diff failed: %s", instance_id, exc)
            diff = ""

        if diff.strip():
            await status(f"Generated diff ({len(diff)} chars, {total_steps} steps)")
        else:
            await status(f"No diff produced ({total_steps} steps)")

        clog.log("result", diff_len=len(diff.strip()), steps=total_steps,
                 qa_steps=qa_steps_used, done_signalled=done_signalled,
                 strategy=strat.name)
        logger.info("[%s] SOLVE COMPLETE: steps=%d diff_len=%d strategy=%s",
                    instance_id, total_steps, len(diff.strip()), strat.name)
        return diff.lstrip()

    finally:
        clog.close()
        await loop.run_in_executor(None, runner.stop)
        await loop.run_in_executor(None, runner.cleanup_image)
