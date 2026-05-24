"""Dual-model plan-execute skill -- frontier planner + cheap executor.

Conserves tokens by using a strong model for research, diagnosis, and task
decomposition (~10 steps) and a cheap model for executing each decomposed
task (~5 steps per task).  The orchestrator iterates deterministically
through the task list; the LLM never decides "what's next."

Reference: other-agents/swe-bench-pro/docs/dual-model-strategy-proposal.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import textwrap
from typing import Callable

from openai import AsyncOpenAI

from usage import tracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLAN_STEP_LIMIT = 10          # frontier model budget for research + plan
EXEC_STEP_LIMIT = 8           # cheap model budget per decomposed task
VERIFY_STEP_LIMIT = 5         # cheap model budget for final verification
ESCALATION_ATTEMPTS = 2       # per-task failures before escalating to frontier
TOOL_RESULT_LIMIT = 30_000
COMMAND_TIMEOUT = 120
COMPACT_THRESHOLD = 200_000

_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _REASONING_MODEL_PREFIXES)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_SHELL_TOOL: dict = {
    "type": "shell",
    "environment": {"type": "local"},
}

_RUN_COMMAND_TOOL: dict = {
    "type": "function",
    "name": "run_command",
    "description": "Execute a shell command. Returns stdout, stderr, and exit code.",
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

_DONE_TOOL: dict = {
    "type": "function",
    "name": "done",
    "description": "Signal that the current phase is complete. Provide your output.",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The output for this phase.",
            },
        },
        "required": ["answer"],
        "additionalProperties": False,
    },
    "strict": True,
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLANNER_PROMPT = textwrap.dedent("""\
    You are an expert analyst and planner. You will receive a task from a
    competition benchmark. Your job is to:

    1. RESEARCH: Use the shell to explore, read files, gather data, and
       understand the problem thoroughly.
    2. PLAN: Decompose the solution into a series of small, deterministic
       tasks that a less capable model can execute one at a time.

    Each decomposed task should be a single focused action -- one file edit,
    one computation, one data extraction.  The executor model will receive
    each task in isolation with no memory of prior tasks, so each task must
    be self-contained.

    When you have finished planning, call `done` with your output in this
    exact format:

    TASKS:
    1. ACTION: <what to do>
       DETAILS: <step-by-step instructions a junior developer could follow>
       VERIFY: <shell command to confirm the action succeeded>

    2. ACTION: <what to do>
       DETAILS: <step-by-step instructions>
       VERIFY: <shell command to confirm>

    FINAL_ANSWER_FORMAT: <describe the exact format the final answer should use>

    If the task is simple enough that decomposition adds no value, output a
    single-task plan.  If you can answer the question directly from your
    research without any execution tasks, use:

    DIRECT_ANSWER: <your answer>

    Be precise. Be efficient. You have {step_limit} shell calls.
""")

EXECUTOR_PROMPT = textwrap.dedent("""\
    You have ONE focused task to complete. Do exactly what is specified.
    Do not deviate, do not add extra work, do not skip steps.

    Use the shell to execute the task. When finished, call `done` with a
    brief summary of what you did and the result.

    You have {step_limit} shell calls.
""")

VERIFY_PROMPT = textwrap.dedent("""\
    You are reviewing completed work. The original task and the results of
    each subtask are provided below. Your job is to:

    1. Verify the work is correct by running any necessary checks.
    2. Produce the FINAL ANSWER in the exact format the original task expects.

    Call `done` with the final answer.

    You have {step_limit} shell calls.
""")

ESCALATION_PROMPT = textwrap.dedent("""\
    A cheaper model failed to complete the following task after
    {attempts} attempts.  Re-analyze the problem and either:
    - Produce a corrected, more detailed task specification, OR
    - Solve it directly yourself.

    Call `done` with either a corrected TASK specification or the result.

    You have {step_limit} shell calls.
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return text[:head] + f"\n... (truncated, {len(text)} total chars) ...\n" + text[-tail:]


async def _run_shell_command(command: str, timeout: int = COMMAND_TIMEOUT) -> dict:
    """Execute a shell command via subprocess and return structured result."""
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "exit_code": -1,
            }
        return {
            "stdout": (stdout_bytes or b"").decode("utf-8", errors="replace"),
            "stderr": (stderr_bytes or b"").decode("utf-8", errors="replace"),
            "exit_code": proc.returncode or 0,
        }
    except Exception as exc:
        return {"stdout": "", "stderr": f"Error: {exc}", "exit_code": -1}


def _parse_tasks(planner_output: str) -> list[dict]:
    """Parse the TASKS: block from the planner's output.

    Returns a list of dicts with keys: action, details, verify.
    """
    tasks: list[dict] = []

    # Find TASKS: block
    tasks_match = re.search(r"TASKS:\s*\n(.*?)(?:FINAL_ANSWER_FORMAT:|$)",
                            planner_output, re.DOTALL)
    if not tasks_match:
        return tasks

    block = tasks_match.group(1)

    # Split on numbered items: "1.", "2.", etc.
    items = re.split(r"\n\s*\d+\.\s+", "\n" + block)

    for item in items:
        item = item.strip()
        if not item:
            continue

        task: dict = {"action": "", "details": "", "verify": ""}

        action_m = re.search(r"ACTION:\s*(.*?)(?:DETAILS:|VERIFY:|$)",
                             item, re.DOTALL | re.IGNORECASE)
        if action_m:
            task["action"] = action_m.group(1).strip()

        details_m = re.search(r"DETAILS:\s*(.*?)(?:VERIFY:|$)",
                              item, re.DOTALL | re.IGNORECASE)
        if details_m:
            task["details"] = details_m.group(1).strip()

        verify_m = re.search(r"VERIFY:\s*(.*?)$", item, re.DOTALL | re.IGNORECASE)
        if verify_m:
            task["verify"] = verify_m.group(1).strip()

        if task["action"]:
            tasks.append(task)

    return tasks


def _parse_direct_answer(planner_output: str) -> str | None:
    """Check if the planner provided a DIRECT_ANSWER instead of tasks."""
    m = re.search(r"DIRECT_ANSWER:\s*(.*?)$", planner_output,
                  re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _parse_answer_format(planner_output: str) -> str:
    """Extract the FINAL_ANSWER_FORMAT hint from the planner output."""
    m = re.search(r"FINAL_ANSWER_FORMAT:\s*(.*?)$", planner_output,
                  re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# Core loop runner (shared by all phases)
# ---------------------------------------------------------------------------

async def _run_loop(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_content: str,
    step_limit: int,
    on_status: Callable | None = None,
    label: str = "unknown",
) -> str:
    """Run a Responses API tool-use loop and return the done answer or last text."""
    is_reasoning = _is_reasoning_model(model)

    if is_reasoning:
        tools = [_SHELL_TOOL, _DONE_TOOL]
    else:
        tools = [_RUN_COMMAND_TOOL, _DONE_TOOL]

    items: list = [{"role": "user", "content": user_content}]
    done_answer: str | None = None

    for step in range(step_limit):
        if on_status:
            await on_status(f"Step {step + 1}/{step_limit}")

        api_kwargs: dict = {
            "model": model,
            "instructions": system_prompt,
            "input": items,
            "tools": tools,
            "parallel_tool_calls": False,
            "store": False,
        }
        if is_reasoning:
            api_kwargs["include"] = ["reasoning.encrypted_content"]
            api_kwargs["context_management"] = [
                {"type": "compaction", "compact_threshold": COMPACT_THRESHOLD},
            ]
            api_kwargs["reasoning"] = {"effort": "high", "summary": "auto"}
            api_kwargs["max_output_tokens"] = 16_000
        else:
            api_kwargs["temperature"] = 0.0
            api_kwargs["max_output_tokens"] = 4096

        try:
            response = await client.responses.create(**api_kwargs)
        except Exception as exc:
            logger.warning("API error at step %d: %s", step, exc)
            break

        tracker.record(response, label=f"dual_model phase={label} step={step+1}")

        # Append output and handle compaction
        items.extend(response.output)
        last_compaction_idx = None
        for i, it in enumerate(items):
            if hasattr(it, "type") and it.type == "compaction":
                last_compaction_idx = i
        if last_compaction_idx is not None and last_compaction_idx > 0:
            items = items[last_compaction_idx:]

        # Collect tool calls
        shell_calls = []
        function_calls = []
        for item in response.output:
            if item.type == "shell_call":
                shell_calls.append(item)
            elif item.type == "function_call":
                function_calls.append(item)

        if not shell_calls and not function_calls:
            # Model returned text directly
            text_parts = []
            for item in response.output:
                if hasattr(item, "type") and item.type == "message":
                    for content in getattr(item, "content", []):
                        if hasattr(content, "text"):
                            text_parts.append(content.text)
                elif hasattr(item, "text"):
                    text_parts.append(item.text)
            if text_parts:
                return "\n".join(text_parts)
            break

        # Process shell_call (reasoning models)
        for sc in shell_calls:
            commands = []
            if hasattr(sc, "action") and sc.action:
                commands = (
                    sc.action.commands
                    if hasattr(sc.action, "commands")
                    else []
                )
            if not commands:
                commands = ["echo '(no command)'"]

            logger.info("[step %d] $ %s", step + 1, commands[0][:80])

            results = []
            for cmd in commands:
                r = await _run_shell_command(cmd)
                results.append({
                    "stdout": _truncate(r["stdout"], TOOL_RESULT_LIMIT),
                    "stderr": _truncate(r["stderr"], TOOL_RESULT_LIMIT),
                    "outcome": {"type": "exit", "exit_code": r["exit_code"]},
                })

            max_output_length = TOOL_RESULT_LIMIT
            if (
                hasattr(sc, "action")
                and hasattr(sc.action, "max_output_length")
                and sc.action.max_output_length
            ):
                max_output_length = sc.action.max_output_length

            items.append({
                "type": "shell_call_output",
                "call_id": sc.call_id,
                "output": results,
                "max_output_length": max_output_length,
            })

        # Process function_call (classic models + done)
        for fc in function_calls:
            name = fc.name
            try:
                args = json.loads(fc.arguments)
            except json.JSONDecodeError:
                args = {}

            if name == "done":
                done_answer = args.get("answer", "")
                items.append({
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": "Phase completed.",
                })
            elif name == "run_command":
                cmd = args.get("command", "echo 'no command'")
                logger.info("[step %d] $ %s", step + 1, cmd[:80])
                r = await _run_shell_command(cmd)
                output = r["stdout"]
                if r["stderr"]:
                    output += f"\nSTDERR: {r['stderr']}"
                output += f"\n[exit code: {r['exit_code']}]"
                items.append({
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": _truncate(output, TOOL_RESULT_LIMIT),
                })

        if done_answer is not None:
            return done_answer

    # Exhausted budget -- return whatever we have
    if done_answer is not None:
        return done_answer

    for item in reversed(items):
        if hasattr(item, "type") and item.type == "message":
            for content in getattr(item, "content", []):
                if hasattr(content, "text"):
                    return content.text
        elif hasattr(item, "text"):
            return item.text

    return ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def solve_dual_model(
    input_text: str,
    task_type: str,
    system_context: str,
    client: AsyncOpenAI,
    frontier_model: str,
    cheap_model: str,
    on_status: Callable | None = None,
) -> str:
    """Solve a task using the dual-model plan-execute approach.

    Phase 1 (frontier): Research the problem and decompose into tasks.
    Phase 2 (cheap):    Execute each task sequentially.
    Phase 3 (cheap):    Verify work and produce the final answer.

    Falls back to frontier model for escalation when cheap model fails.
    """

    # -----------------------------------------------------------------------
    # Phase 1: PLAN (frontier model)
    # -----------------------------------------------------------------------
    if on_status:
        await on_status("Phase 1: Planning with frontier model...")

    planner_system = PLANNER_PROMPT.format(step_limit=PLAN_STEP_LIMIT)
    planner_user = (
        f"## Task type: {task_type}\n\n"
        f"## System context\n{system_context}\n\n"
        f"## Task\n{input_text}"
    )

    planner_output = await _run_loop(
        client=client,
        model=frontier_model,
        system_prompt=planner_system,
        user_content=planner_user,
        step_limit=PLAN_STEP_LIMIT,
        on_status=on_status,
        label="planner",
    )

    logger.info("Planner output (%d chars): %.200s...", len(planner_output),
                planner_output)

    # Check for direct answer (no decomposition needed)
    direct = _parse_direct_answer(planner_output)
    if direct:
        logger.info("Planner returned direct answer")
        return direct

    # Parse tasks
    tasks = _parse_tasks(planner_output)
    answer_format = _parse_answer_format(planner_output)

    if not tasks:
        # Parser couldn't extract tasks -- fall back to treating
        # the entire planner output as the answer
        logger.warning("Could not parse tasks from planner output, "
                       "using raw output as answer")
        return planner_output

    logger.info("Parsed %d tasks from planner", len(tasks))

    # -----------------------------------------------------------------------
    # Phase 2: EXECUTE (cheap model, one task at a time)
    # -----------------------------------------------------------------------
    task_results: list[str] = []

    for i, task in enumerate(tasks):
        if on_status:
            await on_status(
                f"Phase 2: Executing task {i + 1}/{len(tasks)} "
                f"({task['action'][:50]}...)"
            )

        task_user = (
            f"## Your task ({i + 1} of {len(tasks)})\n\n"
            f"ACTION: {task['action']}\n\n"
            f"DETAILS:\n{task['details']}\n"
        )
        if task["verify"]:
            task_user += f"\nVERIFY: After completing the action, run: {task['verify']}\n"

        executor_system = EXECUTOR_PROMPT.format(step_limit=EXEC_STEP_LIMIT)

        result = await _run_loop(
            client=client,
            model=cheap_model,
            system_prompt=executor_system,
            user_content=task_user,
            step_limit=EXEC_STEP_LIMIT,
            on_status=on_status,
            label=f"executor-{i+1}",
        )

        # Run verification if provided
        if task["verify"]:
            verify_result = await _run_shell_command(task["verify"])
            if verify_result["exit_code"] != 0:
                logger.warning("Task %d verification failed (exit %d), "
                               "attempting retry", i + 1,
                               verify_result["exit_code"])

                # Retry once with failure context
                retry_user = (
                    f"{task_user}\n\n"
                    f"## Previous attempt failed\n"
                    f"Result: {result}\n\n"
                    f"Verification output:\n"
                    f"stdout: {_truncate(verify_result['stdout'], 2000)}\n"
                    f"stderr: {_truncate(verify_result['stderr'], 2000)}\n"
                    f"exit code: {verify_result['exit_code']}\n\n"
                    f"Fix the issue and complete the task."
                )

                result = await _run_loop(
                    client=client,
                    model=cheap_model,
                    system_prompt=executor_system,
                    user_content=retry_user,
                    step_limit=EXEC_STEP_LIMIT,
                    on_status=on_status,
                    label=f"executor-{i+1}-retry",
                )

                # Check verification again
                verify_result2 = await _run_shell_command(task["verify"])
                if verify_result2["exit_code"] != 0:
                    # Escalate to frontier model
                    logger.warning("Task %d failed after retry, "
                                   "escalating to frontier", i + 1)
                    if on_status:
                        await on_status(
                            f"Escalating task {i + 1} to frontier model..."
                        )

                    escalation_system = ESCALATION_PROMPT.format(
                        attempts=ESCALATION_ATTEMPTS,
                        step_limit=PLAN_STEP_LIMIT,
                    )
                    escalation_user = (
                        f"## Failed task\n\n"
                        f"ACTION: {task['action']}\n"
                        f"DETAILS: {task['details']}\n\n"
                        f"## Last attempt result\n{result}\n\n"
                        f"## Verification failure\n"
                        f"{_truncate(verify_result2['stdout'], 2000)}\n"
                        f"{_truncate(verify_result2['stderr'], 2000)}"
                    )

                    result = await _run_loop(
                        client=client,
                        model=frontier_model,
                        system_prompt=escalation_system,
                        user_content=escalation_user,
                        step_limit=PLAN_STEP_LIMIT,
                        on_status=on_status,
                        label=f"escalation-{i+1}",
                    )

        task_results.append(f"Task {i + 1}: {task['action']}\nResult: {result}")
        logger.info("Task %d/%d completed", i + 1, len(tasks))

    # -----------------------------------------------------------------------
    # Phase 3: VERIFY + ANSWER (cheap model)
    # -----------------------------------------------------------------------
    if on_status:
        await on_status("Phase 3: Verifying and producing final answer...")

    verify_system = VERIFY_PROMPT.format(step_limit=VERIFY_STEP_LIMIT)
    verify_user = (
        f"## Original task\n{input_text}\n\n"
        f"## Completed subtasks\n" +
        "\n\n---\n\n".join(task_results)
    )
    if answer_format:
        verify_user += f"\n\n## Required answer format\n{answer_format}"

    final_answer = await _run_loop(
        client=client,
        model=cheap_model,
        system_prompt=verify_system,
        user_content=verify_user,
        step_limit=VERIFY_STEP_LIMIT,
        on_status=on_status,
        label="verify",
    )

    return final_answer
