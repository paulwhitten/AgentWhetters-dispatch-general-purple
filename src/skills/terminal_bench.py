"""Terminal Bench 2.0 skill -- multi-turn shell interaction via JSON protocol.

Protocol: terminal-bench-shell-v1
  Turn 1: green sends {"kind":"task","protocol":"terminal-bench-shell-v1","instruction":"..."}
  Purple responds:     {"kind":"exec_request","command":"...","timeout":N}
  Turn 2+: green sends {"kind":"exec_result","exit_code":N,"stdout":"...","stderr":"..."}
  Purple responds:     {"kind":"exec_request","command":"...","timeout":N}
                    or {"kind":"final","output":"..."}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from tools import truncate
from usage import tracker

logger = logging.getLogger("agentwhetters.terminal_bench")

# Max turns before forcing a final response
MAX_TURNS = 80
# Truncate long command output to keep context manageable
MAX_OUTPUT_CHARS = 30_000

SYSTEM_PROMPT = """\
You are an expert systems administrator and software engineer solving tasks \
in a Linux Docker container. You interact by requesting shell commands one at \
a time and receiving their output.

## Approach

1. Read the task instruction carefully before acting.
2. Explore the environment first: ls /, ls /app, cat README*, which <tool>, etc.
3. Plan your approach, then execute step by step.
4. Verify your work after each significant step.
5. When done and verified, call submit_final.

## Efficiency

- Chain related commands: cmd1 && cmd2 && cmd3
- Write multi-step logic as inline scripts: bash -c '...'
- Install packages in one shot: apt-get update && apt-get install -y pkg1 pkg2
- Pipe long output through head/tail/grep to keep it manageable.
- Set timeout appropriately: 30s for quick commands, 120-300s for builds/downloads.
- You have a limited turn budget. Be efficient.

## Common patterns

- **Builds**: read Makefile/CMakeLists.txt, install dependencies, then build. \
Check for build errors and fix them.
- **Git**: use git log --oneline, git reflog, git status, git diff to understand state.
- **Services**: check config syntax (nginx -t), then start, then verify (curl localhost:PORT).
- **Code fixes**: read the code, understand the bug, make minimal targeted changes, test.
- **Crypto/security**: check for installed tools (john, hashcat, openssl), install if needed.
- **Data/ML**: check Python version, install deps with pip, run scripts.
- **Cross-compilation**: identify target arch, install cross toolchain, configure properly.

## Rules

- Never guess at file contents -- always cat/read them.
- Read error messages carefully before retrying.
- If a command fails, diagnose why before trying alternatives.
- When the task says to produce a specific file or output, verify it exists and is correct.
"""

EXEC_COMMAND_TOOL: dict = {
    "type": "function",
    "name": "exec_command",
    "description": (
        "Execute a shell command in the container. Chain commands with && or ;. "
        "Set timeout appropriately for long operations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": (
                    "Timeout in seconds (1-300). Default 30. "
                    "Use 120-300 for builds, downloads, model inference."
                ),
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    "strict": False,
}

SUBMIT_FINAL_TOOL: dict = {
    "type": "function",
    "name": "submit_final",
    "description": (
        "Submit the final result and end the task. "
        "Call this only when the task is complete and verified."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "output": {
                "type": "string",
                "description": "Brief summary of what was accomplished.",
            },
        },
        "required": ["output"],
        "additionalProperties": False,
    },
    "strict": False,
}

TB_TOOLS = [EXEC_COMMAND_TOOL, SUBMIT_FINAL_TOOL]


@dataclass
class TerminalBenchState:
    """Persistent state across multi-turn Terminal Bench interactions."""

    instruction: str = ""
    # OpenAI Responses API items list
    items: list = field(default_factory=list)
    # ID of the last exec_command function_call for pairing with output
    pending_call_id: str | None = None
    turn_count: int = 0


def is_terminal_bench_protocol(input_text: str) -> bool:
    """Detect the initial terminal-bench-shell-v1 task message."""
    try:
        payload = json.loads(input_text)
        return (
            isinstance(payload, dict)
            and payload.get("protocol") == "terminal-bench-shell-v1"
        )
    except (json.JSONDecodeError, TypeError):
        return False


def is_terminal_bench_result(input_text: str) -> bool:
    """Detect an exec_result continuation message."""
    try:
        payload = json.loads(input_text)
        return isinstance(payload, dict) and payload.get("kind") == "exec_result"
    except (json.JSONDecodeError, TypeError):
        return False


async def solve_terminal_bench(
    input_text: str,
    client: AsyncOpenAI,
    model: str,
    *,
    state: TerminalBenchState | None = None,
    on_status=None,
) -> tuple[str, TerminalBenchState]:
    """Process one turn of a Terminal Bench interaction.

    Returns (response_json_string, updated_state).
    """
    payload = json.loads(input_text)

    if state is None:
        # ── First turn: task instruction ──
        state = TerminalBenchState()
        state.instruction = payload.get("instruction", "")
        state.items = [
            {"role": "user", "content": state.instruction},
        ]
        logger.info(
            "Terminal Bench: new task (%d chars instruction)",
            len(state.instruction),
        )
    else:
        # ── Continuation: exec_result ──
        state.turn_count += 1
        exit_code = payload.get("exit_code", -1)
        stdout = payload.get("stdout", "")
        stderr = payload.get("stderr", "")

        # Truncate long output
        stdout = truncate(stdout, MAX_OUTPUT_CHARS)
        stderr = truncate(stderr, MAX_OUTPUT_CHARS)

        # Build tool output text
        result_text = f"exit_code={exit_code}\n"
        if stdout:
            result_text += f"stdout:\n{stdout}\n"
        if stderr:
            result_text += f"stderr:\n{stderr}\n"
        if not stdout and not stderr:
            result_text += "(no output)\n"

        # Append as function_call_output paired with the pending call
        if state.pending_call_id:
            state.items.append({
                "type": "function_call_output",
                "call_id": state.pending_call_id,
                "output": result_text,
            })
            state.pending_call_id = None
        else:
            # Fallback: append as user message
            state.items.append({
                "role": "user",
                "content": f"Command result:\n{result_text}",
            })

    # ── Turn limit check ──
    if state.turn_count >= MAX_TURNS:
        logger.warning("Terminal Bench: turn limit reached (%d)", state.turn_count)
        return json.dumps({"kind": "final", "output": "Turn limit reached."}), state

    if on_status:
        await on_status(f"Terminal Bench: turn {state.turn_count + 1}")

    # ── Ask LLM for next action ──
    try:
        response = await client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=state.items,
            tools=TB_TOOLS,
            parallel_tool_calls=False,
            store=False,
            temperature=1.0,  # required for reasoning models; ignored by non-reasoning
            max_output_tokens=4096,
        )
    except Exception as exc:
        logger.error("Terminal Bench: API error at turn %d: %s", state.turn_count, exc)
        return json.dumps({
            "kind": "final",
            "output": f"API error: {exc}",
        }), state

    tracker.record(response, label=f"terminal_bench turn={state.turn_count}")

    # Append all output items to state
    state.items.extend(response.output)

    # ── Extract function calls ──
    function_calls = [
        it for it in response.output if it.type == "function_call"
    ]

    if function_calls:
        fc = function_calls[0]
        try:
            args = json.loads(fc.arguments)
        except json.JSONDecodeError:
            args = {}

        if fc.name == "exec_command":
            command = args.get("command", "echo 'no command'")
            timeout = args.get("timeout", 30)
            if not isinstance(timeout, int):
                timeout = 30
            timeout = max(1, min(timeout, 300))
            state.pending_call_id = fc.call_id

            logger.info(
                "Terminal Bench [turn %d]: $ %s",
                state.turn_count, command[:120],
            )
            return json.dumps({
                "kind": "exec_request",
                "command": command,
                "timeout": timeout,
            }), state

        elif fc.name == "submit_final":
            output = args.get("output", "Task completed.")
            logger.info(
                "Terminal Bench [turn %d]: FINAL - %s",
                state.turn_count, output[:120],
            )
            return json.dumps({"kind": "final", "output": output}), state

    # ── No tool calls: extract text and treat as final ──
    text_parts = []
    for item in response.output:
        if hasattr(item, "type") and item.type == "message":
            for content in getattr(item, "content", []):
                if hasattr(content, "text"):
                    text_parts.append(content.text)
        elif hasattr(item, "text"):
            text_parts.append(item.text)

    output = "\n".join(text_parts) if text_parts else "Task completed."
    logger.info(
        "Terminal Bench [turn %d]: no tool call, sending final",
        state.turn_count,
    )
    return json.dumps({"kind": "final", "output": output}), state
