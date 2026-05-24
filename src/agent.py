import json
import logging
import os
import time

import httpx
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Message, TaskState, Part, TextPart,
    FilePart, DataPart,
)
from a2a.utils import get_message_text, new_agent_text_message
from openai import AsyncOpenAI, AsyncAzureOpenAI

from classifier import classify_structural, classify_llm
from messenger import Messenger
from prompts import get_system_prompt
from skills.bwim import BWIMState, solve_bwim
from skills.cybergym import CyberGymState, is_test_result, solve_cybergym
from skills.dual_model import solve_dual_model
from skills.swe_bench import solve_instance, get_strategy
from skills.officeqa_rag import solve_officeqa
from skills.terminal_bench import (
    TerminalBenchState,
    is_terminal_bench_protocol,
    is_terminal_bench_result,
    solve_terminal_bench,
)
from skills.mle_bench import solve_mle_bench
from skills.netarena import solve_netarena_malt
from skills.netarena_k8s import solve_netarena_k8s
from skills.web_research import solve_web_research
from tools import (
    DONE_TOOL,
    FINANCE_STEP_LIMIT,
    GENERAL_COMPACT_THRESHOLD,
    GENERAL_STEP_LIMIT,
    GENERAL_TOOL_RESULT_LIMIT,
    RUN_COMMAND_TOOL,
    SHELL_TOOL,
    get_dual_model_tasks,
    is_reasoning_model,
    run_shell_command,
    truncate,
)
from usage import tracker

logger = logging.getLogger("agentwhetters.agent")

# Task types that use the web_research skill (single-call with web search)
_WEB_RESEARCH_TASKS: set[str] = set()  # finance moved to general loop for PDF extraction

# Model routing: task types that get the strong model
_STRONG_TASKS = {
    "swe-bench", "cybersecurity", "finance", "research", "mle-bench",
    "computer-use", "coding", "terminal-bench", "netarena-malt",
    "netarena-k8s",
}


def _format_attachments(attachments: list[dict]) -> str:
    """Format attachment metadata as text to append to user content."""
    lines = ["\n\n## Attachments"]
    for att in attachments:
        if att["type"] == "file":
            lines.append(
                f"- File: {att['name']} (type: {att.get('mime_type', 'unknown')}, "
                f"uri: {att.get('uri', 'N/A')})"
            )
        elif att["type"] == "data":
            data_preview = str(att.get("data", ""))[:500]
            lines.append(
                f"- Data ({att.get('mime_type', 'unknown')}): {data_preview}"
            )
    return "\n".join(lines)


def _make_openai_client() -> AsyncOpenAI:
    """Create an OpenAI client, using Azure when configured."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    if azure_endpoint:
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
        return AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


class Agent:
    """General-purpose agent that adapts strategy based on task type.

    Classification, prompt selection, tool definitions, and shell execution
    are delegated to focused modules (classifier, prompts, tools).
    This class owns dispatch logic and the execution strategies.
    """

    def __init__(self):
        self.messenger = Messenger()
        self.client = _make_openai_client()
        self.model = os.environ.get("AGENT_MODEL", "gpt-5.4")
        self.mid_model = os.environ.get("AGENT_MID_MODEL", "gpt-4.1")
        self.research_model = os.environ.get("AGENT_RESEARCH_MODEL", "gpt-5.4-mini")
        self.cheap_model = os.environ.get("AGENT_CHEAP_MODEL", "gpt-4o-mini")
        # Multi-turn state
        self._active_skill: str | None = None
        self._cybergym_state: CyberGymState | None = None
        self._bwim_state: BWIMState | None = None
        self._terminal_bench_state: TerminalBenchState | None = None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        # Multi-turn: route continuation messages to the active skill
        if self._active_skill == "cybergym" and is_test_result(message):
            self._cybergym_state = await solve_cybergym(
                message, updater, self.client, self.model,
                state=self._cybergym_state,
            )
            tracker.log_summary()
            return
        if self._active_skill == "bwim":
            input_text = get_message_text(message)
            result, self._bwim_state = await solve_bwim(
                input_text, self.client, self.cheap_model,
                state=self._bwim_state,
            )
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=result))],
                name="result",
            )
            tracker.log_summary()
            return
        if self._active_skill == "terminal-bench":
            input_text = get_message_text(message)
            if is_terminal_bench_result(input_text):
                result, self._terminal_bench_state = await solve_terminal_bench(
                    input_text, self.client, self.model,
                    state=self._terminal_bench_state,
                    on_status=lambda msg: updater.update_status(
                        TaskState.working, new_agent_text_message(msg),
                    ),
                )
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text=result))],
                    name="result",
                )
                tracker.log_summary()
                return

        input_text = get_message_text(message)
        attachments = self._extract_attachments(message)
        logger.info("Received message (%d chars, %d attachments)",
                     len(input_text), len(attachments))

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Analyzing task..."),
        )

        # Two-tier classification: structural pre-filter, then LLM
        task_type = classify_structural(input_text)
        if task_type:
            logger.info("Classified task as: %s (structural)", task_type)
        else:
            task_type = await classify_llm(
                input_text, self.client, self.cheap_model,
            )
            logger.info("Classified task as: %s (LLM)", task_type)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Task classified as {task_type}. Working..."),
        )

        # Dispatch to the appropriate execution strategy
        if task_type == "mle-bench":
            result = await solve_mle_bench(
                message, self.client, self.model,
                on_status=lambda msg: updater.update_status(
                    TaskState.working, new_agent_text_message(msg),
                ),
            )
        elif task_type == "swe-bench":
            result = await self._execute_swe_bench(input_text, updater)
        elif task_type == "cybersecurity":
            self._active_skill = "cybergym"
            self._cybergym_state = await solve_cybergym(
                message, updater, self.client, self.model,
            )
            tracker.log_summary()
            return
        elif task_type == "game":
            self._active_skill = "bwim"
            result, self._bwim_state = await solve_bwim(
                input_text, self.client, self.cheap_model,
            )
        elif task_type == "terminal-bench":
            self._active_skill = "terminal-bench"
            result, self._terminal_bench_state = await solve_terminal_bench(
                input_text, self.client, self.model,
                on_status=lambda msg: updater.update_status(
                    TaskState.working, new_agent_text_message(msg),
                ),
            )
        elif task_type == "netarena-malt":
            result = await solve_netarena_malt(
                input_text, self.client, self.model,
            )
        elif task_type == "netarena-k8s":
            result = await solve_netarena_k8s(
                input_text, self.client, self.model,
            )
        elif task_type == "finance":
            result = await solve_officeqa(
                input_text, self.client, self.model,
                cheap_model=self.cheap_model,
                on_status=lambda msg: updater.update_status(
                    TaskState.working, new_agent_text_message(msg),
                ),
            )
        elif task_type in _WEB_RESEARCH_TASKS:
            result = await solve_web_research(
                input_text, self.client, self.research_model,
                on_status=lambda msg: updater.update_status(
                    TaskState.working, new_agent_text_message(msg),
                ),
            )
        elif task_type in get_dual_model_tasks():
            result = await self._execute_dual_model(
                input_text, task_type, updater, attachments,
            )
        else:
            result = await self._execute(
                input_text, task_type, updater, attachments,
            )

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=result))],
            name="result",
        )
        tracker.log_summary()

    # ------------------------------------------------------------------
    # Attachment extraction
    # ------------------------------------------------------------------

    def _extract_attachments(self, message: Message) -> list[dict]:
        """Extract file and data attachments from A2A message parts."""
        attachments = []
        if not message.parts:
            return attachments
        for part in message.parts:
            root = part.root if hasattr(part, "root") else part
            if isinstance(root, FilePart):
                attachments.append({
                    "type": "file",
                    "name": getattr(root, "name", None) or "unnamed",
                    "uri": getattr(root, "uri", None),
                    "mime_type": getattr(root, "mime_type", None),
                })
            elif isinstance(root, DataPart):
                attachments.append({
                    "type": "data",
                    "data": getattr(root, "data", None),
                    "mime_type": getattr(root, "mime_type", None),
                })
        return attachments

    # ------------------------------------------------------------------
    # Model routing
    # ------------------------------------------------------------------

    def _get_model(self, task_type: str) -> str:
        """Return the model to use for the given task type."""
        if task_type in _STRONG_TASKS:
            return self.model
        return self.cheap_model

    # ------------------------------------------------------------------
    # Execution strategies
    # ------------------------------------------------------------------

    async def _execute_dual_model(
        self,
        input_text: str,
        task_type: str,
        updater: TaskUpdater,
        attachments: list[dict] | None = None,
    ) -> str:
        """Execute using the dual-model plan-execute strategy."""
        system_context = get_system_prompt(task_type)
        user_text = input_text
        if attachments:
            user_text += _format_attachments(attachments)

        async def on_status(msg: str) -> None:
            await updater.update_status(
                TaskState.working, new_agent_text_message(msg),
            )

        return await solve_dual_model(
            input_text=user_text,
            task_type=task_type,
            system_context=system_context,
            client=self.client,
            frontier_model=self.model,
            cheap_model=self.cheap_model,
            on_status=on_status,
        )

    async def _execute(
        self,
        input_text: str,
        task_type: str,
        updater: TaskUpdater,
        attachments: list[dict] | None = None,
    ) -> str:
        """General-purpose agentic loop using the Responses API.

        Gives the LLM a shell tool and a done tool, iterating up to
        GENERAL_STEP_LIMIT steps. Works for any task type that does not
        have a specialized skill module.
        """
        system_prompt = get_system_prompt(task_type)
        model = self._get_model(task_type)
        reasoning = is_reasoning_model(model)
        step_limit = FINANCE_STEP_LIMIT if task_type == "finance" else GENERAL_STEP_LIMIT

        tools = [SHELL_TOOL, DONE_TOOL] if reasoning else [RUN_COMMAND_TOOL, DONE_TOOL]

        user_content = input_text
        if attachments:
            user_content += _format_attachments(attachments)

        items: list = [{"role": "user", "content": user_content}]
        done_answer: str | None = None

        for step in range(step_limit):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Step {step + 1}/{step_limit}"),
            )

            api_kwargs: dict = {
                "model": model,
                "instructions": system_prompt,
                "input": items,
                "tools": tools,
                "parallel_tool_calls": False,
                "store": False,
            }
            if reasoning:
                api_kwargs["include"] = ["reasoning.encrypted_content"]
                api_kwargs["context_management"] = [
                    {"type": "compaction",
                     "compact_threshold": GENERAL_COMPACT_THRESHOLD},
                ]
                api_kwargs["reasoning"] = {"effort": "high", "summary": "auto"}
                api_kwargs["max_output_tokens"] = 16_000
            else:
                api_kwargs["temperature"] = 0.0
                api_kwargs["max_output_tokens"] = 4096

            try:
                response = await self.client.responses.create(**api_kwargs)
            except Exception as exc:
                logger.warning("API error at step %d: %s", step, exc)
                break

            tracker.record(response, label=f"general step={step+1}")

            # Append output and handle compaction
            items.extend(response.output)
            last_compaction_idx = None
            for i, it in enumerate(items):
                if hasattr(it, "type") and it.type == "compaction":
                    last_compaction_idx = i
            if last_compaction_idx is not None and last_compaction_idx > 0:
                items = items[last_compaction_idx:]

            # Collect tool calls
            shell_calls = [
                it for it in response.output if it.type == "shell_call"
            ]
            function_calls = [
                it for it in response.output if it.type == "function_call"
            ]

            # No tool calls = model gave a direct text response
            if not shell_calls and not function_calls:
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

            # Process shell_call items (reasoning models)
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
                    r = await run_shell_command(cmd)
                    results.append({
                        "stdout": truncate(r["stdout"], GENERAL_TOOL_RESULT_LIMIT),
                        "stderr": truncate(r["stderr"], GENERAL_TOOL_RESULT_LIMIT),
                        "outcome": {"type": "exit", "exit_code": r["exit_code"]},
                    })

                max_output_length = GENERAL_TOOL_RESULT_LIMIT
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

            # Process function_call items (classic models + done tool)
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
                        "output": "Task completed.",
                    })
                elif name == "run_command":
                    cmd = args.get("command", "echo 'no command'")
                    logger.info("[step %d] $ %s", step + 1, cmd[:80])
                    r = await run_shell_command(cmd)
                    output = r["stdout"]
                    if r["stderr"]:
                        output += f"\nSTDERR: {r['stderr']}"
                    output += f"\n[exit code: {r['exit_code']}]"
                    items.append({
                        "type": "function_call_output",
                        "call_id": fc.call_id,
                        "output": truncate(output, GENERAL_TOOL_RESULT_LIMIT),
                    })

            if done_answer is not None:
                return done_answer

        # Exhausted step budget or API error -- return whatever we have
        if done_answer is not None:
            return done_answer

        for item in reversed(items):
            if hasattr(item, "type") and item.type == "message":
                for content in getattr(item, "content", []):
                    if hasattr(content, "text"):
                        return content.text
            elif isinstance(item, dict) and item.get("type") == "function_call_output":
                continue
            elif hasattr(item, "text"):
                return item.text

        return "I was unable to complete this task within the step budget."

    async def _execute_swe_bench(self, input_text: str, updater: TaskUpdater) -> str:
        """Execute a SWE-bench task using the flat agentic loop."""
        try:
            instance = json.loads(input_text)
        except json.JSONDecodeError as exc:
            return json.dumps({
                "patch": "",
                "error": f"Could not parse instance: {exc}",
            })

        timeout = httpx.Timeout(connect=60.0, read=1800.0, write=60.0, pool=60.0)
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        if azure_endpoint:
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
            client = AsyncAzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=os.environ.get("OPENAI_API_KEY"),
                api_version=api_version,
                timeout=timeout,
            )
        else:
            client = AsyncOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"), timeout=timeout,
            )

        async def on_status(msg: str) -> None:
            await updater.update_status(
                TaskState.working, new_agent_text_message(msg),
            )

        t0 = time.monotonic()
        strategy = get_strategy(os.environ.get("SWE_BENCH_STRATEGY"))
        try:
            patch = await solve_instance(
                instance=instance,
                client=client,
                model=self.model,
                mid_model=self.mid_model,
                on_status=on_status,
                strategy=strategy,
            )
        except Exception as exc:
            logger.exception("solve_instance failed: %s", exc)
            patch = ""

        elapsed = time.monotonic() - t0
        instance_id = instance.get("instance_id", "unknown")
        logger.info("Instance %s solved in %.1fs, patch length: %d",
                     instance_id, elapsed, len(patch))

        if patch:
            return json.dumps({"patch": patch})
        return json.dumps({"patch": "", "error": "Failed to generate patch"})
