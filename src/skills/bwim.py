"""BWIM skill -- spatial block-building agent.

Ported from the tied-1st-place BWIM purple agent. Uses a skills pipeline:
parse → underspec detect → LLM plan → patch → verify → execute → format.
Falls back to direct LLM call if the pipeline fails.

Multi-turn protocol:
  1. Green sends [TASK_DESCRIPTION] + [START_STRUCTURE] → agent responds [BUILD] or [ASK]
  2. If [ASK], green sends Answer → agent responds [BUILD]
  3. Green sends feedback → agent acknowledges
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from usage import tracker

from .bwim_skills.instruction_parser import parse_green_message, ParsedInstruction
from .bwim_skills.build_planner import BuildPlanner
from .bwim_skills.spatial_executor import SpatialExecutor, ExecutionError
from .bwim_skills.underspec_detector import (
    detect_underspec_heuristic,
    patch_instruction_with_color,
    patch_instruction_with_count,
)
from .bwim_skills.response_formatter import format_build_response, validate_build_response
from .bwim_skills.grid import Grid, GridConfig
from .bwim_skills.structure_analyzer import analyze_structure
from .bwim_skills.plan_verifier import (
    verify_plan,
    auto_fix_direction,
    auto_fix_each_end_caps,
    auto_fix_t_shape_extend,
)
from .bwim_skills.plan_patcher import patch_chain_references

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback system prompt
# ---------------------------------------------------------------------------

_FALLBACK_SYSTEM_PROMPT = (
    "You are a block-building agent on a 9x9 grid.\n\n"

    "GRID COORDINATES:\n"
    "- The grid is the x-z plane. Origin (0,0) is the center.\n"
    "- Valid x,z coordinates: [-400,-300,-200,-100,0,100,200,300,400]\n"
    "- Y-axis is vertical (height). Ground level y=50. Each block adds +100.\n"
    "- Valid y coordinates: [50,150,250,350,450]\n"
    "- Format: Color,x,y,z (e.g., Red,0,50,0 means a red block at center, ground level)\n\n"

    "DIRECTIONS (CRITICAL):\n"
    "- 'in front of' = +z direction (increasing z)\n"
    "- 'behind' = -z direction (decreasing z)\n"
    "- 'to the right' = +x direction (increasing x)\n"
    "- 'to the left' = -x direction (decreasing x)\n"
    "- 'on top of' = +y direction (increasing y)\n\n"

    "CORNERS:\n"
    "- bottom left = (-400, 50, 400), bottom right = (400, 50, 400)\n"
    "- top left = (-400, 50, -400), top right = (400, 50, -400)\n\n"

    "YOUR RESPONSE FORMAT:\n"
    "You must respond with ONLY this format:\n\n"

    "[BUILD];Color,x,y,z;Color,x,y,z;...\n"
    "- List ALL blocks that should be on the grid (existing + new)\n"
    "- No spaces, semicolons separate blocks\n"
    "- Colors capitalized (Red, Blue, Green, Yellow, Purple, etc.)\n"
    "- NEVER respond with [ASK] — always BUILD your best guess.\n\n"

    "SPATIAL RULES:\n"
    "- 'highlighted square' / 'middle square' = origin (0,0).\n"
    "- 'each end' of a row = leftmost AND rightmost.\n"
    "- 'in front of the leftmost' = find block with MIN x, then +z.\n"
    "- Chain references: 'stack green behind existing at (0,0)' → green at (0,-100).\n"
    "  'yellow to right of the green one' → yellow at (100,-100).\n"
)

_COLOR_NAMES = {
    "red", "blue", "green", "yellow", "purple", "orange",
    "white", "black", "brown", "pink", "grey", "gray", "cyan",
}

_ANSWER_RE = re.compile(
    r'^Answer:\s*(.+?)(?:\s*\(.*points.*\))?$',
    re.IGNORECASE,
)

_WORD_TO_INT = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class BWIMState:
    """Persistent state for a BWIM context across multi-turn exchanges."""
    pending: dict | None = None  # Pending ASK state
    asked: bool = False  # Whether we already asked this round
    history: list[dict] = field(default_factory=list)
    max_history: int = 5


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _extract_answer_colors(text: str) -> list[str]:
    """Extract ALL color names from an answer message."""
    m = _ANSWER_RE.match(text.strip())
    if not m:
        return []
    answer_body = m.group(1).strip().rstrip(".,!").lower()
    colors: list[str] = []
    for match in re.finditer(r'\b(' + '|'.join(_COLOR_NAMES) + r')\b', answer_body):
        c = match.group(1).capitalize()
        if c not in colors:
            colors.append(c)
    return colors


def _extract_answer_count(text: str) -> int | None:
    """Extract a numeric count from an answer message."""
    m = _ANSWER_RE.match(text.strip())
    if not m:
        return None
    answer_body = m.group(1).strip().rstrip(".,!").lower()
    digit_match = re.search(r'\b(\d+)\b', answer_body)
    if digit_match:
        return int(digit_match.group(1))
    for word, val in _WORD_TO_INT.items():
        if re.search(r'\b' + word + r'\b', answer_body):
            return val
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def solve_bwim(
    message_text: str,
    client: AsyncOpenAI,
    model: str,
    state: BWIMState | None = None,
) -> tuple[str, BWIMState]:
    """Process a BWIM message.

    Returns (response_text, updated_state).
    response_text is a [BUILD] or [ASK] string.
    """
    if state is None:
        state = BWIMState()

    config = GridConfig()
    parsed = parse_green_message(message_text, config)

    # Handle feedback messages (new round)
    if parsed.is_feedback:
        _add_to_history(state, "feedback", parsed.feedback_text)
        state.pending = None
        state.asked = False
        return "[BUILD]", state

    # Check if this is an answer to a question we asked
    pending = state.pending
    state.pending = None
    ask_type = pending.get("ask_type", "color") if pending else "color"

    answered_colors = _extract_answer_colors(parsed.instruction_text)
    answered_count = _extract_answer_count(parsed.instruction_text)

    if pending and (answered_colors or answered_count is not None):
        state.asked = True
        planner = BuildPlanner(client, model, config)

        if ask_type == "compound":
            response = await _handle_compound_answer(
                pending, answered_colors, answered_count, planner, config,
                client, model, state,
            )
        elif ask_type == "count" and answered_count is not None:
            response = await _handle_count_answer(
                pending, answered_count, planner, config,
                client, model, state,
            )
        else:
            response = await _handle_color_answer(
                pending, answered_colors, planner, config,
                client, model, state,
            )
    else:
        # Normal instruction — run the full pipeline
        planner = BuildPlanner(client, model, config)
        response = await _skills_pipeline(
            parsed, planner, config, state, message_text,
        )
        if response is None:
            response = await _direct_llm_call(
                message_text, client, model, state,
            )

    # Hard guard: never send [ASK] more than once per round
    if response.startswith("[ASK]") and state.asked:
        logger.warning("HARD GUARD: suppressing repeated [ASK]")
        response = await _direct_llm_call(message_text, client, model, state)
    elif response.startswith("[ASK]"):
        state.asked = True

    _add_to_history(state, "instruction", parsed.instruction_text)
    _add_to_history(state, "response", response)

    return response, state


# ---------------------------------------------------------------------------
# Answer handlers
# ---------------------------------------------------------------------------

async def _handle_compound_answer(
    pending: dict,
    answered_colors: list[str],
    answered_count: int | None,
    planner: BuildPlanner,
    config: GridConfig,
    client: AsyncOpenAI,
    model: str,
    state: BWIMState,
) -> str:
    """Handle compound answer (both color and count)."""
    original_instruction = pending["parsed"].instruction_text
    instruction_lower = original_instruction.lower()
    instruction_colors = {c for c in _COLOR_NAMES if c in instruction_lower}

    if len(answered_colors) > 1:
        new_colors = [c for c in answered_colors if c.lower() not in instruction_colors]
        color_str = new_colors[0] if new_colors else answered_colors[-1]
    else:
        color_str = answered_colors[0] if answered_colors else "Purple"

    patched_text = patch_instruction_with_color(
        pending["parsed"].instruction_text, color_str,
    )
    if answered_count is not None:
        patched_text = patch_instruction_with_count(patched_text, answered_count)
        pending["parsed"]._answered_count = answered_count

    pending["parsed"].instruction_text = patched_text
    response = await _skills_pipeline(
        pending["parsed"], planner, config, state,
        pending.get("original_input", ""),
        override_count=answered_count,
    )
    if response is None:
        combined = (
            pending.get("original_input", "")
            + f"\n\nThe answer to the question is: {color_str}"
            + (f", {answered_count} blocks" if answered_count else "")
            + f". Use {color_str} for the unspecified blocks. Respond with [BUILD]."
        )
        response = await _direct_llm_call(combined, client, model, state)
    return response


async def _handle_count_answer(
    pending: dict,
    answered_count: int,
    planner: BuildPlanner,
    config: GridConfig,
    client: AsyncOpenAI,
    model: str,
    state: BWIMState,
) -> str:
    """Handle count-only answer."""
    uncounted_color = pending.get("uncounted_color", "")
    patched_text = patch_instruction_with_count(
        pending["parsed"].instruction_text, answered_count,
        target_color=uncounted_color,
    )
    pending["parsed"].instruction_text = patched_text
    pending["parsed"]._answered_count = answered_count
    response = await _skills_pipeline(
        pending["parsed"], planner, config, state,
        pending.get("original_input", ""),
        override_count=answered_count,
    )
    if response is None:
        color_hint = f" {uncounted_color}" if uncounted_color else ""
        combined = (
            pending.get("original_input", "")
            + f"\n\n[IMPORTANT: Build exactly {answered_count}{color_hint} blocks"
            f" for the unspecified stack. Do NOT ask questions."
            f" Respond ONLY with [BUILD].]"
        )
        response = await _direct_llm_call(combined, client, model, state)
    return response


async def _handle_color_answer(
    pending: dict,
    answered_colors: list[str],
    planner: BuildPlanner,
    config: GridConfig,
    client: AsyncOpenAI,
    model: str,
    state: BWIMState,
) -> str:
    """Handle color-only answer."""
    original_instruction = pending["parsed"].instruction_text
    instruction_lower = original_instruction.lower()
    instruction_colors = {c for c in _COLOR_NAMES if c in instruction_lower}

    if len(answered_colors) > 1:
        new_colors = [c for c in answered_colors if c.lower() not in instruction_colors]
        color_str = new_colors[0] if new_colors else answered_colors[-1]
    else:
        color_str = answered_colors[0] if answered_colors else "Purple"

    patched_text = patch_instruction_with_color(
        pending["parsed"].instruction_text, color_str,
    )
    pending["parsed"].instruction_text = patched_text
    response = await _skills_pipeline(
        pending["parsed"], planner, config, state,
        pending.get("original_input", ""),
    )
    if response is None:
        combined = (
            pending.get("original_input", "")
            + f"\n\nThe answer is: {color_str}. "
            "Use this color for the unspecified blocks. Respond with [BUILD]."
        )
        response = await _direct_llm_call(combined, client, model, state)
    return response


# ---------------------------------------------------------------------------
# Skills pipeline
# ---------------------------------------------------------------------------

async def _skills_pipeline(
    parsed: ParsedInstruction,
    planner: BuildPlanner,
    config: GridConfig,
    state: BWIMState,
    original_input: str = "",
    override_count: int | None = None,
) -> str | None:
    """Run the skills-based build pipeline. Returns [BUILD]/[ASK] or None on failure."""
    try:
        # Step 1: Pre-LLM underspec check
        heuristic_result = detect_underspec_heuristic(parsed.instruction_text)
        inferred_count = override_count or heuristic_result.inferred_count or 3

        # Compound: both color and count missing
        if (heuristic_result.has_missing_color
                and heuristic_result.has_missing_number
                and not state.asked):
            state.pending = {
                "parsed": parsed,
                "original_input": original_input,
                "inferred_count": inferred_count,
                "ask_type": "compound",
                "uncounted_color": heuristic_result.uncounted_color,
            }
            question = (
                heuristic_result.suggested_compound_question
                or "What color should the unspecified blocks be, "
                   "and how many blocks should be in that stack?"
            )
            return f"[ASK];{question}"

        # Color missing
        if heuristic_result.has_missing_color and not state.asked:
            state.pending = {
                "parsed": parsed,
                "original_input": original_input,
                "inferred_count": inferred_count,
                "ask_type": "color",
            }
            question = (
                heuristic_result.suggested_question
                or "What color should the unspecified blocks be?"
            )
            return f"[ASK];{question}"

        # Count missing
        if (heuristic_result.has_missing_number
                and not heuristic_result.has_missing_color
                and not state.asked
                and override_count is None):
            state.pending = {
                "parsed": parsed,
                "original_input": original_input,
                "inferred_count": inferred_count,
                "ask_type": "count",
                "uncounted_color": heuristic_result.uncounted_color,
            }
            question = (
                heuristic_result.suggested_count_question
                or "How many blocks should be in the unspecified stack?"
            )
            return f"[ASK];{question}"

        # Already asked — fill with inferred color
        if heuristic_result.has_missing_color:
            fill = heuristic_result.inferred_color or "Purple"
            patched = patch_instruction_with_color(parsed.instruction_text, fill)
            parsed.instruction_text = patched

        # Step 2: Analyze existing structure
        structure_info = analyze_structure(parsed.start_grid)

        # Step 3: Decompose instruction into build steps via LLM
        steps = await planner.decompose(
            parsed.instruction_text,
            parsed.start_grid,
            parsed.speaker,
            structure_hint=structure_info.describe(),
        )
        if not steps:
            return None

        # Step 3b-3d: Deterministic fixes
        steps = patch_chain_references(steps, parsed.start_grid)
        steps = auto_fix_direction(parsed.instruction_text, steps)
        steps = auto_fix_each_end_caps(parsed.instruction_text, steps, parsed.start_grid)
        steps = auto_fix_t_shape_extend(parsed.instruction_text, steps, parsed.start_grid)

        # Verify plan
        verification = verify_plan(
            parsed.instruction_text, steps, len(parsed.start_grid.blocks),
        )
        if verification.has_critical:
            steps = await planner.decompose(
                parsed.instruction_text,
                parsed.start_grid,
                parsed.speaker,
                structure_hint=structure_info.describe(),
                correction_hint=verification.correction_prompt(),
            )
            if not steps:
                return None
            steps = patch_chain_references(steps, parsed.start_grid)
            steps = auto_fix_direction(parsed.instruction_text, steps)
            steps = auto_fix_each_end_caps(parsed.instruction_text, steps, parsed.start_grid)
            steps = auto_fix_t_shape_extend(parsed.instruction_text, steps, parsed.start_grid)

        # Step 4: Resolve uncolored/uncounted steps
        _UNCOLORED = {"uncolored", "unknown", "unspecified", "?"}
        for s in steps:
            if s.color.lower() in _UNCOLORED:
                s.color = heuristic_result.inferred_color or "Purple"
            if isinstance(s.count, str) and s.count.lower() in (
                "uncounted", "unknown", "unspecified", "?"
            ):
                s.count = inferred_count

        # Step 5: Execute steps deterministically
        exec_grid = Grid.from_str(parsed.start_grid.to_str(), config=config)
        executor = SpatialExecutor(exec_grid)
        executor.execute_plan(steps)

        # Step 6: Format response
        response = format_build_response(exec_grid)

        # Step 7: Validate
        is_valid, errors = validate_build_response(response, config)
        if not is_valid:
            logger.warning("Validation failed: %s", errors)
            return None

        return response

    except ExecutionError as exc:
        logger.warning("Execution error in skills pipeline: %s", exc)
        return None
    except Exception as exc:
        logger.warning("Unexpected error in skills pipeline: %s", exc)
        return None


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------

async def _direct_llm_call(
    user_input: str,
    client: AsyncOpenAI,
    model: str,
    state: BWIMState,
) -> str:
    """Fallback: direct LLM call with the spatial system prompt."""
    try:
        messages: list[dict] = [
            {"role": "system", "content": _FALLBACK_SYSTEM_PROMPT},
        ]
        for entry in state.history[-state.max_history:]:
            if entry["type"] == "instruction":
                messages.append({"role": "user", "content": entry["content"]})
            elif entry["type"] == "response":
                messages.append({"role": "assistant", "content": entry["content"]})
            elif entry["type"] == "feedback":
                messages.append({"role": "user", "content": entry["content"]})

        messages.append({"role": "user", "content": user_input})

        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        tracker.record(completion, label="bwim_fallback")
        content = (completion.choices[0].message.content or "").strip()

        if content.startswith("[ASK]"):
            content = "[BUILD]"
        elif not content.startswith("[BUILD]"):
            content = "[BUILD]"

        return content
    except Exception as exc:
        logger.warning("Fallback LLM call failed: %s", exc)
        return "[BUILD]"


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def _add_to_history(state: BWIMState, entry_type: str, content: str) -> None:
    """Add an entry to the conversation history."""
    state.history.append({"type": entry_type, "content": content})
    max_entries = state.max_history * 3
    if len(state.history) > max_entries:
        state.history = state.history[-max_entries:]
