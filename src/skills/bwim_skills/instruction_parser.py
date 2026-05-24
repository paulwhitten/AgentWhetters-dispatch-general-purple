"""Parse green agent messages into structured instruction data."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .grid import Grid, GridConfig


@dataclass
class ParsedInstruction:
    """Structured representation of a green agent message."""
    task_description: str = ""
    speaker: str = ""
    start_structure_str: str = ""
    instruction_text: str = ""
    start_grid: Grid = field(default_factory=Grid)
    is_feedback: bool = False
    feedback_text: str = ""


def parse_green_message(message: str, config: GridConfig | None = None) -> ParsedInstruction:
    """Parse a green agent message into structured fields.

    Handles both instruction messages (with [TASK_DESCRIPTION], [SPEAKER],
    [START_STRUCTURE]) and feedback messages.
    """
    cfg = config or GridConfig()
    result = ParsedInstruction()

    if not message:
        return result

    # Check for feedback messages
    if message.startswith("Feedback:") or message.startswith("A new task is starting"):
        result.is_feedback = True
        result.feedback_text = message
        return result

    # Extract [TASK_DESCRIPTION]
    td_match = re.search(r'\[TASK_DESCRIPTION\]\s*(.*?)(?=\[SPEAKER\]|\[START_STRUCTURE\]|$)', message, re.DOTALL)
    if td_match:
        result.task_description = td_match.group(1).strip()

    # Extract [SPEAKER]
    sp_match = re.search(r'\[SPEAKER\]\s*(.*?)(?=\[START_STRUCTURE\]|\n[A-Z]|$)', message, re.DOTALL)
    if sp_match:
        result.speaker = sp_match.group(1).strip()

    # Extract [START_STRUCTURE]
    # The start structure is on the same line as the tag, contains only block strings (Color,x,y,z;...)
    ss_match = re.search(r'\[START_STRUCTURE\]\s*([^\n]*)', message)
    if ss_match:
        raw = ss_match.group(1).strip()
        # Only treat it as a block string if it looks like Color,x,y,z format
        if raw and re.match(r'^[A-Za-z]+,-?\d+,\d+,-?\d+', raw):
            result.start_structure_str = raw
            result.start_grid = Grid.from_str(raw, config=cfg)
        else:
            result.start_structure_str = ""
            result.start_grid = Grid(config=cfg)
    else:
        result.start_grid = Grid(config=cfg)

    # Extract instruction text (everything after the last structured field)
    # The instruction is typically the last line(s) that don't start with a bracket tag
    lines = message.split("\n")
    instruction_lines = []
    past_tags = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[TASK_DESCRIPTION]") or stripped.startswith("[SPEAKER]") or stripped.startswith("[START_STRUCTURE]"):
            past_tags = True
            # Check if there's content on the same line after [START_STRUCTURE]
            if stripped.startswith("[START_STRUCTURE]"):
                # The next non-empty line after tags is the instruction
                continue
            continue
        if past_tags and stripped:
            instruction_lines.append(stripped)

    # If no structured tags found, treat entire message as instruction
    if not instruction_lines and not result.task_description:
        result.instruction_text = message.strip()
    else:
        result.instruction_text = " ".join(instruction_lines).strip()

    # Handle the common case where instruction is appended after start_structure on same line
    if not result.instruction_text and result.start_structure_str:
        # Look for text after the start structure block string
        after_ss = message.split(result.start_structure_str)
        if len(after_ss) > 1:
            remaining = after_ss[1].strip()
            if remaining:
                result.instruction_text = remaining

    return result
