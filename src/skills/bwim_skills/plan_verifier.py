"""Plan Verifier: post-plan checks that catch common LLM planning errors.

After the LLM planner generates build steps, this module validates them
against the original instruction text to catch systematic errors:

1. Direction consistency — does the plan direction match the instruction?
2. "Each/every" expansion — did the plan expand iterative phrases?
3. No-stacking guard — did horizontal moves become accidental stacks?
4. Block count sanity — does the total block count look reasonable?

Generalises to any instruction — no hard-coded answers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .build_planner import BuildStep
from .grid import Grid
from .structure_analyzer import analyze_structure

logger = logging.getLogger(__name__)

# Direction words → expected plan direction
_LEFT_WORDS = re.compile(
    r'\b(?:going\s+(?:to\s+)?(?:the\s+)?left|towards?\s+(?:the\s+)?left|to\s+the\s+left'
    r'|going\s+left|leftward)\b',
    re.IGNORECASE,
)
_RIGHT_WORDS = re.compile(
    r'\b(?:going\s+(?:to\s+)?(?:the\s+)?right|towards?\s+(?:the\s+)?right|to\s+the\s+right'
    r'|going\s+right|rightward)\b',
    re.IGNORECASE,
)
_FRONT_WORDS = re.compile(
    r'\b(?:going\s+(?:to\s+)?(?:the\s+)?front|towards?\s+(?:the\s+)?(?:front|bottom)'
    r'|going\s+forward|going\s+front)\b',
    re.IGNORECASE,
)
_BACK_WORDS = re.compile(
    r'\b(?:going\s+(?:to\s+)?(?:the\s+)?back|towards?\s+(?:the\s+)?(?:back|top)'
    r'|going\s+back(?:ward)?)\b',
    re.IGNORECASE,
)

# Phrases that signal "do for each"
_EACH_PATTERN = re.compile(
    r'\b(?:each|every|both|all\s+(?:of\s+)?(?:the|them))\b',
    re.IGNORECASE,
)

# "In front of" / "to the left of" etc. should be HORIZONTAL, not stacking
_HORIZONTAL_REL = re.compile(
    r'\b(?:in\s+front\s+of|to\s+the\s+(?:left|right)\s+of|behind|'
    r'directly\s+(?:in\s+front|to\s+the\s+(?:left|right)|behind))\b',
    re.IGNORECASE,
)

# "On top of" — explicit stacking
_STACK_REL = re.compile(
    r'\b(?:on\s+top\s+of|stack(?:ed)?\s+on)\b',
    re.IGNORECASE,
)

# Number words for count extraction
_COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_COUNT_PATTERN = re.compile(
    r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+'
    r'(?:\w+\s+)?(?:blocks?|stack|tower|row)',
    re.IGNORECASE,
)


@dataclass
class VerificationIssue:
    """A single issue found during verification."""
    category: str        # "direction", "each_expansion", "stacking", "count"
    severity: str        # "critical", "warning"
    step_index: int      # which step has the issue (-1 for global)
    description: str
    suggested_fix: str   # instruction for the LLM re-prompt


@dataclass
class VerificationResult:
    """Result of plan verification."""
    issues: list[VerificationIssue] = field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        return any(i.severity == "critical" for i in self.issues)

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    def correction_prompt(self) -> str:
        """Build a correction prompt for re-planning."""
        if not self.issues:
            return ""
        lines = ["CORRECTIONS NEEDED (fix these errors in your plan):"]
        for issue in self.issues:
            lines.append(f"- [{issue.category}] {issue.description}. FIX: {issue.suggested_fix}")
        return "\n".join(lines)


def verify_plan(
    instruction: str,
    steps: list[BuildStep],
    existing_block_count: int = 0,
) -> VerificationResult:
    """Verify a build plan against the instruction text.

    Returns a VerificationResult with any issues found.
    """
    result = VerificationResult()

    _check_direction_consistency(instruction, steps, result)
    _check_each_expansion(instruction, steps, result)
    _check_stacking_vs_horizontal(instruction, steps, result)
    _check_block_counts(instruction, steps, existing_block_count, result)

    if result.has_issues:
        logger.info(
            "Plan verification found %d issue(s) (%d critical)",
            len(result.issues),
            sum(1 for i in result.issues if i.severity == "critical"),
        )
        for issue in result.issues:
            logger.info("  [%s] %s: %s", issue.severity, issue.category, issue.description)

    return result


def _check_direction_consistency(
    instruction: str,
    steps: list[BuildStep],
    result: VerificationResult,
) -> None:
    """Verify that extend_row directions match the instruction text."""
    # Find direction words in instruction
    has_left = bool(_LEFT_WORDS.search(instruction))
    has_right = bool(_RIGHT_WORDS.search(instruction))
    has_front = bool(_FRONT_WORDS.search(instruction))
    has_back = bool(_BACK_WORDS.search(instruction))

    for i, step in enumerate(steps):
        if step.action != "extend_row":
            continue
        plan_dir = step.position.get("direction", "").lower()
        if not plan_dir:
            continue

        # Check for contradictions
        if has_left and not has_right and plan_dir == "right":
            result.issues.append(VerificationIssue(
                category="direction",
                severity="critical",
                step_index=i,
                description=(
                    f"Step {i+1} has direction='right' but instruction says "
                    f"'going to the left'. Direction is REVERSED."
                ),
                suggested_fix=(
                    f"Change step {i+1} direction from 'right' to 'left'. "
                    f"'Going to the left' means decreasing x."
                ),
            ))

        if has_right and not has_left and plan_dir == "left":
            result.issues.append(VerificationIssue(
                category="direction",
                severity="critical",
                step_index=i,
                description=(
                    f"Step {i+1} has direction='left' but instruction says "
                    f"'going to the right'. Direction is REVERSED."
                ),
                suggested_fix=(
                    f"Change step {i+1} direction from 'left' to 'right'. "
                    f"'Going to the right' means increasing x."
                ),
            ))

        if has_front and not has_back and plan_dir in ("back", "behind", "backward"):
            result.issues.append(VerificationIssue(
                category="direction",
                severity="critical",
                step_index=i,
                description=(
                    f"Step {i+1} has direction='{plan_dir}' but instruction says "
                    f"'going to the front'. Direction is REVERSED."
                ),
                suggested_fix=(
                    f"Change step {i+1} direction to 'front'. "
                    f"'Going to the front' means increasing z."
                ),
            ))

        if has_back and not has_front and plan_dir in ("front", "forward"):
            result.issues.append(VerificationIssue(
                category="direction",
                severity="critical",
                step_index=i,
                description=(
                    f"Step {i+1} has direction='{plan_dir}' but instruction says "
                    f"'going to the back'. Direction is REVERSED."
                ),
                suggested_fix=(
                    f"Change step {i+1} direction to 'behind'. "
                    f"'Going to the back' means decreasing z."
                ),
            ))


def _check_each_expansion(
    instruction: str,
    steps: list[BuildStep],
    result: VerificationResult,
) -> None:
    """Check that 'each/every/both' phrases are expanded into multiple steps."""
    each_matches = list(_EACH_PATTERN.finditer(instruction))
    if not each_matches:
        return

    # Look for "in front of each X" or "on top of each X" patterns
    # These require one action per referenced block
    for m in each_matches:
        context_start = max(0, m.start() - 50)
        context_end = min(len(instruction), m.end() + 50)
        context = instruction[context_start:context_end]

        # Check if "each" is part of a positioning phrase
        each_pos = re.search(
            r'(?:in\s+front\s+of|on\s+top\s+of|to\s+the\s+(?:left|right)\s+of|behind)'
            r'\s+each\b',
            context, re.IGNORECASE,
        )
        if not each_pos:
            # Also match "one X to each arm/end/corner"
            each_pos = re.search(
                r'\bone\b.*?\bto\s+each\b|\bone\b.*?\bon\s+each\b'
                r'|\beach\s+(?:end|arm|corner|side)\b'
                r'|\beach\s+of\s+(?:the|them)\b'
                r'|\btop\s+of\s+each\b',
                context, re.IGNORECASE,
            )

        if each_pos:
            # Count how many "place" or "stack" steps exist for this part
            # The instruction with "each" should produce multiple steps
            # We can't determine the exact expected count, but check for
            # suspiciously few steps compared to the pattern
            place_steps = [
                s for s in steps
                if s.action in ("place", "place_relative", "stack")
                and s.count in (1, "1")
            ]
            # If "each" appears but only 1 place step exists, that's suspicious
            # (but not always wrong — need context)
            if len(place_steps) < 2 and len(steps) < 4:
                result.issues.append(VerificationIssue(
                    category="each_expansion",
                    severity="warning",
                    step_index=-1,
                    description=(
                        f"Instruction says '{m.group()}' which suggests "
                        f"multiple placements, but plan has fewer steps than expected."
                    ),
                    suggested_fix=(
                        f"Expand the '{m.group()}' phrase into separate steps — "
                        f"one step for each referenced block/position."
                    ),
                ))


def _check_stacking_vs_horizontal(
    instruction: str,
    steps: list[BuildStep],
    result: VerificationResult,
) -> None:
    """Check that horizontal positioning phrases don't become stacks.

    If the instruction says "in front of X" the step should NOT place at
    the same (x,z) as X (which would stack on top).
    """
    # Find all horizontal relationship phrases
    horiz_matches = list(_HORIZONTAL_REL.finditer(instruction))
    if not horiz_matches:
        return

    # Check if any steps share the same (x,z) with a previous step
    # when the instruction has horizontal relationship words
    positions_so_far: dict[tuple[int, int], int] = {}  # (x,z) → step index
    for i, step in enumerate(steps):
        pos = step.position
        if "x" in pos and "z" in pos:
            xz = (int(pos["x"]), int(pos["z"]))
            if xz in positions_so_far and step.action in ("place", "stack", "place_relative"):
                # Same position as a previous step — could be accidental stacking
                prev_i = positions_so_far[xz]
                prev_step = steps[prev_i]
                # Only flag if the instruction has horizontal words near this step's context
                if prev_step.action in ("place", "stack"):
                    result.issues.append(VerificationIssue(
                        category="stacking",
                        severity="warning",
                        step_index=i,
                        description=(
                            f"Step {i+1} ({step.action} {step.color}) places at "
                            f"({xz[0]},{xz[1]}), same as step {prev_i+1} ({prev_step.action} "
                            f"{prev_step.color}). The instruction uses horizontal "
                            f"positioning ('{horiz_matches[0].group()}') — this might be "
                            f"an accidental stack instead of a horizontal placement."
                        ),
                        suggested_fix=(
                            f"The block in step {i+1} should likely be placed at a "
                            f"different (x,z) from step {prev_i+1}. 'In front of' means "
                            f"+z, 'behind' means -z, 'to the right' means +x, "
                            f"'to the left' means -x."
                        ),
                    ))
            positions_so_far[xz] = i


def _check_block_counts(
    instruction: str,
    steps: list[BuildStep],
    existing_block_count: int,
    result: VerificationResult,
) -> None:
    """Check that the total new block count is reasonable."""
    # Extract counts from instruction
    expected_counts = []
    for m in _COUNT_PATTERN.finditer(instruction):
        val_str = m.group(1).lower()
        if val_str.isdigit():
            expected_counts.append(int(val_str))
        elif val_str in _COUNT_WORDS:
            expected_counts.append(_COUNT_WORDS[val_str])

    if not expected_counts:
        return

    # Sum plan blocks
    plan_total = 0
    for step in steps:
        count = step.count
        if isinstance(count, str):
            if count.lower() in ("uncounted", "unknown", "unspecified", "?"):
                continue
            try:
                count = int(count)
            except ValueError:
                continue
        plan_total += count

    expected_total = sum(expected_counts)

    # Allow some tolerance (the instruction may have implicit counts)
    if plan_total < expected_total - 1:
        result.issues.append(VerificationIssue(
            category="count",
            severity="warning",
            step_index=-1,
            description=(
                f"Plan produces {plan_total} new blocks, but instruction "
                f"mentions counts totalling {expected_total}."
            ),
            suggested_fix=(
                f"Review block counts. The instruction specifies: "
                f"{', '.join(str(c) for c in expected_counts)}. "
                f"Ensure the plan accounts for all of them."
            ),
        ))


def auto_fix_direction(
    instruction: str,
    steps: list[BuildStep],
) -> list[BuildStep]:
    """Automatically fix direction errors in extend_row steps.

    This is a deterministic correction: if the instruction says "going
    left" and the plan says "right", just flip it.  No LLM needed.

    BUG ADDRESSED (run 002439, rounds with L-shape or row extensions):
    gpt-4o-mini frequently reverses left/right in extend_row steps.
    The instruction text is ground truth, so we regex-match direction
    words from the instruction and override the plan when they conflict.

    Only fires when the instruction contains EXACTLY ONE of left/right.
    If both appear (for example "to the right of the left block") we
    cannot safely pick one, so we leave the plan unchanged.

    Called in the pipeline AFTER patch_chain_references and BEFORE
    auto_fix_each_end_caps.
    """
    has_left = bool(_LEFT_WORDS.search(instruction))
    has_right = bool(_RIGHT_WORDS.search(instruction))

    for step in steps:
        if step.action != "extend_row":
            continue
        plan_dir = step.position.get("direction", "").lower()

        if has_left and not has_right and plan_dir == "right":
            logger.info("Auto-fixing direction: right → left for extend_row step")
            step.position["direction"] = "left"

        elif has_right and not has_left and plan_dir == "left":
            logger.info("Auto-fixing direction: left → right for extend_row step")
            step.position["direction"] = "right"

    return steps


# Pattern for "each end" / "both ends" phrases
_EACH_END_PATTERN = re.compile(
    r'\b(?:each|both)\s+ends?\b',
    re.IGNORECASE,
)

# Fallback pattern to parse extension from instruction text
_EXTEND_INSTR_PATTERN = re.compile(
    r'(?:extend|add(?:ing)?)\s+.*?'
    r'(\d+|one|two|three|four|five|six|seven|eight|nine)\s+'
    r'(?:\w+\s+)?blocks?\s+'
    r'(?:to\s+its\s+|going\s+)?'
    r'(right|left|front|behind|forward|back)',
    re.IGNORECASE,
)
_WORD_TO_INT = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
}


def auto_fix_each_end_caps(
    instruction: str,
    steps: list[BuildStep],
    start_grid: Grid,
) -> list[BuildStep]:
    """Fix 'each end' cap positions after extend_row steps.

    BUG ADDRESSED (run 002439, "each end" / "both ends" rounds):
    When the instruction says "stack on each end", the LLM places caps
    at the OLD row endpoints.  After an extend_row the row is longer,
    so one cap should be at the NEW endpoint, not the old one.

    EXAMPLE:
      Instruction: "Extend the red row by 2 blocks to the right,
                     then stack 1 blue on each end."
      Start grid:   Red at x=-100, 0, 100  (row along x)
      LLM plan:     extend_row Red count=2 dir=right (new end x=300)
                    stack Blue at x=-100   (left end -- correct)
                    stack Blue at x=100    (old right end -- WRONG)
      Fixed:        stack Blue at x=300    (new right end -- correct)

    MECHANISM:
    1. Bail out if instruction lacks "each end" / "both ends".
    2. Find the extend_row step (or parse direction/count from the
       instruction text as a fallback when the LLM used place instead).
    3. Compute old endpoint and new endpoint from the start grid.
    4. Scan subsequent stack/place steps; any that sit on the old
       endpoint (which is now interior) get relocated to the new end.

    Fully deterministic, no LLM needed.  Called AFTER auto_fix_direction.
    """
    if not _EACH_END_PATTERN.search(instruction):
        return steps

    # Find extend_row step
    extend_idx = None
    extend_step = None
    for i, step in enumerate(steps):
        if step.action == "extend_row":
            extend_idx = i
            extend_step = step
            break

    if extend_step is None:
        # Fallback: parse extension from instruction text
        ext_match = _EXTEND_INSTR_PATTERN.search(instruction)
        if not ext_match:
            return steps
        count_str = ext_match.group(1).lower()
        direction = ext_match.group(2).lower()
        count = _WORD_TO_INT.get(count_str)
        if count is None:
            try:
                count = int(count_str)
            except ValueError:
                return steps
        if count <= 0:
            return steps
        # Use fallback path with inferred direction/count
        return _fix_caps_for_extension(
            instruction, steps, start_grid, direction, count,
            cap_search_start=0,
        )

    # Get extension direction and count
    pos = extend_step.position
    direction = pos.get("direction", "right").lower()

    count = extend_step.count
    if isinstance(count, str):
        try:
            count = int(count)
        except ValueError:
            return steps
    count = int(count)

    if count <= 0:
        return steps

    return _fix_caps_for_extension(
        instruction, steps, start_grid, direction, count,
        cap_search_start=extend_idx + 1,
    )


def _fix_caps_for_extension(
    instruction: str,
    steps: list[BuildStep],
    start_grid: Grid,
    direction: str,
    count: int,
    cap_search_start: int,
) -> list[BuildStep]:
    """Shared logic: fix cap positions given extension direction and count."""
    config = start_grid.config
    grid_step = config.grid_step

    # Determine which axis changes with extension
    horizontal = direction in ("left", "right")

    # Find ground-level blocks that form the row
    ground_y = config.valid_y[0]
    # Find ground-level blocks that form the row
    ground_y = config.valid_y[0]
    ground_blocks = [b for b in start_grid.blocks if b.y == ground_y]
    if not ground_blocks:
        return steps

    if horizontal:
        # Row runs along x-axis; find the common z value
        z_values = [b.z for b in ground_blocks]
        row_z = max(set(z_values), key=z_values.count)
        row_xs = sorted(b.x for b in ground_blocks if b.z == row_z)
        if not row_xs:
            return steps

        old_min_x = min(row_xs)
        old_max_x = max(row_xs)

        if direction == "right":
            new_max_x = old_max_x + count * grid_step
            old_end = old_max_x
            new_end = new_max_x
        else:
            new_min_x = old_min_x - count * grid_step
            old_end = old_min_x
            new_end = new_min_x

        # Fix subsequent cap steps that land on the old endpoint
        for i, step in enumerate(steps):
            if i < cap_search_start:
                continue
            if step.action not in ("stack", "place", "place_relative"):
                continue

            sx = step.position.get("x")
            sz = step.position.get("z")
            if sx is None or sz is None:
                continue
            try:
                sx, sz = int(sx), int(sz)
            except (ValueError, TypeError):
                continue

            if sz != row_z:
                continue

            # Cap at the old endpoint that is now interior to the row
            if sx == old_end and sx != new_end:
                logger.info(
                    "Auto-fixing each-end cap: x=%d → x=%d "
                    "(new %s end after row extension)",
                    sx, new_end,
                    "right" if direction == "right" else "left",
                )
                step.position["x"] = new_end

    else:
        # Row runs along z-axis (front/back extension)
        x_values = [b.x for b in ground_blocks]
        row_x = max(set(x_values), key=x_values.count)
        row_zs = sorted(b.z for b in ground_blocks if b.x == row_x)
        if not row_zs:
            return steps

        old_min_z = min(row_zs)
        old_max_z = max(row_zs)

        if direction in ("front", "forward", "in_front"):
            new_max_z = old_max_z + count * grid_step
            old_end = old_max_z
            new_end = new_max_z
        else:
            new_min_z = old_min_z - count * grid_step
            old_end = old_min_z
            new_end = new_min_z

        for i, step in enumerate(steps):
            if i < cap_search_start:
                continue
            if step.action not in ("stack", "place", "place_relative"):
                continue

            sx = step.position.get("x")
            sz = step.position.get("z")
            if sx is None or sz is None:
                continue
            try:
                sx, sz = int(sx), int(sz)
            except (ValueError, TypeError):
                continue

            if sx != row_x:
                continue

            if sz == old_end and sz != new_end:
                logger.info(
                    "Auto-fixing each-end cap: z=%d → z=%d "
                    "(new %s end after row extension)",
                    sz, new_end,
                    "front" if direction in ("front", "forward", "in_front") else "back",
                )
                step.position["z"] = new_end

    return steps


# ── T-shape direction keyword mapping ──
_DIRECTION_MAP = {
    ("x", True): "right",
    ("x", False): "left",
    ("z", True): "front",
    ("z", False): "behind",
}

# Triggers for T-shape instructions
_T_SHAPE_PATTERN = re.compile(r'\bt[- ]?shape\b', re.IGNORECASE)
_EXTEND_PATTERN = re.compile(r'\bextend\b', re.IGNORECASE)


def auto_fix_t_shape_extend(
    instruction: str,
    steps: list[BuildStep],
    start_grid: Grid,
) -> list[BuildStep]:
    """Fix extend_row direction and action for T-shape extensions.

    BUG ADDRESSED (runs 013543-014444, 4 rounds each worth -10 = -40 pts):
    When instructed to "extend the longer base" of a T-shape, gpt-4o-mini
    makes two distinct errors:

      ERROR A  direction="on_top" or wrong axis direction.
        The stem runs along the z-axis but the LLM writes direction="on_top"
        or direction="right".  The executor then stacks vertically or extends
        along the wrong axis.

      ERROR B  action="stack" instead of "extend_row" at/near the stem tip.
        Instead of extend_row, the LLM emits a stack step at the stem tip
        coordinates.  The executor stacks vertically on top of the tip block
        rather than extending the stem outward.

    MECHANISM:
    1. Bail out unless the instruction mentions both "t-shape" and "extend".
    2. Run the structure analyzer to identify crossbar, stem, and junction.
    3. Determine which segment to extend ("longer base" maps to the longer
       of crossbar vs stem; default is stem).
    4. Compute the correct extension direction from geometry:
       - Find the junction (shared position between crossbar and stem).
       - The "base" is the segment endpoint farthest from the junction.
       - The extension direction points away from the junction past the base.
    5. If there IS an extend_row step, fix its direction and start position.
    6. If there is NO extend_row step (Error B), search for a stack/place
       step at the stem tip or one grid step past it.  Convert it to
       extend_row with the computed direction and a start position one step
       past the base.

    NOTE on interaction with executor's same-color skip-forward logic:
    If the LLM emits extend_row starting AT the existing stem tip (for example
    extend_row Green count=2 at (0,200) when Green already occupies (0,200)),
    the executor's _handle_extend_row detects the same-color overlap and
    advances the start by one grid step before placing.  That means blocks
    land at z=300 and z=400 rather than stacking at z=200.  See the detailed
    comment in spatial_executor.py _handle_extend_row.

    Fully deterministic, no LLM needed.  Called AFTER auto_fix_each_end_caps.
    """
    if not _T_SHAPE_PATTERN.search(instruction):
        return steps
    if not _EXTEND_PATTERN.search(instruction):
        return steps

    # Find extend_row step — or a stack/place at the stem tip (fallback)
    extend_step = None
    extend_idx = None
    for i, step in enumerate(steps):
        if step.action == "extend_row":
            extend_step = step
            extend_idx = i
            break

    # Analyze the starting structure (needed for both paths)
    analysis = analyze_structure(start_grid)

    # Find the T-shape
    t_shape = None
    for shape in analysis.shapes:
        if shape.shape_type == "T":
            t_shape = shape
            break

    if t_shape is None or t_shape.stem is None:
        return steps

    stem = t_shape.stem
    crossbar = t_shape.crossbar

    # Determine which part to extend
    # "longer base" / "longer part" → the longer of crossbar vs stem
    longer = t_shape.longer_base or "stem"
    inst_lower = instruction.lower()
    if "longer" in inst_lower:
        if longer == "crossbar":
            extend_line = crossbar
        else:
            extend_line = stem
    else:
        # Default to stem extension
        extend_line = stem

    if extend_line is None:
        return steps

    # Find the junction position (shared by crossbar and stem)
    crossbar_set = set(crossbar.positions) if crossbar else set()
    stem_set = set(stem.positions) if stem else set()
    junction_set = crossbar_set & stem_set
    if not junction_set:
        return steps
    junction = junction_set.pop()

    # Determine the axis the extend line runs along and the base position
    if extend_line.direction == "horizontal":
        axis = "x"
        # Base = the end farthest from junction
        d_start = abs(extend_line.start[0] - junction[0])
        d_end = abs(extend_line.end[0] - junction[0])
        base = extend_line.end if d_end >= d_start else extend_line.start
        increasing = base[0] > junction[0]
    else:
        axis = "z"
        d_start = abs(extend_line.start[1] - junction[1])
        d_end = abs(extend_line.end[1] - junction[1])
        base = extend_line.end if d_end >= d_start else extend_line.start
        increasing = base[1] > junction[1]

    correct_direction = _DIRECTION_MAP[(axis, increasing)]

    # ── FALLBACK PATH (Error B): no extend_row step found ──
    #
    # The LLM produced stack/place at the stem tip instead of extend_row.
    # We search for any stack/place/place_relative whose (x,z) matches
    # either the base (stem tip) or one grid step past it in the correct
    # extension direction.  When found we rewrite it to extend_row with:
    #   - action = "extend_row"
    #   - direction = the computed correct_direction
    #   - start position = one grid step past the base
    #
    # Only the FIRST matching step is converted (we break after it).
    # If the LLM produced multiple stack steps at the tip, only one gets
    # fixed.  That is acceptable because the remaining step(s) will stack
    # on a now-empty position (the tip is still occupied by the original
    # structure, so the stack increments y, producing a visible error that
    # the verifier catches on re-plan).  In practice gpt-4o-mini produces
    # a SINGLE stack step with count >= 2 for T-shape extending.
    if extend_step is None:
        grid_step = start_grid.config.grid_step
        base_x, base_z = base[0], base[1]
        for i, step in enumerate(steps):
            if step.action not in ("stack", "place", "place_relative"):
                continue
            sx = step.position.get("x")
            sz = step.position.get("z")
            if sx is None or sz is None:
                continue
            try:
                sx, sz = int(sx), int(sz)
            except (ValueError, TypeError):
                continue
            # Match if the step is at the base (stem tip) or one step
            # past the base in the correct extension direction
            at_base = (sx == base_x and sz == base_z)
            one_past = False
            if axis == "x":
                one_past = (sz == base_z and sx == base_x + (grid_step if increasing else -grid_step))
            else:
                one_past = (sx == base_x and sz == base_z + (grid_step if increasing else -grid_step))

            if at_base or one_past:
                logger.info(
                    "Auto-fixing T-shape: converting %s at (%d,%d) → "
                    "extend_row direction='%s' (stem tip fallback)",
                    step.action, sx, sz, correct_direction,
                )
                step.action = "extend_row"
                step.position["direction"] = correct_direction
                # Start position: one step past the base
                if axis == "x":
                    step.position["x"] = base_x + (grid_step if increasing else -grid_step)
                    step.position["z"] = base_z
                else:
                    step.position["x"] = base_x
                    step.position["z"] = base_z + (grid_step if increasing else -grid_step)
                extend_step = step
                extend_idx = i
                break

        if extend_step is None:
            return steps

    # Check current direction
    current_dir = extend_step.position.get("direction", "").lower()

    # Fix invalid or wrong directions
    invalid_dirs = {"on_top", "on top", "above", "up", ""}
    if current_dir in invalid_dirs or current_dir != correct_direction:
        logger.info(
            "Auto-fixing T-shape extend direction: '%s' → '%s' "
            "(stem runs along %s-axis, base at %s, %s)",
            current_dir, correct_direction,
            axis, base, "increasing" if increasing else "decreasing",
        )
        extend_step.position["direction"] = correct_direction

        # Also fix the starting position for the extend_row
        # The extension should start past the base
        grid_step = start_grid.config.grid_step
        if axis == "x":
            start_x = base[0] + (grid_step if increasing else -grid_step)
            start_z = base[1]
            extend_step.position["x"] = start_x
            extend_step.position["z"] = start_z
        else:
            start_x = base[0]
            start_z = base[1] + (grid_step if increasing else -grid_step)
            extend_step.position["x"] = start_x
            extend_step.position["z"] = start_z

    return steps
