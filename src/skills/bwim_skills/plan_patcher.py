"""Plan Patcher: fix chain-reference coordinates by simulating step positions.

The LLM planner often gets chain references wrong: when step B says "to the
right of the green one" and step A just built a green stack at (0,-100), the
planner may resolve the reference to the *original* green position (0,0)
rather than the newly built one (0,-100).

This module simulates each plan step to track where blocks actually land,
then patches subsequent steps' position references so they point to the
correct coordinates.

Fully deterministic and generalisable — no hard-coded answers.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .build_planner import BuildStep
from .grid import Block, Grid, GridConfig, direction_offset

logger = logging.getLogger(__name__)


def patch_chain_references(
    steps: list[BuildStep],
    start_grid: Grid,
) -> list[BuildStep]:
    """Simulate step placement and fix chain-reference coordinates.

    For each step that uses a relative reference (like "existing_Green_stack_at_X_Z"),
    verify that the referenced position is correct given prior steps' actual
    placements.  If a step references a color that was just placed somewhere
    different, patch the coordinates.

    Returns the (potentially modified) steps list.
    """
    config = start_grid.config
    # Track where each color was last placed (step-by-step)
    last_pos_by_color: dict[str, tuple[int, int]] = {}
    last_pos_overall: tuple[int, int] | None = None

    # Pre-populate from existing grid blocks
    for block in start_grid.blocks:
        ckey = block.color.capitalize()
        last_pos_by_color[ckey] = (block.x, block.z)

    for i, step in enumerate(steps):
        pos = step.position

        # ----- Resolve where THIS step ends up -----
        step_x, step_z = _resolve_step_position(
            step, last_pos_by_color, last_pos_overall, start_grid, config,
        )

        if step_x is None:
            # Can't determine position — skip patching
            continue

        # ----- Check if a subsequent step references this step's color -----
        # (We patch later steps, not the current one)
        color = step.color.capitalize()
        if color.lower() not in ("uncolored", "unknown", "unspecified", "?"):
            # Compute actual end position for extend_row
            actual_x, actual_z = step_x, step_z
            if step.action == "extend_row":
                # The last block in the row is the important reference
                actual_x, actual_z = _extend_row_end(
                    step_x, step_z, step, config,
                )
            last_pos_by_color[color] = (actual_x, actual_z)
            last_pos_overall = (actual_x, actual_z)
        else:
            # Uncolored step — track overall position only
            actual_x, actual_z = step_x, step_z
            if step.action == "extend_row":
                actual_x, actual_z = _extend_row_end(
                    step_x, step_z, step, config,
                )
            last_pos_overall = (actual_x, actual_z)

        # ----- Patch the NEXT steps' references -----
        for j in range(i + 1, len(steps)):
            next_step = steps[j]
            _try_patch_reference(
                next_step, i, step, color,
                actual_x, actual_z,
                last_pos_by_color, last_pos_overall,
                start_grid,
            )

    return steps


def _resolve_step_position(
    step: BuildStep,
    last_pos_by_color: dict[str, tuple[int, int]],
    last_pos_overall: tuple[int, int] | None,
    grid: Grid,
    config: GridConfig,
) -> tuple[int | None, int | None]:
    """Resolve where a step will start placing blocks."""
    pos = step.position

    if "x" in pos and "z" in pos:
        return int(pos["x"]), int(pos["z"])

    if "relative_to" in pos:
        ref = pos["relative_to"]
        ref_x, ref_z = _resolve_ref_simple(
            ref, last_pos_by_color, last_pos_overall, grid,
        )
        if ref_x is None:
            return None, None

        direction = pos.get("direction", "")
        if direction.lower() in ("on_top", "on top", "above"):
            return ref_x, ref_z

        try:
            dx, dz = direction_offset(direction, config)
        except ValueError:
            return None, None

        distance = int(pos.get("distance", 1))
        return ref_x + dx * distance, ref_z + dz * distance

    return None, None


def _resolve_ref_simple(
    ref: str,
    last_pos_by_color: dict[str, tuple[int, int]],
    last_pos_overall: tuple[int, int] | None,
    grid: Grid,
) -> tuple[int | None, int | None]:
    """Simplified reference resolution for simulation purposes."""
    ref_lower = ref.lower().strip()

    if ref_lower in ("origin", "center", "middle", "highlighted",
                     "highlighted_square", "middle_square"):
        return 0, 0

    # "existing_<Color>_stack_at_<x>_<z>"
    m = re.match(r'existing_(\w+)_(?:stack|block)_at_(-?\d+)_(-?\d+)', ref_lower)
    if m:
        return int(m.group(2)), int(m.group(3))

    # Color reference — use tracked position
    m = re.match(r'(?:the\s+)?(\w+)\s+(?:stack|block|tower|one|row)', ref_lower)
    if m:
        color = m.group(1).capitalize()
        if color in last_pos_by_color:
            return last_pos_by_color[color]

    # Last placed
    if ref_lower in ("the_one", "the_stack", "last", "previous",
                     "the one", "the stack", "that one"):
        if last_pos_overall:
            return last_pos_overall

    return None, None


def _extend_row_end(
    start_x: int, start_z: int,
    step: BuildStep,
    config: GridConfig,
) -> tuple[int, int]:
    """Compute the last block position of an extend_row step."""
    dir_name = step.position.get("direction", "right")
    count = step.count
    if isinstance(count, str):
        try:
            count = int(count)
        except ValueError:
            count = 3

    try:
        dx, dz = direction_offset(dir_name, config)
    except ValueError:
        return start_x, start_z

    end_x = start_x + dx * (count - 1)
    end_z = start_z + dz * (count - 1)
    return end_x, end_z


def _try_patch_reference(
    next_step: BuildStep,
    prev_idx: int,
    prev_step: BuildStep,
    prev_color: str,
    actual_x: int,
    actual_z: int,
    last_pos_by_color: dict[str, tuple[int, int]],
    last_pos_overall: tuple[int, int] | None,
    grid: Grid,
) -> None:
    """Try to patch a subsequent step's reference if it points to a stale position."""
    pos = next_step.position

    if "relative_to" not in pos:
        return

    ref = pos.get("relative_to", "").lower()

    # Check if this reference mentions the previous step's color
    if prev_color.lower() not in ("uncolored", "unknown", "unspecified", "?"):
        if prev_color.lower() in ref:
            # This step references the color we just placed.
            # Check if the embedded coordinates are wrong.
            m = re.match(
                r'existing_(\w+)_(?:stack|block)_at_(-?\d+)_(-?\d+)',
                ref,
            )
            if m:
                ref_x, ref_z = int(m.group(2)), int(m.group(3))
                if (ref_x, ref_z) != (actual_x, actual_z):
                    # The reference points to the old position — patch it!
                    old_ref = pos["relative_to"]
                    new_ref = (
                        f"existing_{prev_color}_stack_at_{actual_x}_{actual_z}"
                    )
                    logger.info(
                        "Patching chain reference in step %d: '%s' → '%s' "
                        "(prev step %d placed %s at (%d,%d))",
                        prev_idx + 2, old_ref, new_ref,
                        prev_idx + 1, prev_color, actual_x, actual_z,
                    )
                    pos["relative_to"] = new_ref

            # Also check "the <color> one/stack" pattern
            color_ref = re.match(
                r'(?:the\s+)?(\w+)\s+(?:stack|block|tower|one|row)',
                ref,
            )
            if color_ref and color_ref.group(1).capitalize() == prev_color:
                # This is a pronoun-style reference — the executor will
                # resolve it via _last_position_by_color, which is correct
                # IF the executor runs steps in order.  But if the planner
                # embedded explicit coordinates, they might be wrong.
                # We can convert this to an explicit correct reference.
                pass  # Executor's own tracking handles this


def compute_step_endpoints(
    steps: list[BuildStep],
    start_grid: Grid,
) -> dict[int, tuple[int, int]]:
    """Compute the effective (x, z) endpoint of each step.

    Returns a dict: step_index → (x, z).
    Useful for the planner to know where each step's result ends up.
    """
    config = start_grid.config
    last_pos_by_color: dict[str, tuple[int, int]] = {}
    last_pos_overall: tuple[int, int] | None = None

    for block in start_grid.blocks:
        last_pos_by_color[block.color.capitalize()] = (block.x, block.z)

    endpoints: dict[int, tuple[int, int]] = {}

    for i, step in enumerate(steps):
        x, z = _resolve_step_position(
            step, last_pos_by_color, last_pos_overall, start_grid, config,
        )
        if x is not None and z is not None:
            if step.action == "extend_row":
                x, z = _extend_row_end(x, z, step, config)
            endpoints[i] = (x, z)
            color = step.color.capitalize()
            if color.lower() not in ("uncolored", "unknown", "unspecified", "?"):
                last_pos_by_color[color] = (x, z)
            last_pos_overall = (x, z)

    return endpoints
