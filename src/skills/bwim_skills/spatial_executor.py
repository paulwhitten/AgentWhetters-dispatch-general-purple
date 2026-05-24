"""Spatial Executor: deterministic execution of build plan steps on the Grid model."""

from __future__ import annotations

import logging
import re
from typing import Optional

from .grid import Block, Grid, GridConfig, direction_offset, corner_position
from .build_planner import BuildStep

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Raised when a build step cannot be executed."""
    pass


class SpatialExecutor:
    """Executes build plan steps deterministically on a Grid."""

    def __init__(self, grid: Grid):
        self.grid = grid
        # Track the (x, z) of the last step executed, so chain references
        # like "the X one" or "directly to the right of the red one" resolve
        # to the most-recently placed position.
        self._last_step_position: tuple[int, int] | None = None
        # Track last-placed position per color for "the <color> one" refs
        self._last_position_by_color: dict[str, tuple[int, int]] = {}

    def execute_plan(self, steps: list[BuildStep]) -> Grid:
        """Execute all steps in order. Returns the final grid state."""
        for i, step in enumerate(steps):
            try:
                self.execute_step(step)
                logger.info(
                    "Step %d/%d executed: %s %s %s → %d blocks on grid",
                    i + 1, len(steps), step.action, step.color, step.position,
                    len(self.grid.blocks),
                )
            except ExecutionError as exc:
                logger.warning("Step %d failed: %s", i + 1, exc)
                raise
        return self.grid

    def _record_step_position(self, x: int, z: int, color: str) -> None:
        """Record the position of the most recently placed block/stack."""
        self._last_step_position = (x, z)
        self._last_position_by_color[color.capitalize()] = (x, z)

    def execute_step(self, step: BuildStep) -> None:
        """Execute a single build step, modifying the grid in place."""
        action = step.action.lower().replace("-", "_").replace(" ", "_")

        handlers = {
            "stack": self._handle_stack,
            "place": self._handle_place,
            "place_relative": self._handle_place_relative,
            "extend_row": self._handle_extend_row,
            "place_at_corners": self._handle_place_at_corners,
            "place_along_edge": self._handle_place_along_edge,
        }

        handler = handlers.get(action)
        if handler is None:
            raise ExecutionError(f"Unknown action: {step.action}")

        handler(step)

    def _resolve_count(self, count: int | str, hint_x: int | None = None, hint_z: int | None = None) -> int:
        """Resolve count, defaulting Uncounted to a reasonable guess.
        
        When a count is unspecified, try to match the height of an adjacent stack
        (if hint_x/hint_z are provided) or the tallest stack on the grid. Otherwise 3.
        """
        if isinstance(count, int):
            return count
        if isinstance(count, str):
            if count.lower() in ("uncounted", "unknown", "unspecified", "?"):
                # Try to match adjacent stack height if we have coordinate hints
                if hint_x is not None and hint_z is not None:
                    best = self._guess_count_from_neighbors(hint_x, hint_z)
                    if best:
                        logger.info("Uncounted resolved to %d from neighbor stack height", best)
                        return best
                # Try tallest stack on grid
                if self.grid.blocks:
                    from collections import Counter
                    xz_counts = Counter((b.x, b.z) for b in self.grid.blocks)
                    tallest = xz_counts.most_common(1)[0][1]
                    if tallest > 1:
                        logger.info("Uncounted resolved to %d from tallest stack", tallest)
                        return tallest
                logger.warning("Unresolved count '%s', defaulting to 3", count)
                return 3
            try:
                return int(count)
            except ValueError:
                logger.warning("Invalid count '%s', defaulting to 1", count)
                return 1
        return 1

    def _guess_count_from_neighbors(self, x: int, z: int) -> int | None:
        """Check neighboring columns for stack heights to match."""
        step = self.grid.config.grid_step
        neighbors = [
            (x - step, z), (x + step, z),
            (x, z - step), (x, z + step),
        ]
        heights = []
        for nx, nz in neighbors:
            h = self.grid.stack_height(nx, nz)
            if h > 0:
                heights.append(h)
        if heights:
            return max(heights)
        return None

    def _resolve_color(self, color: str) -> str:
        """Resolve color, using a default if unspecified."""
        if color.lower() in ("uncolored", "unknown", "unspecified", "?"):
            # Use a reasonable default instead of failing
            # Try to pick the most common color already on the grid
            if self.grid.blocks:
                from collections import Counter
                color_counts = Counter(b.color for b in self.grid.blocks)
                default = color_counts.most_common(1)[0][0]
                logger.warning("Unresolved color '%s', defaulting to '%s' (most common on grid)", color, default)
                return default
            logger.warning("Unresolved color '%s', defaulting to 'Red'", color)
            return "Red"
        return color.capitalize()

    def _resolve_position(self, position: dict) -> tuple[int, int]:
        """Resolve a position dict to absolute (x, z) coordinates."""
        if "x" in position and "z" in position:
            return (int(position["x"]), int(position["z"]))

        if "relative_to" in position:
            return self._resolve_relative_position(position)

        raise ExecutionError(f"Cannot resolve position: {position}")

    def _resolve_relative_position(self, position: dict) -> tuple[int, int]:
        """Resolve a relative position to absolute (x, z)."""
        ref = position.get("relative_to", "")
        direction = position.get("direction", "")
        distance = int(position.get("distance", 1))

        # Resolve reference point
        ref_x, ref_z = self._resolve_reference(ref)

        # If direction is "on_top", return same x, z (y handled by stacking)
        if direction.lower() in ("on_top", "on top", "above"):
            return (ref_x, ref_z)

        # Resolve direction offset
        try:
            dx, dz = direction_offset(direction, self.grid.config)
        except ValueError:
            raise ExecutionError(f"Unknown direction: {direction}")

        return (ref_x + dx * distance, ref_z + dz * distance)

    def _resolve_reference(self, ref: str) -> tuple[int, int]:
        """Resolve a reference name to (x, z) coordinates."""
        ref_lower = ref.lower().strip()

        if ref_lower in ("origin", "center", "centre", "middle", "highlighted",
                         "highlighted_square", "middle_square", "center_square",
                         "highlighted_center_square", "highlighted_middle_square"):
            return (0, 0)

        # Parse "existing_<Color>_stack_at_<x>_<z>" pattern
        stack_match = re.match(
            r'existing_(\w+)_(?:stack|block|blocks?)_at_(-?\d+)_(-?\d+)',
            ref_lower,
        )
        if stack_match:
            return (int(stack_match.group(2)), int(stack_match.group(3)))

        # Parse "stack_at_<x>_<z>" or "block_at_<x>_<z>"
        at_match = re.match(r'(?:stack|block|blocks?)_at_(-?\d+)_(-?\d+)', ref_lower)
        if at_match:
            return (int(at_match.group(1)), int(at_match.group(2)))

        # "leftmost_<Color>" or "<Color>_leftmost" or "leftmost_<Color>_block" etc.
        color_extreme_match = re.match(
            r'(?:the[\s_]+)?(?:leftmost|rightmost|frontmost|backmost|rearmost)[\s_]+(\w+?)(?:[\s_]+(?:block|stack|tower|blocks?))?$',
            ref_lower,
        )
        if not color_extreme_match:
            color_extreme_match = re.match(
                r'(?:the[\s_]+)?(\w+?)[\s_]+(?:leftmost|rightmost|frontmost|backmost|rearmost)(?:[\s_]+(?:block|stack|tower|blocks?))?$',
                ref_lower,
            )
        if color_extreme_match:
            color = color_extreme_match.group(1).capitalize()
            color_blocks = [b for b in self.grid.blocks if b.color == color]
            if color_blocks:
                if "leftmost" in ref_lower:
                    b = min(color_blocks, key=lambda b: b.x)
                elif "rightmost" in ref_lower:
                    b = max(color_blocks, key=lambda b: b.x)
                elif "frontmost" in ref_lower:
                    b = max(color_blocks, key=lambda b: b.z)
                elif "backmost" in ref_lower or "rearmost" in ref_lower:
                    b = min(color_blocks, key=lambda b: b.z)
                else:
                    b = color_blocks[0]
                return (b.x, b.z)

        # Positional references (all blocks)
        if ref_lower in ("leftmost",):
            return self._find_extreme("min_x")
        if ref_lower in ("rightmost",):
            return self._find_extreme("max_x")
        if ref_lower in ("frontmost",):
            return self._find_extreme("max_z")
        if ref_lower in ("backmost", "rearmost"):
            return self._find_extreme("min_z")

        # Corner references
        try:
            return corner_position(ref, self.grid.config)
        except ValueError:
            pass

        # Try to find a block by color reference — prefer the most recently
        # placed position for that color (chain reference tracking)
        color_match = re.match(r'(?:the\s+)?(\w+)\s+(?:stack|block|tower|one|row)', ref_lower)
        if color_match:
            color = color_match.group(1).capitalize()
            # First check tracked positions from recent steps (chain refs)
            if color in self._last_position_by_color:
                logger.info("Resolved '%s' to tracked position %s", ref, self._last_position_by_color[color])
                return self._last_position_by_color[color]
            # Fall back to grid search
            color_blocks = [b for b in self.grid.blocks if b.color == color]
            if color_blocks:
                # Return the position of the first block of that color
                return (color_blocks[0].x, color_blocks[0].z)

        # "the one" / "the stack" without color = last placed
        if ref_lower in ("the_one", "the_stack", "last", "previous",
                         "the one", "the stack", "that stack", "that one",
                         "just_built", "just built"):
            if self._last_step_position:
                logger.info("Resolved '%s' to last step position %s", ref, self._last_step_position)
                return self._last_step_position

        # Last resort: try parsing as "x,z"
        coord_match = re.match(r'(-?\d+)[,_\s]+(-?\d+)', ref)
        if coord_match:
            return (int(coord_match.group(1)), int(coord_match.group(2)))

        raise ExecutionError(f"Cannot resolve reference: {ref}")

    def _find_extreme(self, which: str) -> tuple[int, int]:
        """Find the block at an extreme position."""
        if not self.grid.blocks:
            raise ExecutionError(f"Cannot find {which}: grid is empty")

        if which == "min_x":
            b = min(self.grid.blocks, key=lambda b: b.x)
        elif which == "max_x":
            b = max(self.grid.blocks, key=lambda b: b.x)
        elif which == "min_z":
            b = min(self.grid.blocks, key=lambda b: b.z)
        elif which == "max_z":
            b = max(self.grid.blocks, key=lambda b: b.z)
        else:
            raise ExecutionError(f"Unknown extreme: {which}")

        return (b.x, b.z)

    def _handle_stack(self, step: BuildStep) -> None:
        """Stack N blocks vertically at a position."""
        color = self._resolve_color(step.color)
        x, z = self._resolve_position(step.position)
        count = self._resolve_count(step.count, hint_x=x, hint_z=z)

        for _ in range(count):
            ny = self.grid.next_y(x, z)
            if ny > self.grid.config.valid_y[-1]:
                logger.warning("Stack at (%d,%d) reached max height, placed %d blocks", x, z, _)
                break
            self.grid.add(Block(color=color, x=x, y=ny, z=z))
            logger.info("  -> placed %s at (%d,%d,%d) [stack]", color, x, ny, z)
        self._record_step_position(x, z, color)

    def _handle_place(self, step: BuildStep) -> None:
        """Place a single block at an absolute position."""
        color = self._resolve_color(step.color)
        x, z = self._resolve_position(step.position)
        ny = self.grid.next_y(x, z)
        if not self.grid.config.is_valid_position(x, ny, z):
            raise ExecutionError(f"Position ({x},{ny},{z}) is out of bounds")
        self.grid.add(Block(color=color, x=x, y=ny, z=z))
        logger.info("  -> placed %s at (%d,%d,%d) [place]", color, x, ny, z)
        self._record_step_position(x, z, color)

    def _handle_place_relative(self, step: BuildStep) -> None:
        """Place block(s) relative to a reference position."""
        color = self._resolve_color(step.color)
        x, z = self._resolve_position(step.position)
        count = self._resolve_count(step.count, hint_x=x, hint_z=z) if step.count != 1 else 1

        for _ in range(count):
            ny = self.grid.next_y(x, z)
            if not self.grid.config.is_valid_position(x, ny, z):
                raise ExecutionError(f"Position ({x},{ny},{z}) is out of bounds")
            self.grid.add(Block(color=color, x=x, y=ny, z=z))
            logger.info("  -> placed %s at (%d,%d,%d) [place_relative]", color, x, ny, z)
        self._record_step_position(x, z, color)

    def _handle_extend_row(self, step: BuildStep) -> None:
        """Extend a row of blocks in a direction from a starting position."""
        color = self._resolve_color(step.color)

        pos = step.position
        if "x" in pos and "z" in pos:
            start_x, start_z = int(pos["x"]), int(pos["z"])
            dir_name = pos.get("direction", "right")
        elif "relative_to" in pos:
            start_x, start_z = self._resolve_reference(pos["relative_to"])
            dir_name = pos.get("direction", "right")
        else:
            raise ExecutionError(f"extend_row requires position with x,z or relative_to: {pos}")

        count = self._resolve_count(step.count, hint_x=start_x, hint_z=start_z)

        try:
            dx, dz = direction_offset(dir_name, self.grid.config)
        except ValueError:
            raise ExecutionError(f"Unknown direction for extend_row: {dir_name}")

        # ── Start-position adjustment for extend_row ──
        #
        # The LLM frequently says "extend_row Green count=2 at (0,200)"
        # when Green already exists at (0,200).  The intent is "extend
        # FROM the existing block", not "place a new block on top of it".
        #
        # Without adjustment, next_y(0,200) returns y=150 (the stack slot
        # above the existing block) and we get a vertical stack instead
        # of a horizontal extension.  This was the root cause of the
        # T-shape failure in run 184509 (round 1): the LLM produced a
        # correct extend_row step but the executor stacked on top of the
        # stem tip.
        #
        # Two cases:
        #
        # CASE 1  relative_to reference (for example "rightmost"):
        #   Always advance one step in the direction.  The reference IS
        #   the existing block; placing NEXT TO it is the obvious intent.
        #
        # CASE 2  absolute coordinates with same-color overlap:
        #   Advance one step ONLY when the start position already has a
        #   block of the SAME COLOR as the extend_row.  Different-color
        #   overlap (for example Blue extend starting on a Red position)
        #   is left alone because the LLM may genuinely want to start
        #   a new row at that coordinate, stacking on top of the other
        #   color.  See test_extend_row which extends Blue from a Red
        #   position and expects Blue blocks starting AT that position.
        #
        if "relative_to" in pos:
            start_x += dx
            start_z += dz
        else:
            existing = [b for b in self.grid.blocks
                        if b.x == start_x and b.z == start_z]
            if existing and any(b.color == color for b in existing):
                logger.info(
                    "extend_row: start (%d,%d) already has %s, advancing one step %s",
                    start_x, start_z, color, dir_name,
                )
                start_x += dx
                start_z += dz

        cx, cz = start_x, start_z
        for i in range(count):
            if i > 0:
                cx += dx
                cz += dz
            ny = self.grid.next_y(cx, cz)
            if not self.grid.config.is_valid_position(cx, ny, cz):
                logger.warning("extend_row hit grid boundary at (%d,%d)", cx, cz)
                break
            self.grid.add(Block(color=color, x=cx, y=ny, z=cz))
            logger.info("  -> placed %s at (%d,%d,%d) [extend_row]", color, cx, ny, cz)
        # Record the last block position in the row
        self._record_step_position(cx, cz, color)

    def _handle_place_at_corners(self, step: BuildStep) -> None:
        """Place blocks at grid corners."""
        color = self._resolve_color(step.color)
        corners = self.grid.config.corner_positions

        for corner_name, (cx, cz) in corners.items():
            ny = self.grid.next_y(cx, cz)
            if self.grid.config.is_valid_position(cx, ny, cz):
                self.grid.add(Block(color=color, x=cx, y=ny, z=cz))

    def _handle_place_along_edge(self, step: BuildStep) -> None:
        """Place blocks along an entire grid edge."""
        color = self._resolve_color(step.color)
        count = self._resolve_count(step.count)
        pos = step.position
        edge = pos.get("edge", pos.get("direction", "left")).lower()

        valid_xz = self.grid.config.valid_xz
        h = self.grid.config.half_extent

        if edge in ("left", "left_edge"):
            positions = [(-h, z) for z in valid_xz[:count]]
        elif edge in ("right", "right_edge"):
            positions = [(h, z) for z in valid_xz[:count]]
        elif edge in ("top", "top_edge", "back", "back_edge"):
            positions = [(x, -h) for x in valid_xz[:count]]
        elif edge in ("bottom", "bottom_edge", "front", "front_edge"):
            positions = [(x, h) for x in valid_xz[:count]]
        else:
            raise ExecutionError(f"Unknown edge: {edge}")

        for cx, cz in positions:
            ny = self.grid.next_y(cx, cz)
            if self.grid.config.is_valid_position(cx, ny, cz):
                self.grid.add(Block(color=color, x=cx, y=ny, z=cz))

    def _clamp_to_grid(self, x: int, z: int) -> tuple[int, int]:
        """Clamp coordinates to the nearest valid grid position."""
        valid = self.grid.config.valid_xz
        min_v, max_v = valid[0], valid[-1]
        step = self.grid.config.grid_step

        def snap(val: int) -> int:
            clamped = max(min_v, min(max_v, val))
            return round(clamped / step) * step
        return (snap(x), snap(z))
