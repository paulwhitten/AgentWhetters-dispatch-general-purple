"""Structure Analyzer: detect shapes and geometry in existing grid structures.

Analyzes a Grid to produce a structured description of the shapes present,
including lines, L-shapes, T-shapes, stacks, and isolated blocks.  This
description is injected into the LLM planner prompt so that the model
receives pre-computed geometric information instead of having to infer it
from raw coordinate lists.

Generalises to any grid configuration and block arrangement — no hard-coded
answers or task-specific lookup tables.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .grid import Block, Grid, GridConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data classes for analysis results
# ------------------------------------------------------------------

@dataclass
class LineInfo:
    """A contiguous line of blocks on the ground plane."""
    color: str
    positions: list[tuple[int, int]]  # (x, z) in order along the line
    direction: str  # "horizontal" (varying x) or "vertical" (varying z)
    length: int
    start: tuple[int, int]  # first position
    end: tuple[int, int]    # last position

    @property
    def leftmost(self) -> tuple[int, int]:
        return min(self.positions, key=lambda p: p[0])

    @property
    def rightmost(self) -> tuple[int, int]:
        return max(self.positions, key=lambda p: p[0])

    @property
    def frontmost(self) -> tuple[int, int]:
        return max(self.positions, key=lambda p: p[1])

    @property
    def backmost(self) -> tuple[int, int]:
        return min(self.positions, key=lambda p: p[1])


@dataclass
class StackInfo:
    """A vertical stack of blocks at one (x, z)."""
    color: str  # dominant color; "Mixed" if multiple
    x: int
    z: int
    height: int
    colors: list[str]  # bottom to top


@dataclass
class ShapeInfo:
    """Recognised compound shape (T, L, or line)."""
    shape_type: str  # "T", "L", "line", "single", "scattered"
    color: str
    # For T-shapes:
    crossbar: Optional[LineInfo] = None
    stem: Optional[LineInfo] = None
    arm_tips: list[tuple[int, int]] = field(default_factory=list)
    longer_base: Optional[str] = None  # "crossbar" or "stem"
    # For L-shapes:
    longer_side: Optional[LineInfo] = None
    shorter_side: Optional[LineInfo] = None
    corner: Optional[tuple[int, int]] = None
    extend_direction: Optional[str] = None  # direction to continue the longer side
    # For lines:
    line: Optional[LineInfo] = None


@dataclass
class StructureAnalysis:
    """Complete analysis of a grid's existing structure."""
    shapes: list[ShapeInfo] = field(default_factory=list)
    stacks: list[StackInfo] = field(default_factory=list)
    block_count: int = 0
    colors_used: list[str] = field(default_factory=list)
    color_positions: dict[str, dict[str, tuple[int, int]]] = field(default_factory=dict)
    # Per-color extremes: color → {"leftmost": (x,z), "rightmost": ...}

    def describe(self) -> str:
        """Produce a natural-language description for injection into the planner."""
        if self.block_count == 0:
            return "The grid is empty."

        lines: list[str] = []
        lines.append(f"STRUCTURE ANALYSIS ({self.block_count} blocks, colors: {', '.join(self.colors_used)}):")

        # Shapes
        for shape in self.shapes:
            lines.append(self._describe_shape(shape))

        # Stacks
        for stack in self.stacks:
            if stack.height > 1:
                lines.append(
                    f"  Stack at ({stack.x},{stack.z}): {stack.height} blocks tall, "
                    f"colors bottom-to-top: [{', '.join(stack.colors)}]"
                )

        # Color extremes
        for color, extremes in sorted(self.color_positions.items()):
            parts = []
            for label, pos in sorted(extremes.items()):
                parts.append(f"{label}=({pos[0]},{pos[1]})")
            lines.append(f"  {color} positions: {', '.join(parts)}")

        return "\n".join(lines)

    @staticmethod
    def _describe_shape(shape: ShapeInfo) -> str:
        if shape.shape_type == "T":
            desc = f"  T-SHAPE ({shape.color}):"
            if shape.crossbar:
                desc += (
                    f"\n    Crossbar: {shape.crossbar.direction}, "
                    f"{shape.crossbar.length} blocks, "
                    f"from ({shape.crossbar.start[0]},{shape.crossbar.start[1]}) "
                    f"to ({shape.crossbar.end[0]},{shape.crossbar.end[1]})"
                )
            if shape.stem:
                desc += (
                    f"\n    Stem: {shape.stem.direction}, "
                    f"{shape.stem.length} blocks, "
                    f"from ({shape.stem.start[0]},{shape.stem.start[1]}) "
                    f"to ({shape.stem.end[0]},{shape.stem.end[1]})"
                )
            if shape.longer_base:
                desc += f"\n    Longer base: {shape.longer_base}"
            if shape.arm_tips:
                tips = ", ".join(f"({t[0]},{t[1]})" for t in shape.arm_tips)
                desc += f"\n    Arm tips (ends of crossbar): {tips}"
                # Compute extension positions
                if shape.crossbar and len(shape.arm_tips) == 2:
                    t1, t2 = shape.arm_tips
                    if shape.crossbar.direction == "horizontal":
                        ext1 = (min(t1[0], t2[0]) - 100, t1[1])
                        ext2 = (max(t1[0], t2[0]) + 100, t1[1])
                    else:
                        ext1 = (t1[0], min(t1[1], t2[1]) - 100)
                        ext2 = (t1[0], max(t1[1], t2[1]) + 100)
                    desc += (
                        f"\n    To extend arms outward: place at "
                        f"({ext1[0]},{ext1[1]}) and ({ext2[0]},{ext2[1]})"
                    )
            return desc

        elif shape.shape_type == "L":
            desc = f"  L-SHAPE ({shape.color}):"
            if shape.longer_side:
                desc += (
                    f"\n    Longer side: {shape.longer_side.direction}, "
                    f"{shape.longer_side.length} blocks, "
                    f"from ({shape.longer_side.start[0]},{shape.longer_side.start[1]}) "
                    f"to ({shape.longer_side.end[0]},{shape.longer_side.end[1]})"
                )
            if shape.shorter_side:
                desc += (
                    f"\n    Shorter side: {shape.shorter_side.direction}, "
                    f"{shape.shorter_side.length} blocks, "
                    f"from ({shape.shorter_side.start[0]},{shape.shorter_side.start[1]}) "
                    f"to ({shape.shorter_side.end[0]},{shape.shorter_side.end[1]})"
                )
            if shape.corner:
                desc += f"\n    Corner (junction): ({shape.corner[0]},{shape.corner[1]})"
            if shape.extend_direction:
                desc += f"\n    To extend longer side: continue in {shape.extend_direction} direction"
                if shape.longer_side:
                    # Compute the extension start position
                    ls = shape.longer_side
                    if shape.corner:
                        # The end away from the corner
                        far_end = ls.end if ls.start == shape.corner else ls.start
                        if ls.direction == "horizontal":
                            dx = 100 if far_end[0] > shape.corner[0] else -100
                            ext_pos = (far_end[0] + dx, far_end[1])
                        else:
                            dz = 100 if far_end[1] > shape.corner[1] else -100
                            ext_pos = (far_end[0], far_end[1] + dz)
                        desc += f", next block at ({ext_pos[0]},{ext_pos[1]})"
            return desc

        elif shape.shape_type == "line":
            line = shape.line
            if line:
                return (
                    f"  LINE ({shape.color}): {line.direction}, "
                    f"{line.length} blocks, "
                    f"from ({line.start[0]},{line.start[1]}) "
                    f"to ({line.end[0]},{line.end[1]}), "
                    f"leftmost=({line.leftmost[0]},{line.leftmost[1]}), "
                    f"rightmost=({line.rightmost[0]},{line.rightmost[1]})"
                )
            return f"  LINE ({shape.color})"

        elif shape.shape_type == "single":
            return f"  SINGLE BLOCK ({shape.color})"

        return f"  {shape.shape_type.upper()} ({shape.color})"


# ------------------------------------------------------------------
# Analysis functions
# ------------------------------------------------------------------

def analyze_structure(grid: Grid) -> StructureAnalysis:
    """Analyze the grid to detect shapes, stacks, and spatial relationships.

    This is fully deterministic and generalisable — it works on any
    grid configuration and block arrangement.
    """
    analysis = StructureAnalysis()
    analysis.block_count = len(grid.blocks)

    if not grid.blocks:
        return analysis

    analysis.colors_used = grid.colors_used()

    # Compute per-color extremes
    color_groups = grid.by_color()
    for color, blocks in color_groups.items():
        extremes: dict[str, tuple[int, int]] = {}
        if blocks:
            extremes["leftmost"] = min((b.x, b.z) for b in blocks)
            extremes["rightmost"] = max((b.x, b.z) for b in blocks)
            extremes["frontmost"] = max(((b.z, b.x) for b in blocks))
            extremes["frontmost"] = (
                max(blocks, key=lambda b: b.z).x,
                max(blocks, key=lambda b: b.z).z,
            )
            extremes["backmost"] = (
                min(blocks, key=lambda b: b.z).x,
                min(blocks, key=lambda b: b.z).z,
            )
        analysis.color_positions[color] = extremes

    # Detect stacks (multiple blocks at same x,z)
    xz_groups: dict[tuple[int, int], list[Block]] = {}
    for b in grid.blocks:
        xz_groups.setdefault((b.x, b.z), []).append(b)

    for (x, z), blocks in xz_groups.items():
        blocks_sorted = sorted(blocks, key=lambda b: b.y)
        colors = [b.color for b in blocks_sorted]
        dominant = Counter(colors).most_common(1)[0][0]
        analysis.stacks.append(StackInfo(
            color=dominant if len(set(colors)) == 1 else "Mixed",
            x=x, z=z,
            height=len(blocks_sorted),
            colors=colors,
        ))

    # Detect shapes per color (using ground-level footprint)
    for color, blocks in color_groups.items():
        ground_positions = list(set((b.x, b.z) for b in blocks))
        if len(ground_positions) == 0:
            continue
        if len(ground_positions) == 1:
            analysis.shapes.append(ShapeInfo(shape_type="single", color=color))
            continue

        shape = _detect_shape(ground_positions, color, grid.config)
        if shape:
            analysis.shapes.append(shape)

    return analysis


def _detect_shape(
    positions: list[tuple[int, int]],
    color: str,
    config: GridConfig,
) -> ShapeInfo | None:
    """Detect the shape formed by a set of (x, z) positions."""
    step = config.grid_step
    pos_set = set(positions)

    if len(positions) <= 1:
        return ShapeInfo(shape_type="single", color=color)

    # Check if all positions form a line
    line = _detect_line(positions, step)
    if line:
        line.color = color
        return ShapeInfo(shape_type="line", color=color, line=line)

    # Check for T-shape
    t_shape = _detect_t_shape(positions, color, step)
    if t_shape:
        return t_shape

    # Check for L-shape
    l_shape = _detect_l_shape(positions, color, step)
    if l_shape:
        return l_shape

    # Scattered
    return ShapeInfo(shape_type="scattered", color=color)


def _detect_line(
    positions: list[tuple[int, int]],
    step: int,
) -> LineInfo | None:
    """Check if positions form a contiguous line (horizontal or vertical)."""
    xs = sorted(set(p[0] for p in positions))
    zs = sorted(set(p[1] for p in positions))

    # Horizontal line: all same z, x values are contiguous
    if len(zs) == 1:
        if _is_contiguous(xs, step):
            ordered = sorted(positions, key=lambda p: p[0])
            return LineInfo(
                color="",  # filled by caller
                positions=ordered,
                direction="horizontal",
                length=len(ordered),
                start=ordered[0],
                end=ordered[-1],
            )

    # Vertical line: all same x, z values are contiguous
    if len(xs) == 1:
        if _is_contiguous(zs, step):
            ordered = sorted(positions, key=lambda p: p[1])
            return LineInfo(
                color="",
                positions=ordered,
                direction="vertical",
                length=len(ordered),
                start=ordered[0],
                end=ordered[-1],
            )

    return None


def _is_contiguous(values: list[int], step: int) -> bool:
    """Check if sorted values form a contiguous sequence with given step."""
    for i in range(1, len(values)):
        if values[i] - values[i - 1] != step:
            return False
    return True


def _detect_t_shape(
    positions: list[tuple[int, int]],
    color: str,
    step: int,
) -> ShapeInfo | None:
    """Detect a T-shape: a crossbar line intersected by a stem line.

    A T-shape has exactly one position where removing it would split the
    remaining positions into a line + another line (the crossbar and stem).
    The junction point is shared by both.
    """
    pos_set = set(positions)
    n = len(positions)

    if n < 4:
        return None  # T needs at least 4 blocks

    # For each position, check if it's the junction of two perpendicular lines
    for jx, jz in positions:
        # Collect positions in the same row (horizontal through junction)
        h_line = sorted([p for p in positions if p[1] == jz], key=lambda p: p[0])
        # Collect positions in the same column (vertical through junction)
        v_line = sorted([p for p in positions if p[0] == jx], key=lambda p: p[1])

        # Both lines must pass through the junction and be contiguous
        h_xs = [p[0] for p in h_line]
        v_zs = [p[1] for p in v_line]

        if not _is_contiguous(h_xs, step) or not _is_contiguous(v_zs, step):
            continue

        # Together they must cover all positions (junction counted once)
        if len(h_line) + len(v_line) - 1 != n:
            continue

        # Valid T-shape found!
        h_info = LineInfo(
            color=color, positions=h_line, direction="horizontal",
            length=len(h_line), start=h_line[0], end=h_line[-1],
        )
        v_info = LineInfo(
            color=color, positions=v_line, direction="vertical",
            length=len(v_line), start=v_line[0], end=v_line[-1],
        )

        # Determine which is the crossbar vs stem.
        # The crossbar is the one where the junction is NOT at an endpoint.
        # (For a true T, the stem connects at the middle of the crossbar.)
        junction = (jx, jz)

        # Check if junction is an interior point of the horizontal line
        h_interior = len(h_line) >= 3 and h_line[0] != junction and h_line[-1] != junction
        v_interior = len(v_line) >= 3 and v_line[0] != junction and v_line[-1] != junction

        if h_interior:
            crossbar, stem = h_info, v_info
        elif v_interior:
            crossbar, stem = v_info, h_info
        else:
            # Junction is at endpoint of both lines — this is an L-shape,
            # not a T-shape.  Skip and let the L detector handle it.
            continue

        longer_base = "crossbar" if crossbar.length >= stem.length else "stem"

        # Arm tips = endpoints of the crossbar
        arm_tips = [crossbar.start, crossbar.end]

        return ShapeInfo(
            shape_type="T",
            color=color,
            crossbar=crossbar,
            stem=stem,
            arm_tips=arm_tips,
            longer_base=longer_base,
        )

    return None


def _detect_l_shape(
    positions: list[tuple[int, int]],
    color: str,
    step: int,
) -> ShapeInfo | None:
    """Detect an L-shape: two perpendicular lines sharing one endpoint.

    An L-shape has exactly one corner position where two lines meet at
    an endpoint of both.
    """
    pos_set = set(positions)
    n = len(positions)

    if n < 3:
        return None

    for jx, jz in positions:
        junction = (jx, jz)

        # Horizontal and vertical lines through junction
        h_line = sorted([p for p in positions if p[1] == jz], key=lambda p: p[0])
        v_line = sorted([p for p in positions if p[0] == jx], key=lambda p: p[1])

        h_xs = [p[0] for p in h_line]
        v_zs = [p[1] for p in v_line]

        if not _is_contiguous(h_xs, step) or not _is_contiguous(v_zs, step):
            continue

        if len(h_line) + len(v_line) - 1 != n:
            continue

        # For L-shape, junction must be an endpoint of BOTH lines
        h_is_endpoint = (junction == h_line[0] or junction == h_line[-1])
        v_is_endpoint = (junction == v_line[0] or junction == v_line[-1])

        if not (h_is_endpoint and v_is_endpoint):
            continue

        # Need at least 2 blocks in each arm (including junction)
        if len(h_line) < 2 or len(v_line) < 2:
            continue

        h_info = LineInfo(
            color=color, positions=h_line, direction="horizontal",
            length=len(h_line), start=h_line[0], end=h_line[-1],
        )
        v_info = LineInfo(
            color=color, positions=v_line, direction="vertical",
            length=len(v_line), start=v_line[0], end=v_line[-1],
        )

        if h_info.length >= v_info.length:
            longer, shorter = h_info, v_info
        else:
            longer, shorter = v_info, h_info

        # Determine the direction to extend the longer side
        # (away from the corner, continuing the line)
        far_end = longer.end if longer.start == junction else longer.start
        if longer.direction == "horizontal":
            if far_end[0] > junction[0]:
                ext_dir = "right"
            else:
                ext_dir = "left"
        else:
            if far_end[1] > junction[1]:
                ext_dir = "front"
            else:
                ext_dir = "behind"

        return ShapeInfo(
            shape_type="L",
            color=color,
            longer_side=longer,
            shorter_side=shorter,
            corner=junction,
            extend_direction=ext_dir,
        )

    return None
