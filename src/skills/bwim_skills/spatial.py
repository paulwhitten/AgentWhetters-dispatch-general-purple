"""Spatial relationships between blocks on the grid."""

from __future__ import annotations

from typing import Optional

from .grid import Block, Grid, GridConfig


def is_touching(a: Block, b: Block, config: GridConfig) -> bool:
    """True if two blocks share a face (one grid step apart on exactly one axis)."""
    dx = abs(a.x - b.x)
    dy = abs(a.y - b.y)
    dz = abs(a.z - b.z)
    step = config.grid_step
    y_step = config.y_step
    diffs = [
        (dx == step and dy == 0 and dz == 0),
        (dx == 0 and dy == y_step and dz == 0),
        (dx == 0 and dy == 0 and dz == step),
    ]
    return sum(diffs) == 1


def relationship(a: Block, b: Block, config: GridConfig) -> Optional[str]:
    """Describe the spatial relationship from a's point of view toward b."""
    if not is_touching(a, b, config):
        return None

    dx = b.x - a.x
    dy = b.y - a.y
    dz = b.z - a.z
    step = config.grid_step
    y_step = config.y_step

    if dy == y_step and dx == 0 and dz == 0:
        return "on top of"
    if dy == -y_step and dx == 0 and dz == 0:
        return "under"
    if dx == step and dy == 0 and dz == 0:
        return "to the right of"
    if dx == -step and dy == 0 and dz == 0:
        return "to the left of"
    if dx == 0 and dy == 0 and dz == step:
        return "in front of"
    if dz == -step and dx == 0 and dy == 0:
        return "behind"
    return None


def blocks_above(grid: Grid, block: Block) -> list[Block]:
    return sorted(
        [b for b in grid.blocks if b.x == block.x and b.z == block.z and b.y > block.y],
        key=lambda b: b.y,
    )


def blocks_below(grid: Grid, block: Block) -> list[Block]:
    return sorted(
        [b for b in grid.blocks if b.x == block.x and b.z == block.z and b.y < block.y],
        key=lambda b: b.y,
        reverse=True,
    )


def block_on_top_of(grid: Grid, block: Block) -> Optional[Block]:
    target_y = block.y + grid.config.y_step
    for b in grid.blocks:
        if b.x == block.x and b.z == block.z and b.y == target_y:
            return b
    return None


def block_under(grid: Grid, block: Block) -> Optional[Block]:
    target_y = block.y - grid.config.y_step
    for b in grid.blocks:
        if b.x == block.x and b.z == block.z and b.y == target_y:
            return b
    return None


def blocks_next_to(grid: Grid, block: Block) -> list[tuple[str, Block]]:
    step = grid.config.grid_step
    directions = [
        ("right", (step, 0)),
        ("left", (-step, 0)),
        ("front", (0, step)),
        ("behind", (0, -step)),
    ]
    results: list[tuple[str, Block]] = []
    for name, (dx, dz) in directions:
        for b in grid.blocks:
            if b.x == block.x + dx and b.z == block.z + dz and b.y == block.y:
                results.append((name, b))
    return results


def is_connected(grid: Grid) -> bool:
    """Check whether all blocks form a single connected group."""
    if len(grid.blocks) <= 1:
        return True
    components = connected_components(grid)
    return len(components) == 1


def connected_components(grid: Grid) -> list[list[Block]]:
    """Return groups of connected blocks."""
    if not grid.blocks:
        return []

    visited: set[int] = set()
    components: list[list[Block]] = []

    block_index = {(b.x, b.y, b.z): i for i, b in enumerate(grid.blocks)}

    def neighbors(idx: int) -> list[int]:
        b = grid.blocks[idx]
        step = grid.config.grid_step
        y_step = grid.config.y_step
        offsets = [
            (step, 0, 0), (-step, 0, 0),
            (0, y_step, 0), (0, -y_step, 0),
            (0, 0, step), (0, 0, -step),
        ]
        result = []
        for dx, dy, dz in offsets:
            pos = (b.x + dx, b.y + dy, b.z + dz)
            if pos in block_index:
                result.append(block_index[pos])
        return result

    for start in range(len(grid.blocks)):
        if start in visited:
            continue
        component: list[Block] = []
        stack = [start]
        while stack:
            idx = stack.pop()
            if idx in visited:
                continue
            visited.add(idx)
            component.append(grid.blocks[idx])
            stack.extend(neighbors(idx))
        components.append(component)

    return components
