"""Grid model and block operations for a configurable 3D building grid.

Coordinate system (defaults match the 9x9 game board):
- x-z plane is the horizontal grid.
- y is vertical height. Ground level y=50, each block adds +100.
- "In front" = +z, "behind" = -z
- "Right" = +x, "left" = -x
- "Up/on top" = +y
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class GridConfig:
    """Fully describes the grid geometry."""

    grid_size: int = 9
    grid_step: int = 100
    y_ground: int = 50
    y_step: int = 100
    max_height: int = 5

    @property
    def half_extent(self) -> int:
        return (self.grid_size // 2) * self.grid_step

    @property
    def valid_xz(self) -> list[int]:
        h = self.half_extent
        return list(range(-h, h + 1, self.grid_step))

    @property
    def valid_y(self) -> list[int]:
        return [self.y_ground + i * self.y_step for i in range(self.max_height)]

    @property
    def direction_map(self) -> dict[str, tuple[int, int]]:
        s = self.grid_step
        return {
            "right": (s, 0),
            "left": (-s, 0),
            "front": (0, s),
            "forward": (0, s),
            "in_front": (0, s),
            "in front": (0, s),
            "in front of": (0, s),
            "behind": (0, -s),
            "back": (0, -s),
            "backward": (0, -s),
        }

    @property
    def corner_positions(self) -> dict[str, tuple[int, int]]:
        h = self.half_extent
        return {
            "top_left": (-h, -h),
            "top_right": (h, -h),
            "bottom_left": (-h, h),
            "bottom_right": (h, h),
        }

    def is_valid_position(self, x: int, y: int, z: int) -> bool:
        return x in self.valid_xz and y in self.valid_y and z in self.valid_xz

    def describe(self) -> str:
        return (
            f"{self.grid_size}x{self.grid_size} grid, step={self.grid_step}, "
            f"x/z range [{self.valid_xz[0]}..{self.valid_xz[-1]}], "
            f"y ground={self.y_ground}, y step={self.y_step}, "
            f"max height={self.max_height} ({self.valid_y[-1]})"
        )


VALID_COLORS = [
    "Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Pink", "Brown",
    "White", "Black", "Cyan", "Magenta",
]


@dataclass(frozen=True, order=True)
class Block:
    """A single block on the grid."""
    color: str
    x: int
    y: int
    z: int

    def __str__(self) -> str:
        return f"{self.color},{self.x},{self.y},{self.z}"

    @classmethod
    def from_str(cls, s: str) -> Block:
        parts = s.strip().split(",")
        if len(parts) != 4:
            raise ValueError(f"Expected 'Color,x,y,z', got: {s!r}")
        color = parts[0].strip().capitalize()
        x, y, z = int(parts[1]), int(parts[2]), int(parts[3])
        return cls(color=color, x=x, y=y, z=z)

    @property
    def xz(self) -> tuple[int, int]:
        return (self.x, self.z)

    def moved(self, dx: int = 0, dy: int = 0, dz: int = 0) -> Block:
        return Block(color=self.color, x=self.x + dx, y=self.y + dy, z=self.z + dz)

    def with_color(self, color: str) -> Block:
        return Block(color=color.capitalize(), x=self.x, y=self.y, z=self.z)


@dataclass
class Grid:
    """Represents the full state of blocks on the grid."""
    blocks: list[Block] = field(default_factory=list)
    config: GridConfig = field(default_factory=GridConfig)

    @classmethod
    def from_str(cls, s: str, config: GridConfig | None = None) -> Grid:
        grid = cls(config=config or GridConfig())
        if not s or not s.strip():
            return grid
        for part in s.split(";"):
            part = part.strip()
            if part:
                grid.blocks.append(Block.from_str(part))
        return grid

    def to_str(self) -> str:
        return ";".join(str(b) for b in sorted(self.blocks))

    def to_build_response(self) -> str:
        return f"[BUILD];{self.to_str()}"

    def add(self, block: Block) -> None:
        self.blocks.append(block)

    @property
    def positions(self) -> set[tuple[int, int, int]]:
        return {(b.x, b.y, b.z) for b in self.blocks}

    @property
    def xz_positions(self) -> set[tuple[int, int]]:
        return {b.xz for b in self.blocks}

    def blocks_at_xz(self, x: int, z: int) -> list[Block]:
        return sorted(
            [b for b in self.blocks if b.x == x and b.z == z],
            key=lambda b: b.y,
        )

    def stack_height(self, x: int, z: int) -> int:
        return len(self.blocks_at_xz(x, z))

    def top_block_at(self, x: int, z: int) -> Optional[Block]:
        stack = self.blocks_at_xz(x, z)
        return stack[-1] if stack else None

    def next_y(self, x: int, z: int) -> int:
        stack = self.blocks_at_xz(x, z)
        if not stack:
            return self.config.y_ground
        return stack[-1].y + self.config.y_step

    def bounding_box(self) -> dict[str, int]:
        if not self.blocks:
            return {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0, "min_z": 0, "max_z": 0}
        return {
            "min_x": min(b.x for b in self.blocks),
            "max_x": max(b.x for b in self.blocks),
            "min_y": min(b.y for b in self.blocks),
            "max_y": max(b.y for b in self.blocks),
            "min_z": min(b.z for b in self.blocks),
            "max_z": max(b.z for b in self.blocks),
        }

    def ground_footprint(self) -> set[tuple[int, int]]:
        return self.xz_positions

    def by_color(self) -> dict[str, list[Block]]:
        groups: dict[str, list[Block]] = {}
        for b in self.blocks:
            groups.setdefault(b.color, []).append(b)
        return groups

    def by_layer(self) -> dict[int, list[Block]]:
        layers: dict[int, list[Block]] = {}
        for b in self.blocks:
            layers.setdefault(b.y, []).append(b)
        return layers

    def colors_used(self) -> list[str]:
        return sorted(set(b.color for b in self.blocks))

    def describe(self) -> str:
        """Natural language description of the current grid state."""
        if not self.blocks:
            return "The grid is empty."

        lines = [f"{len(self.blocks)} block(s) on the grid."]
        colors = self.by_color()
        for color, blocks in sorted(colors.items()):
            positions = [f"({b.x},{b.y},{b.z})" for b in blocks]
            lines.append(f"  {color}: {len(blocks)} block(s) at {', '.join(positions)}")

        # Describe stacks
        for xz in sorted(self.xz_positions):
            stack = self.blocks_at_xz(*xz)
            if len(stack) > 1:
                stack_colors = [b.color for b in stack]
                lines.append(
                    f"  Stack at ({xz[0]},{xz[1]}): {len(stack)} blocks tall "
                    f"[{', '.join(stack_colors)} from bottom to top]"
                )

        return "\n".join(lines)


def direction_offset(direction: str, config: GridConfig | None = None) -> tuple[int, int]:
    cfg = config or GridConfig()
    dmap = cfg.direction_map
    key = direction.lower().replace("-", "_").strip()
    if key in dmap:
        return dmap[key]
    # Try with spaces preserved
    for k, v in dmap.items():
        if k == key or k.replace("_", " ") == key:
            return v
    raise ValueError(
        f"Unknown direction {direction!r}. Valid: {', '.join(dmap.keys())}"
    )


def corner_position(corner: str, config: GridConfig | None = None) -> tuple[int, int]:
    cfg = config or GridConfig()
    cmap = cfg.corner_positions
    key = corner.lower().replace(" ", "_").replace("-", "_")
    if key not in cmap:
        raise ValueError(f"Unknown corner {corner!r}. Valid: {', '.join(cmap.keys())}")
    return cmap[key]
