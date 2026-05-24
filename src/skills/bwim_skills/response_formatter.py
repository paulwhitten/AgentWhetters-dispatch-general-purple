"""Response formatter and validator for [BUILD] responses."""

from __future__ import annotations

from .grid import Grid, GridConfig, VALID_COLORS


def format_build_response(grid: Grid) -> str:
    """Format the current grid state as a [BUILD] response string."""
    if not grid.blocks:
        return "[BUILD]"
    return grid.to_build_response()


def validate_build_response(response: str, config: GridConfig | None = None) -> tuple[bool, list[str]]:
    """Validate a [BUILD] response string.

    Returns (is_valid, list_of_errors).
    """
    cfg = config or GridConfig()
    errors: list[str] = []

    if not response.startswith("[BUILD]"):
        errors.append("Missing [BUILD] prefix.")
        return (False, errors)

    content = response[7:]
    if content.startswith(";"):
        content = content[1:]

    if not content.strip():
        # Empty build is technically valid (empty grid)
        return (True, [])

    parts = [p.strip() for p in content.split(";") if p.strip()]
    positions_seen: set[tuple[int, int, int]] = set()

    for i, part in enumerate(parts):
        fields = part.split(",")
        if len(fields) != 4:
            errors.append(f"Block {i+1} '{part}': expected 4 fields (Color,x,y,z), got {len(fields)}.")
            continue

        color = fields[0].strip()
        if not color[0].isupper():
            errors.append(f"Block {i+1}: color '{color}' should be capitalized.")

        try:
            x, y, z = int(fields[1]), int(fields[2]), int(fields[3])
        except ValueError:
            errors.append(f"Block {i+1} '{part}': coordinates must be integers.")
            continue

        if x not in cfg.valid_xz:
            errors.append(f"Block {i+1}: x={x} not in valid range.")
        if y not in cfg.valid_y:
            errors.append(f"Block {i+1}: y={y} not in valid range.")
        if z not in cfg.valid_xz:
            errors.append(f"Block {i+1}: z={z} not in valid range.")

        pos = (x, y, z)
        if pos in positions_seen:
            errors.append(f"Block {i+1}: duplicate position ({x},{y},{z}).")
        positions_seen.add(pos)

    return (len(errors) == 0, errors)
