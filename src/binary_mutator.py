"""Binary mutation engine for near-miss PoCs -- zero LLM tokens.

When a PoC gets a near-miss result (PARTIAL_CRASH, WRONG_LOCATION,
WRONG_CRASH), generate deterministic byte mutations to try before
burning another LLM call.

Reference: github.com/sharathbaddam/AgentWhetters-cybergym
"""

from __future__ import annotations

import logging
import random
from typing import List, Tuple

logger = logging.getLogger(__name__)

_BOUNDARY_VALUES = [0x00, 0x01, 0x7E, 0x7F, 0x80, 0x81, 0xFE, 0xFF]

_BOUNDARY_DWORDS = [
    b"\x00\x00\x00\x00",
    b"\x00\x00\x00\x01",
    b"\x7f\xff\xff\xff",
    b"\x80\x00\x00\x00",
    b"\xff\xff\xff\xff",
    b"\xff\xff\xff\xfe",
]

_STRATEGIES = [
    "random_flip", "boundary_byte", "zero_byte", "max_byte",
    "increment", "decrement", "dword_boundary", "swap_bytes",
]


def generate_mutations(
    poc_bytes: bytes, max_mutations: int = 10,
) -> List[Tuple[bytes, str]]:
    """Generate a list of mutated PoC candidates.

    Mutates primarily in the last third of the PoC (trigger region,
    not format headers).
    """
    if not poc_bytes:
        return []

    region_start = max(0, len(poc_bytes) * 2 // 3)
    candidates: List[Tuple[bytes, str]] = []

    for i in range(max_mutations):
        mutated = bytearray(poc_bytes)
        strategy = _STRATEGIES[i % len(_STRATEGIES)]

        if len(mutated) <= region_start:
            region_start = 0

        pos = random.randint(region_start, len(mutated) - 1)

        if strategy == "random_flip":
            mutated[pos] = random.randint(0, 255)
        elif strategy == "boundary_byte":
            mutated[pos] = random.choice(_BOUNDARY_VALUES)
        elif strategy == "zero_byte":
            mutated[pos] = 0x00
        elif strategy == "max_byte":
            mutated[pos] = 0xFF
        elif strategy == "increment":
            mutated[pos] = (mutated[pos] + 1) & 0xFF
        elif strategy == "decrement":
            mutated[pos] = (mutated[pos] - 1) & 0xFF
        elif strategy == "dword_boundary":
            dword_pos = min(pos, len(mutated) - 4)
            if dword_pos >= 0:
                dword = random.choice(_BOUNDARY_DWORDS)
                for j, b in enumerate(dword):
                    if dword_pos + j < len(mutated):
                        mutated[dword_pos + j] = b
        elif strategy == "swap_bytes":
            if len(mutated) > 1:
                pos2 = random.randint(region_start, len(mutated) - 1)
                mutated[pos], mutated[pos2] = mutated[pos2], mutated[pos]

        candidates.append((bytes(mutated), f"binary_mutation #{i + 1}: {strategy} at offset {pos}"))

    return candidates
