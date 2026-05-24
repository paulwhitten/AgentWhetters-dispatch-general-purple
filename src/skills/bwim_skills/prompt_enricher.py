"""Adaptive prompt enrichment for difficult spatial concepts.

Scans the building instruction for known "difficult" phrases that the LLM
frequently gets wrong, and returns targeted concept definitions + worked
micro-examples to inject into the user prompt immediately before the LLM
call.  This puts the relevant rule in the model's local attention window,
right next to the instruction it needs to apply it to.

Design principles:
- Each rule is independent — multiple can fire on the same instruction.
- Rules are short (≤ 5 lines) to avoid bloating the prompt.
- Every rule includes one concrete coordinate example so the model sees
  the expected number format, not just a verbal description.
- Rules are tested against all CSV data to ensure zero false positives
  on fully_spec instructions (i.e., enrichments never HURT correct behavior).
"""

from __future__ import annotations

import atexit
import logging
import re
import signal
import sys
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Module-level counter: tracks how many times each enrichment rule fires
_enrichment_counter: Counter[str] = Counter()
_summary_printed = False


def _print_enrichment_summary(*_args) -> None:
    """Print enrichment fire counts to stderr (always visible) at process exit."""
    global _summary_printed
    if _summary_printed:
        return
    _summary_printed = True

    if not _enrichment_counter:
        print("Enrichment summary: no enrichments fired during this run.",
              file=sys.stderr)
    else:
        total = sum(_enrichment_counter.values())
        print(f"=== ENRICHMENT FIRE SUMMARY ({total} total fires) ===",
              file=sys.stderr)
        for name, count in _enrichment_counter.most_common():
            print(f"  {name:<30s} {count:4d} fires", file=sys.stderr)

    # If called from SIGTERM handler, re-raise so the process exits
    if _args:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        raise SystemExit(143)


atexit.register(_print_enrichment_summary)
signal.signal(signal.SIGTERM, _print_enrichment_summary)


@dataclass
class EnrichmentRule:
    """A single adaptive enrichment rule."""
    name: str
    triggers: list[str]          # regex patterns (case-insensitive)
    enrichment: str              # text to append to user prompt
    _compiled: list[re.Pattern] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._compiled = [re.compile(t, re.IGNORECASE) for t in self.triggers]

    def matches(self, instruction: str) -> bool:
        """Return True if ANY trigger matches the instruction."""
        return any(p.search(instruction) for p in self._compiled)


# ─── Enrichment rule definitions ─────────────────────────────────────────

ENRICHMENT_RULES: list[EnrichmentRule] = [

    # ── 1. "in front of" = +z (NOT stacking, NOT same z) ──
    # Run 173900 failures: 10 rounds (categories 1+2 in stack_issue.md)
    # "in front of the leftmost purple block" → z+100, NOT same z, NOT on top
    EnrichmentRule(
        name="in_front_of",
        triggers=[r"\bin front\b", r"\bfront of\b"],
        enrichment=(
            'REMINDER — "in front of" means z=z+100 or closer/nearer:\n'
            "If a block or stack is at (x, y=50, z), then \"in front of\" it = (x, 50, z+100). "
            "Only z changes. y resets to 50 (ground level for new stack).\n"
            "Example: purple at (x=-200, z=0) → \"in front of\" = (x=-200, z=100). "
            'A common error made with "in front of" is to keep the same z coordinate and stack vertically (on top of the existing block), which is incorrect.\n'
        ),
    ),

    # ── 2. "behind" = -z ──
    # Run 173900 failures: appears in chain reference patterns
    EnrichmentRule(
        name="behind",
        triggers=[r"\bbehind\b"],
        enrichment=(
            'REMINDER — "behind" means z=z-100": or farther away\n'
            "If a block or stack is at (x, y=50, z), then \"behind\" it = (x, 50, z-100). "
            "Only z changes. y resets to 50 for new stack.\n"
            "Example: green at (0,50,0) → \"behind\" = (0,50,-100). "
            "\"to the right of the thing behind\" = (100,50,-100).\n"
        ),
    ),

    # ── 3. "each end" / "both ends" after row extension ──
    # Run 165834 failures: 4 rounds — model uses OLD endpoints after extension
    # The model puts the right cap at x=100 (old end) instead of x=300 (new end)
    EnrichmentRule(
        name="each_end_of_row",
        triggers=[r"\beach end\b", r"\bboth ends\b"],
        enrichment=(
            'REMINDER — "each end" / "both ends" requires finding the ACTUAL endpoints AFTER all modifications:\n'
            "Step-by-step reasoning you MUST follow:\n"
            "1. First, perform the extension (add blocks to the row).\n"
            "2. Then, list ALL blocks in the COMPLETED row.\n"
            "3. Find the minimum and maximum positions along the row axis.\n"
            "4. Those min and max positions are the NEW two ends.\n"
            "5. Place blocks on top of THOSE positions.\n\n"
            "Worked example:\n"
            "  Original row: Red at x=[-100, 0, 100] at z=0.\n"
            '  Instruction: "Extend by 2 to the right. Place block on top of each end."\n'
            "  After extending right: new blocks at x=200, x=300.\n"
            "  FULL row is now: x=[-100, 0, 100, 200, 300].\n"
            "  The ends of the FULL row: min x=-100, max x=300.\n"
            "  WRONG: using old end x=100 — that is in the MIDDLE of the extended row!\n"
            "  CORRECT: caps at (-100,150,0) and (300,150,0).\n"
        ),
    ),

    # ── 4. L-shape: extend AWAY from junction, NEVER stack ──
    # Run 190047 failures: 4/4 — model extended THROUGH junction (wrong direction)
    # CRITICAL BUG FIX: old enrichment had wrong example direction
    EnrichmentRule(
        name="l_shape_extend",
        triggers=[r"\bl[- ]?shape\b", r"\bshape of an l\b"],
        enrichment=(
            "REMINDER — L-shape extension direction:\n"
            "The JUNCTION (corner) is where the two arms of the L meet.\n"
            '"Extend the longer side" = add blocks at the end FARTHEST FROM THE JUNCTION.\n'
            "Do NOT extend through the junction — that would break the L-shape.\n"
            "Example: L at [(0,50,-100),(0,50,0),(0,50,100),(100,50,100)].\n"
            "  Longer arm: z=-100 to z=100 (3 blocks). Short arm: x=100 (1 block). Junction at z=100.\n"
            "  Extend by 2 → AWAY from junction → (0,50,-200) and (0,50,-300).\n"
            "  NOT (0,50,200) — that goes through the junction!\n"
            '"Add to shorter side" → continue outward: (200,50,100).\n'
            "ALL extensions at y=50. NEVER stack vertically.\n"
        ),
    ),

    # ── 5. "leftmost"/"rightmost" + directional placement ──
    # Run 173900 failures: appears in "in front of the leftmost" pattern
    EnrichmentRule(
        name="most_ref_direction",
        triggers=[r"\bleftmost\b", r"\brightmost\b"],
        enrichment=(
            "REMINDER — \"leftmost\" = block with smallest x; \"rightmost\" = largest x:\n"
            "\"In front of the leftmost block\" = find block with min x, then place at "
            "(that_x, 50, that_z + 100). x stays the same, only z increases.\n"
            "Example: row at x=[-200,-100,0] at z=0. Leftmost = (-200,50,0). "
            "\"in front of leftmost\" = (-200, 50, 100).\n"
        ),
    ),

    # ── 6. Corner coordinates ──
    EnrichmentRule(
        name="corner_directions",
        triggers=[r"\btop left\b", r"\btop right\b", r"\bbottom left\b", r"\bbottom right\b",
                  r"\bcorner\b", r"\beach corner\b"],
        enrichment=(
            "REMINDER — corner coordinates (all at ground level y=50):\n"
            "\"bottom left corner\" = (x=-400, y=50, z=400)\n"
            "\"bottom right corner\" = (x=400, y=50, z=400)\n"
            "\"top left corner\" = (x=-400, y=50, z=-400)\n"
            "\"top right corner\" = (x=400, y=50, z=-400)\n"
            "\"top/bottom\" = map view — \"top\" = far/back (-z), \"bottom\" = near/front (+z).\n"
            "\"in front of\" top left (-400,50,-400) = (-400, 50, -300) [z+100].\n"
        ),
    ),

    # ── 7. Chain reference: "the X one" = the JUST-BUILT structure ──
    # Run 190047 failures: 7 rounds — model stacks ON TOP of row blocks instead of
    # placing at the position AFTER the row's last block
    EnrichmentRule(
        name="chain_reference",
        triggers=[
            r"\bthe\s+\w+\s+one\b",
            r"\bthe\s+\w+\s+ones\b",
            r"\bthe\s+\w+\s+stack\b",
            r"\bthe\s+\w+\s+tower\b",
            r"\byou just\s+(?:built|placed)\b",
            r"\bto the (?:right|left) of the\b.*\brow\b",
            r"\bto the (?:right|left) of the\b.*\bline\b",
            r"\bright of the\s+\w+\s+row\b",
        ],
        enrichment=(
            'REMINDER — "the [color] one/stack/tower" or "you just built" = the structure JUST built:\n'
            "When step B says \"to the right of the green one,\" it refers to the "
            "green blocks built in the PREVIOUS step — at that step's position, NOT the original grid.\n"
            "Example: original green at (0,50,0). Step A builds 3 green behind → (0,z=-100). "
            "Step B: \"right of the green one\" = (100, z=-100), NOT (100, z=0).\n"
            '"To the right of a ROW" = at position (row\'s rightmost x + 100, same z), starting at y=50.\n'
            "Do NOT stack on top of a block IN the row. The new stack goes AFTER the row's last block.\n"
            "Example: row at x=[100,200,300]. \"Stack to the right of this row\" → x=400, y=50.\n"
            "NOT x=200 y=150 — that is stacking ON TOP, which is wrong.\n"
        ),
    ),

    # ── 8. "extend" a row/line (not L-shape) ──
    EnrichmentRule(
        name="extend_row",
        triggers=[r"\bextend\b(?!.*\bl[- ]?shape\b)"],
        enrichment=(
            "REMINDER — \"extend\" a row = add blocks CONTINUING the row direction:\n"
            "Find the current row direction and add blocks at the next positions.\n"
            "\"Extend by adding two to its right\" from row ending at x=100 "
            "→ new blocks at x=200 and x=300. NOT stacking on top.\n"
        ),
    ),

    # ── 9. "on top of" after a horizontal reference ──
    EnrichmentRule(
        name="on_top_specific_block",
        triggers=[r"\bon top of\b.*\b(?:last|that|middle|second|leftmost|rightmost|each end|each)\b"],
        enrichment=(
            "REMINDER — \"on top of [specific block]\" means stack VERTICALLY at that block's (x,z):\n"
            "If the referenced block is at (x, y, z), new block goes at (x, y+100, z). "
            "x and z stay the same.\n"
            "\"on top of the second block\" in row [(0,50,0),(-100,50,0),(-200,50,0)] "
            "→ second block is (-100,50,0), stack at (-100, 150, 0).\n"
        ),
    ),

    # ── 10. "starting from/at" + "going to/towards" ──
    EnrichmentRule(
        name="starting_going",
        triggers=[r"\bstarting\s+(?:from|at)\b.*\bgoing\s+(?:to|towards)\b"],
        enrichment=(
            "REMINDER — \"starting from X going to the left\" = place blocks with decreasing x:\n"
            "The FIRST block is placed AT the starting position. "
            "Subsequent blocks step in the stated direction.\n"
            "\"Starting at origin going left, 3 blocks\" → (0,50,0), (-100,50,0), (-200,50,0).\n"
            "\"Leftmost\" after this is (-200,50,0). \"In front of leftmost\" = (-200,50,100).\n"
        ),
    ),

    # ── 11. "horizontal row" = same y, not stacking ──
    EnrichmentRule(
        name="horizontal_row_from_stack",
        triggers=[r"\bhorizontal\s+row\b"],
        enrichment=(
            "REMINDER — \"horizontal row\" means blocks placed along x or z axis at y=50 (ground):\n"
            "A horizontal row is NOT stacking. Each block in the row is at the SAME y.\n"
            "\"horizontal row of two blocks to the right\" from (100,50,0) "
            "→ blocks at (200,50,0) and (300,50,0), all at y=50.\n"
        ),
    ),

    # ── 12. stack / tower = vertical column at one (x,z) ──
    EnrichmentRule(
        name="stack_definition",
        triggers=[r"\bstack\b", r"\btower\b"],
        enrichment=(
            'REMINDER — A "stack" or "tower" = blocks piled VERTICALLY at ONE (x,z) position:\n'
            "All blocks share the SAME x and z. Only y changes: y=50, 150, 250, 350, ...\n"
            "When placing a NEW stack next to an existing structure, the new stack starts at y=50 "
            "(ground level). Do NOT stack on top of the neighboring position.\n"
        ),
    ),

    # ── 13. stack/tower + directional placement (left/right/front/behind) ──
    # Combines stack and direction — the most common compound failure
    # Run 190047 failures: 3 rounds — "left side" confused with z-axis movement
    EnrichmentRule(
        name="stack_directional",
        triggers=[
            r"\bstack\b.*\b(?:left|right|behind)\b",
            r"\b(?:left|right|behind)\b.*\bstack\b",
            r"\bstack\b.*\bin front\b",
            r"\bin front\b.*\bstack\b",
            r"\btower\b.*\b(?:left|right|behind|in front)\b",
            r"\b(?:left|right|behind)\b.*\btower\b",
            r"\bin front\b.*\btower\b",
            r"\bleft side\b",
            r"\bright side\b",
        ],
        enrichment=(
            "REMINDER — Placing a stack/tower ADJACENT to another structure:\n"
            "The new stack is VERTICAL at the neighboring (x,z), starting at y=50 (ground).\n"
            "The new stack shares the reference's z (for left/right) or x (for front/behind).\n"
            "• LEFT / LEFT SIDE of ref at (x,z) → new stack at (x-100, z). z is UNCHANGED.\n"
            "• RIGHT / RIGHT SIDE of ref at (x,z) → new stack at (x+100, z). z is UNCHANGED.\n"
            "• IN FRONT of ref at (x,z) → new stack at (x, z+100). x is UNCHANGED.\n"
            "• BEHIND ref at (x,z) → new stack at (x, z-100). x is UNCHANGED.\n"
            '"Left side" = "to the left" = ALWAYS -x. NEVER change z for left/right.\n'
            "Do NOT confuse \"left\" with \"behind\" — left changes x, behind changes z.\n"
        ),
    ),

    # ── 14. T-shape anatomy + extending ──
    # Run 165834 failures: 4/4 — model stacks vertically instead of continuing
    # stem direction. It adds (0,50,300) correctly but then (0,150,200) instead of
    # (0,50,400). The key error is losing track of the extension direction.
    EnrichmentRule(
        name="t_shape",
        triggers=[r"\bt[- ]?shape\b"],
        enrichment=(
            "REMINDER — T-shape: identify parts, find axis, find end, then extend.\n\n"
            "Step 1. FIND THE JUNCTION: the block shared by two perpendicular lines.\n"
            "Step 2. IDENTIFY CROSSBAR vs STEM:\n"
            "  CROSSBAR = the bar with blocks on BOTH sides of junction (junction bisects it).\n"
            "  STEM = the bar with blocks on only ONE side of junction (junction is at one end).\n"
            "Step 3. FIND WHICH AXIS THE STEM RUNS ALONG:\n"
            "  If stem blocks differ in z but share x → stem runs along the z-axis.\n"
            "  If stem blocks differ in x but share z → stem runs along the x-axis.\n"
            "Step 4. FIND THE BASE (stem end farthest from junction) and EXTENSION DIRECTION:\n"
            "  From junction toward base, is the coordinate increasing or decreasing?\n"
            "  z increasing → direction = 'front'. z decreasing → direction = 'behind'.\n"
            "  x increasing → direction = 'right'. x decreasing → direction = 'left'.\n"
            "  Extension adds blocks PAST the base in this SAME direction. y=50 ALWAYS.\n"
            "  WRONG direction = 'on_top'. NEVER use on_top for extend_row. That stacks vertically.\n"
            "Step 5. ARMS = two halves of crossbar from junction to each tip.\n"
            "  'Add to each arm' = place 1 block outward from each tip, along crossbar axis. y=50.\n\n"
            "Worked example (stem along +z):\n"
            "  Crossbar along x at z=-100: (-100,50,-100), (0,50,-100), (100,50,-100)\n"
            "  Stem along z at x=0: (0,50,-100), (0,50,0), (0,50,100), (0,50,200)\n"
            "  Junction = (0,50,-100). Stem blocks differ in z, so stem runs along z-axis.\n"
            "  Base = (0,50,200). From junction z=-100 to base z=200: z increases → direction = 'front'.\n"
            "  Extend by 2 with direction='front': (0,50,300) and (0,50,400).\n"
            "    WRONG: direction='on_top' produces (0,150,200) — stacks vertically, breaks the T!\n"
            "  Arm tips: (-100,50,-100) and (100,50,-100). Outward along x → (-200,50,-100) and (200,50,-100).\n"
            "  JSON steps: extend_row direction='front', then two place steps for arm tips.\n"
        ),
    ),

    # ── 15. "towards the bottom" = +z ──
    EnrichmentRule(
        name="towards_bottom",
        triggers=[r"\btowards the bottom\b", r"\bbottom of the grid\b"],
        enrichment=(
            "REMINDER — \"towards the bottom of the grid\" = +z direction (increasing z):\n"
            "\"Bottom\" in map view = near/front = +z.\n"
            "\"Place blocks going towards the bottom\" = increase z: z, z+100, z+200, ...\n"
        ),
    ),
]


def get_enrichments(instruction: str) -> str:
    """Scan instruction for difficult concepts and return enrichment text.

    Returns an empty string if no enrichments are needed.
    The returned text should be appended to the user prompt, after the
    instruction, before the LLM call.
    """
    fired: list[str] = []
    for rule in ENRICHMENT_RULES:
        if rule.matches(instruction):
            fired.append(rule.enrichment)
            _enrichment_counter[rule.name] += 1
            logger.debug("Enrichment rule '%s' fired for: %s", rule.name, instruction[:60])

    if not fired:
        return ""

    return "\n--- ADAPTIVE REMINDERS (for this specific instruction) ---\n" + "\n".join(fired)


def get_fired_rule_names(instruction: str) -> list[str]:
    """Return list of rule names that fire for the given instruction.

    Useful for testing and logging.
    """
    return [rule.name for rule in ENRICHMENT_RULES if rule.matches(instruction)]
