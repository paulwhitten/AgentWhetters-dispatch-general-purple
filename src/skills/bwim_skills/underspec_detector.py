"""Detect underspecified instructions (missing color or number).

Strategy:
- Missing NUMBER is worth asking about when the color is already known.
  Empirical analysis shows heuristic count inference is only 64.6%
  accurate (0% on variant_a trials). Asking costs -5 but a correct
  answer earns +10, so net +5 beats guessing (EV +2.9).
- Missing COLOR is only worth asking if the color is genuinely
  unpredictable, that is, a block-placing phrase has no color and we
  cannot infer it from any color mentioned elsewhere in the instruction.
  If the instruction mentions exactly one color, we can reuse it.
- Only one question is allowed per round. Priority: color > count.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from .grid import VALID_COLORS

logger = logging.getLogger(__name__)

# Color words that may appear in instructions (lowercase for matching)
_COLOR_WORDS = {c.lower() for c in VALID_COLORS}

# Map of word numbers to ints
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


@dataclass
class UnderspecResult:
    """Result of underspecification detection."""
    has_missing_color: bool = False
    has_missing_number: bool = False
    suggested_question: str = ""
    suggested_count_question: str = ""
    details: str = ""
    # Colors found in the instruction that can be used for inference
    colors_in_instruction: list[str] = field(default_factory=list)
    # Suggested default color if we decide not to ask
    inferred_color: str = ""
    # Suggested default count if we decide not to ask
    inferred_count: int = 0
    # Color of the first uncounted phrase (used to ask a specific question)
    uncounted_color: str = ""
    # All uncounted phrases with their colors: [(color, phrase_text), ...]
    uncounted_phrases: list[tuple[str, str]] = field(default_factory=list)
    # Compound question when both color and count are missing
    suggested_compound_question: str = ""


def _extract_colors_from_text(text: str) -> list[str]:
    """Extract all color words from text in order of appearance."""
    colors = []
    for m in re.finditer(r'\b(' + '|'.join(_COLOR_WORDS) + r')\b', text.lower()):
        colors.append(m.group(1).capitalize())
    return colors


def _count_block_placing_phrases(text: str) -> list[dict]:
    """Parse block-placing phrases and determine which have/lack color and count."""
    text_lower = text.lower()
    phrases = []

    # Match phrases like "stack N color blocks", "build a color stack",
    # "place two blocks", "stack blocks", etc.
    patterns = [
        # "stack/place/build/add/put [N] [color] blocks/stack/tower"
        r'(?:stack|place|build|add|put|extend)\s+'
        r'(?:a\s+)?'
        r'(?:(?:tower|row|stack|line|horizontal\s+row)\s+of\s+)?'
        r'(?:(?:\d+|one|two|three|four|five|six|seven|eight|nine)\s+)?'
        r'(?:(?:' + '|'.join(_COLOR_WORDS) + r')\s+)?'
        r'(?:blocks?|stack|tower|row)\b',
        # "build a [color] stack/tower"
        r'(?:build|finish\s+with)\s+(?:a\s+)?'
        r'(?:(?:' + '|'.join(_COLOR_WORDS) + r')\s+)?'
        r'(?:stack|tower)',
    ]

    for pattern in patterns:
        for m in re.finditer(pattern, text_lower):
            phrase = m.group()
            has_color = any(c in phrase for c in _COLOR_WORDS)
            has_number = bool(re.search(
                r'\b(?:\d+|one|two|three|four|five|six|seven|eight|nine)\b', phrase
            ))
            phrases.append({
                'phrase': phrase,
                'has_color': has_color,
                'has_number': has_number,
                'start': m.start(),
                'end': m.end(),
            })

    # Post-processing: detect counts in "of N blocks" patterns that follow
    # the matched phrase.  Handles "a yellow stack of three blocks" where
    # the regex captures "build a yellow stack" but misses the trailing
    # "of three blocks".
    _COUNT_AFTER = re.compile(
        r'\s+of\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine)\b'
    )
    for p in phrases:
        if not p['has_number']:
            lookahead = text_lower[p['end']:p['end'] + 30]
            if _COUNT_AFTER.match(lookahead):
                p['has_number'] = True

    return phrases


def detect_underspec_heuristic(instruction: str) -> UnderspecResult:
    """Detect missing colors or numbers using regex heuristics.

    Only flags genuinely missing colors that cannot be inferred.
    Never flags missing numbers — those are resolved with heuristics.
    """
    result = UnderspecResult()
    text = instruction.lower()

    # Extract all colors mentioned anywhere in the instruction
    all_colors = _extract_colors_from_text(text)
    result.colors_in_instruction = all_colors
    unique_colors = list(dict.fromkeys(all_colors))  # dedupe, preserve order

    # Parse block-placing phrases
    phrases = _count_block_placing_phrases(text)
    colored = [p for p in phrases if p['has_color']]

    # Check for phrases without color — but filter out sub-phrases that
    # are contained within a longer colored phrase (e.g., "build a stack"
    # inside "build a stack of three blue blocks") and phrases whose
    # lookahead context already contains a color word.
    colorless_phrases = [p for p in phrases if not p['has_color']]

    # Filter: remove sub-phrases overlapping with colored phrases
    def _overlaps(cp: dict) -> bool:
        for col in colored:
            if cp['start'] >= col['start'] and cp['end'] <= col['end']:
                return True
            if col['start'] >= cp['start'] and col['end'] <= cp['end']:
                return True
        return False

    colorless_phrases = [p for p in colorless_phrases if not _overlaps(p)]

    # Filter: skip phrases whose 30-char lookahead already has a color
    def _context_has_color(p: dict) -> bool:
        lookahead = text[p['end']:p['end'] + 30]
        for c in _COLOR_WORDS:
            if re.search(r'\b' + c + r'\b', lookahead):
                return True
        return False

    colorless_phrases = [p for p in colorless_phrases if not _context_has_color(p)]

    if colorless_phrases:
        if len(unique_colors) == 0:
            # No colors at all in the instruction — truly unknown
            result.has_missing_color = True
            result.details = "No color words found in instruction at all. "
            result.suggested_question = "What color should the unspecified blocks be?"
        elif len(unique_colors) == 1:
            # Single color mentioned but some phrases lack color.
            # Always ask — at 85% build success, asking (+5 net) beats
            # inference (~54% correct → expected -10×0.46 + 10×0.54 = +0.8).
            # Keep inferred_color as fallback for the answer-path patcher.
            result.has_missing_color = True
            result.inferred_color = unique_colors[0]
            result.details = (
                f"Single color '{unique_colors[0]}' but colorless phrases present. "
                f"Asking is +EV vs inference. "
            )
            result.suggested_question = (
                "What color should the unspecified blocks be?"
            )
        else:
            # Multiple colors mentioned but some phrases lack color.
            # The missing color is genuinely unpredictable — ASK.
            # Data shows color_under targets always use a color NOT in the
            # instruction, so guessing is ~0% accurate.  Asking costs -5
            # but correct answer earns +10 → net +5 beats guessing (-10).
            result.has_missing_color = True
            result.details = (
                f"Multiple colors ({unique_colors}) but some phrases lack color. "
                f"Must ask — cannot reliably infer. "
            )
            result.suggested_question = (
                "What color should the unspecified blocks be?"
            )

    # Check for phrases without number.
    # Only flag as genuinely missing when the uncounted phrase uses plural
    # "blocks" or a bare group noun (stack/tower). Singular "a block" means
    # count=1 and is not missing.
    numbered_phrases = [p for p in phrases if p['has_number']]
    numberless_phrases = [p for p in phrases if not p['has_number']]

    # Remove numberless sub-phrases that overlap with numbered phrases
    # (e.g. "build a stack" inside "build a stack of three yellow blocks")
    def _overlaps_numbered(cp: dict) -> bool:
        for np in numbered_phrases:
            if cp['start'] >= np['start'] and cp['end'] <= np['end']:
                return True
            if np['start'] >= cp['start'] and np['end'] <= cp['end']:
                return True
        return False

    numberless_phrases = [p for p in numberless_phrases if not _overlaps_numbered(p)]

    # Filter numberless phrases to those with genuinely unknown counts
    truly_missing = []
    for p in numberless_phrases:
        pt = p['phrase']
        # Plural "blocks" without a number → genuinely unknown count
        if re.search(r'\bblocks\b', pt):
            truly_missing.append(p)
        # Bare group noun (stack/tower) without "block/blocks" → unknown
        elif re.search(r'\b(stack|tower)\b', pt) and not re.search(r'\bblocks?\b', pt):
            truly_missing.append(p)
        # Singular "block" (often preceded by "a") → count is 1, not missing

    if truly_missing:
        result.has_missing_number = True
        # Default: match the count of the first specified stack, or use 3
        # Extract the first explicit count from the instruction
        count_match = re.search(
            r'\b(?:(\d+)|(one|two|three|four|five|six|seven|eight|nine))\b', text
        )
        if count_match:
            if count_match.group(1):
                result.inferred_count = int(count_match.group(1))
            else:
                result.inferred_count = _WORD_NUMBERS.get(count_match.group(2), 3)
        else:
            result.inferred_count = 3  # reasonable default

        # Extract the color of each uncounted phrase so the question
        # can reference it.  This helps the green agent (which only sees
        # coordinates) identify the correct stack.
        for p in truly_missing:
            phrase_colors = []
            for c in _COLOR_WORDS:
                if re.search(r'\b' + c + r'\b', p['phrase']):
                    phrase_colors.append(c.capitalize())
            # Also check the 30-char context after the phrase for a color
            if not phrase_colors:
                lookahead = text[p['end']:p['end'] + 40]
                for c in _COLOR_WORDS:
                    if re.search(r'\b' + c + r'\b', lookahead):
                        phrase_colors.append(c.capitalize())
                        break
            color = phrase_colors[0] if phrase_colors else ""
            result.uncounted_phrases.append((color, p['phrase']))

        # Set the primary uncounted color (first phrase with a color)
        for color, _phrase in result.uncounted_phrases:
            if color:
                result.uncounted_color = color
                break

        result.details += (
            f"Missing count detected, inferred as {result.inferred_count} "
            f"(matched from instruction or default). "
        )
        if result.uncounted_phrases:
            result.details += (
                f"Uncounted phrases: "
                f"{[(c, p) for c, p in result.uncounted_phrases]}. "
            )

        # Generate a color-specific count question so the green agent
        # can identify the correct stack from its coordinate data.
        if result.uncounted_color:
            color_lower = result.uncounted_color.lower()
            result.suggested_count_question = (
                f"How many {color_lower} blocks should be in the "
                f"{color_lower} stack?"
            )
        else:
            result.suggested_count_question = (
                "How many blocks should be in the unspecified stack?"
            )

    # Generate compound question when both color and count are missing.
    # This allows asking for both in a single -5 cost round.
    if result.has_missing_color and result.has_missing_number:
        result.suggested_compound_question = (
            "What color should the unspecified blocks be, "
            "and how many blocks should be in that stack?"
        )

    return result


def patch_instruction_with_color(instruction: str, color: str) -> str:
    """Insert a color word into colorless block-placing phrases.

    Used to patch the instruction BEFORE sending to the LLM, so the planner
    receives a fully-specified instruction and doesn't have to guess colors.

    Examples:
        >>> patch_instruction_with_color(
        ...     "Build a tower of four blocks in front", "Blue")
        'Build a tower of four blue blocks in front'
        >>> patch_instruction_with_color(
        ...     "Then place a block on top", "Red")
        'Then place a red block on top'
    """
    phrases = _count_block_placing_phrases(instruction)
    colored = [p for p in phrases if p['has_color']]
    colorless = [p for p in phrases if not p['has_color']]

    if not colorless:
        return instruction

    # Remove colorless phrases that overlap with a colored phrase
    def _overlaps_colored(cp: dict) -> bool:
        for col in colored:
            if cp['start'] >= col['start'] and cp['end'] <= col['end']:
                return True
            if col['start'] >= cp['start'] and col['end'] <= cp['end']:
                return True
        return False

    colorless = [p for p in colorless if not _overlaps_colored(p)]

    if not colorless:
        return instruction

    # Skip phrases whose immediate context already contains a color
    # e.g. "build a tower" when the full clause is "build a tower of three blue blocks"
    def _context_has_color(p: dict) -> bool:
        lookahead = instruction[p['end']:p['end'] + 30].lower()
        for c in _COLOR_WORDS:
            if re.search(r'\b' + c + r'\b', lookahead):
                return True
        return False

    colorless = [p for p in colorless if not _context_has_color(p)]

    if not colorless:
        return instruction

    # Deduplicate overlapping — keep the longest match at each start position
    by_start: dict[int, dict] = {}
    for p in colorless:
        s = p['start']
        if s not in by_start or (p['end'] - p['start']) > (by_start[s]['end'] - by_start[s]['start']):
            by_start[s] = p
    unique = sorted(by_start.values(), key=lambda p: p['start'], reverse=True)

    # Insert color word right before the final noun (blocks/block/stack/tower/row)
    # Process right-to-left so earlier positions stay valid
    result = instruction
    color_lower = color.lower()
    for p in unique:
        phrase_text = result[p['start']:p['end']]
        m = re.search(r'\b(blocks?|stack|tower|row)\s*$', phrase_text, re.IGNORECASE)
        if m:
            insert_pos = p['start'] + m.start()
            result = result[:insert_pos] + color_lower + ' ' + result[insert_pos:]

    return result


def patch_instruction_with_count(instruction: str, count: int,
                                 target_color: str = "") -> str:
    """Insert a count into uncounted block-placing phrases.

    Used to patch the instruction BEFORE sending to the LLM, so the planner
    receives a fully-specified instruction and does not have to guess counts.

    If *target_color* is provided, only patches the phrase that matches
    that color.  Otherwise patches all uncounted phrases.

    Examples:
        >>> patch_instruction_with_count(
        ...     "Build a blue stack in front of the yellow one.", 3)
        'Build a stack of 3 blue blocks in front of the yellow one.'
        >>> patch_instruction_with_count(
        ...     "Finish with a yellow stack on the left.", 4, "yellow")
        'Finish with a stack of 4 yellow blocks on the left.'
    """
    phrases = _count_block_placing_phrases(instruction)
    numbered = [p for p in phrases if p['has_number']]
    numberless = [p for p in phrases if not p['has_number']]

    if not numberless:
        return instruction

    # Remove sub-phrases overlapping with numbered phrases
    def _overlaps_numbered(cp: dict) -> bool:
        for np in numbered:
            if cp['start'] >= np['start'] and cp['end'] <= np['end']:
                return True
            if np['start'] >= cp['start'] and np['end'] <= cp['end']:
                return True
        return False

    numberless = [p for p in numberless if not _overlaps_numbered(p)]

    # Filter: only genuinely missing counts (plural "blocks" or bare group noun)
    truly_missing = []
    for p in numberless:
        pt = p['phrase']
        if re.search(r'\bblocks\b', pt):
            truly_missing.append(p)
        elif re.search(r'\b(stack|tower)\b', pt) and not re.search(r'\bblocks?\b', pt):
            truly_missing.append(p)

    if not truly_missing:
        return instruction

    # If target_color specified, only patch phrases with that color
    if target_color:
        tc = target_color.lower()
        matching = [p for p in truly_missing if tc in p['phrase']]
        # Also check the phrase context (color word might be just before
        # the matched phrase, for example "a blue stack")
        if not matching:
            for p in truly_missing:
                lookahead = instruction[p['end']:p['end'] + 40].lower()
                if re.search(r'\b' + tc + r'\b', lookahead):
                    matching.append(p)
        if matching:
            truly_missing = matching

    # Deduplicate: keep longest match at each start position
    by_start: dict[int, dict] = {}
    for p in truly_missing:
        s = p['start']
        if s not in by_start or (p['end'] - p['start']) > (by_start[s]['end'] - by_start[s]['start']):
            by_start[s] = p
    unique = sorted(by_start.values(), key=lambda p: p['start'], reverse=True)

    # Rewrite each uncounted phrase: "build a blue stack" -> "build a stack of 3 blue blocks"
    # Strategy: insert count before the color or before the final noun
    result = instruction
    for p in unique:
        phrase_text = result[p['start']:p['end']]
        # Pattern: "[verb] [a] [color] stack/tower" -> "[verb] a stack of N [color] blocks"
        # Or: "[verb] [a] [color] blocks" -> "[verb] N [color] blocks"
        noun_m = re.search(r'\b(blocks?|stack|tower|row)\s*$', phrase_text, re.IGNORECASE)
        if not noun_m:
            continue

        # Find color within the phrase
        phrase_color = ""
        for c in _COLOR_WORDS:
            if re.search(r'\b' + c + r'\b', phrase_text.lower()):
                phrase_color = c
                break

        # Rebuild as: same verb prefix + "a stack of N [color] blocks"
        noun_word = noun_m.group(1).lower()
        if noun_word in ("stack", "tower"):
            # "build a blue stack" -> "build a stack of 3 blue blocks"
            # Find the start of the noun phrase (skip verb)
            color_and_noun = phrase_text[noun_m.start():].strip()
            prefix = phrase_text[:noun_m.start()].rstrip()
            if phrase_color:
                new_noun = f"a stack of {count} {phrase_color} blocks"
                # Remove color from prefix if it's there
                prefix = re.sub(r'\b' + phrase_color + r'\s*', '', prefix, flags=re.IGNORECASE).rstrip()
                # Remove "a " from prefix if present (we add it in new_noun)
                prefix = re.sub(r'\ba\s*$', '', prefix).rstrip()
            else:
                new_noun = f"a stack of {count} blocks"
                prefix = re.sub(r'\ba\s*$', '', prefix).rstrip()
            replacement = prefix + " " + new_noun
        else:
            # "stack blue blocks" -> "stack 3 blue blocks"
            # Insert count before the color or before "blocks"
            if phrase_color:
                color_m = re.search(r'\b' + phrase_color + r'\b', phrase_text, re.IGNORECASE)
                if color_m:
                    insert_pos = color_m.start()
                    replacement = phrase_text[:insert_pos] + f"{count} " + phrase_text[insert_pos:]
                else:
                    insert_pos = noun_m.start()
                    replacement = phrase_text[:insert_pos] + f"{count} " + phrase_text[insert_pos:]
            else:
                insert_pos = noun_m.start()
                replacement = phrase_text[:insert_pos] + f"{count} " + phrase_text[insert_pos:]

        result = result[:p['start']] + replacement + result[p['end']:]

    return result


def detect_underspec_from_plan(plan_steps: list[dict]) -> UnderspecResult:
    """Detect underspecification from LLM plan output.

    Checks if the LLM output contains 'Uncolored' or 'Uncounted' placeholders.
    For 'Uncounted', provides a heuristic count instead of flagging to ask.
    For 'Uncolored', only flags if genuinely unpredictable.
    """
    result = UnderspecResult()

    # Collect all specified colors from the plan for inference
    plan_colors = []
    plan_counts = []
    for step in plan_steps:
        color = str(step.get("color", ""))
        count = step.get("count", "")
        if color.lower() not in ("uncolored", "unknown", "unspecified", "?", ""):
            plan_colors.append(color.capitalize())
        if isinstance(count, int) or (isinstance(count, str) and count.isdigit()):
            plan_counts.append(int(count))

    unique_plan_colors = list(dict.fromkeys(plan_colors))

    has_uncolored = False
    has_uncounted = False

    for step in plan_steps:
        color = str(step.get("color", ""))
        count = step.get("count", "")

        if color.lower() in ("uncolored", "unknown", "unspecified", "?"):
            has_uncolored = True

        if str(count).lower() in ("uncounted", "unknown", "unspecified", "?"):
            has_uncounted = True

    # ── Color resolution ──
    if has_uncolored:
        if len(unique_plan_colors) == 1:
            # Only one color in the plan — safe to infer
            result.inferred_color = unique_plan_colors[0]
            result.details += (
                f"LLM flagged unspecified color, inferring "
                f"'{unique_plan_colors[0]}' (only color in plan). "
            )
        else:
            # Multiple colors OR no colors — cannot reliably guess.
            # Always ASK.  Guessing is ~0% accurate on color_under trials.
            result.has_missing_color = True
            result.details += (
                "LLM flagged unspecified color, must ask "
                f"(plan colors: {unique_plan_colors}). "
            )

    # ── Count resolution ──
    if has_uncounted:
        result.has_missing_number = True
        if plan_counts:
            result.inferred_count = plan_counts[-1]
        else:
            result.inferred_count = 3
        result.details += (
            f"LLM flagged unspecified count, inferred as "
            f"{result.inferred_count}. "
        )

    # ── Build question ──
    # Only one question per round. Priority: color > count.
    if result.has_missing_color:
        result.suggested_question = (
            "What color should the unspecified blocks be?"
        )
    # Generate a count question for pipeline to use when color is known.
    if result.has_missing_number:
        result.suggested_count_question = (
            "How many blocks should be in the unspecified stack?"
        )

    return result
