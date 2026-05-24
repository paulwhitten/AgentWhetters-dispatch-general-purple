"""Web research skill -- single-call LLM with web_search for document QA.

Handles tasks where the primary challenge is finding and reasoning over
information available on the public web (e.g., OfficeQA Treasury Bulletin
questions, BrowseComp, general research queries).

Uses the OpenAI Responses API with the built-in web_search tool so the
model can search and retrieve web content natively, rather than manually
scraping via shell commands.
"""

from __future__ import annotations

import logging
from typing import Callable

from openai import AsyncOpenAI

from usage import tracker

logger = logging.getLogger(__name__)

STEP_LIMIT = 10

_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _REASONING_MODEL_PREFIXES)


SYSTEM_PROMPT = """\
You are an expert research analyst. Answer the question precisely using \
web search to find authoritative sources.

Search strategy:
- Search for the specific document, report, or dataset mentioned in the question.
- For U.S. Treasury data, search fraser.stlouisfed.org or treasury.gov.
- Cross-reference multiple sources when possible.
- Pay close attention to units (millions, billions, trillions), time periods, \
fiscal years vs calendar years, and numerical precision.

REQUIRED OUTPUT FORMAT:
<REASONING>
[Show your search queries, sources found, data extracted, and calculations]
</REASONING>
<FINAL_ANSWER>
[Your precise answer -- just the value, no extra text]
</FINAL_ANSWER>

You MUST wrap your final answer in <FINAL_ANSWER> tags or it will be scored \
as incorrect. If you cannot find the answer, still provide your best estimate \
in <FINAL_ANSWER> tags.
"""


async def solve_web_research(
    input_text: str,
    client: AsyncOpenAI,
    model: str,
    on_status: Callable | None = None,
) -> str:
    """Solve a research task using web search.

    Uses the Responses API with web_search_preview for reasoning models,
    falling back to a multi-step loop for non-reasoning models.
    """
    if on_status:
        await on_status("Researching with web search...")

    is_reasoning = _is_reasoning_model(model)

    tools: list[dict] = [{"type": "web_search_preview"}]

    api_kwargs: dict = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": [{"role": "user", "content": input_text}],
        "tools": tools,
        "store": False,
    }

    if is_reasoning:
        api_kwargs["reasoning"] = {"effort": "high", "summary": "auto"}
        api_kwargs["max_output_tokens"] = 16_000
    else:
        api_kwargs["temperature"] = 0.0
        api_kwargs["max_output_tokens"] = 4096

    try:
        response = await client.responses.create(**api_kwargs)
        tracker.record(response, label="web_research")
    except Exception as exc:
        logger.error("Web research API call failed: %s", exc)
        return f"<FINAL_ANSWER>Error: {exc}</FINAL_ANSWER>"

    # Extract text from response output
    result = _extract_text(response)

    if not result:
        logger.warning("Web research returned empty result, retrying without web search")
        # Fallback: try without web_search in case it's not supported
        api_kwargs.pop("tools", None)
        try:
            response = await client.responses.create(**api_kwargs)
            tracker.record(response, label="web_research_fallback")
            result = _extract_text(response)
        except Exception as exc:
            logger.error("Fallback API call failed: %s", exc)
            return f"<FINAL_ANSWER>Error: {exc}</FINAL_ANSWER>"

    if not result:
        return "<FINAL_ANSWER>Unable to determine answer</FINAL_ANSWER>"

    logger.info("Web research result (%d chars): %.200s...", len(result), result)
    return result


def _extract_text(response) -> str:
    """Extract text content from a Responses API response."""
    # Use output_text if available (simplest path)
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    # Fall back to iterating output items
    text_parts = []
    for item in response.output:
        if hasattr(item, "type") and item.type == "message":
            for content in getattr(item, "content", []):
                if hasattr(content, "text"):
                    text_parts.append(content.text)
        elif hasattr(item, "text"):
            text_parts.append(item.text)
    return "\n".join(text_parts)
