"""Task classification -- structural pre-filter and LLM fallback."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from usage import tracker

logger = logging.getLogger(__name__)

TASK_TYPES = [
    "swe-bench",       # SWE-bench Pro: code repair with Docker container
    "cybersecurity",    # CyberGym: vulnerability analysis + PoC
    "game",             # BWIM: spatial building tasks
    "safety",           # Pi-Bench: policy compliance
    "finance",          # OfficeQA: financial QA
    "negotiation",      # MAizeBargAIn: multi-turn negotiation
    "mle-bench",        # MLE-Bench: Kaggle ML competitions
    "research",         # Mind2Web, BrowseComp+, FieldWorkArena
    "computer-use",     # CAR-bench, OSWorld: desktop/browser interaction
    "terminal-bench",    # Terminal Bench 2.0: containerized shell tasks
    "netarena-malt",    # NetArena MALT: data center capacity planning code gen
    "netarena-k8s",     # NetArena K8s: Kubernetes network policy debugging
    "coding",           # Terminal Bench, NetArena: general coding tasks
    "general",          # Fallback for unrecognized tasks
]

CLASSIFIER_PROMPT = """You are a task classifier for an AI agent competition. Given a task message, classify it into exactly one category.

Categories:
- swe-bench: Software engineering bug fix / feature request with a code repository. Usually contains instance_id, problem_statement, docker_image fields.
- cybersecurity: Vulnerability analysis, exploit development, CVE analysis, proof-of-concept code.
- game: Spatial building, grid-based construction, block placement (e.g., Minecraft-style).
- safety: Policy compliance, regulation checking, content moderation decisions.
- finance: Financial analysis, treasury, fiscal, revenue questions over documents/data.
- negotiation: Multi-turn bargaining, offers, counteroffers, deal-making scenarios.
- research: Data science, machine learning competitions, web research, information retrieval, form filling.
- mle-bench: MLE-Bench Kaggle competition with competition.tar.gz data and instructions about submission.csv.
- computer-use: Desktop automation, browser interaction, GUI navigation tasks.
- terminal-bench: Terminal Bench 2.0 tasks using terminal-bench-shell-v1 JSON protocol.
- netarena-malt: Data center network capacity planning tasks that ask for Python code to process networkx graphs (process_graph function, EK_PORT, physical_capacity_bps).
- netarena-k8s: Kubernetes network policy debugging with connectivity mismatches and kubectl commands.
- coding: General programming tasks, terminal commands, network configuration (not SWE-bench format).
- general: Anything that doesn't clearly fit the above categories.

Respond with ONLY the category name, nothing else."""


def classify_structural(input_text: str) -> str | None:
    """Structural pre-filter: detect task type from message format.

    Returns a task type string if the format is unambiguous,
    or None if LLM classification is needed.
    """
    text_lower = input_text.lower()

    # MLE-Bench: Kaggle ML competition instructions with tar.gz
    if "mle-bench" in text_lower and "submission" in text_lower:
        return "mle-bench"

    # SWE-bench Pro: JSON with instance_id + problem_statement + docker_image
    if '"instance_id"' in text_lower and '"problem_statement"' in text_lower:
        return "swe-bench"

    # BWIM: characteristic markers
    if "[task_description]" in text_lower and "[start_structure]" in text_lower:
        return "game"

    # MAizeBargAIn: JSON with negotiation fields
    if '"batna"' in text_lower and '"valuations"' in text_lower:
        return "negotiation"

    # Terminal Bench 2.0: JSON protocol with terminal-bench-shell-v1
    if '"terminal-bench-shell-v1"' in input_text:
        return "terminal-bench"

    # NetArena MALT: data center capacity planning graph queries
    if "process_graph" in input_text and "physical_capacity_bps" in input_text:
        return "netarena-malt"

    # NetArena K8s: Kubernetes network policy debugging
    if "networkpolicy" in text_lower and ("mismatch" in text_lower or "kubectl" in text_lower):
        return "netarena-k8s"

    # Terminal Bench 2.0: fallback detection via exec URL
    if "/exec/" in input_text and (
        "terminal" in text_lower or "command" in text_lower
    ):
        return "coding"

    # CAR-bench: automotive tool-calling with ToolCallsData
    if '"toolcallsdata"' in text_lower or "tool_definitions" in text_lower:
        return "computer-use"

    # Pi-Bench: policy + record_decision tool
    if "record_decision" in text_lower and (
        "allow" in text_lower
        or "deny" in text_lower
        or "escalate" in text_lower
    ):
        return "safety"

    # OfficeQA: Treasury/fiscal document QA
    if "<final_answer>" in text_lower or "treasury bulletin" in text_lower:
        return "finance"

    # OfficeQA: fiscal/treasury questions about specific financial metrics
    fiscal_terms = ("fiscal year", "public debt", "federal debt", "treasury",
                    "national debt", "budget deficit", "budget surplus",
                    "government receipts", "government expenditures")
    if any(term in text_lower for term in fiscal_terms):
        return "finance"

    # CyberGym: vulnerability analysis with test_vulnerable action
    if "test_vulnerable" in text_lower and (
        "proof-of-concept" in text_lower
        or "poc" in text_lower
        or "vulnerability" in text_lower
    ):
        return "cybersecurity"

    return None


async def classify_llm(
    input_text: str,
    client: AsyncOpenAI,
    model: str,
) -> str:
    """Use the cheap LLM to classify ambiguous tasks."""
    truncated = input_text[:4000] if len(input_text) > 4000 else input_text
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": truncated},
            ],
            max_tokens=20,
            temperature=0.0,
        )
        tracker.record(response, label="classifier")
        result = (response.choices[0].message.content or "").strip().lower()
        if result in TASK_TYPES:
            return result
        logger.warning("LLM classifier returned unknown type: %s", result)
        return "general"
    except Exception as e:
        logger.warning("LLM classification failed: %s", e)
        return "general"
