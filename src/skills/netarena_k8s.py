"""NetArena K8s skill -- multi-turn Kubernetes network policy debugging.

The K8s benchmark evaluates agents on fixing injected network policy
misconfigurations in a microservices environment. The green agent sends
connectivity status with mismatches; we respond with kubectl commands.

Protocol: multi-turn text-in / text-out over A2A (stateless from our side --
green provides full accumulated context each turn).
Metrics: correctness (all mismatches fixed), safety (didn't increase mismatches),
iterations (fewer is better).
"""

from __future__ import annotations

import logging
import textwrap

from openai import AsyncOpenAI

from usage import tracker

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert Kubernetes network engineer debugging network policy
    misconfigurations in Google's Online Boutique microservices demo.

    Architecture (desired communication):
    - User/loadgenerator -> frontend (HTTP)
    - frontend -> checkout, ad, recommendation, productcatalog, cart, shipping, currency, payment, email
    - checkout -> payment, shipping, email, currency
    - recommendation -> productcatalog
    - cart -> redis-cart

    Your task: Fix network policy mismatches by issuing kubectl commands.

    Rules:
    - Provide ONE command at a time
    - Use `kubectl patch` with `--type=json` or `--type=merge` (NOT `kubectl edit`)
    - Do NOT use: kubectl exec, kubectl create, kubectl apply, kubectl delete, kubectl describe, bash, sudo
    - Do NOT change policies that are already correct
    - First inspect the relevant policy with `kubectl get networkpolicy <name> -o yaml`
    - Then patch to fix the mismatch

    Common fixes:
    - "Expected: True, Actual: False" means traffic is blocked that should be allowed
      -> Add the missing ingress/egress rule
    - "Expected: False, Actual: True" means traffic is allowed that should be blocked
      -> Restrict the ingress/egress rules to only allow expected sources

    Response format: Put your command in a markdown code block:
    ```
    kubectl ...
    ```

    Think step by step about which policy needs fixing and what the correct
    rule should be, then provide exactly ONE command.
""")


def is_netarena_k8s(input_text: str) -> bool:
    """Detect if the input is a NetArena K8s query.

    K8s prompts contain connectivity mismatch information and network policy context.
    """
    markers = (
        "networkpolicy",
        "kubectl",
        "microservices",
        "Mismatch:",
        "connectivity status",
    )
    text_lower = input_text.lower()
    # Need at least 2 markers (case-insensitive for most, exact for Mismatch)
    count = 0
    for m in markers:
        if m.lower() in text_lower:
            count += 1
    return count >= 2


async def solve_netarena_k8s(
    input_text: str,
    client: AsyncOpenAI,
    model: str,
) -> str:
    """Solve a NetArena K8s query by generating kubectl commands.

    Each message from the green includes the full connectivity log.
    We respond with a single kubectl command in a code block.
    """
    logger.info("NetArena K8s: generating kubectl command")

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
        ],
        temperature=0.0,
        max_completion_tokens=2048,
    )

    result = response.choices[0].message.content or ""

    tracker.record(response, label="netarena-k8s")

    logger.info("NetArena K8s: response generated (%d chars)", len(result))
    return result
