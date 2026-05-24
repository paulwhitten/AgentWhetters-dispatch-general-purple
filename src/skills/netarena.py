"""NetArena MALT skill -- single-turn code generation for network graph queries.

The MALT benchmark evaluates agents on data center capacity planning tasks.
The green agent sends a prompt asking for Python code (a `process_graph` function)
that manipulates a networkx graph. We respond with that code.

Protocol: single-turn text-in / text-out over A2A.
Metrics: correctness, safety (no graph corruption), latency.
"""

from __future__ import annotations

import logging
import textwrap

from openai import AsyncOpenAI

from usage import tracker

logger = logging.getLogger(__name__)

# System prompt to enhance code generation quality beyond what the green provides
SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert network engineer writing Python code for networkx graph queries.

    CRITICAL RULES:
    - Write a single `def process_graph(graph_data):` function
    - `copy`, `nx`, `json` are already imported -- do NOT add imports
    - Always start with: `graph_copy = copy.deepcopy(graph_data)`
    - Do NOT include example usage or main blocks
    - ALWAYS include 'updated_graph' in the return using the safety cleanup pattern shown below

    AVAILABLE HELPER FUNCTIONS (already in scope, use them directly):
    - solid_step_add_node_to_graph(graph_data, new_node, parent_node_name=None)
        new_node = {'name': 'node_name', 'type': 'EK_PACKET_SWITCH'}
        Returns modified graph_data
    - solid_step_remove_node_from_graph(graph_data, node_name)
        node_name is a string. Returns modified graph_data
    - solid_step_list_child_nodes(graph_data, parent_node)
        parent_node = {'name': 'node_name'}
        Returns list of child node name strings
    - solid_step_counting_query(graph_data, node1, node2=None)
        node1 = {'name': 'node_name', 'type': 'EK_PACKET_SWITCH'}
        node2 = {'type': 'EK_PORT'} or None
        If node2 is None: counts all nodes of node1's type in entire graph
        If node2 given: counts node2-type nodes contained within node1
        Returns integer
    - solid_step_rank_child_nodes(graph_data, parent_node_name)
        parent_node_name is a string
        Returns list of tuples: [(child_name, total_capacity_bps), ...] sorted descending
    - solid_step_update_node_value(graph_data, child_node_name, new_value)
        Updates physical_capacity_bps of an EK_PORT node. Returns modified graph_data

    RETURN FORMAT:
    - 'type': one of 'text', 'list', 'table', 'graph'
    - 'data': the answer data:
        * 'text' -> a string (convert numbers to str)
        * 'list' -> a Python list
        * 'table' -> list of lists (rows)
        * 'graph' -> the modified graph_copy object itself (NOT node_link_data)
    - 'updated_graph': ALWAYS include. Create a safety-cleaned copy:
        _sg = copy.deepcopy(graph_copy)
        _sg.remove_nodes_from(list(nx.isolates(_sg)))
        Then use nx.readwrite.json_graph.node_link_data(_sg)

    EXAMPLES:
    # Add a node:
    def process_graph(graph_data):
        graph_copy = copy.deepcopy(graph_data)
        new_node = {'name': 'new_EK_PACKET_SWITCH_1', 'type': 'EK_PACKET_SWITCH'}
        graph_copy = solid_step_add_node_to_graph(graph_copy, new_node, 'parent_name')
        _sg = copy.deepcopy(graph_copy)
        _sg.remove_nodes_from(list(nx.isolates(_sg)))
        return {'type': 'graph', 'data': graph_copy, 'updated_graph': nx.readwrite.json_graph.node_link_data(_sg)}

    # Remove a node:
    def process_graph(graph_data):
        graph_copy = copy.deepcopy(graph_data)
        graph_copy = solid_step_remove_node_from_graph(graph_copy, 'node_to_remove')
        _sg = copy.deepcopy(graph_copy)
        _sg.remove_nodes_from(list(nx.isolates(_sg)))
        return {'type': 'graph', 'data': graph_copy, 'updated_graph': nx.readwrite.json_graph.node_link_data(_sg)}

    # List children:
    def process_graph(graph_data):
        graph_copy = copy.deepcopy(graph_data)
        children = solid_step_list_child_nodes(graph_copy, {'name': 'parent_name'})
        _sg = copy.deepcopy(graph_copy)
        _sg.remove_nodes_from(list(nx.isolates(_sg)))
        return {'type': 'list', 'data': children, 'updated_graph': nx.readwrite.json_graph.node_link_data(_sg)}

    # Rank children by capacity:
    def process_graph(graph_data):
        graph_copy = copy.deepcopy(graph_data)
        ranked = solid_step_rank_child_nodes(graph_copy, 'parent_name')
        _sg = copy.deepcopy(graph_copy)
        _sg.remove_nodes_from(list(nx.isolates(_sg)))
        return {'type': 'list', 'data': ranked, 'updated_graph': nx.readwrite.json_graph.node_link_data(_sg)}

    # Remove then list (composite):
    def process_graph(graph_data):
        graph_copy = copy.deepcopy(graph_data)
        graph_copy = solid_step_remove_node_from_graph(graph_copy, 'node_to_remove')
        children = solid_step_list_child_nodes(graph_copy, {'name': 'parent_name'})
        _sg = copy.deepcopy(graph_copy)
        _sg.remove_nodes_from(list(nx.isolates(_sg)))
        return {'type': 'list', 'data': children, 'updated_graph': nx.readwrite.json_graph.node_link_data(_sg)}

    # Count nodes:
    def process_graph(graph_data):
        graph_copy = copy.deepcopy(graph_data)
        count = solid_step_counting_query(graph_copy, {'name': 'container', 'type': 'EK_AGG_BLOCK'}, {'type': 'EK_PORT'})
        _sg = copy.deepcopy(graph_copy)
        _sg.remove_nodes_from(list(nx.isolates(_sg)))
        return {'type': 'text', 'data': str(count), 'updated_graph': nx.readwrite.json_graph.node_link_data(_sg)}

    Output ONLY the function in a ```python code block. Be concise and use the helper functions.
""")


def is_netarena_malt(input_text: str) -> bool:
    """Detect if the input is a NetArena MALT query.

    The green agent sends prompts containing characteristic MALT domain terms.
    """
    markers = (
        "process_graph",
        "EK_PORT",
        "physical_capacity_bps",
        "node_link_data",
    )
    # Need at least 2 markers to avoid false positives
    count = sum(1 for m in markers if m in input_text)
    return count >= 2


async def solve_netarena_malt(
    input_text: str,
    client: AsyncOpenAI,
    model: str,
) -> str:
    """Solve a NetArena MALT query by generating process_graph code.

    The green already provides a well-formed prompt with instructions and
    examples. We add a system prompt to reinforce best practices and let
    the model generate the code.
    """
    logger.info("NetArena MALT: generating process_graph code")

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
        ],
        temperature=0.0,
        max_completion_tokens=4096,
    )

    result = response.choices[0].message.content or ""

    tracker.record(response, label="netarena-malt")

    logger.info("NetArena MALT: response generated (%d chars)", len(result))
    return result
