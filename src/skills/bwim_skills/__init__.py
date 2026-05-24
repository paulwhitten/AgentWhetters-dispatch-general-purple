"""Skills package for the purple builder agent.

Provides spatial awareness, build planning, and instruction decomposition
capabilities that enable low-cost models to accurately build block structures.
"""

from .grid import Block, Grid, GridConfig, VALID_COLORS
from .instruction_parser import ParsedInstruction, parse_green_message
from .build_planner import BuildPlanner, BuildStep
from .spatial_executor import SpatialExecutor
from .response_formatter import format_build_response, validate_build_response
from .structure_analyzer import analyze_structure, StructureAnalysis
from .plan_verifier import verify_plan, auto_fix_direction, auto_fix_each_end_caps, auto_fix_t_shape_extend
from .plan_patcher import patch_chain_references
from .underspec_detector import (
    detect_underspec_heuristic,
    detect_underspec_from_plan,
    patch_instruction_with_color,
)
from .prompt_enricher import get_enrichments, get_fired_rule_names

__all__ = [
    "Block",
    "Grid",
    "GridConfig",
    "VALID_COLORS",
    "ParsedInstruction",
    "parse_green_message",
    "BuildPlanner",
    "BuildStep",
    "SpatialExecutor",
    "format_build_response",
    "validate_build_response",
    "analyze_structure",
    "StructureAnalysis",
    "verify_plan",
    "auto_fix_direction",
    "auto_fix_each_end_caps",
    "auto_fix_t_shape_extend",
    "patch_chain_references",
    "detect_underspec_heuristic",
    "detect_underspec_from_plan",
    "patch_instruction_with_color",
    "get_enrichments",
    "get_fired_rule_names",
]
