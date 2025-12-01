"""Providers module - DEPRECATED, use chuk_mcp_solver.solver instead.

This module is maintained for backward compatibility.
Please use chuk_mcp_solver.solver in new code.
"""

import warnings

# Import from new location for backward compatibility
from chuk_mcp_solver.solver import ORToolsSolver as ORToolsProvider
from chuk_mcp_solver.solver import SolverProvider
from chuk_mcp_solver.solver import get_solver as get_provider

warnings.warn(
    "chuk_mcp_solver.providers is deprecated, use chuk_mcp_solver.solver instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SolverProvider", "ORToolsProvider", "get_provider", "get_provider_for_tool"]


def get_provider_for_tool(tool_name: str) -> SolverProvider:
    """Get provider for a specific tool - DEPRECATED.

    Use get_solver() instead.
    """
    return get_provider("ortools")
