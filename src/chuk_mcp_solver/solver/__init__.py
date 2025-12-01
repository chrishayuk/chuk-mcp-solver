"""Constraint solver implementations.

This package provides the solver interface and concrete implementations.
Currently supports Google OR-Tools CP-SAT solver.
"""

from chuk_mcp_solver.solver.ortools import ORToolsSolver
from chuk_mcp_solver.solver.provider import SolverProvider

__all__ = ["SolverProvider", "ORToolsSolver", "get_solver"]


def get_solver(solver_type: str = "ortools") -> SolverProvider:
    """Get a solver instance by type.

    Args:
        solver_type: Type of solver to create. Currently supported: "ortools"

    Returns:
        A solver instance.

    Raises:
        ValueError: If solver_type is not supported.

    Example:
        >>> solver = get_solver("ortools")
        >>> response = await solver.solve_constraint_model(request)
    """
    if solver_type == "ortools":
        return ORToolsSolver()
    else:
        raise ValueError(f"Unknown solver type: {solver_type}. Supported types: ortools")
