"""Abstract solver provider interface.

Defines the contract that all solver implementations must follow.
"""

from abc import ABC, abstractmethod

from chuk_mcp_solver.models import SolveConstraintModelRequest, SolveConstraintModelResponse


class SolverProvider(ABC):
    """Abstract base class for constraint solver providers.

    All solver implementations (OR-Tools, Gurobi, etc.) must inherit from this
    and implement the solve_constraint_model method.
    """

    @abstractmethod
    async def solve_constraint_model(
        self, request: SolveConstraintModelRequest
    ) -> SolveConstraintModelResponse:
        """Solve a constraint/optimization model.

        Args:
            request: The constraint model to solve.

        Returns:
            SolveConstraintModelResponse with solution status and results.

        Raises:
            ValueError: If the request is invalid or contains unsupported features.
        """
        pass
