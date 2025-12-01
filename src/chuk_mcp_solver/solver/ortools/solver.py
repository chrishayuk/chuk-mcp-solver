"""OR-Tools CP-SAT solver implementation.

Main solver class that orchestrates constraint solving using Google OR-Tools.
"""

import logging

from ortools.sat.python import cp_model

from chuk_mcp_solver.models import (
    Explanation,
    SolveConstraintModelRequest,
    SolveConstraintModelResponse,
    SolverMode,
    SolverStatus,
    VariableDomainType,
)
from chuk_mcp_solver.solver.ortools.constraints import build_constraint
from chuk_mcp_solver.solver.ortools.objectives import build_objective, configure_solver
from chuk_mcp_solver.solver.ortools.responses import build_failure_response, build_success_response
from chuk_mcp_solver.solver.provider import SolverProvider

logger = logging.getLogger(__name__)


class ORToolsSolver(SolverProvider):
    """Google OR-Tools CP-SAT constraint solver.

    High-performance constraint programming solver supporting:
    - Integer and boolean variables
    - Linear constraints
    - Global constraints (all_different, element, table)
    - Scheduling constraints (cumulative, no_overlap)
    - Routing constraints (circuit)
    - Inventory constraints (reservoir)
    - Multi-objective optimization
    """

    async def solve_constraint_model(
        self, request: SolveConstraintModelRequest
    ) -> SolveConstraintModelResponse:
        """Solve a constraint/optimization model using OR-Tools CP-SAT.

        Args:
            request: The constraint model to solve.

        Returns:
            SolveConstraintModelResponse with solution status and results.

        Raises:
            ValueError: If the request is invalid or contains unsupported features.
        """
        # Validate request
        self._validate_request(request)

        try:
            # Build CP-SAT model
            model = cp_model.CpModel()
            var_map = self._build_variables(model, request)
            self._build_constraints(model, request, var_map)

            # Set objective if optimizing
            if request.mode == SolverMode.OPTIMIZE and request.objective:
                build_objective(model, request.objective, var_map)

            # Solve
            solver = cp_model.CpSolver()
            configure_solver(solver, request)

            # Add warm-start solution hints if provided
            if request.search and request.search.warm_start_solution:
                for var_id, value in request.search.warm_start_solution.items():
                    if var_id in var_map:
                        model.AddHint(var_map[var_id], value)

            status = solver.Solve(model)

            # Convert status
            solver_status = self._convert_status(status, request.mode)

            # Build response
            if solver_status in (
                SolverStatus.OPTIMAL,
                SolverStatus.FEASIBLE,
                SolverStatus.SATISFIED,
            ):
                return build_success_response(solver_status, solver, var_map, request)
            else:
                return build_failure_response(solver_status)

        except Exception as e:
            logger.error(f"Error solving model: {e}")
            return SolveConstraintModelResponse(
                status=SolverStatus.ERROR,
                explanation=Explanation(summary=f"Solver error: {str(e)}"),
            )

    def _validate_request(self, request: SolveConstraintModelRequest) -> None:
        """Validate the solve request.

        Args:
            request: The request to validate.

        Raises:
            ValueError: If the request is invalid.
        """
        if request.mode == SolverMode.OPTIMIZE and not request.objective:
            raise ValueError("Objective required when mode is 'optimize'")

        if request.mode == SolverMode.SATISFY and request.objective:
            logger.warning(
                "Objective provided in 'satisfy' mode will be ignored. "
                "Use mode='optimize' to optimize the objective."
            )

    def _build_variables(
        self, model: cp_model.CpModel, request: SolveConstraintModelRequest
    ) -> dict[str, cp_model.IntVar]:
        """Build CP-SAT variables from request.

        Args:
            model: The CP-SAT model.
            request: The request containing variable definitions.

        Returns:
            Mapping from variable ID to CP-SAT variable.
        """
        var_map = {}

        for var_def in request.variables:
            if var_def.domain.type == VariableDomainType.BOOL:
                # Boolean variables are [0, 1] integer variables
                cp_var = model.NewIntVar(0, 1, var_def.id)
            else:  # INTEGER
                cp_var = model.NewIntVar(var_def.domain.lower, var_def.domain.upper, var_def.id)

            var_map[var_def.id] = cp_var

        return var_map

    def _build_constraints(
        self,
        model: cp_model.CpModel,
        request: SolveConstraintModelRequest,
        var_map: dict[str, cp_model.IntVar],
    ) -> None:
        """Build all constraints.

        Args:
            model: The CP-SAT model.
            request: The request containing constraint definitions.
            var_map: Mapping from variable ID to CP-SAT variable.
        """
        for constraint in request.constraints:
            build_constraint(model, constraint, var_map)

    def _convert_status(self, cp_status: int, mode: SolverMode) -> SolverStatus:
        """Convert CP-SAT status to SolverStatus.

        Args:
            cp_status: The CP-SAT status code.
            mode: The solver mode (satisfy or optimize).

        Returns:
            The corresponding SolverStatus.
        """
        if cp_status == cp_model.OPTIMAL:
            # In satisfy mode, treat OPTIMAL as SATISFIED
            if mode == SolverMode.SATISFY:
                return SolverStatus.SATISFIED
            else:
                return SolverStatus.OPTIMAL
        elif cp_status == cp_model.FEASIBLE:
            if mode == SolverMode.SATISFY:
                return SolverStatus.SATISFIED
            else:
                return SolverStatus.FEASIBLE
        elif cp_status == cp_model.INFEASIBLE:
            return SolverStatus.INFEASIBLE
        elif cp_status == cp_model.UNKNOWN:
            return SolverStatus.TIMEOUT
        elif cp_status == cp_model.MODEL_INVALID:
            return SolverStatus.ERROR
        else:
            return SolverStatus.ERROR
