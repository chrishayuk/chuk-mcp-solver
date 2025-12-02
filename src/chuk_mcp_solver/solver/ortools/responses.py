"""Response builders for OR-Tools CP-SAT solver.

This module handles solution response construction and analysis.
"""

from ortools.sat.python import cp_model

from chuk_mcp_solver.models import (
    BindingConstraint,
    ConstraintKind,
    ConstraintSense,
    Explanation,
    LinearConstraintParams,
    Solution,
    SolutionVariable,
    SolveConstraintModelRequest,
    SolveConstraintModelResponse,
    SolverStatus,
)

# Tolerance for checking binding constraints
BINDING_TOLERANCE = 1e-6


def build_success_response(
    status: SolverStatus,
    solver: cp_model.CpSolver,
    var_map: dict[str, cp_model.IntVar],
    request: SolveConstraintModelRequest,
    solve_time_ms: int = 0,
) -> SolveConstraintModelResponse:
    """Build a successful solution response.

    Args:
        status: The solver status.
        solver: The CP-SAT solver with solution.
        var_map: Mapping from variable ID to CP-SAT variable.
        request: The original request.
        solve_time_ms: Actual wall-clock solve time in milliseconds.

    Returns:
        SolveConstraintModelResponse with solution data.
    """
    # Extract solution values
    solution_vars = []
    for var_def in request.variables:
        cp_var = var_map[var_def.id]
        value = solver.Value(cp_var)
        solution_vars.append(
            SolutionVariable(
                id=var_def.id,
                value=float(value),
                metadata=var_def.metadata,
            )
        )

    solution = Solution(variables=solution_vars)

    # Get objective value if applicable
    objective_value = None
    optimality_gap = None
    if request.objective:
        objective_value = solver.ObjectiveValue()

        # Calculate optimality gap for optimization problems
        # Gap = 100 * |best_bound - current_value| / |current_value|
        if status in (SolverStatus.FEASIBLE, SolverStatus.TIMEOUT_BEST):
            try:
                best_bound = solver.BestObjectiveBound()
                if objective_value != 0:
                    optimality_gap = (
                        100.0 * abs(best_bound - objective_value) / abs(objective_value)
                    )
                else:
                    # Handle zero objective value case
                    optimality_gap = abs(best_bound - objective_value)
            except Exception:
                # BestObjectiveBound() may not be available in all cases
                pass
        elif status == SolverStatus.OPTIMAL:
            # Optimal solution has zero gap
            optimality_gap = 0.0

    # Identify binding constraints
    binding_constraints = identify_binding_constraints(solver, var_map, request)

    # Build explanation
    summary = build_solution_summary(
        status, objective_value, len(binding_constraints), optimality_gap
    )
    explanation = Explanation(
        summary=summary,
        binding_constraints=binding_constraints,
    )

    return SolveConstraintModelResponse(
        status=status,
        objective_value=objective_value,
        optimality_gap=optimality_gap,
        solve_time_ms=solve_time_ms,
        solutions=[solution],
        explanation=explanation,
    )


def build_failure_response(status: SolverStatus) -> SolveConstraintModelResponse:
    """Build a failure response.

    Args:
        status: The solver status (infeasible, unbounded, timeout, error).

    Returns:
        SolveConstraintModelResponse with failure information.
    """
    summary_map = {
        SolverStatus.INFEASIBLE: "Problem is infeasible. No solution exists that satisfies all constraints.",
        SolverStatus.UNBOUNDED: "Problem is unbounded. Objective can be improved infinitely.",
        SolverStatus.TIMEOUT: "Solver timed out before finding optimal solution. Try increasing max_time_ms or simplifying the problem.",
        SolverStatus.ERROR: "Solver encountered an error. Check constraint definitions and variable domains.",
    }

    summary = summary_map.get(status, "Solver failed to find solution. Check problem formulation.")

    return SolveConstraintModelResponse(
        status=status,
        explanation=Explanation(summary=summary),
    )


def identify_binding_constraints(
    solver: cp_model.CpSolver,
    var_map: dict[str, cp_model.IntVar],
    request: SolveConstraintModelRequest,
) -> list[BindingConstraint]:
    """Identify constraints that are tight/binding in the solution.

    Args:
        solver: The CP-SAT solver with solution.
        var_map: Mapping from variable ID to CP-SAT variable.
        request: The original request.

    Returns:
        List of binding constraints.
    """
    binding = []

    for constraint in request.constraints:
        # Only analyze linear constraints for now
        if constraint.kind != ConstraintKind.LINEAR:
            continue

        params: LinearConstraintParams = constraint.params  # type: ignore[assignment]

        # Evaluate LHS with solution values
        lhs_value = sum(term.coef * solver.Value(var_map[term.var]) for term in params.terms)

        # Check if constraint is binding (tight)
        is_binding = False
        if params.sense == ConstraintSense.LESS_EQUAL:
            is_binding = abs(lhs_value - params.rhs) < BINDING_TOLERANCE
        elif params.sense == ConstraintSense.GREATER_EQUAL:
            is_binding = abs(lhs_value - params.rhs) < BINDING_TOLERANCE
        else:  # EQUAL
            is_binding = True  # Equality constraints are always binding

        if is_binding:
            binding.append(
                BindingConstraint(
                    id=constraint.id,
                    sense=params.sense,
                    lhs_value=lhs_value,
                    rhs=params.rhs,
                    metadata=constraint.metadata,
                )
            )

    return binding


def build_solution_summary(
    status: SolverStatus,
    objective_value: float | None,
    num_binding: int,
    optimality_gap: float | None = None,
) -> str:
    """Build a human-readable solution summary.

    Args:
        status: The solver status.
        objective_value: The objective value, if any.
        num_binding: Number of binding constraints.
        optimality_gap: Optimality gap percentage, if available.

    Returns:
        Summary string.
    """
    if status == SolverStatus.OPTIMAL:
        if objective_value is not None:
            summary = f"Found optimal solution with objective value {objective_value:.2f}."
        else:
            summary = "Found optimal solution."
    elif status in (SolverStatus.FEASIBLE, SolverStatus.TIMEOUT_BEST):
        if objective_value is not None:
            summary = f"Found feasible solution with objective value {objective_value:.2f}"
            if optimality_gap is not None and optimality_gap > 0:
                summary += f" (gap: {optimality_gap:.2f}% from best bound)"
            else:
                summary += " (may not be optimal)"
            summary += "."
        else:
            summary = "Found feasible solution (may not be optimal)."
    else:  # SATISFIED
        summary = "Satisfied all constraints."

    if num_binding > 0:
        summary += f" {num_binding} constraint(s) are binding (tight)."

    return summary
