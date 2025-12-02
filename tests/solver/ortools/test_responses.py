"""Tests for solver status conversion and edge cases."""

from ortools.sat.python import cp_model

from chuk_mcp_solver.models import SolverMode, SolverStatus
from chuk_mcp_solver.solver.ortools import ORToolsSolver as ORToolsProvider
from chuk_mcp_solver.solver.ortools.responses import build_failure_response


def test_status_conversion_model_invalid():
    """Test conversion of MODEL_INVALID status."""
    provider = ORToolsProvider()
    status = provider._convert_status(cp_model.MODEL_INVALID, SolverMode.SATISFY)
    assert status == SolverStatus.ERROR


def test_status_conversion_unknown():
    """Test conversion of UNKNOWN status (timeout)."""
    provider = ORToolsProvider()
    status = provider._convert_status(cp_model.UNKNOWN, SolverMode.SATISFY)
    assert status == SolverStatus.TIMEOUT


def test_status_conversion_infeasible():
    """Test conversion of INFEASIBLE status."""
    provider = ORToolsProvider()
    status = provider._convert_status(cp_model.INFEASIBLE, SolverMode.SATISFY)
    assert status == SolverStatus.INFEASIBLE


def test_status_conversion_optimal_optimize_mode():
    """Test OPTIMAL in optimize mode."""
    provider = ORToolsProvider()
    status = provider._convert_status(cp_model.OPTIMAL, SolverMode.OPTIMIZE)
    assert status == SolverStatus.OPTIMAL


def test_status_conversion_optimal_satisfy_mode():
    """Test OPTIMAL in satisfy mode becomes SATISFIED."""
    provider = ORToolsProvider()
    status = provider._convert_status(cp_model.OPTIMAL, SolverMode.SATISFY)
    assert status == SolverStatus.SATISFIED


def test_status_conversion_feasible_optimize_mode():
    """Test FEASIBLE in optimize mode."""
    provider = ORToolsProvider()
    status = provider._convert_status(cp_model.FEASIBLE, SolverMode.OPTIMIZE)
    assert status == SolverStatus.FEASIBLE


def test_status_conversion_feasible_satisfy_mode():
    """Test FEASIBLE in satisfy mode becomes SATISFIED."""
    provider = ORToolsProvider()
    status = provider._convert_status(cp_model.FEASIBLE, SolverMode.SATISFY)
    assert status == SolverStatus.SATISFIED


def test_build_failure_response_infeasible():
    """Test building failure response for infeasible."""
    response = build_failure_response(SolverStatus.INFEASIBLE)
    assert response.status == SolverStatus.INFEASIBLE
    assert "infeasible" in response.explanation.summary.lower()


def test_build_failure_response_unbounded():
    """Test building failure response for unbounded."""
    response = build_failure_response(SolverStatus.UNBOUNDED)
    assert response.status == SolverStatus.UNBOUNDED
    assert "unbounded" in response.explanation.summary.lower()


def test_build_failure_response_timeout():
    """Test building failure response for timeout."""
    response = build_failure_response(SolverStatus.TIMEOUT)
    assert response.status == SolverStatus.TIMEOUT
    assert "timed out" in response.explanation.summary.lower()


def test_build_failure_response_error():
    """Test building failure response for error."""
    response = build_failure_response(SolverStatus.ERROR)
    assert response.status == SolverStatus.ERROR
    assert "error" in response.explanation.summary.lower()


# Tests for optimality gap calculation and summary generation
async def test_build_success_response_with_feasible_and_gap():
    """Test response building for FEASIBLE status with optimality gap."""
    from chuk_mcp_solver.models import (
        Objective,
        SearchConfig,
        SolveConstraintModelRequest,
        Variable,
        VariableDomain,
        VariableDomainType,
    )
    from chuk_mcp_solver.solver import get_solver

    solver = get_solver("ortools")

    # Create a problem that will likely get FEASIBLE (not OPTIMAL) with short timeout
    request = SolveConstraintModelRequest(
        mode=SolverMode.OPTIMIZE,
        variables=[
            Variable(
                id=f"x{i}",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=100),
            )
            for i in range(15)
        ],
        constraints=[],
        objective=Objective(
            sense="max",
            terms=[{"var": f"x{i}", "coef": i + 1} for i in range(15)],
        ),
        search=SearchConfig(max_time_ms=5),  # Very short timeout
    )

    response = await solver.solve_constraint_model(request)

    # Should be OPTIMAL or FEASIBLE
    assert response.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)

    # Check metrics exist
    assert hasattr(response, "optimality_gap")
    assert hasattr(response, "solve_time_ms")

    if response.status == SolverStatus.FEASIBLE:
        # Gap should be present for FEASIBLE
        assert response.optimality_gap is not None
        # Explanation should mention gap if > 0
        assert response.explanation is not None


async def test_build_success_response_timeout_best_with_gap():
    """Test response building for TIMEOUT_BEST status."""
    from chuk_mcp_solver.models import (
        Objective,
        SearchConfig,
        SolveConstraintModelRequest,
        Variable,
        VariableDomain,
        VariableDomainType,
    )
    from chuk_mcp_solver.solver import get_solver

    solver = get_solver("ortools")

    # Problem with very short timeout and return_partial_solution enabled
    request = SolveConstraintModelRequest(
        mode=SolverMode.OPTIMIZE,
        variables=[
            Variable(
                id=f"x{i}",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            )
            for i in range(10)
        ],
        constraints=[],
        objective=Objective(
            sense="max",
            terms=[{"var": f"x{i}", "coef": i + 1} for i in range(10)],
        ),
        search=SearchConfig(
            max_time_ms=1,  # Very short
            return_partial_solution=True,
        ),
    )

    response = await solver.solve_constraint_model(request)

    # Could be OPTIMAL, TIMEOUT_BEST, or TIMEOUT_NO_SOLUTION
    assert response.status in (
        SolverStatus.OPTIMAL,
        SolverStatus.TIMEOUT_BEST,
        SolverStatus.TIMEOUT_NO_SOLUTION,
    )

    if response.status == SolverStatus.TIMEOUT_BEST:
        # Should have solution
        assert len(response.solutions) > 0
        # Should have optimality gap
        assert response.optimality_gap is not None
        # Explanation should mention timeout
        assert "timed out" in response.explanation.summary.lower()


async def test_build_success_response_optimal_no_objective_value():
    """Test response with OPTIMAL but no objective (satisfy mode)."""
    from chuk_mcp_solver.models import (
        SolveConstraintModelRequest,
        Variable,
        VariableDomain,
        VariableDomainType,
    )
    from chuk_mcp_solver.solver import get_solver

    solver = get_solver("ortools")

    request = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x", domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10)
            )
        ],
        constraints=[],
    )

    response = await solver.solve_constraint_model(request)

    assert response.status == SolverStatus.SATISFIED
    assert response.objective_value is None
    assert response.optimality_gap is None
    assert response.explanation is not None
    # Summary should not mention objective
    assert "optimal solution" not in response.explanation.summary.lower()


async def test_build_success_response_zero_objective_value():
    """Test optimality gap calculation when objective value is zero."""
    from chuk_mcp_solver.models import (
        Objective,
        SolveConstraintModelRequest,
        Variable,
        VariableDomain,
        VariableDomainType,
    )
    from chuk_mcp_solver.solver import get_solver

    solver = get_solver("ortools")

    # Create problem where optimal value is 0
    request = SolveConstraintModelRequest(
        mode=SolverMode.OPTIMIZE,
        variables=[
            Variable(
                id="x", domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=0)
            )
        ],
        constraints=[],
        objective=Objective(sense="max", terms=[{"var": "x", "coef": 1}]),
    )

    response = await solver.solve_constraint_model(request)

    assert response.status == SolverStatus.OPTIMAL
    assert response.objective_value == 0
    assert response.optimality_gap == 0.0  # Optimal has 0 gap even with 0 value
