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
