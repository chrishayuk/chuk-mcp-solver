"""Tests for diagnostics and health checks."""

from chuk_mcp_solver.diagnostics import (
    HealthStatus,
    SolverHealth,
    check_solver_health,
    compute_problem_hash,
    diagnose_infeasibility,
    get_solver_diagnostics,
)
from chuk_mcp_solver.models import (
    Constraint,
    ConstraintKind,
    LinearConstraintParams,
    LinearTerm,
    SolveConstraintModelRequest,
    SolverMode,
    SolverStatus,
    Variable,
    VariableDomain,
    VariableDomainType,
)
from chuk_mcp_solver.observability import (
    SolveOutcome,
    SolverMetrics,
    get_global_statistics,
    reset_global_statistics,
)


def test_solver_health_to_dict():
    """Test SolverHealth to_dict conversion."""
    health = SolverHealth(
        status=HealthStatus.HEALTHY,
        message="All systems operational",
        details={"total_solves": 100},
    )

    result = health.to_dict()

    assert result["status"] == "healthy"
    assert result["message"] == "All systems operational"
    assert result["details"]["total_solves"] == 100


def test_check_solver_health_no_solves():
    """Test health check with no solves yet."""
    reset_global_statistics()

    health = check_solver_health()

    assert health.status == HealthStatus.HEALTHY
    assert "no solves yet" in health.message.lower()
    assert health.details["total_solves"] == 0


def test_check_solver_health_healthy():
    """Test health check with good performance."""
    reset_global_statistics()
    stats = get_global_statistics()

    # Record 10 successful solves
    for i in range(10):
        stats.record(
            SolverMetrics(
                problem_id=f"p{i}",
                timestamp="2025-01-01T00:00:00",
                num_variables=10,
                num_constraints=5,
                solve_time_ms=50.0,
                outcome=SolveOutcome.SUCCESS,
            )
        )

    health = check_solver_health()

    assert health.status == HealthStatus.HEALTHY
    assert "normally" in health.message.lower()


def test_check_solver_health_degraded_error_rate():
    """Test health check with elevated error rate."""
    reset_global_statistics()
    stats = get_global_statistics()

    # 8 successful, 2 errors = 20% error rate (degraded threshold)
    for i in range(8):
        stats.record(
            SolverMetrics(
                problem_id=f"p{i}",
                timestamp="2025-01-01T00:00:00",
                num_variables=10,
                num_constraints=5,
                solve_time_ms=50.0,
                outcome=SolveOutcome.SUCCESS,
            )
        )

    for i in range(2):
        stats.record(
            SolverMetrics(
                problem_id=f"e{i}",
                timestamp="2025-01-01T00:00:00",
                num_variables=10,
                num_constraints=5,
                solve_time_ms=50.0,
                outcome=SolveOutcome.ERROR,
            )
        )

    health = check_solver_health()

    assert health.status == HealthStatus.DEGRADED
    assert "elevated error rate" in health.message.lower()


def test_check_solver_health_unhealthy_error_rate():
    """Test health check with high error rate."""
    reset_global_statistics()
    stats = get_global_statistics()

    # 7 successful, 3 errors = 30% error rate (unhealthy threshold)
    for i in range(7):
        stats.record(
            SolverMetrics(
                problem_id=f"p{i}",
                timestamp="2025-01-01T00:00:00",
                num_variables=10,
                num_constraints=5,
                solve_time_ms=50.0,
                outcome=SolveOutcome.SUCCESS,
            )
        )

    for i in range(3):
        stats.record(
            SolverMetrics(
                problem_id=f"e{i}",
                timestamp="2025-01-01T00:00:00",
                num_variables=10,
                num_constraints=5,
                solve_time_ms=50.0,
                outcome=SolveOutcome.ERROR,
            )
        )

    health = check_solver_health()

    assert health.status == HealthStatus.UNHEALTHY
    assert "high error rate" in health.message.lower()


def test_check_solver_health_degraded_timeout_rate():
    """Test health check with high timeout rate."""
    reset_global_statistics()
    stats = get_global_statistics()

    # 5 successful, 6 timeouts = 54.5% timeout rate (> 50% degraded threshold)
    for i in range(5):
        stats.record(
            SolverMetrics(
                problem_id=f"p{i}",
                timestamp="2025-01-01T00:00:00",
                num_variables=10,
                num_constraints=5,
                solve_time_ms=50.0,
                outcome=SolveOutcome.SUCCESS,
            )
        )

    for i in range(6):
        stats.record(
            SolverMetrics(
                problem_id=f"t{i}",
                timestamp="2025-01-01T00:00:00",
                num_variables=10,
                num_constraints=5,
                solve_time_ms=5000.0,
                outcome=SolveOutcome.TIMEOUT,
            )
        )

    health = check_solver_health()

    # Timeout rate > 50% triggers degraded
    assert health.status == HealthStatus.DEGRADED
    assert "timeout rate" in health.message.lower()


def test_compute_problem_hash_deterministic():
    """Test that problem hash is deterministic."""
    request1 = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            ),
            Variable(
                id="y",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            ),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind=ConstraintKind.LINEAR,
                params=LinearConstraintParams(
                    terms=[
                        LinearTerm(var="x", coef=1),
                        LinearTerm(var="y", coef=1),
                    ],
                    sense="<=",
                    rhs=10,
                ),
            )
        ],
    )

    request2 = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            ),
            Variable(
                id="y",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            ),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind=ConstraintKind.LINEAR,
                params=LinearConstraintParams(
                    terms=[
                        LinearTerm(var="x", coef=1),
                        LinearTerm(var="y", coef=1),
                    ],
                    sense="<=",
                    rhs=10,
                ),
            )
        ],
    )

    hash1 = compute_problem_hash(request1)
    hash2 = compute_problem_hash(request2)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex digest


def test_compute_problem_hash_different_problems():
    """Test that different problems get different hashes."""
    request1 = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            )
        ],
        constraints=[],
    )

    request2 = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=20),
            )  # Different upper bound
        ],
        constraints=[],
    )

    hash1 = compute_problem_hash(request1)
    hash2 = compute_problem_hash(request2)

    assert hash1 != hash2


def test_diagnose_infeasibility_invalid_bounds():
    """Test diagnosis of variable with invalid bounds."""
    request = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(
                    type=VariableDomainType.INTEGER, lower=10, upper=5
                ),  # Invalid: lower > upper
            )
        ],
        constraints=[],
    )

    diagnosis = diagnose_infeasibility(request)

    assert diagnosis.status == SolverStatus.INFEASIBLE
    assert len(diagnosis.variable_bounds_conflicts) == 1
    assert diagnosis.variable_bounds_conflicts[0]["variable"] == "x"
    assert "invalid bounds" in diagnosis.suggestions[0].lower()


def test_diagnose_infeasibility_fixed_variable():
    """Test diagnosis of fixed variable (lower == upper)."""
    request = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=5, upper=5),  # Fixed
            )
        ],
        constraints=[],
    )

    diagnosis = diagnose_infeasibility(request)

    assert "fixed to" in diagnosis.suggestions[0].lower()


def test_diagnose_infeasibility_multiple_equalities():
    """Test diagnosis of multiple equality constraints on same variable."""
    request = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            )
        ],
        constraints=[
            Constraint(
                id="c1",
                kind=ConstraintKind.LINEAR,
                params=LinearConstraintParams(
                    terms=[LinearTerm(var="x", coef=1)],
                    sense="==",
                    rhs=5,
                ),
            ),
            Constraint(
                id="c2",
                kind=ConstraintKind.LINEAR,
                params=LinearConstraintParams(
                    terms=[LinearTerm(var="x", coef=1)],
                    sense="==",
                    rhs=7,
                ),
            ),
        ],
    )

    diagnosis = diagnose_infeasibility(request)

    assert len(diagnosis.conflicting_constraints) >= 2
    assert "c1" in diagnosis.conflicting_constraints
    assert "c2" in diagnosis.conflicting_constraints


def test_diagnose_infeasibility_optimize_without_objective():
    """Test diagnosis of optimize mode without objective."""
    request = SolveConstraintModelRequest(
        mode=SolverMode.OPTIMIZE,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            )
        ],
        constraints=[],
        objective=None,  # Missing objective
    )

    diagnosis = diagnose_infeasibility(request)

    assert any("objective" in s.lower() for s in diagnosis.suggestions)


def test_diagnose_infeasibility_generic_suggestions():
    """Test that generic suggestions are provided when no specific issues found."""
    request = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            )
        ],
        constraints=[],
    )

    diagnosis = diagnose_infeasibility(request)

    # Should have generic suggestions
    assert len(diagnosis.suggestions) > 0
    assert any("relax" in s.lower() for s in diagnosis.suggestions)


def test_get_solver_diagnostics():
    """Test comprehensive solver diagnostics."""
    reset_global_statistics()
    stats = get_global_statistics()

    # Record some solves
    for i in range(5):
        stats.record(
            SolverMetrics(
                problem_id=f"p{i}",
                timestamp="2025-01-01T00:00:00",
                num_variables=10,
                num_constraints=5,
                solve_time_ms=50.0 + i * 10,
                outcome=SolveOutcome.SUCCESS,
            )
        )

    diagnostics = get_solver_diagnostics()

    assert "health" in diagnostics
    assert "statistics" in diagnostics
    assert "recent_outcomes" in diagnostics
    assert "recent_solve_times" in diagnostics
    assert "version" in diagnostics

    assert diagnostics["health"]["status"] == "healthy"
    assert diagnostics["statistics"]["total_solves"] == 5
    assert len(diagnostics["recent_outcomes"]) == 5
    assert len(diagnostics["recent_solve_times"]) == 5
