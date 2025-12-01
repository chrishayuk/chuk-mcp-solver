"""Pydantic models for constraint solver.

This module defines all data models, enums, and types used throughout the solver.
No magic strings - all constants are defined as enums or module-level constants.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ============================================================================
# Constants
# ============================================================================

# API versioning
API_VERSION = "1.0.0"

# Constraint kinds
CONSTRAINT_KIND_LINEAR = "linear"
CONSTRAINT_KIND_ALL_DIFFERENT = "all_different"
CONSTRAINT_KIND_ELEMENT = "element"
CONSTRAINT_KIND_TABLE = "table"
CONSTRAINT_KIND_IMPLICATION = "implication"
CONSTRAINT_KIND_CUMULATIVE = "cumulative"
CONSTRAINT_KIND_CIRCUIT = "circuit"
CONSTRAINT_KIND_RESERVOIR = "reservoir"
CONSTRAINT_KIND_NO_OVERLAP = "no_overlap"


# ============================================================================
# Enums - No Magic Strings
# ============================================================================


class SolverMode(str, Enum):
    """Solver execution mode."""

    SATISFY = "satisfy"
    OPTIMIZE = "optimize"


class VariableDomainType(str, Enum):
    """Variable domain types supported by the solver."""

    BOOL = "bool"
    INTEGER = "integer"


class ConstraintSense(str, Enum):
    """Comparison operators for linear constraints."""

    LESS_EQUAL = "<="
    GREATER_EQUAL = ">="
    EQUAL = "=="


class ObjectiveSense(str, Enum):
    """Optimization direction."""

    MINIMIZE = "min"
    MAXIMIZE = "max"


class ConstraintKind(str, Enum):
    """Types of constraints supported by the solver."""

    LINEAR = CONSTRAINT_KIND_LINEAR
    ALL_DIFFERENT = CONSTRAINT_KIND_ALL_DIFFERENT
    ELEMENT = CONSTRAINT_KIND_ELEMENT
    TABLE = CONSTRAINT_KIND_TABLE
    IMPLICATION = CONSTRAINT_KIND_IMPLICATION
    CUMULATIVE = CONSTRAINT_KIND_CUMULATIVE
    CIRCUIT = CONSTRAINT_KIND_CIRCUIT
    RESERVOIR = CONSTRAINT_KIND_RESERVOIR
    NO_OVERLAP = CONSTRAINT_KIND_NO_OVERLAP


class SolverStatus(str, Enum):
    """Solution status returned by the solver."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    SATISFIED = "satisfied"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIMEOUT = "timeout"
    ERROR = "error"


# ============================================================================
# Variable Domain Models
# ============================================================================


class VariableDomain(BaseModel):
    """Domain specification for a decision variable.

    Defines the type and bounds for a variable in the constraint model.
    """

    type: VariableDomainType = Field(
        ...,
        description="Variable domain type: 'bool' for binary variables, 'integer' for integer variables",
    )
    lower: int = Field(
        default=0,
        description="Lower bound (inclusive) for integer variables; ignored for bool variables",
    )
    upper: int = Field(
        default=1,
        description="Upper bound (inclusive) for integer variables; ignored for bool variables",
    )


# ============================================================================
# Variable Models
# ============================================================================


class Variable(BaseModel):
    """Decision variable in the constraint model.

    Each variable represents a choice to be made by the solver.
    """

    id: str = Field(
        ...,
        description="Unique identifier for this variable",
        min_length=1,
    )
    domain: VariableDomain = Field(
        ...,
        description="Domain specification (type and bounds)",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata for explanations and context; echoed in solutions",
    )


# ============================================================================
# Constraint Parameter Models
# ============================================================================


class LinearTerm(BaseModel):
    """A term in a linear expression: coefficient * variable."""

    var: str = Field(
        ...,
        description="Variable identifier",
        min_length=1,
    )
    coef: float = Field(
        ...,
        description="Coefficient multiplying the variable",
    )


class LinearConstraintParams(BaseModel):
    """Parameters for a linear constraint: sum(terms) sense rhs."""

    terms: list[LinearTerm] = Field(
        ...,
        description="List of linear terms forming the left-hand side",
        min_length=1,
    )
    sense: ConstraintSense = Field(
        ...,
        description="Comparison operator: '<=', '>=', or '=='",
    )
    rhs: float = Field(
        ...,
        description="Right-hand side constant value",
    )


class AllDifferentParams(BaseModel):
    """Parameters for an all-different constraint."""

    vars: list[str] = Field(
        ...,
        description="List of variable identifiers that must all take different values",
        min_length=2,
    )


class ElementParams(BaseModel):
    """Parameters for an element constraint: target = array[index]."""

    index_var: str = Field(
        ...,
        description="Integer variable used as index into the array",
        min_length=1,
    )
    array: list[int] = Field(
        ...,
        description="Constant integer array to index into",
        min_length=1,
    )
    target_var: str = Field(
        ...,
        description="Variable that equals array[index_var]",
        min_length=1,
    )


class TableParams(BaseModel):
    """Parameters for a table constraint: allowed tuples."""

    vars: list[str] = Field(
        ...,
        description="List of variable identifiers forming the tuple",
        min_length=1,
    )
    allowed_tuples: list[list[int]] = Field(
        ...,
        description="List of allowed integer tuples for the variables",
        min_length=1,
    )


class CumulativeParams(BaseModel):
    """Parameters for a cumulative constraint: resource capacity over time."""

    start_vars: list[str] = Field(
        ...,
        description="List of start time variable identifiers",
        min_length=1,
    )
    duration_vars: list[str] | list[int] = Field(
        ...,
        description="List of duration variable identifiers or constant durations",
        min_length=1,
    )
    demand_vars: list[str] | list[int] = Field(
        ...,
        description="List of demand variable identifiers or constant demands",
        min_length=1,
    )
    capacity: int = Field(
        ...,
        description="Maximum cumulative resource capacity",
        ge=0,
    )


class CircuitParams(BaseModel):
    """Parameters for a circuit constraint: routing/tour problem."""

    arcs: list[tuple[int, int, str]] = Field(
        ...,
        description="List of (from_node, to_node, arc_var) tuples forming possible connections",
        min_length=1,
    )


class ReservoirParams(BaseModel):
    """Parameters for a reservoir constraint: inventory/stock management."""

    time_vars: list[str] = Field(
        ...,
        description="List of time variable identifiers when events occur",
        min_length=1,
    )
    level_changes: list[int] = Field(
        ...,
        description="Change in level at each time point (positive=production, negative=consumption)",
        min_length=1,
    )
    min_level: int = Field(
        default=0,
        description="Minimum reservoir level (default 0)",
    )
    max_level: int = Field(
        ...,
        description="Maximum reservoir level (capacity)",
        ge=0,
    )


class NoOverlapParams(BaseModel):
    """Parameters for a no-overlap constraint: disjunctive scheduling."""

    start_vars: list[str] = Field(
        ...,
        description="List of start time variable identifiers",
        min_length=1,
    )
    duration_vars: list[str] | list[int] = Field(
        ...,
        description="List of duration variable identifiers or constant durations",
        min_length=1,
    )


class ImplicationParams(BaseModel):
    """Parameters for an implication constraint: if bool_var then nested_constraint."""

    if_var: str = Field(
        ...,
        description="Boolean variable: when true, the nested constraint must hold",
        min_length=1,
    )
    then: Constraint = Field(
        ...,
        description="Nested constraint that becomes active when if_var is true",
    )


# ============================================================================
# Constraint Models
# ============================================================================


class Constraint(BaseModel):
    """Constraint in the constraint model.

    Each constraint restricts the values that variables can take.
    """

    id: str = Field(
        ...,
        description="Unique identifier for this constraint",
        min_length=1,
    )
    kind: ConstraintKind = Field(
        ...,
        description="Type of constraint: linear, all_different, element, table, implication, cumulative, circuit, reservoir, or no_overlap",
    )
    params: (
        LinearConstraintParams
        | AllDifferentParams
        | ElementParams
        | TableParams
        | ImplicationParams
        | CumulativeParams
        | CircuitParams
        | ReservoirParams
        | NoOverlapParams
    ) = Field(
        ...,
        description="Parameters specific to the constraint kind",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata for explanations (e.g., human description)",
    )


# ============================================================================
# Objective Models
# ============================================================================


class Objective(BaseModel):
    """Objective function for optimization.

    Defines a linear objective to minimize or maximize.
    Supports multi-objective via lexicographic ordering.
    """

    sense: ObjectiveSense = Field(
        ...,
        description="Optimization direction: 'min' to minimize, 'max' to maximize",
    )
    terms: list[LinearTerm] = Field(
        ...,
        description="Linear terms forming the objective function",
        min_length=1,
    )
    priority: int = Field(
        default=1,
        description="Priority level for lexicographic multi-objective (higher = more important)",
        ge=1,
    )
    weight: float = Field(
        default=1.0,
        description="Weight for weighted-sum multi-objective optimization",
        gt=0.0,
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional explanation/label for the objective",
    )


# ============================================================================
# Search Configuration
# ============================================================================


class SearchConfig(BaseModel):
    """Solver search configuration and limits."""

    max_time_ms: int | None = Field(
        default=None,
        description="Maximum solver time in milliseconds",
        ge=1,
    )
    max_solutions: int = Field(
        default=1,
        description="Maximum number of solutions to return",
        ge=1,
    )
    num_search_workers: int | None = Field(
        default=None,
        description="Number of parallel search workers (default: auto)",
        ge=1,
    )
    log_search_progress: bool = Field(
        default=False,
        description="Enable search progress logging",
    )
    warm_start_solution: dict[str, int] | None = Field(
        default=None,
        description="Optional warm-start solution hint (variable_id -> value mapping)",
    )


# ============================================================================
# Request Model
# ============================================================================


class SolveConstraintModelRequest(BaseModel):
    """Request to solve a constraint/optimization model.

    This is the main input to the solve_constraint_model tool.
    """

    mode: SolverMode = Field(
        ...,
        description="Solver mode: 'satisfy' to find any feasible solution, 'optimize' to find optimal solution",
    )
    variables: list[Variable] = Field(
        ...,
        description="List of decision variables",
        min_length=1,
    )
    constraints: list[Constraint] = Field(
        ...,
        description="List of constraints",
    )
    objective: Objective | list[Objective] | None = Field(
        default=None,
        description="Objective function(s); required when mode is 'optimize'. Can be single objective or list for multi-objective optimization",
    )
    search: SearchConfig | None = Field(
        default=None,
        description="Optional solver search/limits configuration",
    )


# ============================================================================
# Solution Models
# ============================================================================


class SolutionVariable(BaseModel):
    """Variable value in a solution."""

    id: str = Field(
        ...,
        description="Variable identifier",
    )
    value: float = Field(
        ...,
        description="Assigned value in this solution",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Original metadata from the variable definition",
    )


class Solution(BaseModel):
    """A single solution to the constraint model."""

    variables: list[SolutionVariable] = Field(
        ...,
        description="Variable assignments in this solution",
    )
    derived: dict[str, Any] | None = Field(
        default=None,
        description="Optional derived metrics computed from the solution (e.g., makespan, counts)",
    )


class BindingConstraint(BaseModel):
    """Information about a constraint that is tight/critical in the solution."""

    id: str = Field(
        ...,
        description="Constraint identifier",
    )
    sense: ConstraintSense | None = Field(
        default=None,
        description="Constraint sense (for linear constraints)",
    )
    lhs_value: float = Field(
        ...,
        description="Evaluated left-hand side under the solution",
    )
    rhs: float = Field(
        ...,
        description="Right-hand side value",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional constraint metadata",
    )


class Explanation(BaseModel):
    """Human-readable explanation of the solution."""

    summary: str = Field(
        ...,
        description="High-level textual summary of the result",
    )
    binding_constraints: list[BindingConstraint] = Field(
        default_factory=list,
        description="Constraints that are tight/critical in the solution",
    )


# ============================================================================
# Response Model
# ============================================================================


class SolveConstraintModelResponse(BaseModel):
    """Response from solving a constraint/optimization model."""

    apiversion: str = Field(
        default=API_VERSION,
        description="API version",
    )
    status: SolverStatus = Field(
        ...,
        description="Solution status: optimal, feasible, satisfied, infeasible, unbounded, timeout, or error",
    )
    objective_value: float | None = Field(
        default=None,
        description="Objective value for the best solution, if applicable",
    )
    solutions: list[Solution] = Field(
        default_factory=list,
        description="List of solutions; usually length 1 unless max_solutions > 1",
    )
    explanation: Explanation | None = Field(
        default=None,
        description="Optional human-readable explanation of the result",
    )
