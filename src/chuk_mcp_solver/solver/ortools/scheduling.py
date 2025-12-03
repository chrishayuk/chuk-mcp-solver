"""Scheduling problem converters.

This module converts high-level scheduling problems to/from CP-SAT models.
"""

from chuk_mcp_solver.models import (
    Constraint,
    ConstraintKind,
    ConstraintSense,
    CumulativeParams,
    LinearConstraintParams,
    LinearTerm,
    NoOverlapParams,
    Objective,
    ObjectiveSense,
    SchedulingExplanation,
    SchedulingObjective,
    SearchConfig,
    SolveConstraintModelRequest,
    SolveConstraintModelResponse,
    SolverMode,
    SolverStatus,
    SolveSchedulingProblemRequest,
    SolveSchedulingProblemResponse,
    TaskAssignment,
    Variable,
    VariableDomain,
    VariableDomainType,
)


def convert_scheduling_to_cpsat(
    request: SolveSchedulingProblemRequest,
) -> SolveConstraintModelRequest:
    """Convert high-level scheduling problem to CP-SAT model.

    Args:
        request: High-level scheduling request

    Returns:
        CP-SAT constraint model request
    """
    variables = []
    constraints = []

    # Calculate horizon (upper bound on makespan)
    if request.max_makespan:
        horizon = request.max_makespan
    else:
        # Sum of all task durations as base upper bound
        horizon = sum(task.duration for task in request.tasks)

        # Account for earliest_start times (a task starting late could push makespan beyond sum of durations)
        for task in request.tasks:
            if task.earliest_start is not None:
                min_completion = task.earliest_start + task.duration
                horizon = max(horizon, min_completion)

    # Create start and end time variables for each task
    for task in request.tasks:
        # Start time variable
        start_lower = task.earliest_start if task.earliest_start is not None else 0
        start_upper = task.deadline if task.deadline is not None else horizon

        variables.append(
            Variable(
                id=f"start_{task.id}",
                domain=VariableDomain(
                    type=VariableDomainType.INTEGER, lower=start_lower, upper=start_upper
                ),
                metadata={"task": task.id, "type": "start_time"},
            )
        )

        # End time variable (lower bound = start_lower + duration)
        end_lower = start_lower + task.duration
        end_upper = horizon

        variables.append(
            Variable(
                id=f"end_{task.id}",
                domain=VariableDomain(
                    type=VariableDomainType.INTEGER, lower=end_lower, upper=end_upper
                ),
                metadata={"task": task.id, "type": "end_time"},
            )
        )

    # Add duration constraints: end = start + duration
    for task in request.tasks:
        constraints.append(
            Constraint(
                id=f"duration_{task.id}",
                kind=ConstraintKind.LINEAR,
                params=LinearConstraintParams(
                    terms=[
                        LinearTerm(var=f"end_{task.id}", coef=1),
                        LinearTerm(var=f"start_{task.id}", coef=-1),
                    ],
                    sense=ConstraintSense.EQUAL,
                    rhs=task.duration,
                ),
                metadata={"description": f"Task {task.id} duration = {task.duration}"},
            )
        )

    # Add precedence constraints for dependencies
    for task in request.tasks:
        for dep_id in task.dependencies:
            constraints.append(
                Constraint(
                    id=f"precedence_{dep_id}_to_{task.id}",
                    kind=ConstraintKind.LINEAR,
                    params=LinearConstraintParams(
                        terms=[
                            LinearTerm(var=f"start_{task.id}", coef=1),
                            LinearTerm(var=f"end_{dep_id}", coef=-1),
                        ],
                        sense=ConstraintSense.GREATER_EQUAL,
                        rhs=0,
                    ),
                    metadata={"description": f"Task {task.id} must start after {dep_id} completes"},
                )
            )

    # Add deadline constraints
    for task in request.tasks:
        if task.deadline is not None:
            constraints.append(
                Constraint(
                    id=f"deadline_{task.id}",
                    kind=ConstraintKind.LINEAR,
                    params=LinearConstraintParams(
                        terms=[LinearTerm(var=f"end_{task.id}", coef=1)],
                        sense=ConstraintSense.LESS_EQUAL,
                        rhs=task.deadline,
                    ),
                    metadata={"description": f"Task {task.id} must complete by {task.deadline}"},
                )
            )

    # Add resource capacity constraints (cumulative)
    for resource in request.resources:
        # Collect tasks using this resource
        start_vars = []
        duration_vals = []
        demand_vals = []

        for task in request.tasks:
            if resource.id in task.resources_required:
                start_vars.append(f"start_{task.id}")
                duration_vals.append(task.duration)
                demand_vals.append(task.resources_required[resource.id])

        if start_vars:
            constraints.append(
                Constraint(
                    id=f"capacity_{resource.id}",
                    kind=ConstraintKind.CUMULATIVE,
                    params=CumulativeParams(
                        start_vars=start_vars,
                        duration_vars=duration_vals,
                        demand_vars=demand_vals,
                        capacity=resource.capacity,
                    ),
                    metadata={
                        "description": f"Resource {resource.id} capacity limit ({resource.capacity})"
                    },
                )
            )

    # Add no-overlap constraints for task groups
    for group in request.no_overlap_tasks:
        if len(group) >= 2:
            constraints.append(
                Constraint(
                    id=f"no_overlap_{'_'.join(group)}",
                    kind=ConstraintKind.NO_OVERLAP,
                    params=NoOverlapParams(
                        start_vars=[f"start_{tid}" for tid in group],
                        duration_vars=[
                            next(t.duration for t in request.tasks if t.id == tid) for tid in group
                        ],
                    ),
                    metadata={"description": f"Tasks {', '.join(group)} cannot overlap"},
                )
            )

    # Create objective based on scheduling objective
    objective = None
    if request.objective == SchedulingObjective.MINIMIZE_MAKESPAN:
        # Create makespan variable (max of all end times)
        variables.append(
            Variable(
                id="makespan",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=horizon),
                metadata={"type": "makespan"},
            )
        )

        # makespan >= end_i for all tasks
        for task in request.tasks:
            constraints.append(
                Constraint(
                    id=f"makespan_bounds_{task.id}",
                    kind=ConstraintKind.LINEAR,
                    params=LinearConstraintParams(
                        terms=[
                            LinearTerm(var="makespan", coef=1),
                            LinearTerm(var=f"end_{task.id}", coef=-1),
                        ],
                        sense=ConstraintSense.GREATER_EQUAL,
                        rhs=0,
                    ),
                    metadata={"description": "Makespan lower bound"},
                )
            )

        objective = Objective(
            sense=ObjectiveSense.MINIMIZE, terms=[LinearTerm(var="makespan", coef=1)]
        )

    # TODO: Implement other objectives (MINIMIZE_COST, MINIMIZE_LATENESS)

    return SolveConstraintModelRequest(
        mode=SolverMode.OPTIMIZE if objective else SolverMode.SATISFY,
        variables=variables,
        constraints=constraints,
        objective=objective,
        search=SearchConfig(
            max_time_ms=request.max_time_ms,
            return_partial_solution=request.return_partial_solution,
        ),
    )


def convert_cpsat_to_scheduling_response(
    cpsat_response: SolveConstraintModelResponse,
    original_request: SolveSchedulingProblemRequest,
) -> SolveSchedulingProblemResponse:
    """Convert CP-SAT solution back to scheduling domain.

    Args:
        cpsat_response: CP-SAT solver response
        original_request: Original scheduling request

    Returns:
        High-level scheduling response
    """
    if cpsat_response.status in (
        SolverStatus.INFEASIBLE,
        SolverStatus.UNBOUNDED,
        SolverStatus.ERROR,
    ):
        # No solution
        return SolveSchedulingProblemResponse(
            status=cpsat_response.status,
            solve_time_ms=cpsat_response.solve_time_ms,
            explanation=SchedulingExplanation(
                summary=cpsat_response.explanation.summary
                if cpsat_response.explanation
                else f"Problem is {cpsat_response.status.value}"
            ),
        )

    if cpsat_response.status == SolverStatus.TIMEOUT_NO_SOLUTION:
        return SolveSchedulingProblemResponse(
            status=cpsat_response.status,
            solve_time_ms=cpsat_response.solve_time_ms,
            explanation=SchedulingExplanation(
                summary=cpsat_response.explanation.summary
                if cpsat_response.explanation
                else "Timeout with no solution found",
                recommendations=[
                    "Increase max_time_ms",
                    "Simplify problem (fewer tasks or looser constraints)",
                    "Check for infeasibility (conflicting deadlines/dependencies)",
                ],
            ),
        )

    if not cpsat_response.solutions:
        return SolveSchedulingProblemResponse(
            status=cpsat_response.status,
            solve_time_ms=cpsat_response.solve_time_ms,
            explanation=SchedulingExplanation(summary="No solution available"),
        )

    # Extract variable values
    solution = cpsat_response.solutions[0]
    var_values = {v.id: int(v.value) for v in solution.variables}

    # Build task assignments
    schedule = []
    for task in original_request.tasks:
        start_time = var_values.get(f"start_{task.id}", 0)
        end_time = var_values.get(f"end_{task.id}", start_time + task.duration)

        schedule.append(
            TaskAssignment(
                task_id=task.id,
                start_time=start_time,
                end_time=end_time,
                resources_used=task.resources_required,
                metadata=task.metadata,
            )
        )

    # Calculate makespan
    makespan = var_values.get("makespan")
    if makespan is None:
        makespan = max((t.end_time for t in schedule), default=0)

    # Build explanation
    summary_parts = []
    if cpsat_response.status == SolverStatus.OPTIMAL:
        summary_parts.append(f"Found optimal schedule completing in {makespan} time units")
    elif cpsat_response.status in (SolverStatus.FEASIBLE, SolverStatus.TIMEOUT_BEST):
        summary_parts.append(f"Found feasible schedule completing in {makespan} time units")
        if cpsat_response.optimality_gap:
            summary_parts.append(f"(gap: {cpsat_response.optimality_gap:.2f}%)")
    else:
        summary_parts.append("Schedule found")

    summary_parts.append(f"with {len(schedule)} tasks")
    if original_request.resources:
        summary_parts.append(f"using {len(original_request.resources)} resources")

    explanation = SchedulingExplanation(summary=" ".join(summary_parts))

    return SolveSchedulingProblemResponse(
        status=cpsat_response.status,
        makespan=makespan,
        schedule=schedule,
        solve_time_ms=cpsat_response.solve_time_ms,
        optimality_gap=cpsat_response.optimality_gap,
        explanation=explanation,
    )
