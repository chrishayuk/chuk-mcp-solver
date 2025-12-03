"""Assignment problem converters.

This module converts high-level assignment problems to/from CP-SAT models.
"""

from chuk_mcp_solver.models import (
    Assignment,
    AssignmentExplanation,
    AssignmentObjective,
    ConstraintKind,
    ConstraintSense,
    LinearConstraintParams,
    LinearTerm,
    Objective,
    ObjectiveSense,
    SearchConfig,
    SolveAssignmentProblemRequest,
    SolveAssignmentProblemResponse,
    SolveConstraintModelRequest,
    SolveConstraintModelResponse,
    SolverMode,
    SolverStatus,
    Variable,
    VariableDomain,
    VariableDomainType,
)
from chuk_mcp_solver.models import Constraint as ConstraintModel


def _build_cost_matrix(
    request: SolveAssignmentProblemRequest,
) -> tuple[list[list[float]], set[tuple[int, int]]]:
    """Build cost matrix from request.

    Args:
        request: Assignment request

    Returns:
        Tuple of (cost_matrix, incompatible_pairs) where:
        - cost_matrix[task_idx][agent_idx] = cost to assign task to agent
        - incompatible_pairs is set of (task_idx, agent_idx) that are incompatible
    """
    if request.cost_matrix is not None:
        return request.cost_matrix, set()

    # Build cost matrix from agent cost_multiplier and task duration
    n_tasks = len(request.tasks)
    n_agents = len(request.agents)
    matrix = [[0.0 for _ in range(n_agents)] for _ in range(n_tasks)]
    incompatible = set()

    for task_idx, task in enumerate(request.tasks):
        for agent_idx, agent in enumerate(request.agents):
            # Check skill compatibility
            if task.required_skills:
                has_all_skills = all(skill in agent.skills for skill in task.required_skills)
                if not has_all_skills:
                    # Mark as incompatible
                    incompatible.add((task_idx, agent_idx))
                    matrix[task_idx][agent_idx] = 0.0  # Cost doesn't matter, will be forbidden
                else:
                    matrix[task_idx][agent_idx] = agent.cost_multiplier * task.duration
            else:
                matrix[task_idx][agent_idx] = agent.cost_multiplier * task.duration

    return matrix, incompatible


def convert_assignment_to_cpsat(
    request: SolveAssignmentProblemRequest,
) -> SolveConstraintModelRequest:
    """Convert high-level assignment problem to CP-SAT model.

    Creates an assignment model with:
    - Binary variables for each (task, agent) pair
    - Constraints ensuring each task assigned to exactly one agent (or optionally unassigned)
    - Agent capacity constraints
    - Skill matching via cost matrix
    - Cost/count/balance optimization objective

    Args:
        request: High-level assignment request

    Returns:
        CP-SAT constraint model request
    """
    variables = []
    constraints = []

    n_tasks = len(request.tasks)
    n_agents = len(request.agents)
    cost_matrix, incompatible_pairs = _build_cost_matrix(request)

    # Create binary assignment variable for each (task, agent) pair
    for task_idx, task in enumerate(request.tasks):
        for agent_idx, agent in enumerate(request.agents):
            var_id = f"assign_t{task_idx}_a{agent_idx}"
            variables.append(
                Variable(
                    id=var_id,
                    domain=VariableDomain(type=VariableDomainType.BOOL),
                    metadata={
                        "task_id": task.id,
                        "agent_id": agent.id,
                        "cost": cost_matrix[task_idx][agent_idx],
                    },
                )
            )

    # Add constraints to forbid incompatible assignments (skill mismatches)
    for task_idx, agent_idx in incompatible_pairs:
        var_id = f"assign_t{task_idx}_a{agent_idx}"
        task_id = request.tasks[task_idx].id
        agent_id = request.agents[agent_idx].id
        constraints.append(
            ConstraintModel(
                id=f"forbid_{task_id}_to_{agent_id}",
                kind=ConstraintKind.LINEAR,
                params=LinearConstraintParams(
                    terms=[LinearTerm(var=var_id, coef=1)],
                    sense=ConstraintSense.EQUAL,
                    rhs=0,
                ),
                metadata={
                    "description": f"Task {task_id} incompatible with agent {agent_id} (missing skills)"
                },
            )
        )

    # Task assignment constraints: each task assigned to exactly one agent (or zero if not force_assign_all)
    for task_idx, task in enumerate(request.tasks):
        terms = [
            LinearTerm(var=f"assign_t{task_idx}_a{agent_idx}", coef=1)
            for agent_idx in range(n_agents)
        ]

        if request.force_assign_all:
            # Must assign to exactly one agent
            constraints.append(
                ConstraintModel(
                    id=f"task_{task.id}_assignment",
                    kind=ConstraintKind.LINEAR,
                    params=LinearConstraintParams(
                        terms=terms,
                        sense=ConstraintSense.EQUAL,
                        rhs=1,
                    ),
                    metadata={
                        "description": f"Task {task.id} must be assigned to exactly one agent"
                    },
                )
            )
        else:
            # Can assign to at most one agent (or leave unassigned)
            constraints.append(
                ConstraintModel(
                    id=f"task_{task.id}_assignment",
                    kind=ConstraintKind.LINEAR,
                    params=LinearConstraintParams(
                        terms=terms,
                        sense=ConstraintSense.LESS_EQUAL,
                        rhs=1,
                    ),
                    metadata={
                        "description": f"Task {task.id} assigned to at most one agent (can be unassigned)"
                    },
                )
            )

    # Agent capacity constraints: each agent can handle at most capacity tasks
    for agent_idx, agent in enumerate(request.agents):
        terms = [
            LinearTerm(var=f"assign_t{task_idx}_a{agent_idx}", coef=1)
            for task_idx in range(n_tasks)
        ]

        constraints.append(
            ConstraintModel(
                id=f"agent_{agent.id}_capacity",
                kind=ConstraintKind.LINEAR,
                params=LinearConstraintParams(
                    terms=terms,
                    sense=ConstraintSense.LESS_EQUAL,
                    rhs=agent.capacity,
                ),
                metadata={
                    "description": f"Agent {agent.id} can handle at most {agent.capacity} tasks"
                },
            )
        )

    # Create objective based on assignment objective
    objective = None
    if request.objective == AssignmentObjective.MINIMIZE_COST:
        # Minimize total cost (incompatible pairs already forbidden by constraints)
        cost_terms = []
        for task_idx in range(n_tasks):
            for agent_idx in range(n_agents):
                if (task_idx, agent_idx) not in incompatible_pairs:
                    cost = int(cost_matrix[task_idx][agent_idx] * 100)  # Scale to integers
                    cost_terms.append(LinearTerm(var=f"assign_t{task_idx}_a{agent_idx}", coef=cost))
        objective = Objective(sense=ObjectiveSense.MINIMIZE, terms=cost_terms)

    elif request.objective == AssignmentObjective.MAXIMIZE_ASSIGNMENTS:
        # Maximize number of assigned tasks
        assignment_terms = []
        for task_idx in range(n_tasks):
            for agent_idx in range(n_agents):
                assignment_terms.append(LinearTerm(var=f"assign_t{task_idx}_a{agent_idx}", coef=1))
        objective = Objective(sense=ObjectiveSense.MAXIMIZE, terms=assignment_terms)

    elif request.objective == AssignmentObjective.BALANCE_LOAD:
        # Minimize maximum load - we'll use a different approach
        # Add a variable for max load and minimize it
        variables.append(
            Variable(
                id="max_load",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=n_tasks),
                metadata={"description": "Maximum load across all agents"},
            )
        )

        # For each agent, constrain: load <= max_load
        for agent_idx, agent in enumerate(request.agents):
            load_terms = [
                LinearTerm(var=f"assign_t{task_idx}_a{agent_idx}", coef=1)
                for task_idx in range(n_tasks)
            ]
            load_terms.append(LinearTerm(var="max_load", coef=-1))

            constraints.append(
                ConstraintModel(
                    id=f"balance_agent_{agent.id}",
                    kind=ConstraintKind.LINEAR,
                    params=LinearConstraintParams(
                        terms=load_terms,
                        sense=ConstraintSense.LESS_EQUAL,
                        rhs=0,
                    ),
                    metadata={"description": f"Agent {agent.id} load <= max_load"},
                )
            )

        # Minimize max_load
        objective = Objective(
            sense=ObjectiveSense.MINIMIZE, terms=[LinearTerm(var="max_load", coef=1)]
        )

    return SolveConstraintModelRequest(
        mode=SolverMode.OPTIMIZE,
        variables=variables,
        constraints=constraints,
        objective=objective,
        search=SearchConfig(
            max_time_ms=request.max_time_ms,
            return_partial_solution=request.return_partial_solution,
            enable_solution_caching=False,  # Disable caching for assignment problems (agent IDs may collide)
        ),
    )


def convert_cpsat_to_assignment_response(
    cpsat_response: SolveConstraintModelResponse,
    original_request: SolveAssignmentProblemRequest,
) -> SolveAssignmentProblemResponse:
    """Convert CP-SAT solution back to assignment domain.

    Args:
        cpsat_response: CP-SAT solver response
        original_request: Original assignment request

    Returns:
        High-level assignment response
    """
    if cpsat_response.status in (
        SolverStatus.INFEASIBLE,
        SolverStatus.UNBOUNDED,
        SolverStatus.ERROR,
    ):
        # No solution
        return SolveAssignmentProblemResponse(
            status=cpsat_response.status,
            solve_time_ms=cpsat_response.solve_time_ms,
            explanation=AssignmentExplanation(
                summary=cpsat_response.explanation.summary
                if cpsat_response.explanation
                else f"Problem is {cpsat_response.status.value}"
            ),
        )

    if cpsat_response.status == SolverStatus.TIMEOUT_NO_SOLUTION:
        return SolveAssignmentProblemResponse(
            status=cpsat_response.status,
            solve_time_ms=cpsat_response.solve_time_ms,
            explanation=AssignmentExplanation(
                summary=cpsat_response.explanation.summary
                if cpsat_response.explanation
                else "Timeout with no solution found",
                recommendations=[
                    "Increase max_time_ms",
                    "Reduce number of tasks or agents",
                    "Relax force_assign_all constraint",
                    "Add more agent capacity",
                ],
            ),
        )

    if not cpsat_response.solutions:
        return SolveAssignmentProblemResponse(
            status=cpsat_response.status,
            solve_time_ms=cpsat_response.solve_time_ms,
            explanation=AssignmentExplanation(summary="No solution available"),
        )

    # Extract solution
    solution = cpsat_response.solutions[0]
    cost_matrix, _ = _build_cost_matrix(original_request)

    # Extract assignments
    assignments = []
    agent_load: dict[str, int] = {agent.id: 0 for agent in original_request.agents}
    unassigned_tasks = []
    total_cost = 0.0

    for task_idx, task in enumerate(original_request.tasks):
        assigned = False
        for agent_idx, agent in enumerate(original_request.agents):
            var_id = f"assign_t{task_idx}_a{agent_idx}"
            var = next((v for v in solution.variables if v.id == var_id), None)

            if var and var.value == 1:
                cost = cost_matrix[task_idx][agent_idx]
                assignments.append(Assignment(task_id=task.id, agent_id=agent.id, cost=cost))
                agent_load[agent.id] += 1
                total_cost += cost
                assigned = True
                break

        if not assigned:
            unassigned_tasks.append(task.id)

    # Build explanation
    summary_parts = []
    if cpsat_response.status == SolverStatus.OPTIMAL:
        num_agents_used = len([load for load in agent_load.values() if load > 0])
        summary_parts.append(
            f"Optimal assignment: {len(assignments)} tasks assigned to {num_agents_used} agents"
        )
    elif cpsat_response.status in (SolverStatus.FEASIBLE, SolverStatus.TIMEOUT_BEST):
        num_agents_used = len([load for load in agent_load.values() if load > 0])
        summary_parts.append(
            f"Feasible assignment: {len(assignments)} tasks assigned to {num_agents_used} agents"
        )
        if cpsat_response.optimality_gap:
            summary_parts.append(f"(gap: {cpsat_response.optimality_gap:.2f}%)")
    else:
        summary_parts.append(f"Assignment: {len(assignments)} tasks assigned")

    if unassigned_tasks:
        summary_parts.append(f", {len(unassigned_tasks)} unassigned")

    # Identify over/underutilized agents
    overloaded_agents = []
    underutilized_agents = []

    if agent_load:
        avg_load = sum(agent_load.values()) / len(agent_load)
        for agent in original_request.agents:
            load = agent_load[agent.id]
            if load > avg_load * 1.5:  # More than 50% above average
                overloaded_agents.append(agent.id)
            elif load < avg_load * 0.5 and agent.capacity > 0:  # Less than 50% of average
                underutilized_agents.append(agent.id)

    explanation = AssignmentExplanation(
        summary=" ".join(summary_parts),
        overloaded_agents=overloaded_agents,
        underutilized_agents=underutilized_agents,
    )

    return SolveAssignmentProblemResponse(
        status=cpsat_response.status,
        assignments=assignments,
        unassigned_tasks=unassigned_tasks,
        agent_load=agent_load,
        total_cost=total_cost,
        solve_time_ms=cpsat_response.solve_time_ms,
        optimality_gap=cpsat_response.optimality_gap,
        explanation=explanation,
    )
