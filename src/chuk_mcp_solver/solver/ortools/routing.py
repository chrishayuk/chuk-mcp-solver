"""Routing problem converters.

This module converts high-level routing problems (TSP/VRP) to/from CP-SAT models.
"""

import math

from chuk_mcp_solver.models import (
    CircuitParams,
    ConstraintKind,
    LinearTerm,
    Objective,
    ObjectiveSense,
    Route,
    RoutingExplanation,
    RoutingObjective,
    SearchConfig,
    SolveConstraintModelRequest,
    SolveConstraintModelResponse,
    SolverMode,
    SolveRoutingProblemRequest,
    SolveRoutingProblemResponse,
    SolverStatus,
    Variable,
    VariableDomain,
    VariableDomainType,
)
from chuk_mcp_solver.models import Constraint as ConstraintModel


def _calculate_euclidean_distance(coord1: tuple[float, float], coord2: tuple[float, float]) -> int:
    """Calculate Euclidean distance between two coordinates.

    Args:
        coord1: (x, y) or (lat, lon)
        coord2: (x, y) or (lat, lon)

    Returns:
        Distance rounded to nearest integer
    """
    dx = coord1[0] - coord2[0]
    dy = coord1[1] - coord2[1]
    return int(math.sqrt(dx * dx + dy * dy))


def _build_distance_matrix(request: SolveRoutingProblemRequest) -> list[list[int]]:
    """Build distance matrix from request.

    Args:
        request: Routing request

    Returns:
        Distance matrix where [i][j] = distance from location i to j
    """
    if request.distance_matrix is not None:
        return request.distance_matrix

    # Use Euclidean distance from coordinates
    n = len(request.locations)
    matrix = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                loc_i = request.locations[i]
                loc_j = request.locations[j]

                if loc_i.coordinates is None or loc_j.coordinates is None:
                    # Default to large distance if coordinates missing
                    matrix[i][j] = 999999
                else:
                    matrix[i][j] = _calculate_euclidean_distance(
                        loc_i.coordinates, loc_j.coordinates
                    )

    return matrix


def convert_routing_to_cpsat(
    request: SolveRoutingProblemRequest,
) -> SolveConstraintModelRequest:
    """Convert high-level routing problem to CP-SAT model.

    For single vehicle (TSP): Creates circuit constraint with all locations.
    For multiple vehicles (VRP): More complex model with vehicle assignment.

    Args:
        request: High-level routing request

    Returns:
        CP-SAT constraint model request
    """
    variables = []
    constraints = []

    n = len(request.locations)
    distance_matrix = _build_distance_matrix(request)

    # For now, implement single-vehicle TSP (Phase 4.1.2a)
    # Multi-vehicle VRP will come in Phase 4.1.2b
    if len(request.vehicles) > 1:
        raise NotImplementedError(
            "Multi-vehicle routing not yet implemented. For now, use single vehicle TSP by providing 0 or 1 vehicles."
        )

    # Single vehicle TSP using circuit constraint
    arc_vars = []
    distance_terms = []

    # Create boolean variable for each possible arc (i, j)
    for i in range(n):
        for j in range(n):
            if i != j:  # No self-loops
                arc_id = f"arc_{i}_{j}"

                variables.append(
                    Variable(
                        id=arc_id,
                        domain=VariableDomain(type=VariableDomainType.BOOL),
                        metadata={
                            "from": request.locations[i].id,
                            "to": request.locations[j].id,
                            "distance": distance_matrix[i][j],
                        },
                    )
                )

                arc_vars.append((i, j, arc_id))

                # Add to objective
                distance = distance_matrix[i][j]
                distance_terms.append(LinearTerm(var=arc_id, coef=distance))

    # Add circuit constraint - ensures valid Hamiltonian circuit
    constraints.append(
        ConstraintModel(
            id="hamiltonian_circuit",
            kind=ConstraintKind.CIRCUIT,
            params=CircuitParams(arcs=arc_vars),
            metadata={"description": "Must form complete tour visiting each location once"},
        )
    )

    # Create objective based on routing objective
    objective = None
    if request.objective == RoutingObjective.MINIMIZE_DISTANCE:
        objective = Objective(sense=ObjectiveSense.MINIMIZE, terms=distance_terms)
    elif request.objective == RoutingObjective.MINIMIZE_TIME:
        # For TSP, time = distance + service times (constant)
        # So minimizing distance also minimizes time
        objective = Objective(sense=ObjectiveSense.MINIMIZE, terms=distance_terms)
    elif request.objective == RoutingObjective.MINIMIZE_COST:
        # Apply cost_per_distance from vehicle (if provided)
        if request.vehicles:
            cost_per_dist = request.vehicles[0].cost_per_distance
            cost_terms = [
                LinearTerm(var=term.var, coef=term.coef * cost_per_dist) for term in distance_terms
            ]
            objective = Objective(sense=ObjectiveSense.MINIMIZE, terms=cost_terms)
        else:
            objective = Objective(sense=ObjectiveSense.MINIMIZE, terms=distance_terms)
    else:
        # MINIMIZE_VEHICLES doesn't apply to single-vehicle TSP
        objective = Objective(sense=ObjectiveSense.MINIMIZE, terms=distance_terms)

    return SolveConstraintModelRequest(
        mode=SolverMode.OPTIMIZE,
        variables=variables,
        constraints=constraints,
        objective=objective,
        search=SearchConfig(
            max_time_ms=request.max_time_ms,
            return_partial_solution=request.return_partial_solution,
        ),
    )


def convert_cpsat_to_routing_response(
    cpsat_response: SolveConstraintModelResponse,
    original_request: SolveRoutingProblemRequest,
) -> SolveRoutingProblemResponse:
    """Convert CP-SAT solution back to routing domain.

    Args:
        cpsat_response: CP-SAT solver response
        original_request: Original routing request

    Returns:
        High-level routing response
    """
    if cpsat_response.status in (
        SolverStatus.INFEASIBLE,
        SolverStatus.UNBOUNDED,
        SolverStatus.ERROR,
    ):
        # No solution
        return SolveRoutingProblemResponse(
            status=cpsat_response.status,
            solve_time_ms=cpsat_response.solve_time_ms,
            explanation=RoutingExplanation(
                summary=cpsat_response.explanation.summary
                if cpsat_response.explanation
                else f"Problem is {cpsat_response.status.value}"
            ),
        )

    if cpsat_response.status == SolverStatus.TIMEOUT_NO_SOLUTION:
        return SolveRoutingProblemResponse(
            status=cpsat_response.status,
            solve_time_ms=cpsat_response.solve_time_ms,
            explanation=RoutingExplanation(
                summary=cpsat_response.explanation.summary
                if cpsat_response.explanation
                else "Timeout with no solution found",
                recommendations=[
                    "Increase max_time_ms",
                    "Reduce number of locations",
                    "Provide distance_matrix instead of coordinates for faster solving",
                ],
            ),
        )

    if not cpsat_response.solutions:
        return SolveRoutingProblemResponse(
            status=cpsat_response.status,
            solve_time_ms=cpsat_response.solve_time_ms,
            explanation=RoutingExplanation(summary="No solution available"),
        )

    # Extract solution
    solution = cpsat_response.solutions[0]

    # Build distance matrix
    distance_matrix = _build_distance_matrix(original_request)
    n = len(original_request.locations)

    # Extract selected arcs
    arcs = {}
    for var in solution.variables:
        if var.value == 1 and var.id.startswith("arc_"):
            parts = var.id.split("_")
            from_idx = int(parts[1])
            to_idx = int(parts[2])
            arcs[from_idx] = to_idx

    # Reconstruct tour starting from location 0
    tour_indices = [0]
    current = 0
    while len(tour_indices) < n:
        if current not in arcs:
            break
        next_loc = arcs[current]
        tour_indices.append(next_loc)
        current = next_loc

    # Build route
    sequence = [original_request.locations[i].id for i in tour_indices]

    # Calculate total distance and time
    total_distance = 0
    total_time = 0

    for i in range(len(tour_indices)):
        from_idx = tour_indices[i]
        to_idx = tour_indices[(i + 1) % len(tour_indices)]
        total_distance += distance_matrix[from_idx][to_idx]

        # Add service time at current location
        total_time += original_request.locations[from_idx].service_time

    # Add travel time (assuming time = distance for now)
    total_time += total_distance

    # Calculate cost
    cost_per_dist = 1.0
    fixed_cost = 0.0
    if original_request.vehicles:
        cost_per_dist = original_request.vehicles[0].cost_per_distance
        fixed_cost = original_request.vehicles[0].fixed_cost

    total_cost = total_distance * cost_per_dist + fixed_cost

    vehicle_id = original_request.vehicles[0].id if original_request.vehicles else "vehicle_1"

    route = Route(
        vehicle_id=vehicle_id,
        sequence=sequence,
        total_distance=total_distance,
        total_time=total_time,
        total_cost=total_cost,
        load_timeline=[],  # TODO: implement for capacity-constrained routing
    )

    # Build explanation
    summary_parts = []
    if cpsat_response.status == SolverStatus.OPTIMAL:
        summary_parts.append(f"Found optimal route visiting {n} locations")
    elif cpsat_response.status in (SolverStatus.FEASIBLE, SolverStatus.TIMEOUT_BEST):
        summary_parts.append(f"Found feasible route visiting {n} locations")
        if cpsat_response.optimality_gap:
            summary_parts.append(f"(gap: {cpsat_response.optimality_gap:.2f}%)")
    else:
        summary_parts.append(f"Route visiting {n} locations")

    summary_parts.append(f"with total distance {total_distance}")

    explanation = RoutingExplanation(summary=" ".join(summary_parts))

    return SolveRoutingProblemResponse(
        status=cpsat_response.status,
        routes=[route],
        total_distance=total_distance,
        total_time=total_time,
        total_cost=total_cost,
        vehicles_used=1,
        solve_time_ms=cpsat_response.solve_time_ms,
        optimality_gap=cpsat_response.optimality_gap,
        explanation=explanation,
    )
