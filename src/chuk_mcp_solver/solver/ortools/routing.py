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
    Solution,
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
    n_locations = len(request.locations)
    n_vehicles = len(request.vehicles) if request.vehicles else 1
    distance_matrix = _build_distance_matrix(request)

    # Multi-vehicle VRP
    if n_vehicles > 1:
        return _build_vrp_model(request, n_locations, n_vehicles, distance_matrix)

    # Single vehicle TSP using circuit constraint
    variables: list[Variable] = []
    constraints: list[ConstraintModel] = []
    arc_vars = []
    distance_terms = []

    # Create boolean variable for each possible arc (i, j)
    for i in range(n_locations):
        for j in range(n_locations):
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
    n_locations = len(original_request.locations)
    n_vehicles = len(original_request.vehicles) if original_request.vehicles else 1
    distance_matrix = _build_distance_matrix(original_request)

    # Multi-vehicle VRP
    if n_vehicles > 1:
        return _extract_vrp_routes(
            cpsat_response, original_request, solution, n_locations, n_vehicles, distance_matrix
        )

    # Single-vehicle TSP
    n = n_locations

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


def _build_vrp_model(
    request: SolveRoutingProblemRequest,
    n_locs: int,
    n_vehs: int,
    distance_matrix: list[list[int]],
) -> SolveConstraintModelRequest:
    """Build multi-vehicle VRP model.

    Args:
        request: Original routing request
        n_locs: Number of locations
        n_vehs: Number of vehicles
        distance_matrix: Distance matrix

    Returns:
        CP-SAT constraint model request
    """
    from chuk_mcp_solver.models import (
        ConstraintSense,
        LinearConstraintParams,
    )

    variables: list[Variable] = []
    constraints: list[ConstraintModel] = []

    # Create arc variables: arc[i][j][v] = vehicle v goes from i to j
    arc_vars: dict[tuple[int, int, int], str] = {}

    for v_idx in range(n_vehs):
        for i in range(n_locs):
            for j in range(n_locs):
                # No self-loops for VRP (vehicles can skip locations entirely)
                if i != j:
                    arc_id = f"arc_{i}_{j}_v{v_idx}"

                    variables.append(
                        Variable(
                            id=arc_id,
                            domain=VariableDomain(type=VariableDomainType.BOOL),
                            metadata={
                                "from": request.locations[i].id,
                                "to": request.locations[j].id,
                                "vehicle": request.vehicles[v_idx].id,
                                "distance": distance_matrix[i][j],
                            },
                        )
                    )

                    arc_vars[(i, j, v_idx)] = arc_id

    # Flow conservation constraints per vehicle per location
    # For each vehicle and each location: incoming flow = outgoing flow
    for v_idx in range(n_vehs):
        for loc_idx in range(n_locs):
            incoming = []
            outgoing = []

            for i in range(n_locs):
                if i != loc_idx and (i, loc_idx, v_idx) in arc_vars:
                    incoming.append(LinearTerm(var=arc_vars[(i, loc_idx, v_idx)], coef=1))

            for j in range(n_locs):
                if j != loc_idx and (loc_idx, j, v_idx) in arc_vars:
                    outgoing.append(LinearTerm(var=arc_vars[(loc_idx, j, v_idx)], coef=1))

            # Flow in = flow out
            if incoming and outgoing:
                constraints.append(
                    ConstraintModel(
                        id=f"flow_balance_v{v_idx}_loc{loc_idx}",
                        kind=ConstraintKind.LINEAR,
                        params=LinearConstraintParams(
                            terms=incoming
                            + [LinearTerm(var=t.var, coef=-t.coef) for t in outgoing],
                            sense=ConstraintSense.EQUAL,
                            rhs=0,
                        ),
                        metadata={
                            "description": f"Flow balance for vehicle {request.vehicles[v_idx].id} at {request.locations[loc_idx].id}"
                        },
                    )
                )

    # Each customer visited exactly once (depot can be visited multiple times)
    # Assuming depot is at index 0
    for j in range(1, n_locs):  # Skip depot
        terms = []
        for i in range(n_locs):
            if i != j:
                for v_idx in range(n_vehs):
                    if (i, j, v_idx) in arc_vars:
                        terms.append(LinearTerm(var=arc_vars[(i, j, v_idx)], coef=1))

        if terms:  # Only add constraint if there are terms
            constraints.append(
                ConstraintModel(
                    id=f"visit_once_{j}",
                    kind=ConstraintKind.LINEAR,
                    params=LinearConstraintParams(terms=terms, sense=ConstraintSense.EQUAL, rhs=1),
                    metadata={
                        "description": f"Location {request.locations[j].id} visited exactly once"
                    },
                )
            )

    # Depot start/end constraints: each vehicle can leave depot at most once
    # And must return to depot if it leaves
    for v_idx in range(n_vehs):
        depot_outgoing = []
        depot_incoming = []

        for j in range(1, n_locs):  # Depot to customers
            if (0, j, v_idx) in arc_vars:
                depot_outgoing.append(LinearTerm(var=arc_vars[(0, j, v_idx)], coef=1))

        for i in range(1, n_locs):  # Customers to depot
            if (i, 0, v_idx) in arc_vars:
                depot_incoming.append(LinearTerm(var=arc_vars[(i, 0, v_idx)], coef=1))

        # Vehicle leaves depot at most once
        if depot_outgoing:
            constraints.append(
                ConstraintModel(
                    id=f"depot_start_v{v_idx}",
                    kind=ConstraintKind.LINEAR,
                    params=LinearConstraintParams(
                        terms=depot_outgoing, sense=ConstraintSense.LESS_EQUAL, rhs=1
                    ),
                    metadata={
                        "description": f"Vehicle {request.vehicles[v_idx].id} leaves depot at most once"
                    },
                )
            )

        # If vehicle leaves depot, it must return: outgoing = incoming
        if depot_outgoing and depot_incoming:
            constraints.append(
                ConstraintModel(
                    id=f"depot_return_v{v_idx}",
                    kind=ConstraintKind.LINEAR,
                    params=LinearConstraintParams(
                        terms=depot_outgoing
                        + [LinearTerm(var=t.var, coef=-t.coef) for t in depot_incoming],
                        sense=ConstraintSense.EQUAL,
                        rhs=0,
                    ),
                    metadata={
                        "description": f"Vehicle {request.vehicles[v_idx].id} returns to depot"
                    },
                )
            )

    # MTZ subtour elimination: use position variables u[i,v] for customer ordering
    # u[i,v] represents the position of location i in vehicle v's route (0 if not visited)
    # Constraint: if arc (i,j,v) is used, then u[j,v] = u[i,v] + 1
    position_vars: dict[tuple[int, int], str] = {}

    for v_idx in range(n_vehs):
        for i in range(1, n_locs):  # Customers only (depot has implicit position 0)
            pos_var_id = f"pos_{i}_v{v_idx}"
            variables.append(
                Variable(
                    id=pos_var_id,
                    domain=VariableDomain(
                        type=VariableDomainType.INTEGER,
                        lower=0,
                        upper=n_locs - 1,  # Max position
                    ),
                    metadata={
                        "location": request.locations[i].id,
                        "vehicle": request.vehicles[v_idx].id,
                    },
                )
            )
            position_vars[(i, v_idx)] = pos_var_id

    # MTZ constraints: if arc (i,j,v) is active, then pos[j,v] >= pos[i,v] + 1
    # Formulated as: pos[j,v] - pos[i,v] >= 1 - M * (1 - arc[i,j,v])
    # Or: pos[j,v] - pos[i,v] + M * arc[i,j,v] >= 1 where M = n_locs
    M = n_locs
    for v_idx in range(n_vehs):
        for i in range(1, n_locs):  # From customer
            for j in range(1, n_locs):  # To customer
                if i != j and (i, j, v_idx) in arc_vars:
                    constraints.append(
                        ConstraintModel(
                            id=f"mtz_{i}_{j}_v{v_idx}",
                            kind=ConstraintKind.LINEAR,
                            params=LinearConstraintParams(
                                terms=[
                                    LinearTerm(var=position_vars[(j, v_idx)], coef=1),
                                    LinearTerm(var=position_vars[(i, v_idx)], coef=-1),
                                    LinearTerm(var=arc_vars[(i, j, v_idx)], coef=-M),
                                ],
                                sense=ConstraintSense.GREATER_EQUAL,
                                rhs=1 - M,
                            ),
                            metadata={
                                "description": f"MTZ subtour elimination for {i}->{j} on vehicle {request.vehicles[v_idx].id}"
                            },
                        )
                    )

        # Position from depot: if arc (0,j,v) is used, pos[j,v] = 1
        # Formulated as: pos[j,v] >= arc[0,j,v]  AND  pos[j,v] <= arc[0,j,v] * M
        for j in range(1, n_locs):
            if (0, j, v_idx) in arc_vars:
                # Lower bound: pos[j,v] >= arc[0,j,v]
                constraints.append(
                    ConstraintModel(
                        id=f"mtz_depot_start_{j}_v{v_idx}",
                        kind=ConstraintKind.LINEAR,
                        params=LinearConstraintParams(
                            terms=[
                                LinearTerm(var=position_vars[(j, v_idx)], coef=1),
                                LinearTerm(var=arc_vars[(0, j, v_idx)], coef=-1),
                            ],
                            sense=ConstraintSense.GREATER_EQUAL,
                            rhs=0,
                        ),
                        metadata={
                            "description": f"Position from depot for {j} on vehicle {request.vehicles[v_idx].id}"
                        },
                    )
                )

    # Capacity constraints per vehicle (if demands present)
    if any(loc.demand > 0 for loc in request.locations):
        for v_idx in range(n_vehs):
            vehicle = request.vehicles[v_idx]

            # Sum of demands on arcs entering customers (not depot)
            demand_terms = []
            for j in range(1, n_locs):  # Customers only
                for i in range(n_locs):
                    if i != j and (i, j, v_idx) in arc_vars:
                        demand = request.locations[j].demand
                        if demand > 0:
                            demand_terms.append(
                                LinearTerm(var=arc_vars[(i, j, v_idx)], coef=demand)
                            )

            if demand_terms:
                constraints.append(
                    ConstraintModel(
                        id=f"capacity_v{v_idx}",
                        kind=ConstraintKind.LINEAR,
                        params=LinearConstraintParams(
                            terms=demand_terms,
                            sense=ConstraintSense.LESS_EQUAL,
                            rhs=vehicle.capacity,
                        ),
                        metadata={"description": f"Capacity limit for {vehicle.id}"},
                    )
                )

    # Vehicle usage boolean: vehicle_used[v] = 1 if vehicle v is used
    # Only needed for minimize_vehicles and minimize_cost objectives
    vehicle_used_vars = []
    if request.objective in (RoutingObjective.MINIMIZE_VEHICLES, RoutingObjective.MINIMIZE_COST):
        for v_idx in range(n_vehs):
            vehicle_used_id = f"vehicle_used_v{v_idx}"
            variables.append(
                Variable(
                    id=vehicle_used_id,
                    domain=VariableDomain(type=VariableDomainType.BOOL),
                    metadata={"vehicle": request.vehicles[v_idx].id, "type": "usage_indicator"},
                )
            )
            vehicle_used_vars.append(vehicle_used_id)

            # vehicle_used[v] >= any arc from depot (excluding self-loop)
            # This ensures vehicle_used=1 if vehicle leaves depot
            for j in range(1, n_locs):
                if (0, j, v_idx) in arc_vars:
                    constraints.append(
                        ConstraintModel(
                            id=f"link_usage_v{v_idx}_to_{j}",
                            kind=ConstraintKind.LINEAR,
                            params=LinearConstraintParams(
                                terms=[
                                    LinearTerm(var=vehicle_used_id, coef=1),
                                    LinearTerm(var=arc_vars[(0, j, v_idx)], coef=-1),
                                ],
                                sense=ConstraintSense.GREATER_EQUAL,
                                rhs=0,
                            ),
                            metadata={
                                "description": f"Link usage for {request.vehicles[v_idx].id} to {request.locations[j].id}"
                            },
                        )
                    )

    # Build objective
    objective_terms: list[LinearTerm] = []

    if request.objective == RoutingObjective.MINIMIZE_VEHICLES:
        # Minimize number of vehicles used
        for var_id in vehicle_used_vars:
            objective_terms.append(LinearTerm(var=var_id, coef=1))

    elif request.objective == RoutingObjective.MINIMIZE_COST:
        # Minimize total cost: fixed_cost * vehicle_used + distance_cost
        for v_idx, vehicle in enumerate(request.vehicles):
            # Fixed cost
            if vehicle.fixed_cost > 0 and v_idx < len(vehicle_used_vars):
                objective_terms.append(
                    LinearTerm(var=vehicle_used_vars[v_idx], coef=int(vehicle.fixed_cost))
                )

            # Distance cost
            for i, j, v in arc_vars:
                if v == v_idx:
                    dist_cost = int(distance_matrix[i][j] * vehicle.cost_per_distance)
                    objective_terms.append(LinearTerm(var=arc_vars[(i, j, v)], coef=dist_cost))

    else:  # MINIMIZE_DISTANCE or MINIMIZE_TIME
        # Sum all arc distances weighted by usage
        for (i, j, _v_idx), arc_id in arc_vars.items():
            objective_terms.append(LinearTerm(var=arc_id, coef=distance_matrix[i][j]))

    objective = (
        Objective(sense=ObjectiveSense.MINIMIZE, terms=objective_terms) if objective_terms else None
    )

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


def _extract_vrp_routes(
    cpsat_response: SolveConstraintModelResponse,
    original_request: SolveRoutingProblemRequest,
    solution: Solution,
    n_locs: int,
    n_vehs: int,
    distance_matrix: list[list[int]],
) -> SolveRoutingProblemResponse:
    """Extract routes from VRP solution.

    Args:
        cpsat_response: CP-SAT response
        original_request: Original request
        solution: Solution object
        n_locs: Number of locations
        n_vehs: Number of vehicles
        distance_matrix: Distance matrix

    Returns:
        Routing response with multiple routes
    """
    routes: list[Route] = []
    total_distance = 0
    total_time = 0
    total_cost = 0.0
    vehicles_used = 0

    # Extract which vehicles are used
    vehicle_used = {}
    for var in solution.variables:
        if var.id.startswith("vehicle_used_v"):
            v_idx = int(var.id.split("_v")[1])
            vehicle_used[v_idx] = var.value == 1

    # For each vehicle, extract its route
    for v_idx in range(n_vehs):
        vehicle = original_request.vehicles[v_idx]

        # Extract arcs for this vehicle
        arcs = {}
        for var in solution.variables:
            if var.value == 1 and var.id.startswith("arc_") and var.id.endswith(f"_v{v_idx}"):
                # Parse arc_i_j_vN
                parts = var.id.split("_")
                from_idx = int(parts[1])
                to_idx = int(parts[2])
                arcs[from_idx] = to_idx

        # Skip if vehicle has no arcs (not used)
        if not arcs:
            continue

        vehicles_used += 1

        # Reconstruct tour starting from depot (index 0)
        tour_indices = [0]
        current = 0
        visited = {0}

        # Follow arcs until we return to depot or run out of arcs
        max_iterations = n_locs + 1  # Safety limit
        iterations = 0
        while current in arcs and iterations < max_iterations:
            iterations += 1
            next_loc = arcs[current]

            if next_loc == 0 and len(tour_indices) > 1:
                # Returned to depot - add it and stop
                tour_indices.append(0)
                break

            if next_loc in visited:
                # Avoid infinite loop (shouldn't happen with valid solution)
                break

            tour_indices.append(next_loc)
            visited.add(next_loc)
            current = next_loc

        # Build sequence of location IDs
        sequence = [original_request.locations[i].id for i in tour_indices]

        # Calculate route distance, time, and load timeline
        route_distance = 0
        route_time = 0
        load_timeline: list[tuple[str, int]] = []
        current_load = 0

        for i in range(len(tour_indices) - 1):
            from_idx = tour_indices[i]
            to_idx = tour_indices[i + 1]
            route_distance += distance_matrix[from_idx][to_idx]

            # Update load (pick up at customer, drop off at depot)
            loc = original_request.locations[to_idx]
            if to_idx == 0:  # Depot - drop off
                current_load = 0
            else:  # Customer - pick up
                current_load += loc.demand

            load_timeline.append((loc.id, current_load))

            # Add service time
            route_time += loc.service_time

        # Add travel time
        route_time += route_distance

        # Calculate cost
        route_cost = route_distance * vehicle.cost_per_distance + vehicle.fixed_cost

        routes.append(
            Route(
                vehicle_id=vehicle.id,
                sequence=sequence,
                total_distance=route_distance,
                total_time=route_time,
                total_cost=route_cost,
                load_timeline=load_timeline,
            )
        )

        total_distance += route_distance
        total_time += route_time
        total_cost += route_cost

    # Build explanation
    summary_parts = []
    if cpsat_response.status == SolverStatus.OPTIMAL:
        summary_parts.append("Found optimal solution")
    elif cpsat_response.status in (SolverStatus.FEASIBLE, SolverStatus.TIMEOUT_BEST):
        summary_parts.append("Found feasible solution")
        if cpsat_response.optimality_gap:
            summary_parts.append(f"(gap: {cpsat_response.optimality_gap:.2f}%)")

    summary_parts.append(f"using {vehicles_used} of {n_vehs} vehicles")
    summary_parts.append(f"visiting {n_locs - 1} customers")  # -1 for depot
    summary_parts.append(f"with total distance {total_distance}")

    explanation = RoutingExplanation(summary=" ".join(summary_parts))

    return SolveRoutingProblemResponse(
        status=cpsat_response.status,
        routes=routes,
        total_distance=total_distance,
        total_time=total_time,
        total_cost=total_cost,
        vehicles_used=vehicles_used,
        solve_time_ms=cpsat_response.solve_time_ms,
        optimality_gap=cpsat_response.optimality_gap,
        explanation=explanation,
    )
