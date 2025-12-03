"""Assignment problem demonstration.

This example shows how to use the solve_assignment_problem tool for
task-to-agent assignment problems with various objectives and constraints.

Demonstrates:
- Minimize cost: Assign tasks to cheapest qualified agents
- Skill matching: Ensure agents have required skills for tasks
- Capacity constraints: Limit number of tasks per agent
- Load balancing: Distribute work evenly across agents
- Maximize assignments: Assign as many tasks as possible
"""

import asyncio

from chuk_mcp_solver.server import solve_assignment_problem


async def demo_simple_assignment():
    """Demo 1: Simple task assignment to minimize cost."""
    print("=" * 80)
    print("Demo 1: Simple Task Assignment (Minimize Cost)")
    print("=" * 80)
    print("Assigning 3 tasks to 2 workers with different hourly rates")
    print()

    response = await solve_assignment_problem(
        agents=[
            {"id": "alice", "capacity": 2, "cost_multiplier": 1.0},  # $50/hour
            {"id": "bob", "capacity": 2, "cost_multiplier": 1.5},  # $75/hour
        ],
        tasks=[
            {"id": "task_1", "duration": 2},  # 2 hours
            {"id": "task_2", "duration": 3},  # 3 hours
            {"id": "task_3", "duration": 1},  # 1 hour
        ],
        objective="minimize_cost",
    )

    print(f"Status: {response.status}")
    print(f"Total cost: ${response.total_cost * 50:.2f}")
    print("\nAssignments:")
    for assignment in response.assignments:
        cost_per_hour = assignment.cost * 50
        duration = next(
            t["duration"]
            for t in [
                {"id": "task_1", "duration": 2},
                {"id": "task_2", "duration": 3},
                {"id": "task_3", "duration": 1},
            ]
            if t["id"] == assignment.task_id
        )
        print(f"  {assignment.task_id} → {assignment.agent_id} (${cost_per_hour:.2f}, {duration}h)")

    print("\nAgent workload:")
    for agent_id, load in response.agent_load.items():
        print(f"  {agent_id}: {load} tasks")

    print(f"\n{response.explanation.summary}")
    print()


async def demo_skill_matching():
    """Demo 2: Assignment with skill requirements."""
    print("=" * 80)
    print("Demo 2: Assignment with Skill Matching")
    print("=" * 80)
    print("Assigning development tasks to engineers based on required skills")
    print()

    response = await solve_assignment_problem(
        agents=[
            {"id": "alice", "capacity": 2, "skills": ["python", "docker", "aws"]},
            {"id": "bob", "capacity": 2, "skills": ["react", "typescript", "nodejs"]},
            {"id": "charlie", "capacity": 2, "skills": ["python", "react", "postgres"]},
        ],
        tasks=[
            {"id": "backend_api", "duration": 5, "required_skills": ["python", "docker"]},
            {"id": "frontend_ui", "duration": 4, "required_skills": ["react", "typescript"]},
            {"id": "database_migration", "duration": 2, "required_skills": ["python", "postgres"]},
            {"id": "deployment", "duration": 3, "required_skills": ["docker", "aws"]},
        ],
        objective="minimize_cost",
    )

    print(f"Status: {response.status}")
    print("\nAssignments:")
    for assignment in response.assignments:
        print(f"  {assignment.task_id} → {assignment.agent_id}")

    print("\nAgent workload:")
    for agent_id, load in response.agent_load.items():
        print(f"  {agent_id}: {load} tasks")

    print(f"\n{response.explanation.summary}")
    if response.explanation.overloaded_agents:
        print(f"Overloaded agents: {', '.join(response.explanation.overloaded_agents)}")
    if response.explanation.underutilized_agents:
        print(f"Underutilized agents: {', '.join(response.explanation.underutilized_agents)}")
    print()


async def demo_load_balancing():
    """Demo 3: Balance workload across agents."""
    print("=" * 80)
    print("Demo 3: Load Balancing")
    print("=" * 80)
    print("Distributing 9 tasks evenly across 3 servers")
    print()

    response = await solve_assignment_problem(
        agents=[
            {"id": "server_1", "capacity": 5},
            {"id": "server_2", "capacity": 5},
            {"id": "server_3", "capacity": 5},
        ],
        tasks=[{"id": f"job_{i}", "duration": 1} for i in range(9)],
        objective="balance_load",
    )

    print(f"Status: {response.status}")
    print("\nAssignments:")
    for agent_id, load in response.agent_load.items():
        tasks = [a.task_id for a in response.assignments if a.agent_id == agent_id]
        print(f"  {agent_id}: {load} tasks - {', '.join(tasks)}")

    max_load = max(response.agent_load.values())
    min_load = min(response.agent_load.values())
    print(f"\nLoad distribution: min={min_load}, max={max_load}, variance={max_load - min_load}")
    print(f"\n{response.explanation.summary}")
    print()


async def demo_capacity_limits():
    """Demo 4: Assignment with capacity constraints."""
    print("=" * 80)
    print("Demo 4: Capacity Constraints")
    print("=" * 80)
    print("Assigning tasks with limited agent capacity")
    print()

    response = await solve_assignment_problem(
        agents=[
            {"id": "specialist_1", "capacity": 1, "cost_multiplier": 1.0},  # Can only take 1 task
            {"id": "specialist_2", "capacity": 1, "cost_multiplier": 1.2},  # Can only take 1 task
            {"id": "generalist", "capacity": 3, "cost_multiplier": 1.5},  # Can take 3 tasks
        ],
        tasks=[
            {"id": "urgent_task", "duration": 1, "priority": 3},
            {"id": "important_task", "duration": 1, "priority": 2},
            {"id": "normal_task_1", "duration": 1, "priority": 1},
            {"id": "normal_task_2", "duration": 1, "priority": 1},
        ],
        objective="minimize_cost",
    )

    print(f"Status: {response.status}")
    print(f"Total cost: {response.total_cost:.2f}")
    print("\nAssignments:")
    for assignment in response.assignments:
        print(f"  {assignment.task_id} → {assignment.agent_id} (cost: {assignment.cost:.2f})")

    print("\nAgent utilization:")
    for agent in [
        {"id": "specialist_1", "capacity": 1},
        {"id": "specialist_2", "capacity": 1},
        {"id": "generalist", "capacity": 3},
    ]:
        load = response.agent_load[agent["id"]]
        capacity = agent["capacity"]
        util_pct = (load / capacity * 100) if capacity > 0 else 0
        print(f"  {agent['id']}: {load}/{capacity} tasks ({util_pct:.0f}% utilized)")

    print(f"\n{response.explanation.summary}")
    print()


async def demo_maximize_assignments():
    """Demo 5: Maximize number of completed tasks."""
    print("=" * 80)
    print("Demo 5: Maximize Assignments (Optional Tasks)")
    print("=" * 80)
    print("Assign as many tasks as possible given limited capacity")
    print()

    response = await solve_assignment_problem(
        agents=[
            {"id": "team_member_1", "capacity": 2},
            {"id": "team_member_2", "capacity": 2},
        ],
        tasks=[
            {"id": "feature_a", "duration": 1},
            {"id": "feature_b", "duration": 1},
            {"id": "feature_c", "duration": 1},
            {"id": "feature_d", "duration": 1},
            {"id": "feature_e", "duration": 1},
        ],
        objective="maximize_assignments",
        force_assign_all=False,  # Some tasks can remain unassigned
    )

    print(f"Status: {response.status}")
    print(f"Assigned: {len(response.assignments)} out of 5 tasks")
    print("\nCompleted tasks:")
    for assignment in response.assignments:
        print(f"  {assignment.task_id} → {assignment.agent_id}")

    if response.unassigned_tasks:
        print(f"\nUnassigned tasks (backlog): {', '.join(response.unassigned_tasks)}")

    print("\nAgent workload:")
    for agent_id, load in response.agent_load.items():
        print(f"  {agent_id}: {load} tasks")

    print(f"\n{response.explanation.summary}")
    print()


async def demo_infeasible_assignment():
    """Demo 6: Infeasible assignment (not enough capacity)."""
    print("=" * 80)
    print("Demo 6: Infeasible Assignment")
    print("=" * 80)
    print("Attempting to assign more tasks than capacity allows")
    print()

    response = await solve_assignment_problem(
        agents=[
            {"id": "worker", "capacity": 1},  # Can only handle 1 task
        ],
        tasks=[
            {"id": "task_1", "duration": 1},
            {"id": "task_2", "duration": 1},
        ],
        objective="minimize_cost",
        force_assign_all=True,  # Must assign all tasks
    )

    print(f"Status: {response.status}")
    print(f"Explanation: {response.explanation.summary}")
    print()


async def demo_mixed_priorities():
    """Demo 7: Assignment with cost matrix (custom costs)."""
    print("=" * 80)
    print("Demo 7: Custom Cost Matrix")
    print("=" * 80)
    print("Using custom cost matrix to encode task-agent preferences")
    print()

    # Cost matrix: cost[task_idx][agent_idx]
    # Lower cost = better match
    cost_matrix = [
        [100, 500, 300],  # Task 0: alice is best match
        [400, 100, 200],  # Task 1: bob is best match
        [200, 300, 100],  # Task 2: charlie is best match
    ]

    response = await solve_assignment_problem(
        agents=[
            {"id": "alice", "capacity": 2},
            {"id": "bob", "capacity": 2},
            {"id": "charlie", "capacity": 2},
        ],
        tasks=[
            {"id": "research_task"},
            {"id": "coding_task"},
            {"id": "review_task"},
        ],
        cost_matrix=cost_matrix,
        objective="minimize_cost",
    )

    print(f"Status: {response.status}")
    print(f"Total cost: {response.total_cost}")
    print("\nAssignments (matched by cost preferences):")
    for assignment in response.assignments:
        print(f"  {assignment.task_id} → {assignment.agent_id} (cost: {assignment.cost})")

    print(f"\n{response.explanation.summary}")
    print()


async def main():
    """Run all assignment demos."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "ASSIGNMENT PROBLEM DEMONSTRATIONS" + " " * 25 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    await demo_simple_assignment()
    await demo_skill_matching()
    await demo_load_balancing()
    await demo_capacity_limits()
    await demo_maximize_assignments()
    await demo_infeasible_assignment()
    await demo_mixed_priorities()

    print("=" * 80)
    print("All demos completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
