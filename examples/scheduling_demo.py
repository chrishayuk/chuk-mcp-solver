"""Scheduling problem demonstration.

This example shows how to use the high-level solve_scheduling_problem tool
to solve various scheduling scenarios without directly working with CP-SAT.
"""

import asyncio

from chuk_mcp_solver.server import solve_scheduling_problem


async def demo_simple_project_schedule():
    """Demo 1: Simple project with sequential dependencies."""
    print("=" * 70)
    print("Demo 1: Simple Project Schedule (build → test → deploy)")
    print("=" * 70)

    response = await solve_scheduling_problem(
        tasks=[
            {"id": "build", "duration": 10},
            {"id": "test", "duration": 5, "dependencies": ["build"]},
            {"id": "deploy", "duration": 3, "dependencies": ["test"]},
        ],
        objective="minimize_makespan",
    )

    print(f"Status: {response.status}")
    print(f"Makespan: {response.makespan} time units")
    print(f"Solve time: {response.solve_time_ms}ms")
    print("\nSchedule:")
    for task in response.schedule:
        print(f"  {task.task_id}: start={task.start_time}, end={task.end_time}")
    print(f"\n{response.explanation.summary}")
    print()


async def demo_parallel_tasks():
    """Demo 2: Independent tasks that can run in parallel."""
    print("=" * 70)
    print("Demo 2: Parallel Task Execution")
    print("=" * 70)

    response = await solve_scheduling_problem(
        tasks=[
            {"id": "frontend", "duration": 8},
            {"id": "backend", "duration": 10},
            {"id": "database", "duration": 6},
            {
                "id": "integration",
                "duration": 4,
                "dependencies": ["frontend", "backend", "database"],
            },
        ],
        objective="minimize_makespan",
    )

    print(f"Status: {response.status}")
    print(f"Makespan: {response.makespan} time units")
    print("  (frontend, backend, database run in parallel)")
    print("\nSchedule:")
    for task in sorted(response.schedule, key=lambda t: t.start_time):
        deps = (
            "after "
            + ", ".join([t.task_id for t in response.schedule if t.end_time == task.start_time])
            if task.start_time > 0
            else "immediately"
        )
        print(f"  {task.task_id}: start={task.start_time}, end={task.end_time} ({deps})")
    print()


async def demo_resource_constraints():
    """Demo 3: Tasks competing for limited resources."""
    print("=" * 70)
    print("Demo 3: Resource Constrained Scheduling")
    print("=" * 70)

    response = await solve_scheduling_problem(
        tasks=[
            {"id": "task_a", "duration": 5, "resources_required": {"cpu": 2}},
            {"id": "task_b", "duration": 3, "resources_required": {"cpu": 3}},
            {"id": "task_c", "duration": 4, "resources_required": {"cpu": 1}},
        ],
        resources=[{"id": "cpu", "capacity": 4}],
        objective="minimize_makespan",
    )

    print(f"Status: {response.status}")
    print(f"Makespan: {response.makespan} time units")
    print("Resource: 4 CPU units available")
    print("\nSchedule:")
    for task in sorted(response.schedule, key=lambda t: (t.start_time, t.task_id)):
        cpu_used = task.resources_used.get("cpu", 0)
        print(f"  {task.task_id}: start={task.start_time}, end={task.end_time}, cpu={cpu_used}")

    # Show timeline
    print("\nResource usage timeline (4 CPU capacity):")
    for t in range(response.makespan + 1):
        active = [task for task in response.schedule if task.start_time <= t < task.end_time]
        total_cpu = sum(task.resources_used.get("cpu", 0) for task in active)
        bar = "█" * total_cpu + "░" * (4 - total_cpu)
        tasks_str = ", ".join(task.task_id for task in active) if active else "idle"
        print(f"  t={t:2d}: [{bar}] {total_cpu}/4 CPU ({tasks_str})")
    print()


async def demo_deadlines_and_release_times():
    """Demo 4: Tasks with earliest start times and deadlines."""
    print("=" * 70)
    print("Demo 4: Deadlines and Release Times")
    print("=" * 70)

    response = await solve_scheduling_problem(
        tasks=[
            {"id": "prep", "duration": 2, "earliest_start": 0},
            {"id": "main_work", "duration": 6, "dependencies": ["prep"], "deadline": 10},
            {"id": "review", "duration": 3, "dependencies": ["main_work"], "earliest_start": 8},
        ],
        objective="minimize_makespan",
    )

    print(f"Status: {response.status}")
    print(f"Makespan: {response.makespan} time units")
    print("\nSchedule:")
    for task in response.schedule:
        constraints = []
        original_task = next(
            t
            for t in [
                {"id": "prep", "earliest_start": 0},
                {"id": "main_work", "deadline": 10},
                {"id": "review", "earliest_start": 8},
            ]
            if t["id"] == task.task_id
        )
        if "earliest_start" in original_task:
            constraints.append(f"release≥{original_task['earliest_start']}")
        if "deadline" in original_task:
            constraints.append(f"deadline≤{original_task['deadline']}")
        constraints_str = " [" + ", ".join(constraints) + "]" if constraints else ""
        print(f"  {task.task_id}: start={task.start_time}, end={task.end_time}{constraints_str}")
    print()


async def demo_infeasible_problem():
    """Demo 5: Infeasible problem (impossible deadline)."""
    print("=" * 70)
    print("Demo 5: Infeasible Problem Detection")
    print("=" * 70)

    response = await solve_scheduling_problem(
        tasks=[
            {"id": "long_task", "duration": 10, "deadline": 5},  # Impossible!
        ],
        objective="minimize_makespan",
    )

    print(f"Status: {response.status}")
    print(f"Makespan: {response.makespan}")
    print(f"\n{response.explanation.summary}")
    print("\nThis demonstrates the solver's ability to detect impossible constraints.")
    print()


async def demo_complex_project():
    """Demo 6: More complex project with mixed constraints."""
    print("=" * 70)
    print("Demo 6: Complex Project (DevOps Pipeline)")
    print("=" * 70)

    response = await solve_scheduling_problem(
        tasks=[
            # Phase 1: Setup (can run in parallel)
            {"id": "setup_env", "duration": 2},
            {"id": "install_deps", "duration": 5},
            # Phase 2: Build (depends on setup)
            {
                "id": "compile",
                "duration": 8,
                "dependencies": ["setup_env", "install_deps"],
                "resources_required": {"cpu": 4},
            },
            # Phase 3: Testing (depends on build)
            {
                "id": "unit_tests",
                "duration": 4,
                "dependencies": ["compile"],
                "resources_required": {"cpu": 2},
            },
            {
                "id": "integration_tests",
                "duration": 6,
                "dependencies": ["compile"],
                "resources_required": {"cpu": 2},
            },
            # Phase 4: Quality checks (depends on tests)
            {
                "id": "lint",
                "duration": 2,
                "dependencies": ["compile"],
                "resources_required": {"cpu": 1},
            },
            {
                "id": "security_scan",
                "duration": 5,
                "dependencies": ["compile"],
                "resources_required": {"cpu": 1},
            },
            # Phase 5: Deploy (depends on everything)
            {
                "id": "deploy",
                "duration": 3,
                "dependencies": ["unit_tests", "integration_tests", "lint", "security_scan"],
                "deadline": 35,
            },
        ],
        resources=[{"id": "cpu", "capacity": 4}],
        objective="minimize_makespan",
    )

    print(f"Status: {response.status}")
    print(f"Makespan: {response.makespan} time units")
    print(f"Solve time: {response.solve_time_ms}ms")

    # Group by phase
    phases = {
        "Setup": ["setup_env", "install_deps"],
        "Build": ["compile"],
        "Testing": ["unit_tests", "integration_tests"],
        "Quality": ["lint", "security_scan"],
        "Deploy": ["deploy"],
    }

    print("\nSchedule by phase:")
    for phase_name, task_ids in phases.items():
        print(f"\n  {phase_name}:")
        for task_id in task_ids:
            task = next(t for t in response.schedule if t.task_id == task_id)
            cpu = task.resources_used.get("cpu", 0)
            cpu_str = f", cpu={cpu}" if cpu > 0 else ""
            print(f"    {task.task_id}: [{task.start_time}-{task.end_time}]{cpu_str}")

    print(f"\n{response.explanation.summary}")
    print()


async def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "SCHEDULING PROBLEM EXAMPLES" + " " * 26 + "║")
    print("║" + " " * 68 + "║")
    print("║  High-level scheduling API for task optimization with" + " " * 13 + "║")
    print("║  dependencies, resources, deadlines, and release times." + " " * 12 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")

    await demo_simple_project_schedule()
    await demo_parallel_tasks()
    await demo_resource_constraints()
    await demo_deadlines_and_release_times()
    await demo_infeasible_problem()
    await demo_complex_project()

    print("=" * 70)
    print("All demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
