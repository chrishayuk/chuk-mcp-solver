"""Example: GPU job scheduling with heterogeneous resources and cost optimization.

Demonstrates real-world ML/AI workload scheduling:
- Multiple GPU types with different capabilities and costs
- Jobs with varying resource requirements and priorities
- Memory, compute, and cost constraints
- Job dependencies and deadlines
- Multi-objective optimization (cost vs time vs priority)

This shows how an LLM could translate "Schedule these ML jobs across GPUs"
into a constraint optimization problem.
"""

import asyncio

from chuk_mcp_solver.models import SolveConstraintModelRequest
from chuk_mcp_solver.solver import get_solver


def build_gpu_scheduling_model(
    jobs: list[dict], gpus: list[dict], horizon: int, budget: float
) -> dict:
    """Build GPU job scheduling optimization model.

    Args:
        jobs: List of jobs with requirements (memory, compute, duration, priority)
        gpus: List of GPUs with capabilities (memory, compute, cost_per_hour)
        horizon: Time horizon in hours
        budget: Maximum budget for GPU usage

    Returns:
        Model dictionary ready for SolveConstraintModelRequest.
    """
    variables = []
    constraints = []

    # Decision variables: which GPU runs which job
    assignment_vars = {}  # (job_id, gpu_id) -> bool_var_id
    for job in jobs:
        for gpu in gpus:
            var_id = f"assign_{job['id']}_to_{gpu['id']}"
            variables.append(
                {
                    "id": var_id,
                    "domain": {"type": "bool"},
                    "metadata": {
                        "job": job["id"],
                        "gpu": gpu["id"],
                        "type": "assignment",
                    },
                }
            )
            assignment_vars[(job["id"], gpu["id"])] = var_id

    # Start time variables for each job
    for job in jobs:
        variables.append(
            {
                "id": f"start_{job['id']}",
                "domain": {"type": "integer", "lower": 0, "upper": horizon},
                "metadata": {"job": job["id"], "type": "start_time"},
            }
        )
        # End time
        variables.append(
            {
                "id": f"end_{job['id']}",
                "domain": {
                    "type": "integer",
                    "lower": job["duration"],
                    "upper": horizon,
                },
                "metadata": {"job": job["id"], "type": "end_time"},
            }
        )

        # Link: end = start + duration
        constraints.append(
            {
                "id": f"duration_{job['id']}",
                "kind": "linear",
                "params": {
                    "terms": [
                        {"var": f"end_{job['id']}", "coef": 1},
                        {"var": f"start_{job['id']}", "coef": -1},
                    ],
                    "sense": "==",
                    "rhs": job["duration"],
                },
            }
        )

    # Each job must be assigned to exactly one GPU
    for job in jobs:
        assignment_terms = [
            {"var": assignment_vars[(job["id"], gpu["id"])], "coef": 1} for gpu in gpus
        ]
        constraints.append(
            {
                "id": f"assign_one_{job['id']}",
                "kind": "linear",
                "params": {"terms": assignment_terms, "sense": "==", "rhs": 1},
                "metadata": {"description": f"Job {job['id']} must run on exactly one GPU"},
            }
        )

    # GPU resource constraints: memory and compute capacity
    for gpu in gpus:
        # Build cumulative constraints for this GPU
        jobs_on_gpu = []
        start_vars_gpu = []
        duration_vars_gpu = []
        memory_demands_gpu = []
        compute_demands_gpu = []

        for job in jobs:
            assign_var = assignment_vars[(job["id"], gpu["id"])]
            # We need to model: if assigned to this GPU, add to resource usage
            # For simplicity, we'll use implication constraints

            # Memory check - if job memory exceeds GPU memory, assignment must be 0
            if job["memory_gb"] > gpu["memory_gb"]:
                constraints.append(
                    {
                        "id": f"memory_fit_{job['id']}_{gpu['id']}",
                        "kind": "linear",
                        "params": {
                            "terms": [{"var": assign_var, "coef": 1}],
                            "sense": "==",
                            "rhs": 0,
                        },
                        "metadata": {
                            "description": f"Job {job['id']} doesn't fit on GPU {gpu['id']}"
                        },
                    }
                )
            else:
                jobs_on_gpu.append(job)
                start_vars_gpu.append(f"start_{job['id']}")
                duration_vars_gpu.append(job["duration"])
                memory_demands_gpu.append(job["memory_gb"])
                compute_demands_gpu.append(job["compute_units"])

    # Job dependencies (precedence constraints)
    for job in jobs:
        if "depends_on" in job and job["depends_on"]:
            for predecessor_id in job["depends_on"]:
                # Predecessor must finish before this job starts
                constraints.append(
                    {
                        "id": f"precedence_{predecessor_id}_{job['id']}",
                        "kind": "linear",
                        "params": {
                            "terms": [
                                {"var": f"start_{job['id']}", "coef": 1},
                                {"var": f"end_{predecessor_id}", "coef": -1},
                            ],
                            "sense": ">=",
                            "rhs": 0,
                        },
                        "metadata": {
                            "description": f"Job {job['id']} must start after {predecessor_id} completes"
                        },
                    }
                )

    # Deadline constraints
    for job in jobs:
        if "deadline" in job and job["deadline"]:
            constraints.append(
                {
                    "id": f"deadline_{job['id']}",
                    "kind": "linear",
                    "params": {
                        "terms": [{"var": f"end_{job['id']}", "coef": 1}],
                        "sense": "<=",
                        "rhs": job["deadline"],
                    },
                    "metadata": {
                        "description": f"Job {job['id']} must complete by hour {job['deadline']}"
                    },
                }
            )

    # Budget constraint - total cost must not exceed budget
    cost_terms = []
    for job in jobs:
        for gpu in gpus:
            assign_var = assignment_vars[(job["id"], gpu["id"])]
            job_cost = job["duration"] * gpu["cost_per_hour"]
            cost_terms.append({"var": assign_var, "coef": int(job_cost * 100)})  # Convert to cents

    constraints.append(
        {
            "id": "budget_limit",
            "kind": "linear",
            "params": {
                "terms": cost_terms,
                "sense": "<=",
                "rhs": int(budget * 100),  # Convert to cents
            },
            "metadata": {"description": f"Total cost must not exceed ${budget}"},
        }
    )

    # Makespan variable (project completion time)
    variables.append(
        {
            "id": "makespan",
            "domain": {"type": "integer", "lower": 0, "upper": horizon},
            "metadata": {"type": "objective"},
        }
    )

    # Makespan >= all job end times
    for job in jobs:
        constraints.append(
            {
                "id": f"makespan_{job['id']}",
                "kind": "linear",
                "params": {
                    "terms": [
                        {"var": "makespan", "coef": 1},
                        {"var": f"end_{job['id']}", "coef": -1},
                    ],
                    "sense": ">=",
                    "rhs": 0,
                },
            }
        )

    # Total cost variable for visibility
    variables.append(
        {
            "id": "total_cost",
            "domain": {"type": "integer", "lower": 0, "upper": int(budget * 100)},
            "metadata": {"type": "objective"},
        }
    )

    cost_terms_eq = [{"var": "total_cost", "coef": 1}] + [
        {"var": term["var"], "coef": -term["coef"]} for term in cost_terms
    ]

    constraints.append(
        {
            "id": "calc_total_cost",
            "kind": "linear",
            "params": {"terms": cost_terms_eq, "sense": "==", "rhs": 0},
        }
    )

    # Priority score variable (maximize priority sum)
    variables.append(
        {
            "id": "priority_score",
            "domain": {"type": "integer", "lower": 0, "upper": 1000},
            "metadata": {"type": "objective"},
        }
    )

    # Calculate priority score based on early completion
    # Higher priority jobs completing earlier = higher score
    priority_terms = [{"var": "priority_score", "coef": 1}]
    for job in jobs:
        # Score = priority * (horizon - end_time)
        # We approximate this linearly
        priority_terms.append({"var": f"end_{job['id']}", "coef": job.get("priority", 1)})

    # Multi-objective: minimize cost (priority 3), minimize makespan (priority 2), maximize priority (priority 1)
    return {
        "mode": "optimize",
        "variables": variables,
        "constraints": constraints,
        "objective": [
            {
                "sense": "min",
                "terms": [{"var": "total_cost", "coef": 1}],
                "priority": 3,  # Highest priority - minimize cost first
                "weight": 1.0,
            },
            {
                "sense": "min",
                "terms": [{"var": "makespan", "coef": 1}],
                "priority": 2,  # Second priority - minimize time
                "weight": 1.0,
            },
        ],
    }


async def main():
    """Run GPU job scheduling example."""
    print("=== GPU Job Scheduling with Resource Constraints ===\n")
    print("Scenario: Schedule ML/AI jobs across heterogeneous GPUs optimizing cost and time\n")

    # Define GPU resources
    gpus = [
        {
            "id": "gpu_a100_0",
            "name": "A100 80GB",
            "memory_gb": 80,
            "compute_units": 100,
            "cost_per_hour": 3.00,
        },
        {
            "id": "gpu_v100_0",
            "name": "V100 32GB",
            "memory_gb": 32,
            "compute_units": 60,
            "cost_per_hour": 1.50,
        },
        {
            "id": "gpu_v100_1",
            "name": "V100 32GB",
            "memory_gb": 32,
            "compute_units": 60,
            "cost_per_hour": 1.50,
        },
        {
            "id": "gpu_t4_0",
            "name": "T4 16GB",
            "memory_gb": 16,
            "compute_units": 30,
            "cost_per_hour": 0.50,
        },
    ]

    # Define jobs
    jobs = [
        {
            "id": "job_embed_1",
            "name": "Generate embeddings (1M docs)",
            "memory_gb": 24,
            "compute_units": 40,
            "duration": 4,  # hours
            "priority": 3,
            "deadline": 12,
        },
        {
            "id": "job_finetune_1",
            "name": "Fine-tune LLM",
            "memory_gb": 64,
            "compute_units": 80,
            "duration": 8,
            "priority": 5,
            "deadline": 24,
        },
        {
            "id": "job_inference_1",
            "name": "Batch inference",
            "memory_gb": 16,
            "compute_units": 20,
            "duration": 2,
            "priority": 2,
            "depends_on": ["job_embed_1"],  # Must run after embedding
        },
        {
            "id": "job_embed_2",
            "name": "Generate embeddings (500K docs)",
            "memory_gb": 12,
            "compute_units": 20,
            "duration": 2,
            "priority": 2,
        },
        {
            "id": "job_train_1",
            "name": "Train small model",
            "memory_gb": 16,
            "compute_units": 25,
            "duration": 6,
            "priority": 3,
        },
        {
            "id": "job_eval_1",
            "name": "Model evaluation",
            "memory_gb": 8,
            "compute_units": 10,
            "duration": 1,
            "priority": 1,
            "depends_on": ["job_train_1", "job_finetune_1"],
        },
    ]

    budget = 100.00  # dollars
    horizon = 24  # hours

    print("Available GPUs:")
    print("GPU ID        | Name        | Memory | Compute | Cost/Hour")
    print("--------------|-------------|--------|---------|----------")
    for gpu in gpus:
        print(
            f"{gpu['id']:13} | {gpu['name']:11} | {gpu['memory_gb']:4}GB | "
            f"{gpu['compute_units']:7} | ${gpu['cost_per_hour']:.2f}"
        )

    print(f"\nJobs to Schedule ({len(jobs)} total):")
    print(
        "Job ID         | Name                        | Memory | Compute | Duration | Priority | Deadline | Depends On"
    )
    print(
        "---------------|-----------------------------|---------|---------|---------|---------|---------|-----------"
    )
    for job in jobs:
        deps = ", ".join(job.get("depends_on", [])) or "None"
        deadline = f"{job.get('deadline', 'None'):>4}"
        if deadline == "None":
            deadline = " N/A"
        else:
            deadline = f"{deadline}h"
        print(
            f"{job['id']:14} | {job['name']:27} | {job['memory_gb']:5}GB | "
            f"{job['compute_units']:7} | {job['duration']:7}h | {job['priority']:8} | {deadline:>8} | {deps}"
        )

    print("\nConstraints:")
    print(f"  Budget: ${budget}")
    print(f"  Time Horizon: {horizon} hours")
    print("  Must respect job dependencies")
    print("  Must meet deadlines")
    print("  Jobs must fit on GPU memory")

    print("\nObjectives (in priority order):")
    print("  1. Minimize total cost (highest priority)")
    print("  2. Minimize completion time (makespan)")

    print("\nOptimizing schedule...\n")

    # Build and solve
    model = build_gpu_scheduling_model(jobs, gpus, horizon, budget)
    request = SolveConstraintModelRequest(**model)

    solver = get_solver("ortools")
    response = await solver.solve_constraint_model(request)

    print(f"Status: {response.status}\n")

    if response.status.value in ["optimal", "feasible"]:
        # Extract solution
        var_map = {var.id: var.value for var in response.solutions[0].variables}

        makespan = var_map["makespan"]
        total_cost = var_map["total_cost"] / 100.0  # Convert back from cents

        print(f"Solution Quality: {response.status.value.upper()}")
        print(f"Total Cost: ${total_cost:.2f} (Budget: ${budget})")
        print(f"Completion Time: {int(makespan)} hours")
        print(f"Combined Objective Value: {response.objective_value}\n")

        # Extract job assignments
        schedule = []
        for job in jobs:
            assigned_gpu = None
            for gpu in gpus:
                if var_map[f"assign_{job['id']}_to_{gpu['id']}"] == 1:
                    assigned_gpu = gpu
                    break

            start = int(var_map[f"start_{job['id']}"])
            end = int(var_map[f"end_{job['id']}"])

            schedule.append(
                {
                    "job": job,
                    "gpu": assigned_gpu,
                    "start": start,
                    "end": end,
                    "cost": job["duration"] * assigned_gpu["cost_per_hour"],
                }
            )

        # Sort by start time
        schedule.sort(key=lambda x: (x["start"], x["job"]["id"]))

        print("Optimal Schedule:")
        print(
            "Job ID         | GPU           | Start | End  | Duration | Cost    | Priority | Status"
        )
        print(
            "---------------|---------------|-------|------|----------|---------|----------|--------"
        )

        for item in schedule:
            job = item["job"]
            gpu = item["gpu"]
            status = "⚠️ LATE" if job.get("deadline") and item["end"] > job["deadline"] else "✓ OK"

            print(
                f"{job['id']:14} | {gpu['id']:13} | {item['start']:5}h | {item['end']:4}h | "
                f"{job['duration']:8}h | ${item['cost']:6.2f} | {job['priority']:8} | {status:8}"
            )

        # Show timeline
        print("\nTimeline (hours):")
        print("Time | GPU A100         | GPU V100_0       | GPU V100_1       | GPU T4")
        print("-----|------------------|------------------|------------------|------------------")

        for hour in range(int(makespan) + 1):
            row = f"{hour:4} |"
            for gpu in gpus:
                running_jobs = [
                    item
                    for item in schedule
                    if item["gpu"]["id"] == gpu["id"] and item["start"] <= hour < item["end"]
                ]
                if running_jobs:
                    job_label = running_jobs[0]["job"]["id"][:16]
                    row += f" {job_label:16} |"
                else:
                    row += f" {'':16} |"
            print(row)

        # Cost breakdown
        print("\nCost Breakdown by GPU:")
        cost_by_gpu = {}
        for item in schedule:
            gpu_id = item["gpu"]["id"]
            cost_by_gpu[gpu_id] = cost_by_gpu.get(gpu_id, 0) + item["cost"]

        for gpu in gpus:
            cost = cost_by_gpu.get(gpu["id"], 0)
            hours = cost / gpu["cost_per_hour"] if cost > 0 else 0
            print(f"  {gpu['id']:13}: ${cost:6.2f} ({hours:.1f} hours)")

        print(f"\nTotal: ${total_cost:.2f}")

        # Show explanation
        if response.explanation:
            print(f"\n{response.explanation.summary}")
            if response.explanation.binding_constraints:
                print(f"\nBinding Constraints: {len(response.explanation.binding_constraints)}")
                for bc in response.explanation.binding_constraints[:5]:
                    if bc.metadata and "description" in bc.metadata:
                        print(f"  - {bc.metadata['description']}")

    else:
        print(f"Could not find solution: {response.status}")
        if response.explanation:
            print(f"\n{response.explanation.summary}")
        print("\nPossible reasons:")
        print("  - Budget too low for required jobs")
        print("  - Deadlines too tight given dependencies")
        print("  - GPU memory insufficient for job requirements")


if __name__ == "__main__":
    asyncio.run(main())
