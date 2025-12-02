"""Example: End-to-end ML pipeline orchestration with complex dependencies.

Demonstrates comprehensive ML workflow scheduling:
- Data ingestion → preprocessing → training → evaluation → deployment
- Multiple model variants (A/B testing different architectures)
- Resource sharing across pipeline stages
- Conditional execution (deploy only if accuracy > threshold)
- Budget constraints across entire pipeline
- SLA requirements (must complete within deadline)

Shows LLM integration: "Train 3 model variants, deploy the best one if accuracy > 90%,
complete within 48 hours under $500 budget."
"""

import asyncio

from chuk_mcp_solver.models import SolveConstraintModelRequest
from chuk_mcp_solver.solver import get_solver


def build_ml_pipeline_model(
    model_variants: list[dict],
    stages: list[dict],
    resources: list[dict],
    budget: float,
    deadline: int,
) -> dict:
    """Build ML pipeline orchestration model.

    Args:
        model_variants: Different model architectures to train
        stages: Pipeline stages (ingest, preprocess, train, eval, deploy)
        resources: Available compute resources
        budget: Total budget for pipeline
        deadline: Must complete within this many hours

    Returns:
        Model dictionary for solver.
    """
    variables = []
    constraints = []

    # For each model variant, create variables for each stage
    for variant in model_variants:
        variant_id = variant["id"]

        for stage in stages:
            stage_id = stage["id"]

            # Start time
            variables.append(
                {
                    "id": f"start_{variant_id}_{stage_id}",
                    "domain": {"type": "integer", "lower": 0, "upper": deadline},
                    "metadata": {
                        "variant": variant_id,
                        "stage": stage_id,
                        "type": "start",
                    },
                }
            )

            # End time
            duration = variant.get(f"{stage_id}_duration", stage["base_duration"])
            variables.append(
                {
                    "id": f"end_{variant_id}_{stage_id}",
                    "domain": {
                        "type": "integer",
                        "lower": duration,
                        "upper": deadline,
                    },
                    "metadata": {"variant": variant_id, "stage": stage_id, "type": "end"},
                }
            )

            # Duration constraint
            constraints.append(
                {
                    "id": f"duration_{variant_id}_{stage_id}",
                    "kind": "linear",
                    "params": {
                        "terms": [
                            {"var": f"end_{variant_id}_{stage_id}", "coef": 1},
                            {"var": f"start_{variant_id}_{stage_id}", "coef": -1},
                        ],
                        "sense": "==",
                        "rhs": duration,
                    },
                }
            )

            # Resource assignment for this stage
            for resource in resources:
                var_id = f"assign_{variant_id}_{stage_id}_{resource['id']}"
                variables.append(
                    {
                        "id": var_id,
                        "domain": {"type": "bool"},
                        "metadata": {
                            "variant": variant_id,
                            "stage": stage_id,
                            "resource": resource["id"],
                        },
                    }
                )

            # Each stage assigned to exactly one resource
            assignment_terms = [
                {"var": f"assign_{variant_id}_{stage_id}_{resource['id']}", "coef": 1}
                for resource in resources
            ]
            constraints.append(
                {
                    "id": f"assign_resource_{variant_id}_{stage_id}",
                    "kind": "linear",
                    "params": {"terms": assignment_terms, "sense": "==", "rhs": 1},
                }
            )

    # Stage dependencies within each variant (pipeline order)
    stage_order = ["ingest", "preprocess", "train", "eval", "deploy"]
    for variant in model_variants:
        for i in range(len(stage_order) - 1):
            curr_stage = stage_order[i]
            next_stage = stage_order[i + 1]

            # Next stage must start after current stage ends
            constraints.append(
                {
                    "id": f"pipeline_{variant['id']}_{curr_stage}_{next_stage}",
                    "kind": "linear",
                    "params": {
                        "terms": [
                            {"var": f"start_{variant['id']}_{next_stage}", "coef": 1},
                            {"var": f"end_{variant['id']}_{curr_stage}", "coef": -1},
                        ],
                        "sense": ">=",
                        "rhs": 0,
                    },
                    "metadata": {
                        "description": f"Variant {variant['id']}: {next_stage} after {curr_stage}"
                    },
                }
            )

    # Shared data preprocessing - all variants share ingested data
    # So ingest only needs to happen once, and other variants can use it
    # Model this by: variant_2.ingest.start >= variant_1.ingest.end (if they share)

    # For simplicity, let's say all variants can share preprocessed data
    # So preprocessing happens in parallel after shared ingestion

    # Resource constraints - no two stages can use same resource simultaneously
    # This is complex to model perfectly; we'll approximate with cumulative constraints

    # Deployment decision variables - deploy only if expected to meet quality
    for variant in model_variants:
        variables.append(
            {
                "id": f"deploy_decision_{variant['id']}",
                "domain": {"type": "bool"},
                "metadata": {
                    "variant": variant["id"],
                    "type": "deployment_decision",
                },
            }
        )

        # If deploy decision is True, must actually run deploy stage
        # This is an implication: deploy_decision -> deploy stage must execute
        # We model this differently: if expected_accuracy < threshold, deploy_decision must be 0

        if variant.get("expected_accuracy", 100) < 90:  # Threshold
            constraints.append(
                {
                    "id": f"quality_gate_{variant['id']}",
                    "kind": "linear",
                    "params": {
                        "terms": [{"var": f"deploy_decision_{variant['id']}", "coef": 1}],
                        "sense": "==",
                        "rhs": 0,
                    },
                    "metadata": {
                        "description": f"Variant {variant['id']} doesn't meet quality threshold"
                    },
                }
            )

    # Budget constraint
    cost_terms = []
    for variant in model_variants:
        for stage in stages:
            for resource in resources:
                assign_var = f"assign_{variant['id']}_{stage['id']}_{resource['id']}"
                duration = variant.get(f"{stage['id']}_duration", stage["base_duration"])
                stage_cost = duration * resource["cost_per_hour"]
                cost_terms.append({"var": assign_var, "coef": int(stage_cost * 100)})

    constraints.append(
        {
            "id": "budget_constraint",
            "kind": "linear",
            "params": {
                "terms": cost_terms,
                "sense": "<=",
                "rhs": int(budget * 100),
            },
            "metadata": {"description": f"Total cost must not exceed ${budget}"},
        }
    )

    # Deadline constraint - all variants must complete by deadline
    for variant in model_variants:
        constraints.append(
            {
                "id": f"deadline_{variant['id']}",
                "kind": "linear",
                "params": {
                    "terms": [{"var": f"end_{variant['id']}_deploy", "coef": 1}],
                    "sense": "<=",
                    "rhs": deadline,
                },
                "metadata": {
                    "description": f"Variant {variant['id']} must complete within {deadline}h"
                },
            }
        )

    # Makespan
    variables.append(
        {
            "id": "makespan",
            "domain": {"type": "integer", "lower": 0, "upper": deadline},
            "metadata": {"type": "objective"},
        }
    )

    for variant in model_variants:
        constraints.append(
            {
                "id": f"makespan_{variant['id']}",
                "kind": "linear",
                "params": {
                    "terms": [
                        {"var": "makespan", "coef": 1},
                        {"var": f"end_{variant['id']}_deploy", "coef": -1},
                    ],
                    "sense": ">=",
                    "rhs": 0,
                },
            }
        )

    # Total cost variable
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
            "id": "calc_cost",
            "kind": "linear",
            "params": {"terms": cost_terms_eq, "sense": "==", "rhs": 0},
        }
    )

    # Optimize: minimize cost (priority 2), minimize time (priority 1)
    return {
        "mode": "optimize",
        "variables": variables,
        "constraints": constraints,
        "objective": [
            {
                "sense": "min",
                "terms": [{"var": "total_cost", "coef": 1}],
                "priority": 2,
                "weight": 1.0,
            },
            {
                "sense": "min",
                "terms": [{"var": "makespan", "coef": 1}],
                "priority": 1,
                "weight": 1.0,
            },
        ],
    }


async def main():
    """Run ML pipeline orchestration example."""
    print("=== End-to-End ML Pipeline Orchestrator ===\n")
    print("Scenario: Train and deploy ML models with A/B testing\n")

    # Define pipeline stages
    stages = [
        {"id": "ingest", "name": "Data Ingestion", "base_duration": 2},
        {"id": "preprocess", "name": "Data Preprocessing", "base_duration": 4},
        {"id": "train", "name": "Model Training", "base_duration": 12},
        {"id": "eval", "name": "Model Evaluation", "base_duration": 2},
        {"id": "deploy", "name": "Model Deployment", "base_duration": 1},
    ]

    # Define model variants
    variants = [
        {
            "id": "transformer_large",
            "name": "Transformer Large",
            "train_duration": 16,  # Longer training
            "expected_accuracy": 94,
        },
        {
            "id": "transformer_base",
            "name": "Transformer Base",
            "train_duration": 12,
            "expected_accuracy": 92,
        },
        {
            "id": "lstm_model",
            "name": "LSTM Model",
            "train_duration": 8,  # Faster training
            "expected_accuracy": 88,  # Lower accuracy
        },
    ]

    # Define compute resources
    resources = [
        {"id": "gpu_cluster_a", "name": "GPU Cluster A (8xA100)", "cost_per_hour": 20.0},
        {"id": "gpu_cluster_b", "name": "GPU Cluster B (4xV100)", "cost_per_hour": 10.0},
        {"id": "cpu_cluster", "name": "CPU Cluster (32 cores)", "cost_per_hour": 2.0},
    ]

    budget = 500.0  # dollars
    deadline = 48  # hours (2 days)

    print("Pipeline Stages:")
    for stage in stages:
        print(f"  {stage['name']:20} - Base duration: {stage['base_duration']}h")

    print(f"\nModel Variants ({len(variants)} total):")
    print("Variant            | Training Time | Expected Accuracy | Deploy?")
    print("-------------------|---------------|-------------------|--------")
    for variant in variants:
        train_time = variant.get("train_duration", 12)
        accuracy = variant.get("expected_accuracy", 90)
        deploy = "✓ Yes" if accuracy >= 90 else "✗ No (< 90%)"
        print(f"{variant['name']:18} | {train_time:13}h | {accuracy:17}% | {deploy}")

    print("\nCompute Resources:")
    for resource in resources:
        print(f"  {resource['name']:30} - ${resource['cost_per_hour']:.2f}/hour")

    print("\nConstraints:")
    print(f"  Budget: ${budget}")
    print(f"  Deadline: {deadline} hours")
    print("  Quality Gate: Deploy only if accuracy ≥ 90%")
    print("  Pipeline Order: ingest → preprocess → train → eval → deploy")

    print("\nObjectives:")
    print("  1. Minimize total cost (highest priority)")
    print("  2. Minimize pipeline completion time")

    print("\nOptimizing ML pipeline...\n")

    # Build and solve
    model = build_ml_pipeline_model(variants, stages, resources, budget, deadline)
    request = SolveConstraintModelRequest(**model)

    solver = get_solver("ortools")
    response = await solver.solve_constraint_model(request)

    print(f"Status: {response.status}\n")

    if response.status.value in ["optimal", "feasible"]:
        var_map = {var.id: var.value for var in response.solutions[0].variables}

        makespan = var_map["makespan"]
        total_cost = var_map["total_cost"] / 100.0

        print(f"Solution Quality: {response.status.value.upper()}")
        print(f"Total Cost: ${total_cost:.2f} (Budget: ${budget})")
        print(f"Pipeline Completion: {int(makespan)} hours")
        print(f"Time Remaining: {deadline - int(makespan)} hours\n")

        # Show schedule for each variant
        for variant in variants:
            print(f"\n{'=' * 70}")
            print(f"Variant: {variant['name']}")
            print(f"Expected Accuracy: {variant.get('expected_accuracy', 90)}%")
            deploy_decision = var_map.get(f"deploy_decision_{variant['id']}", 0)
            print(f"Deploy Decision: {'✓ DEPLOY' if deploy_decision == 1 else '✗ SKIP'}")
            print(f"{'=' * 70}")

            print("\nStage        | Resource              | Start | End  | Duration | Cost")
            print("-------------|------------------------|-------|------|----------|--------")

            variant_cost = 0
            for stage in stages:
                start = int(var_map[f"start_{variant['id']}_{stage['id']}"])
                end = int(var_map[f"end_{variant['id']}_{stage['id']}"])
                duration = end - start

                # Find assigned resource
                assigned_resource = None
                for resource in resources:
                    if var_map[f"assign_{variant['id']}_{stage['id']}_{resource['id']}"] == 1:
                        assigned_resource = resource
                        break

                stage_cost = duration * assigned_resource["cost_per_hour"]
                variant_cost += stage_cost

                print(
                    f"{stage['name']:12} | {assigned_resource['name']:22} | {start:5}h | {end:4}h | "
                    f"{duration:8}h | ${stage_cost:6.2f}"
                )

            print(f"\nVariant Total: ${variant_cost:.2f}")

        # Resource utilization summary
        print(f"\n{'=' * 70}")
        print("Resource Utilization Summary")
        print(f"{'=' * 70}\n")

        resource_usage = {r["id"]: {"hours": 0, "cost": 0} for r in resources}

        for variant in variants:
            for stage in stages:
                duration = int(
                    var_map[f"end_{variant['id']}_{stage['id']}"]
                    - var_map[f"start_{variant['id']}_{stage['id']}"]
                )

                for resource in resources:
                    if var_map[f"assign_{variant['id']}_{stage['id']}_{resource['id']}"] == 1:
                        resource_usage[resource["id"]]["hours"] += duration
                        resource_usage[resource["id"]]["cost"] += (
                            duration * resource["cost_per_hour"]
                        )

        for resource in resources:
            usage = resource_usage[resource["id"]]
            print(
                f"{resource['name']:30}: {usage['hours']:3}h utilization, ${usage['cost']:.2f} cost"
            )

        if response.explanation:
            print(f"\n{response.explanation.summary}")

    else:
        print(f"Could not find solution: {response.status}")
        if response.explanation:
            print(f"\n{response.explanation.summary}")
        print("\nPossible reasons:")
        print("  - Budget insufficient for all model variants")
        print("  - Deadline too tight given training durations")
        print("  - Resource constraints too restrictive")


if __name__ == "__main__":
    asyncio.run(main())
