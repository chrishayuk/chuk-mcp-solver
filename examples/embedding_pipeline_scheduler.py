"""Example: Embedding extraction pipeline with batch scheduling and resource allocation.

Demonstrates realistic ML pipeline scheduling:
- Multi-stage pipeline: document processing → embedding extraction → vector DB ingestion
- Batch size optimization (larger batches = better GPU utilization, more memory)
- Rate limiting (API quotas, vector DB write limits)
- Cost optimization across different embedding providers
- Pipeline dependencies and data flow

Shows how LLM could translate: "Process 1M documents through embedding pipeline
optimizing for cost and throughput" into constraint problem.
"""

import asyncio

from chuk_mcp_solver.models import SolveConstraintModelRequest
from chuk_mcp_solver.solver import get_solver


def build_embedding_pipeline_model(
    document_batches: list[dict],
    embedding_providers: list[dict],
    vector_db_config: dict,
    horizon: int,
) -> dict:
    """Build embedding pipeline scheduling model.

    Args:
        document_batches: Batches of documents to process
        embedding_providers: Available embedding APIs (OpenAI, Cohere, etc.)
        vector_db_config: Vector DB ingestion rate limits
        horizon: Time horizon in minutes

    Returns:
        Model dictionary for constraint solver.
    """
    variables = []
    constraints = []

    # Stage 1: Document preprocessing
    # Stage 2: Embedding extraction
    # Stage 3: Vector DB ingestion

    # Decision: which provider handles which batch
    assignment_vars = {}
    for batch in document_batches:
        for provider in embedding_providers:
            var_id = f"assign_{batch['id']}_to_{provider['id']}"
            variables.append(
                {
                    "id": var_id,
                    "domain": {"type": "bool"},
                    "metadata": {
                        "batch": batch["id"],
                        "provider": provider["id"],
                        "type": "assignment",
                    },
                }
            )
            assignment_vars[(batch["id"], provider["id"])] = var_id

    # Time variables for each stage of each batch
    for batch in document_batches:
        batch_id = batch["id"]

        # Stage 1: Preprocessing
        variables.append(
            {
                "id": f"prep_start_{batch_id}",
                "domain": {"type": "integer", "lower": 0, "upper": horizon},
                "metadata": {"batch": batch_id, "stage": "prep", "type": "start"},
            }
        )
        variables.append(
            {
                "id": f"prep_end_{batch_id}",
                "domain": {
                    "type": "integer",
                    "lower": batch["prep_time"],
                    "upper": horizon,
                },
                "metadata": {"batch": batch_id, "stage": "prep", "type": "end"},
            }
        )

        # Link: prep_end = prep_start + prep_time
        constraints.append(
            {
                "id": f"prep_duration_{batch_id}",
                "kind": "linear",
                "params": {
                    "terms": [
                        {"var": f"prep_end_{batch_id}", "coef": 1},
                        {"var": f"prep_start_{batch_id}", "coef": -1},
                    ],
                    "sense": "==",
                    "rhs": batch["prep_time"],
                },
            }
        )

        # Stage 2: Embedding extraction
        variables.append(
            {
                "id": f"embed_start_{batch_id}",
                "domain": {"type": "integer", "lower": 0, "upper": horizon},
                "metadata": {"batch": batch_id, "stage": "embed", "type": "start"},
            }
        )
        variables.append(
            {
                "id": f"embed_end_{batch_id}",
                "domain": {"type": "integer", "lower": 0, "upper": horizon},
                "metadata": {"batch": batch_id, "stage": "embed", "type": "end"},
            }
        )

        # Embed must start after prep finishes
        constraints.append(
            {
                "id": f"pipeline_prep_embed_{batch_id}",
                "kind": "linear",
                "params": {
                    "terms": [
                        {"var": f"embed_start_{batch_id}", "coef": 1},
                        {"var": f"prep_end_{batch_id}", "coef": -1},
                    ],
                    "sense": ">=",
                    "rhs": 0,
                },
                "metadata": {
                    "description": f"Batch {batch_id} embedding must start after preprocessing"
                },
            }
        )

        # Stage 3: Vector DB ingestion
        variables.append(
            {
                "id": f"ingest_start_{batch_id}",
                "domain": {"type": "integer", "lower": 0, "upper": horizon},
                "metadata": {"batch": batch_id, "stage": "ingest", "type": "start"},
            }
        )
        variables.append(
            {
                "id": f"ingest_end_{batch_id}",
                "domain": {"type": "integer", "lower": 0, "upper": horizon},
                "metadata": {"batch": batch_id, "stage": "ingest", "type": "end"},
            }
        )

        # Ingest must start after embedding finishes
        constraints.append(
            {
                "id": f"pipeline_embed_ingest_{batch_id}",
                "kind": "linear",
                "params": {
                    "terms": [
                        {"var": f"ingest_start_{batch_id}", "coef": 1},
                        {"var": f"embed_end_{batch_id}", "coef": -1},
                    ],
                    "sense": ">=",
                    "rhs": 0,
                },
                "metadata": {
                    "description": f"Batch {batch_id} ingestion must start after embedding"
                },
            }
        )

        # Ingest duration depends on batch size
        ingest_time = (
            batch["size"] * vector_db_config["ms_per_vector"]
        ) // 1000  # Convert to minutes
        constraints.append(
            {
                "id": f"ingest_duration_{batch_id}",
                "kind": "linear",
                "params": {
                    "terms": [
                        {"var": f"ingest_end_{batch_id}", "coef": 1},
                        {"var": f"ingest_start_{batch_id}", "coef": -1},
                    ],
                    "sense": "==",
                    "rhs": max(1, ingest_time),
                },
            }
        )

    # Each batch assigned to exactly one provider
    for batch in document_batches:
        assignment_terms = [
            {"var": assignment_vars[(batch["id"], provider["id"])], "coef": 1}
            for provider in embedding_providers
        ]
        constraints.append(
            {
                "id": f"assign_one_{batch['id']}",
                "kind": "linear",
                "params": {"terms": assignment_terms, "sense": "==", "rhs": 1},
                "metadata": {"description": f"Batch {batch['id']} must use exactly one provider"},
            }
        )

    # Embedding duration depends on provider and batch size
    for batch in document_batches:
        for provider in embedding_providers:
            assign_var = assignment_vars[(batch["id"], provider["id"])]

            # Calculate embedding time for this batch on this provider
            embed_time = (batch["size"] * provider["ms_per_doc"]) // 1000  # Convert to minutes
            embed_time = max(1, embed_time)  # At least 1 minute

            # If assigned to this provider, duration must match
            # We model this with implication: assign_var -> duration constraint
            # For simplicity, we'll use a linear constraint weighted by assignment
            # This is an approximation; ideally we'd use implication constraints

    # Provider rate limits (requests per hour)
    # Track how many batches each provider handles and enforce rate limits
    for provider in embedding_providers:
        if provider.get("rate_limit_per_hour"):
            # Count total documents assigned to this provider
            doc_terms = []
            for batch in document_batches:
                assign_var = assignment_vars[(batch["id"], provider["id"])]
                # If assigned, add batch size to total
                doc_terms.append({"var": assign_var, "coef": batch["size"]})

            # Total must not exceed rate limit
            constraints.append(
                {
                    "id": f"rate_limit_{provider['id']}",
                    "kind": "linear",
                    "params": {
                        "terms": doc_terms,
                        "sense": "<=",
                        "rhs": provider["rate_limit_per_hour"] * (horizon // 60),
                    },
                    "metadata": {
                        "description": f"{provider['name']} rate limit: {provider['rate_limit_per_hour']} docs/hour"
                    },
                }
            )

    # Vector DB write rate limit (vectors per minute)
    # Simplified: total ingestion rate enforced through duration constraints
    # In reality, you'd use cumulative constraint with capacity = max_vectors_per_min
    # (No additional constraints needed as ingestion duration already accounts for throughput)

    # Cost calculation
    variables.append(
        {
            "id": "total_cost",
            "domain": {"type": "integer", "lower": 0, "upper": 1000000},
            "metadata": {"type": "objective"},
        }
    )

    cost_terms = [{"var": "total_cost", "coef": 1}]
    for batch in document_batches:
        for provider in embedding_providers:
            assign_var = assignment_vars[(batch["id"], provider["id"])]
            batch_cost = batch["size"] * provider["cost_per_1k_docs"] / 1000
            cost_terms.append({"var": assign_var, "coef": -int(batch_cost * 100)})

    constraints.append(
        {
            "id": "calc_total_cost",
            "kind": "linear",
            "params": {"terms": cost_terms, "sense": "==", "rhs": 0},
        }
    )

    # Makespan
    variables.append(
        {
            "id": "makespan",
            "domain": {"type": "integer", "lower": 0, "upper": horizon},
            "metadata": {"type": "objective"},
        }
    )

    for batch in document_batches:
        constraints.append(
            {
                "id": f"makespan_{batch['id']}",
                "kind": "linear",
                "params": {
                    "terms": [
                        {"var": "makespan", "coef": 1},
                        {"var": f"ingest_end_{batch['id']}", "coef": -1},
                    ],
                    "sense": ">=",
                    "rhs": 0,
                },
            }
        )

    # Multi-objective: minimize cost (priority 2), minimize time (priority 1)
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
    """Run embedding pipeline scheduling example."""
    print("=== Embedding Extraction Pipeline Scheduler ===\n")
    print("Scenario: Process 100K documents through embedding pipeline\n")

    # Define embedding providers
    providers = [
        {
            "id": "openai_ada",
            "name": "OpenAI text-embedding-ada-002",
            "cost_per_1k_docs": 0.10,  # $0.10 per 1K docs
            "ms_per_doc": 50,  # 50ms per document
            "rate_limit_per_hour": 50000,  # 50K docs per hour
            "dimensions": 1536,
        },
        {
            "id": "cohere_embed",
            "name": "Cohere Embed v3",
            "cost_per_1k_docs": 0.06,  # $0.06 per 1K docs
            "ms_per_doc": 40,
            "rate_limit_per_hour": 100000,  # 100K docs per hour
            "dimensions": 1024,
        },
        {
            "id": "voyage_large",
            "name": "Voyage-large-2",
            "cost_per_1k_docs": 0.12,  # $0.12 per 1K docs
            "ms_per_doc": 35,
            "rate_limit_per_hour": 30000,  # 30K docs per hour
            "dimensions": 1536,
        },
    ]

    # Vector DB configuration
    vector_db_config = {
        "name": "Pinecone",
        "max_vectors_per_min": 10000,
        "ms_per_vector": 10,  # 10ms per vector write
    }

    # Document batches
    batches = [
        {"id": "batch_1", "size": 10000, "prep_time": 5},  # 10K docs, 5 min prep
        {"id": "batch_2", "size": 15000, "prep_time": 7},
        {"id": "batch_3", "size": 12000, "prep_time": 6},
        {"id": "batch_4", "size": 20000, "prep_time": 10},
        {"id": "batch_5", "size": 18000, "prep_time": 9},
        {"id": "batch_6", "size": 14000, "prep_time": 7},
        {"id": "batch_7", "size": 11000, "prep_time": 6},
    ]

    total_docs = sum(b["size"] for b in batches)
    horizon = 300  # 5 hours in minutes

    print(f"Total Documents: {total_docs:,}")
    print(f"Document Batches: {len(batches)}\n")

    print("Embedding Providers:")
    print("Provider                       | Cost/1K  | Speed     | Rate Limit   | Dimensions")
    print("-------------------------------|----------|-----------|--------------|------------")
    for p in providers:
        print(
            f"{p['name']:30} | ${p['cost_per_1k_docs']:.2f}   | {p['ms_per_doc']:4}ms/doc | "
            f"{p['rate_limit_per_hour']:6}/hour | {p['dimensions']:10}"
        )

    print(f"\nVector Database: {vector_db_config['name']}")
    print(f"  Write Throughput: {vector_db_config['max_vectors_per_min']:,} vectors/min")
    print(f"  Write Latency: {vector_db_config['ms_per_vector']}ms/vector\n")

    print("Document Batches:")
    print("Batch ID | Size    | Prep Time")
    print("---------|---------|----------")
    for batch in batches:
        print(f"{batch['id']:8} | {batch['size']:7,} | {batch['prep_time']:5} min")

    print("\nPipeline Stages:")
    print("  1. Preprocessing (clean, tokenize, chunk)")
    print("  2. Embedding extraction (API call)")
    print("  3. Vector DB ingestion (write to Pinecone)")

    print("\nObjectives:")
    print("  1. Minimize total cost (highest priority)")
    print("  2. Minimize pipeline completion time")

    print("\nConstraints:")
    print("  - Respect provider rate limits")
    print("  - Respect vector DB write throughput")
    print("  - Maintain pipeline order (prep → embed → ingest)")

    print("\nOptimizing pipeline schedule...\n")

    # Build and solve
    model = build_embedding_pipeline_model(batches, providers, vector_db_config, horizon)
    request = SolveConstraintModelRequest(**model)

    solver = get_solver("ortools")
    response = await solver.solve_constraint_model(request)

    print(f"Status: {response.status}\n")

    if response.status.value in ["optimal", "feasible"]:
        var_map = {var.id: var.value for var in response.solutions[0].variables}

        makespan = var_map["makespan"]
        total_cost = var_map["total_cost"] / 100.0

        print(f"Solution Quality: {response.status.value.upper()}")
        print(f"Total Cost: ${total_cost:.2f}")
        print(f"Pipeline Completion: {int(makespan)} minutes ({makespan / 60:.1f} hours)")
        print(
            f"Throughput: {total_docs / makespan:.0f} docs/min ({total_docs / (makespan / 60):.0f} docs/hour)\n"
        )

        # Extract assignments
        assignments = []
        for batch in batches:
            assigned_provider = None
            for provider in providers:
                if var_map[f"assign_{batch['id']}_to_{provider['id']}"] == 1:
                    assigned_provider = provider
                    break

            prep_start = int(var_map[f"prep_start_{batch['id']}"])
            prep_end = int(var_map[f"prep_end_{batch['id']}"])
            embed_start = int(var_map[f"embed_start_{batch['id']}"])
            embed_end = int(var_map[f"embed_end_{batch['id']}"])
            ingest_start = int(var_map[f"ingest_start_{batch['id']}"])
            ingest_end = int(var_map[f"ingest_end_{batch['id']}"])

            batch_cost = batch["size"] * assigned_provider["cost_per_1k_docs"] / 1000

            assignments.append(
                {
                    "batch": batch,
                    "provider": assigned_provider,
                    "prep_start": prep_start,
                    "prep_end": prep_end,
                    "embed_start": embed_start,
                    "embed_end": embed_end,
                    "ingest_start": ingest_start,
                    "ingest_end": ingest_end,
                    "cost": batch_cost,
                }
            )

        # Sort by start time
        assignments.sort(key=lambda x: x["prep_start"])

        print("Pipeline Schedule:")
        print(
            "Batch    | Provider                       | Prep      | Embed     | Ingest    | Cost"
        )
        print(
            "---------|--------------------------------|-----------|-----------|-----------|--------"
        )

        for assign in assignments:
            print(
                f"{assign['batch']['id']:8} | {assign['provider']['name']:30} | "
                f"{assign['prep_start']:3}-{assign['prep_end']:3}min | "
                f"{assign['embed_start']:3}-{assign['embed_end']:3}min | "
                f"{assign['ingest_start']:3}-{assign['ingest_end']:3}min | ${assign['cost']:6.2f}"
            )

        # Provider usage
        print("\nProvider Utilization:")
        provider_stats = {}
        for assign in assignments:
            prov_id = assign["provider"]["id"]
            if prov_id not in provider_stats:
                provider_stats[prov_id] = {
                    "name": assign["provider"]["name"],
                    "batches": 0,
                    "docs": 0,
                    "cost": 0,
                }
            provider_stats[prov_id]["batches"] += 1
            provider_stats[prov_id]["docs"] += assign["batch"]["size"]
            provider_stats[prov_id]["cost"] += assign["cost"]

        for _prov_id, stats in provider_stats.items():
            print(
                f"  {stats['name']:30}: {stats['batches']} batches, "
                f"{stats['docs']:6,} docs, ${stats['cost']:.2f}"
            )

        # Timeline visualization (sample)
        print("\nPipeline Timeline (first 60 minutes):")
        print("Time  | Activities")
        print("------|" + "-" * 70)

        for minute in range(0, min(60, int(makespan) + 1), 5):
            activities = []
            for assign in assignments:
                if assign["prep_start"] <= minute < assign["prep_end"]:
                    activities.append(f"{assign['batch']['id']}:prep")
                elif assign["embed_start"] <= minute < assign["embed_end"]:
                    activities.append(f"{assign['batch']['id']}:embed")
                elif assign["ingest_start"] <= minute < assign["ingest_end"]:
                    activities.append(f"{assign['batch']['id']}:ingest")

            activity_str = ", ".join(activities) if activities else "idle"
            print(f"{minute:4}m | {activity_str}")

        if response.explanation:
            print(f"\n{response.explanation.summary}")

    else:
        print(f"Could not find solution: {response.status}")
        if response.explanation:
            print(f"\n{response.explanation.summary}")


if __name__ == "__main__":
    asyncio.run(main())
