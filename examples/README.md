# CHUK MCP Solver Examples

This directory contains example scripts demonstrating various use cases for the constraint solver.

## Running Examples

Each example is a standalone Python script that can be run directly:

```bash
# Install the package first
uv pip install -e .

# Run an example
python examples/sudoku_solver.py
python examples/project_scheduler.py
python examples/knapsack_optimizer.py
```

## Examples by Category

### 🤖 Phase 2 & Phase 3 Feature Demos (NEW)

These examples demonstrate the latest features designed for LLM/AI agent use:

#### Validation Demo (`validation_demo.py`) 🆕
Demonstrates Phase 2 validation features:
- Pre-solve model validation
- Actionable error messages for LLMs
- Smart typo detection ("Did you mean...?")
- Three-level severity (ERROR, WARNING, INFO)
- How validation helps LLMs self-correct

```bash
python examples/validation_demo.py
```

#### Caching Demo (`caching_demo.py`) 🆕
Demonstrates Phase 3 caching features:
- Automatic solution caching with problem hashing
- Cache hit performance improvements (10-100x speedup)
- LRU eviction and TTL management
- Cache statistics and hit rate tracking
- When to enable/disable caching

```bash
python examples/caching_demo.py
```

#### Advanced Search Demo (`advanced_search_demo.py`) 🆕
Demonstrates Phase 3 search features:
- Partial solutions (best-so-far on timeout)
- Search strategy hints (first-fail, random)
- Deterministic solving with random seeds
- Warm-start solution hints
- Anytime algorithms (progressive improvement)

```bash
python examples/advanced_search_demo.py
```

### 📚 Core Constraint Satisfaction & Optimization

#### 1. Sudoku Solver (`sudoku_solver.py`)
Demonstrates:
- Constraint satisfaction (no objective)
- `all_different` global constraints
- Integer variables with domains
- Solving logic puzzles

#### 2. Project Scheduler (`project_scheduler.py`)
Demonstrates:
- Optimization with objective
- Linear constraints for precedence and resources
- Minimizing makespan (project duration)
- Binding constraint analysis

#### 3. Knapsack Optimizer (`knapsack_optimizer.py`)
Demonstrates:
- Binary (boolean) variables
- Capacity constraints
- Value maximization
- Classic optimization problem

### 🔧 Advanced Constraint Types

#### 4. Resource Scheduler (`resource_scheduler.py`)
Demonstrates:
- Cumulative constraints (resource capacity)
- CPU/memory allocation scheduling
- Resource utilization timeline
- Makespan optimization

#### 5. Delivery Router (`delivery_router.py`)
Demonstrates:
- Circuit constraints (Hamiltonian path)
- Vehicle routing / TSP
- Distance matrix optimization
- Route visualization

#### 6. Inventory Manager (`inventory_manager.py`)
Demonstrates:
- Reservoir constraints (stock levels)
- Production and consumption scheduling
- Safety stock maintenance
- Timeline visualization

### 🎯 Multi-Objective & AI Orchestration

#### 7. Multi-Objective Planner (`multi_objective_planner.py`)
Demonstrates:
- Multi-objective optimization with priorities
- Cost vs latency trade-offs
- Cloud instance selection
- Priority-based lexicographic ordering

#### 8. Tool Selection (`tool_selector.py`)
Demonstrates:
- Implication constraints (conditional logic)
- AI model/tool selection
- Cost/latency trade-offs
- Practical MCP use case

### 🚀 Complex Real-World Examples

#### 9. GPU Job Scheduler (`gpu_job_scheduler.py`)
Demonstrates:
- Heterogeneous resource allocation
- Job dependencies and deadlines
- Budget constraints
- ML/AI workload optimization

#### 10. Embedding Pipeline Scheduler (`embedding_pipeline_scheduler.py`)
Demonstrates:
- Multi-stage pipeline orchestration
- Rate limit constraints
- Provider selection (OpenAI, Cohere, Voyage)
- Throughput optimization

#### 11. ML Pipeline Orchestrator (`ml_pipeline_orchestrator.py`)
Demonstrates:
- End-to-end ML pipeline
- Conditional deployment (quality gates)
- Multi-variant training
- A/B testing optimization

## Example Structure

Each example follows this pattern:

1. **Problem Setup**: Define the problem parameters
2. **Model Building**: Create variables, constraints, and objective
3. **Solving**: Call the solver
4. **Result Display**: Show the solution in a readable format

## Adapting Examples

These examples can be easily adapted for your own use cases:

- Modify problem parameters (sizes, bounds, costs)
- Add new constraints
- Change the objective function
- Experiment with different constraint types
