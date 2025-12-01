# CHUK MCP Solver

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-compatible-brightgreen.svg)](https://modelcontextprotocol.io)
[![Tests](https://img.shields.io/badge/tests-105%20passed-success.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-91%25-brightgreen.svg)](tests/)

🔧 **General-purpose constraint and optimization solver as an MCP server**

A powerful Model Context Protocol (MCP) server that provides constraint satisfaction and optimization capabilities to LLMs and AI agents. Built on Google OR-Tools CP-SAT solver, it enables sophisticated decision-making for scheduling, resource allocation, puzzles, and more.

## Features

✨ **General Constraint Solver**
- Integer and boolean variables
- Linear constraints
- Global constraints (all_different, element, table)
- Implication constraints (conditional logic)
- Scheduling constraints (cumulative, no_overlap)
- Routing constraints (circuit)
- Inventory constraints (reservoir)
- Satisfaction and optimization modes

🎯 **Wide Range of Use Cases**
- Project scheduling and resource allocation
- Logic puzzles (Sudoku, etc.)
- Knapsack and packing problems
- Tool/model selection under constraints
- Configuration optimization
- Budget allocation

🚀 **Production Ready**
- Async/await native
- Type-safe with Pydantic models
- Comprehensive test coverage (>90%)
- Clean architecture with provider pattern
- Configurable via environment or YAML

📊 **Rich Solutions**
- Optimal and feasible solutions
- Multi-objective optimization (priority-based)
- Warm-start from previous solutions
- Parallel search workers
- Binding constraint analysis
- Human-readable explanations
- Metadata preservation

✅ **Production Quality**
- 105 comprehensive tests (all passing)
- 91% test coverage
- Type-safe with mypy
- Extensive error handling

## Example Use Cases

**🏗️ DevOps Team**: "Schedule 20 deployment tasks across 5 servers with CPU/memory limits while minimizing total deployment time"
- Uses cumulative constraints to manage resource capacity
- Optimizes makespan while respecting dependencies
- See: [`resource_scheduler.py`](examples/resource_scheduler.py)

**🚚 Logistics Company**: "Plan delivery routes for 10 trucks visiting 50 customers to minimize total distance"
- Uses circuit constraints for vehicle routing (TSP/VRP)
- Handles time windows and capacity constraints
- See: [`delivery_router.py`](examples/delivery_router.py)

**📦 Warehouse Manager**: "Schedule production runs and customer orders while maintaining safety stock of 500 units"
- Uses reservoir constraints to track inventory levels
- Prevents stockouts and overstock situations
- See: [`inventory_manager.py`](examples/inventory_manager.py)

**☁️ Cloud Architect**: "Select AWS instances to meet requirements while minimizing cost, then latency"
- Uses multi-objective optimization with priorities
- Balances competing objectives (cost vs performance)
- See: [`multi_objective_planner.py`](examples/multi_objective_planner.py)

**🤖 AI Platform**: "Route 100 user requests to GPT-4, GPT-3.5, or Claude to minimize cost under $50 budget"
- Uses implication constraints for conditional logic
- Selects optimal model for each task based on capabilities
- See: [`tool_selector.py`](examples/tool_selector.py)

**🎯 Project Manager**: "Schedule 10 tasks with dependencies to minimize project completion time"
- Uses linear constraints for precedence relationships
- Optimizes critical path and resource allocation
- See: [`project_scheduler.py`](examples/project_scheduler.py)

**🧩 Puzzle Solver**: "Solve a Sudoku puzzle or find valid N-Queens placement"
- Uses all_different constraints for logic puzzles
- Demonstrates pure constraint satisfaction
- See: [`sudoku_solver.py`](examples/sudoku_solver.py)

**💼 Budget Planner**: "Allocate $10,000 across 20 initiatives to maximize ROI under capacity constraints"
- Uses knapsack optimization for resource allocation
- Handles multiple constraints (budget, headcount, time)
- See: [`knapsack_optimizer.py`](examples/knapsack_optimizer.py)

### LLM/AI Agent Examples

**"Claude, I need to schedule a team meeting with 5 people. Alice is only free Mon/Wed, Bob can't do mornings, and Carol must attend before David. Find a time that works."**
- LLM extracts: 5 people, availability constraints, precedence constraint
- Solver finds: Valid meeting time satisfying all constraints
- Response: "Schedule meeting Wednesday 2-3pm: Alice, Bob, Carol attend first half; David joins after Carol confirms"

**"Help me plan a road trip visiting San Francisco, LA, Vegas, and Phoenix in the shortest route starting from Seattle."**
- LLM converts to: TSP problem with 5 cities
- Solver optimizes: Circuit constraint for minimum distance route
- Response: "Optimal route (1,247 miles): Seattle → SF (808mi) → LA (382mi) → Vegas (270mi) → Phoenix (297mi) → Seattle (1,440mi)"

**"I have $500/month for AI API costs. I need to process 10,000 text requests and 2,000 image requests. What's the cheapest mix of GPT-4, GPT-3.5, and Claude?"**
- LLM builds: Cost optimization problem with budget constraint
- Solver finds: Optimal model selection minimizing cost
- Response: "Use GPT-3.5 for 8,000 text ($40), Claude for 2,000 text ($30), GPT-4 for 2,000 images ($400). Total: $470/month"

**"I'm deploying a microservice that needs 16 CPU cores and 32GB RAM. Minimize cost but keep latency under 50ms. What AWS instances should I use?"**
- LLM creates: Multi-objective problem (cost priority 1, latency priority 2)
- Solver optimizes: Instance selection meeting requirements
- Response: "Deploy 4x c5.large instances (16 cores, 32GB total) at $340/month with 30ms latency"

**"We have 3 devs, 2 designers, 1 PM. Schedule 15 tasks over 2 weeks where: Task A needs 2 devs for 3 days, Task B needs 1 designer + 1 dev for 2 days, all tasks have dependencies."**
- LLM extracts: Resource requirements, durations, dependencies
- Solver schedules: Cumulative resource constraints + precedence
- Response: "Project completes in 12 days. Task A: Days 1-3 (Alice, Bob). Task B: Days 4-5 (Carol, David)..."

**"I need to maintain 500 units of inventory. I have supplier deliveries on days 1, 7, 14 and customer orders on days 3, 5, 10, 15. When should I schedule each delivery to never run out?"**
- LLM models: Reservoir constraint problem with stock levels
- Solver finds: Valid delivery schedule maintaining safety stock
- Response: "Schedule delivery 1 on day 0 (300 units), delivery 2 on day 6 (250 units), delivery 3 on day 12 (200 units). Stock never drops below 500."

**"Find me a valid Sudoku solution for this puzzle..."**
- LLM recognizes: Constraint satisfaction problem
- Solver finds: Valid solution using all_different constraints
- Response: Shows completed Sudoku grid

**"I have 10 research papers to review. Each needs 2-4 hours. Some must be done before others. I have 20 hours this week. Create an optimal schedule."**
- LLM extracts: Tasks, durations, precedence, time budget
- Solver optimizes: Maximize papers reviewed in 20 hours
- Response: "Can complete 7 papers in 20 hours: Paper A (2h, Mon 9-11am), Paper D (3h, Mon 11am-2pm)..."

## Installation

### Quick Start with uvx (Recommended)

```bash
# Run directly without installation
uvx chuk-mcp-solver
```

### Install from PyPI

```bash
# With pip
pip install chuk-mcp-solver

# With uv (faster)
uv pip install chuk-mcp-solver
```

### Local Development

```bash
# Clone and install
git clone https://github.com/chuk-ai/chuk-mcp-solver.git
cd chuk-mcp-solver
uv pip install -e ".[dev]"
```

## Quick Start

### As an MCP Server

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "solver": {
      "command": "uvx",
      "args": ["chuk-mcp-solver"]
    }
  }
}
```

Or use locally:

```json
{
  "mcpServers": {
    "solver": {
      "command": "uv",
      "args": ["run", "chuk-mcp-solver"],
      "cwd": "/path/to/chuk-mcp-solver"
    }
  }
}
```

### Programmatic Usage

```python
from chuk_mcp_solver.models import SolveConstraintModelRequest
from chuk_mcp_solver.solver import get_solver

# Define a simple optimization problem
request = SolveConstraintModelRequest(
    mode="optimize",
    variables=[
        {"id": "x", "domain": {"type": "integer", "lower": 0, "upper": 10}},
        {"id": "y", "domain": {"type": "integer", "lower": 0, "upper": 10}},
    ],
    constraints=[
        {
            "id": "capacity",
            "kind": "linear",
            "params": {
                "terms": [{"var": "x", "coef": 2}, {"var": "y", "coef": 3}],
                "sense": "<=",
                "rhs": 15,
            },
        }
    ],
    objective={
        "sense": "max",
        "terms": [{"var": "x", "coef": 5}, {"var": "y", "coef": 4}],
    },
)

# Solve
solver = get_solver("ortools")
response = await solver.solve_constraint_model(request)

print(f"Status: {response.status}")
print(f"Objective: {response.objective_value}")
for var in response.solutions[0].variables:
    print(f"  {var.id} = {var.value}")
```

## Examples

The `examples/` directory contains 8 complete examples demonstrating different constraint types and use cases:

### Logic & Constraint Satisfaction

**Sudoku Solver** (`sudoku_solver.py`)
- **Constraints**: all_different
- **Use Case**: Logic puzzles, constraint satisfaction

```bash
python examples/sudoku_solver.py
```

Solves a 4x4 Sudoku puzzle using all_different constraints for rows, columns, and boxes.

### Optimization Problems

**Knapsack Optimizer** (`knapsack_optimizer.py`)
- **Constraints**: Linear (capacity), binary variables
- **Use Case**: Resource allocation, packing problems

```bash
python examples/knapsack_optimizer.py
```

Classic 0/1 knapsack problem: maximize value subject to weight capacity.

**Project Scheduler** (`project_scheduler.py`)
- **Constraints**: Linear (precedence), optimization
- **Use Case**: Task scheduling with dependencies

```bash
python examples/project_scheduler.py
```

Minimizes project makespan with task precedence constraints.

### Advanced Scheduling Constraints

**Resource Scheduler** (`resource_scheduler.py`) 🆕
- **Constraints**: Cumulative (resource capacity)
- **Use Case**: CPU/memory allocation, worker scheduling

```bash
python examples/resource_scheduler.py
```

Schedules tasks with resource demand under capacity limits. Shows resource utilization timeline.

**Delivery Router** (`delivery_router.py`) 🆕
- **Constraints**: Circuit (Hamiltonian path)
- **Use Case**: TSP, vehicle routing, delivery optimization

```bash
python examples/delivery_router.py
```

Finds optimal delivery route visiting all customers with minimum distance.

**Inventory Manager** (`inventory_manager.py`) 🆕
- **Constraints**: Reservoir (stock levels)
- **Use Case**: Production/consumption scheduling, inventory management

```bash
python examples/inventory_manager.py
```

Manages inventory levels with production and consumption events, maintaining safety stock.

### Multi-Objective & AI Orchestration

**Multi-Objective Planner** (`multi_objective_planner.py`) 🆕
- **Constraints**: Multi-objective optimization
- **Use Case**: Cloud deployment, trade-off analysis

```bash
python examples/multi_objective_planner.py
```

Optimizes cloud deployment with multiple objectives: minimize cost (priority 1), minimize latency (priority 2).

**Tool/Model Selection** (`tool_selector.py`)
- **Constraints**: Implication (conditional logic)
- **Use Case**: MCP tool orchestration, model selection

```bash
python examples/tool_selector.py
```

Selects optimal AI models/tools for tasks under budget constraints using implication constraints.

### Example Output

**Resource Scheduler Output:**
```
Status: OPTIMAL
Minimum Project Duration: 9 time units

Resource Utilization Timeline:
Time | Utilization | Running Tasks
-----|-------------|------------------
   0 | 4/4 ████    | task_B, task_C
   1 | 4/4 ████    | task_B, task_C
   2 | 3/4 ███     | task_B
   4 | 4/4 ████    | task_A, task_D
...
```

**Delivery Router Output:**
```
Status: OPTIMAL
Minimum Total Distance: 46 km

Route: Warehouse → Customer_D → Customer_C → Customer_B → Customer_A → Warehouse
```

## Tool Reference

### `solve_constraint_model`

Solve a general constraint or optimization model.

**Parameters:**

- `mode` (str): `"satisfy"` for any feasible solution, `"optimize"` for best solution
- `variables` (list): Decision variables with domains
- `constraints` (list): Constraints to satisfy
- `objective` (dict, optional): Objective function (required if mode is `"optimize"`)
- `search` (dict, optional): Search configuration (time limits, etc.)

**Variable Schema:**

```python
{
    "id": "unique_id",
    "domain": {
        "type": "bool" | "integer",
        "lower": 0,  # for integer
        "upper": 10  # for integer
    },
    "metadata": {...}  # optional
}
```

**Constraint Types:**

1. **Linear**: `sum(coef * var) sense rhs`
   ```python
   {
       "id": "c1",
       "kind": "linear",
       "params": {
           "terms": [{"var": "x", "coef": 2}, {"var": "y", "coef": 3}],
           "sense": "<=",  # "<=", ">=", or "=="
           "rhs": 10
       }
   }
   ```

2. **All Different**: Variables must have distinct values
   ```python
   {
       "id": "c2",
       "kind": "all_different",
       "params": {"vars": ["x", "y", "z"]}
   }
   ```

3. **Element**: Array indexing `target = array[index]`
   ```python
   {
       "id": "c3",
       "kind": "element",
       "params": {
           "index_var": "idx",
           "array": [10, 20, 30],
           "target_var": "result"
       }
   }
   ```

4. **Table**: Allowed tuples
   ```python
   {
       "id": "c4",
       "kind": "table",
       "params": {
           "vars": ["x", "y"],
           "allowed_tuples": [[0, 1], [1, 0]]
       }
   }
   ```

5. **Implication**: If-then constraint
   ```python
   {
       "id": "c5",
       "kind": "implication",
       "params": {
           "if_var": "use_feature",
           "then": {
               "id": "cost",
               "kind": "linear",
               "params": {...}
           }
       }
   }
   ```

6. **Cumulative**: Resource scheduling with capacity
   ```python
   {
       "id": "c6",
       "kind": "cumulative",
       "params": {
           "start_vars": ["s1", "s2", "s3"],
           "duration_vars": [3, 4, 2],  # or variable IDs
           "demand_vars": [2, 1, 3],    # or variable IDs
           "capacity": 5
       }
   }
   ```

7. **Circuit**: Routing/Hamiltonian circuit
   ```python
   {
       "id": "c7",
       "kind": "circuit",
       "params": {
           "arcs": [
               (0, 1, "arc_0_1"),  # (from_node, to_node, bool_var)
               (1, 2, "arc_1_2"),
               # ...
           ]
       }
   }
   ```

8. **Reservoir**: Inventory/stock management
   ```python
   {
       "id": "c8",
       "kind": "reservoir",
       "params": {
           "time_vars": ["t1", "t2", "t3"],
           "level_changes": [5, -3, -2],  # production/consumption
           "min_level": 0,
           "max_level": 10
       }
   }
   ```

9. **No-Overlap**: Disjunctive scheduling
   ```python
   {
       "id": "c9",
       "kind": "no_overlap",
       "params": {
           "start_vars": ["s1", "s2", "s3"],
           "duration_vars": [3, 4, 2]  # or variable IDs
       }
   }
   ```

**Multi-Objective Optimization:**

```python
{
    "mode": "optimize",
    "objective": [
        {
            "sense": "max",
            "terms": [{"var": "x", "coef": 1}],
            "priority": 2,  # Higher priority
            "weight": 1.0
        },
        {
            "sense": "max",
            "terms": [{"var": "y", "coef": 1}],
            "priority": 1,  # Lower priority
            "weight": 1.0
        }
    ]
}
```

**Search Configuration:**

```python
{
    "search": {
        "max_time_ms": 5000,
        "max_solutions": 1,
        "num_search_workers": 4,
        "log_search_progress": false,
        "warm_start_solution": {"x": 5, "y": 3}
    }
}
```

**Response Schema:**

```python
{
    "status": "optimal" | "feasible" | "satisfied" | "infeasible" | "unbounded" | "timeout" | "error",
    "objective_value": 42.0,  # if applicable
    "solutions": [
        {
            "variables": [
                {"id": "x", "value": 5, "metadata": {...}}
            ],
            "derived": {...}  # optional computed metrics
        }
    ],
    "explanation": {
        "summary": "Found optimal solution...",
        "binding_constraints": [...]  # tight constraints
    }
}
```

## Configuration

### Environment Variables

```bash
# Provider selection
export CHUK_SOLVER_PROVIDER=ortools

# Tool-specific provider
export CHUK_SOLVER_TOOL_PROVIDER=ortools

# Config file location
export CHUK_SOLVER_CONFIG=/path/to/config.yaml
```

### YAML Configuration

Create `~/.config/chuk-mcp-solver/config.yaml`:

```yaml
default_provider: ortools

tool_providers:
  solve_constraint_model: ortools
```

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/chuk-ai/chuk-mcp-solver.git
cd chuk-mcp-solver

# Install with dev dependencies
uv pip install -e ".[dev]"
```

### Testing

```bash
# Run tests
make test

# Run with coverage (requires 90%+)
make test-cov

# Run specific tests
pytest tests/test_models.py -v

# Current stats: 105 tests, 91% coverage ✅
```

### Code Quality

```bash
# Lint
make lint

# Format
make format

# Type check
make typecheck

# All checks
make check
```

### Running Locally

```bash
# Run server
make run

# Or directly
python -m chuk_mcp_solver.server
```

## Architecture

```
chuk-mcp-solver/
├── src/chuk_mcp_solver/
│   ├── __init__.py          # Package metadata
│   ├── server.py            # MCP server + tools
│   ├── models.py            # Pydantic models + enums
│   ├── config.py            # Configuration management
│   └── solver/              # Solver implementations
│       ├── __init__.py      # Solver factory (get_solver)
│       ├── provider.py      # Abstract solver interface
│       └── ortools/         # OR-Tools implementation
│           ├── solver.py    # Main ORToolsSolver class
│           ├── constraints.py  # Constraint builders
│           ├── objectives.py   # Objective builders
│           └── responses.py    # Response builders
├── tests/                   # Comprehensive test suite
│   ├── test_solver.py       # Factory tests
│   └── solver/ortools/      # OR-Tools tests (mirrors source)
│       ├── test_solver.py
│       ├── test_constraints.py
│       ├── test_responses.py
│       └── test_edge_cases.py
├── examples/                # Example scripts
└── pyproject.toml           # Package configuration
```

**Key Design Patterns:**

- **Modular Architecture**: Focused modules with single responsibilities
- **Solver Pattern**: Pluggable solver backends via abstract interface
- **Factory Function**: Simple `get_solver()` for solver instantiation
- **Pydantic Models**: Type-safe throughout
- **Async Native**: Non-blocking I/O
- **No Magic Strings**: Enums for all constants
- **Mirrored Test Structure**: Tests match source organization

## Use Cases

### Scheduling & Resource Allocation
- **Project Scheduling**: Task scheduling with precedence constraints → [`project_scheduler.py`](examples/project_scheduler.py)
- **Resource-Constrained Scheduling**: CPU/memory/worker allocation with capacity limits → [`resource_scheduler.py`](examples/resource_scheduler.py)
- **Inventory Management**: Production/consumption planning with stock levels → [`inventory_manager.py`](examples/inventory_manager.py)
- **Shift Rostering**: Employee scheduling with availability and skill constraints
- **Meeting Scheduling**: Calendar optimization with participant constraints

### Routing & Logistics
- **Vehicle Routing**: Delivery route optimization (TSP/VRP) → [`delivery_router.py`](examples/delivery_router.py)
- **Circuit Planning**: Hamiltonian path/circuit problems
- **Network Design**: Optimal path selection in graphs
- **Warehouse Optimization**: Pick path optimization

### Optimization Problems
- **Knapsack Problems**: Resource allocation under weight/capacity limits → [`knapsack_optimizer.py`](examples/knapsack_optimizer.py)
- **Budget Allocation**: Optimal spending across categories
- **Portfolio Selection**: Asset selection under risk/return constraints
- **Packing Problems**: Bin packing, cutting stock

### AI/LLM Orchestration
- **Multi-Model Selection**: Choose optimal AI models under budget → [`tool_selector.py`](examples/tool_selector.py)
- **Multi-Objective Planning**: Balance cost, latency, quality trade-offs → [`multi_objective_planner.py`](examples/multi_objective_planner.py)
- **Rate-Limit Aware Scheduling**: Task scheduling respecting API limits
- **Capability-Based Routing**: Route requests to appropriate models
- **Cost-Latency Optimization**: Minimize cost while meeting SLAs

### Configuration & Selection
- **System Configuration**: Parameter optimization under constraints
- **Feature Selection**: Optimal feature subset selection
- **Bundle Recommendations**: Best product/service combinations
- **Resource Sizing**: Cloud instance selection and sizing

### Logic Puzzles
- **Sudoku**: Constraint satisfaction puzzles → [`sudoku_solver.py`](examples/sudoku_solver.py)
- **Kakuro, KenKen**: Arithmetic constraint puzzles
- **Logic Grids**: Deductive reasoning puzzles
- **N-Queens**: Placement problems

## Roadmap

### Completed ✅

- [x] Cumulative constraints (resource scheduling)
- [x] Circuit constraints (routing/TSP)
- [x] Reservoir constraints (inventory management)
- [x] No-overlap constraints (disjunctive scheduling)
- [x] Multi-objective optimization (priority-based)
- [x] Warm-start from previous solutions
- [x] Parallel search workers
- [x] Search progress logging

### Planned 🚧

- [ ] Solution enumeration (find N diverse solutions)
- [ ] Relaxation and feasibility analysis
- [ ] Export to MPS/LP formats
- [ ] Visualization of solutions and Gantt charts
- [ ] Advanced search strategies (custom heuristics)
- [ ] Symmetry breaking
- [ ] Decomposition strategies

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure `make check` passes
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- 📖 [Documentation](https://github.com/chuk-ai/chuk-mcp-solver#readme)
- 🐛 [Issue Tracker](https://github.com/chuk-ai/chuk-mcp-solver/issues)
- 💬 [Discussions](https://github.com/chuk-ai/chuk-mcp-solver/discussions)

## Acknowledgments

Built with:
- [Google OR-Tools](https://developers.google.com/optimization) - CP-SAT solver
- [Pydantic](https://pydantic.dev) - Data validation
- [chuk-mcp-server](https://github.com/chuk-ai/chuk-mcp-server) - MCP framework

---

Made with ❤️ by [CHUK AI](https://chuk.ai)
