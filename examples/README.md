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

## Examples

### 1. Sudoku Solver (`sudoku_solver.py`)
Demonstrates:
- Constraint satisfaction (no objective)
- `all_different` global constraints
- Integer variables with domains
- Solving logic puzzles

### 2. Project Scheduler (`project_scheduler.py`)
Demonstrates:
- Optimization with objective
- Linear constraints for precedence and resources
- Minimizing makespan (project duration)
- Binding constraint analysis

### 3. Knapsack Optimizer (`knapsack_optimizer.py`)
Demonstrates:
- Binary (boolean) variables
- Capacity constraints
- Value maximization
- Classic optimization problem

### 4. Tool Selection (`tool_selector.py`)
Demonstrates:
- Implication constraints
- Cost/latency trade-offs
- Multi-objective considerations
- Practical MCP use case

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
