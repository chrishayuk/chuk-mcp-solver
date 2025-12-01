"""Example: Solving a 4x4 Sudoku puzzle.

Demonstrates constraint satisfaction using all_different constraints.
"""

import asyncio

from chuk_mcp_solver.models import SolveConstraintModelRequest
from chuk_mcp_solver.solver import get_solver


def build_4x4_sudoku_model(given_values: list[list[int]]) -> dict:
    """Build a 4x4 Sudoku model.

    Args:
        given_values: 4x4 grid with 0 for empty cells, 1-4 for given values.

    Returns:
        Model dictionary ready for SolveConstraintModelRequest.
    """
    variables = []
    constraints = []

    # Create variables (one per cell)
    for row in range(4):
        for col in range(4):
            variables.append(
                {
                    "id": f"cell_{row}_{col}",
                    "domain": {"type": "integer", "lower": 1, "upper": 4},
                    "metadata": {"row": row, "col": col},
                }
            )

    # Add given values as constraints
    for row in range(4):
        for col in range(4):
            if given_values[row][col] != 0:
                constraints.append(
                    {
                        "id": f"given_{row}_{col}",
                        "kind": "linear",
                        "params": {
                            "terms": [{"var": f"cell_{row}_{col}", "coef": 1}],
                            "sense": "==",
                            "rhs": given_values[row][col],
                        },
                        "metadata": {"description": f"Given value at ({row},{col})"},
                    }
                )

    # Row constraints: all different in each row
    for row in range(4):
        row_vars = [f"cell_{row}_{col}" for col in range(4)]
        constraints.append(
            {
                "id": f"row_{row}",
                "kind": "all_different",
                "params": {"vars": row_vars},
                "metadata": {"description": f"Row {row} all different"},
            }
        )

    # Column constraints: all different in each column
    for col in range(4):
        col_vars = [f"cell_{row}_{col}" for row in range(4)]
        constraints.append(
            {
                "id": f"col_{col}",
                "kind": "all_different",
                "params": {"vars": col_vars},
                "metadata": {"description": f"Column {col} all different"},
            }
        )

    # 2x2 block constraints
    for block_row in range(2):
        for block_col in range(2):
            block_vars = []
            for r in range(2):
                for c in range(2):
                    row = block_row * 2 + r
                    col = block_col * 2 + c
                    block_vars.append(f"cell_{row}_{col}")

            constraints.append(
                {
                    "id": f"block_{block_row}_{block_col}",
                    "kind": "all_different",
                    "params": {"vars": block_vars},
                    "metadata": {"description": f"Block ({block_row},{block_col}) all different"},
                }
            )

    return {
        "mode": "satisfy",
        "variables": variables,
        "constraints": constraints,
    }


def display_grid(solution_vars: list, size: int = 4) -> None:
    """Display the solved grid.

    Args:
        solution_vars: List of SolutionVariable objects.
        size: Grid size (default 4x4).
    """
    # Extract values
    grid = [[0] * size for _ in range(size)]
    for var in solution_vars:
        row = var.metadata["row"]
        col = var.metadata["col"]
        grid[row][col] = int(var.value)

    # Print grid
    print("\nSolved 4x4 Sudoku:")
    print("┌─────┬─────┐")
    for row in range(size):
        if row == 2:
            print("├─────┼─────┤")
        print("│", end="")
        for col in range(size):
            if col == 2:
                print(" │", end="")
            print(f" {grid[row][col]}", end="")
        print(" │")
    print("└─────┴─────┘")


async def main() -> None:
    """Run the Sudoku solver example."""
    print("=== 4x4 Sudoku Solver Example ===\n")

    # Define a puzzle (0 = empty)
    puzzle = [
        [1, 0, 0, 0],
        [0, 0, 2, 0],
        [0, 3, 0, 0],
        [0, 0, 0, 4],
    ]

    print("Puzzle:")
    print("┌─────┬─────┐")
    for row in range(4):
        if row == 2:
            print("├─────┼─────┤")
        print("│", end="")
        for col in range(4):
            if col == 2:
                print(" │", end="")
            val = puzzle[row][col]
            print(f" {val if val != 0 else '.'}", end="")
        print(" │")
    print("└─────┴─────┘")

    # Build model
    model_dict = build_4x4_sudoku_model(puzzle)
    request = SolveConstraintModelRequest(**model_dict)

    # Solve
    print("\nSolving...")
    solver = get_solver("ortools")
    response = await solver.solve_constraint_model(request)

    # Display results
    print(f"\nStatus: {response.status}")

    if response.solutions:
        display_grid(response.solutions[0].variables)
    else:
        print("No solution found!")

    if response.explanation:
        print(f"\n{response.explanation.summary}")


if __name__ == "__main__":
    asyncio.run(main())
