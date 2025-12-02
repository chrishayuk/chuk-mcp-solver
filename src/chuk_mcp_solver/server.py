"""MCP server for constraint and optimization solving.

Exposes constraint solving and optimization capabilities via MCP tools.
"""

import logging
import sys

from chuk_mcp_server import ChukMCPServer, tool

from chuk_mcp_solver.models import SolveConstraintModelRequest, SolveConstraintModelResponse
from chuk_mcp_solver.providers import get_provider_for_tool

logger = logging.getLogger(__name__)


@tool  # type: ignore[arg-type]
async def solve_constraint_model(
    mode: str,
    variables: list[dict],
    constraints: list[dict],
    objective: dict | None = None,
    search: dict | None = None,
) -> SolveConstraintModelResponse:
    """Solve a general constraint or optimization model.

    This tool solves discrete optimization and constraint satisfaction problems.
    It supports integer and boolean variables, linear constraints, global constraints
    (all_different, element, table), implications, and linear objectives.

    Use cases include:
    - Project scheduling and resource allocation
    - Sudoku and logic puzzles
    - Configuration optimization
    - Tool/model selection under constraints
    - Routing and assignment problems
    - Budget allocation

    Args:
        mode: Solver mode - 'satisfy' to find any feasible solution,
              'optimize' to find the best solution according to the objective.
        variables: List of decision variables, each with:
            - id (str): Unique identifier
            - domain (dict): Domain specification with:
                - type (str): 'bool' or 'integer'
                - lower (int): Lower bound for integers (default 0)
                - upper (int): Upper bound for integers (default 1)
            - metadata (dict, optional): Context for explanations
        constraints: List of constraints, each with:
            - id (str): Unique identifier
            - kind (str): Constraint type - 'linear', 'all_different', 'element',
                         'table', or 'implication'
            - params (dict): Constraint-specific parameters:
                For 'linear': terms (list of {var, coef}), sense ('<=', '>=', '=='), rhs (number)
                For 'all_different': vars (list of variable ids)
                For 'element': index_var (str), array (list of int), target_var (str)
                For 'table': vars (list of str), allowed_tuples (list of lists)
                For 'implication': if_var (str), then (nested constraint dict)
            - metadata (dict, optional): Description and context
        objective: Optional objective function (required if mode='optimize'):
            - sense (str): 'min' or 'max'
            - terms (list): Linear terms as {var, coef}
            - metadata (dict, optional): Description
        search: Optional search configuration:
            - max_time_ms (int): Maximum solver time in milliseconds
            - max_solutions (int): Maximum solutions to return (default 1)

    Returns:
        SolveConstraintModelResponse containing:
            - status: 'optimal', 'feasible', 'satisfied', 'infeasible', 'unbounded',
                     'timeout', or 'error'
            - objective_value: Objective value if applicable
            - solutions: List of solutions with variable assignments
            - explanation: Human-readable summary and binding constraints

    Tips for LLMs:
        - Start with a small model to test; gradually add complexity.
        - For Sudoku: use 'all_different' constraints for rows, columns, and blocks.
        - For scheduling: use linear constraints for precedence and capacity.
        - Variable metadata is useful for building readable explanations.
        - Constraint metadata helps identify which constraints are tight.
        - If infeasible, check constraint metadata to diagnose conflicts.
        - Use 'satisfy' mode for puzzles; 'optimize' mode for cost/time minimization.

    Example (simple knapsack):
        ```python
        response = await solve_constraint_model(
            mode="optimize",
            variables=[
                {"id": "take_item_1", "domain": {"type": "bool"}},
                {"id": "take_item_2", "domain": {"type": "bool"}},
            ],
            constraints=[
                {
                    "id": "capacity",
                    "kind": "linear",
                    "params": {
                        "terms": [
                            {"var": "take_item_1", "coef": 3},
                            {"var": "take_item_2", "coef": 5},
                        ],
                        "sense": "<=",
                        "rhs": 7,
                    },
                }
            ],
            objective={
                "sense": "max",
                "terms": [
                    {"var": "take_item_1", "coef": 10},
                    {"var": "take_item_2", "coef": 15},
                ],
            },
        )
        ```
    """
    # Construct request model from dict inputs
    request_data = {
        "mode": mode,
        "variables": variables,
        "constraints": constraints,
        "objective": objective,
        "search": search,
    }

    request = SolveConstraintModelRequest(**request_data)

    # Get provider and solve
    provider = get_provider_for_tool("solve_constraint_model")
    response = await provider.solve_constraint_model(request)

    return response


def main() -> None:
    """Main entry point for the MCP server."""
    # Default to stdio for MCP compatibility (Claude Desktop, mcp-cli)
    transport = "stdio"

    # Allow HTTP mode via command line
    if len(sys.argv) > 1 and sys.argv[1] in ["http", "--http"]:
        transport = "http"
        logger.warning("Starting CHUK MCP Solver in HTTP mode")

    # Suppress logging in STDIO mode
    if transport == "stdio":
        # Set chuk_mcp_server loggers to ERROR only
        logging.getLogger("chuk_mcp_server").setLevel(logging.ERROR)
        logging.getLogger("chuk_mcp_server.core").setLevel(logging.ERROR)
        logging.getLogger("chuk_mcp_server.stdio_transport").setLevel(logging.ERROR)
        # Suppress httpx logging
        logging.getLogger("httpx").setLevel(logging.ERROR)

    # Create and run server
    server = ChukMCPServer("chuk-mcp-solver")

    if transport == "stdio":
        server.run(stdio=True)
    else:
        # Bind to all interfaces for Docker containers
        server.run(host="0.0.0.0", port=8000)  # nosec B104


if __name__ == "__main__":
    main()
