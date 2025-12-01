"""Shared pytest fixtures for chuk-mcp-solver tests."""

import pytest


@pytest.fixture
def simple_knapsack_request() -> dict:
    """Simple knapsack problem for testing."""
    return {
        "mode": "optimize",
        "variables": [
            {"id": "item_1", "domain": {"type": "bool"}},
            {"id": "item_2", "domain": {"type": "bool"}},
            {"id": "item_3", "domain": {"type": "bool"}},
        ],
        "constraints": [
            {
                "id": "capacity",
                "kind": "linear",
                "params": {
                    "terms": [
                        {"var": "item_1", "coef": 3},
                        {"var": "item_2", "coef": 5},
                        {"var": "item_3", "coef": 2},
                    ],
                    "sense": "<=",
                    "rhs": 7,
                },
                "metadata": {"description": "Total weight cannot exceed 7"},
            }
        ],
        "objective": {
            "sense": "max",
            "terms": [
                {"var": "item_1", "coef": 10},
                {"var": "item_2", "coef": 15},
                {"var": "item_3", "coef": 8},
            ],
        },
    }


@pytest.fixture
def simple_sudoku_cell_request() -> dict:
    """Simple 4-cell all-different problem."""
    return {
        "mode": "satisfy",
        "variables": [
            {"id": "cell_0", "domain": {"type": "integer", "lower": 1, "upper": 4}},
            {"id": "cell_1", "domain": {"type": "integer", "lower": 1, "upper": 4}},
            {"id": "cell_2", "domain": {"type": "integer", "lower": 1, "upper": 4}},
            {"id": "cell_3", "domain": {"type": "integer", "lower": 1, "upper": 4}},
        ],
        "constraints": [
            {
                "id": "all_diff",
                "kind": "all_different",
                "params": {"vars": ["cell_0", "cell_1", "cell_2", "cell_3"]},
            },
            {
                "id": "cell_0_fixed",
                "kind": "linear",
                "params": {"terms": [{"var": "cell_0", "coef": 1}], "sense": "==", "rhs": 1},
            },
        ],
    }


@pytest.fixture
def infeasible_request() -> dict:
    """An infeasible problem for testing."""
    return {
        "mode": "satisfy",
        "variables": [
            {"id": "x", "domain": {"type": "integer", "lower": 0, "upper": 10}},
        ],
        "constraints": [
            {
                "id": "lower_bound",
                "kind": "linear",
                "params": {"terms": [{"var": "x", "coef": 1}], "sense": ">=", "rhs": 5},
            },
            {
                "id": "upper_bound",
                "kind": "linear",
                "params": {"terms": [{"var": "x", "coef": 1}], "sense": "<=", "rhs": 4},
            },
        ],
    }
