"""Budget allocation demonstration.

This example shows how to use the high-level solve_budget_allocation tool
to solve knapsack and portfolio selection problems without directly working
with CP-SAT constraints.
"""

import asyncio

from chuk_mcp_solver.server import solve_budget_allocation


async def demo_simple_knapsack():
    """Demo 1: Simple knapsack problem."""
    print("=" * 70)
    print("Demo 1: Simple Knapsack Problem")
    print("=" * 70)

    response = await solve_budget_allocation(
        items=[
            {"id": "item_A", "cost": 5000, "value": 12000},
            {"id": "item_B", "cost": 3000, "value": 7000},
            {"id": "item_C", "cost": 4000, "value": 9000},
            {"id": "item_D", "cost": 2000, "value": 5000},
        ],
        budgets=[{"resource": "money", "limit": 10000}],
        objective="maximize_value",
    )

    print(f"Status: {response.status}")
    print(f"Selected items: {', '.join(response.selected_items)}")
    print(f"Total cost: ${response.total_cost:,.2f}")
    print(f"Total value: ${response.total_value:,.2f}")
    print(f"Budget remaining: ${response.resource_slack['money']:,.2f}")
    print(f"Solve time: {response.solve_time_ms}ms")
    print(f"\n{response.explanation.summary}")
    print()


async def demo_portfolio_selection():
    """Demo 2: Project portfolio selection with dependencies."""
    print("=" * 70)
    print("Demo 2: Project Portfolio Selection")
    print("=" * 70)

    response = await solve_budget_allocation(
        items=[
            {"id": "backend_api", "cost": 8000, "value": 5000},
            {
                "id": "web_frontend",
                "cost": 6000,
                "value": 12000,
                "dependencies": ["backend_api"],
            },
            {
                "id": "mobile_app",
                "cost": 7000,
                "value": 10000,
                "dependencies": ["backend_api"],
            },
            {"id": "analytics", "cost": 4000, "value": 6000},
            {"id": "marketing", "cost": 5000, "value": 8000},
        ],
        budgets=[{"resource": "money", "limit": 20000}],
        objective="maximize_value",
    )

    print(f"Status: {response.status}")
    print("\nSelected projects:")
    for i, item in enumerate(response.selected_items, 1):
        print(f"  {i}. {item}")

    print(f"\nBudget: ${response.resource_usage['money']:,.2f} / $20,000")
    print(f"Remaining: ${response.resource_slack['money']:,.2f}")
    print(f"Total value: ${response.total_value:,.2f}")
    print(f"\n{response.explanation.summary}")

    if response.explanation.binding_constraints:
        print("\nBinding constraints:")
        for constraint in response.explanation.binding_constraints:
            print(f"  - {constraint}")
    print()


async def demo_feature_prioritization():
    """Demo 3: Feature prioritization with conflicts."""
    print("=" * 70)
    print("Demo 3: Feature Prioritization with Conflicts")
    print("=" * 70)

    response = await solve_budget_allocation(
        items=[
            {
                "id": "mobile_checkout",
                "cost": 5000,
                "value": 15000,
                "conflicts": ["web_checkout_redesign"],
            },
            {
                "id": "web_checkout_redesign",
                "cost": 6000,
                "value": 14000,
                "conflicts": ["mobile_checkout"],
            },
            {"id": "payment_methods", "cost": 4000, "value": 10000},
            {"id": "email_receipts", "cost": 2000, "value": 5000},
            {"id": "order_tracking", "cost": 3000, "value": 8000},
        ],
        budgets=[{"resource": "money", "limit": 12000}],
        objective="maximize_value",
    )

    print(f"Status: {response.status}")
    print("\nSelected features:")
    for i, item in enumerate(response.selected_items, 1):
        print(f"  {i}. {item}")

    print(f"\nTotal effort: ${response.total_cost:,.2f}")
    print(f"Total business value: ${response.total_value:,.2f}")
    print("\nNote: mobile_checkout and web_checkout_redesign are mutually exclusive")
    print(f"{response.explanation.summary}")
    print()


async def demo_multi_resource_allocation():
    """Demo 4: Multi-resource allocation (budget + headcount + time)."""
    print("=" * 70)
    print("Demo 4: Multi-Resource Allocation")
    print("=" * 70)

    response = await solve_budget_allocation(
        items=[
            {
                "id": "feature_A",
                "cost": 5000,
                "value": 15000,
                "resources_required": {"headcount": 2, "time": 3},
            },
            {
                "id": "feature_B",
                "cost": 3000,
                "value": 10000,
                "resources_required": {"headcount": 1, "time": 2},
            },
            {
                "id": "feature_C",
                "cost": 4000,
                "value": 12000,
                "resources_required": {"headcount": 2, "time": 1},
            },
            {
                "id": "feature_D",
                "cost": 2000,
                "value": 6000,
                "resources_required": {"headcount": 1, "time": 1},
            },
        ],
        budgets=[
            {"resource": "money", "limit": 10000},
            {"resource": "headcount", "limit": 4},
            {"resource": "time", "limit": 5},
        ],
        objective="maximize_value",
    )

    print(f"Status: {response.status}")
    print("\nSelected features:")
    for i, item in enumerate(response.selected_items, 1):
        print(f"  {i}. {item}")

    print("\nResource utilization:")
    print(f"  Money:     ${response.resource_usage.get('money', 0):,.2f} / $10,000")
    print(f"  Headcount: {response.resource_usage.get('headcount', 0):.0f} / 4 people")
    print(f"  Time:      {response.resource_usage.get('time', 0):.0f} / 5 months")

    print("\nResource slack:")
    for resource, slack in response.resource_slack.items():
        print(f"  {resource.capitalize()}: {slack:.2f}")

    print(f"\nTotal value: ${response.total_value:,.2f}")
    print(f"{response.explanation.summary}")
    print()


async def demo_maximize_count():
    """Demo 5: Maximize number of items under budget."""
    print("=" * 70)
    print("Demo 5: Maximize Item Count")
    print("=" * 70)

    response = await solve_budget_allocation(
        items=[
            {"id": "quick_win_1", "cost": 1000, "value": 3000},
            {"id": "quick_win_2", "cost": 1500, "value": 4000},
            {"id": "quick_win_3", "cost": 2000, "value": 5000},
            {"id": "big_project", "cost": 8000, "value": 20000},
            {"id": "medium_project", "cost": 4000, "value": 10000},
        ],
        budgets=[{"resource": "money", "limit": 10000}],
        objective="maximize_count",
    )

    print(f"Status: {response.status}")
    print(f"\nSelected {len(response.selected_items)} items:")
    for i, item in enumerate(response.selected_items, 1):
        print(f"  {i}. {item}")

    print(f"\nTotal cost: ${response.total_cost:,.2f}")
    print(f"Total value: ${response.total_value:,.2f}")
    print(f"Budget remaining: ${response.resource_slack['money']:,.2f}")
    print("\nObjective was to maximize COUNT, not VALUE")
    print(f"{response.explanation.summary}")
    print()


async def demo_minimize_cost_with_threshold():
    """Demo 6: Minimize cost while meeting value threshold."""
    print("=" * 70)
    print("Demo 6: Minimize Cost with Value Threshold")
    print("=" * 70)

    response = await solve_budget_allocation(
        items=[
            {"id": "expensive_A", "cost": 10000, "value": 25000},
            {"id": "efficient_B", "cost": 4000, "value": 12000},
            {"id": "efficient_C", "cost": 5000, "value": 15000},
            {"id": "small_D", "cost": 2000, "value": 5000},
        ],
        budgets=[{"resource": "money", "limit": 30000}],
        objective="minimize_cost",
        min_value_threshold=25000,  # Must achieve at least this much value
    )

    print(f"Status: {response.status}")
    print(f"\nSelected items to meet ${25000:,.2f} value threshold:")
    for i, item in enumerate(response.selected_items, 1):
        print(f"  {i}. {item}")

    print(f"\nMinimum cost: ${response.total_cost:,.2f}")
    print(f"Value achieved: ${response.total_value:,.2f}")
    print(f"Budget used: {response.total_cost / 30000 * 100:.1f}%")
    print(f"\n{response.explanation.summary}")
    print()


async def demo_item_count_constraints():
    """Demo 7: Min/max item count constraints."""
    print("=" * 70)
    print("Demo 7: Item Count Constraints")
    print("=" * 70)

    response = await solve_budget_allocation(
        items=[
            {"id": "priority_1", "cost": 5000, "value": 15000},
            {"id": "priority_2", "cost": 4000, "value": 12000},
            {"id": "priority_3", "cost": 3000, "value": 9000},
            {"id": "priority_4", "cost": 2000, "value": 6000},
            {"id": "priority_5", "cost": 1000, "value": 3000},
        ],
        budgets=[{"resource": "money", "limit": 15000}],
        objective="maximize_value",
        min_items=3,  # Must select at least 3 items
        max_items=4,  # But no more than 4 items
    )

    print(f"Status: {response.status}")
    print(f"\nSelected {len(response.selected_items)} items (min: 3, max: 4):")
    for i, item in enumerate(response.selected_items, 1):
        print(f"  {i}. {item}")

    print(f"\nTotal cost: ${response.total_cost:,.2f}")
    print(f"Total value: ${response.total_value:,.2f}")
    print(f"{response.explanation.summary}")
    print()


async def main():
    """Run all allocation demos."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 16 + "BUDGET ALLOCATION EXAMPLES" + " " * 26 + "║")
    print("║" + " " * 68 + "║")
    print("║  High-level allocation API for knapsack and portfolio" + " " * 14 + "║")
    print("║  selection with dependencies, conflicts, and resources." + " " * 12 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")

    await demo_simple_knapsack()
    await demo_portfolio_selection()
    await demo_feature_prioritization()
    await demo_multi_resource_allocation()
    await demo_maximize_count()
    await demo_minimize_cost_with_threshold()
    await demo_item_count_constraints()

    print("=" * 70)
    print("All allocation demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
