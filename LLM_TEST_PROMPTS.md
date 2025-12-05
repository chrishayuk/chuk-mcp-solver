# LLM Test Prompts for CHUK MCP Solver

This document contains test prompts for each MCP tool. Use these to verify the solver works correctly with LLMs.

---

## 1. `solve_constraint_model` - General Constraint Solving

### Test 1.1: Simple Optimization (Knapsack)
```
Use the budget allocation tool to solve this packing problem:

I need to pack items for a camping trip. I have:
- Tent: 5kg, value 100
- Sleeping bag: 3kg, value 80
- Food: 2kg, value 60
- Water: 4kg, value 70
- First aid kit: 1kg, value 50

My backpack can only carry 10kg. What items should I take to maximize value?
```

**Expected behavior**: LLM uses `solve_budget_allocation` with weight budget constraint (or `solve_constraint_model` with binary variables and linear constraint for weight).

---

### Test 1.2: Logic Puzzle (Sudoku)
```
Use the constraint solver to find a solution to this 4x4 Sudoku puzzle:

2 _ | _ 3
_ 3 | 2 _
----+----
3 _ | _ 2
_ 2 | 3 _

Each row, column, and 2x2 box must contain numbers 1-4 exactly once.
Use the solve_constraint_model tool with all_different constraints.
```

**Expected behavior**: LLM uses `solve_constraint_model` with `all_different` constraints for rows, columns, and boxes.

---

### Test 1.3: Resource Scheduling
```
I need to schedule 3 tasks on a server with 4 CPU cores:

Task A: runs for 3 hours, needs 2 cores
Task B: runs for 2 hours, needs 3 cores
Task C: runs for 4 hours, needs 1 core

What's the earliest I can finish all tasks if they can run in parallel?
```

**Expected behavior**: LLM uses `cumulative` constraint for CPU capacity and optimizes makespan.

---

## 2. `solve_scheduling_problem` - High-Level Task Scheduling

### Test 2.1: Simple Project Schedule
```
I'm managing a software deployment with these tasks:

1. Build code (2 hours)
2. Run tests (3 hours) - must happen after build
3. Deploy to staging (1 hour) - must happen after tests
4. Deploy to production (1 hour) - must happen after staging deployment

How long will the whole process take? Give me the start and end time for each task.
```

**Expected behavior**: LLM uses `solve_scheduling_problem` with dependencies, gets makespan of 7 hours.

---

### Test 2.2: Resource-Constrained Scheduling
```
I have 3 developers working on 5 tasks:

Task 1: 4 hours, needs 2 developers
Task 2: 3 hours, needs 1 developer
Task 3: 2 hours, needs 2 developers (must finish before Task 4)
Task 4: 3 hours, needs 1 developer
Task 5: 5 hours, needs 2 developers

Only 3 developers are available at any time. What's the optimal schedule?
```

**Expected behavior**: LLM uses resources with capacity=3, automatic cumulative constraints.

---

### Test 2.3: Deadlines and Release Times
```
I need to schedule 3 marketing tasks:

- Design assets: 5 days duration, can start anytime
- Write copy: 3 days duration, must start after day 2 (waiting for client input)
- Launch campaign: 2 days duration, must complete by day 12 (hard deadline)
- Launch depends on both design and copy being complete

Can this be done? If so, what's the schedule?
```

**Expected behavior**: LLM uses `earliest_start` and `deadline` parameters, checks feasibility.

---

## 3. `solve_routing_problem` - TSP/VRP Optimization

### Test 3.1: Simple TSP (Traveling Salesman)
```
I need to visit 4 cities starting from my home:

- Home: (0, 0)
- City A: (10, 5)
- City B: (5, 10)
- City C: (15, 8)

What's the shortest route that visits all cities and returns home?
```

**Expected behavior**: Single-vehicle TSP with coordinates, minimize distance.

---

### Test 3.2: Multi-Vehicle VRP with Capacity
```
I run a delivery company with 2 vans (each carries 30 packages) delivering to 4 stores:

- Warehouse: (0, 0) - no packages needed
- Store A: (10, 5) - needs 12 packages
- Store B: (5, 10) - needs 15 packages
- Store C: (15, 15) - needs 18 packages
- Store D: (8, 3) - needs 10 packages

Each van starts at the warehouse. What's the optimal route for each van to minimize total distance?
```

**Expected behavior**: Multi-vehicle VRP with capacity constraints, 2 vehicles used.

---

### Test 3.3: Service Times and Cost Optimization
```
I have a maintenance van visiting 3 client sites:

- Depot: (0, 0), no service time
- Client 1: (20, 10), 30 minutes service time
- Client 2: (10, 20), 45 minutes service time
- Client 3: (30, 15), 20 minutes service time

The van costs $50 fixed cost + $2 per mile driven. Minimize total cost.
```

**Expected behavior**: Uses `service_time` and vehicle costs, minimize cost objective.

---

## 4. `solve_budget_allocation` - Portfolio Selection

### Test 4.1: Simple Budget Allocation
```
I have $10,000 to invest in projects:

- Project A: costs $5,000, returns $12,000 value
- Project B: costs $3,000, returns $7,000 value
- Project C: costs $4,000, returns $9,000 value
- Project D: costs $6,000, returns $11,000 value

Which projects should I fund to maximize value?
```

**Expected behavior**: Classic knapsack, selects projects A and C for $9,000 cost, $21,000 value.

---

### Test 4.2: Dependencies and Conflicts
```
I'm planning a product roadmap with $20,000 budget:

Features:
- Backend API: $8,000 cost, $5,000 value
- Frontend Web: $6,000 cost, $12,000 value (requires Backend API)
- Mobile App: $7,000 cost, $10,000 value (requires Backend API)
- Analytics: $3,000 cost, $8,000 value

Mobile App and Frontend Web conflict (can't do both this quarter).

What should I build?
```

**Expected behavior**: Uses dependencies and conflicts, respects constraints.

---

### Test 4.3: Multi-Resource Constraints
```
I'm allocating Q1 resources across 4 initiatives:

Initiative A: $10k, 3 engineers, 2 months → value 50
Initiative B: $5k, 2 engineers, 1 month → value 30
Initiative C: $8k, 4 engineers, 2 months → value 45
Initiative D: $3k, 1 engineer, 1 month → value 20

Budget limit: $15k
Headcount limit: 5 engineers
Time limit: 3 months

What initiatives should I pursue?
```

**Expected behavior**: Multi-resource constraints (money, headcount, time).

---

## 5. `solve_assignment_problem` - Task-Agent Matching

### Test 5.1: Cost Minimization Assignment
```
I need to assign 3 tasks to 3 team members:

Team:
- Alice: $50/hour
- Bob: $75/hour
- Carol: $60/hour

Tasks:
- Task 1: 2 hours
- Task 2: 3 hours
- Task 3: 1 hour

Each person can handle 1 task. Minimize total cost.
```

**Expected behavior**: Assigns cheapest resources: Alice→Task 2, Carol→Task 1, Bob→Task 3.

---

### Test 5.2: Skill Matching
```
Assign 4 development tasks to 3 engineers:

Engineers (max 2 tasks each):
- Alice: skills [Python, Docker, AWS]
- Bob: skills [React, TypeScript, Node.js]
- Charlie: skills [Python, React, PostgreSQL]

Tasks:
- Backend API: needs [Python, Docker]
- Frontend UI: needs [React, TypeScript]
- Database migration: needs [Python, PostgreSQL]
- Deployment: needs [Docker, AWS]

Who should do what?
```

**Expected behavior**: Skill-aware assignment respecting capacity limits.

---

### Test 5.3: Load Balancing
```
I have 9 jobs to distribute across 3 servers (each handles max 5 jobs):

Jobs: job_1, job_2, ..., job_9 (each takes 1 hour)

Distribute jobs evenly across servers to balance load.
```

**Expected behavior**: Uses `objective="balance_load"`, distributes 3 jobs per server.

---

## How to Use These Prompts

1. **Copy a prompt** from any section above
2. **Paste into Claude** (or any LLM with MCP access to this solver)
3. **Verify the LLM**:
   - Recognizes the problem type
   - Selects the correct MCP tool
   - Constructs a valid request
   - Parses and explains the solution

## Expected LLM Workflow

For each prompt, the LLM should:

1. ✅ **Understand** the problem domain
2. ✅ **Select** the appropriate tool (or `solve_constraint_model` for custom cases)
3. ✅ **Extract** parameters from the natural language description
4. ✅ **Call** the MCP tool with properly formatted arguments
5. ✅ **Parse** the response (status, solutions, explanation)
6. ✅ **Explain** the solution in user-friendly terms

## Success Criteria

- ✅ All prompts should work without modification
- ✅ LLM should never need to ask "what format" for parameters
- ✅ Coordinates as lists `[x, y]` should work (auto-coerced to tuples)
- ✅ Responses should be optimal or feasible
- ✅ LLM should provide clear, actionable explanations

## Troubleshooting

If a prompt fails:

1. Check the exact error message
2. Verify the LLM selected the right tool
3. Check parameter formats (especially lists vs tuples)
4. Ensure all required fields are provided
5. Look for typos in variable/location names

---

**Last Updated**: 2025-12-03
**Solver Version**: 0.3.0
**MCP Tools**: 5 (constraint_model, scheduling, routing, budget_allocation, assignment)
