# Quick LLM Test Prompts - One-Liners

Copy-paste these into Claude to quickly test each tool.

---

## 1️⃣ Budget Allocation - Knapsack
```
Use solve_budget_allocation: I need to pack a 10kg backpack. Items: Tent (5kg, value 100), Sleeping bag (3kg, value 80), Food (2kg, value 60), Water (4kg, value 70), First aid (1kg, value 50). What should I take to maximize value?
```

---

## 2️⃣ Scheduling - Simple Project
```
Use solve_scheduling_problem: I'm deploying software: Build (2 hours) → Test (3 hours) → Stage (1 hour) → Production (1 hour). Each step depends on the previous. How long does it take total and when does each start?
```

---

## 3️⃣ Routing - Multi-Vehicle VRP ✨
```
Use solve_routing_problem: I have 2 delivery vans (each can carry 30 items) and need to deliver to 4 stores:
  - Store A at (10, 5): needs 12 items
  - Store B at (5, 10): needs 15 items
  - Store C at (15, 15): needs 18 items
  - Store D at (8, 3): needs 10 items

Starting from depot at (0, 0), what's the best route for each van?
```

---

## 4️⃣ Budget Allocation - Project Selection
```
Use solve_budget_allocation: I have $10k budget. Projects: A ($5k, value $12k), B ($3k, value $7k), C ($4k, value $9k), D ($6k, value $11k). Which should I fund to maximize value?
```

---

## 5️⃣ Assignment - Task Matching
```
Use solve_assignment_problem: Assign 3 tasks to 3 people: Alice ($50/hr), Bob ($75/hr), Carol ($60/hr). Tasks: 1 (2hrs), 2 (3hrs), 3 (1hr). Each person does 1 task. Minimize cost.
```

---

## Expected Results Summary

| Tool | Prompt | Expected Outcome |
|------|--------|------------------|
| **Budget Allocation** | Knapsack packing | Sleeping bag + Food + Water + First aid = 10kg, value 260 |
| **Scheduling** | Software deployment | 7 hours total: Build(0-2), Test(2-5), Stage(5-6), Prod(6-7) |
| **Routing** | 2-van delivery | Van 1: 2 stores, Van 2: 2 stores, total ~69 distance |
| **Budget Allocation** | Project selection | Fund A + C = $9k cost, $21k value |
| **Assignment** | Task assignment | Alice→Task2 ($150), Carol→Task1 ($120), Bob→Task3 ($75) |

---

## All Tests in One Prompt

Want to test everything at once? Try this:

```
I need help with 5 optimization problems:

1. PACKING: I have a 10kg backpack. Items: Tent (5kg, $100 value), Sleeping bag (3kg, $80), Food (2kg, $60), Water (4kg, $70), First aid (1kg, $50). What maximizes value?

2. SCHEDULING: Software deployment tasks: Build (2hrs) → Test (3hrs) → Stage (1hr) → Production (1hr). Each depends on previous. When does each start/end?

3. ROUTING: 2 vans (30 capacity each) from depot (0,0) to: Store A (10,5) needs 12, Store B (5,10) needs 15, Store C (15,15) needs 18, Store D (8,3) needs 10. Best routes?

4. BUDGET: $10k for projects: A ($5k, $12k value), B ($3k, $7k value), C ($4k, $9k value), D ($6k, $11k value). Which to fund?

5. ASSIGNMENT: Assign Task1 (2hrs), Task2 (3hrs), Task3 (1hr) to Alice ($50/hr), Bob ($75/hr), Carol ($60/hr). One task each. Minimize cost.

Solve all 5.
```

The LLM should recognize each problem type and use the appropriate tool!

---

**Pro tip**: After testing, ask Claude "Which tools did you use?" to verify it selected the right ones.
