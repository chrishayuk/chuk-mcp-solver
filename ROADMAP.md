# CHUK MCP Solver - Roadmap to the Ultimate Solver

**Vision:** Transform `chuk-mcp-solver` into the definitive optimization co-processor for LLMs—a single MCP server that can understand natural-language optimization problems, choose the right solving approach, build and refine models iteratively, and explain results in human-friendly terms.

**Guiding Principle:** "LLM's optimization co-processor", not just "OR-Tools wrapper"

---

## Current State (v0.1.3)

### ✅ Phases 1-3 Complete

You've built a solid foundation with:

- **Trust & Foundations (Phase 1)**
  - Structured observability and logging
  - Health checks and diagnostics
  - Problem hashing for deduplication
  - Infeasibility diagnosis
  - Deterministic solving (random seeds)
  - Solution metadata tracking

- **Developer Experience (Phase 2)**
  - Pre-solve model validation
  - Actionable error messages for LLMs
  - Smart typo detection ("Did you mean...?")
  - Three-level validation severity (ERROR, WARNING, INFO)
  - Detailed validation suggestions

- **Power & Performance (Phase 3)**
  - Solution caching with LRU + TTL
  - Partial solutions (best-so-far on timeout)
  - Search strategy hints (first-fail, random, etc.)
  - Cache statistics and hit rate tracking
  - Warm-start solution hints

- **Rich Constraint Set**
  - 9 constraint types: linear, all_different, element, table, implication, cumulative, circuit, reservoir, no_overlap
  - Multi-objective optimization (priority-based)
  - Parallel search workers
  - 170+ tests, 93% coverage

**Current Gap:** Still a single backend (OR-Tools CP-SAT), no sessions/iteration support, minimal explanation layer, no high-level problem-type abstractions.

---

## Phase 4: LLM-Native Problem Schemas (Next Up - 1-2 months)

**Goal:** Make it trivial for LLMs to solve common optimization problems without thinking in raw CP-SAT constraints.

### 4.1 Opinionated High-Level Schemas

Create small, domain-specific schemas that hide OR-Tools internals:

- **`SolveSchedulingProblemRequest`**
  - Input: tasks (id, duration, resources), workers (capacity, availability), dependencies
  - Output: Gantt chart data + explanations
  - Maps to: cumulative/no_overlap constraints + linear precedence

- **`SolveRoutingProblemRequest`**
  - Input: locations (id, coords), vehicles (capacity, start/end), time windows
  - Output: routes with distances/times
  - Maps to: circuit constraints + distance matrix

- **`SolveBudgetAllocationRequest`**
  - Input: items (cost, value, constraints), budgets (limits by category)
  - Output: optimal selection + sensitivity analysis
  - Maps to: knapsack-style linear constraints

- **`SolveAssignmentProblemRequest`**
  - Input: agents, tasks, costs/preferences matrix
  - Output: optimal assignment + unassigned items
  - Maps to: all_different + linear objective

**Deliverables:**
- [ ] Define 4 high-level request/response schemas in `models.py`
- [ ] Add converter functions: high-level → `SolveConstraintModelRequest`
- [ ] Create 4 new MCP tools: `solve_scheduling_problem`, `solve_routing_problem`, etc.
- [ ] Add examples in `examples/high_level/` demonstrating each
- [ ] Update README with "LLM Quick Start" section using these schemas

**Success Criteria:** LLM can solve "schedule 10 tasks across 3 workers" without knowing what a `cumulative` constraint is.

### 4.2 Enhanced Timeout & Status Handling

Make partial results and timeouts a first-class story:

**Request Model Additions:**
```python
class SearchConfig(BaseModel):
    max_time_ms: int = 120000  # Already exists
    return_best_if_timeout: bool = True  # Already exists
    num_solutions: int | None = None  # NEW: solution pools
    optimality_gap_threshold: float | None = None  # NEW: stop when gap < X%
```

**Response Status Enum:**
```python
class SolveStatus(str, Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    TIMEOUT_BEST = "timeout_best"  # NEW: distinguish timeout with best-so-far
    TIMEOUT_NO_SOLUTION = "timeout_no_solution"  # NEW
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    MODEL_ERROR = "error"
```

**Response Fields:**
```python
class SolveConstraintModelResponse(BaseModel):
    status: SolveStatus
    objective_value: float | None
    optimality_gap: float | None  # NEW: % gap from best bound
    solve_time_ms: int  # NEW: actual wall-clock time
    solutions: list[Solution]  # Can be > 1 if num_solutions set
    explanation: SolveExplanation
```

**Deliverables:**
- [ ] Add new status codes to `models.py`
- [ ] Add `optimality_gap`, `solve_time_ms` to response
- [ ] Support `num_solutions` for solution enumeration
- [ ] Add tests for timeout scenarios
- [ ] Update examples to show timeout handling

### 4.3 Diagnostic Tools

**New MCP Tools:**

1. **`solver_analyze_infeasibility(model)`**
   ```python
   {
     "conflicting_constraints": ["c1", "c5", "c8"],
     "minimal_infeasible_subset": [...],
     "explanation": "Constraints c1 (x <= 5) and c8 (x >= 10) are contradictory"
   }
   ```

2. **`solver_get_binding_constraints(model, solution)`**
   ```python
   {
     "tight_constraints": [
       {"id": "c3", "slack": 0, "description": "Budget limit fully utilized"}
     ],
     "loose_constraints": [
       {"id": "c7", "slack": 15.3, "description": "Time limit has 15.3 hours slack"}
     ]
   }
   ```

3. **`solver_explain_solution(model, solution)`**
   ```python
   {
     "summary": "Optimal schedule completes in 42 hours using 3 workers",
     "key_drivers": ["Budget constraint", "Worker 2 availability"],
     "key_variables": [
       {"id": "start_task_5", "value": 12, "impact": "Critical path bottleneck"}
     ],
     "tradeoffs": [
       "Adding 1 worker reduces time by 8 hours but increases cost by $500"
     ]
   }
   ```

**Implementation Strategy:**
- Use OR-Tools' constraint inspection APIs where possible
- Post-hoc analysis of slacks/shadows for the rest
- Start with simple template-based explanations, evolve to smarter analysis

**Deliverables:**
- [ ] Implement 3 diagnostic tools in `server.py`
- [ ] Add explanation generation logic in `solver/ortools/responses.py`
- [ ] Add tests for each diagnostic tool
- [ ] Create example: `examples/diagnostics_demo.py`

---

## Phase 5: Stateful Sessions & Iteration (2-4 months)

**Goal:** Enable iterative model refinement—build, solve, debug, tweak, re-solve—all tracked in a persistent session.

### 5.1 Session Management

**New Architecture:**
```python
# In-memory session store (later: chuk-artifacts backend)
sessions: dict[str, SolverSession] = {}

class SolverSession:
    id: str
    problem_type: str  # "scheduling", "routing", etc.
    model: SolveConstraintModelRequest | None
    solutions: list[SolveConstraintModelResponse]
    history: list[SessionEvent]  # model changes, solve attempts
    created_at: datetime
    metadata: dict[str, Any]
```

**New MCP Tools:**

1. **`solver_create_session(problem_type, metadata)`** → `session_id`
2. **`solver_set_model(session_id, model_json)`**
3. **`solver_get_model(session_id)`** → current model
4. **`solver_solve_session(session_id, solve_params)`** → solution
5. **`solver_get_session_history(session_id)`** → all events
6. **`solver_delete_session(session_id)`**

**LLM Workflow:**
```
1. CREATE session for "scheduling"
2. SET initial model (approximate, may have issues)
3. SOLVE → infeasible
4. GET model, tweak constraint c3
5. SOLVE → feasible
6. GET history → see all attempts
```

**Integration with chuk-artifacts (later):**
```
/solver/sessions/{session_id}/
  model.json          # current model
  solutions/
    1.json           # first solve
    2.json           # second solve
  history.jsonl      # event log
```

**Deliverables:**
- [ ] Define `SolverSession` model in `models.py`
- [ ] Implement in-memory session store in `server.py`
- [ ] Add 6 session management tools
- [ ] Add session history tracking (all model mutations + solve attempts)
- [ ] Add tests for session lifecycle
- [ ] Create example: `examples/iterative_refinement.py`
- [ ] (Later) Add chuk-artifacts persistence layer

### 5.2 Replay & Audit Trail

**Tool: `solver_get_session_history(session_id)`**

Returns structured timeline:
```python
{
  "events": [
    {
      "timestamp": "2025-12-02T10:00:00Z",
      "type": "session_created",
      "data": {"problem_type": "scheduling"}
    },
    {
      "timestamp": "2025-12-02T10:01:00Z",
      "type": "model_set",
      "data": {"num_variables": 50, "num_constraints": 120}
    },
    {
      "timestamp": "2025-12-02T10:02:00Z",
      "type": "solve_attempted",
      "data": {
        "status": "infeasible",
        "time_ms": 1234,
        "explanation": "..."
      }
    },
    {
      "timestamp": "2025-12-02T10:05:00Z",
      "type": "model_updated",
      "data": {
        "changes": [
          {"path": "constraints[5].params.rhs", "old": 100, "new": 120}
        ]
      }
    },
    ...
  ]
}
```

**Use Cases:**
- Debugging: "Why did solve #3 fail but #4 succeed?"
- Explanation: "Show me how the LLM refined this model"
- Visualization: Render timeline in chuk-mcp-remotion (future)

**Deliverables:**
- [ ] Add event logging to session store
- [ ] Implement `solver_get_session_history` tool
- [ ] Add model diff detection for `model_updated` events
- [ ] Add tests for history tracking
- [ ] Document session workflow in README

---

## Phase 6: Multi-Backend Architecture (3-6 months)

**Goal:** Move beyond CP-SAT. Route problems to the right solver backend automatically or explicitly.

### 6.1 Backend Abstraction

**New Architecture:**
```python
# src/chuk_mcp_solver/solver/provider.py
class SolverProvider(ABC):
    @abstractmethod
    async def solve_constraint_model(
        request: SolveConstraintModelRequest
    ) -> SolveConstraintModelResponse:
        ...

    @abstractmethod
    def can_handle(request: SolveConstraintModelRequest) -> bool:
        """Return True if this backend can solve this problem"""
        ...

# Backends:
# - CPSATProvider (current ORToolsSolver)
# - LPMIPProvider (OR-Tools linear solver)
# - SMTProvider (Z3, optional)
# - MetaheuristicProvider (simulated annealing, GA, etc.)
```

**Request Model Addition:**
```python
class SolveConstraintModelRequest(BaseModel):
    mode: SolveMode
    backend: Literal["auto", "cp_sat", "lp_mip", "smt", "metaheuristic"] = "auto"
    # ... rest of fields
```

**Backend Selection Logic (`"auto"`):**
```python
def select_backend(request: SolveConstraintModelRequest) -> str:
    # Pure boolean/SAT → cp_sat
    if all_vars_boolean(request.variables):
        return "cp_sat"

    # Linear with continuous vars, no integrality → lp_mip
    if is_linear_program(request):
        return "lp_mip"

    # Heavy logical constraints → smt
    if has_complex_logic(request):
        return "smt"

    # Default
    return "cp_sat"
```

**Deliverables:**
- [ ] Refactor current `ORToolsSolver` → `CPSATProvider`
- [ ] Create `SolverProvider` abstract base class
- [ ] Implement backend selection logic
- [ ] Add `backend` field to request schema
- [ ] Add tests for backend routing
- [ ] Update docs with backend comparison table

### 6.2 LP/MIP Backend

**Implementation:**
```python
# src/chuk_mcp_solver/solver/ortools/lp_mip.py
class LPMIPProvider(SolverProvider):
    """Linear/Mixed-Integer Programming via OR-Tools pywraplp"""

    def can_handle(request):
        return (
            all_constraints_linear(request.constraints) and
            no_global_constraints(request.constraints)
        )
```

**Use Cases:**
- Pure LP: continuous variables, linear objective, linear constraints
- MIP: mixed integer/continuous
- Better for large-scale linear problems than CP-SAT

**Deliverables:**
- [ ] Implement `LPMIPProvider` using OR-Tools' `pywraplp`
- [ ] Add LP/MIP-specific constraint builders
- [ ] Add tests comparing CP-SAT vs LP/MIP on linear problems
- [ ] Add example: `examples/linear_program.py`
- [ ] Document when to use LP/MIP vs CP-SAT

### 6.3 SMT Backend (Optional)

**Implementation:**
```python
# src/chuk_mcp_solver/solver/z3/smt.py (requires Z3 dependency)
class SMTProvider(SolverProvider):
    """SMT solving via Z3 for logical/bitvector problems"""
```

**Use Cases:**
- Heavy if-then-else logic
- Bitvector arithmetic
- Quantified formulas
- Program verification-style constraints

**Deliverables:**
- [ ] Add `z3-solver` as optional dependency
- [ ] Implement `SMTProvider`
- [ ] Add tests for SMT-specific problems
- [ ] Add example: `examples/smt_logic_puzzle.py`
- [ ] Document SMT use cases

### 6.4 Metaheuristic Backend (Future)

For black-box or highly nonlinear objectives:
- Simulated annealing
- Genetic algorithms
- Particle swarm optimization

**Deliverables:**
- [ ] Research lightweight metaheuristic libraries
- [ ] Implement `MetaheuristicProvider` (basic)
- [ ] Add tests for nonlinear problems
- [ ] Document when metaheuristics are appropriate

---

## Phase 7: LLM Integration & Prompts (1-2 months, parallel with Phase 4-6)

**Goal:** Official prompt templates and best practices for LLMs using the solver.

### 7.1 Prompt Templates

Create `prompts/` directory with domain-specific templates:

**`prompts/scheduling.md`:**
```markdown
# Scheduling Problem Prompt

You are using CHUK MCP Solver to solve scheduling problems.

## Your Task
1. Parse user's scheduling request into structured data
2. Convert to `SolveSchedulingProblemRequest` JSON
3. Call `solve_scheduling_problem` tool
4. Interpret results and explain to user

## Schema
{...JSON schema for SolveSchedulingProblemRequest...}

## Validation Checklist
- [ ] All tasks have positive durations
- [ ] No resource capacity is negative
- [ ] Dependencies form a DAG (no cycles)
- [ ] At least one objective is defined
- [ ] If conflicting constraints detected, ask user for clarification

## Examples

### Example 1: Simple Project Schedule
User: "Schedule 5 tasks with these dependencies: A before B, B before C..."

Thought: Extract tasks, durations, dependencies...

JSON:
{
  "tasks": [
    {"id": "A", "duration": 3},
    ...
  ],
  "dependencies": [["A", "B"], ...],
  "objective": {"type": "minimize_makespan"}
}

MCP Call: solve_scheduling_problem(...)

Response: "Optimal schedule completes in 12 hours: Task A starts at 0..."

### Example 2: Resource-Constrained
...

## Common Pitfalls
- Forgetting to define resource capacities
- Using negative durations
- Creating dependency cycles
```

**Other Templates:**
- `prompts/routing.md`
- `prompts/budgeting.md`
- `prompts/assignment.md`

**Deliverables:**
- [ ] Create `prompts/` directory
- [ ] Write 4 domain-specific prompt templates
- [ ] Include 2-3 end-to-end examples per template
- [ ] Add "LLM Best Practices" section to README
- [ ] Consider: System prompt snippet for Claude Desktop config

### 7.2 Validation Checklists

Embed LLM-friendly checklists in documentation:

**Pre-Solve Checklist (for LLMs):**
```markdown
Before calling the solver, verify:
1. [ ] All variable domains are valid (lower <= upper)
2. [ ] All referenced variables exist
3. [ ] Constraint parameters are within valid ranges
4. [ ] If optimization, objective is defined
5. [ ] Time/resource constraints don't contradict
6. [ ] If user intent is ambiguous, ask clarifying question
```

**Post-Solve Checklist:**
```markdown
When presenting solution to user:
1. [ ] Explain status (optimal/feasible/infeasible)
2. [ ] If infeasible, explain why (use solver_analyze_infeasibility)
3. [ ] Highlight key tradeoffs
4. [ ] Suggest improvements if partial solution
5. [ ] Ask if user wants to explore alternatives
```

**Deliverables:**
- [ ] Add "LLM Checklists" section to README
- [ ] Include checklist in each prompt template
- [ ] Add validation reminders to tool descriptions

---

## Phase 8: Multi-Objective & Scenario Analysis (2-3 months)

**Goal:** First-class support for multiple objectives and "what-if" analysis.

### 8.1 Enhanced Multi-Objective Support

**Current State:**
- Priority-based multi-objective (lexicographic)
- Weighted sum (manual by user)

**Enhancements:**

1. **Automatic Scalarization:**
```python
class ObjectiveStrategy(str, Enum):
    LEXICOGRAPHIC = "lexicographic"  # Existing: priority order
    WEIGHTED_SUM = "weighted_sum"    # New: w1*obj1 + w2*obj2
    EPSILON_CONSTRAINT = "epsilon"   # New: optimize obj1, constrain obj2

class MultiObjectiveConfig(BaseModel):
    strategy: ObjectiveStrategy = "lexicographic"
    weights: dict[str, float] | None = None  # For weighted_sum
    epsilon_constraints: dict[str, tuple[float, float]] | None = None
```

2. **Objective Component Breakdown:**
```python
class SolveConstraintModelResponse(BaseModel):
    # ... existing fields ...
    objective_components: dict[str, float] | None = None
    # e.g., {"cost": 1200, "time": 45, "emissions": 8.3}
```

**Deliverables:**
- [ ] Add `ObjectiveStrategy` enum to `models.py`
- [ ] Implement weighted sum and epsilon constraint methods
- [ ] Return objective component breakdown in response
- [ ] Add tests for different objective strategies
- [ ] Add example: `examples/multi_objective_strategies.py`

### 8.2 Scenario Comparison Tool

**New Tool: `solver_run_scenarios(base_model, scenarios)`**

```python
{
  "base_model": {...},
  "scenarios": [
    {
      "id": "add_worker",
      "description": "Add 1 worker",
      "modifications": [
        {"path": "resources[0].capacity", "operation": "add", "value": 1}
      ]
    },
    {
      "id": "reduce_deadline",
      "description": "Tighten deadline by 20%",
      "modifications": [
        {"path": "constraints[deadline].params.rhs", "operation": "multiply", "value": 0.8}
      ]
    }
  ]
}
```

**Response:**
```python
{
  "baseline": {
    "objective": 100,
    "status": "optimal",
    "solve_time_ms": 234
  },
  "scenarios": [
    {
      "id": "add_worker",
      "objective": 85,  # 15% improvement
      "status": "optimal",
      "delta_vs_baseline": -15,
      "key_changes": "Makespan reduced by 3 hours"
    },
    {
      "id": "reduce_deadline",
      "objective": null,
      "status": "infeasible",
      "explanation": "Deadline too tight, no feasible solution"
    }
  ],
  "comparison_table": "..." # formatted table
}
```

**Use Cases:**
- "What if we add another server?"
- "What's the impact of reducing the budget by 10%?"
- Sensitivity analysis for LLM-driven decision making

**Deliverables:**
- [ ] Implement `solver_run_scenarios` tool in `server.py`
- [ ] Add JSON Patch-style model modification logic
- [ ] Generate comparison table/summary
- [ ] Add tests for scenario analysis
- [ ] Add example: `examples/scenario_comparison.py`

---

## Phase 9: Visualization & CHUK Stack Integration (2-4 months)

**Goal:** Rich visual outputs and deep integration with other CHUK MCPs.

### 9.1 Solution Export

**New Tool: `solver_export_solution(session_id, format, path)`**

**Supported Formats:**
- `"json"`: Raw solution JSON
- `"csv"`: Variable assignments as CSV
- `"gantt_json"`: Gantt chart data (for scheduling)
- `"route_geojson"`: Route data with coordinates (for routing)
- `"markdown"`: Human-readable report

**Integration with chuk-virtual-fs:**
```python
# LLM workflow:
1. solver_export_solution(session_id, "gantt_json", "/solver/schedule.json")
2. fs_read("/solver/schedule.json")
3. Pass to chuk-mcp-remotion for video rendering
```

**Deliverables:**
- [ ] Implement `solver_export_solution` tool
- [ ] Add export format converters
- [ ] Add Gantt chart JSON schema
- [ ] Add route GeoJSON export
- [ ] Add Markdown report template
- [ ] Add tests for each export format
- [ ] Document integration with chuk-virtual-fs

### 9.2 Visualization via chuk-mcp-remotion

**New Tool: `solver_render_schedule(session_id, renderer)`**

**For Scheduling Problems:**
- Generate animated Gantt chart
- Show resource utilization over time
- Highlight critical path

**For Routing Problems:**
- Animate vehicle routes on a map
- Show delivery sequence

**Example:**
```python
# In chuk-mcp-remotion (future):
render_gantt_chart(
    tasks=[
        {"id": "A", "start": 0, "duration": 3, "worker": "Alice"},
        ...
    ],
    timeline_duration_sec=10,
    output_path="/videos/schedule.mp4"
)
```

**Deliverables:**
- [ ] Define data schemas for visualization (Gantt, route map)
- [ ] Add `solver_render_schedule` tool (delegates to remotion)
- [ ] Coordinate with chuk-mcp-remotion for templates
- [ ] Add example: `examples/visualize_schedule.py`
- [ ] Document visualization workflow in README

### 9.3 Jupyter Notebook Export

**New Tool: `solver_export_notebook(session_id, path)`**

Generate a Jupyter notebook with:
- Problem description
- Model specification
- Solve code (runnable)
- Solution visualization
- Sensitivity analysis

**Use Case:** LLM helps user solve a problem, then exports reproducible notebook for sharing/documentation.

**Deliverables:**
- [ ] Implement notebook template generator
- [ ] Add `solver_export_notebook` tool
- [ ] Add tests for notebook generation
- [ ] Add example notebook to `examples/`

---

## Phase 10: Performance, Scale & Production Ops (3-6 months)

**Goal:** Production-grade performance characteristics and operational transparency.

### 10.1 Benchmarks & Performance Documentation

**Create `PERFORMANCE.md`:**
```markdown
# Performance Benchmarks

## Methodology
- Hardware: Apple M2 Pro, 16GB RAM
- OR-Tools version: 9.10
- CP-SAT settings: default + 4 workers

## Results

### Scheduling Problems
| Problem | Variables | Constraints | Solve Time | Memory |
|---------|-----------|-------------|------------|--------|
| 100 tasks, 5 workers | 500 | 1200 | 1.2s | 45MB |
| 500 tasks, 10 workers | 2500 | 6000 | 8.4s | 180MB |
| 1000 tasks, 20 workers | 5000 | 12000 | 45s | 420MB |

### Routing Problems (TSP)
| Cities | Variables | Solve Time | Optimal Distance |
|--------|-----------|------------|------------------|
| 10 | 90 | 0.1s | 2834 km |
| 50 | 2450 | 2.3s | 8127 km |
| 100 | 9900 | 18s | 11456 km |

### Knapsack Problems
...

## Soft Limits (Interactive Use)
- **Recommended:** < 1000 variables, < 5000 constraints
- **Acceptable:** < 5000 variables, < 20000 constraints
- **Advanced:** Use parallel workers, warm starts, or decomposition

## Tips for Large Problems
1. Enable `num_search_workers: 4` or more
2. Use `max_time_ms` with `return_partial_solution: true`
3. Provide warm-start hints if available
4. Consider domain reduction pre-processing
5. Break into subproblems if possible
```

**Deliverables:**
- [ ] Create `PERFORMANCE.md` with benchmarks
- [ ] Add benchmark scripts to `benchmarks/` directory
- [ ] Run benchmarks on standard hardware
- [ ] Document soft limits and scaling tips
- [ ] Add CI job to track performance regression (optional)

### 10.2 Advanced Caching

**Enhancements to Current Cache:**

1. **Persistent Cache (optional):**
```python
# Current: in-memory only
# Enhancement: Redis or SQLite backend

class CacheBackend(ABC):
    @abstractmethod
    async def get(key: str) -> CacheEntry | None: ...
    @abstractmethod
    async def set(key: str, value: CacheEntry, ttl: int): ...

class RedisCacheBackend(CacheBackend):
    # For distributed deployments
    ...

class SQLiteCacheBackend(CacheBackend):
    # For local persistence
    ...
```

2. **Semantic Caching:**
```python
# Problem variations that produce same solution:
# - Variable order changes
# - Constraint order changes
# - Equivalent constraint formulations

def normalize_problem(request: SolveConstraintModelRequest) -> str:
    """Canonicalize problem for semantic cache matching"""
    # Sort variables, constraints, etc.
    ...
```

3. **Cache Warming:**
```python
# New tool: solver_warm_cache(common_problems)
# Pre-solve common problem templates to populate cache
```

**Deliverables:**
- [ ] Design pluggable cache backend interface
- [ ] Implement SQLite cache backend (optional)
- [ ] Add problem normalization for semantic caching
- [ ] Add `solver_warm_cache` tool
- [ ] Add cache eviction strategies (LRU, LFU, TTL)
- [ ] Add tests for cache backends
- [ ] Document caching strategies in README

### 10.3 Observability & Metrics

**Enhanced Logging:**
```python
# Structured logs for:
- Solve attempts (request_id, status, time, gap)
- Cache hits/misses
- Backend selection decisions
- Validation failures
- Session lifecycle events
```

**Metrics Endpoint (if running as HTTP server):**
```json
GET /metrics
{
  "total_solves": 12453,
  "cache_hit_rate": 0.34,
  "avg_solve_time_ms": 1234,
  "backend_usage": {
    "cp_sat": 10234,
    "lp_mip": 2100,
    "smt": 119
  },
  "status_distribution": {
    "optimal": 8234,
    "feasible": 2100,
    "infeasible": 1500,
    "timeout": 619
  }
}
```

**Deliverables:**
- [ ] Enhance observability module with structured events
- [ ] Add metrics collection (in-memory)
- [ ] Add `solver_get_metrics` MCP tool
- [ ] Add `/metrics` HTTP endpoint (if server mode)
- [ ] Add Prometheus exporter (optional)
- [ ] Document observability setup in README

---

## Phase 11: Advanced Features (6-12 months, ongoing)

**Goal:** Cutting-edge solver capabilities for power users.

### 11.1 Solution Enumeration

**Current:** Return 1 optimal or N best solutions
**Enhancement:** Find diverse solutions, Pareto frontiers

**New Search Option:**
```python
class SearchConfig(BaseModel):
    # ... existing fields ...
    diverse_solutions: bool = False
    diversity_metric: Literal["hamming", "euclidean"] = "hamming"
    min_diversity_threshold: float = 0.2
```

**Use Case:** "Show me 5 different ways to schedule this project, each at least 20% different from the others"

**Deliverables:**
- [ ] Implement solution pool with diversity filtering
- [ ] Add diversity metric calculation
- [ ] Add tests for diverse solution generation
- [ ] Add example: `examples/diverse_solutions.py`

### 11.2 Symmetry Breaking

Automatically detect and break symmetries:
- Identical workers
- Interchangeable tasks
- Symmetric constraints

**Deliverables:**
- [ ] Research OR-Tools symmetry breaking
- [ ] Add automatic symmetry detection
- [ ] Add symmetry-breaking constraints
- [ ] Measure performance improvement

### 11.3 Decomposition Strategies

For very large problems:
- Temporal decomposition (schedule in chunks)
- Spatial decomposition (partition graph)
- Hierarchical decomposition (coarse → fine)

**Deliverables:**
- [ ] Research decomposition heuristics
- [ ] Implement temporal decomposition for scheduling
- [ ] Add tests for decomposition
- [ ] Document when to use decomposition

### 11.4 Custom Search Heuristics

Allow users to define custom variable/value selection:

```python
class SearchConfig(BaseModel):
    # ... existing fields ...
    custom_heuristic: dict[str, Any] | None = None
    # e.g., {"type": "phase", "var_order": ["critical_tasks", "rest"]}
```

**Deliverables:**
- [ ] Design custom heuristic API
- [ ] Implement phase-based search
- [ ] Add tests for custom heuristics
- [ ] Document heuristic design in advanced guide

### 11.5 Explanation Generation (Advanced)

**Current:** Template-based explanations
**Enhancement:** LLM-powered explanations

**Workflow:**
```python
# After solving:
1. Extract solution facts (binding constraints, key variables)
2. Pass to small LLM (GPT-4o-mini, Claude Haiku) with prompt:
   "Explain this optimization result in plain language..."
3. Return enriched explanation
```

**Deliverables:**
- [ ] Add optional LLM-powered explanation mode
- [ ] Design explanation prompt template
- [ ] Add configuration for explanation model
- [ ] Add tests comparing template vs LLM explanations
- [ ] Document explanation options

---

## Success Metrics

How do we know we've achieved "Ultimate Solver MCP" status?

### Phase 4-5 (LLM-Native + Sessions):
- [ ] LLM can solve 4 common problem types without raw CP-SAT knowledge
- [ ] LLM can iteratively refine infeasible model to feasible in < 3 attempts
- [ ] 90%+ of timeout scenarios return useful partial solutions

### Phase 6 (Multi-Backend):
- [ ] Backend auto-selection chooses correctly 95%+ of the time
- [ ] LP/MIP backend outperforms CP-SAT on linear problems by 10x+
- [ ] At least 3 production backends available (CP-SAT, LP/MIP, +1)

### Phase 7 (LLM Integration):
- [ ] Prompt templates exist for top 4 problem types
- [ ] Validation checklists reduce invalid requests by 80%+
- [ ] Example-driven docs enable LLM to solve new problem types

### Phase 8 (Multi-Objective):
- [ ] Weighted sum, lexicographic, epsilon methods all supported
- [ ] Scenario comparison runs 5+ scenarios in < 10 seconds
- [ ] Objective breakdown helps users understand tradeoffs

### Phase 9 (Visualization):
- [ ] Gantt charts auto-generated for all scheduling solutions
- [ ] Route maps auto-generated for all routing solutions
- [ ] Jupyter notebook export works end-to-end

### Phase 10 (Performance):
- [ ] Published benchmarks for 10+ problem types
- [ ] Soft limits documented and tested
- [ ] Cache hit rate > 30% in production workloads
- [ ] Metrics endpoint tracks all key stats

### Phase 11 (Advanced):
- [ ] Diverse solution enumeration available
- [ ] Symmetry breaking improves solve time by 30%+ on symmetric problems
- [ ] Decomposition handles 10,000+ variable problems

---

## Concrete Next Steps (Month 1)

Based on priority and dependencies, here's what to tackle first:

### Week 1-2: Phase 4.1 - High-Level Schemas
1. Define `SolveSchedulingProblemRequest` schema
2. Implement converter to `SolveConstraintModelRequest`
3. Add `solve_scheduling_problem` MCP tool
4. Create example: `examples/high_level/simple_schedule.py`
5. Test with LLM (Claude Desktop)

### Week 3-4: Phase 4.2 & 4.3 - Enhanced Status + Diagnostics
1. Add `TIMEOUT_BEST` and `TIMEOUT_NO_SOLUTION` statuses
2. Add `optimality_gap` and `solve_time_ms` to response
3. Implement `solver_analyze_infeasibility` tool
4. Implement `solver_explain_solution` tool (basic)
5. Add tests and examples

### Month 2: Complete Phase 4
1. Add remaining 3 high-level schemas (routing, budgeting, assignment)
2. Polish diagnostic tools
3. Update README with "LLM Quick Start"
4. Create prompt templates (basic versions)

### Month 3: Phase 5.1 - Sessions
1. Implement in-memory session store
2. Add 6 session management tools
3. Add session history tracking
4. Create iterative refinement example
5. Test full LLM workflow

---

## Open Questions & Decisions Needed

1. **Multi-Backend Priority:** Which second backend is most valuable? LP/MIP (best for scale) vs SMT (best for logic)?
   - **Recommendation:** LP/MIP first (broader applicability)

2. **Session Persistence:** In-memory first, or chuk-artifacts from day 1?
   - **Recommendation:** In-memory first (simpler), add persistence later

3. **Explanation Strategy:** Template-based or LLM-powered?
   - **Recommendation:** Start template-based, add optional LLM mode later

4. **Visualization Ownership:** Build in solver MCP or delegate to remotion/stage?
   - **Recommendation:** Delegate to remotion, solver just exports data

5. **Caching Backend:** When to add persistent cache?
   - **Recommendation:** After Phase 5, when sessions make persistence more critical

---

## Summary

This roadmap transforms `chuk-mcp-solver` from "excellent OR-Tools wrapper" to "LLM's ultimate optimization co-processor" over ~12 months:

- **Now-3 months (Phases 4-5):** LLM-native interfaces, sessions, iteration
- **3-6 months (Phase 6):** Multi-backend architecture (LP/MIP, SMT)
- **6-9 months (Phases 7-8):** LLM integration polish, multi-objective, scenarios
- **9-12 months (Phases 9-10):** Visualization, performance, production ops
- **12+ months (Phase 11):** Advanced features (diversity, symmetry, decomposition)

Each phase builds on the last. Phases 4-5 are highest ROI for LLM usability. Phase 6 unlocks scale. Phases 7-10 are production polish. Phase 11 is power-user territory.

**The key insight:** You're not just building a solver—you're building a *reasoning augmentation layer* for LLMs. Every feature should ask: "Does this make it easier for an LLM to help a human solve hard problems?"

Let's start with Phase 4 and make scheduling problems trivial for LLMs. 🚀
