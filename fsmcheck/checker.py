"""The three checks: safety, deadlock and liveness.

Every check returns :class:`~fsmcheck.graph.CheckResult` with one of:

* ``"pass"``    -- exploration finished and no violation exists;
* ``"fail"``    -- a counterexample was found *and* independently replayed;
* ``"unknown"`` -- a resource budget stopped exploration first.

Counterexamples for safety/deadlock are the shortest possible paths, ties
broken by lexicographically smallest transition-id sequence.  Liveness
counterexamples are a finite prefix followed by a non-empty closed loop that
never visits an accepting state.
"""

from __future__ import annotations

from .graph import (
    Counterexample, Trace, CheckResult, Exploration,
    explore, verify_against_trace,
    PASS, FAIL, UNKNOWN,
)
from .model import Model, evaluate_bool


def _index(model: Model) -> dict[str, int]:
    return {name: i for i, name in enumerate(model.variable_names)}


def _empty_exploration(model: Model) -> Exploration:
    """An exploration containing only the initial state and no edges."""
    initial = model.initial
    return Exploration(
        model=model,
        adjacency={initial: []},
        order=[initial],
        parent={initial: (None, None)},
        states_explored=1,
        edges_explored=0,
        complete=True,
    )


def _result(kind, status, exploration, *, budgets, counterexample=None,
            message=""):
    return CheckResult(
        status=status,
        kind=kind,
        counterexample=counterexample,
        states_explored=exploration.states_explored,
        edges_explored=exploration.edges_explored,
        state_budget=budgets[0],
        edge_budget=budgets[1],
        message=message,
    )


def _unknown(kind, exploration, budgets):
    return _result(kind, UNKNOWN, exploration, budgets=budgets,
                   message=exploration.budget_reason)


# ---------------------------------------------------------------------- #
# Safety
# ---------------------------------------------------------------------- #
def check_safety(model: Model, predicate, *,
                 state_budget=10_000, edge_budget=50_000) -> CheckResult:
    """Check that *predicate* holds in every reachable state.

    A state where it is false is a safety violation.
    """
    expr = model.prepare_predicate(predicate, "safety predicate")
    index = _index(model)
    budgets = (state_budget, edge_budget)

    # The initial state can fail via an empty path; that never needs any
    # exploration budget and must be reported even when budgets are tiny.
    initial = model.initial
    if not evaluate_bool(expr, initial, index):
        ce = Counterexample("safety", Trace([], [initial]), None,
                            f"predicate false at {model.state_dict(initial)}")
        verify_against_trace(model, ce, safety_predicate=expr)
        return _result("safety", FAIL,
                       _empty_exploration(model), budgets=budgets,
                       counterexample=ce)

    def stop_at(state, edges):
        return "safety violation" if not evaluate_bool(expr, state, index) \
            else None

    exp = explore(model, state_budget=state_budget,
                  edge_budget=edge_budget, stop_at=stop_at)
    if exp.budget_reason == "safety violation":
        bad = exp.order[-1]
        trace = exp.path_to(bad)
        ce = Counterexample("safety", trace, None,
                            f"predicate false at {model.state_dict(bad)}")
        verify_against_trace(model, ce, safety_predicate=expr)
        return _result("safety", FAIL, exp, budgets=budgets,
                       counterexample=ce)
    if not exp.complete:
        return _unknown("safety", exp, budgets)
    return _result("safety", PASS, exp, budgets=budgets)


# ---------------------------------------------------------------------- #
# Deadlock
# ---------------------------------------------------------------------- #
def check_deadlock(model: Model, *,
                   state_budget=10_000, edge_budget=50_000) -> CheckResult:
    """A deadlock is a reachable state with no enabled transition that is
    not declared terminal."""
    budgets = (state_budget, edge_budget)

    # Empty-path deadlock in the initial state (independent of budgets).
    initial = model.initial
    if next(model.successors(initial), None) is None \
            and not model.is_terminal(initial):
        ce = Counterexample("deadlock", Trace([], [initial]), None,
                            f"no enabled transition at "
                            f"{model.state_dict(initial)}")
        verify_against_trace(model, ce)
        return _result("deadlock", FAIL,
                       _empty_exploration(model), budgets=budgets,
                       counterexample=ce)

    def stop_at(state, edges):
        # ``edges`` were already enumerated (and budgeted) by explore(); an
        # empty list means no enabled in-bounds transition.
        if not edges and not model.is_terminal(state):
            return "deadlock"
        return None

    exp = explore(model, state_budget=state_budget,
                  edge_budget=edge_budget, stop_at=stop_at)
    if exp.budget_reason == "deadlock":
        stuck = exp.order[-1]
        trace = exp.path_to(stuck)
        ce = Counterexample("deadlock", trace, None,
                            f"no enabled transition at "
                            f"{model.state_dict(stuck)}")
        verify_against_trace(model, ce)
        return _result("deadlock", FAIL, exp, budgets=budgets,
                       counterexample=ce)
    if not exp.complete:
        return _unknown("deadlock", exp, budgets)
    return _result("deadlock", PASS, exp, budgets=budgets)


# ---------------------------------------------------------------------- #
# Liveness: every infinite execution visits accept infinitely often
# ---------------------------------------------------------------------- #
def check_liveness(model: Model, *,
                   state_budget=10_000, edge_budget=50_000) -> CheckResult:
    """Without fairness assumptions, a violation exists exactly when some
    reachable state lies on a cycle entirely inside the subgraph induced by
    the non-accepting states: from there an execution can loop forever and
    never accept again.  Finite executions (dead ends) do not violate."""
    kind = "liveness"
    budgets = (state_budget, edge_budget)
    exp = explore(model, state_budget=state_budget,
                  edge_budget=edge_budget)
    if not exp.complete:
        return _unknown(kind, exp, budgets)

    bad_components = _cyclic_components(model, exp)
    if not bad_components:
        return _result(kind, PASS, exp, budgets=budgets)

    candidates = []
    for component in bad_components:
        entry, prefix = _best_entry(model, exp, component)
        loop = _closed_walk(model, exp, component, entry)
        candidates.append((prefix.length, list(prefix.transitions),
                           loop.length, list(loop.transitions),
                           entry, prefix, loop))
    candidates.sort(key=lambda c: (c[0], c[1], c[2], c[3]))
    _, _, _, _, _, prefix, loop = candidates[0]

    ce = Counterexample(
        kind, prefix, loop,
        "prefix reaches a cyclic SCC of non-accepting states; "
        "repeating the loop never accepts again",
    )
    verify_against_trace(model, ce)
    return _result(kind, FAIL, exp, budgets=budgets, counterexample=ce)


def _cyclic_components(model: Model, exp: Exploration):
    """Kosaraju SCCs of the subgraph induced by non-accepting reachable
    states; keep only the cyclic ones (self-loop, or size >= 2)."""
    nodes = [s for s in exp.adjacency if not model.is_accept(s)]
    node_set = set(nodes)

    def internal_out(s):
        return [(tid, d) for tid, d in exp.adjacency[s] if d in node_set]

    # First pass: finishing order on the induced graph.
    seen: set = set()
    finish_order: list = []
    for start in nodes:
        if start in seen:
            continue
        stack = [(start, iter(internal_out(start)))]
        seen.add(start)
        while stack:
            state, it = stack[-1]
            advanced = False
            for _, dst in it:
                if dst not in seen:
                    seen.add(dst)
                    stack.append((dst, iter(internal_out(dst))))
                    advanced = True
                    break
            if not advanced:
                finish_order.append(state)
                stack.pop()

    # Reverse edges inside the induced graph.
    reverse: dict = {s: [] for s in nodes}
    for s in nodes:
        for tid, d in internal_out(s):
            reverse[d].append((tid, s))

    # Second pass: assign components on the reverse graph in reverse
    # finishing order.
    assigned: set = set()
    components: list[list] = []
    for start in reversed(finish_order):
        if start in assigned:
            continue
        comp = []
        stack = [start]
        assigned.add(start)
        while stack:
            s = stack.pop()
            comp.append(s)
            for _, d in reverse[s]:
                if d not in assigned:
                    assigned.add(d)
                    stack.append(d)
        components.append(comp)

    cyclic = []
    for comp in components:
        if len(comp) >= 2:  # strongly connected + >1 node => has a cycle
            cyclic.append(comp)
            continue
        (only,) = comp
        if any(d == only for _, d in internal_out(only)):
            cyclic.append(comp)
    return cyclic


def _best_entry(model: Model, exp: Exploration, component):
    """Choose the component entry whose shortest global path from the
    initial state is lexicographically smallest (it may pass through
    accepting states)."""
    best = None
    for state in component:
        trace = exp.path_to(state)
        key = (trace.length, list(trace.transitions))
        if best is None or key < best[0]:
            best = (key, state, trace)
    return best[1], best[2]


def _closed_walk(model: Model, exp: Exploration, component, entry) -> Trace:
    """Shortest lexicographically smallest non-empty walk from *entry* back
    to itself, staying inside *component*.

    Tried as ``first edge`` + shortest BFS path back to entry, with first
    edges enumerated in transition-id order.
    """
    members = set(component)
    candidates = []
    for first_id, first_dst in exp.adjacency[entry]:
        if first_dst not in members:
            continue
        back = _shortest_path(exp, members, first_dst, entry)
        if back is None:
            continue
        ids = [first_id] + back.transitions
        states = [entry, first_dst] + back.states[1:]
        candidates.append((len(ids), ids, states))
    candidates.sort(key=lambda c: (c[0], c[1]))
    _, ids, states = candidates[0]
    return Trace(ids, states)


def _shortest_path(exp: Exploration, allowed, source, target):
    """BFS within *allowed* states from *source* to *target* (zero edges
    allowed when source == target); lex-smallest among shortest paths."""
    if source == target:
        return Trace([], [source])
    parent = {source: (None, None)}
    queue = [source]
    head = 0
    while head < len(queue):
        state = queue[head]
        head += 1
        for tid, dst in exp.adjacency[state]:
            if dst not in allowed or dst in parent:
                continue
            parent[dst] = (tid, state)
            if dst == target:
                queue.append(dst)
                head = len(queue)
                break
            queue.append(dst)
    if target not in parent:
        return None
    ids: list[str] = []
    states: list = []
    cur = target
    while cur is not None:
        tid, par = parent[cur]
        states.append(cur)
        if tid is not None:
            ids.append(tid)
        cur = par
    states.reverse()
    ids.reverse()
    return Trace(ids, states)
