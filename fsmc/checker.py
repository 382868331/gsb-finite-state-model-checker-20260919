"""Explicit-state finite model checking: safety, deadlocks, liveness.

Results are always one of:

* :data:`Status.PASS`    -- exploration completed, property holds
* :data:`Status.FAIL`    -- a replayable counterexample was found
* :data:`Status.UNKNOWN` -- a state/edge budget was exhausted first

Counterexamples carry step-by-step states and are independently replayed
from the initial state: guard evaluation, simultaneous assignment and the
end-point violation are all recomputed rather than trusted from the search.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from . import astx
from .model import Model

__all__ = [
    "Status",
    "Budgets",
    "Counterexample",
    "Step",
    "Result",
    "ReplayError",
    "check_safety",
    "check_deadlock",
    "check_liveness",
]

DEFAULT_STATE_BUDGET = 10_000
DEFAULT_EDGE_BUDGET = 50_000


class Status(Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Budgets:
    """Exploration limits.

    ``max_states`` counts distinct discovered states (including the initial
    state); ``max_edges`` counts fired transitions. Exhausting either one
    yields :data:`Status.UNKNOWN` unless a replayable counterexample was
    already found.
    """

    max_states: int = DEFAULT_STATE_BUDGET
    max_edges: int = DEFAULT_EDGE_BUDGET

    def __post_init__(self) -> None:
        if not isinstance(self.max_states, int) or self.max_states < 1:
            raise ValueError("max_states must be a positive int")
        if not isinstance(self.max_edges, int) or self.max_edges < 0:
            raise ValueError("max_edges must be a non-negative int")


class ReplayError(RuntimeError):
    """An independent counterexample replay failed to reproduce the witness."""


@dataclass(frozen=True)
class Step:
    """One replayable transition firing."""

    transition_id: str
    source: tuple[int, ...]
    target: tuple[int, ...]


@dataclass(frozen=True)
class Counterexample:
    """A replayable witness.

    Safety / deadlock: ``states`` has length ``len(steps) + 1`` and
    ``bad_state`` is the violating end state. ``steps`` (and hence
    ``transition_ids``) may be empty when the initial state already fails.

    Liveness: ``prefix_states`` / ``prefix`` lead from the initial state to
    the cycle (the prefix is allowed to pass through accepting states), and
    ``cycle_states`` / ``cycle`` is a non-empty closed walk avoiding accept.
    """

    kind: str  # "safety" | "deadlock" | "liveness"
    steps: tuple[Step, ...] = ()
    states: tuple[tuple[int, ...], ...] = ()
    bad_state: tuple[int, ...] | None = None
    prefix: tuple[Step, ...] = ()
    cycle: tuple[Step, ...] = ()
    prefix_states: tuple[tuple[int, ...], ...] = ()
    cycle_states: tuple[tuple[int, ...], ...] = ()

    @property
    def transition_ids(self) -> tuple[str, ...]:
        return tuple(step.transition_id for step in self.steps)

    @property
    def length(self) -> int:
        return len(self.steps)


@dataclass(frozen=True)
class Result:
    status: Status
    property_name: str
    counterexample: Counterexample | None = None
    states_explored: int = 0
    edges_explored: int = 0
    reason: str = ""

    @property
    def passed(self) -> bool:
        return self.status is Status.PASS

    @property
    def failed(self) -> bool:
        return self.status is Status.FAIL

    @property
    def unknown(self) -> bool:
        return self.status is Status.UNKNOWN


# --------------------------------------------------------------------------- #
# Reachable-graph exploration with budgets
# --------------------------------------------------------------------------- #

@dataclass
class _Graph:
    # state -> [(transition_id, target)] in id enumeration order
    adjacency: dict
    # states whose successor list may be incomplete because a budget fired
    # while they were being expanded
    partial: set
    state_count: int
    edge_count: int
    complete: bool


def _explore(model: Model, budgets: Budgets) -> _Graph:
    """BFS over reachable states; transitions enumerated in sorted-id order."""
    initial = model.initial_state
    adjacency: dict[tuple[int, ...], list] = {}
    partial: set[tuple[int, ...]] = set()
    visited = {initial}
    queue: deque[tuple[int, ...]] = deque([initial])
    edge_count = 0

    while queue:
        state = queue.popleft()
        outgoing: list[tuple[str, tuple[int, ...]]] = []
        truncated = False
        for tid, target in model.successors(state):
            if edge_count >= budgets.max_edges:
                truncated = True
                break
            edge_count += 1
            outgoing.append((tid, target))
            if target not in visited:
                if len(visited) >= budgets.max_states:
                    truncated = True
                    break
                visited.add(target)
                queue.append(target)
        adjacency[state] = outgoing
        if truncated:
            partial.add(state)
            return _Graph(adjacency, partial, len(visited), edge_count, False)

    return _Graph(adjacency, partial, len(visited), edge_count, True)


# --------------------------------------------------------------------------- #
# Independent replay
# --------------------------------------------------------------------------- #

def _replay(model: Model, path, endpoint_check):
    """Rebuild a path from the initial state, re-checking everything.

    Guards are re-evaluated, simultaneous assignments are recomputed against
    the old state (out-of-bounds results disable the transition) and the
    supplied end-point check must fail at the final state.
    """
    transitions_by_id = {t.id: t for t in model.transitions}
    index = model._lookup()
    bounds = model.bounds
    states = [model.initial_state]
    steps: list[Step] = []
    state = states[0]
    for tid in path:
        transition = transitions_by_id.get(tid)
        if transition is None:
            raise ReplayError("unknown transition id %r" % tid)
        if not astx.evaluate(transition.guard, state, index):
            raise ReplayError("guard of %r does not hold at %r" % (tid, state))
        new_values = list(state)
        for target, rhs in transition.assignments.items():
            value = astx.evaluate(rhs, state, index)
            slot = index[target]
            if value not in bounds[slot]:
                raise ReplayError(
                    "transition %r produces out-of-bounds value for %r"
                    % (tid, target)
                )
            new_values[slot] = value
        target_state = tuple(new_values)
        steps.append(Step(tid, state, target_state))
        state = target_state
        states.append(state)
    if not endpoint_check(state):
        raise ReplayError("replayed end state does not violate the property")
    return tuple(steps), tuple(states)


# --------------------------------------------------------------------------- #
# Shortest, id-lexicographically smallest path to a violating state
# --------------------------------------------------------------------------- #

def _shortest_bad_path(graph: _Graph, initial, is_bad):
    """BFS for the shortest id sequence to a bad state.

    Successor lists are in sorted transition-id order, so the first parent
    that reaches a state gives the lexicographically smallest shortest path.
    States whose expansion was truncated by a budget are not reported.
    """
    if is_bad(initial):
        return (), initial

    parents: dict[tuple[int, ...], tuple[tuple[int, ...], str]] = {}
    frontier = deque([initial])
    seen = {initial}
    while frontier:
        state = frontier.popleft()
        for tid, target in graph.adjacency.get(state, ()):
            if target in seen:
                continue
            seen.add(target)
            parents[target] = (state, tid)
            if is_bad(target):
                path: list[str] = []
                cur = target
                while cur != initial:
                    parent, edge_id = parents[cur]
                    path.append(edge_id)
                    cur = parent
                path.reverse()
                return tuple(path), target
            frontier.append(target)
    return None


def _predicate_fn(model: Model, predicate):
    """Validate a boolean AST predicate and return its evaluator.

    Callables are deliberately rejected: the library forbids arbitrary
    callbacks; predicates must be AST nodes checked up front.
    """
    if callable(predicate):
        raise TypeError(
            "callable predicates are forbidden (no arbitrary callbacks): "
            "pass a boolean AST node instead"
        )
    model.check()
    names = frozenset(model.variable_names)
    astx.validate(predicate, names, expected="bool")
    index = model._lookup()
    return lambda state: astx.evaluate(predicate, state, index)


def _unknown(property_name: str, graph: _Graph) -> Result:
    return Result(
        Status.UNKNOWN, property_name,
        states_explored=graph.state_count,
        edges_explored=graph.edge_count,
        reason="budget exhausted before the reachable graph was fully explored",
    )


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #

def check_safety(
    model: Model,
    invariant: tuple,
    budgets: Budgets | None = None,
    *,
    property_name: str = "safety",
) -> Result:
    """Check that a boolean invariant holds in every reachable state.

    FAIL returns the shortest counterexample (ties broken by the
    lexicographically smallest id sequence); the empty path is used when the
    initial state itself violates the invariant.
    """
    budgets = budgets or Budgets()
    holds = _predicate_fn(model, invariant)
    graph = _explore(model, budgets)

    is_bad = lambda state: not holds(state)
    found = _shortest_bad_path(graph, model.initial_state, is_bad)
    if found is not None:
        path, bad_state = found
        steps, states = _replay(model, path, is_bad)
        return Result(
            Status.FAIL, property_name,
            counterexample=Counterexample(
                "safety", steps=steps, states=states, bad_state=bad_state
            ),
            states_explored=graph.state_count, edges_explored=graph.edge_count,
        )
    if not graph.complete:
        return _unknown(property_name, graph)
    return Result(
        Status.PASS, property_name,
        states_explored=graph.state_count, edges_explored=graph.edge_count,
    )


# --------------------------------------------------------------------------- #
# Deadlock freedom
# --------------------------------------------------------------------------- #

def check_deadlock(
    model: Model,
    terminal: tuple,
    budgets: Budgets | None = None,
    *,
    property_name: str = "deadlock-freedom",
) -> Result:
    """Check absence of deadlocks.

    A deadlock is a reachable state that has no enabled transition and does
    not satisfy the boolean ``terminal`` predicate (legitimate termination).
    States whose expansion was cut off by a budget are never reported.
    """
    budgets = budgets or Budgets()
    is_terminal = _predicate_fn(model, terminal)
    graph = _explore(model, budgets)

    def is_bad(state):
        # A state is a proven sink only if it was expanded completely.
        # States whose expansion was cut off (partial) and states merely
        # discovered but never expanded (absent from adjacency) do not
        # qualify: their enabled transitions are unknown.
        if state not in graph.adjacency or state in graph.partial:
            return False
        return not graph.adjacency[state] and not is_terminal(state)

    found = _shortest_bad_path(graph, model.initial_state, is_bad)
    if found is not None:
        path, bad_state = found
        steps, states = _replay(model, path, is_bad)
        return Result(
            Status.FAIL, property_name,
            counterexample=Counterexample(
                "deadlock", steps=steps, states=states, bad_state=bad_state
            ),
            states_explored=graph.state_count, edges_explored=graph.edge_count,
        )
    if not graph.complete:
        return _unknown(property_name, graph)
    return Result(
        Status.PASS, property_name,
        states_explored=graph.state_count, edges_explored=graph.edge_count,
    )


# --------------------------------------------------------------------------- #
# Liveness: every infinite execution visits accept infinitely often
# --------------------------------------------------------------------------- #

def _cyclic_sccs(nodes, successors):
    """Cyclic SCCs via iterative Tarjan (safe for 10k+ nodes).

    ``successors(node)`` returns the nodes the node edges to inside the
    induced subgraph. A singleton SCC is cyclic only with a self-loop.
    """
    index: dict = {}
    low: dict = {}
    on_stack: set = set()
    stack: list = []
    components: list[list] = []
    counter = 0

    for root in nodes:
        if root in index:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        work: list[tuple[object, object]] = [(root, iter(successors(root)))]

        while work:
            v, iterator = work[-1]
            descended = False
            for w in iterator:
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, iter(successors(w))))
                    descended = True
                    break
                if w in on_stack:
                    if index[w] < low[v]:
                        low[v] = index[w]
            if descended:
                continue

            work.pop()
            if low[v] == index[v]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == v:
                        break
                if len(component) > 1 or v in successors(v):
                    components.append(component)
            if work:
                parent = work[-1][0]
                if low[v] < low[parent]:
                    low[parent] = low[v]

    return components


def _find_path(graph: _Graph, source, is_target, within=None,
               match_source: bool = True):
    """Shortest id-lexicographically smallest edge-id path.

    ``is_target`` is a predicate on states; ``within`` optionally restricts
    usable intermediate/target states (an induced subgraph). ``match_source``
    disables the zero-length match (used when searching for a non-empty
    closed walk back to the source). Returns ``(id_path, reached_state)`` or
    None.
    """
    if within is None:
        allowed = None
    else:
        allowed = within
    if match_source and is_target(source) and (
        allowed is None or source in allowed
    ):
        return (), source
    parents = {source: None}
    frontier = deque([source])
    while frontier:
        state = frontier.popleft()
        for tid, target in graph.adjacency.get(state, ()):
            if allowed is not None and target not in allowed:
                continue
            # With source-matching disabled, an edge back to the source
            # closes a non-empty walk (self-loop or a longer cycle).
            if not match_source and target == source and is_target(target):
                path: list[str] = []
                if state != source:
                    cur = state
                    while cur != source:
                        parent, edge_id = parents[cur]
                        path.append(edge_id)
                        cur = parent
                    path.reverse()
                return tuple(path + [tid]), target
            if target in parents:
                continue
            parents[target] = (state, tid)
            if is_target(target):
                path: list[str] = []
                cur = target
                while cur != source:
                    parent, edge_id = parents[cur]
                    path.append(edge_id)
                    cur = parent
                path.reverse()
                return tuple(path), target
            frontier.append(target)
    return None


def _materialize(model: Model, path, start):
    """Rebuild steps/states along an id path starting at ``start``."""
    transitions_by_id = {t.id: t for t in model.transitions}
    index = model._lookup()
    bounds = model.bounds
    steps: list[Step] = []
    states = [start]
    state = start
    for tid in path:
        transition = transitions_by_id[tid]
        if not astx.evaluate(transition.guard, state, index):
            raise ReplayError("guard of %r does not hold at %r" % (tid, state))
        new_values = list(state)
        for target, rhs in transition.assignments.items():
            value = astx.evaluate(rhs, state, index)
            slot = index[target]
            if value not in bounds[slot]:
                raise ReplayError(
                    "transition %r produces an out-of-bounds value" % tid
                )
            new_values[slot] = value
        target_state = tuple(new_values)
        steps.append(Step(tid, state, target_state))
        state = target_state
        states.append(state)
    return tuple(steps), tuple(states)


def check_liveness(
    model: Model,
    accept: tuple,
    budgets: Budgets | None = None,
    *,
    property_name: str = "acceptance",
) -> Result:
    """Check that every infinite execution visits ``accept`` infinitely often.

    No fairness is assumed. The complete reachable graph is built first;
    then the induced subgraph on non-accepting states is searched for a
    cyclic SCC. Reachable cyclic SCC -> FAIL with a prefix (which may pass
    through accept) followed by a non-empty closed loop that never visits
    accept. Finite termination outside accept is not a violation. Requires
    the complete graph, so budget exhaustion yields UNKNOWN.
    """
    budgets = budgets or Budgets()
    is_accept = _predicate_fn(model, accept)
    graph = _explore(model, budgets)
    if not graph.complete:
        return Result(
            Status.UNKNOWN, property_name,
            states_explored=graph.state_count,
            edges_explored=graph.edge_count,
            reason="liveness needs the complete reachable graph; budget "
                   "exhausted first",
        )

    non_accept = {s for s in graph.adjacency if not is_accept(s)}

    def successors(state):
        return [t for _, t in graph.adjacency[state] if t in non_accept]

    cyclic = _cyclic_sccs(list(non_accept), successors)
    if not cyclic:
        return Result(
            Status.PASS, property_name,
            states_explored=graph.state_count, edges_explored=graph.edge_count,
        )

    best = None  # (sort key, prefix ids, entry, loop ids)
    for component in cyclic:
        comp_set = frozenset(component)
        reach = _find_path(
            graph, model.initial_state, comp_set.__contains__
        )
        if reach is None:  # pragma: no cover - all graph states are reachable
            continue
        prefix_ids, entry = reach
        loop = _find_path(
            graph, entry, lambda s: s == entry, within=comp_set,
            match_source=False,
        )
        if loop is None:  # pragma: no cover - cyclic SCC guarantees one
            continue
        loop_ids = loop[0]
        key = (prefix_ids, loop_ids)
        if best is None or key < best[0]:
            best = (key, prefix_ids, entry, loop_ids)

    _, prefix_ids, entry, loop_ids = best
    prefix_steps, prefix_states = _materialize(
        model, prefix_ids, model.initial_state
    )
    cycle_steps, cycle_states = _materialize(model, loop_ids, entry)

    # Independent verification of the witness shape.
    if not cycle_steps:
        raise ReplayError("liveness counterexample has an empty cycle")
    if cycle_states[0] != cycle_states[-1]:
        raise ReplayError("liveness loop is not closed")
    if prefix_states[-1] != cycle_states[0]:
        raise ReplayError("prefix does not connect to the cycle")
    if any(is_accept(s) for s in cycle_states[:-1]):
        raise ReplayError("liveness loop visits an accepting state")

    return Result(
        Status.FAIL, property_name,
        counterexample=Counterexample(
            "liveness",
            prefix=prefix_steps, cycle=cycle_steps,
            prefix_states=prefix_states, cycle_states=cycle_states,
        ),
        states_explored=graph.state_count, edges_explored=graph.edge_count,
    )
