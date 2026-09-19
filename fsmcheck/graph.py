"""Reachability graph exploration, counterexample traces and replay.

Exploration is plain breadth-first search over states (tuples of ints).
Edges are enumerated in transition-ID lexicographic order, which is exactly
the order :class:`~fsmcheck.model.Model` exposes them.

Budgets bound the work: at most ``state_budget`` distinct states and
``edge_budget`` enumerated edges.  When a budget is hit before exploration
finishes the result is :data:`UNKNOWN`, never a pretended proof.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .model import Model, evaluate_bool


DEFAULT_STATE_BUDGET = 10_000
DEFAULT_EDGE_BUDGET = 50_000

PASS = "pass"
FAIL = "fail"
UNKNOWN = "unknown"


class ReplayError(Exception):
    """A counterexample could not be independently replayed."""


@dataclass(slots=True)
class Trace:
    """A finite path: ``states[0]`` is the initial state and firing
    ``transitions[i]`` leads from ``states[i]`` to ``states[i + 1]``."""

    transitions: list[str]
    states: list[tuple[int, ...]]

    @property
    def length(self) -> int:
        return len(self.transitions)


@dataclass(slots=True)
class Counterexample:
    """A replay-verified violation.

    For safety/deadlock ``loop`` is ``None`` and ``prefix`` is the whole
    path (it may be empty for a failure in the initial state).  For liveness
    ``prefix`` reaches a state inside a cyclic, never-accepting SCC and
    ``loop`` is the non-empty closed walk to repeat forever.
    """

    kind: str
    prefix: Trace
    loop: Trace | None = None
    detail: str = ""


@dataclass(slots=True)
class CheckResult:
    """Outcome of one check, always one of pass / fail / unknown."""

    status: str
    kind: str
    counterexample: Counterexample | None = None
    states_explored: int = 0
    edges_explored: int = 0
    state_budget: int = DEFAULT_STATE_BUDGET
    edge_budget: int = DEFAULT_EDGE_BUDGET
    message: str = ""

    def __str__(self) -> str:  # compact human-readable summary
        head = f"{self.kind}: {self.status.upper()}"
        if self.counterexample is not None:
            ce = self.counterexample
            head += (f" (prefix length {ce.prefix.length}"
                     + (f", loop length {ce.loop.length}"
                        if ce.loop is not None else "")
                     + f", ids={ce.prefix.transitions}"
                     + (f" ++ {ce.loop.transitions}"
                        if ce.loop is not None else "")
                     + ")")
        head += f" [states={self.states_explored}/{self.state_budget}," \
                f" edges={self.edges_explored}/{self.edge_budget}]"
        if self.message:
            head += f" {self.message}"
        return head


@dataclass(slots=True)
class Exploration:
    model: Model
    adjacency: dict[tuple[int, ...], list[tuple[str, tuple[int, ...]]]]
    order: list[tuple[int, ...]]           # BFS discovery order
    parent: dict[tuple[int, ...], tuple[str | None, tuple[int, ...] | None]]
    states_explored: int
    edges_explored: int
    complete: bool
    budget_reason: str = ""

    def path_to(self, state) -> Trace:
        """Shortest discovery path from the initial state to *state*.

        Because successors are enumerated in ID order and BFS keeps paths
        ordered by (length, id-sequence), the first discovery of a state is
        the lexicographically smallest shortest path to it.
        """
        ids: list[str] = []
        states: list[tuple[int, ...]] = []
        cur = state
        while cur is not None:
            trans_id, par = self.parent[cur]
            states.append(cur)
            if trans_id is not None:
                ids.append(trans_id)
            cur = par
        states.reverse()
        ids.reverse()
        return Trace(ids, states)


def explore(model: Model, *,
            state_budget: int = DEFAULT_STATE_BUDGET,
            edge_budget: int = DEFAULT_EDGE_BUDGET,
            stop_at=None) -> Exploration:
    """BFS the reachable graph from the model's unique initial state.

    *stop_at(state, edges)* is called after *state*'s outgoing edges have
    been enumerated (``edges`` is the ``[(id, destination), ...]`` list);
    returning a string ends exploration early as "complete" (a failure was
    found, not a budget cutoff).
    """
    initial = model.initial
    adjacency: dict = {initial: []}
    parent: dict = {initial: (None, None)}
    order: list = []
    queue = deque([initial])
    states_count = 1
    edges_count = 0
    complete = True
    reason = ""

    while queue:
        state = queue.popleft()
        order.append(state)

        edges = adjacency[state]
        truncated = False
        for trans_id, dst in model.successors(state):
            edges_count += 1
            edges.append((trans_id, dst))
            if edges_count > edge_budget:
                complete = False
                reason = f"edge budget {edge_budget} exceeded"
                truncated = True
                break
            if dst not in parent:
                if states_count >= state_budget:
                    complete = False
                    reason = (f"state budget {state_budget} exceeded "
                              f"(next state would be #{states_count + 1})")
                    truncated = True
                    break
                states_count += 1
                parent[dst] = (trans_id, state)
                adjacency[dst] = []
                queue.append(dst)
        if truncated:
            break

        if stop_at is not None:
            hit = stop_at(state, edges)
            if hit is not None:
                reason = hit
                break

    return Exploration(
        model=model,
        adjacency=adjacency,
        order=order,
        parent=parent,
        states_explored=states_count,
        edges_explored=edges_count,
        complete=complete,
        budget_reason=reason,
    )


# ---------------------------------------------------------------------- #
# Independent replay
# ---------------------------------------------------------------------- #
def replay(model: Model, transitions, *, start=None) -> Trace:
    """Recompute a path from scratch.

    Every transition id must exist, every guard must hold in the current
    state, and every simultaneous assignment must stay in bounds; otherwise
    :class:`ReplayError` is raised.  Nothing here trusts BFS bookkeeping.
    """
    by_id = {t.id: t for t in model.transitions}
    state = tuple(model.initial if start is None else start)
    states = [state]
    for trans_id in transitions:
        trans = by_id.get(trans_id)
        if trans is None:
            raise ReplayError(f"unknown transition id {trans_id!r}")
        if not model.enabled(trans, state):
            raise ReplayError(
                f"guard of {trans_id!r} does not hold at "
                f"{model.state_dict(state)}"
            )
        nxt = model.fire(trans, state)
        if nxt is None:
            raise ReplayError(
                f"{trans_id!r} disabled: assignment leaves variable bounds"
            )
        state = nxt
        states.append(state)
    return Trace(list(transitions), states)


def replay_prefix_loop(model: Model, counterexample: Counterexample):
    """Replay a liveness counterexample: prefix then the closed loop.

    Checks guards/bounds on every edge, loop closure, non-emptiness and that
    no loop state is accepting.
    """
    prefix = replay(model, counterexample.prefix.transitions)
    if prefix.states != counterexample.prefix.states:
        raise ReplayError("prefix states do not match the reported trace")
    entry = prefix.states[-1]
    loop = replay(model, counterexample.loop.transitions, start=entry)
    if loop.states != counterexample.loop.states:
        raise ReplayError("loop states do not match the reported trace")
    if not loop.transitions:
        raise ReplayError("liveness loop must be non-empty")
    if loop.states[-1] != entry:
        raise ReplayError("loop does not close back at the prefix endpoint")
    for state in loop.states[:-1]:
        if model.is_accept(state):
            raise ReplayError("loop passes through an accepting state")
    return prefix, loop


def verify_against_trace(model: Model, counterexample: Counterexample,
                         safety_predicate=None) -> None:
    """Full independent verification, including the endpoint violation."""
    if counterexample.kind == "liveness":
        replay_prefix_loop(model, counterexample)
        return
    trace = replay(model, counterexample.prefix.transitions)
    if trace.states != counterexample.prefix.states:
        raise ReplayError("replayed states do not match the reported trace")
    end = trace.states[-1]
    if counterexample.kind == "safety":
        if safety_predicate is None:
            raise ReplayError("safety replay needs the predicate expression")
        if evaluate_bool(safety_predicate, end, _index(model)):
            raise ReplayError("endpoint actually satisfies the safety predicate")
    elif counterexample.kind == "deadlock":
        if next(model.successors(end), None) is not None:
            raise ReplayError("endpoint still has enabled transitions")
        if model.is_terminal(end):
            raise ReplayError("endpoint is a terminal state, not a deadlock")
    else:  # pragma: no cover
        raise ReplayError(f"unknown counterexample kind {counterexample.kind}")


def _index(model: Model):
    return {name: i for i, name in enumerate(model.variable_names)}
