"""Finite-state model: bounded integer variables, unique initial state,
guarded simultaneous-assignment transitions.
"""

from __future__ import annotations

from . import astx
from .astx import AstError

__all__ = ["Model", "Transition", "ModelError", "Bounds"]

MAX_VARS = 6
MAX_TRANSITIONS = 20


class ModelError(ValueError):
    """Raised when a model fails static pre-checks."""


class Bounds:
    """Inclusive integer range ``[lo, hi]`` of one variable."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo: int, hi: int):
        if not isinstance(lo, int) or isinstance(lo, bool):
            raise ModelError("bound lo must be an int")
        if not isinstance(hi, int) or isinstance(hi, bool):
            raise ModelError("bound hi must be an int")
        if lo > hi:
            raise ModelError("empty range: %d > %d" % (lo, hi))
        self.lo = lo
        self.hi = hi

    def __contains__(self, value: int) -> bool:
        return self.lo <= value <= self.hi

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Bounds)
            and self.lo == other.lo
            and self.hi == other.hi
        )

    def __hash__(self) -> int:
        return hash((self.lo, self.hi))

    def __repr__(self) -> str:
        return "Bounds(%d, %d)" % (self.lo, self.hi)


class Transition:
    """One transition: id, boolean guard and simultaneous assignments.

    ``assignments`` maps a variable name to an integer-typed AST. Every
    right-hand side is evaluated against the *old* state; unassigned
    variables keep their value.
    """

    __slots__ = ("id", "guard", "assignments")

    def __init__(self, id: str, guard: tuple, assignments: dict[str, tuple] | None = None):
        if not isinstance(id, str) or not id:
            raise ModelError("transition id must be a non-empty string")
        self.id = id
        self.guard = guard
        self.assignments = dict(assignments or {})

    def __repr__(self) -> str:
        return "Transition(%r, ...)" % self.id


class Model:
    """A finite-state concurrent model.

    >>> m = Model()
    >>> m.declare("x", 0, 2, initial=0)
    >>> m.add_transition(Transition("t1", astx.lt(astx.variable("x"),
    ...                                           astx.const(2)),
    ...                             {"x": astx.add(astx.variable("x"),
    ...                                            astx.const(1))}))
    """

    def __init__(self):
        self._names: list[str] = []
        self._bounds: list[Bounds] = []
        self._initial: list[int] | None = None
        self._transitions: list[Transition] = []
        self._ids: set[str] = set()

    # -- construction ----------------------------------------------------- #

    def declare(self, name: str, lo: int, hi: int, initial: int) -> None:
        if not isinstance(name, str) or not name:
            raise ModelError("variable name must be a non-empty string")
        if name in self._lookup():
            raise ModelError("duplicate variable name %r" % name)
        if len(self._names) >= MAX_VARS:
            raise ModelError("at most %d variables" % MAX_VARS)
        bounds = Bounds(lo, hi)
        if initial not in bounds:
            raise ModelError(
                "initial value %d of %r out of bounds %s"
                % (initial, name, bounds)
            )
        self._names.append(name)
        self._bounds.append(bounds)
        if self._initial is None:
            self._initial = []
        self._initial.append(initial)

    def add_transition(self, transition: Transition) -> None:
        if len(self._transitions) >= MAX_TRANSITIONS:
            raise ModelError("at most %d transitions" % MAX_TRANSITIONS)
        if transition.id in self._ids:
            raise ModelError("duplicate transition id %r" % transition.id)
        if not isinstance(transition, Transition):
            raise ModelError("Transition expected")
        self._ids.add(transition.id)
        self._transitions.append(transition)

    # -- introspection ---------------------------------------------------- #

    @property
    def variable_names(self) -> tuple[str, ...]:
        return tuple(self._names)

    @property
    def bounds(self) -> tuple[Bounds, ...]:
        return tuple(self._bounds)

    @property
    def transitions(self) -> tuple[Transition, ...]:
        """Transitions sorted by id (lexicographic enumeration order)."""
        return tuple(sorted(self._transitions, key=lambda t: t.id))

    @property
    def initial_state(self) -> tuple[int, ...]:
        if self._initial is None or len(self._initial) != len(self._names):
            raise ModelError(
                "unique initial state is incomplete: declare variables first"
            )
        return tuple(self._initial)

    def _lookup(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self._names)}

    # -- static checks ---------------------------------------------------- #

    def check(self) -> None:
        """Run all pre-checks: shape, types, unknown variables, AST depth."""
        if not self._names:
            raise ModelError("model has no variables")
        if self._initial is None or len(self._initial) != len(self._names):
            raise ModelError("unique initial state is not fully defined")
        names = frozenset(self._names)
        for transition in self.transitions:
            try:
                astx.validate(transition.guard, names, expected="bool")
                for target, rhs in transition.assignments.items():
                    if target not in names:
                        raise ModelError(
                            "transition %r assigns unknown variable %r"
                            % (transition.id, target)
                        )
                    astx.validate(rhs, names, expected="int")
            except AstError as exc:
                raise ModelError(
                    "transition %r: %s" % (transition.id, exc)
                ) from exc

    # -- semantics -------------------------------------------------------- #

    def successors(self, state: tuple[int, ...]):
        """Yield ``(transition_id, next_state)`` in id lexicographic order.

        Simultaneous assignment: all right-hand sides are read from the
        supplied (old) state. A transition whose result leaves any variable
        out of bounds is disabled.
        """
        index = self._lookup()
        for transition in self.transitions:
            if not astx.evaluate(transition.guard, state, index):
                continue
            new_values = list(state)
            in_bounds = True
            for target, rhs in transition.assignments.items():
                value = astx.evaluate(rhs, state, index)
                slot = index[target]
                if value not in self._bounds[slot]:
                    in_bounds = False
                    break
                new_values[slot] = value
            if in_bounds:
                yield transition.id, tuple(new_values)

    def enabled_ids(self, state: tuple[int, ...]) -> tuple[str, ...]:
        return tuple(tid for tid, _ in self.successors(state))
