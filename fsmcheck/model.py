"""Model definition: bounded integer variables, transitions, predicates.

A :class:`Model` is fully validated at construction time:

* at most 6 variables, at most 20 transitions, unique names/IDs;
* initial values inside every variable's inclusive bounds;
* every expression type-checks (int vs bool operands, result type);
* every referenced variable exists;
* every AST has depth <= 32;
* assignment targets are distinct and exist.

Expressions are evaluated by :func:`evaluate_int` / :func:`evaluate_bool`,
never through ``eval`` or callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import ast

MAX_VARIABLES = 6
MAX_TRANSITIONS = 20
MAX_AST_DEPTH = 32


class ModelError(ValueError):
    """Raised when a model fails construction-time validation."""


@dataclass(slots=True, frozen=True)
class Variable:
    """A named integer variable with inclusive lower/upper bounds."""

    name: str
    lower: int
    upper: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ModelError("variable name must be a non-empty string")
        if not all(isinstance(v, int) and not isinstance(v, bool)
                   for v in (self.lower, self.upper)):
            raise ModelError(f"bounds of {self.name!r} must be ints")
        if self.lower > self.upper:
            raise ModelError(
                f"variable {self.name!r}: lower bound {self.lower} > "
                f"upper bound {self.upper}"
            )


@dataclass(slots=True, frozen=True)
class Transition:
    """One named transition: ``guard`` enables it, ``assigns`` fires it."""

    id: str
    guard: ast.Node          # boolean expression
    assigns: dict[str, ast.Node]  # variable -> integer RHS expression

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ModelError("transition id must be a non-empty string")
        if not isinstance(self.guard, ast.Node):
            raise ModelError(f"transition {self.id!r}: guard must be an AST")
        if not isinstance(self.assigns, dict):
            raise ModelError(
                f"transition {self.id!r}: assigns must be a dict "
                "(possibly empty, meaning a state-preserving transition)"
            )


def coerce(value) -> ast.Node:
    """Coerce a Python literal in a model into an AST node.

    Accepts ``int`` (not ``bool``), ``bool``, :class:`ast.Node`, strings
    (treated as variable references).  Everything else is rejected.
    """
    if isinstance(value, ast.Node):
        return value
    if isinstance(value, bool):
        return ast.Bool(value)
    if isinstance(value, int):
        return ast.Int(value)
    if isinstance(value, str):
        return ast.Var(value)
    raise ModelError(
        f"cannot use {type(value).__name__} as an expression: use int, bool, "
        "str (variable name) or an fsmcheck.ast node"
    )


def coerce_tree(value) -> ast.Node:
    """Recursively turn a nested expression into pure AST nodes.

    Raw ``int``/``bool``/``str`` leaves are coerced anywhere in the tree so
    models can write ``Add(Var("x"), 1)`` instead of ``Add(Var("x"), Int(1))``.
    Iterative: untrusted input is never recursed over with the Python stack.
    """
    rebuilt: dict[int, ast.Node] = {}
    stack: list = [(value, False)]
    while stack:
        raw, processed = stack.pop()
        node = coerce(raw)
        if not processed:
            stack.append((raw, True))
            # Children are the actual stored objects (nodes or raw leaves).
            for child_raw in ast.children(node):
                stack.append((child_raw, False))
            continue
        kids = ast.children(node)
        if not kids:
            rebuilt[id(raw)] = node
        elif isinstance(node, ast.Not):
            rebuilt[id(raw)] = type(node)(rebuilt[id(kids[0])])
        else:  # all remaining composite nodes are Binary subclasses
            rebuilt[id(raw)] = type(node)(
                rebuilt[id(kids[0])], rebuilt[id(kids[1])]
            )
    return rebuilt[id(value)]


_INT_BINOPS: tuple[type[ast.Binary], ...] = (ast.Add, ast.Sub)
_CMP_BINOPS: tuple[type[ast.Binary], ...] = (
    ast.Lt, ast.Le, ast.Gt, ast.Ge, ast.Eq, ast.Ne,
)
_BOOL_BINOPS: tuple[type[ast.Binary], ...] = (ast.And, ast.Or)


class Model:
    """A validated finite-state model ready to be checked."""

    def __init__(self, variables, initial, transitions,
                 *, accept=None, terminal=None):
        self._variables: list[Variable] = []
        for var in variables:
            if not isinstance(var, Variable):
                raise ModelError("variables must be Variable instances")
            self._variables.append(var)
        if not 1 <= len(self._variables) <= MAX_VARIABLES:
            raise ModelError(
                f"model needs 1..{MAX_VARIABLES} variables, "
                f"got {len(self._variables)}"
            )
        names = [v.name for v in self._variables]
        if len(set(names)) != len(names):
            raise ModelError("duplicate variable names")
        self._var_index = {name: i for i, name in enumerate(names)}
        self._bounds = {v.name: (v.lower, v.upper) for v in self._variables}

        if not isinstance(initial, dict):
            raise ModelError("initial must be a dict of variable -> int")
        if set(initial) != set(names):
            raise ModelError(
                "initial must give a value for every variable exactly once"
            )
        self._initial: tuple[int, ...] = tuple(
            self._checked_initial(name, initial[name]) for name in names
        )

        self._transitions = self._validate_transitions(transitions)
        self.validate_transition_expressions()

        self._accept = self.prepare_predicate(accept, "accept predicate")
        self._terminal = self.prepare_predicate(terminal, "terminal predicate")

    def prepare_predicate(self, value, label="predicate"):
        """Validate an ad-hoc boolean predicate (type, names, depth).

        Used for predicates supplied directly to a check as well as the
        model-level accept/terminal predicates.
        """
        if value is None:
            return None
        node = coerce_tree(value)
        self._typecheck(node, set(self._var_index), label, expect_bool=True)
        self._check_depth(node, label)
        return node

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def _checked_initial(self, name, value):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ModelError(f"initial value of {name!r} must be an int")
        lower, upper = self._bounds[name]
        if not lower <= value <= upper:
            raise ModelError(
                f"initial value {value} of {name!r} outside [{lower}, {upper}]"
            )
        return value

    def _validate_transitions(self, raw):
        if not isinstance(raw, (list, tuple)):
            raise ModelError("transitions must be a list")
        if not 1 <= len(raw) <= MAX_TRANSITIONS:
            raise ModelError(
                f"model needs 1..{MAX_TRANSITIONS} transitions, "
                f"got {len(raw)}"
            )
        result: list[Transition] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, Transition):
                raise ModelError("transitions must be Transition instances")
            if item.id in seen:
                raise ModelError(f"duplicate transition id {item.id!r}")
            seen.add(item.id)
            assigns = {target: coerce_tree(rhs) for target, rhs in
                       item.assigns.items()}
            unknown = set(assigns) - set(self._var_index)
            if unknown:
                raise ModelError(
                    f"transition {item.id!r}: assignment to unknown "
                    f"variable(s) {sorted(unknown)}"
                )
            result.append(
                Transition(item.id, coerce_tree(item.guard), assigns)
            )
        result.sort(key=lambda t: t.id)  # lexicographic enumeration order
        return result

    def _check_depth(self, node, where):
        found = ast.depth(node)
        if found > MAX_AST_DEPTH:
            raise ModelError(
                f"AST depth {found} in {where} exceeds {MAX_AST_DEPTH}"
            )

    def _typecheck(self, node, names, where, *, expect_bool):
        """Iterative post-order type inference / variable / depth check."""
        # Map id(node) -> "int" | "bool"; avoids recursion over user input.
        types: dict[int, str] = {}
        stack: list = [(node, False)]
        while stack:
            current, processed = stack.pop()
            if processed:
                self._infer_type(current, types, names, where)
                continue
            if id(current) in types:
                continue
            stack.append((current, True))
            for child in ast.children(current):
                stack.append((child, False))
        got = types[id(node)]
        want = "bool" if expect_bool else "int"
        if got != want:
            raise ModelError(
                f"{where}: expression has type {got}, expected {want}"
            )
        self._check_depth(node, where)

    def _infer_type(self, node, types, names, where):
        if isinstance(node, ast.Int):
            if not isinstance(node.value, int) or isinstance(node.value, bool):
                raise ModelError(f"{where}: Int literal must hold an int")
            types[id(node)] = "int"
        elif isinstance(node, ast.Bool):
            types[id(node)] = "bool"
        elif isinstance(node, ast.Var):
            if node.name not in names:
                raise ModelError(
                    f"{where}: unknown variable {node.name!r}"
                )
            types[id(node)] = "int"
        elif isinstance(node, ast.Not):
            if types[id(node.operand)] != "bool":
                raise ModelError(f"{where}: Not needs a boolean operand")
            types[id(node)] = "bool"
        elif isinstance(node, ast.Binary):
            left_t = types[id(node.left)]
            right_t = types[id(node.right)]
            if isinstance(node, _INT_BINOPS):
                if left_t != "int" or right_t != "int":
                    raise ModelError(
                        f"{where}: {type(node).__name__} needs integer operands"
                    )
                types[id(node)] = "int"
            elif isinstance(node, _CMP_BINOPS):
                if left_t != "int" or right_t != "int":
                    raise ModelError(
                        f"{where}: comparison needs integer operands"
                    )
                types[id(node)] = "bool"
            elif isinstance(node, _BOOL_BINOPS):
                if left_t != "bool" or right_t != "bool":
                    raise ModelError(
                        f"{where}: {type(node).__name__} needs boolean operands"
                    )
                types[id(node)] = "bool"
            else:  # pragma: no cover - Binary is abstract over these groups
                raise ModelError(f"{where}: unsupported expression node")
        else:  # pragma: no cover
            raise ModelError(f"{where}: unsupported expression node")

    def validate_transition_expressions(self):
        """Type-check / depth-check every guard and assignment RHS."""
        names = set(self._var_index)
        for trans in self._transitions:
            self._typecheck(trans.guard, names, f"guard {trans.id!r}",
                            expect_bool=True)
            for target, rhs in trans.assigns.items():
                self._typecheck(rhs, names, f"assignment {target} of "
                                            f"{trans.id!r}", expect_bool=False)

    # ------------------------------------------------------------------ #
    # Read-only views
    # ------------------------------------------------------------------ #
    @property
    def variable_names(self) -> list[str]:
        return [v.name for v in self._variables]

    @property
    def bounds(self) -> dict[str, tuple[int, int]]:
        return dict(self._bounds)

    @property
    def initial(self) -> tuple[int, ...]:
        return self._initial

    @property
    def transitions(self) -> list[Transition]:
        return list(self._transitions)

    def state_dict(self, state) -> dict[str, int]:
        return {name: state[i] for i, name in enumerate(self.variable_names)}

    # ------------------------------------------------------------------ #
    # Semantics
    # ------------------------------------------------------------------ #
    def enabled(self, transition, state) -> bool:
        """True iff *transition*'s guard holds in *state*."""
        return evaluate_bool(transition.guard, state, self._var_index)

    def fire(self, transition, state):
        """Apply simultaneous assignment using the OLD state.

        Returns the new state, or ``None`` when any RHS is out of bounds
        (the transition is then disabled).
        """
        values: dict[str, int] = {}
        for target, rhs in transition.assigns.items():
            value = evaluate_int(rhs, state, self._var_index)
            lower, upper = self._bounds[target]
            if not lower <= value <= upper:
                return None
            values[target] = value
        new = list(state)
        for target, value in values.items():
            new[self._var_index[target]] = value
        return tuple(new)

    def successors(self, state):
        """Yield ``(transition_id, next_state)`` in ID lexicographic order."""
        for trans in self._transitions:
            if self.enabled(trans, state):
                nxt = self.fire(trans, state)
                if nxt is not None:
                    yield trans.id, nxt

    def is_accept(self, state) -> bool:
        if self._accept is None:
            return False
        return evaluate_bool(self._accept, state, self._var_index)

    def is_terminal(self, state) -> bool:
        if self._terminal is None:
            return False
        return evaluate_bool(self._terminal, state, self._var_index)


# ---------------------------------------------------------------------- #
# Expression evaluation (explicit walker; never eval/exec/callbacks)
# ---------------------------------------------------------------------- #
def _evaluate(node, state, index):
    values: dict[int, object] = {}
    stack: list = [(node, False)]
    while stack:
        current, processed = stack.pop()
        if processed:
            values[id(current)] = _eval_node(current, values, state, index)
            continue
        if id(current) in values:
            continue
        stack.append((current, True))
        for child in ast.children(current):
            stack.append((child, False))
    return values[id(node)]


def _eval_node(node, values, state, index):
    if isinstance(node, ast.Int):
        return node.value
    if isinstance(node, ast.Bool):
        return node.value
    if isinstance(node, ast.Var):
        return state[index[node.name]]
    if isinstance(node, ast.Not):
        return not values[id(node.operand)]
    a = values[id(node.left)]
    b = values[id(node.right)]
    if isinstance(node, ast.Add):
        return a + b
    if isinstance(node, ast.Sub):
        return a - b
    if isinstance(node, ast.Lt):
        return a < b
    if isinstance(node, ast.Le):
        return a <= b
    if isinstance(node, ast.Gt):
        return a > b
    if isinstance(node, ast.Ge):
        return a >= b
    if isinstance(node, ast.Eq):
        return a == b
    if isinstance(node, ast.Ne):
        return a != b
    if isinstance(node, ast.And):
        return bool(a) and bool(b)
    if isinstance(node, ast.Or):
        return bool(a) or bool(b)
    raise ModelError("unsupported expression node during evaluation")  # pragma


def evaluate_int(node, state, index) -> int:
    value = _evaluate(node, state, index)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ModelError("expression did not produce an int")
    return value


def evaluate_bool(node, state, index) -> bool:
    value = _evaluate(node, state, index)
    if not isinstance(value, bool):
        raise ModelError("expression did not produce a bool")
    return value
