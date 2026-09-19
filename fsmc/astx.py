"""Small safe expression AST for guards, assignments and predicates.

Node shapes (plain immutable tuples, no eval / no callbacks):

* ``("const", int)``                 -- integer constant
* ``("var", "name")``                -- named bounded integer variable
* ``("add", l, r)`` / ``("sub", l, r)`` -- integer addition / subtraction
* ``("lt"|"le"|"eq"|"ne"|"ge"|"gt", l, r)`` -- integer comparison, yields bool
* ``("not", x)``                     -- boolean negation
* ``("and", (x, ...))``              -- boolean conjunction (empty tuple == True)
* ``("or",  (x, ...))``              -- boolean disjunction (empty tuple == False)

Expressions are statically type checked (int vs bool), referenced variables
must exist, and AST depth is bounded by :data:`MAX_DEPTH`.
"""

from __future__ import annotations

import operator

__all__ = [
    "MAX_DEPTH",
    "AstError",
    "const",
    "variable",
    "add",
    "sub",
    "lt",
    "le",
    "eq",
    "ne",
    "ge",
    "gt",
    "land",
    "lor",
    "lnot",
    "validate",
    "evaluate",
]

MAX_DEPTH = 32

_INT = "int"
_BOOL = "bool"

_TAGS = {
    "const",
    "var",
    "add",
    "sub",
    "lt",
    "le",
    "eq",
    "ne",
    "ge",
    "gt",
    "not",
    "and",
    "or",
}

_CMP_OPS = {
    "lt": operator.lt,
    "le": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
    "ge": operator.ge,
    "gt": operator.gt,
}


class AstError(ValueError):
    """Raised for malformed / ill-typed / too deep AST nodes."""


# --------------------------------------------------------------------------- #
# Constructors
# --------------------------------------------------------------------------- #

def const(value: int) -> tuple:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AstError("integer constant expected, got %r" % (value,))
    return ("const", value)


def variable(name: str) -> tuple:
    if not isinstance(name, str) or not name:
        raise AstError("variable name must be a non-empty string")
    return ("var", name)


def add(left: tuple, right: tuple) -> tuple:
    return ("add", left, right)


def sub(left: tuple, right: tuple) -> tuple:
    return ("sub", left, right)


def lt(left: tuple, right: tuple) -> tuple:
    return ("lt", left, right)


def le(left: tuple, right: tuple) -> tuple:
    return ("le", left, right)


def eq(left: tuple, right: tuple) -> tuple:
    return ("eq", left, right)


def ne(left: tuple, right: tuple) -> tuple:
    return ("ne", left, right)


def ge(left: tuple, right: tuple) -> tuple:
    return ("ge", left, right)


def gt(left: tuple, right: tuple) -> tuple:
    return ("gt", left, right)


def lnot(operand: tuple) -> tuple:
    return ("not", operand)


def land(*operands: tuple) -> tuple:
    return ("and", tuple(operands))


def lor(*operands: tuple) -> tuple:
    return ("or", tuple(operands))


def true() -> tuple:
    """Tautology: empty conjunction."""
    return ("and", ())


def false() -> tuple:
    """Contradiction: empty disjunction."""
    return ("or", ())


# --------------------------------------------------------------------------- #
# Static validation
# --------------------------------------------------------------------------- #

def _is_node(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) >= 2
        and isinstance(value[0], str)
        and value[0] in _TAGS
    )


def _check(node: object, names: frozenset[str], depth: int) -> str:
    if depth > MAX_DEPTH:
        raise AstError("AST depth exceeds %d" % MAX_DEPTH)
    if not _is_node(node):
        raise AstError("not a well-formed AST node: %r" % (node,))
    tag = node[0]

    if tag == "const":
        value = node[1]
        if not isinstance(value, int) or isinstance(value, bool):
            raise AstError("integer constant expected, got %r" % (value,))
        return _INT

    if tag == "var":
        name = node[1]
        if not isinstance(name, str) or not name:
            raise AstError("variable name must be a non-empty string")
        if name not in names:
            raise AstError("unknown variable %r" % name)
        return _INT

    if tag in ("add", "sub"):
        _, left, right = node
        if _check(left, names, depth + 1) != _INT:
            raise AstError("%s expects integer operands" % tag)
        if _check(right, names, depth + 1) != _INT:
            raise AstError("%s expects integer operands" % tag)
        return _INT

    if tag in _CMP_OPS:
        _, left, right = node
        if _check(left, names, depth + 1) != _INT:
            raise AstError("comparison %s expects integer operands" % tag)
        if _check(right, names, depth + 1) != _INT:
            raise AstError("comparison %s expects integer operands" % tag)
        return _BOOL

    if tag == "not":
        if _check(node[1], names, depth + 1) != _BOOL:
            raise AstError("not expects a boolean operand")
        return _BOOL

    # and / or
    operands = node[1]
    if not isinstance(operands, tuple):
        raise AstError("%s operands must be a tuple" % tag)
    for operand in operands:
        if _check(operand, names, depth + 1) != _BOOL:
            raise AstError("%s expects boolean operands" % tag)
    return _BOOL


def validate(node: object, variable_names, expected: str = _BOOL) -> None:
    """Validate ``node`` against the model's variable names.

    ``expected`` is ``"bool"`` (guards / predicates) or ``"int"``
    (assignment right-hand sides).
    """
    names = frozenset(variable_names)
    if expected not in (_INT, _BOOL):
        raise ValueError("expected must be 'int' or 'bool'")
    actual = _check(node, names, 1)
    if actual != expected:
        raise AstError(
            "expected %s expression but got %s" % (expected, actual)
        )


# --------------------------------------------------------------------------- #
# Safe evaluation (no eval / exec / callbacks)
# --------------------------------------------------------------------------- #

def evaluate(node: tuple, values: tuple, index: dict[str, int]):
    """Evaluate a validated node against a concrete state tuple."""
    tag = node[0]

    if tag == "const":
        return node[1]
    if tag == "var":
        return values[index[node[1]]]
    if tag == "add":
        return evaluate(node[1], values, index) + evaluate(node[2], values, index)
    if tag == "sub":
        return evaluate(node[1], values, index) - evaluate(node[2], values, index)
    if tag in _CMP_OPS:
        return _CMP_OPS[tag](
            evaluate(node[1], values, index),
            evaluate(node[2], values, index),
        )
    if tag == "not":
        return not evaluate(node[1], values, index)
    if tag == "and":
        for operand in node[1]:
            if not evaluate(operand, values, index):
                return False
        return True
    if tag == "or":
        for operand in node[1]:
            if evaluate(operand, values, index):
                return True
        return False

    raise AstError("unknown AST tag %r" % (tag,))
