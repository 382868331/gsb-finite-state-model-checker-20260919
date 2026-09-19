"""Small expression AST used by guards, predicates and assignment right-hand sides.

An expression is *always* a tree of :class:`Node` objects (or plain ``int`` /
``str`` / ``bool`` literals that :mod:`fsmcheck.model` coerces).  There is no
``eval`` and there are no user callbacks anywhere: the checker walks these
nodes itself, so a model can never execute arbitrary code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Node:
    """Abstract base class for every AST node."""


@dataclass(slots=True, frozen=True)
class Int(Node):
    """Integer literal."""

    value: int


@dataclass(slots=True, frozen=True)
class Bool(Node):
    """Boolean literal."""

    value: bool


@dataclass(slots=True, frozen=True)
class Var(Node):
    """Reference to a named state variable (always integer typed)."""

    name: str


@dataclass(slots=True, frozen=True)
class Binary(Node):
    left: Node
    right: Node


class Add(Binary):
    """Integer addition (both operands must be integer typed)."""


class Sub(Binary):
    """Integer subtraction (both operands must be integer typed)."""


class Lt(Binary):
    """``left < right`` -- boolean result."""


class Le(Binary):
    """``left <= right`` -- boolean result."""


class Gt(Binary):
    """``left > right`` -- boolean result."""


class Ge(Binary):
    """``left >= right`` -- boolean result."""


class Eq(Binary):
    """``left == right`` -- boolean result."""


class Ne(Binary):
    """``left != right`` -- boolean result."""


class And(Binary):
    """Boolean conjunction (both operands must be boolean typed)."""


class Or(Binary):
    """Boolean disjunction (both operands must be boolean typed)."""


@dataclass(slots=True, frozen=True)
class Not(Node):
    """Boolean negation."""

    operand: Node


def children(node: Node) -> tuple[Node, ...]:
    """Return the direct child expressions of *node*."""
    if isinstance(node, Binary):
        return (node.left, node.right)
    if isinstance(node, Not):
        return (node.operand,)
    return ()


def depth(node: Node) -> int:
    """Longest root-to-leaf path length (a leaf has depth 1).

    Iterative on purpose: the depth check runs *before* the tree is trusted,
    so it must not rely on recursion over attacker-controlled input.
    """
    stack: list[tuple[Node, int]] = [(node, 1)]
    deepest = 0
    while stack:
        current, level = stack.pop()
        if level > deepest:
            deepest = level
        for child in children(current):
            stack.append((child, level + 1))
    return max(deepest, 1)


def walk(node: Node):
    """Yield every node in the tree (parents before children)."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        for child in children(current):
            stack.append(child)
