"""Tests for the safe expression AST: typing, unknown vars, depth limit."""

import unittest

from fsmc import (
    MAX_DEPTH,
    AstError,
    add,
    const,
    eq,
    ge,
    gt,
    land,
    le,
    lnot,
    lor,
    lt,
    ne,
    sub,
    variable,
)
from fsmc.astx import evaluate, validate

NAMES = frozenset({"x", "y"})
INDEX = {"x": 0, "y": 1}


class EvalSmoke(unittest.TestCase):
    def test_arith_and_compare(self):
        expr = lt(add(variable("x"), const(2)), sub(variable("y"), const(1)))
        validate(expr, NAMES, "bool")
        self.assertTrue(evaluate(expr, (1, 5), INDEX))
        self.assertFalse(evaluate(expr, (4, 5), INDEX))

    def test_all_comparisons(self):
        table = {
            lt: (1, 2), le: (1, 1), eq: (2, 2), ne: (1, 2),
            ge: (3, 3), gt: (4, 2),
        }
        for builder, (a, b) in table.items():
            node = builder(const(a), const(b))
            self.assertTrue(evaluate(node, (), {}))

    def test_boolean_operators(self):
        t = land()  # empty and == true
        f = lor()  # empty or == false
        self.assertTrue(evaluate(t, (), {}))
        self.assertFalse(evaluate(f, (), {}))
        self.assertTrue(evaluate(lor(t, f), (), {}))
        self.assertFalse(evaluate(land(t, f), (), {}))
        self.assertTrue(evaluate(lnot(f), (), {}))
        self.assertFalse(evaluate(lnot(t), (), {}))

    def test_short_circuit_is_value_correct(self):
        # No short-circuit side effects are possible (pure AST); just check
        # nested mixed results.
        expr = land(
            ge(variable("x"), const(0)),
            lor(lt(variable("y"), const(0)), ne(variable("y"), const(9))),
        )
        validate(expr, NAMES)
        self.assertTrue(evaluate(expr, (0, 3), INDEX))
        self.assertFalse(evaluate(expr, (0, 9), INDEX))


class TypeChecking(unittest.TestCase):
    def test_guard_must_be_bool(self):
        with self.assertRaises(AstError):
            validate(add(const(1), const(1)), NAMES, "bool")

    def test_rhs_must_be_int(self):
        with self.assertRaises(AstError):
            validate(lt(variable("x"), const(1)), NAMES, "int")

    def test_arith_rejects_bool_operands(self):
        with self.assertRaises(AstError):
            validate(add(lt(variable("x"), const(1)), const(1)), NAMES)

    def test_comparison_rejects_bool_operands(self):
        with self.assertRaises(AstError):
            validate(
                lt(eq(variable("x"), const(1)), const(1)), NAMES
            )

    def test_not_rejects_int(self):
        with self.assertRaises(AstError):
            validate(lnot(variable("x")), NAMES)

    def test_and_rejects_int_operand(self):
        with self.assertRaises(AstError):
            validate(land(lt(variable("x"), const(1)), variable("y")), NAMES)

    def test_unknown_variable(self):
        with self.assertRaises(AstError) as ctx:
            validate(eq(variable("z"), const(0)), NAMES)
        self.assertIn("unknown variable", str(ctx.exception))

    def test_bool_constant_rejected(self):
        with self.assertRaises(AstError):
            const(True)

    def test_malformed_node(self):
        for bad in [("const",), 7, "x", None, ("bogus", 1), (1, 2)]:
            with self.assertRaises(AstError):
                validate(bad, NAMES)

    def test_and_operands_must_be_tuple(self):
        with self.assertRaises(AstError):
            validate(("and", [lt(variable("x"), const(1))]), NAMES)


class DepthLimit(unittest.TestCase):
    def _chain(self, adds: int):
        node = const(0)
        for _ in range(adds):
            node = add(node, const(0))
        return node

    def test_depth_at_limit_passes(self):
        # 31 nested adds -> deepest leaf at depth 32 == MAX_DEPTH.
        validate(self._chain(MAX_DEPTH - 1), NAMES, "int")

    def test_depth_over_limit_fails(self):
        with self.assertRaises(AstError) as ctx:
            validate(self._chain(MAX_DEPTH), NAMES, "int")
        self.assertIn("depth", str(ctx.exception))

    def test_boolean_chain_depth(self):
        node = lt(variable("x"), const(0))
        for _ in range(MAX_DEPTH):
            node = lnot(node)
        with self.assertRaises(AstError):
            validate(node, NAMES)


if __name__ == "__main__":
    unittest.main()
