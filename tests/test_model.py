"""Tests for model construction, pre-checks and simultaneous assignment."""

import unittest

from fsmc import (
    Model,
    ModelError,
    Transition,
    add,
    const,
    eq,
    land,
    lt,
    sub,
    variable,
)


def make_swap_model():
    """x, y in 0..2; one transition swaps x and y simultaneously."""
    m = Model()
    m.declare("x", 0, 2, initial=1)
    m.declare("y", 0, 2, initial=2)
    m.add_transition(
        Transition(
            "swap",
            land(),
            {"x": variable("y"), "y": variable("x")},
        )
    )
    return m


class Construction(unittest.TestCase):
    def test_initial_unique_state(self):
        m = make_swap_model()
        self.assertEqual(m.initial_state, (1, 2))

    def test_too_many_variables(self):
        m = Model()
        for i in range(6):
            m.declare("v%d" % i, 0, 1, initial=0)
        with self.assertRaises(ModelError):
            m.declare("v6", 0, 1, initial=0)

    def test_too_many_transitions(self):
        m = Model()
        m.declare("x", 0, 1, initial=0)
        for i in range(20):
            m.add_transition(
                Transition("t%02d" % i, land(), {"x": const(0)})
            )
        with self.assertRaises(ModelError):
            m.add_transition(Transition("extra", land(), {"x": const(0)}))

    def test_duplicate_ids_and_names(self):
        m = Model()
        m.declare("x", 0, 1, initial=0)
        m.add_transition(Transition("t", land(), {"x": const(0)}))
        with self.assertRaises(ModelError):
            m.add_transition(Transition("t", land(), {"x": const(0)}))
        with self.assertRaises(ModelError):
            m.declare("x", 0, 1, initial=0)

    def test_initial_out_of_bounds(self):
        m = Model()
        with self.assertRaises(ModelError):
            m.declare("x", 1, 2, initial=0)

    def test_bad_bounds(self):
        m = Model()
        with self.assertRaises(ModelError):
            m.declare("x", 2, 1, initial=1)

    def test_check_rejects_unknown_var_in_guard_and_assignment(self):
        m = Model()
        m.declare("x", 0, 1, initial=0)
        m.add_transition(
            Transition("t", eq(variable("z"), const(0)), {"x": const(1)})
        )
        with self.assertRaises(ModelError):
            m.check()

        m2 = Model()
        m2.declare("x", 0, 1, initial=0)
        m2.add_transition(Transition("t", land(), {"z": const(1)}))
        with self.assertRaises(ModelError):
            m2.check()

    def test_check_rejects_type_errors(self):
        m = Model()
        m.declare("x", 0, 1, initial=0)
        m.add_transition(
            Transition("t", land(), {"x": lt(variable("x"), const(1))})
        )
        with self.assertRaises(ModelError):
            m.check()


class Semantics(unittest.TestCase):
    def test_simultaneous_assignment_reads_old_state(self):
        m = make_swap_model()
        m.check()
        successors = dict(m.successors((1, 2)))
        # Sequential execution could not yield (2, 1) from x:=y, y:=x
        # regardless of order when values differ; here it is a clean swap.
        self.assertEqual(successors, {"swap": (2, 1)})
        successors = dict(m.successors((2, 1)))
        self.assertEqual(successors, {"swap": (1, 2)})

    def test_simultaneous_dependent_rhs(self):
        # x' = x + 1, y' = x (old x): at (1, 0) -> (2, 1)
        m = Model()
        m.declare("x", 0, 3, initial=1)
        m.declare("y", 0, 3, initial=0)
        m.add_transition(
            Transition("step", land(),
                       {"x": add(variable("x"), const(1)),
                        "y": variable("x")})
        )
        m.check()
        self.assertEqual(dict(m.successors((1, 0))), {"step": (2, 1)})

    def test_out_of_bounds_result_disables_transition(self):
        # x in [0,1]; inc is guarded true but x+1 at x==1 leaves the range,
        # so the transition must be disabled (no successor).
        m = Model()
        m.declare("x", 0, 1, initial=0)
        m.add_transition(
            Transition("inc", land(), {"x": add(variable("x"), const(1))})
        )
        m.check()
        self.assertEqual(m.enabled_ids((0,)), ("inc",))
        self.assertEqual(m.enabled_ids((1,)), ())

    def test_negative_result_disables(self):
        m = Model()
        m.declare("x", 0, 2, initial=1)
        m.add_transition(
            Transition("dec", land(), {"x": sub(variable("x"), const(2))})
        )
        m.check()
        # 1 - 2 = -1 out of bounds -> disabled
        self.assertEqual(m.enabled_ids((1,)), ())
        # 2 - 2 = 0 in bounds -> enabled
        self.assertEqual(m.enabled_ids((2,)), ("dec",))

    def test_false_guard_disables(self):
        m = Model()
        m.declare("x", 0, 2, initial=0)
        m.add_transition(
            Transition("guarded", lt(variable("x"), const(0)),
                       {"x": const(1)})
        )
        m.check()
        self.assertEqual(m.enabled_ids((0,)), ())

    def test_enumeration_id_order(self):
        m = Model()
        m.declare("x", 0, 3, initial=0)
        for tid in ["z_last", "a_first", "m_mid"]:
            m.add_transition(
                Transition(tid, lt(variable("x"), const(3)),
                           {"x": add(variable("x"), const(1))})
        )
        m.check()
        ids = [tid for tid, _ in m.successors((0,))]
        self.assertEqual(ids, ["a_first", "m_mid", "z_last"])

    def test_unassigned_variables_kept(self):
        m = Model()
        m.declare("x", 0, 2, initial=1)
        m.declare("y", 0, 2, initial=2)
        m.add_transition(Transition("bump", land(),
                                    {"x": const(2)}))
        m.check()
        self.assertEqual(dict(m.successors((1, 2))), {"bump": (2, 2)})


if __name__ == "__main__":
    unittest.main()
