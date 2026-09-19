"""Demonstration of the fsmc finite-state model checking library.

Run with ``python demo.py``. It computes real results for small models and
prints one proven PASS plus genuinely triggered FAIL witnesses (safety,
deadlock, liveness) and a budget-limited UNKNOWN. All output is produced by
the library; nothing is hard-coded.
"""

from __future__ import annotations

import time

from fsmc import (
    Budgets,
    Model,
    Status,
    Transition,
    add,
    check_deadlock,
    check_liveness,
    check_safety,
    const,
    eq,
    gt,
    land,
    le,
    lnot,
    lor,
    lt,
    variable,
)

LINE = "-" * 68


def state_str(model: Model, state: tuple[int, ...]) -> str:
    return "(" + ", ".join(
        "%s=%d" % (name, value)
        for name, value in zip(model.variable_names, state)
    ) + ")"


def print_result(title: str, model: Model, result) -> None:
    print(LINE)
    print(title)
    print("  status : %s" % result.status.value.upper())
    print(
        "  counts : %d states, %d edges"
        % (result.states_explored, result.edges_explored)
    )
    if result.reason:
        print("  reason : %s" % result.reason)
    ce = result.counterexample
    if ce is None:
        return
    if ce.kind in ("safety", "deadlock"):
        print("  length : %d transition(s)" % ce.length)
        if ce.length == 0:
            print("  path   : <empty> (initial state already violates)")
        else:
            print("  ids    : %s" % " -> ".join(ce.transition_ids))
        print("  replay :")
        for i, state in enumerate(ce.states):
            print("    %d: %s" % (i, state_str(model, state)))
    else:
        p_ids = [s.transition_id for s in ce.prefix]
        c_ids = [s.transition_id for s in ce.cycle]
        print("  prefix : %s" % (" -> ".join(p_ids) or "<empty>"))
        print("  loop   : %s   (repeat forever)" % " -> ".join(c_ids))
        print("  replay :")
        for i, state in enumerate(ce.prefix_states):
            print("    %d: %s" % (i, state_str(model, state)))
        n = len(ce.prefix_states) - 1
        for i, state in enumerate(ce.cycle_states[1:], start=1):
            tag = " <= cycle closes" if i == len(ce.cycle_states) - 1 else ""
            print("    %d: %s%s" % (n + i, state_str(model, state), tag))


def demo_safety_pass() -> None:
    """A bounded counter 0..3; invariant x <= 3 always holds."""
    m = Model()
    m.declare("x", 0, 3, initial=0)
    m.add_transition(
        Transition("inc", lt(variable("x"), const(3)),
                   {"x": add(variable("x"), const(1))})
    )
    m.add_transition(
        Transition("reset", gt(variable("x"), const(0)),
                   {"x": const(0)})
    )
    result = check_safety(m, le(variable("x"), const(3)),
                          property_name="x <= 3")
    print_result("[PASS] SAFETY  bounded counter keeps x <= 3", m, result)
    assert result.status is Status.PASS


def demo_safety_fail() -> None:
    """Two independent one-bit requesters; both requesting is forbidden.

    Two shortest paths reach the bad state; the checker must return the
    lexicographically smallest id sequence: raise_a then raise_b.
    """
    m = Model()
    m.declare("a", 0, 1, initial=0)
    m.declare("b", 0, 1, initial=0)
    m.add_transition(
        Transition("raise_a", eq(variable("a"), const(0)),
                   {"a": const(1)})
    )
    m.add_transition(
        Transition("raise_b", eq(variable("b"), const(0)),
                   {"b": const(1)})
    )
    m.add_transition(
        Transition("clear_a", eq(variable("a"), const(1)),
                   {"a": const(0)})
    )
    m.add_transition(
        Transition("clear_b", eq(variable("b"), const(1)),
                   {"b": const(0)})
    )
    invariant = lnot(
        land(eq(variable("a"), const(1)), eq(variable("b"), const(1)))
    )
    result = check_safety(m, invariant,
                          property_name="not (a == 1 and b == 1)")
    print_result("[FAIL] SAFETY  mutual exclusion between a and b", m, result)
    assert result.status is Status.FAIL
    assert result.counterexample.transition_ids == ("raise_a", "raise_b")


def demo_deadlock_fail() -> None:
    """A one-shot latch: once closed it cannot move and is not terminal."""
    m = Model()
    m.declare("latch", 0, 1, initial=0)
    m.add_transition(
        Transition("close", eq(variable("latch"), const(0)),
                   {"latch": const(1)})
    )
    result = check_deadlock(m, lor(), property_name="latch never jams")
    print_result("[FAIL] DEADLOCK  closed latch with no exit", m, result)
    assert result.status is Status.FAIL


def demo_liveness_pass() -> None:
    """Counter that can always reset to 0: every infinite run sees x == 0."""
    m = Model()
    m.declare("x", 0, 3, initial=0)
    m.add_transition(
        Transition("inc", lt(variable("x"), const(3)),
                   {"x": add(variable("x"), const(1))})
    )
    m.add_transition(
        Transition("reset", gt(variable("x"), const(0)),
                   {"x": const(0)})
    )
    result = check_liveness(m, eq(variable("x"), const(0)),
                            property_name="visit x == 0 infinitely often")
    print_result("[PASS] LIVENESS  reset keeps every run visiting 0",
                 m, result)
    assert result.status is Status.PASS


def demo_liveness_fail() -> None:
    """Run may pass through the accepting state and then get trapped in a
    non-accepting 2-cycle: 0 -> 1(accept) -> 2 <-> 3."""
    m = Model()
    m.declare("pc", 0, 3, initial=0)
    m.add_transition(
        Transition("enter", eq(variable("pc"), const(0)),
                   {"pc": const(1)})
    )
    m.add_transition(
        Transition("leave", eq(variable("pc"), const(1)),
                   {"pc": const(2)})
    )
    m.add_transition(
        Transition("toggle_fwd", eq(variable("pc"), const(2)),
                   {"pc": const(3)})
    )
    m.add_transition(
        Transition("toggle_back", eq(variable("pc"), const(3)),
                   {"pc": const(2)})
    )
    result = check_liveness(
        m, eq(variable("pc"), const(1)),
        property_name="visit pc == 1 infinitely often",
    )
    print_result("[FAIL] LIVENESS  escape through accept into a 2-cycle",
                 m, result)
    assert result.status is Status.FAIL
    ce = result.counterexample
    assert [s.transition_id for s in ce.prefix] == ["enter", "leave"]
    assert [s.transition_id for s in ce.cycle] == [
        "toggle_fwd", "toggle_back"
    ]


def demo_unknown() -> None:
    """A small model explored under a tiny state/edge budget -> UNKNOWN."""
    m = Model()
    m.declare("x", 0, 9, initial=0)
    m.add_transition(
        Transition("inc", lt(variable("x"), const(9)),
                   {"x": add(variable("x"), const(1))})
    )
    m.add_transition(
        Transition("dec", gt(variable("x"), const(0)),
                   {"x": add(variable("x"), const(-1))})
    )
    result = check_safety(
        m, le(variable("x"), const(9)),
        budgets=Budgets(max_states=4, max_edges=4),
        property_name="x <= 9 with a 4-state/4-edge budget",
    )
    print_result("[UNKNOWN] BUDGET  exploration cut off early", m, result)
    assert result.status is Status.UNKNOWN
    assert result.states_explored <= 4
    assert result.edges_explored <= 4


def main() -> None:
    start = time.perf_counter()
    print("fsmc demonstration -- results are computed by the checker")
    demo_safety_pass()
    demo_safety_fail()
    demo_deadlock_fail()
    demo_liveness_pass()
    demo_liveness_fail()
    demo_unknown()
    print(LINE)
    print("elapsed: %.2f s" % (time.perf_counter() - start))


if __name__ == "__main__":
    main()
