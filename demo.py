"""Demo for fsmcheck: one proven-correct model and real failures.

Run with ``python demo.py``.  Everything printed is actually computed by
the checker; the script finishes in well under 8 seconds and exits 0 even
though some checks report FAIL (that is the point of a model checker).
"""

from __future__ import annotations

import time

from fsmcheck import (
    Model, Variable, Transition,
    Var, Add, Lt, Le, Eq, And, Or, Not,
    check_safety, check_deadlock, check_liveness,
    replay, replay_prefix_loop, PASS, FAIL, UNKNOWN,
)

LINE = "=" * 70


def state_str(model, state):
    return ", ".join(f"{k}={v}" for k, v in model.state_dict(state).items())


def show_result(result):
    print(f"    {result}")


def walk_finite(model, result):
    """Independently replay a safety/deadlock counterexample step by step."""
    ce = result.counterexample
    trace = replay(model, ce.prefix.transitions)
    print("    independent replay, state by state:")
    print(f"      step 0: {state_str(model, trace.states[0])}  (initial)")
    for i, trans_id in enumerate(ce.prefix.transitions, start=1):
        print(f"      --[{trans_id}]-->")
        print(f"      step {i}: {state_str(model, trace.states[i])}")


def walk_liveness(model, result):
    ce = result.counterexample
    prefix, loop = replay_prefix_loop(model, ce)
    print("    independent replay of the liveness counterexample:")
    print(f"      step 0: {state_str(model, prefix.states[0])}  (initial)")
    n = 0
    for tid, nxt in zip(ce.prefix.transitions, prefix.states[1:]):
        n += 1
        mark = " accept" if model.is_accept(prefix.states[n - 1]) else ""
        print(f"      --[{tid}]-->{mark}")
        print(f"      step {n}: {state_str(model, nxt)}")
    print("      --- now repeat this closed loop forever, never accepting:")
    for tid in ce.loop.transitions:
        print(f"      --[{tid}]-->")
    print(f"      back to: {state_str(model, loop.states[-1])}")


# ---------------------------------------------------------------------- #
# Models
# ---------------------------------------------------------------------- #
def mutex_models():
    """Two processes p,q (0 idle, 1 waiting, 2 critical) sharing a lock L."""
    variables = [
        Variable("p", 0, 2), Variable("q", 0, 2), Variable("L", 0, 1),
    ]
    initial = {"p": 0, "q": 0, "L": 0}
    mutex = Not(And(Eq(Var("p"), 2), Eq(Var("q"), 2)))

    good = [
        Transition("p_req",   Eq(Var("p"), 0), {"p": 1}),
        Transition("p_enter", And(Eq(Var("p"), 1), Eq(Var("L"), 0)),
                   {"p": 2, "L": 1}),
        Transition("p_exit",  Eq(Var("p"), 2), {"p": 0, "L": 0}),
        Transition("q_req",   Eq(Var("q"), 0), {"q": 1}),
        Transition("q_enter", And(Eq(Var("q"), 1), Eq(Var("L"), 0)),
                   {"q": 2, "L": 1}),
        Transition("q_exit",  Eq(Var("q"), 2), {"q": 0, "L": 0}),
    ]
    # Broken version: enter ignores the lock and never sets it, so both
    # processes can be critical at the same time.
    bad = [
        Transition("p_req",   Eq(Var("p"), 0), {"p": 1}),
        Transition("p_enter", Eq(Var("p"), 1), {"p": 2}),
        Transition("p_exit",  Eq(Var("p"), 2), {"p": 0}),
        Transition("q_req",   Eq(Var("q"), 0), {"q": 1}),
        Transition("q_enter", Eq(Var("q"), 1), {"q": 2}),
        Transition("q_exit",  Eq(Var("q"), 2), {"q": 0}),
    ]
    return (
        Model(variables, initial, good),
        Model(variables, initial, bad),
        mutex,
    )


def deadlock_model():
    """Two counters that each need a resource the other holds first run:
    a can only move at x=0,y=1; b only at x=1,y=0 -- neither ever enabled
    from (1,1), which is not terminal."""
    return Model(
        variables=[Variable("x", 0, 1), Variable("y", 0, 1)],
        initial={"x": 1, "y": 1},
        transitions=[
            Transition("a_move",
                       And(Eq(Var("x"), 0), Eq(Var("y"), 1)),
                       {"x": 1, "y": 0}),
            Transition("b_move",
                       And(Eq(Var("x"), 1), Eq(Var("y"), 0)),
                       {"x": 0, "y": 1}),
        ],
    )


def liveness_model():
    """Counter x: 0(accept) -> 1 -> 2(accept) -> 3 <-> 4 forever.

    An execution can pass through accepting states on its prefix and then
    spin in the 3-4 ring, never accepting again: a real liveness violation
    under no-fairness assumptions."""
    return Model(
        variables=[Variable("x", 0, 4)],
        initial={"x": 0},
        transitions=[
            Transition("f1", Eq(Var("x"), 0), {"x": 1}),
            Transition("f2", Eq(Var("x"), 1), {"x": 2}),
            Transition("f3", Eq(Var("x"), 2), {"x": 3}),
            Transition("g4", Eq(Var("x"), 3), {"x": 4}),
            Transition("g5", Eq(Var("x"), 4), {"x": 3}),
        ],
        accept=Or(Eq(Var("x"), 0), Eq(Var("x"), 2)),
    )


def chain_model():
    """A long chain used only to demonstrate the budget -> UNKNOWN path."""
    return Model(
        variables=[Variable("x", 0, 30)],
        initial={"x": 0},
        transitions=[
            Transition("inc", Lt(Var("x"), 30), {"x": Add(Var("x"), 1)}),
        ],
    )


def main():
    start = time.perf_counter()
    print(LINE)
    print("fsmcheck demo - explicit finite-state model checking")
    print(LINE)

    # ---- 1. Correct protocol: everything proven PASS ---------------- #
    good_mutex, _, mutex_pred = mutex_models()
    print("\n[1] Correct two-process lock protocol (8-state model)")
    safety = check_safety(good_mutex, mutex_pred)
    dead = check_deadlock(good_mutex)
    print(f"    mutual exclusion (never both critical):")
    show_result(safety)
    print(f"    deadlock freedom:")
    show_result(dead)
    assert safety.status == PASS and dead.status == PASS

    # ---- 2. Broken protocol: a real safety FAIL, replayed ------------ #
    _, bad_mutex, _ = mutex_models()
    print("\n[2] Same protocol but enter ignores the lock")
    result = check_safety(bad_mutex, mutex_pred)
    show_result(result)
    assert result.status == FAIL
    ce = result.counterexample
    print(f"    shortest counterexample: {len(ce.prefix.transitions)} "
          f"transition(s), ids={ce.prefix.transitions}")
    print(f"    violation: {ce.detail}")
    walk_finite(bad_mutex, result)

    # ---- 3. A genuine deadlock FAIL ---------------------------------- #
    print("\n[3] Resource protocol stuck from its initial state")
    result = check_deadlock(deadlock_model())
    show_result(result)
    assert result.status == FAIL
    walk_finite(deadlock_model(), result)

    # ---- 4. Liveness FAIL: accept prefix into a bad forever ring ----- #
    print("\n[4] Progress: every infinite run must accept infinitely often")
    live = liveness_model()
    result = check_liveness(live)
    show_result(result)
    assert result.status == FAIL
    walk_liveness(live, result)

    # ---- 5. Finite termination is NOT a liveness failure ------------- #
    terminating = Model(
        variables=[Variable("x", 0, 1)],
        initial={"x": 1},
        transitions=[Transition("stop", Eq(Var("x"), 1), {"x": 0})],
        accept=Eq(Var("x"), 1),
        terminal=Eq(Var("x"), 0),
    )
    print("\n[5] A run that simply terminates (finite, not a violation)")
    result = check_liveness(terminating)
    show_result(result)
    assert result.status == PASS

    # ---- 6. Budget exhausted: honest UNKNOWN, never a fake pass ------ #
    print("\n[6] Exploration stopped by a tiny state budget")
    result = check_safety(chain_model(), Le(Var("x"), 30),
                          state_budget=5, edge_budget=50_000)
    show_result(result)
    assert result.status == UNKNOWN
    print("    -> reported UNKNOWN with counts; no pass/fail was guessed")

    elapsed = time.perf_counter() - start
    print("\n" + LINE)
    print(f"demo finished in {elapsed:.2f}s (budget ~8s); all computed, "
          f"all assertions held")
    print(LINE)


if __name__ == "__main__":
    main()
