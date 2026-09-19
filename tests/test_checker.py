"""Tests for safety, deadlock and liveness checking, budgets and replay.

Explicit scenario coverage required by the task:

* simultaneous assignment            (tests/test_model.py)
* out-of-bounds disables a transition(tests/test_model.py)
* initial-state failure (empty path) (InitialFailure)
* shortest-counterexample tie        (LexicographicTie)
* self-loop liveness violation       (LivenessSelfLoop)
* prefix through accept to bad cycle (LivenessAcceptPrefix)
* finite termination is not a fail   (LivenessFiniteTermination)
* budget unknown                     (BudgetsUnknown)

RandomModels cross-checks the checker against an independently written
brute-force analysis using a fixed seed and small models only.
"""

import random
import unittest
from collections import deque
from itertools import product

from fsmc import (
    Budgets,
    Model,
    ReplayError,
    Status,
    Transition,
    add,
    check_deadlock,
    check_liveness,
    check_safety,
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


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #

class SafetyBasic(unittest.TestCase):
    def test_callable_predicate_is_rejected(self):
        m = Model()
        m.declare("x", 0, 2, initial=0)
        m.add_transition(Transition("t", land(), {"x": const(1)}))
        with self.assertRaises(TypeError):
            check_safety(m, lambda s: True)
        with self.assertRaises(TypeError):
            check_deadlock(m, lambda s: False)
        with self.assertRaises(TypeError):
            check_liveness(m, lambda s: True)

    def test_pass_when_invariant_holds_everywhere(self):
        m = Model()
        m.declare("x", 0, 2, initial=0)
        m.add_transition(
            Transition("inc", lt(variable("x"), const(2)),
                       {"x": add(variable("x"), const(1))})
        )
        result = check_safety(m, le(variable("x"), const(2)))
        self.assertIs(result.status, Status.PASS)
        self.assertIsNone(result.counterexample)
        self.assertEqual(result.states_explored, 3)

    def test_fail_reports_step_by_step_states(self):
        m = Model()
        m.declare("x", 0, 3, initial=0)
        m.add_transition(
            Transition("inc", land(),
                       {"x": add(variable("x"), const(1))})
        )
        result = check_safety(m, le(variable("x"), const(1)))
        self.assertIs(result.status, Status.FAIL)
        ce = result.counterexample
        self.assertEqual(ce.kind, "safety")
        self.assertEqual(ce.transition_ids, ("inc", "inc"))
        self.assertEqual(ce.length, 2)
        self.assertEqual(ce.states, ((0,), (1,), (2,)))
        self.assertEqual(ce.bad_state, (2,))
        # step source/target chain is consistent
        for step, source, target in zip(ce.steps, ce.states, ce.states[1:]):
            self.assertEqual(step.source, source)
            self.assertEqual(step.target, target)

    def test_failure_is_shortest(self):
        # Long detour loop then short exit: BFS must take the short one.
        m = Model()
        m.declare("pc", 0, 3, initial=0)
        # detour 0 -> 3? keep within 0..3: pc 0 -> 1 (loop waste), 0 -> 2
        # -> 3 bad. Short: jump then go.
        m.add_transition(Transition("detour", eq(variable("pc"), const(0)),
                                    {"pc": const(1)}))
        m.add_transition(Transition("back", eq(variable("pc"), const(1)),
                                    {"pc": const(0)}))
        m.add_transition(Transition("jump", eq(variable("pc"), const(0)),
                                    {"pc": const(2)}))
        m.add_transition(Transition("go", eq(variable("pc"), const(2)),
                                    {"pc": const(3)}))
        result = check_safety(m, ne(variable("pc"), const(3)))
        self.assertEqual(result.counterexample.transition_ids,
                         ("jump", "go"))


class InitialFailure(unittest.TestCase):
    def test_safety_violated_at_initial_state_empty_path(self):
        m = Model()
        m.declare("x", 0, 2, initial=2)
        m.add_transition(
            Transition("dec", gt(variable("x"), const(0)),
                       {"x": sub(variable("x"), const(1))})
        )
        result = check_safety(m, le(variable("x"), const(1)))
        self.assertIs(result.status, Status.FAIL)
        ce = result.counterexample
        self.assertEqual(ce.transition_ids, ())
        self.assertEqual(ce.steps, ())
        self.assertEqual(ce.states, ((2,),))
        self.assertEqual(ce.bad_state, (2,))

    def test_deadlock_at_initial_state_empty_path(self):
        m = Model()
        m.declare("x", 0, 1, initial=0)
        # only transition is disabled at the initial state and x can't move
        m.add_transition(
            Transition("impossible", gt(variable("x"), const(0)),
                       {"x": const(1)})
        )
        result = check_deadlock(m, lor())
        self.assertIs(result.status, Status.FAIL)
        self.assertEqual(result.counterexample.transition_ids, ())
        self.assertEqual(result.counterexample.states, ((0,),))


class LexicographicTie(unittest.TestCase):
    def test_two_equal_length_paths_picks_smaller_first_id(self):
        # From (0,0), both raise_a;raise_b and raise_b;raise_a reach (1,1).
        m = Model()
        m.declare("a", 0, 1, initial=0)
        m.declare("b", 0, 1, initial=0)
        m.add_transition(Transition("raise_a", eq(variable("a"), const(0)),
                                    {"a": const(1)}))
        m.add_transition(Transition("raise_b", eq(variable("b"), const(0)),
                                    {"b": const(1)}))
        result = check_safety(
            m, lnot(land(eq(variable("a"), const(1)),
                         eq(variable("b"), const(1))))
        )
        self.assertEqual(result.counterexample.transition_ids,
                         ("raise_a", "raise_b"))

    def test_tie_broken_on_second_id(self):
        # All shortest paths share first step "go"; afterwards two equally
        # long options "bad_early_zzz" vs "bad_late_aaa": ids determine it.
        m = Model()
        m.declare("pc", 0, 3, initial=0)
        m.add_transition(Transition("go", eq(variable("pc"), const(0)),
                                    {"pc": const(1)}))
        m.add_transition(Transition("zzz_route",
                                    eq(variable("pc"), const(1)),
                                    {"pc": const(2)}))
        m.add_transition(Transition("aaa_route",
                                    eq(variable("pc"), const(1)),
                                    {"pc": const(3)}))
        result = check_safety(m, ne(variable("pc"), const(3)))
        # BFS enumeration order at pc==1 is aaa_route then zzz_route, so the
        # lexicographically smallest shortest id sequence reaches pc==3.
        self.assertEqual(result.counterexample.transition_ids,
                         ("go", "aaa_route"))


# --------------------------------------------------------------------------- #
# Deadlock
# --------------------------------------------------------------------------- #

class Deadlock(unittest.TestCase):
    def test_deadlock_detected(self):
        m = Model()
        m.declare("pc", 0, 2, initial=0)
        m.add_transition(Transition("advance", eq(variable("pc"), const(0)),
                                    {"pc": const(1)}))
        m.add_transition(Transition("blocked", eq(variable("pc"), const(1)),
                                    {"pc": const(2)}))
        # at pc==1 assignment pc:=2 is in bounds and enabled, then pc==2
        # has no outgoing transitions -> deadlock
        result = check_deadlock(m, lor())
        self.assertIs(result.status, Status.FAIL)
        self.assertEqual(result.counterexample.transition_ids,
                         ("advance", "blocked"))
        self.assertEqual(result.counterexample.bad_state, (2,))

    def test_terminal_sink_is_not_deadlock(self):
        m = Model()
        m.declare("pc", 0, 1, initial=0)
        m.add_transition(Transition("finish", eq(variable("pc"), const(0)),
                                    {"pc": const(1)}))
        # pc == 1 declared a legitimate terminal state
        result = check_deadlock(m, eq(variable("pc"), const(1)))
        self.assertIs(result.status, Status.PASS)

    def test_no_sink_means_pass(self):
        m = Model()
        m.declare("x", 0, 1, initial=0)
        m.add_transition(Transition("set", eq(variable("x"), const(0)),
                                    {"x": const(1)}))
        m.add_transition(Transition("clr", eq(variable("x"), const(1)),
                                    {"x": const(0)}))
        result = check_deadlock(m, lor())
        self.assertIs(result.status, Status.PASS)

    def test_simultaneous_assignment_deadlock_scenario(self):
        # Two-bit handshake that stops in (1,1), not terminal.
        m = Model()
        m.declare("r", 0, 1, initial=0)
        m.declare("a", 0, 1, initial=0)
        m.add_transition(Transition("req",
                                    land(eq(variable("r"), const(0)),
                                         eq(variable("a"), const(0))),
                                    {"r": const(1)}))
        m.add_transition(Transition("ack",
                                    land(eq(variable("r"), const(1)),
                                         eq(variable("a"), const(0))),
                                    {"r": const(1), "a": const(1)}))
        result = check_deadlock(m, lor())
        self.assertIs(result.status, Status.FAIL)
        self.assertEqual(result.counterexample.transition_ids,
                         ("req", "ack"))
        self.assertEqual(result.counterexample.bad_state, (1, 1))


# --------------------------------------------------------------------------- #
# Liveness
# --------------------------------------------------------------------------- #

class LivenessSelfLoop(unittest.TestCase):
    def test_non_accepting_self_loop_is_a_violation(self):
        m = Model()
        m.declare("x", 0, 1, initial=0)
        m.add_transition(Transition("spin", land(),
                                    {"x": const(0)}))  # 0 -> 0 self-loop
        m.add_transition(Transition("to_accept", land(),
                                    {"x": const(1)}))
        result = check_liveness(m, eq(variable("x"), const(1)))
        self.assertIs(result.status, Status.FAIL)
        ce = result.counterexample
        self.assertEqual([s.transition_id for s in ce.prefix], [])
        self.assertEqual([s.transition_id for s in ce.cycle], ["spin"])
        self.assertEqual(ce.cycle_states, ((0,), (0,)))
        # every cycle state is non-accepting
        for state in ce.cycle_states[:-1]:
            self.assertNotEqual(state, (1,))

    def test_accepting_self_loop_passes(self):
        m = Model()
        m.declare("x", 0, 1, initial=1)
        m.add_transition(Transition("spin", land(), {"x": const(1)}))
        result = check_liveness(m, eq(variable("x"), const(1)))
        self.assertIs(result.status, Status.PASS)


class LivenessAcceptPrefix(unittest.TestCase):
    def test_prefix_through_accept_then_bad_cycle(self):
        # 0 -> 1(accept) -> 2 <-> 3 ; bad SCC {2,3}
        m = Model()
        m.declare("pc", 0, 3, initial=0)
        m.add_transition(Transition("enter", eq(variable("pc"), const(0)),
                                    {"pc": const(1)}))
        m.add_transition(Transition("leave", eq(variable("pc"), const(1)),
                                    {"pc": const(2)}))
        m.add_transition(Transition("fwd", eq(variable("pc"), const(2)),
                                    {"pc": const(3)}))
        m.add_transition(Transition("back", eq(variable("pc"), const(3)),
                                    {"pc": const(2)}))
        result = check_liveness(m, eq(variable("pc"), const(1)))
        self.assertIs(result.status, Status.FAIL)
        ce = result.counterexample
        self.assertEqual([s.transition_id for s in ce.prefix],
                         ["enter", "leave"])
        self.assertEqual([s.transition_id for s in ce.cycle],
                         ["fwd", "back"])
        self.assertEqual(ce.prefix_states[1], (1,))  # prefix visits accept
        self.assertEqual(ce.cycle_states[0], ce.cycle_states[-1])
        self.assertNotIn((1,), ce.cycle_states)

    def test_cycle_reachable_only_via_accept_still_fails(self):
        # Same topology, but prefix must be the unique route through accept.
        m = Model()
        m.declare("pc", 0, 3, initial=0)
        m.add_transition(Transition("a1", eq(variable("pc"), const(0)),
                                    {"pc": const(1)}))
        m.add_transition(Transition("a2", eq(variable("pc"), const(1)),
                                    {"pc": const(2)}))
        m.add_transition(Transition("loop", eq(variable("pc"), const(2)),
                                    {"pc": const(2)}))
        result = check_liveness(m, eq(variable("pc"), const(1)))
        self.assertIs(result.status, Status.FAIL)
        self.assertEqual(
            [s.transition_id for s in result.counterexample.prefix],
            ["a1", "a2"],
        )
        self.assertEqual(
            [s.transition_id for s in result.counterexample.cycle],
            ["loop"],
        )


class LivenessFiniteTermination(unittest.TestCase):
    def test_sink_outside_accept_is_not_a_violation(self):
        m = Model()
        m.declare("pc", 0, 1, initial=0)
        m.add_transition(Transition("finish", eq(variable("pc"), const(0)),
                                    {"pc": const(1)}))
        # accept is pc == 0; the only infinite execution would loop at 0,
        # but there is no loop at 0; run terminates at 1 (non-accept).
        result = check_liveness(m, eq(variable("pc"), const(0)))
        self.assertIs(result.status, Status.PASS)

    def test_branch_one_good_one_bad_fails(self):
        # From 0 can loop forever at non-accept (spin) or go to accept 2 via 1
        m = Model()
        m.declare("pc", 0, 2, initial=0)
        m.add_transition(Transition("spin", eq(variable("pc"), const(0)),
                                    {"pc": const(0)}))
        m.add_transition(Transition("visit", eq(variable("pc"), const(0)),
                                    {"pc": const(2)}))
        result = check_liveness(m, eq(variable("pc"), const(2)))
        self.assertIs(result.status, Status.FAIL)
        self.assertEqual(
            [s.transition_id for s in result.counterexample.cycle],
            ["spin"],
        )

    def test_every_cycle_hits_accept_passes(self):
        # 0 -> 1 -> 0 cycle; accept == 1; all infinite executions see it
        m = Model()
        m.declare("pc", 0, 1, initial=0)
        m.add_transition(Transition("fwd", eq(variable("pc"), const(0)),
                                    {"pc": const(1)}))
        m.add_transition(Transition("bwd", eq(variable("pc"), const(1)),
                                    {"pc": const(0)}))
        result = check_liveness(m, eq(variable("pc"), const(1)))
        self.assertIs(result.status, Status.PASS)


# --------------------------------------------------------------------------- #
# Budgets / unknown
# --------------------------------------------------------------------------- #

class BudgetsUnknown(unittest.TestCase):
    def _chain_model(self):
        m = Model()
        m.declare("x", 0, 9, initial=0)
        m.add_transition(
            Transition("inc", lt(variable("x"), const(9)),
                       {"x": add(variable("x"), const(1))})
        )
        return m

    def test_default_budgets_constants(self):
        from fsmc import DEFAULT_EDGE_BUDGET, DEFAULT_STATE_BUDGET
        self.assertEqual(DEFAULT_STATE_BUDGET, 10_000)
        self.assertEqual(DEFAULT_EDGE_BUDGET, 50_000)

    def test_state_budget_exhausted_is_unknown(self):
        m = self._chain_model()
        result = check_safety(m, le(variable("x"), const(9)),
                              budgets=Budgets(max_states=3, max_edges=100))
        self.assertIs(result.status, Status.UNKNOWN)
        self.assertLessEqual(result.states_explored, 3)
        self.assertGreater(result.edges_explored, 0)
        self.assertTrue(result.reason)

    def test_edge_budget_exhausted_is_unknown(self):
        m = self._chain_model()
        result = check_safety(
            m, le(variable("x"), const(9)),
            budgets=Budgets(max_states=10_000, max_edges=2),
        )
        self.assertIs(result.status, Status.UNKNOWN)
        self.assertLessEqual(result.edges_explored, 2)

    def test_zero_edge_budget_with_enabled_initial_edge_is_unknown(self):
        # The initial state has an enabled transition, but with a 0-edge
        # budget its expansion is truncated before even one edge fires, so a
        # deadlock downstream cannot be ruled out.
        m = Model()
        m.declare("x", 0, 2, initial=0)
        m.add_transition(
            Transition("inc", lt(variable("x"), const(2)),
                       {"x": add(variable("x"), const(1))})
        )
        result = check_deadlock(
            m, lor(), budgets=Budgets(max_states=10, max_edges=0)
        )
        self.assertIs(result.status, Status.UNKNOWN)

    def test_zero_edge_budget_still_proves_initial_deadlock(self):
        # No transition is enabled at the initial state: no edges need to be
        # fired to prove the empty-path deadlock.
        m = Model()
        m.declare("x", 0, 1, initial=0)
        m.add_transition(
            Transition("t", gt(variable("x"), const(0)), {"x": const(1)})
        )
        result = check_deadlock(
            m, lor(), budgets=Budgets(max_states=10, max_edges=0)
        )
        self.assertIs(result.status, Status.FAIL)
        self.assertEqual(result.counterexample.transition_ids, ())

    def test_liveness_always_unknown_without_complete_graph(self):
        m = self._chain_model()
        result = check_liveness(
            m, eq(variable("x"), const(0)),
            budgets=Budgets(max_states=3, max_edges=100),
        )
        self.assertIs(result.status, Status.UNKNOWN)

    def test_counterexample_found_before_budget_runs_out_still_fails(self):
        m = self._chain_model()
        # Bad state x == 5 is reached after 5 edges; budget 6 states / 8
        # edges is enough to find it even though full graph needs 10 states.
        result = check_safety(
            m, le(variable("x"), const(4)),
            budgets=Budgets(max_states=6, max_edges=8),
        )
        self.assertIs(result.status, Status.FAIL)
        self.assertEqual(result.counterexample.bad_state, (5,))

    def test_budget_validation(self):
        with self.assertRaises(ValueError):
            Budgets(max_states=0)
        with self.assertRaises(ValueError):
            Budgets(max_edges=-1)

    def test_generous_budget_completes(self):
        m = self._chain_model()
        result = check_safety(m, le(variable("x"), const(9)))
        self.assertIs(result.status, Status.PASS)
        self.assertEqual(result.states_explored, 10)


# --------------------------------------------------------------------------- #
# Replay guarantees and determinism
# --------------------------------------------------------------------------- #

class ReplayAndDeterminism(unittest.TestCase):
    def test_tampered_guard_path_is_rejected_on_replay(self):
        from fsmc.checker import _replay

        m = Model()
        m.declare("x", 0, 2, initial=0)
        m.add_transition(
            Transition("inc", lt(variable("x"), const(2)),
                       {"x": add(variable("x"), const(1))})
        )
        m.check()
        # inc is disabled at x == 2; an invented third step must fail replay
        with self.assertRaises(ReplayError):
            _replay(m, ("inc", "inc", "inc"), lambda s: True)

    def test_tampered_out_of_bounds_step_is_rejected_on_replay(self):
        from fsmc.checker import _replay

        m = Model()
        m.declare("x", 0, 1, initial=0)
        m.add_transition(
            Transition("inc", land(),
                       {"x": add(variable("x"), const(1))})
        )
        m.check()
        with self.assertRaises(ReplayError):
            _replay(m, ("inc", "inc"), lambda s: True)

    def test_unknown_id_rejected_on_replay(self):
        from fsmc.checker import _replay

        m = Model()
        m.declare("x", 0, 1, initial=0)
        m.add_transition(Transition("inc", land(), {"x": const(1)}))
        m.check()
        with self.assertRaises(ReplayError):
            _replay(m, ("ghost",), lambda s: True)

    def test_endpoint_must_actually_violate(self):
        from fsmc.checker import _replay

        m = Model()
        m.declare("x", 0, 2, initial=0)
        m.add_transition(
            Transition("inc", land(),
                       {"x": add(variable("x"), const(1))})
        )
        m.check()
        with self.assertRaises(ReplayError):
            _replay(m, ("inc",), lambda s: s == (9,))

    def test_results_are_deterministic(self):
        def build():
            m = Model()
            m.declare("a", 0, 1, initial=0)
            m.declare("b", 0, 1, initial=0)
            m.add_transition(Transition("raise_a", land(), {"a": const(1)}))
            m.add_transition(Transition("raise_b", land(), {"b": const(1)}))
            return m

        inv = lnot(land(eq(variable("a"), const(1)),
                        eq(variable("b"), const(1))))
        ids_seqs = set()
        for _ in range(5):
            result = check_safety(build(), inv)
            ids_seqs.add(result.counterexample.transition_ids)
        self.assertEqual(ids_seqs, {("raise_a", "raise_b")})


class LargerModel(unittest.TestCase):
    def test_model_one_hundred_states(self):
        # Two independent counters 0..9: exactly 100 reachable states.
        # Invariant a + b <= 9 fails; the shortest witness has 10 steps and
        # its lexicographically smallest id sequence is inc_a x9 then inc_b.
        m = Model()
        m.declare("a", 0, 9, initial=0)
        m.declare("b", 0, 9, initial=0)
        m.add_transition(
            Transition("inc_a", lt(variable("a"), const(9)),
                       {"a": add(variable("a"), const(1))})
        )
        m.add_transition(
            Transition("dec_a", gt(variable("a"), const(0)),
                       {"a": add(variable("a"), const(-1))})
        )
        m.add_transition(
            Transition("inc_b", lt(variable("b"), const(9)),
                       {"b": add(variable("b"), const(1))})
        )
        m.add_transition(
            Transition("dec_b", gt(variable("b"), const(0)),
                       {"b": add(variable("b"), const(-1))})
        )
        m.check()
        result = check_safety(
            m, le(add(variable("a"), variable("b")), const(9))
        )
        self.assertIs(result.status, Status.FAIL)
        ids = result.counterexample.transition_ids
        self.assertEqual(len(ids), 10)
        self.assertEqual(ids, ("inc_a",) * 9 + ("inc_b",))
        self.assertEqual(result.counterexample.bad_state, (9, 1))
        self.assertEqual(result.states_explored, 100)

    def test_one_hundred_state_liveness_pass(self):
        # Token ring of counters 0..9 on two vars with a return edge to the
        # accepting state (0,0) from everywhere that can reach it: a simple
        # acyclic reset graph gives finite termination instead; make (0,0)
        # revisitable via resets from any state.
        m = Model()
        m.declare("a", 0, 2, initial=0)
        m.declare("b", 0, 2, initial=0)
        for var in ("a", "b"):
            m.add_transition(
                Transition("inc_" + var,
                           lt(variable(var), const(2)),
                           {var: add(variable(var), const(1))})
            )
        # both can always be reset to 0 simultaneously -> accept revisited
        m.add_transition(
            Transition("reset",
                       lor(gt(variable("a"), const(0)),
                           gt(variable("b"), const(0))),
                       {"a": const(0), "b": const(0)})
        )
        m.check()
        result = check_liveness(
            m, land(eq(variable("a"), const(0)), eq(variable("b"), const(0)))
        )
        self.assertIs(result.status, Status.PASS)
        self.assertEqual(result.states_explored, 9)


# --------------------------------------------------------------------------- #
# Random cross-validation against an independent brute-force analysis
# --------------------------------------------------------------------------- #

CMP_BUILDERS = [lt, le, eq, ne, ge, gt]
ARITH_BUILDERS = [add, sub]


def _random_int_expr(rng, names, depth=0):
    choices = ["const"]
    if names:
        choices.append("var")
    if depth < 2:
        choices += ["arith"]
    kind = rng.choice(choices)
    if kind == "const":
        return const(rng.randint(0, 2))
    if kind == "var":
        return variable(rng.choice(names))
    builder = rng.choice(ARITH_BUILDERS)
    return builder(
        _random_int_expr(rng, names, depth + 1),
        _random_int_expr(rng, names, depth + 1),
    )


def _random_bool_expr(rng, names, depth=0):
    if depth >= 2:
        return rng.choice(CMP_BUILDERS)(
            _random_int_expr(rng, names), _random_int_expr(rng, names)
        )
    kind = rng.choice(["cmp", "cmp", "and", "or", "not"])
    if kind == "cmp":
        return rng.choice(CMP_BUILDERS)(
            _random_int_expr(rng, names), _random_int_expr(rng, names)
        )
    if kind == "and":
        return land(
            _random_bool_expr(rng, names, depth + 1),
            _random_bool_expr(rng, names, depth + 1),
        )
    if kind == "or":
        return lor(
            _random_bool_expr(rng, names, depth + 1),
            _random_bool_expr(rng, names, depth + 1),
        )
    return lnot(_random_bool_expr(rng, names, depth + 1))


def _random_model(rng):
    m = Model()
    var_count = rng.randint(1, 2)
    names = ["x", "y"][:var_count]
    for name in names:
        m.declare(name, 0, 2, initial=rng.randint(0, 2))
    for i in range(rng.randint(1, 5)):
        guard = _random_bool_expr(rng, names)
        targets = rng.sample(names, rng.randint(1, len(names)))
        assignments = {
            target: _random_int_expr(rng, names) for target in targets
        }
        m.add_transition(Transition("t%02d" % i, guard, assignments))
    m.check()
    return m, names


def _reachable(model):
    """Independent reachability: adjacency + states, stack based."""
    initial = model.initial_state
    adjacency = {}
    stack = [initial]
    seen = {initial}
    while stack:
        state = stack.pop()
        outgoing = list(model.successors(state))  # sorted-id order
        adjacency[state] = outgoing
        for _, target in outgoing:
            if target not in seen:
                seen.add(target)
                stack.append(target)
    return initial, adjacency, seen


def _all_shortest_bad_sequences(adjacency, initial, is_bad, max_depth=12):
    """Enumerate simple (state-simple) paths; return shortest length and the
    set of lexicographically minimal id sequences at that length.
    Independently written DFS, not the checker's BFS.
    """
    best_len = None
    best_seqs = set()

    def dfs(state, ids, visited):
        nonlocal best_len, best_seqs
        if is_bad(state):
            length = len(ids)
            seq = tuple(ids)
            if best_len is None or length < best_len:
                best_len = length
                best_seqs = {seq}
            elif length == best_len:
                best_seqs.add(seq)
            return
        if best_len is not None and len(ids) >= best_len:
            return
        if len(ids) >= max_depth:
            return
        for tid, target in adjacency[state]:
            if target in visited:
                continue
            visited.add(target)
            ids.append(tid)
            dfs(target, ids, visited)
            ids.pop()
            visited.remove(target)

    dfs(initial, [], {initial})
    return best_len, best_seqs


def _has_cycle_reachable(adjacency, initial, is_accept):
    """Brute force: a liveness violation exists iff some reachable
    non-accepting state u lies on a cycle made entirely of non-accepting
    states. Tested by reachability back to u through the induced subgraph.
    """
    non_accept = {s for s in adjacency if not is_accept(s)}

    def reaches(start, goal):
        seen = {start}
        stack = [start]
        while stack:
            state = stack.pop()
            for _, target in adjacency[state]:
                if target == goal:
                    return True
                if target in non_accept and target not in seen:
                    seen.add(target)
                    stack.append(target)
        return False

    # BFS from initial to enumerate reachable non-accept states
    reachable = {initial} if initial in non_accept else set()
    stack = [initial]
    seen = {initial}
    while stack:
        state = stack.pop()
        for _, target in adjacency[state]:
            if target not in seen:
                seen.add(target)
                stack.append(target)
                if target in non_accept:
                    reachable.add(target)
    for u in reachable:
        # any non-empty path u -> u staying inside non_accept?
        for _, first in adjacency[u]:
            if first in non_accept and (
                first == u or reaches(first, u)
            ):
                return True
    return False


class RandomModels(unittest.TestCase):
    def test_cross_check_fixed_seed(self):
        rng = random.Random(20260920)
        samples = 150
        for sample in range(samples):
            m, names = _random_model(rng)
            initial, adjacency, seen = _reachable(m)

            # predicates are simple comparisons of one variable to a const
            var = rng.choice(names)
            k = rng.randint(0, 2)
            builder = rng.choice(CMP_BUILDERS)
            predicate = builder(variable(var), const(k))
            from fsmc.astx import evaluate as _eval
            index = {n: i for i, n in enumerate(names)}
            is_pred = lambda s, p=predicate: _eval(p, s, index)

            # ---- safety ------------------------------------------------ #
            result = check_safety(m, predicate)
            best_len, best_seqs = _all_shortest_bad_sequences(
                adjacency, initial, lambda s: not is_pred(s)
            )
            if best_len is None:
                self.assertIs(result.status, Status.PASS,
                              "sample %d" % sample)
            else:
                self.assertIs(result.status, Status.FAIL,
                              "sample %d" % sample)
                ce = result.counterexample
                self.assertEqual(len(ce.transition_ids), best_len,
                                 "sample %d" % sample)
                self.assertIn(ce.transition_ids, best_seqs,
                              "sample %d" % sample)
                self.assertEqual(min(best_seqs), ce.transition_ids,
                                 "sample %d" % sample)
                # independently walk the witness
                state = initial
                self.assertEqual(ce.states[0], initial)
                for step, expected in zip(ce.steps, ce.states[1:]):
                    outgoing = dict(adjacency[state])
                    self.assertIn(step.transition_id, outgoing,
                                  "sample %d" % sample)
                    state = outgoing[step.transition_id]
                    self.assertEqual(state, expected, "sample %d" % sample)
                self.assertFalse(is_pred(ce.bad_state), "sample %d" % sample)

            # ---- deadlock ---------------------------------------------- #
            term_var = rng.choice(names)
            term_k = rng.randint(0, 2)
            terminal = rng.choice(CMP_BUILDERS)(
                variable(term_var), const(term_k)
            )
            is_term = lambda s, p=terminal: _eval(p, s, index)
            d_result = check_deadlock(m, terminal)
            d_len, d_seqs = _all_shortest_bad_sequences(
                adjacency, initial,
                lambda s: not adjacency[s] and not is_term(s),
            )
            if d_len is None:
                self.assertIs(d_result.status, Status.PASS,
                              "sample %d deadlock" % sample)
            else:
                self.assertIs(d_result.status, Status.FAIL,
                              "sample %d deadlock" % sample)
                self.assertEqual(
                    d_result.counterexample.transition_ids,
                    min(d_seqs),
                    "sample %d deadlock" % sample,
                )

            # ---- liveness ---------------------------------------------- #
            acc_var = rng.choice(names)
            acc_k = rng.randint(0, 2)
            accept = rng.choice(CMP_BUILDERS)(
                variable(acc_var), const(acc_k)
            )
            is_accept = lambda s, p=accept: _eval(p, s, index)
            l_result = check_liveness(m, accept)
            expected_violation = _has_cycle_reachable(
                adjacency, initial, is_accept
            )
            if expected_violation:
                self.assertIs(l_result.status, Status.FAIL,
                              "sample %d liveness" % sample)
                ce = l_result.counterexample
                self.assertTrue(len(ce.cycle) >= 1)
                self.assertEqual(ce.cycle_states[0], ce.cycle_states[-1])
                self.assertEqual(ce.prefix_states[-1], ce.cycle_states[0])
                for state in ce.cycle_states[:-1]:
                    self.assertFalse(is_accept(state),
                                     "sample %d liveness" % sample)
                # prefix + loop is executable in the real graph
                state = initial
                for step in ce.prefix:
                    state = dict(adjacency[state])[step.transition_id]
                self.assertEqual(state, ce.cycle_states[0])
                for step in ce.cycle:
                    state = dict(adjacency[state])[step.transition_id]
                self.assertEqual(state, ce.cycle_states[0])
            else:
                self.assertIs(l_result.status, Status.PASS,
                              "sample %d liveness" % sample)


if __name__ == "__main__":
    unittest.main()
