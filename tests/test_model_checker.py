"""Tests for the fsmcheck finite-state model checker.

Covers every scenario required by TASK.md:
simultaneous assignment, out-of-bounds disabling, initial-state failure,
short-counterexample tie breaking, self loops, accept-prefix into a bad
loop, finite termination, budget unknown -- plus construction-time
validation, independent replay and a fixed-seed randomized differential
check against a separately written reference.
"""

import os
import random
import re
import unittest

from fsmcheck import (
    Model, Variable, Transition, ModelError,
    Int, Bool, Var, Add, Sub, Lt, Le, Gt, Ge, Eq, Ne, And, Or, Not,
    check_safety, check_deadlock, check_liveness,
    replay, replay_prefix_loop, ReplayError,
    PASS, FAIL, UNKNOWN,
)


def V(name):
    return Var(name)


class SimultaneousAssignmentTests(unittest.TestCase):
    def test_swap_reads_old_state(self):
        # x=1, y=0 -- one transition swaps them.  Both RHS read the OLD
        # state, so the result is (x=0, y=1), not (0, 0) or (1, 1).
        m = Model(
            variables=[Variable("x", 0, 1), Variable("y", 0, 1)],
            initial={"x": 1, "y": 0},
            transitions=[
                Transition("swap", Eq(V("x"), 1),
                           {"x": V("y"), "y": V("x")}),
            ],
        )
        self.assertEqual(list(m.successors(m.initial)),
                         [("swap", (0, 1))])

    def test_two_assignments_both_use_old_values(self):
        # x=2, y=3: x <- y+1 and y <- x+1 simultaneously -> (4, 3).
        m = Model(
            variables=[Variable("x", 0, 9), Variable("y", 0, 9)],
            initial={"x": 2, "y": 3},
            transitions=[
                Transition("t", Bool(True),
                           {"x": Add(V("y"), 1), "y": Add(V("x"), 1)}),
            ],
        )
        self.assertEqual(list(m.successors(m.initial)), [("t", (4, 3))])


class OutOfBoundsDisablesTests(unittest.TestCase):
    def _model(self, terminal=None):
        # inc would compute x+1=2 outside [0,1]: disabled by the bounds.
        return Model(
            variables=[Variable("x", 0, 1)],
            initial={"x": 1},
            transitions=[
                Transition("inc", Bool(True), {"x": Add(V("x"), 1)}),
            ],
            terminal=terminal,
        )

    def test_fire_returns_none(self):
        m = self._model()
        trans = m.transitions[0]
        self.assertTrue(m.enabled(trans, m.initial))  # guard holds...
        self.assertIsNone(m.fire(trans, m.initial))    # ...but OOB disables

    def test_successors_empty_deadlock(self):
        m = self._model(terminal=None)
        self.assertEqual(list(m.successors(m.initial)), [])
        result = check_deadlock(m)
        self.assertEqual(result.status, FAIL)
        self.assertEqual(result.counterexample.prefix.transitions, [])
        self.assertEqual(result.counterexample.prefix.states, [(1,)])

    def test_terminal_nonaccepting_end_is_not_deadlock(self):
        m = self._model(terminal=Eq(V("x"), 1))
        self.assertEqual(check_deadlock(m).status, PASS)


class InitialStateFailureTests(unittest.TestCase):
    def test_safety_empty_path(self):
        m = Model(
            variables=[Variable("x", 0, 1)],
            initial={"x": 1},
            transitions=[Transition("a", Eq(V("x"), 1), {"x": 0})],
        )
        result = check_safety(m, Eq(V("x"), 0))
        self.assertEqual(result.status, FAIL)
        trace = result.counterexample.prefix
        self.assertEqual(trace.length, 0)
        self.assertEqual(trace.transitions, [])
        self.assertEqual(trace.states, [(1,)])

    def test_initial_deadlock_empty_path(self):
        m = Model(
            variables=[Variable("x", 0, 1)],
            initial={"x": 1},
            transitions=[Transition("a", Eq(V("x"), 0), {"x": 1})],
        )
        result = check_deadlock(m)
        self.assertEqual(result.status, FAIL)
        self.assertEqual(result.counterexample.prefix.length, 0)

    def test_initial_failure_reported_even_with_zero_budget(self):
        m = Model(
            variables=[Variable("x", 0, 1)],
            initial={"x": 1},
            transitions=[Transition("a", Bool(True), {"x": 0})],
        )
        result = check_safety(m, Eq(V("x"), 0),
                              state_budget=1, edge_budget=0)
        self.assertEqual(result.status, FAIL)


class TieBreakingTests(unittest.TestCase):
    def test_shortest_then_lex_smallest_ids(self):
        # From x=0 both ta (->1) and tb (->2) violate x<=0 in one step;
        # both are shortest, so the lexicographically smaller id wins.
        m = Model(
            variables=[Variable("x", 0, 2)],
            initial={"x": 0},
            transitions=[
                Transition("tb", Eq(V("x"), 0), {"x": 2}),
                Transition("ta", Eq(V("x"), 0), {"x": 1}),
            ],
        )
        result = check_safety(m, Le(V("x"), 0))
        self.assertEqual(result.status, FAIL)
        self.assertEqual(result.counterexample.prefix.transitions, ["ta"])

    def test_shortest_preferred_over_lex(self):
        # za reaches a violation in one step; aa only after two.  Shortest
        # length beats the lexicographically smaller id.
        m = Model(
            variables=[Variable("x", 0, 3)],
            initial={"x": 0},
            transitions=[
                Transition("za", Eq(V("x"), 0), {"x": 3}),
                Transition("aa", Eq(V("x"), 0), {"x": 1}),
                Transition("ab", Eq(V("x"), 1), {"x": 3}),
            ],
        )
        result = check_safety(m, Le(V("x"), 2))  # x=3 violates
        self.assertEqual(result.status, FAIL)
        self.assertEqual(result.counterexample.prefix.transitions, ["za"])


class SelfLoopTests(unittest.TestCase):
    def test_self_loop_is_liveness_violation(self):
        m = Model(
            variables=[Variable("x", 0, 0)],
            initial={"x": 0},
            transitions=[
                Transition("spin", Bool(True), {"x": 0}),
            ],
        )
        result = check_liveness(m)
        self.assertEqual(result.status, FAIL)
        ce = result.counterexample
        self.assertEqual(ce.prefix.transitions, [])
        self.assertEqual(ce.loop.transitions, ["spin"])
        self.assertEqual(ce.loop.states, [(0,), (0,)])
        replay_prefix_loop(m, ce)  # independent replay must accept it


class AcceptPrefixBadLoopTests(unittest.TestCase):
    def test_prefix_through_accept_into_nonaccepting_ring(self):
        # 0(accept) ->1 ->2(accept) ->3 <->4 (never accept again)
        m = Model(
            variables=[Variable("x", 0, 4)],
            initial={"x": 0},
            transitions=[
                Transition("t1", Eq(V("x"), 0), {"x": 1}),
                Transition("t2", Eq(V("x"), 1), {"x": 2}),
                Transition("t3", Eq(V("x"), 2), {"x": 3}),
                Transition("t4", Eq(V("x"), 3), {"x": 4}),
                Transition("t5", Eq(V("x"), 4), {"x": 3}),
            ],
            accept=Or(Eq(V("x"), 0), Eq(V("x"), 2)),
        )
        result = check_liveness(m)
        self.assertEqual(result.status, FAIL)
        ce = result.counterexample
        self.assertEqual(ce.prefix.transitions, ["t1", "t2", "t3"])
        self.assertEqual(ce.loop.transitions, ["t4", "t5"])
        self.assertTrue(ce.loop.length >= 1)
        self.assertEqual(ce.loop.states[0], ce.loop.states[-1])
        for state in ce.loop.states[:-1]:
            self.assertFalse(m.is_accept(state))
        # the prefix is allowed to pass through accepting states
        self.assertTrue(m.is_accept(ce.prefix.states[0]))
        self.assertTrue(m.is_accept(ce.prefix.states[2]))
        replay_prefix_loop(m, ce)


class FiniteTerminationTests(unittest.TestCase):
    def test_dead_end_is_not_liveness_violation(self):
        # Starts accepting, takes one step into a terminal non-accepting
        # state.  There is no infinite execution, so the liveness property
        # ("every infinite execution accepts infinitely often") holds.
        m = Model(
            variables=[Variable("x", 0, 1)],
            initial={"x": 1},
            transitions=[
                Transition("stop", Eq(V("x"), 1), {"x": 0}),
            ],
            accept=Eq(V("x"), 1),
            terminal=Eq(V("x"), 0),
        )
        self.assertEqual(check_liveness(m).status, PASS)
        # ...and it is not flagged as a deadlock either.
        self.assertEqual(check_deadlock(m).status, PASS)

    def test_nonterminal_dead_end_is_deadlock_but_not_liveness(self):
        m = Model(
            variables=[Variable("x", 0, 1)],
            initial={"x": 1},
            transitions=[
                Transition("stop", Eq(V("x"), 1), {"x": 0}),
            ],
        )
        self.assertEqual(check_liveness(m).status, PASS)
        self.assertEqual(check_deadlock(m).status, FAIL)


class BudgetTests(unittest.TestCase):
    def _chain(self):
        # 0 -> 1 -> ... -> 10 (x=10 is a safety violation), 11 states.
        return Model(
            variables=[Variable("x", 0, 10)],
            initial={"x": 0},
            transitions=[
                Transition("inc", Lt(V("x"), 10),
                           {"x": Add(V("x"), 1)}),
            ],
        )

    def test_default_budgets_pass_full_exploration(self):
        result = check_safety(self._chain(), Le(V("x"), 10))
        self.assertEqual(result.status, PASS)
        self.assertEqual(result.states_explored, 11)

    def test_state_budget_unknown_with_counts(self):
        result = check_safety(self._chain(), Le(V("x"), 10),
                              state_budget=3, edge_budget=50_000)
        self.assertEqual(result.status, UNKNOWN)
        self.assertLessEqual(result.states_explored, 3)
        self.assertIn("state budget", result.message)

    def test_edge_budget_unknown_with_counts(self):
        result = check_safety(self._chain(), Le(V("x"), 10),
                              state_budget=10_000, edge_budget=0)
        self.assertEqual(result.status, UNKNOWN)
        self.assertIn("edge budget", result.message)
        self.assertGreaterEqual(result.edges_explored, 1)

    def test_violation_beyond_budget_is_unknown_not_fail(self):
        # The violation at x=10 cannot be reached within a 3-state budget,
        # so the honest answer is unknown, not fail and not pass.
        result = check_safety(self._chain(), Le(V("x"), 9),
                              state_budget=3, edge_budget=50_000)
        self.assertEqual(result.status, UNKNOWN)

    def test_liveness_budget_unknown(self):
        m = Model(
            variables=[Variable("x", 0, 4)],
            initial={"x": 0},
            transitions=[
                Transition("inc", Lt(V("x"), 4),
                           {"x": Add(V("x"), 1)}),
                Transition("back", Eq(V("x"), 4), {"x": 0}),
            ],
        )
        result = check_liveness(m, state_budget=2)
        self.assertEqual(result.status, UNKNOWN)


class ValidationTests(unittest.TestCase):
    def test_too_many_variables(self):
        with self.assertRaises(ModelError):
            Model(
                variables=[Variable(f"v{i}", 0, 1) for i in range(7)],
                initial={f"v{i}": 0 for i in range(7)},
                transitions=[Transition("t", Bool(True),
                                        {"v0": Int(0)})],
            )

    def test_transition_count_bounds(self):
        one_var = [Variable("x", 0, 0)]
        init = {"x": 0}
        with self.assertRaises(ModelError):
            Model(variables=one_var, initial=init, transitions=[])
        with self.assertRaises(ModelError):
            Model(
                variables=one_var, initial=init,
                transitions=[Transition(f"t{i:02d}", Bool(True),
                                        {"x": 0}) for i in range(21)],
            )

    def test_duplicate_variable_and_transition_ids(self):
        with self.assertRaises(ModelError):
            Model(
                variables=[Variable("x", 0, 1), Variable("x", 0, 1)],
                initial={"x": 0},
                transitions=[Transition("t", Bool(True), {"x": 0})],
            )
        with self.assertRaises(ModelError):
            Model(
                variables=[Variable("x", 0, 1)],
                initial={"x": 0},
                transitions=[
                    Transition("t", Bool(True), {"x": 0}),
                    Transition("t", Bool(True), {"x": 0}),
                ],
            )

    def test_unknown_variable_rejected(self):
        with self.assertRaises(ModelError):
            Model(
                variables=[Variable("x", 0, 1)],
                initial={"x": 0},
                transitions=[Transition("t", Lt(V("y"), 1), {"x": 0})],
            )
        with self.assertRaises(ModelError):
            Model(
                variables=[Variable("x", 0, 1)],
                initial={"x": 0},
                transitions=[Transition("t", Bool(True), {"y": 0})],
            )

    def test_type_mismatches_rejected(self):
        good = [Variable("x", 0, 2)]
        init = {"x": 0}
        # guard must be boolean
        with self.assertRaises(ModelError):
            Model(variables=good, initial=init,
                  transitions=[Transition("t", Add(V("x"), 1), {"x": 0})])
        # arithmetic needs int operands
        with self.assertRaises(ModelError):
            Model(variables=good, initial=init,
                  transitions=[Transition("t", Bool(True),
                                          {"x": Add(V("x"), Bool(True))})])
        # comparison needs int operands
        with self.assertRaises(ModelError):
            Model(variables=good, initial=init,
                  transitions=[Transition("t", Lt(Bool(False), V("x")),
                                         {"x": 0})])
        # And needs boolean operands
        with self.assertRaises(ModelError):
            Model(variables=good, initial=init,
                  transitions=[Transition("t", And(Bool(True), V("x")),
                                         {"x": 0})])
        # Not needs a boolean operand
        with self.assertRaises(ModelError):
            Model(variables=good, initial=init,
                  transitions=[Transition("t", Not(V("x")), {"x": 0})])
        # assignment RHS must be integer typed
        with self.assertRaises(ModelError):
            Model(variables=good, initial=init,
                  transitions=[Transition("t", Bool(True),
                                          {"x": Bool(True)})])

    def test_callable_is_rejected_no_callbacks(self):
        with self.assertRaises(ModelError):
            Model(
                variables=[Variable("x", 0, 1)],
                initial={"x": 0},
                transitions=[Transition("t", lambda s: True,  # type: ignore
                                        {"x": 0})],
            )

    def test_initial_value_must_be_complete_and_in_bounds(self):
        with self.assertRaises(ModelError):
            Model(variables=[Variable("x", 0, 1)], initial={"x": 2},
                  transitions=[Transition("t", Bool(True), {"x": 0})])
        with self.assertRaises(ModelError):
            Model(variables=[Variable("x", 0, 1)], initial={},
                  transitions=[Transition("t", Bool(True), {"x": 0})])

    def test_ast_depth_limit(self):
        def nested(depth):
            node = V("x")
            for _ in range(depth - 1):
                node = Sub(node, Int(1))
            return node

        ok = Model(
            variables=[Variable("x", -1000, 1000)],
            initial={"x": 0},
            transitions=[Transition("t", Bool(True), {"x": nested(32)})],
        )
        self.assertEqual(len(ok.transitions), 1)
        with self.assertRaises(ModelError):
            Model(
                variables=[Variable("x", -1000, 1000)],
                initial={"x": 0},
                transitions=[Transition("t", Bool(True),
                                        {"x": nested(33)})],
            )

    def test_predicate_unknown_var_and_type(self):
        m = Model(
            variables=[Variable("x", 0, 1)],
            initial={"x": 0},
            transitions=[Transition("t", Bool(True), {"x": 0})],
        )
        with self.assertRaises(ModelError):
            check_safety(m, Lt(V("y"), 1))
        with self.assertRaises(ModelError):
            check_safety(m, Add(V("x"), 1))


class NoEvalSourceTests(unittest.TestCase):
    def test_no_eval_exec_or_dynamic_code_in_package(self):
        package_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        lib_dir = os.path.join(package_dir, "fsmcheck")
        pattern = re.compile(r"\b(eval|exec)\s*\(")
        for name in os.listdir(lib_dir):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(lib_dir, name), encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    self.assertIsNone(
                        pattern.search(line),
                        f"{name}:{lineno} uses eval/exec: {line.strip()}",
                    )


class ReplayTests(unittest.TestCase):
    def test_independent_replay_matches_step_states(self):
        m = Model(
            variables=[Variable("x", 0, 3)],
            initial={"x": 0},
            transitions=[
                Transition("inc", Lt(V("x"), 3), {"x": Add(V("x"), 1)}),
                Transition("dec", Gt(V("x"), 0), {"x": Sub(V("x"), 1)}),
            ],
        )
        trace = replay(m, ["inc", "inc", "dec"])
        self.assertEqual(trace.states, [(0,), (1,), (2,), (1,)])
        # state at every step is nameable as a dict
        self.assertEqual(m.state_dict(trace.states[2]), {"x": 2})

    def test_replay_rejects_bogus_id_and_failed_guard(self):
        m = Model(
            variables=[Variable("x", 0, 1)],
            initial={"x": 0},
            transitions=[Transition("a", Eq(V("x"), 1), {"x": 0})],
        )
        with self.assertRaises(ReplayError):
            replay(m, ["nope"])
        with self.assertRaises(ReplayError):
            replay(m, ["a"])  # guard x==1 does not hold at x=0

    def test_tampered_counterexample_fails_verification(self):
        from fsmcheck import Counterexample, Trace, verify_against_trace
        m = Model(
            variables=[Variable("x", 0, 2)],
            initial={"x": 0},
            transitions=[
                Transition("ok", Eq(V("x"), 0), {"x": 0}),
                Transition("bad", Eq(V("x"), 0), {"x": 2}),
            ],
        )
        # Claim that "ok" leads to the violating state x=2.
        fake = Counterexample(
            "safety",
            Trace(["ok"], [(0,), (2,)]),
            None,
            "fabricated",
        )
        with self.assertRaises(ReplayError):
            verify_against_trace(m, fake,
                                 safety_predicate=Le(V("x"), 1))

    def test_liveness_tampered_loop_closure_rejected(self):
        from fsmcheck import Counterexample, Trace
        m = Model(
            variables=[Variable("x", 0, 1)],
            initial={"x": 0},
            transitions=[
                Transition("a", Eq(V("x"), 0), {"x": 1}),
                Transition("b", Eq(V("x"), 1), {"x": 0}),
            ],
        )
        # Loop claims to close at x=0 but replays to x=1.
        fake = Counterexample(
            "liveness",
            Trace([], [(0,)]),
            Trace(["a"], [(0,), (0,)]),
            "fabricated",
        )
        with self.assertRaises(ReplayError):
            replay_prefix_loop(m, fake)


class ExpressionCoverageTests(unittest.TestCase):
    def test_all_operators(self):
        m = Model(
            variables=[Variable("x", 0, 5), Variable("y", 0, 5)],
            initial={"x": 2, "y": 3},
            transitions=[
                Transition("ge", Ge(V("y"), V("x")), {"x": 0, "y": 0}),
            ],
        )
        # Use a predicate exercising Ne, Gt, Or, Not, Sub in one go.
        pred = Or(
            And(Gt(V("y"), 0), Ne(V("x"), 9)),
            Not(Eq(Sub(V("y"), V("x")), 100)),
        )
        self.assertEqual(check_safety(m, pred).status, PASS)
        # Eq/Le/Lt/Add already covered elsewhere; evaluate a false case too.
        self.assertEqual(
            check_safety(m, Or(Eq(V("x"), 9), Lt(V("y"), 0))).status,
            FAIL,
        )


# ---------------------------------------------------------------------- #
# Fixed-seed randomized differential testing
# ---------------------------------------------------------------------- #
def _build_random_model(rng):
    """Build a small 1- or 2-variable model with +/- constant transitions."""
    var_count = rng.randint(1, 2)
    variables = []
    initial = {}
    for i in range(var_count):
        upper = rng.randint(1, 4)
        name = chr(ord("x") + i)
        variables.append(Variable(name, 0, upper))
        initial[name] = rng.randint(0, upper)
    transitions = []
    used_ids = set()
    for i in range(rng.randint(1, 6)):
        tid = f"t{i:02d}"
        if tid in used_ids:
            continue
        used_ids.add(tid)
        name = chr(ord("x") + rng.randrange(var_count))
        upper = dict((v.name, v.upper) for v in variables)[name]
        delta = rng.choice([-2, -1, 1, 2])
        guard = And(
            Le(Int(0), V(name)),          # always true structurally
            Le(V(name), Int(upper + 5)),  # also always true in range
        )
        assigns = {name: Add(V(name), delta)}
        transitions.append(Transition(tid, guard, assigns))
    if not transitions:
        transitions.append(Transition("t00", Bool(True), {"x": Int(0)}))
    accept = Eq(V("x"), rng.randint(0, variables[0].upper))
    terminal = And(Gt(V("x"), 0), Bool(False))  # never terminal
    return Model(variables=variables, initial=initial,
                 transitions=transitions, accept=accept, terminal=terminal)


def _reachable(m):
    """Independent reference reachability (plain set BFS)."""
    seen = {m.initial}
    frontier = [m.initial]
    while frontier:
        state = frontier.pop()
        for _, dst in m.successors(state):
            if dst not in seen:
                seen.add(dst)
                frontier.append(dst)
    return seen


def _reference_safety(m, predicate):
    expr = m.prepare_predicate(predicate, "ref safety")
    index = dict((n, i) for i, n in enumerate(m.variable_names))
    from fsmcheck.model import evaluate_bool
    return all(evaluate_bool(expr, s, index) for s in _reachable(m))


def _reference_deadlock(m):
    for state in _reachable(m):
        if next(m.successors(state), None) is None \
                and not m.is_terminal(state):
            return True
    return False


def _reference_liveness(m):
    """Violation iff a non-accepting reachable state can return to itself
    via >=1 edge staying entirely among non-accepting states."""
    nonaccept = {s for s in _reachable(m) if not m.is_accept(s)}

    def returns(state):
        stack = [(state, iter(m.successors(state)))]
        seen_edges = 0
        visited = {state}
        while stack:
            cur, it = stack[-1]
            advanced = False
            for tid, dst in it:
                seen_edges += 1
                if dst not in nonaccept:
                    continue
                if dst == state and seen_edges >= 1:
                    return True
                if dst not in visited:
                    visited.add(dst)
                    stack.append((dst, iter(m.successors(dst))))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
        return False

    return any(returns(s) for s in nonaccept)


def _lex_min_shortest_violation(m, predicate):
    """BFS reference for the lex-smallest shortest id sequence to a state
    where predicate is false."""
    from fsmcheck.model import evaluate_bool
    expr = m.prepare_predicate(predicate, "ref safety")
    index = dict((n, i) for i, n in enumerate(m.variable_names))

    def bad(s):
        return not evaluate_bool(expr, s, index)

    if bad(m.initial):
        return []
    # Layer BFS; parent stores the lex-min shortest sequence.
    best = {m.initial: []}
    frontier = [m.initial]
    while frontier:
        winners = []
        next_layer = {}
        for state in frontier:
            prefix = best[state]
            for tid, dst in m.successors(state):
                seq = prefix + [tid]
                if dst in best:  # reached in an earlier (shorter) layer
                    continue
                if dst not in next_layer or seq < next_layer[dst]:
                    next_layer[dst] = seq
        for dst, seq in next_layer.items():
            best[dst] = seq
            if bad(dst):
                winners.append(seq)
        if winners:
            return min(winners)
        frontier = list(next_layer)
    return None  # no violation


class RandomDifferentialTests(unittest.TestCase):
    def test_random_models_match_reference(self):
        rng = random.Random(20260920)
        checked = 0
        for _ in range(60):
            m = _build_random_model(rng)
            # safety predicate: x must not exceed a random threshold
            threshold = rng.randint(0, 3)
            pred = Le(V("x"), threshold)

            safety = check_safety(m, pred)
            if safety.status == PASS:
                self.assertTrue(_reference_safety(m, pred))
                self.assertIsNone(_lex_min_shortest_violation(m, pred))
            elif safety.status == FAIL:
                self.assertFalse(_reference_safety(m, pred))
                ids = safety.counterexample.prefix.transitions
                self.assertEqual(ids, _lex_min_shortest_violation(m, pred))
                # independently replay the reported path
                replayed = replay(m, ids)
                self.assertEqual(replayed.states[-1],
                                 safety.counterexample.prefix.states[-1])

            dl = check_deadlock(m)
            self.assertEqual(dl.status == FAIL, _reference_deadlock(m))

            lv = check_liveness(m)
            self.assertEqual(lv.status == FAIL, _reference_liveness(m))
            if lv.status == FAIL:
                replay_prefix_loop(m, lv.counterexample)
            checked += 1
        self.assertEqual(checked, 60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
