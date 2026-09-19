"""fsmcheck: a small finite-state explicit model checker.

Public API
----------
Model building:
    Variable, Transition, Model, ModelError

Expressions (small AST, no eval, no callbacks):
    Int, Bool, Var, Add, Sub, Lt, Le, Gt, Ge, Eq, Ne, And, Or, Not

Checks (each returns :class:`CheckResult`):
    check_safety, check_deadlock, check_liveness

Result objects:
    CheckResult, Counterexample, Trace, ReplayError
    PASS, FAIL, UNKNOWN

Typical use::

    from fsmcheck import Model, Variable, Transition, Var as V, Add, Le
    from fsmcheck import check_safety

    model = Model(
        variables=[Variable("x", 0, 3)],
        initial={"x": 0},
        transitions=[Transition("inc", Le(V("x"), 2), {"x": Add(V("x"), 1)})],
    )
    print(check_safety(model, Le(V("x"), 3)))
"""

from .ast import (
    Node, Int, Bool, Var,
    Add, Sub, Lt, Le, Gt, Ge, Eq, Ne, And, Or, Not,
)
from .model import (
    Variable, Transition, Model, ModelError,
    MAX_VARIABLES, MAX_TRANSITIONS, MAX_AST_DEPTH,
)
from .graph import (
    CheckResult, Counterexample, Trace, ReplayError, Exploration,
    PASS, FAIL, UNKNOWN,
    DEFAULT_STATE_BUDGET, DEFAULT_EDGE_BUDGET,
    explore, replay, replay_prefix_loop, verify_against_trace,
)
from .checker import check_safety, check_deadlock, check_liveness

__all__ = [
    # AST
    "Node", "Int", "Bool", "Var",
    "Add", "Sub", "Lt", "Le", "Gt", "Ge", "Eq", "Ne", "And", "Or", "Not",
    # model
    "Variable", "Transition", "Model", "ModelError",
    "MAX_VARIABLES", "MAX_TRANSITIONS", "MAX_AST_DEPTH",
    # graph / results
    "CheckResult", "Counterexample", "Trace", "ReplayError", "Exploration",
    "PASS", "FAIL", "UNKNOWN",
    "DEFAULT_STATE_BUDGET", "DEFAULT_EDGE_BUDGET",
    "explore", "replay", "replay_prefix_loop", "verify_against_trace",
    # checks
    "check_safety", "check_deadlock", "check_liveness",
]
