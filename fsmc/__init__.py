"""fsmc -- a small finite-state model checking library.

Explicit-state exploration of models with up to six named bounded integer
variables and at most twenty uniquely identified guarded transitions. Finds
safety violations, deadlocks and liveness (acceptance) counterexamples with
step-by-step replay, and reports UNKNOWN when state/edge budgets run out.
"""

from .astx import (
    MAX_DEPTH,
    AstError,
    add,
    const,
    eq,
    false,
    ge,
    gt,
    land,
    le,
    lnot,
    lor,
    lt,
    ne,
    sub,
    true,
    variable,
)
from .checker import (
    DEFAULT_EDGE_BUDGET,
    DEFAULT_STATE_BUDGET,
    Budgets,
    Counterexample,
    Result,
    ReplayError,
    Status,
    Step,
    check_deadlock,
    check_liveness,
    check_safety,
)
from .model import MAX_TRANSITIONS, MAX_VARS, Bounds, Model, ModelError, Transition

__all__ = [
    # AST
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
    "lnot",
    "land",
    "lor",
    "true",
    "false",
    # model
    "Model",
    "Transition",
    "Bounds",
    "ModelError",
    "MAX_VARS",
    "MAX_TRANSITIONS",
    # checking
    "Status",
    "Budgets",
    "Counterexample",
    "Step",
    "Result",
    "ReplayError",
    "check_safety",
    "check_deadlock",
    "check_liveness",
    "DEFAULT_STATE_BUDGET",
    "DEFAULT_EDGE_BUDGET",
]
