import numpy as np
import pytest

from oracle import available_selectors, make_selector, RoutingContext
from oracle.routing import LinUCB, TypeUCB, ACRouterSelector, HashingFeatures


def ctx(prompt="x", feats=None, task_type=None):
    return RoutingContext(program_id="p", prompt=prompt, features=feats, task_type=task_type)


def test_registry_has_builtins():
    names = set(available_selectors())
    assert {"linucb", "ucb", "lints", "epsilon_greedy", "acrouter", "routellm", "fixed", "random", "round_robin", "callable"} <= names


def test_linucb_learns_context():
    rng = np.random.default_rng(0)
    sel = LinUCB(["strong", "weak"], dim=4, alpha_ucb=0.5, seed=0)
    # strong is better when x[0] > 0, weak when x[0] < 0
    for _ in range(400):
        x = np.array([rng.choice([-1.0, 1.0]), rng.normal(), rng.normal(), 1.0])
        m = sel.select(ctx(feats=x))
        r = (1.0 if x[0] > 0 else 0.3) if m == "strong" else (0.3 if x[0] > 0 else 1.0)
        sel.update(ctx(feats=x), m, r)
    assert sel.select(ctx(feats=np.array([1.0, 0, 0, 1.0]))) == "strong"
    assert sel.select(ctx(feats=np.array([-1.0, 0, 0, 1.0]))) == "weak"
    assert sel.estimate(ctx(feats=np.array([1.0, 0, 0, 1.0])), "strong") > sel.estimate(ctx(feats=np.array([1.0, 0, 0, 1.0])), "weak")


def test_type_ucb_per_cell():
    sel = TypeUCB(["strong", "weak"], delta=0.1)
    for _ in range(100):
        for t, good in (("swe", "strong"), ("tau2", "weak")):
            m = sel.select(ctx(task_type=t))
            sel.update(ctx(task_type=t), m, 1.0 if m == good else 0.0)
    assert sel.estimate(ctx(task_type="swe"), "strong") > sel.estimate(ctx(task_type="swe"), "weak")
    assert sel.estimate(ctx(task_type="tau2"), "weak") > sel.estimate(ctx(task_type="tau2"), "strong")


def test_acrouter_memory_rule():
    sel = ACRouterSelector(["strong", "weak"], k=5, min_sim=0.1)
    assert sel.select(ctx("book a flight to Paris")) == "strong"  # empty memory -> first model
    for _ in range(5):
        sel.update(ctx("book a flight to Rome"), "weak", 1.0)
        sel.update(ctx("book a flight to Rome"), "strong", 1.0)
    assert sel.select(ctx("book a flight to Paris")) == "weak"  # tie -> cheapest
    sel2 = ACRouterSelector(["strong", "weak"], k=5, min_sim=0.1)
    for _ in range(5):
        sel2.update(ctx("fix the failing unit test"), "weak", 0.0)
        sel2.update(ctx("fix the failing unit test"), "strong", 1.0)
    assert sel2.select(ctx("fix the broken unit test")) == "strong"


def test_make_selector_kwargs_and_errors():
    s = make_selector("fixed", ["a", "b"], model="b")
    assert s.select(ctx()) == "b"
    with pytest.raises(KeyError):
        make_selector("nope", ["a"])
    with pytest.raises(KeyError):
        s.update(ctx(), "zzz", 1.0)


def test_hashing_features_shape():
    f = HashingFeatures(dim=32)
    x = f("fix the test", {})
    assert x.shape == (32,) and x[-1] == 1.0
