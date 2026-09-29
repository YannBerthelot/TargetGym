"""The version-2 reward contract, checked on every registered environment.

``docs/reward-shaping.md`` states it: the reward is minus a sum of
non-negative cost terms -- floor-normalised tracking, running cost above the
hold-phase consumption, a failure charge -- with no product between them, no
concave tracking shape, and every number in it documented with the script or
source that produced it. These tests hold the code to that page.
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import reward as R
from target_gym.provenance import BASELINES_PATH
from target_gym.registry import REGISTRY

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "target_gym"

SPECS = list(REGISTRY.values())
IDS = [s.name for s in SPECS]

# Parameters the reward reads, whose value must be traceable in PHYSICS.md.
REWARD_PARAM_PREFIXES = ("e_floor", "e_tol", "c_hold", "rho_floor")
REWARD_PARAM_NAMES = {
    "tracking_exponent",
    "running_weight",
    "fatigue_weight",
    "rod_wear_weight",
    "imbalance_price",
    "fade_price",
    "gas_price",
    "comfort_price",
    "comfort_tolerance",
    "failure_cost",
    "restart_steps",
    "restart_in_place",
}


def _reward_params(params) -> list[str]:
    return sorted(
        k
        for k in vars(params)
        if k.startswith(REWARD_PARAM_PREFIXES) or k in REWARD_PARAM_NAMES
    )


def _a_state(spec, seed=0, steps=5):
    env = spec.make_env()
    p = spec.make_test_params()
    obs, state = env.reset_env(jax.random.PRNGKey(seed), p)
    step = jax.jit(env.step_env)
    a = jnp.zeros(
        np.atleast_1d(env.action_space(p).sample(jax.random.PRNGKey(0))).shape
    )
    for _ in range(steps):
        obs, state, _, _, _ = step(jax.random.PRNGKey(0), state, a, p)
    return env, p, state


@pytest.fixture(params=SPECS, ids=IDS)
def spec(request):
    return request.param


def test_reward_is_minus_the_sum_of_non_negative_terms(spec):
    """Additive by construction: the scalar reward equals ``-sum(terms)`` and
    each term is a cost (>= 0), on a real state."""
    env, p, state = _a_state(spec)
    terms = env.reward_terms(state, p)
    for name, value in terms.items():
        assert float(value) >= 0.0, f"{spec.name}: term {name} is negative"
    assert float(env.compute_reward(state, p)) == pytest.approx(
        -float(sum(terms.values())), rel=1e-5, abs=1e-6
    )


def test_tracking_and_running_terms_have_no_cross_term(spec):
    """Scaling the running cost's parameters leaves the tracking term exactly
    where it was, and vice versa: no product between the two."""
    env, p, state = _a_state(spec)
    base = env.reward_terms(state, p)
    if "running" not in base:
        pytest.skip(f"{spec.name}: no running cost")
    running_knobs = {
        k: v * 3.0
        for k, v in vars(p).items()
        if k
        in {
            "running_weight",
            "fatigue_weight",
            "rod_wear_weight",
            "gas_price",
            "fade_price",
        }
    }
    if running_knobs:
        scaled = env.reward_terms(state, p.replace(**running_knobs))
        assert float(scaled["tracking"]) == float(base["tracking"])
    # The tracking shape's own knobs (tolerance, exponent) never reach the
    # running term. (Where a wear price is pegged to the tracking floor cost --
    # reactor, wind -- ``e_floor`` scales both by design, so it is not used.)
    tracking_knobs = {
        k: v * 3.0 + 0.1
        for k, v in vars(p).items()
        if k in {"e_tol", "e_tol_altitude", "comfort_tolerance", "tracking_exponent"}
    }
    scaled = env.reward_terms(state, p.replace(**tracking_knobs))
    assert float(scaled["running"]) == pytest.approx(float(base["running"]), rel=1e-6)


@pytest.mark.parametrize("p_exp", [1.0, 2.0])
@pytest.mark.parametrize("e_tol", [0.0, 0.5])
def test_tracking_cost_is_convex_in_the_error(p_exp, e_tol):
    """Second finite differences of the tracking cost are non-negative on a
    grid through the dead zone: convex, never concave (the log shape was)."""
    e = np.linspace(-5.0, 5.0, 401)
    c = np.asarray(R.tracking_cost(e, 0.3, e_tol, p_exp, xp=np), dtype=np.float64)
    second = c[2:] - 2 * c[1:-1] + c[:-2]
    assert second.min() >= -1e-9


def test_version2_reward_has_no_log_shape_and_no_precision_floor(spec):
    """The terms function must not call the log-scaled shape or read the
    version-1 resolution floor."""
    env = spec.make_env()
    fn = env.reward_terms
    module = inspect.getmodule(inspect.unwrap(fn))
    # The env's own module (env_jax) delegates to env.py; check every
    # ``compute_reward_terms*`` function in the plant's package.
    package_dir = pathlib.Path(inspect.getfile(module)).parent
    sources = "\n".join(
        f.read_text()
        for f in package_dir.glob("*.py")
        if not f.name.startswith("rendering")
    )
    tree = ast.parse(sources)
    found = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith(
            "compute_reward_terms"
        ):
            found += 1
            body = ast.unparse(node)
            assert (
                "log_scaled_reward" not in body
            ), f"{spec.name}: {node.name} uses the log shape"
            assert (
                "precision_floor" not in body
            ), f"{spec.name}: {node.name} reads precision_floor"
    assert found >= 1, f"{spec.name}: no compute_reward_terms function"


def test_reward_version_1_is_still_the_shipped_reward(spec):
    """Every plant keeps its version-1 reward constructible: on a real state
    it is bounded in [0, 1] as the capped log reward was, while version 2 is
    a cost."""
    env, p, state = _a_state(spec)
    v1 = float(env.compute_reward(state, p.replace(reward_version=1)))
    v2 = float(env.compute_reward(state, p))
    assert 0.0 <= v1 <= 1.0 + 1e-6
    assert v2 <= 0.0


@pytest.mark.parametrize("name", ["first_order", "cstr"])
def test_version_1_reproduces_its_recorded_baseline(name):
    """The v1 baseline file is kept, and the v1 reward reproduces it."""
    from target_gym.runners.runners import baseline_policy, rollout

    v1 = json.loads((BASELINES_PATH.parent / "baseline_returns_v1.json").read_text())[
        name
    ]
    spec = REGISTRY[name]
    p = spec.make_test_params(reward_version=1)
    for seed in range(2):
        _, _, r = rollout(spec, p, baseline_policy(spec, "pid", p), seed=seed)
        assert float(r.sum()) == pytest.approx(v1["pid_returns"][seed], abs=1e-3)


def test_reward_parameters_are_documented(spec):
    """Every floor, tolerance, hold-phase consumption, weight and price the
    reward reads is named in the plant's PHYSICS.md next to the script or
    source it came from."""
    env = spec.make_env()
    module = inspect.getmodule(type(env))
    physics = pathlib.Path(inspect.getfile(module)).parent / "PHYSICS.md"
    assert physics.exists(), f"{spec.name}: no PHYSICS.md"
    text = physics.read_text()
    p = spec.make_test_params()
    missing = [k for k in _reward_params(p) if f"`{k}`" not in text]
    assert not missing, f"{spec.name}: undocumented reward parameters {missing}"
    # Where the table states a number for a scalar parameter, it must be the
    # parameter's value (a name in backticks alone documents nothing). Rows
    # are ``| `name` | value | source |``; the first number of the value cell
    # is compared to within 2%, or exactly for booleans. Rows whose value is
    # prose (a formula, "documented minimum") are not checked.
    wrong = []
    for k in _reward_params(p):
        v = getattr(p, k)
        if not isinstance(v, (int, float, bool)) or isinstance(v, bool):
            continue
        m = re.search(rf"^\| `{re.escape(k)}`(?:, `[^`]+`)* \| ([^|]*) \|", text, re.M)
        if not m:
            continue
        cell = m.group(1)
        if not re.match(r"\s*[-+]?\d", cell) or re.match(r"\s*\d+(\.\d+)? x ", cell):
            continue  # prose or a formula ("2 x the envelope's cost")
        nums = [
            float(x.replace(" ", "").replace("_", ""))
            for x in re.findall(
                r"[-+]?\d(?:[\d_]|\s(?=\d))*(?:\.\d+)?(?:[eE][-+]?\d+)?", cell
            )
        ]
        # A cell may list one value per task (the aircraft's c_hold): any match.
        if not any(abs(n - float(v)) <= 0.02 * max(abs(float(v)), 1e-12) for n in nums):
            wrong.append(f"{k}: PHYSICS.md says {nums}, params say {v}")
    assert not wrong, f"{spec.name}: {wrong}"
    assert re.search(
        r"measure_hold\.py|floor_\w+\.py|closed form|documented minimum", text
    ), f"{spec.name}: PHYSICS.md does not say how the floor was obtained"


@pytest.mark.slow
def test_mpc_does_not_beat_a_measured_floor(spec):
    """Floor sanity: where ``e_floor`` is a measured or certified floor (not a
    documented minimum on a deterministic plant), the shipped MPC's long-run
    tracking cost after burn-in is at or above the floor's cost. A floor the
    MPC beats is wrong."""
    from target_gym.eval import evaluate_controller

    p = spec.make_test_params()
    if spec.make_mpc is None or getattr(p, "floor_is_documented_minimum", True):
        pytest.skip(f"{spec.name}: no measured floor to check against")
    m = evaluate_controller(spec.name, "mpc", seeds=2)
    assert m["tracking"] >= 0.98 * float(p.rho_floor_tracking), (
        f"{spec.name}: MPC tracking cost {m['tracking']:.4g} below the floor "
        f"{float(p.rho_floor_tracking):.4g}"
    )


def test_a_trip_is_charged_once_and_the_plant_restarts_at_once(spec):
    """``base.failure_kernel``: when the physics' proposal leaves the envelope
    the step is scored on that proposal -- every term zero, the failure term
    the trip cost ``restart_steps * failure_cost`` -- and the state the window
    continues from is a fresh draw on the same clock. A proposal inside the
    envelope passes through unchanged."""
    from target_gym.base import failure_kernel

    env, p, state = _a_state(spec)
    if getattr(p, "restart_in_place", False):
        pytest.skip(f"{spec.name}: restarts in place (its own test below)")
    key = jax.random.PRNGKey(7)
    proposal = state.replace(time=state.time + 1)

    class Trips:
        is_terminated = staticmethod(lambda s, params: jnp.ones((), bool))
        reset_env = staticmethod(env.reset_env)

    out, tripped, scored = failure_kernel(Trips(), key, state, proposal, p)
    assert bool(tripped)
    assert int(out.time) == int(proposal.time)
    assert scored is proposal
    fresh = env.reset_env(key, p)[1]
    for name in vars(fresh):
        if name == "time":
            continue
        np.testing.assert_array_equal(
            np.asarray(getattr(out, name)), np.asarray(getattr(fresh, name))
        )
    out, tripped, scored = failure_kernel(env, key, state, proposal, p)
    assert not bool(tripped)
    assert scored is proposal
    for name in vars(proposal):
        np.testing.assert_array_equal(
            np.asarray(getattr(out, name)), np.asarray(getattr(proposal, name))
        )
    # The reward's failure term is the trip cost, and nothing else, on a
    # state outside the envelope; zero on one inside.
    terms = env.reward_terms(state, p)
    assert float(terms["failure"]) == 0.0
    assert float(R.trip_cost(p)) == float(p.restart_steps) * float(p.failure_cost)
    out_of_envelope = R.with_trip(terms, jnp.ones((), bool), R.trip_cost(p))
    assert float(out_of_envelope["failure"]) == pytest.approx(float(R.trip_cost(p)))
    assert all(float(v) == 0.0 for k, v in out_of_envelope.items() if k != "failure")


def test_a_plant_that_restarts_in_place_keeps_its_state_and_pays_every_step():
    """``restart_in_place``: the building keeps its state through a trip --
    the proposal is both the state the window continues from and the state
    scored -- so every step outside the envelope is charged."""
    from target_gym.base import failure_kernel

    spec = REGISTRY["hvac"]
    env, p, state = _a_state(spec)
    assert bool(p.restart_in_place)
    proposal = state.replace(time=state.time + 1, T_air=jnp.asarray(p.T_air_max + 1.0))
    out, tripped, scored = failure_kernel(
        env, jax.random.PRNGKey(0), state, proposal, p
    )
    assert bool(tripped)
    assert scored is proposal
    assert float(out.T_air) == pytest.approx(float(p.T_air_max) + 1.0)
    terms = env.reward_terms(out, p)
    assert float(terms["failure"]) == pytest.approx(float(R.trip_cost(p)))
    assert float(terms["tracking"]) == 0.0 and float(terms["running"]) == 0.0
