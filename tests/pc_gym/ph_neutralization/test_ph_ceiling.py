"""The pH task's ceiling: the perfect hold on the permit's low edge.

Contract (owner, 2026-11-23): the ceiling is the reagent that holds the discharge
exactly on the permit's low edge, pH 6.00, against the episode's own influent flow,
buffer flow and acid strength. It is the steady balance's base flow at every step, so
a window's start or end gains it nothing.

Where the expected values come from, none of them the function under test:

- the base flow that leaves a mix of the three streams at pH 6 is the root, in the
  base flow, of the charge balance in ``env.py``'s docstring on the streams'
  flow-weighted invariants (PHYSICS.md section 1), found here by Brent's method in
  float64;
- a plant dosed with those flows from a start on the edge stays on the edge: the
  plant itself is the check, through ``env.step_env``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from target_gym import registry
from target_gym.benchmark import episode_keys

#: Half the electrode's 0.01 pH resolution: held to within what a probe can read.
EDGE_TOLERANCE = 0.005


def _pH(Wa, Wb, p):
    """The charge balance's root in float64 (the equation of env.py's docstring)."""

    def balance(pH):
        carbonate = (1 + 2 * 10 ** (pH - p.pK2)) / (
            1 + 10 ** (p.pK1 - pH) + 10 ** (pH - p.pK2)
        )
        return Wa + 10 ** (pH - 14) - 10**-pH + Wb * carbonate

    return brentq(balance, 0.0, 14.0, xtol=1e-14)


def _base_flow_for(pH, q1, q2, acid, p):
    """The base flow whose steady mix with ``q1`` of acid at ``acid`` times nominal
    and ``q2`` of buffer sits at ``pH``: the streams' invariants mix by flow."""
    Wa1 = float(p.Wa1) * acid

    def miss(q3):
        total = q1 + q2 + q3
        Wa = (q1 * Wa1 + q2 * float(p.Wa2) + q3 * float(p.Wa3)) / total
        Wb = (q1 * float(p.Wb1) + q2 * float(p.Wb2) + q3 * float(p.Wb3)) / total
        return _pH(Wa, Wb, p) - pH

    return brentq(miss, 0.0, 60.0, xtol=1e-12)


@pytest.fixture(scope="module")
def spec():
    return registry.get("ph_neutralization")


@pytest.mark.parametrize(
    "q1, q2, acid",
    # Nominal flows; strong acid in a high flow with no buffer; weak acid in a low
    # flow with heavy buffering.
    [(16.6, 0.55, 1.0), (19.92, 0.0, 1.2), (13.28, 1.25, 0.8)],
)
def test_the_hold_flow_leaves_the_mix_on_the_permits_low_edge(spec, q1, q2, acid):
    from target_gym.pc_gym.ph_neutralization.oracle import hold_flow

    p = spec.make_test_params()
    flow = float(hold_flow(float(p.permit_low), q1, q2, acid, p))
    # 1e-4: float32 on a balance whose terms differ by three orders of magnitude.
    assert flow == pytest.approx(_base_flow_for(6.0, q1, q2, acid, p), rel=1e-4)


def test_on_a_quiet_plant_the_hold_costs_the_steady_balances_reagent(spec):
    from target_gym.pc_gym.ph_neutralization.oracle import perfect_hold

    p = spec.make_test_params(
        q2_noise_std=0.0, acid_upset_size=0.0, flow_swing=0.0, max_steps_in_episode=60
    )
    hold = perfect_hold(spec.make_env(), p, seed=0)
    # Nothing moves: every step costs the nominal mix's base flow at pH 6, in litres
    # over one 5 s step at 1 L per L.
    litres = _base_flow_for(6.0, 16.6, 0.55, 1.0, p) * 5.0 / 1000.0
    assert hold["cost"] == pytest.approx(litres, rel=1e-4)
    assert hold["feasible"]


@pytest.mark.parametrize("seed", [0, 1])
def test_a_plant_dosed_with_the_hold_stays_on_the_edge_at_the_holds_cost(spec, seed):
    from target_gym.pc_gym.ph_neutralization.oracle import perfect_hold

    env = spec.make_env()
    # The scored bad day (acid strength and flow both move, the buffer drifts), half
    # an hour of it, starting on the edge.
    p = spec.make_test_params(setpoint_pH=6.0, max_steps_in_episode=360)
    hold = perfect_hold(env, p, seed=seed)
    assert hold["feasible"], "pick a seed whose hold the stage-1 valve can pass"

    resets, steps = episode_keys(jax.random.PRNGKey(seed), 1, 360)
    _, state = env.reset_env(resets[0], p)
    raw = 2.0 * (jnp.asarray(hold["flow"]) - p.q3_min) / (p.q3_max - p.q3_min) - 1.0
    actions = jnp.stack([raw, -jnp.ones_like(raw)], axis=1)  # the trim shut

    def step(state, ka):
        key, action = ka
        _, state, _, _, _ = env.step_env(key, state, action, p)
        return state, (state.pH, env.reward_terms(state, p)["reagent"])

    _, (pH, reagent) = jax.jit(lambda s: jax.lax.scan(step, s, (steps[:, 0], actions)))(
        state
    )
    np.testing.assert_allclose(np.asarray(pH), 6.0, atol=EDGE_TOLERANCE)
    # The reagent the plant was charged is the hold's cost.
    assert float(np.asarray(reagent).mean()) == pytest.approx(hold["cost"], rel=1e-5)
