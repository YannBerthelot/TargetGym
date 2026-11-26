"""The pH task's ceiling: the perfect hold on the permit's low edge, less what the
tanks hold above it at the start.

Contract (owner, 2026-11-23 and 2026-11-24): the ceiling is the reagent that holds the
discharge exactly on the permit's low edge, pH 6.00, against the episode's own influent
flow, buffer flow and acid strength: the steady balance's base flow at every step
(``hold``). An episode starts on the reference's set point, above the edge, so the two
tanks then hold base the edge does not need (``inventory``, in litres). A policy can
spend it once, and none that stays in the permit can spend more: the ceiling's ``cost``
per step is the hold's less the inventory spread over the episode. The episode is named
by its keys, the reset's and each step's, which the caller draws as the scoring runner
does (``benchmark.episode_keys``): the oracle module draws nothing itself.

Where the expected values come from, none of them the function under test:

- the base flow that leaves a mix of the three streams at a pH is the root, in the
  base flow, of the charge balance in ``env.py``'s docstring on the streams'
  flow-weighted invariants (PHYSICS.md section 1), found here by Brent's method in
  float64;
- a plant dosed with those flows from a start on the edge stays on the edge: the
  plant itself is the check, through ``env.step_env``;
- the inventory by counting base: a tank at rest at pH 6.5 is the steady mix there, a
  fraction ``f = q3 / (q1 + q2 + q3)`` of it base. The water in it (the rest) needs
  ``q3_edge / (q1 + q2)`` of base per unit to sit on the edge. The difference, on both
  tanks' volumes, is what the tanks hold above the edge.
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


def _keys(seed, steps):
    """One episode's reset key and step keys, as ``run_policy`` draws them."""
    resets, per_step = episode_keys(jax.random.PRNGKey(seed), 1, steps)
    return resets[0], per_step[:, 0]


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
    hold = perfect_hold(spec.make_env(), p, *_keys(0, 60))
    # Nothing moves: every step costs the nominal mix's base flow at pH 6, in litres
    # over one 5 s step at 1 L per L.
    litres = _base_flow_for(6.0, 16.6, 0.55, 1.0, p) * 5.0 / 1000.0
    assert hold["hold"] == pytest.approx(litres, rel=1e-4)
    assert hold["feasible"]


def test_the_ceiling_spends_once_the_base_the_tanks_hold_above_the_edge(spec):
    from target_gym.pc_gym.ph_neutralization.oracle import perfect_hold

    # A quiet plant at rest on the shipped set point, pH 6.5, for 2 h.
    p = spec.make_test_params(
        q2_noise_std=0.0, acid_upset_size=0.0, flow_swing=0.0, setpoint_pH=6.5
    )
    steps = int(p.max_steps_in_episode)
    hold = perfect_hold(spec.make_env(), p, *_keys(0, steps))
    water = 16.6 + 0.55  # mL/s of influent and buffer
    at_start = _base_flow_for(6.5, 16.6, 0.55, 1.0, p)
    on_edge = _base_flow_for(6.0, 16.6, 0.55, 1.0, p)
    base = at_start / (water + at_start)  # the base fraction of a tank at rest
    # Two tanks of 20 L: the base they hold, less what their water needs on the edge.
    litres = 2 * 20.0 * (base - (1.0 - base) * on_edge / water)
    # 1e-3: a small difference of float32 fractions.
    assert hold["inventory"] == pytest.approx(litres, rel=1e-3)
    assert 1.0 < litres < 4.0  # litres of base, of some 87 the hold uses in 2 h
    # At 1 L per L, the ceiling is the hold less that base spread over the episode.
    assert hold["cost"] == pytest.approx(hold["hold"] - litres / steps, rel=1e-4)


@pytest.mark.parametrize("seed", [0, 1])
def test_a_plant_dosed_with_the_hold_stays_on_the_edge_at_the_holds_cost(spec, seed):
    from target_gym.pc_gym.ph_neutralization.oracle import perfect_hold

    env = spec.make_env()
    # The scored bad day (acid strength and flow both move, the buffer drifts), half
    # an hour of it, starting on the edge.
    p = spec.make_test_params(setpoint_pH=6.0, max_steps_in_episode=360)
    reset_key, step_keys = _keys(seed, 360)
    hold = perfect_hold(env, p, reset_key, step_keys)
    assert hold["feasible"], "pick a seed whose hold the stage-1 valve can pass"
    # Started on the edge, the tanks hold nothing above it: the ceiling is the hold.
    assert hold["inventory"] == pytest.approx(0.0, abs=1e-3)

    # The same episode, replayed: the hold names it by the keys it was given.
    _, state = env.reset_env(reset_key, p)
    raw = 2.0 * (jnp.asarray(hold["flow"]) - p.q3_min) / (p.q3_max - p.q3_min) - 1.0
    actions = jnp.stack([raw, -jnp.ones_like(raw)], axis=1)  # the trim shut

    def step(state, ka):
        key, action = ka
        _, state, _, _, _ = env.step_env(key, state, action, p)
        return state, (state.pH, env.reward_terms(state, p)["reagent"])

    _, (pH, reagent) = jax.jit(lambda s: jax.lax.scan(step, s, (step_keys, actions)))(
        state
    )
    np.testing.assert_allclose(np.asarray(pH), 6.0, atol=EDGE_TOLERANCE)
    # The reagent the plant was charged is the hold's cost.
    assert float(np.asarray(reagent).mean()) == pytest.approx(hold["hold"], rel=1e-5)
