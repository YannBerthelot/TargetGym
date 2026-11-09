"""The glass furnace's oracle plans the task's own reward on the episode it will meet.

Contract (charter J3, the task's oracle rule; ``glass_furnace/oracle.py``):
- the cost the oracle reports for a plan is the cost the env charges when that plan is
  replayed through the benchmark's runner on the same seed;
- that cost is the reward's: on a step that loses no glass, the fuel burnt at its heating
  value and the gas price (env_jax.compute_reward's docstring, REALITY.md section 6);
- a solve meets the fining and canal limits on every step and stops at a KKT point;
- with a horizon shorter than the fining's dead time (11.8 h, the ladder row's step
  test) and a canal that its own loop holds, no valve move can reach a limit, so the
  cheapest plan is the valve closed to its minimum.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.benchmark import run_policy, task_info
from target_gym.glass_furnace import env_jax as E
from target_gym.glass_furnace import oracle as O
from target_gym.registry import REGISTRY

SPEC = REGISTRY["glass_furnace"]
# Each test compiles the 2D plant's rollout, and the solves its Jacobian: 2 to 12 s.
pytestmark = pytest.mark.medium


def _replayed(env, params, seed, valve):
    """Mean cost per step when ``valve`` (kg/s per step) is replayed on ``seed``."""

    class Plan:
        oracle = True

        def reset(self, key, obs, task):
            return None

        def act(self, carry, obs, t, state=None):
            return np.asarray(E.to_action(valve[t], params))[None], carry

    task = task_info("glass_furnace", env, params, 1)
    out = run_policy(Plan(), env, params, task, seeds=[seed])
    return float(-out["reward"][0].mean())


def test_a_step_without_lost_glass_costs_its_fuel_at_the_gas_price():
    # No walk and no event: the gas sits at its nominal heating value, so each step
    # costs fuel * mean firing * 120 s * LHV / 3.6e9 J/MWh * 60.5 EUR/MWh.
    env = SPEC.make_env()
    p = SPEC.make_test_params(max_steps_in_episode=4, walk_sd=(0.0,) * 5)
    reset, keys = O.episode_keys(7, 4)
    _, state = env.reset_env(reset, p)
    cost, running, fining, canal, _ = O.rollout(env, p, state, keys, jnp.full(4, 0.5))
    assert np.all(np.asarray(fining) >= p.fining_pass)  # the design state passes
    assert np.all(np.asarray(canal) >= p.canal_target)
    by_hand = 0.5 * float(E.P.mean_firing(p.plant)) * 120.0 * p.plant.LHV / 3.6e9 * 60.5
    np.testing.assert_allclose(np.asarray(cost), by_hand, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(running), by_hand, rtol=1e-5)


def test_the_planned_cost_of_a_plan_is_its_replayed_cost():
    # A stepped valve on a seed with its walks and a gas swing: the oracle's rollout
    # and the runner's step-by-step replay at sensor noise 1 charge the same episode.
    # Tolerance: float32 rounding over 8 steps of a ~46 EUR cost.
    env = SPEC.make_env()
    p = SPEC.make_test_params(max_steps_in_episode=8, gas_swing=0.06,
                              sensor_noise_scale=1.0)  # fmt: skip
    valve = np.array([0.30, 0.30, 0.70, 0.70, 0.45, 0.45, 0.60, 0.60])
    reset, keys = O.episode_keys(3, 8)
    _, state = env.reset_env(reset, p)
    planned = float(jnp.mean(O.rollout(env, p, state, keys, jnp.asarray(valve))[0]))
    assert planned == pytest.approx(_replayed(env, p, 3, valve), rel=1e-5)


def test_a_horizon_inside_the_dead_time_closes_the_valve():
    env = SPEC.make_env()
    p = SPEC.make_test_params(max_steps_in_episode=30)  # one hour
    reset, keys = O.episode_keys(5, 30)
    _, state = env.reset_env(reset, p)
    plan = O.solve(env, p, state, keys, knot_steps=10)
    assert plan.status == 0 and not plan.capped
    np.testing.assert_allclose(plan.knots, p.fuel_min, atol=1e-6)
    assert plan.lost_steps == 0 and plan.kkt < O.KKT_TOL
    assert plan.cost == pytest.approx(_replayed(env, p, 5, plan.valve), rel=1e-5)


def test_the_registered_oracle_plays_its_plan_for_the_seed():
    env = SPEC.make_env()
    p = SPEC.make_test_params(max_steps_in_episode=30)
    oracle = SPEC.make_mpc(env, p, seed=5, knot_hours=1.0 / 3.0)
    reset, _ = O.episode_keys(5, 30)
    obs, state = env.reset_env(reset, p)
    action = oracle.step(obs, state)
    assert action == pytest.approx(float(E.to_action(oracle.plans[0].valve[0], p)[0]))
    assert len(oracle.plans) == 1 and oracle.plans[0].status == 0
