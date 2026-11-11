"""The glass furnace's oracle plans the task's own reward on the episode it will meet.

Contract (charter J3, the task's oracle rule; ``glass_furnace/oracle.py``):
- the cost the oracle reports for a plan is the cost the env charges when that plan is
  replayed through the benchmark's runner on the same seed;
- that cost is the reward's: on a step that loses no glass, the fuel burnt at its heating
  value and the gas price (env_jax.compute_reward's docstring, REALITY.md section 6);
- a solve meets the fining and canal limits on every step and stops at a KKT point;
- the furnace is continuing and the episode's time limit cuts it, unseen by any agent
  (docs/rl-protocol.md), so the plan's last knot, held for ``TAIL_HOURS`` past the
  episode on the walks' mean path, loses no glass, and the plan is the cheapest over the
  episode and that tail: the tail's fuel is charged in its objective, though the cost it
  reports is the episode's;
- a plan that gets the tail's fuel free buys episode fuel with it: it costs less over the
  episode and more over the episode and the tail;
- without that tail, a horizon shorter than the fining's dead time (11.8 h, the ladder
  row's step test) and a canal that its own loop holds let no valve move reach a limit,
  so the cheapest plan is the valve closed to its minimum.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.benchmark import run_policy, task_info
from target_gym.glass_furnace import env_jax as E
from target_gym.glass_furnace import oracle as O
from target_gym.glass_furnace import physical as P
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


def _held_past_the_episode(env, p, state, keys, plan):
    """Replay ``plan`` on the env's step from ``state``, then hold its last knot on the
    walks' mean path (no innovation) for ``TAIL_HOURS``: per tail step, the fining share,
    the canal (K) and the env's charge (EUR)."""
    step = jax.jit(env.step_env)
    for t in range(len(keys)):
        _, state, _, _, _ = step(keys[t], state, E.to_action(plan.valve[t], p), p)
    tail = p.replace(walk_sd=(0.0,) * len(E.WALKS))
    hold = E.to_action(plan.knots[-1], p)
    n_tail = int(round(O.TAIL_HOURS * 3600.0 / float(p.delta_t)))
    out = []
    for k in jax.random.split(jax.random.PRNGKey(0), n_tail):
        _, state, r, _, _ = step(k, state, hold, tail)
        read = P.sensors(state.plant, p.grid, p.plant)
        out.append((float(read["fining"]), float(read["canal"]), -float(r)))
    return np.array(out).T


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


def test_without_a_tail_a_horizon_inside_the_dead_time_closes_the_valve():
    env = SPEC.make_env()
    p = SPEC.make_test_params(max_steps_in_episode=30)  # one hour
    reset, keys = O.episode_keys(5, 30)
    _, state = env.reset_env(reset, p)
    plan = O.solve(env, p, state, keys, knot_steps=10, tail_hours=0.0)
    assert plan.status == 0 and not plan.capped
    np.testing.assert_allclose(plan.knots, p.fuel_min, atol=1e-6)
    assert plan.lost_steps == 0 and plan.kkt < O.KKT_TOL
    assert plan.cost == pytest.approx(_replayed(env, p, 5, plan.valve), rel=1e-5)


def test_the_last_knot_held_past_the_episode_loses_no_glass():
    # One hour of episode, inside the fining's dead time: without the tail the plan
    # closes the valve, and held there the fining falls under its pass hours later.
    env = SPEC.make_env()
    p = SPEC.make_test_params(max_steps_in_episode=30)
    reset, keys = O.episode_keys(5, 30)
    _, state = env.reset_env(reset, p)
    plan = O.solve(env, p, state, keys, knot_steps=10)
    assert plan.lost_steps == 0
    # The reported cost is the episode's replayed cost, the tail left out.
    assert plan.cost == pytest.approx(_replayed(env, p, 5, plan.valve), rel=1e-5)
    fining, canal, _ = _held_past_the_episode(env, p, state, keys, plan)
    assert fining.min() >= p.fining_pass
    assert canal.min() >= p.canal_target


def test_the_plan_is_cheapest_over_the_episode_and_its_tail():
    # Six hours on 2 h knots: the design state's fining margin lets a plan coast through
    # the episode and pay the fining back in the tail. Charged for the tail's fuel, the
    # plan beats one that gets that fuel free over the episode and the tail together;
    # the free one beats it over the episode alone. Each is optimal for its own bill,
    # so these hold at any two optima, and they differ only if the free tail is used.
    # Margin: 1e-3 of the bill, far above float32's 1e-5 over these steps (ours).
    env = SPEC.make_env()
    p = SPEC.make_test_params(max_steps_in_episode=180)
    reset, keys = O.episode_keys(5, 180)
    _, state = env.reset_env(reset, p)
    bills = {}
    for charged in (True, False):
        plan = O.solve(env, p, state, keys, knot_steps=60, charge_tail=charged)
        assert plan.lost_steps == 0
        episode = 180 * _replayed(env, p, 5, plan.valve)
        fining, canal, tail = _held_past_the_episode(env, p, state, keys, plan)
        assert fining.min() >= p.fining_pass and canal.min() >= p.canal_target
        bills[charged] = (episode, episode + tail.sum())
        if charged:  # the tail's planned fuel bill is the one the env charges
            assert plan.tail_running == pytest.approx(tail.mean(), rel=1e-5)
    assert bills[True][1] < (1.0 - 1e-3) * bills[False][1]
    assert bills[False][0] < (1.0 - 1e-3) * bills[True][0]


def test_a_last_iterate_that_loses_glass_is_raised_at_least_cost():
    # Six hours on 2 h knots. Lowered by 0.02 kg/s, the plan's middle knot loses glass
    # hours later, when its fuel reaches the fining. Left no iteration, the solve's plan
    # is that start repaired. Raising where the glass is lost costs about the plan it
    # came from; raising every knot alike also pays for the last knot's fuel through the
    # 24 h tail. Margin: 1e-3 of the bill, above a first-order repair's overshoot and
    # float32's 1e-5 (ours).
    env = SPEC.make_env()
    p = SPEC.make_test_params(max_steps_in_episode=180)
    reset, keys = O.episode_keys(5, 180)
    _, state = env.reset_env(reset, p)
    best = O.solve(env, p, state, keys, knot_steps=60)
    start = best.knots - np.array([0.0, 0.02, 0.0])
    lowered = O.Plan._make([np.repeat(start, 60), start] + [None] * 12)
    fining, canal, _ = _held_past_the_episode(env, p, state, keys, lowered)
    assert fining.min() < p.fining_pass or canal.min() < p.canal_target
    plan = O.solve(env, p, state, keys, knot_steps=60, start=start, max_iter=0)
    bills = []
    for q in (best, plan):
        assert q.lost_steps == 0
        fining, canal, tail = _held_past_the_episode(env, p, state, keys, q)
        assert fining.min() >= p.fining_pass and canal.min() >= p.canal_target
        bills.append(180 * _replayed(env, p, 5, q.valve) + tail.sum())
    assert bills[1] < (1.0 + 1e-3) * bills[0]


def test_the_registered_oracle_plays_its_plan_for_the_seed():
    env = SPEC.make_env()
    p = SPEC.make_test_params(max_steps_in_episode=30)
    oracle = SPEC.make_mpc(env, p, seed=5, knot_hours=1.0 / 3.0)
    reset, _ = O.episode_keys(5, 30)
    obs, state = env.reset_env(reset, p)
    action = oracle.step(obs, state)
    assert action == pytest.approx(float(E.to_action(oracle.plans[0].valve[0], p)[0]))
    assert len(oracle.plans) == 1 and oracle.plans[0].lost_steps == 0
