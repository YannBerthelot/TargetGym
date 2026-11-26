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

Without a seed (the owner's ruling of 2026-11-16; docs/proposals/glass-furnace-room.md,
"The fair line"):
- the oracle is shown no future draw: it plans on the walks' mean path from the true
  state, a new plan each knot, and when no valve meets the canal's limit or the
  override's cap, the plan it had goes on;
- it prices the fining mark instead of limiting it: each planned step is charged, over
  its fuel and plates, the bill of a lost step times the chance that the fining comes
  out under its mark, the forecast's error taken as normal with the spread the model's
  own walks give at that lead (no episode's draws), read at the episode's first plan
  over the fining's dead time and a knot, past which a later plan can still answer;
- so the wider the spread, the further over the mark a plan keeps the fining, and the
  cost a plan reports stays the env's, without the charge.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.benchmark import episode_keys, run_policy, task_info
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


@pytest.mark.parametrize("seed, n", [(0, 1), (5, 30)])
def test_the_oracle_replays_the_keys_the_benchmark_runs_its_seed_on(seed, n):
    # Contract: the oracle plans on the disturbances the scored episode meets,
    # which come from benchmark.episode_keys for one episode on PRNGKey(seed).
    resets, steps = episode_keys(jax.random.PRNGKey(seed), 1, n)
    reset, keys = O.episode_keys(seed, n)
    np.testing.assert_array_equal(reset, resets[0])
    np.testing.assert_array_equal(keys, steps[:, 0])


def test_the_chance_of_a_lost_step_is_the_normal_tail_under_the_mark():
    # Standard normal table: half on the mark, 0.15866 one spread over it, 0.02275
    # two, 3.167e-5 four, and all but 0.02275 two spreads under it. Tolerance: the
    # table's last digit.
    table = (
        (0.0, 0.5),
        (1.0, 0.15866),
        (2.0, 0.02275),
        (4.0, 3.167e-5),
        (-2.0, 0.97725),
    )
    for over, chance in table:
        got = float(O.chance_under(0.5 + over * 0.007, 0.5, 0.007))
        assert got == pytest.approx(chance, rel=2e-3)


def test_the_forecast_spread_is_the_walks_own():
    # Four hours with the valve held, eight sampled paths. Without walks every path is
    # the forecast, to float32's rounding of a share. With them the spread shows well
    # over that rounding. The walks are first order and the fining's answer to them
    # close to linear at their size, so on the same draws half the walks give about
    # half the spread (within a fifth, ours: the fining band and the blocks' release
    # are not linear).
    env = SPEC.make_env()
    p = SPEC.make_test_params()
    reset, _ = O.episode_keys(5, 1)
    _, state = env.reset_env(reset, p)
    half = p.replace(walk_sd=tuple(0.5 * s for s in p.walk_sd))
    none = p.replace(walk_sd=(0.0,) * len(E.WALKS))
    whole, halved, still = (
        np.asarray(O.forecast_spread(env, q, state, 120, paths=8)) for q in (p, half, none)
    )  # fmt: skip
    assert whole.shape == (120,)
    np.testing.assert_allclose(still, 0.0, atol=1e-6)
    assert whole[-1] > 1e-5
    assert halved[-1] == pytest.approx(0.5 * whole[-1], rel=0.2)


def test_a_priced_fining_mark_is_kept_further_the_wider_the_spread():
    # Six hours on 2 h knots, as above: the plan coasts on the design state's fining
    # and pays it back in the tail. With the mark limited, the plan keeps it and no
    # more. With the mark priced, a step two spreads over the mark is still charged
    # 416.88 EUR * 0.02275 = 9.5 EUR, a fifth of its fuel bill, so the plan stays
    # further than that, and further the wider the spread. A priced plan meets the
    # limit's own rows, so on the env's bill it is no cheaper than the limit's plan,
    # which is the cheapest there. Margin on the bills: 1e-3, as above (ours).
    env = SPEC.make_env()
    p = SPEC.make_test_params(max_steps_in_episode=180)
    reset, keys = O.episode_keys(5, 180)
    _, state = env.reset_env(reset, p)
    over, bills = {}, {}
    for spread in (None, 0.002, 0.007):
        priced = None if spread is None else np.full(1, spread)
        plan = O.solve(env, p, state, keys, knot_steps=60, fining_spread=priced)
        episode = _replayed(env, p, 5, plan.valve)
        assert plan.cost == pytest.approx(
            episode, rel=1e-5
        )  # the env's, no charge in it
        played = O.rollout(env, p, state, keys, jnp.asarray(plan.valve))[2]
        held, canal, tail = _held_past_the_episode(env, p, state, keys, plan)
        assert canal.min() >= p.canal_target
        over[spread] = min(float(played.min()), held.min()) - p.fining_pass
        bills[spread] = 180 * episode + tail.sum()
    assert 0.0 <= over[None] < over[0.002] < over[0.007]
    assert over[0.002] > 2 * 0.002 and over[0.007] > 2 * 0.007
    assert bills[0.002] > (1.0 - 1e-3) * bills[None]
    assert bills[0.007] > (1.0 - 1e-3) * bills[None]


def _first_plans(oracle, env, p, seed, n):
    """``oracle`` played ``n`` steps on ``seed``: per step, the plans made so far, the
    action, and whether the step kept both marks."""
    reset, keys = O.episode_keys(seed, n)
    obs, state = env.reset_env(reset, p)
    step = jax.jit(env.step_env)
    made, played, kept = [], [], []
    for t in range(n):
        action = oracle.step(obs, state)
        made.append(len(oracle.plans))
        played.append(action)
        obs, state, _, _, _ = step(keys[t], state, jnp.asarray([action]), p)
        read = P.sensors(state.plant, p.grid, p.plant)
        kept.append(bool(read["fining"] >= p.fining_pass and read["canal"] >= p.canal_target))  # fmt: skip
    return made, played, kept


def test_without_a_seed_the_oracle_plans_again_each_knot():
    # Forty minutes on 20 min knots: a plan at steps 0 and 10, each played for its
    # first knot, and the design state's forty minutes keep both marks.
    env = SPEC.make_env()
    p = SPEC.make_test_params(max_steps_in_episode=20)
    oracle = SPEC.make_mpc(env, p, knot_hours=1.0 / 3.0, spread=np.full(1, 0.002))
    made, played, kept = _first_plans(oracle, env, p, 5, 20)
    assert made == [1] * 10 + [2] * 10
    firsts = [float(E.to_action(q.knots[0], p)[0]) for q in oracle.plans]
    np.testing.assert_allclose(played, np.repeat(firsts, 10), atol=1e-6)
    assert all(kept)


def test_when_no_valve_can_meet_the_canal_the_plan_it_had_goes_on():
    # A bath 20 K colder than the plan left it: the canal is under its target at once
    # and no fuel reaches it within the forty minutes, so no plan meets the limits.
    env = SPEC.make_env()
    p = SPEC.make_test_params(max_steps_in_episode=20)
    oracle = SPEC.make_mpc(env, p, knot_hours=1.0 / 3.0, spread=np.full(1, 0.002))
    reset, _ = O.episode_keys(5, 20)
    obs, state = env.reset_env(reset, p)
    oracle.step(obs, state)
    had = oracle.plans[0]
    cold = state.replace(
        time=state.time + 10, plant=state.plant.replace(T=state.plant.T - 20.0)
    )
    action = oracle.step(obs, cold)
    assert action == pytest.approx(float(E.to_action(had.knots[1], p)[0]))
    assert oracle.kept == 1


def test_the_spread_is_read_from_the_model_at_the_first_plan():
    # No spread given: the oracle samples the walks from the state it is first shown,
    # over the fining's dead time and a knot (11.8 h and 20 min: 364 steps). Its plan
    # is the plan of an oracle handed that spread.
    env = SPEC.make_env()
    p = SPEC.make_test_params(max_steps_in_episode=10)
    reset, _ = O.episode_keys(5, 10)
    obs, state = env.reset_env(reset, p)
    spread = O.forecast_spread(env, p, state, 364, paths=4)
    oracle = SPEC.make_mpc(env, p, knot_hours=1.0 / 3.0, paths=4)
    handed = SPEC.make_mpc(env, p, knot_hours=1.0 / 3.0, spread=spread)
    assert oracle.step(obs, state) == pytest.approx(handed.step(obs, state))
    np.testing.assert_allclose(oracle.plans[0].knots, handed.plans[0].knots, atol=1e-6)
