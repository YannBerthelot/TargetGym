"""The thermal and energy oracles plan the task's own score.

Contract (task request, charter J3: the ceiling is the best cost an oracle finds
with the true model, the true state and the future): for any plan, the cost the
oracle's planner charges it equals the cost the env charges when the plan is
replayed on it from the same state with the same future. Checked on tiny cases
built to make every priced term non-zero, at smoothing width 0, where the
planner's model is exact:

- boiler_drum: the imbalance at the price the reward states, GBP 80/MWh of
  steam's generation at 3.5 kg/kWh: 2 s / 3.5 / 1000 * 80 = GBP 0.0457 per
  kg/s off the plan past the meter's 0.047 kg/s;
- hvac: kelvin-hours outside the band at the work-loss price, 0.25 h *
  0.02 * 0.10 * 150 * 31.8 = EUR 2.385 per K per step, and the electricity;
- wind_turbine: lost production and pitch wear (the tower's fatigue is not
  planned, PHYSICS.md; the comparison leaves it out on both sides);
- battery: the plan is the baselines, and the oracle's cost of them is what
  the scoring run (``benchmark.run_policy``) charges them; its search finds
  no plan a fine grid beats.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.benchmark import episode_keys, episodes_of, run_policy, task_info
from target_gym.oracles.controller import enable_x64, float64, hinge_costs
from target_gym.registry import REGISTRY

pytestmark = pytest.mark.medium


def plan_and_replay(task, model, data_of, moves, start=lambda s: s, **over):
    """(planned, replayed, env terms) of ``moves`` from the seed-100 reset."""
    spec = REGISTRY[task]
    env = spec.make_env()
    p = spec.make_test_params(**over)
    resets, steps = episode_keys(jax.random.PRNGKey(100), 1, len(moves))
    keys = steps[:, 0]
    with enable_x64():
        p64 = float64(p)
        st = start(float64(env.reset_env(resets[0], p)[1]))
        tm = model(env, p64)
        d = {"task": float64(data_of(env, p64, st, keys)), "widths": jnp.zeros(5)}
        x, planned = tm.state_x(st), 0.0
        for t, u in enumerate(moves):
            x, c, y = tm.step(d, x, jnp.asarray(u, jnp.float64), t)[:3]
            planned += float(c + hinge_costs(tm.hinges, jnp.asarray(y)))
        s, replayed, terms = st, 0.0, {}
        for t, u in enumerate(moves):
            u = jnp.asarray(u, jnp.float64)
            a = u if tm.action is None else tm.action(s, u)
            _, s, r, _, info = env.step_env(keys[t], s, a, p64)
            assert not bool(info["tripped"])
            replayed -= float(r)
            for k, v in env.reward_terms(s, p64).items():
                terms[k] = terms.get(k, 0.0) + float(v)
    return planned, replayed, terms


def test_boiler_plan_costs_what_the_env_charges():
    from target_gym.boiler_drum.oracle import boiler_model, ceiling_data

    # Pressure 3 bar off nominal: the valve passes 3/p_nom more than the plan.
    planned, replayed, terms = plan_and_replay(
        "boiler_drum",
        boiler_model,
        ceiling_data,
        [[1.0, -1.0], [0.5, 0.0], [-1.0, 1.0], [0.0, 0.3]],
        start=lambda s: s.replace(pressure=s.pressure + 3.0),
    )
    assert terms["imbalance"] > 0.0
    np.testing.assert_allclose(planned, replayed, rtol=1e-9)


def test_boiler_hinge_is_priced_per_kg_of_steam_at_the_imbalance_price():
    from target_gym.boiler_drum.oracle import boiler_model

    spec = REGISTRY["boiler_drum"]
    (hinge,) = boiler_model(spec.make_env(), spec.make_test_params()).hinges
    assert hinge.absolute and hinge.tol == pytest.approx(0.047)
    assert hinge.weight == pytest.approx(2.0 / 3.5 / 1000.0 * 80.0)


def test_hvac_plan_costs_what_the_env_charges():
    from target_gym.hvac.oracle import ceiling_data, hvac_model

    # A cold building at 09:00, occupied: the room under the band. Full
    # supply, then water below the core (no heat), then in between.
    planned, replayed, terms = plan_and_replay(
        "hvac",
        hvac_model,
        ceiling_data,
        [[1.0], [-1.0], [-0.2], [0.4]],
        start=lambda s: s.replace(time=36, T_mass=s.T_mass - 4.0, T_core=19.5),
    )
    assert terms["comfort"] > 0.0 and terms["electricity"] > 0.0
    np.testing.assert_allclose(planned, replayed, rtol=1e-9)


def test_hvac_comfort_hinges_are_priced_at_the_work_loss():
    from target_gym.hvac.oracle import hvac_model

    spec = REGISTRY["hvac"]
    hinges = hvac_model(spec.make_env(), spec.make_test_params()).hinges
    assert [h.weight for h in hinges] == pytest.approx(
        [0.25 * 0.02 * 0.10 * 150.0 * 31.8] * 3
    )


def test_wind_plan_costs_what_the_env_charges_bar_the_tower():
    from target_gym.energy.wind_turbine.oracle import ceiling_data, wind_model

    planned, replayed, terms = plan_and_replay(
        "wind_turbine",
        wind_model,
        ceiling_data,
        [[1.0, -1.0], [0.5, -1.0], [-1.0, 0.2], [0.0, 1.0], [0.3, 0.0]],
    )
    assert terms["lost"] > 0.0 and terms["pitch"] > 0.0
    np.testing.assert_allclose(planned, replayed - terms["tower"], rtol=1e-9)


class Schedule:
    """Plays a fixed raw action per step."""

    name = "schedule"

    def __init__(self, actions):
        self.actions = np.asarray(actions, np.float32)

    def reset(self, key, obs, task):
        return 0

    def act(self, carry, obs, t):
        return self.actions[t][None], carry


def test_battery_ceiling_is_the_scored_cost_of_its_plan_and_beats_a_fine_grid():
    from target_gym.energy.battery import oracle as O

    # A 40 s block of 10 s periods, gate closure 10 s ahead: two periods to
    # declare, both in a price spike.
    spec = REGISTRY["battery"]
    env = spec.make_env()
    p = spec.make_test_params(
        max_steps_in_episode=40, period_seconds=10.0, gate_closure_seconds=10.0,
        price_event_range=(300.0, 300.0), price_event_first=(2, 2),
    )  # fmt: skip
    resets, steps = episode_keys(jax.random.PRNGKey(100), 1, 40)
    st = env.reset_env(resets[0], p)[1]
    res = O.ceiling_search(env, p, st, steps[:, 0])
    assert res["periods"] == [2, 3] and res["converged"]
    B, periods = jnp.asarray(res["baselines_raw"]), jnp.asarray(res["periods"])
    actions = [float(O.raw_action(B, periods, t, p)[0]) for t in range(40)]
    rollout = run_policy(
        Schedule(actions), env, p, task_info("battery", env, p, 1), seeds=[100]
    )
    (episode,) = episodes_of(rollout)
    np.testing.assert_allclose(episode.cost.sum(), res["cost"], rtol=1e-5)
    axis = np.linspace(-1.0, 1.0, 41)
    grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    f = O.evaluator(env, p, st, steps[:, 0], periods, batch=len(grid))
    assert res["cost"] <= f(grid)[0].min() + 1e-6
