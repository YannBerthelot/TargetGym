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

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.benchmark import episode_keys, episodes_of, run_policy, task_info
from target_gym.oracles.controller import enable_x64, float64, hinge_costs
from target_gym.registry import REGISTRY

pytestmark = pytest.mark.medium


def plan_and_replay(task, model, data_of, moves, start=lambda s: s, **over):
    """(planned, replayed, env terms) of ``moves`` from the seed-100 reset.
    Both sides run under x64, so a ceiling's disturbance path is read under
    x64 too (``as_called``): the replay's draws are x64's, and the check is
    the model's exactness on its data. That the path is the float32 scored
    run's is ``test_ceiling_weather_is_the_float32_episodes_under_x64``."""
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


def as_called(fn, *args):
    """Runs ``fn`` in the caller's mode: the exact path for an x64 replay."""
    return fn(*args)


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
        partial(ceiling_data, run_as=as_called),
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
        partial(ceiling_data, run_as=as_called),
        [[1.0, -1.0], [0.5, -1.0], [-1.0, 0.2], [0.0, 1.0], [0.3, 0.0]],
    )
    assert terms["lost"] > 0.0 and terms["pitch"] > 0.0
    np.testing.assert_allclose(planned, replayed - terms["tower"], rtol=1e-9)


def test_wind_plan_pays_for_power_the_converter_will_not_deliver():
    """The converter holds power at the set point (owner, 2026-11-23), so the
    env charges nothing above it: it is never delivered. The model keeps its
    torque free and pays for power above the set point instead, a surrogate
    priced at ten times the energy price (ours: an exact penalty needs only to
    exceed what shedding power can buy, and at one times it the review's
    ceiling stayed within 0.04 % of the set point), so that a plan stays where
    model and env agree. Three steps of full torque rise from the reset, which
    sits on the set point: the env's power stays on it, the model's passes it,
    and the plan pays 10 x 60 $/MWh for every MWh of the excess."""
    from target_gym.energy.wind_turbine.env import electrical_power
    from target_gym.energy.wind_turbine.oracle import ceiling_data, wind_model

    spec = REGISTRY["wind_turbine"]
    env, p = spec.make_env(), spec.make_test_params()
    moves = [[0.0, 1.0]] * 3
    resets, steps = episode_keys(jax.random.PRNGKey(100), 1, len(moves))
    keys = steps[:, 0]
    with enable_x64():
        p64 = float64(p)
        st = float64(env.reset_env(resets[0], p)[1])
        tm = wind_model(env, p64)
        d = {"task": float64(ceiling_data(env, p64, st, keys, run_as=as_called)),
             "widths": jnp.zeros(5)}  # fmt: skip
        x, s_env, excess, paid = tm.state_x(st), st, 0.0, 0.0
        for t, u in enumerate(moves):
            u = jnp.asarray(u, jnp.float64)
            x, c, y, power = tm.step(d, x, u, t)[:4]
            with_it = float(hinge_costs(tm.hinges, jnp.asarray(y)))
            without = float(hinge_costs(tm.hinges, jnp.asarray(y), surrogates=False))
            paid += with_it - without
            _, s_env, _, _, _ = env.step_env(keys[t], s_env, tm.action(s_env, u), p64)
            set_point = float(s_env.target_power)
            excess += max(float(power[0]) - set_point, 0.0)
            delivered = float(electrical_power(s_env.omega, s_env.torque, p64))
            assert delivered <= set_point * (1.0 + 1e-9)
    assert excess > 1.0e5  # W: the model's torque did pass the limit
    mwh = excess * float(p.delta_t) / 3600.0 / 1.0e6
    assert paid == pytest.approx(10.0 * 60.0 * mwh, rel=1e-9)


def _wind_plant_run(env, p, state, keys, moves):
    """The float32 plant's own run of ``moves`` as the recorder replays a plan:
    each move from the pitch and torque the plant is at (``wind_model``'s
    action). (Torque, power, set point) after every step."""
    from target_gym.energy.wind_turbine.oracle import wind_model

    act = wind_model(env, p, rest=False).action
    s, out = state, []
    for t, u in enumerate(np.asarray(moves, np.float32)):
        _, s, _, _, _ = env.step_env(keys[t], s, act(s, jnp.asarray(u)), p)
        out.append([float(s.torque), float(s.power), float(s.target_power)])
    return np.array(out).T


def test_wind_trimmed_plan_runs_on_the_plant_without_the_converter_clipping():
    """The env runs a ceiling's plan open loop, each move from the torque the
    plant is at. Where the plant ends a step over the set point the converter
    clips its torque, the moves that follow start from the clipped torque, and
    the rotor is braked less than planned from then on. The solver's model
    reads Cp and Ct smoothed (K2), so a plan resting on the set point there
    ends 0.8 to 11 kW of 5 MW over it on the plant's exact tables (the plans
    measured, 2026-11-24): those 3.8 kW and more over tripped the plant (normal
    seed 102 on every re-solve, near_rated seed 101).

    The owner chose (2026-11-24): the plan's torque is trimmed on the exact
    tables to the set point less 0.001 % of it (``trim_plan``). Off the cap the
    float32 plant follows a float64 plan (measured on that seed over 2400
    steps: no step clipped, speed 1.2e-5 of rated apart).

    Forty steps from the reset, which sits on the set point, inside S1's torque
    limit as a solver's plan is (47.4 kN m, 4.3 kN m over the reset's). The plan
    raises the torque 375 N m a step for ten steps (0.1 of the 3.75 kN m reach,
    S1's 15 kN m/s over 0.25 s) to 3.75 kN m over, holds it, dips 4.5 kN m in
    four steps, comes back and holds, with the pitch moving a little both ways.
    On over half the steps it asks for more than the converter delivers (the
    rotor slows under the load, so the cap's torque rises toward the plan's).

    - Untrimmed, the plant clips: its torque ends over 1 kN m under the plan's.
    - Trimmed, the pitch moves are the plan's, and the plant ends every step on
      the torque the trimmed plan says, within 0.5 N m (float32 against
      float64: 43 kN m at 6e-8 a step, forty steps): the converter clipped
      nothing.
    - Where the plan asked for more than the cap, the plant delivers within
      0.002 % of the set point: the trim takes off no more than it has to."""
    from target_gym.energy.wind_turbine.oracle import (
        ceiling_data,
        trim_plan,
        wind_model,
    )

    spec = REGISTRY["wind_turbine"]
    env, p = spec.make_env(), spec.make_test_params()
    moves = np.array(
        [[0.05, 0.1]] * 10
        + [[-0.05, 0.0]] * 10
        + [[0.0, -0.3]] * 4
        + [[0.0, 0.3]] * 4
        + [[0.02, 0.0]] * 12
    )
    resets, steps = episode_keys(jax.random.PRNGKey(100), 1, len(moves))
    keys = steps[:, 0]
    st = env.reset_env(resets[0], p)[1]
    data = ceiling_data(env, p, st, keys)
    with enable_x64():
        p64 = float64(p)
        x0 = wind_model(env, p64, rest=False).state_x(float64(st))
        trimmed = np.asarray(trim_plan(env, p64, float64(data), x0, jnp.asarray(moves)))
    reach = 15000.0 * 0.25  # N m of torque a step at S1's rate limit
    asked = float(st.torque) + np.cumsum(moves[:, 1]) * reach
    planned = float(st.torque) + np.cumsum(trimmed[:, 1]) * reach

    clipped, _, _ = _wind_plant_run(env, p, st, keys, moves)
    assert np.max(asked - clipped) > 1000.0

    torque, power, set_point = _wind_plant_run(env, p, st, keys, trimmed)
    np.testing.assert_allclose(trimmed[:, 0], moves[:, 0], atol=1e-12)
    np.testing.assert_allclose(torque, planned, atol=0.5)
    more = asked - planned > 1.0
    assert more.sum() > 20
    assert np.all(power[more] >= set_point[more] * (1.0 - 2.0e-5))


def test_wind_trim_leaves_a_plan_under_the_cap_as_it_is():
    """A plan that never asks for more than the set point is not the trim's
    business: from the reset with the torque 20 % under the set point's, forty
    small moves stay under it, and come back unchanged."""
    from target_gym.energy.wind_turbine.oracle import (
        ceiling_data,
        trim_plan,
        wind_model,
    )

    spec = REGISTRY["wind_turbine"]
    env, p = spec.make_env(), spec.make_test_params()
    moves = np.random.default_rng(0).uniform(-1.0, 1.0, (40, 2)) * [1.0, 0.2]
    resets, steps = episode_keys(jax.random.PRNGKey(100), 1, len(moves))
    keys = steps[:, 0]
    st = env.reset_env(resets[0], p)[1]
    st = st.replace(torque=0.8 * st.torque)
    data = ceiling_data(env, p, st, keys)
    _, power, set_point = _wind_plant_run(env, p, st, keys, moves)
    assert np.all(power < 0.95 * set_point)
    with enable_x64():
        p64 = float64(p)
        x0 = wind_model(env, p64, rest=False).state_x(float64(st))
        trimmed = np.asarray(trim_plan(env, p64, float64(data), x0, jnp.asarray(moves)))
    np.testing.assert_allclose(trimmed, moves, atol=1e-9)


def test_wind_plan_on_its_own_rainflow_rows_costs_what_the_env_charges():
    """The outer loop's contract (Loew and Bottasso 2022, online rainflow): with
    the tower's rows frozen at the plan's own turning points, the planned
    fatigue is the env's rainflow count (ASTM E1049, owner 2026-11-23), so the
    whole score, tower included, is what the env charges. Rainflow pairs up the
    record's turning points; each pair's range is charged at its two ends, about
    their midpoint where both are in the plan, which is exact at the plan the
    rows were counted on (oracle.rainflow_rows).

    Since the converter holds power at the set point (owner, 2026-11-23), the
    model and the env agree only under that limit, so the moves start with the
    torque 20 % under it and move it a fifth as fast as the pitch; the run
    checks that it stays under."""
    from target_gym.energy.wind_turbine.env import tower_base_moment
    from target_gym.energy.wind_turbine.oracle import (
        ceiling_data,
        rainflow_rows,
        wind_model,
    )

    moves = np.random.default_rng(0).uniform(-1.0, 1.0, (40, 2)) * [1.0, 0.2]

    def start(s):
        return s.replace(torque=0.8 * s.torque)

    spec = REGISTRY["wind_turbine"]
    env = spec.make_env()
    p = spec.make_test_params()
    resets, steps = episode_keys(jax.random.PRNGKey(100), 1, len(moves))
    with enable_x64():
        p64 = float64(p)
        st = start(float64(env.reset_env(resets[0], p)[1]))
        tm = wind_model(env, p64)
        task = float64(ceiling_data(env, p64, st, steps[:, 0], run_as=as_called))
        d = {"task": task, "widths": jnp.zeros(5)}
        x, moments, above = tm.state_x(st), [], []
        for t, u in enumerate(moves):
            x, _, y = tm.step(d, x, jnp.asarray(u), t)[:3]
            moments.append(float(tower_base_moment(x[3], p64)))
            above.append(float(y[4]))  # W over the set point
    assert max(above) < 0.0  # under the converter's limit throughout
    frozen = rainflow_rows(
        np.asarray(st.moment_stack, float),
        int(st.moment_depth),
        float(st.moment),
        np.asarray(moments),
        float(p.fatigue_exponent),
    )
    assert (frozen[1] > 0).sum() >= 4  # the moves swing the tower

    def data_of(env, p, st, keys):
        return (*ceiling_data(env, p, st, keys, run_as=as_called), frozen)

    planned, replayed, terms = plan_and_replay(
        "wind_turbine", wind_model, data_of, moves.tolist(), start=start
    )
    assert terms["tower"] > 0.0 and terms["lost"] > 0.0
    np.testing.assert_allclose(planned, replayed, rtol=1e-9)


def test_wind_ceiling_plans_on_the_episodes_wind_even_where_a_held_run_trips():
    """From 1 % under the overspeed level the reset's pitch and torque held
    overspeed the rotor, and a trip redraws the turbulence. The wind does not
    depend on the rotor, so the ceiling's must be the one a feathering run from
    the reset's own speed, which does not trip, sees."""
    from target_gym.energy.wind_turbine.env import omega_rated
    from target_gym.energy.wind_turbine.oracle import ceiling_data

    spec = REGISTRY["wind_turbine"]
    env, p = spec.make_env(), spec.make_test_params()
    resets, steps = episode_keys(jax.random.PRNGKey(100), 1, 40)
    st = env.reset_env(resets[0], p)[1]
    # At fine pitch in this seed's 19 m/s the rotor runs away.
    fast = st.replace(
        omega=jnp.float32(1.19 * float(omega_rated(p))), pitch=jnp.float32(0.0)
    )
    path = np.asarray(ceiling_data(env, p, fast, steps[:, 0])[-1])

    def run(action):
        def f(s, k):
            _, s1, _, _, info = env.step_env(k, s, action(s), p)
            return s1, (s1.v_wind, info["tripped"])

        return f

    held = env.initial_action(fast, p)
    assert bool(jnp.any(jax.lax.scan(run(lambda s: held), fast, steps[:, 0])[1][1]))
    feather = run(lambda s: jnp.array([1.0, 1.0]))
    wind, tripped = jax.lax.scan(feather, st, steps[:, 0])[1]
    assert not bool(jnp.any(tripped))
    np.testing.assert_array_equal(path[:, 0], np.asarray(wind))


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


def test_boiler_receding_oracle_holds_the_steam_under_water_at_or_above_zero():
    """The env holds the steam under the water surface at 0 (``compute_next
    _state``), so a plan that drives it below 0 mispredicts the level; the
    ceiling holds it as a limit, and the registry's receding slot must too."""
    from target_gym.boiler_drum.oracle import boiler_model, make_boiler_drum_mpc
    from target_gym.experts.mpc import plan_params

    spec = REGISTRY["boiler_drum"]
    env = spec.make_env()
    p = plan_params(spec, spec.make_test_params())
    receding = make_boiler_drum_mpc(env, p).model
    with enable_x64():
        ceiling = boiler_model(env, float64(p), ceiling=True)
    floors = [(q.lo, q.hi, q.price) for q in receding.limits]
    assert floors == [(q.lo, q.hi, q.price) for q in ceiling.limits]


@pytest.mark.parametrize("task", ["wind_turbine", "hvac"])
def test_ceiling_weather_is_the_float32_episodes_under_x64(task):
    """The ceiling plans on the episode's weather, which the scored run draws
    in float32. ``jax.random.normal`` draws other values under x64, so the
    path must not depend on whether the caller has x64 on."""
    import importlib

    module = (
        "target_gym.energy.wind_turbine.oracle"
        if task == "wind_turbine"
        else ("target_gym.hvac.oracle")
    )
    ceiling_data = importlib.import_module(module).ceiling_data
    spec = REGISTRY[task]
    env, p = spec.make_env(), spec.make_test_params()
    resets, steps = episode_keys(jax.random.PRNGKey(100), 1, 30)
    st = env.reset_env(resets[0], p)[1]
    scored = np.asarray(ceiling_data(env, p, st, steps[:, 0])[-1])
    with enable_x64():
        planned = np.asarray(
            ceiling_data(env, float64(p), float64(st), steps[:, 0])[-1]
        )
    assert np.any(scored != scored[0])  # the weather moves on this seed
    np.testing.assert_allclose(planned, scored, rtol=1e-5, atol=1e-6)


def test_boiler_oracle_backs_the_trip_levels_off_by_the_level_reading():
    """K11 backs a limit off by the planner's smoothing error, by the noise it
    cannot foresee, and by the plant's precision. The boiler's planner smooths
    nothing (no width sites) and its plant draws nothing after the reset, but
    a plan riding a trip level exactly is tripped by the float32 plant, which
    ends about 1e-6 m off the float64 plan; the inclusive trip timer then runs.
    Like the ceiling, the receding slot plans one sigma of the level's reading
    inside both trip levels, and nothing off the floor of the steam under the
    water, which the env clips."""
    from target_gym import records

    env = REGISTRY["boiler_drum"].make_env()
    (sigma,) = [r.sigma for r in env.readings if r.field == "level"]
    row = records.load_row("oracle_widths", "boiler_drum")
    assert row["sites"] == {}
    assert row["limit_backoffs"] == pytest.approx([sigma, 0.0])
