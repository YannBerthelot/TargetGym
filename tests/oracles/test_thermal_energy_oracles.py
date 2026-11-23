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


def test_wind_receding_plan_charges_the_episodes_own_curtailment():
    """The oracle is the problem definition, so its receding model reads the
    de-rate the episode carries (``state.derate_level``, its start and its end),
    not its own params' (owner, 2026-11-23). The shipped oracle plans on the
    registered task's params, which have no de-rate; the episode here is
    de-rated to 0.9 from 1 s (step 4), with no turbulence and no tower term so
    the mean path is the wind. Torque falls at its rate limit, so power drops
    under both ratings and lost production is priced on every step: the plan
    then charges what the env charges, to float64 rounding. Read at the params'
    rating of 1.0, it charges 0.1 of rated power more per step once the de-rate
    starts."""
    from target_gym.energy.wind_turbine.oracle import wind_model

    spec = REGISTRY["wind_turbine"]
    env = spec.make_env()
    still = dict(turbulence_iref=0.0, tower_share=0.0)
    run = spec.make_test_params(derate_to=0.9, derate_window=(1.0, 1.0), **still)
    moves = [[0.0, -1.0]] * 10  # torque stays above 0, inside the move rows
    resets, steps = episode_keys(jax.random.PRNGKey(100), 1, len(moves))
    with enable_x64():
        st = float64(env.reset_env(resets[0], run)[1])
        tm = wind_model(env, float64(spec.make_test_params(**still)))
        d = {"task": tm.data(st), "widths": jnp.zeros(5)}
        x, planned = tm.state_x(st), 0.0
        for t, u in enumerate(moves):
            x, c, y = tm.step(d, x, jnp.asarray(u, jnp.float64), t)[:3]
            planned += float(c + hinge_costs(tm.hinges, jnp.asarray(y)))
        p64, s, replayed = float64(run), st, 0.0
        for t, u in enumerate(moves):
            a = tm.action(s, jnp.asarray(u, jnp.float64))
            _, s, r, _, info = env.step_env(steps[t, 0], s, a, p64)
            assert not bool(info["tripped"])
            replayed -= float(r)
    np.testing.assert_allclose(planned, replayed, rtol=1e-9)


def test_wind_plan_on_its_own_turning_points_costs_what_the_env_charges():
    """The outer loop's contract (Loew and Bottasso 2022, online rainflow): with
    the tower's turning points frozen at the plan's own, the planned fatigue is
    the env's half-cycle count, so the whole score, tower included, is what the
    env charges. The env closes a half-cycle at each turning point of the base
    moment, 0.5 (range / 1 MN m)^m at the tower's price per unit of design
    damage (env.compute_reward_terms)."""
    from target_gym.energy.wind_turbine.env import tower_base_moment
    from target_gym.energy.wind_turbine.oracle import (
        ceiling_data,
        turning_points,
        wind_model,
    )

    moves = np.random.default_rng(0).uniform(-1.0, 1.0, (40, 2))
    spec = REGISTRY["wind_turbine"]
    env = spec.make_env()
    p = spec.make_test_params()
    resets, steps = episode_keys(jax.random.PRNGKey(100), 1, len(moves))
    with enable_x64():
        p64 = float64(p)
        st = float64(env.reset_env(resets[0], p)[1])
        tm = wind_model(env, p64)
        task = float64(ceiling_data(env, p64, st, steps[:, 0], run_as=as_called))
        d = {"task": task, "widths": jnp.zeros(5)}
        x, moments = tm.state_x(st), []
        for t, u in enumerate(moves):
            x = tm.step(d, x, jnp.asarray(u), t)[0]
            moments.append(float(tower_base_moment(x[3], p64)))
    frozen = turning_points(
        float(st.moment), float(st.moment_turn), float(st.moment_dir), moments
    )
    assert frozen[0].sum() >= 2  # the moves swing the tower

    def data_of(env, p, st, keys):
        return (*ceiling_data(env, p, st, keys, run_as=as_called), frozen)

    planned, replayed, terms = plan_and_replay(
        "wind_turbine", wind_model, data_of, moves.tolist()
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


# --- Wind: the overspeed level backed off over the pitch's window (K11h, protocol
# Amendment 6, owner 2026-11-23) ------------------------------------------------
# The receding plan forecasts the turbulence decaying, so K11h backs the overspeed
# level off by 2 s_j |d omega_j / d v| over j = 1 to the window: the run-up of a
# wind deviation the plan does not see, held from the solve with the move held,
# s_j = sigma sqrt(1 - a^2j) its spread over j steps (IEC 61400-1 [S13]: sigma =
# Iref (0.75 V + 5.6); a = exp(-dt / T), T = Kaimal L_u / V). The window is the
# time the pitch rate limit needs to take back a 2-sigma gust, over the registered
# mean winds, 13 to 25 m/s, where the pitch regulates: 6 steps (1.5 s).

WIND_GRID = np.arange(13.0, 25.01, 1.0)


def _wind_reset(spec, p, v):
    """The float64 operating point the reset establishes at mean wind ``v``,
    with no turbulence: rated speed, its pitch and torque."""
    env = spec.make_env()
    still = p.replace(v_mean_range=(float(v), float(v)), turbulence_iref=0.0)
    return env, float64(still), float64(env.reset_env(jax.random.PRNGKey(0), still)[1])


def test_wind_pitch_window_is_the_rate_limits_time_to_take_back_a_2_sigma_gust():
    """dbeta = 2 sigma (dtau/dv) / |dtau/dbeta| at each operating point; the
    window is the largest dbeta / pitch_rate_max over 13 to 25 m/s, in whole
    0.25 s steps: about 1.28 s at 13 m/s, so 6 steps."""
    from target_gym.energy.wind_turbine.env import aerodynamic_torque
    from target_gym.energy.wind_turbine.oracle import HOLD_WINDOW

    spec = REGISTRY["wind_turbine"]
    p = spec.make_test_params()
    seconds = []
    with enable_x64():
        for v in WIND_GRID:
            _, p64, st = _wind_reset(spec, p, v)

            def tau(wind, pitch, st=st, p64=p64):
                return aerodynamic_torque(st.omega, wind, pitch, p64)[0]

            dv = float(jax.grad(tau, 0)(float(v), st.pitch))
            dbeta = float(jax.grad(tau, 1)(float(v), st.pitch))
            sigma = float(p.turbulence_iref) * (0.75 * v + 5.6)
            seconds.append(2.0 * sigma * dv / abs(dbeta) / float(p.pitch_rate_max))
    assert max(seconds) == pytest.approx(1.28, abs=0.01)
    assert HOLD_WINDOW == int(np.ceil(max(seconds) / float(p.delta_t))) == 6


def _hold_backoff(spec, p, grid, window):
    """K11h's term from the float64 env, by central differences on the mean wind:
    the worst over the grid and j <= window of 2 s_j |d omega_j / d v|."""
    worst, h = 0.0, 1e-3
    with enable_x64():
        for v in grid:
            env, p64, st = _wind_reset(spec, p, v)
            held = env.initial_action(st, p64)

            def omegas(dv, st=st, p64=p64, held=held, env=env):
                s, out = st.replace(v_mean=st.v_mean + dv), []
                for t in range(window):
                    s = env.step_env(jax.random.PRNGKey(t), s, held, p64)[1]
                    out.append(float(s.omega))
                return np.array(out)

            gain = np.abs(omegas(h) - omegas(-h)) / (2.0 * h)
            sigma = float(p.turbulence_iref) * (0.75 * v + 5.6)
            T = float(p.kaimal_factor) * float(p.turbulence_lambda1) / v
            a = np.exp(-float(p.delta_t) / T)
            s_j = sigma * np.sqrt(1.0 - a ** (2 * np.arange(1, window + 1)))
            worst = max(worst, float(np.max(2.0 * s_j * gain)))
    return worst


@pytest.mark.parametrize("scenario", [{}, {"turbulence_iref": 0.16}])
def test_wind_oracle_backs_the_overspeed_level_off_by_a_held_gust_over_the_window(
    scenario,
):
    """The planned overspeed bound (the limit less the solver's back-off) is the
    overspeed level less K11's smoothing gap and K11h's term, from the params the
    oracle is built on: the registered task, and gusty_class_a's Iref of 0.16,
    which backs it off further. Central differences against the exact derivative:
    1e-3 relative."""
    from target_gym.api import make_oracle
    from target_gym.energy.wind_turbine.env import omega_rated
    from target_gym.oracles.controller import widths_row

    spec = REGISTRY["wind_turbine"]
    p = spec.make_test_params(**scenario)
    oracle = make_oracle("wind_turbine", p)  # the planner's copy has no turbulence
    (limit,) = oracle.model.limits
    row = widths_row("wind_turbine")
    planned = limit.hi - row["limit_backoffs"][0]
    level = float(p.overspeed_factor) * float(omega_rated(float64(p)))
    term = _hold_backoff(spec, p, WIND_GRID, 6)
    assert level - planned == pytest.approx(row["limit_gaps"][0] + term, rel=1e-3)
    assert term > 0.09


def test_wind_ceiling_keeps_the_overspeed_level_itself():
    """K11h is the receding plan's: the ceiling knows the wind ahead, so its
    model's limit is the overspeed level (it backs off by its own
    ``ceiling_backoffs``)."""
    from target_gym.energy.wind_turbine.env import omega_rated
    from target_gym.energy.wind_turbine.oracle import wind_model

    spec = REGISTRY["wind_turbine"]
    with enable_x64():
        p64 = float64(spec.make_test_params())
        (limit,) = wind_model(spec.make_env(), p64, rest=False).limits
    level = float(p64.overspeed_factor) * float(omega_rated(p64))
    # float32 params read in float64 (rated speed's 1e-7), far under K11h's 0.1.
    assert limit.hi == pytest.approx(level, rel=1e-6)


def test_wind_receding_plan_sees_the_extreme_gust_the_env_applies():
    """The extreme operating gust is in the problem: its start is in the data and
    its size is the env's (``state.gust_size``, or in a window the size the
    scenario's turbulence gives at the mean wind, env.py ``gust``). The
    planner's copy of the params has no turbulence (``plan_params``), which gave
    the plan a gust of size 0. Gust from 1 s; each step, the plan's departure
    from its mean path is the env's gust departure, v_wind - v_mean - v_turb,
    to float64 rounding."""
    from target_gym.api import make_oracle
    from target_gym.energy.wind_turbine.oracle import wind_step

    spec = REGISTRY["wind_turbine"]
    env = spec.make_env()
    run = spec.make_test_params(gust=1.0, gust_window=(1.0, 1.0))
    oracle = make_oracle("wind_turbine", run)
    m = oracle.model
    resets, steps = episode_keys(jax.random.PRNGKey(100), 1, 40)
    with enable_x64():
        p64 = float64(run)
        st = float64(env.reset_env(resets[0], run)[1])
        task, x, s = m.data(st), m.state_x(st), st
        decay = jnp.exp(
            -p64.delta_t / (p64.kaimal_factor * p64.turbulence_lambda1 / st.v_mean)
        )
        planned, applied = [], []
        for t in range(40):
            v_plan = wind_step(
                env, oracle.params, x[:6], jnp.zeros(2), t, task, jnp.zeros(5)
            )[1]
            planned.append(float(v_plan - st.v_mean - st.v_turb * decay ** (t + 1)))
            s = env.step_env(steps[t, 0], s, env.initial_action(s, p64), p64)[1]
            applied.append(float(s.v_wind - s.v_mean - s.v_turb))
    assert max(abs(a) for a in applied) > 1.0  # the gust is on, metres a second
    np.testing.assert_allclose(planned, applied, atol=1e-9)
