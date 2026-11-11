"""The wind turbine as a continuing plant (long-strata draft 4, S1, E4).

Contract:
- a curtailment carries its rating: started in the curtailment regime, it runs to
  its return through a switch to normal operation, and lasts ``derate_hold``;
- a gust carries its size: the wind's departure is the same whichever regime's
  params the plant runs on after the onset, and it ends with the gust;
- the mean wind is no longer drawn once and held: in a stream it moves, inside its
  regime's ``v_mean_range``, and keeps the window's draw (uniform on the range) as
  its long-run law;
- at a switch the range's ends move to the new regime's as a first-order lag of the
  mean wind's correlation time ``v_mean_tau``, so the mean wind does not jump: it is
  the quantile of its score on the range where the ends stand (owner, 2026-11-11);
- the event clocks keep exact time however long the plant has run;
- a stream starts with neither the window's gust nor its curtailment.

Oracles: the scenarios' declared ranges, sizes and hold, ``P_rated``, the gust's
10.5 s, a Kolmogorov-Smirnov bound for the uniform law, a first-order lag's
closed form and the normal law's CDF (SciPy).
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ndtr

from target_gym.stream import make_stream

RATING = 4  # observation channel: the rating's power, MW
#: Steps every run here takes: a minute at 0.25 s.
N = 240


@pytest.fixture(scope="module")
def wind():
    """The stream's turbine, its scenarios' params as the stream runs them, and its
    reset, event hook and an ``N``-step run holding an action, compiled once."""
    stream = make_stream("wind_turbine")
    env = stream.env
    names = [s.name for s in stream.scenarios]

    def params(name, **kw):
        p = stream.scenario_params(stream.default_params, names.index(name))
        return p.replace(**kw)

    @jax.jit
    def run(state, p, key):
        action = env.initial_action(state, p)

        def body(s, k):
            obs, s1, *_ = env.step_env(k, s, action, p)
            # The wind's departure from its mean and turbulence: the gust.
            return s1, (obs, s1.v_wind - s.v_mean - s1.v_turb, s1.v_mean)

        return jax.lax.scan(body, state, jax.random.split(key, N))[1]

    return SimpleNamespace(
        env=env,
        params=params,
        reset=jax.jit(env.reset_env),
        start=jax.jit(env.start_event),
        traced=run,
        run=lambda state, p: jax.tree_util.tree_map(
            np.asarray, run(state, p, jax.random.PRNGKey(9))
        ),
    )


def test_a_curtailment_runs_through_a_switch_and_returns_after_its_hold(wind):
    cut = wind.params("curtailment", derate_hold=30.0)  # s
    normal = wind.params("normal")
    _, s = wind.reset(jax.random.PRNGKey(0), normal)
    s = wind.start(s.replace(time=jnp.int32(77)), jax.random.PRNGKey(1), cut)
    obs, _, _ = wind.run(s, normal)
    hold = int(cut.derate_hold / cut.delta_t)  # 120 steps
    rated = normal.P_rated / 1e6
    np.testing.assert_allclose(
        obs[: hold - 1, RATING], cut.derate_to * rated, rtol=1e-6
    )
    np.testing.assert_allclose(obs[hold:, RATING], rated, rtol=1e-6)


def test_a_gust_keeps_its_size_through_a_switch_and_ends(wind):
    # The mean wind held, so the departure is the gust's alone.
    gusty = wind.params("extreme_gust", v_mean_tau=0.0)
    other = wind.params("gusty_class_a", v_mean_tau=0.0)
    _, s = wind.reset(jax.random.PRNGKey(2), gusty)
    s = wind.start(s, jax.random.PRNGKey(3), gusty)
    _, kept, _ = wind.run(s, gusty)
    _, switched, _ = wind.run(s, other)
    np.testing.assert_allclose(switched, kept, atol=1e-4)
    gust_steps = int(10.5 / gusty.delta_t)  # 42
    assert np.abs(kept[:gust_steps]).max() > 1.0  # m/s: a gust happened
    np.testing.assert_allclose(kept[gust_steps + 1 :], 0.0, atol=1e-4)


def test_the_mean_wind_moves_inside_its_range_with_the_window_s_law(wind):
    p = wind.params("near_rated", v_mean_tau=2.0)  # s: fast, so N steps forget
    lo, hi = p.v_mean_range
    keys = jax.random.split(jax.random.PRNGKey(4), 1024)
    starts = jax.vmap(wind.reset, in_axes=(0, None))(keys, p)[1]
    # Each plant its own world: a stream's draws come from its own seed.
    worlds = jax.random.split(jax.random.PRNGKey(5), 1024)
    ends = jax.vmap(lambda s, k: wind.traced(s, p, k)[2][-1])(starts, worlds)
    begun = np.asarray(starts.v_mean)
    ends = np.asarray(ends)
    assert ((ends >= lo) & (ends <= hi)).all()
    assert np.abs(ends - begun).mean() > 0.1 * (hi - lo)  # it moved
    # Kolmogorov-Smirnov against uniform on the range: 1.63 / sqrt(n) is the 1 %
    # critical value for n = 1024.
    u = np.sort((ends - lo) / (hi - lo))
    grid = np.arange(1, u.size + 1) / u.size
    assert np.max(np.abs(grid - u)) < 1.63 / np.sqrt(u.size)


def test_the_event_clocks_keep_exact_time_however_long_the_plant_has_run(wind):
    cut = wind.params("curtailment", derate_hold=30.0, v_mean_tau=0.0)
    _, s = wind.reset(jax.random.PRNGKey(5), cut)
    ratings = []
    for t0 in (10, 200_000_010):  # 50 million s later: float32 counts it in 4 s
        started = wind.start(s.replace(time=jnp.int32(t0)), jax.random.PRNGKey(6), cut)
        ratings.append(wind.run(started, cut)[0][:, RATING])
    np.testing.assert_array_equal(ratings[0], ratings[1])


@pytest.mark.parametrize("name", ["extreme_gust", "curtailment"])
def test_a_stream_starts_without_the_window_s_events(wind, name):
    # The window would gust inside 60 to 580 s, or curtail inside 60 to 300 s.
    p = wind.params(name, v_mean_tau=0.0)
    _, s = wind.reset(jax.random.PRNGKey(7), p)
    for _ in range(10):  # the window's ten minutes, a minute at a time
        obs, departure, _ = wind.run(s, p)
        np.testing.assert_allclose(obs[:, RATING], p.P_rated / 1e6, rtol=1e-6)
        np.testing.assert_allclose(departure, 0.0, atol=1e-4)
        s = s.replace(time=s.time + N)


def test_at_a_switch_the_mean_wind_s_range_moves_over_its_correlation_time(wind):
    tau = 60.0  # s: the N steps of 0.25 s
    normal = wind.params("normal", v_mean_tau=tau)  # 13 to 25 m/s
    near = wind.params("near_rated", v_mean_tau=tau)  # 11.4 to 14 m/s
    _, s = wind.reset(jax.random.PRNGKey(4), normal)

    @jax.jit
    def scores(state):
        def body(s, k):
            _, s1, *_ = wind.env.step_env(k, s, wind.env.initial_action(s, near), near)
            return s1, (s1.v_mean, s1.v_mean_z)

        keys = jax.random.split(jax.random.PRNGKey(9), N)
        return jax.lax.scan(body, state, keys)[1]

    v, z = (np.asarray(a, np.float64) for a in scores(s))
    left = np.exp(-np.arange(1, N + 1) * near.delta_t / tau)
    lo, hi = 11.4 + (13.0 - 11.4) * left, 14.0 + (25.0 - 14.0) * left
    np.testing.assert_allclose(v, lo + (hi - lo) * ndtr(z), rtol=1e-5)
