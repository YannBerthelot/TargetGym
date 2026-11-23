"""The wind the rotor sees: the hub's turbulence through the rotor's spatial filter.

Contract, from the sources (REALITY.md), not from the code:

* The hub's wind is a point: the mean, IEC's normal turbulence about it
  (sigma1 = Iref (0.75 V + 5.6), the Kaimal integral time scale) and, in its
  scenario, one extreme operating gust [S13, S19].
* A rotor averages the turbulence over its disc. The filter is Petru's Eq. (5.7)
  [S24], after Wilkie, Leithead and Anderson (1990), on the turbulence about
  the mean::

      H(s) = (sqrt2 + b s) / ((sqrt2 + b sqrt(a) s) (1 + b s / sqrt(a)))
      b = gamma R / V,   gamma = 1.3,   a = 0.55

  with R the rotor's radius and V the mean wind at the hub. Its gain is 1 at
  rest, so a steady wind reaches the rotor whole.
* The extreme operating gust is one coherent rise and fall over the whole rotor
  (IEC, via [S13]), so it is not averaged.
* An episode starts in established operation (rule 17): the filter is drawn
  from the law it has after a long run, given the hub's turbulence.
* ``rotor_averaging = 0`` switches the filter off: the rotor sees the hub's wind.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import signal

from target_gym.energy.wind_turbine import env as W
from target_gym.registry import REGISTRY

SPEC = REGISTRY["wind_turbine"]
ENV = SPEC.make_env()
DT = 0.25
GAMMA, A = 1.3, 0.55  # [S24 Eq. 5.7]
R = 63.0  # m [S1 Table 1-1]


def _published(f_hz, v_mean):
    """|H(j 2 pi f)| of [S24 Eq. 5.7] at mean wind ``v_mean``."""
    s = 2j * math.pi * f_hz
    b = GAMMA * R / v_mean
    return abs(
        (math.sqrt(2) + b * s)
        / ((math.sqrt(2) + b * math.sqrt(A) * s) * (1 + b * s / math.sqrt(A)))
    )


@jax.jit
def _filtered(u, v_mean):
    """The rotor's turbulence for the hub's series ``u``, from rest."""
    p = W.WindTurbineParams()

    def body(lags, x):
        fast, slow, seen = W.rotor_turbulence(lags[0], lags[1], x, v_mean, p)
        return (fast, slow), seen

    return jax.lax.scan(body, (jnp.zeros(()), jnp.zeros(())), u)[1]


def test_a_steady_wind_reaches_the_rotor_whole():
    seen = _filtered(jnp.ones(2400), 18.0)
    assert float(seen[-1]) == pytest.approx(1.0, abs=1e-4)


# Each lag is stepped exactly with the hub's turbulence held over the 0.25 s
# step, so a sine comes out sin(x) / x smaller, x = pi f dt: 0.989 at the
# tower's 0.324 Hz and closer to 1 below it. Tolerance 3 %.
@pytest.mark.parametrize("v_mean", [13.0, 18.0, 25.0])
@pytest.mark.parametrize("f_hz", [0.02, 0.05, 0.1, 0.324])
def test_the_rotor_attenuates_a_sine_as_the_published_filter_does(f_hz, v_mean):
    t = np.arange(4800) * DT  # 20 min, settled long before the last 5 min
    seen = np.asarray(_filtered(jnp.sin(2 * math.pi * f_hz * t), v_mean), float)
    late = slice(3600, None)
    basis = np.stack(
        [np.sin(2 * math.pi * f_hz * t[late]), np.cos(2 * math.pi * f_hz * t[late])], 1
    )
    amplitude = np.linalg.norm(np.linalg.lstsq(basis, seen[late], rcond=None)[0])
    assert amplitude == pytest.approx(_published(f_hz, v_mean), rel=0.03)


def _held(p, seed, steps):
    """The plant under its reset's own pitch and torque, the overspeed level
    lifted so nothing restarts: (hub turbulence, rotor's wind, mean) per step,
    and the state it started from."""
    key = jax.random.PRNGKey(seed)
    _, s0 = ENV.reset_env(key, p)
    free = p.replace(overspeed_factor=1e9)
    held = ENV.initial_action(s0, p)

    def body(s, k):
        s1 = ENV.step_env(k, s, held, free)[1]
        return s1, (s1.v_turb, s1.v_wind, s1.v_mean)

    out = jax.lax.scan(body, s0, jax.random.split(key, steps))[1]
    return s0, [np.asarray(x, float) for x in out]


def test_the_rotors_wind_is_the_hubs_turbulence_through_the_published_filter():
    """Differential: the same transfer function discretised by scipy (zero-order
    hold) on the plant's own hub turbulence. The plant holds each new value
    over its step, so its series runs one step ahead of scipy's. Compared after
    100 s, 16 of the slow lag's 6 s, when the reset's own draw has died away.
    Tolerance 2e-3 m/s on a few m/s: float32 against float64."""
    v = 18.0
    p = SPEC.make_test_params(v_mean_range=(v, v), sensor_noise_scale=0.0)
    _, (hub, rotor, mean) = _held(p, seed=3, steps=2400)
    b = GAMMA * R / v
    num = [b, math.sqrt(2)]
    den = np.polymul([b * math.sqrt(A), math.sqrt(2)], [b / math.sqrt(A), 1.0])
    num_d, den_d, _ = signal.cont2discrete((num, den), DT, method="zoh")
    reference = signal.lfilter(np.ravel(num_d), den_d, hub)
    np.testing.assert_allclose(
        (rotor - mean)[:-1][400:], reference[1:][400:], atol=2e-3
    )


def test_with_rotor_averaging_off_the_rotor_sees_the_hubs_wind():
    p = SPEC.make_test_params(rotor_averaging=0.0, sensor_noise_scale=0.0)
    _, (hub, rotor, mean) = _held(p, seed=5, steps=200)
    np.testing.assert_allclose(rotor, mean + hub, rtol=1e-6)


def test_the_rotor_sees_a_smoother_wind_than_the_hub():
    """What the filter is for: at 18 m/s it passes 0.108 of a sine at the tower's
    0.324 Hz and 0.52 at 0.05 Hz [S24], so the rotor's wind moves far less from
    one 0.25 s step to the next than the hub's, and somewhat less overall."""
    p = SPEC.make_test_params(v_mean_range=(18.0, 18.0), sensor_noise_scale=0.0)
    _, (hub, rotor, mean) = _held(p, seed=7, steps=2400)
    assert np.diff(rotor).std() < 0.25 * np.diff(hub).std()
    assert 0.5 * hub.std() < (rotor - mean).std() < hub.std()


# 4000 episodes: a standard deviation is known to 1.1 %, a correlation near 0.9
# to 0.003. Tolerances 5 % and 0.02.
@pytest.mark.medium
def test_an_episode_starts_with_the_filter_in_its_long_run_law():
    p = SPEC.make_test_params(v_mean_range=(18.0, 18.0), sensor_noise_scale=0.0)
    free = p.replace(overspeed_factor=1e9)

    def one(key):
        _, s0 = ENV.reset_env(key, p)
        held = ENV.initial_action(s0, p)

        def body(s, k):
            return ENV.step_env(k, s, held, free)[1], None

        s1 = jax.lax.scan(body, s0, jax.random.split(key, 400))[0]
        return (
            jnp.stack([s0.v_wind - s0.v_mean, s0.v_turb]),
            jnp.stack([s1.v_wind - s1.v_mean, s1.v_turb]),
        )

    start, later = jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(0), 4000))
    start, later = np.asarray(start, float), np.asarray(later, float)
    assert start[:, 0].std() == pytest.approx(later[:, 0].std(), rel=0.05)
    assert np.corrcoef(start.T)[0, 1] == pytest.approx(
        np.corrcoef(later.T)[0, 1], abs=0.02
    )


def test_a_trip_restarts_the_plant_in_the_same_wind():
    """The wind does not know the turbine tripped (long strata, S2): the filter
    carries through a restart with the rest of the outside world."""
    assert {"v_lag_fast", "v_lag_slow"} <= set(ENV.OUTSIDE_WORLD)


# The receding oracle knows no future draw: it plans on the wind's mean path
# from where the wind is now (oracle.py, model). The filter is linear, so that
# path is the average of the plant's own wind over the draws to come. 4000
# draws of the next 10 s from one state; the average of 4000 is known to its
# standard deviation over sqrt(4000), and the test allows 5 of those.
@pytest.mark.medium
def test_the_oracles_wind_ahead_is_the_average_of_the_plants_wind_to_come():
    from target_gym.energy.wind_turbine import oracle as O

    p = SPEC.make_test_params(v_mean_range=(18.0, 18.0), sensor_noise_scale=0.0)
    free = p.replace(overspeed_factor=1e9)
    _, s0 = ENV.reset_env(jax.random.PRNGKey(11), p)
    held = ENV.initial_action(s0, p)
    steps = 40

    def run(key):
        def body(s, k):
            s1 = ENV.step_env(k, s, held, free)[1]
            return s1, s1.v_wind

        return jax.lax.scan(body, s0, jax.random.split(key, steps))[1]

    winds = np.asarray(
        jax.jit(jax.vmap(run))(jax.random.split(jax.random.PRNGKey(1), 4000)), float
    )
    model = O.wind_model(ENV, p)
    task, x = model.data(s0), model.state_x(s0)[:6]
    ahead = np.array(
        [
            float(O.wind_step(ENV, p, x, jnp.zeros(2), t, task, jnp.zeros(3))[1])
            for t in range(steps)
        ]
    )
    error = winds.std(axis=0) / math.sqrt(winds.shape[0])
    assert np.all(np.abs(ahead - winds.mean(axis=0)) <= 5.0 * error + 1e-4), (
        ahead - winds.mean(axis=0),
        error,
    )
