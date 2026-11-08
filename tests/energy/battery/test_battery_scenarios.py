"""The battery's scenario suite (charter rule 20), each from REALITY.md.

Contract:
- The suite builds through ``target_gym.scenarios.suite`` and each scenario
  runs an episode.
- normal: flat base price, energy-neutral signal (PNNL-ESS App. B).
- price spike: two settlement periods (one hourly price) between GBP 150 and
  491/MWh, the GB 2024 hourly high (Electricity Maps).
- negative price: two periods between GBP -28.2/MWh, the GB 2024 low, and 0.
- long deviation: a one-way response of 0.19 to 0.46 p.u., DR's response to a
  50 to 100 mHz mean deviation (ENTSOE-LER sec. 1, NESO-GUIDE p. 4), for 30
  to 90 min.
- wrong estimate: the BMS error spread of MPS-ACCURE's outliers, 20 %.
- long deviation on a wrong estimate: both, the record's bad day.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import scenarios
from target_gym.registry import REGISTRY

SPEC = REGISTRY["battery"]
SUITE = {s.name: s for s in scenarios.suite(SPEC)}
ENV = SPEC.make_env()


def _params(name, **extra):
    return scenarios.params_for(SPEC, SUITE[name], sensor_noise_scale=0.0, **extra)


def test_the_suite_has_the_documented_windows():
    assert set(SUITE) == {
        "normal",
        "price spike",
        "negative price",
        "long deviation",
        "wrong estimate",
        "long deviation, wrong estimate",
    }
    assert all(s.source for s in SUITE.values())


@pytest.mark.parametrize("name", sorted(SUITE))
def test_each_scenario_runs(name):
    params = _params(name, max_steps_in_episode=30)
    obs, state = ENV.reset_env(jax.random.PRNGKey(0), params)
    for _ in range(30):
        obs, state, reward, done, _ = ENV.step_env(
            jax.random.PRNGKey(1), state, jnp.zeros(1), params
        )
    assert np.all(np.isfinite(np.asarray(obs))) and np.isfinite(float(reward))


def _resets(name, n=200):
    params = _params(name)
    return jax.vmap(lambda k: ENV.reset_env(jax.random.PRNGKey(k), params)[1])(
        jnp.arange(n)
    )


def test_a_price_spike_is_one_hour_inside_the_family_range():
    prices = np.asarray(_resets("price spike").prices[:, :8])
    high = prices > 74.0
    assert np.all(high.sum(axis=1) == 2)
    assert prices[high].min() >= 150.0 and prices[high].max() <= 491.0


def test_a_negative_price_is_one_hour_inside_the_family_range():
    prices = np.asarray(_resets("negative price").prices[:, :8])
    low = prices < 74.0
    assert np.all(low.sum(axis=1) == 2)
    assert prices[low].min() >= -28.2 and prices[low].max() <= 0.0


def test_a_long_deviation_is_one_way_inside_the_family_range():
    states = _resets("long deviation")
    size = np.abs(np.asarray(states.bias_pu))
    length = np.asarray(states.bias_end - states.bias_start)
    assert size.min() >= 0.19 and size.max() <= 0.46
    assert length.min() >= 1800.0 and length.max() <= 5400.0


def test_a_long_deviation_moves_the_signal_one_way():
    """Over the event the mean response is the deviation's, on average over 40
    seeds within 10 % of it. One seed is not enough: the signal's own slow
    spread, before the event, is about 0.15 p.u."""
    params = _params("long deviation")
    steps = int(params.max_steps_in_episode)
    contracted = float(params.power_max / (1.0 + params.reserve_fraction))

    def one(seed):
        _, state = ENV.reset_env(jax.random.PRNGKey(seed), params)

        def body(s, _):
            _, s1, _, _, _ = ENV.step_env(
                jax.random.PRNGKey(seed + 1000), s, jnp.zeros(1), params
            )
            return s1, s1.regulation

        _, reg = jax.lax.scan(body, state, None, length=steps)
        t = jnp.arange(1, steps + 1) * params.delta_t
        inside = (t >= state.bias_start) & (t < state.bias_end)
        mean = jnp.sum(jnp.where(inside, reg, 0.0)) / jnp.sum(inside) / contracted
        return mean / state.bias_pu

    ratios = np.asarray(jax.jit(jax.vmap(one))(jnp.arange(40)))
    assert ratios.mean() == pytest.approx(1.0, abs=0.1)


def test_a_wrong_estimate_has_the_outliers_spread():
    states = _resets("wrong estimate", n=400)
    sd = float(np.std(np.asarray(states.soc_error)))
    assert sd == pytest.approx(0.20, rel=0.15)
