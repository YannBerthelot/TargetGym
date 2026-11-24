"""``plane``'s mountain-wave scenarios (charter B5; plane/REALITY.md, "What
threatens the speed in cruise").

Contract, with its sources:

- ``wave_downdraft``. Once in the window the air sinks under the aircraft: "a
  large or prolonged downdraft ... due to orographic waves. The downdraft may have
  a negative vertical speed of more than 500 ft/min" [SF21 p. 13]. One downdraft,
  still air before and after, its peak between 3.0 m/s, where ICAO has a severe
  mountain wave [SIGMET], and 4.5 m/s, the top of the Sierra Wave Project's weak
  class [WILDMANN]. It lasts 25 km of track, the scale of the one recorded case
  at cruise level [BRAMBERGER Figure 2], at the assigned speed.
- ``wave_headwind_loss``. Once in the window the along-track wind turns toward a
  tailwind over 25 km of track and stays there: about 20 m/s over about 25 km in
  the recorded case [BRAMBERGER Figure 2]. The scenario draws 10 to 20 m/s (the
  lower half ours).
- The air data read a wave at once: as the tailwind rises the airspeed falls by
  as much, before the aircraft has moved.
- Each weighs 1 % of cruise time (owner, 2026-11-23: no source gives how often an
  airliner meets one). No other scenario has a wave.
- A wave is not in the observation, as a gust is not; a wind sensor shows it.

Hand-derived: 25 km at the M 0.78 schedule's 231 to 236 m/s is 106 to 108 s. A
half-sine downdraft of peak A is below A / 2 for two thirds of its length.
"""

import jax
import numpy as np
import pytest

import target_gym as tg
from target_gym.plane.env_jax import Airplane2D
from target_gym.plane.scenarios import suite

SEEDS = (100, 101, 102)
TRACK = 25_000.0  # m [BRAMBERGER Figure 2]


def _scenario(name):
    return next(s for s in suite() if s.name == name)


#: One wind-sensor environment and one compiled step for the whole module: the
#: parameters are an argument, so every scenario runs the same program.
_SENSOR = Airplane2D(observe_wind=True)
_STEP = jax.jit(_SENSOR.step_env)


def _window(name, seed, **changes):
    """The wind a sensor reads through one calm window of scenario ``name`` flown
    on the trim action: ``(along-track m/s, vertical m/s, airspeed m/s, assigned
    true airspeed m/s)``, one row a second."""
    sensor, step = _SENSOR, _STEP
    _, params = tg.make(
        "plane",
        **{
            **dict(_scenario(name).overrides),
            "turbulence_sigma": 0.0,
            "sensor_noise_scale": 0.0,
            **changes,
        },
    )
    key = jax.random.PRNGKey(seed)
    obs, state = sensor.reset_env(key, params)
    action = sensor.trim_action(state, params)
    assigned = float(obs[sensor.obs_index["target_speed"]])
    rows = [np.asarray(obs)]
    for k in jax.random.split(key, int(params.max_steps_in_episode) - 1):
        obs, state, *_ = step(k, state, action, params)
        rows.append(np.asarray(obs))
    rows = np.stack(rows)
    return rows[:, 12], rows[:, 13], rows[:, sensor.obs_index["airspeed"]], assigned


@pytest.mark.parametrize("seed", SEEDS)
def test_the_downdraft_is_one_sink_of_severe_wave_size_over_25_km_of_track(seed):
    along, vertical, _, assigned = _window("wave_downdraft", seed)
    assert np.all(along == 0.0)
    assert vertical.max() <= 0.0
    assert 3.0 <= -vertical.min() <= 4.5
    sinking = np.flatnonzero(vertical < 0.0)
    assert np.all(np.diff(sinking) == 1)  # one stretch, still air either side
    assert len(sinking) == pytest.approx(TRACK / assigned, rel=0.1)
    # A half-sine: below half its peak for two thirds of its length.
    deep = np.count_nonzero(vertical < 0.5 * vertical.min())
    assert deep == pytest.approx(2.0 / 3.0 * len(sinking), rel=0.1)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_head_wind_falls_by_10_to_20_m_s_over_25_km_and_stays(seed):
    along, vertical, _, assigned = _window("wave_headwind_loss", seed)
    assert np.all(vertical == 0.0)
    gain = along[-1] - along[0]
    assert along[0] == 0.0
    assert 10.0 <= gain <= 20.0
    assert np.all(np.diff(along) >= 0.0)  # toward a tailwind, never back
    turning = np.flatnonzero((along > 0.01 * gain) & (along < 0.99 * gain))
    assert len(turning) <= TRACK / assigned
    assert len(turning) >= 0.7 * TRACK / assigned


def test_the_air_data_read_the_lost_head_wind_at_once():
    """Over the first ten seconds of the turn the aircraft's inertia keeps its
    ground speed, so the airspeed falls by what the tailwind gains (10 %: drag
    falls with the airspeed and gives a little of it back)."""
    along, _, airspeed, _ = _window("wave_headwind_loss", SEEDS[0])
    start = int(np.flatnonzero(along > 0.0)[0])
    gained = along[start + 10] - along[start - 1]
    lost = airspeed[start - 1] - airspeed[start + 10]
    assert gained > 0.2
    assert lost == pytest.approx(gained, rel=0.1)


@pytest.mark.parametrize(
    "name", [s.name for s in suite() if not s.name.startswith("wave_")]
)
def test_no_other_scenario_has_a_wave(name):
    along, vertical, _, _ = _window(name, SEEDS[0], max_steps_in_episode=600)
    assert np.all(along == 0.0) and np.all(vertical == 0.0)


def test_each_wave_scenario_weighs_one_percent_of_cruise_time():
    weights = {s.name: s.weight for s in suite()}
    total = sum(weights.values())
    for name in ("wave_downdraft", "wave_headwind_loss"):
        assert weights[name] / total == pytest.approx(0.01, abs=1e-4)


def test_a_wave_is_not_in_the_observation():
    env, params = tg.make("plane", **dict(_scenario("wave_downdraft").overrides))
    obs, _ = env.reset_env(jax.random.PRNGKey(0), params)
    assert obs.shape == (12,)


@pytest.mark.medium
@pytest.mark.parametrize("name", ["wave_downdraft", "wave_headwind_loss"])
def test_the_reference_flies_a_wave_without_a_trip(name):
    """The reference keeps flying through each wave on four seeds. Its cost is
    printed, not asserted: it is the room's numerator."""
    env, params = tg.make("plane", **dict(_scenario(name).overrides))
    expert = tg.make_expert("plane")
    step, act = jax.jit(env.step_env), jax.jit(expert.step)
    for seed in (100, 101, 102, 103):
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset_env(key, params)
        memory = expert.reset()
        cost, tripped = 0.0, False
        for k in jax.random.split(key, int(params.max_steps_in_episode)):
            action, memory = act(memory, obs)
            obs, state, reward, _, info = step(k, state, action, params)
            tripped |= bool(info["tripped"])
            cost -= float(reward)
        assert not tripped
        print(f"{name}, seed {seed}: reference cost {cost:.1f}")
