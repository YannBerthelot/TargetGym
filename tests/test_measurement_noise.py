"""Measurement noise, tested against docs/proposals/measurement-noise.md.

Contract, taken from the proposal and not from the code:

- A reading is ``value + scale * sigma * epsilon``, then rounded to the
  instrument's step. ``epsilon`` is standard normal, per reading and update.
- ``sensor_noise_scale`` defaults to 1, and 0 gives the noise-free task.
- Setpoints stay exact. Trips, termination and ``reward`` read the true state.
- A reading is a pure function of the state, so ``get_obs`` needs no key, and
  the same episode key gives the same noise to every policy.
- A slow instrument holds ``epsilon`` between updates while the value stays
  current.
- ``info["measured_reward"]`` holds the tracking and consumption terms on the
  measured copy, with the trip charge from the true state. On a measured copy
  the cost has a floor of ``(sigma / e_floor)^2`` per step.
- A channel reading does not inherit the noise of the fields behind it.
- The oracle's planning params are noise-free.

The parametrised tests read the registry, so a task that declares readings is
held to the shared contract the day it declares them.
"""

import dataclasses
import operator
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from target_gym.base import (
    ChannelReading,
    EnvParams,
    EnvState,
    FieldReading,
    measured_state,
)
from target_gym.benchmark import episodes_of, run_policy, task_info
from target_gym.eval import time_in_band
from target_gym.experts.mpc import plan_params
from target_gym.pc_gym.four_tank.env import FourTankParams
from target_gym.pc_gym.four_tank.env_jax import FourTank
from target_gym.registry import REGISTRY


def _declares_readings(spec):
    return bool(getattr(spec.make_env(), "readings", ()))


NOISY = [s for s in REGISTRY.values() if _declares_readings(s)]
noisy_tasks = pytest.mark.parametrize("spec", NOISY, ids=lambda s: s.name)

#: Enough steps to see several draws and, on the pumps-off policy, a trip.
STEPS = 40


def _bounds(env, params):
    space = env.action_space(params)
    low = jnp.broadcast_to(space.low, space.shape).astype(jnp.float32)
    high = jnp.broadcast_to(space.high, space.shape).astype(jnp.float32)
    return low, high


def _own_env(spec):
    """An instance of the task to change: ``spec.make_env()`` is shared, so a
    test that sets ``readings`` on it would change every later test."""
    return spec.env_factory()


def _unrounded(env):
    """Rounding hides ``epsilon``, and so does a late value, so read the draws
    with no digital step and no age."""
    env.readings = tuple(
        dataclasses.replace(r, step=0.0, dead_time=0.0, sampled=False)
        for r in env.readings
    )
    return env


def _rollout(env, params, key, action, steps=STEPS, step_stream=0):
    """``[(obs, state, reward, terminated, info)]``, the reset first.
    ``step_stream`` picks the per-step keys, which only ``key`` must outlive."""
    step = jax.jit(env.step_env)
    obs, state = env.reset_env(key, params)
    out = [(obs, state, None, None, None)]
    stream = jax.random.fold_in(key, step_stream)
    for t in range(steps):
        obs, state, reward, done, info = step(
            jax.random.fold_in(stream, t), state, action, params
        )
        out.append((obs, state, reward, done, info))
    return out


def _quiet(params):
    return params.replace(sensor_noise_scale=0.0)


def _added(env, state, params):
    """``(what each reading adds at state, the values it adds to)``, flat.

    Read at the readings, not across the observation: a channel derived from a
    reading (an aircraft's path error, its ``gamma``) takes a share of the same
    draw that depends on the plant's state.
    """
    measured = measured_state(env, state, params)
    fields = [
        operator.attrgetter(r.field)
        for r in env.readings
        if isinstance(r, FieldReading)
    ]
    values = [np.ravel(f(state)) for f in fields]
    added = [np.ravel(f(measured)) - v for f, v in zip(fields, values)]
    channels = [r.channel for r in env.readings if isinstance(r, ChannelReading)]
    if channels:
        true = np.asarray(env.get_obs(state, _quiet(params)))[channels]
        read = np.asarray(env.get_obs(state, params))[channels]
        values.append(true)
        added.append(read - true)
    return np.concatenate(added), np.concatenate(values)


# ---------------------------------------------------------------------------
# One episode key, one noise path
# ---------------------------------------------------------------------------


@noisy_tasks
def test_two_policies_on_one_episode_key_see_identical_noise_draws(spec):
    env = _unrounded(_own_env(spec))
    params = spec.make_test_params()
    key = jax.random.PRNGKey(3)
    low, high = _bounds(env, params)

    def draws(action, step_stream):
        rows = _rollout(env, params, key, action, step_stream=step_stream)
        added, values = zip(*(_added(env, s, params) for _, s, *_ in rows))
        return np.stack(added), np.stack(values)

    # Different step keys too: the noise belongs to the episode key alone.
    a, true_a = draws(low, 1)
    b, true_b = draws(high, 2)
    # Four float32 steps of the larger value: the subtraction's rounding.
    tol = 4 * np.spacing(np.maximum(np.abs(true_a), np.abs(true_b)).astype(np.float32))
    assert np.all(np.abs(a - b) <= tol)
    assert np.any(a != 0.0), "no reading carried any noise"
    assert np.any(a[1:] != a[:-1]), "the draws never changed"


def test_noise_path_survives_a_trip():
    """A policy that trips restarts the plant, not the instruments: the noise
    after the restart is the noise of a policy that never tripped."""
    env = _unrounded(FourTank())
    params = FourTankParams()
    key = jax.random.PRNGKey(3)
    pumps_off = _rollout(env, params, key, jnp.array([-1.0, -1.0]), step_stream=1)
    steady = _rollout(env, params, key, jnp.array([0.0, 0.0]), step_stream=2)
    assert any(bool(info["tripped"]) for *_, info in pumps_off[1:])
    assert not any(bool(info["tripped"]) for *_, info in steady[1:])
    for (obs_a, state_a, *_), (obs_b, state_b, *_) in zip(pumps_off, steady):
        np.testing.assert_allclose(
            obs_a - env.get_obs(state_a, _quiet(params)),
            obs_b - env.get_obs(state_b, _quiet(params)),
            atol=3e-7,  # 4 float32 steps at the 1.5 m overflow level
        )


@noisy_tasks
def test_a_reading_is_a_pure_function_of_the_state(spec):
    """Experts, tuners and Ajax's ``get_raw_obs`` all call ``get_obs(state,
    params)``, and each must see what the step returned."""
    env, params = spec.make_env(), spec.make_test_params()
    key = jax.random.PRNGKey(0)
    low, _ = _bounds(env, params)
    obs, state = env.reset_env(key, params)
    np.testing.assert_array_equal(obs, env.get_obs(state, params))
    obs, state, *_ = jax.jit(env.step_env)(key, state, low, params)
    # Compiled into the step, a derived channel rounds a float32 step or so
    # apart: 1e-3 m on the racetrack's 15 km coordinates, or 1 part in 1.6e7.
    np.testing.assert_allclose(obs, env.get_obs(state, params), rtol=1e-6, atol=2e-3)
    np.testing.assert_array_equal(
        env.get_obs(state, params), env.get_obs(state, params)
    )


@noisy_tasks
def test_a_step_keeps_the_episodes_instruments(spec):
    """The noise belongs to the episode key, so every step must carry it. A
    ``compute_next_state`` that builds a new state from its fields drops the key
    to its default, and from the second step every episode on every seed then
    meets one noise path."""
    env, params = spec.make_env(), spec.make_test_params()
    key = jax.random.PRNGKey(6)
    low, _ = _bounds(env, params)
    _, state = env.reset_env(key, params)
    assert np.any(np.asarray(state.sensor_key) != 0)
    new = jax.jit(env.step_env)(key, state, low, params)[1]
    np.testing.assert_array_equal(new.sensor_key, state.sensor_key)


def test_different_episode_keys_give_different_noise():
    env = _unrounded(FourTank())
    params = FourTankParams()
    _, a = env.reset_env(jax.random.PRNGKey(0), params)
    _, b = env.reset_env(jax.random.PRNGKey(1), params)
    b = b.replace(h1=a.h1, h2=a.h2, h3=a.h3, h4=a.h4, target_h1=a.target_h1)
    b = b.replace(target_h2=a.target_h2)
    assert not np.array_equal(env.get_obs(a, params), env.get_obs(b, params))


# ---------------------------------------------------------------------------
# Scale, targets
# ---------------------------------------------------------------------------


def test_sensor_noise_is_on_by_default():
    assert EnvParams().sensor_noise_scale == 1.0
    assert FourTankParams().sensor_noise_scale == 1.0


@noisy_tasks
def test_scale_zero_is_the_noise_free_task_exactly(spec):
    """Rounding goes too. An instrument's age stays: a reading with no scatter
    and no step still reports late (tests/test_sensor_delay.py)."""
    noisy, free = _own_env(spec), _own_env(spec)
    free.readings = tuple(
        dataclasses.replace(r, sigma=0.0, step=0.0)
        for r in free.readings
        if r.dead_time > 0 or r.sampled
    )
    params = spec.make_test_params()
    key = jax.random.PRNGKey(5)
    low, high = _bounds(noisy, params)
    action = 0.5 * (low + high)
    for a, b in zip(
        _rollout(noisy, _quiet(params), key, action),
        _rollout(free, params, key, action),
    ):
        np.testing.assert_array_equal(a[0], b[0])
        jax.tree_util.tree_map(np.testing.assert_array_equal, a[1], b[1])
        if a[2] is not None:
            np.testing.assert_array_equal(a[2], b[2])
            np.testing.assert_array_equal(a[3], b[3])
            np.testing.assert_array_equal(
                a[4]["measured_reward"], b[4]["measured_reward"]
            )
            if not free.readings:  # a late reading's measured cost is late too
                np.testing.assert_array_equal(a[4]["measured_reward"], a[2])


@noisy_tasks
def test_scale_scales_the_noise(spec):
    """Twice the scale gives twice the scatter of the same draws."""
    env = _unrounded(_own_env(spec))
    params = spec.make_test_params()
    _, state = env.reset_env(jax.random.PRNGKey(1), params)
    one, values = _added(env, state, params)
    two, _ = _added(env, state, params.replace(sensor_noise_scale=2.0))
    tol = 8 * np.spacing(np.abs(np.asarray(values, np.float32)))
    assert np.all(np.abs(two - 2.0 * one) <= tol)


@noisy_tasks
def test_targets_stay_exact(spec):
    env, params = spec.make_env(), spec.make_test_params()
    key = jax.random.PRNGKey(2)
    low, _ = _bounds(env, params)
    index = jnp.atleast_1d(jnp.array(env.obs_target_index))
    for obs, state, *_ in _rollout(env, params, key, low, steps=5):
        np.testing.assert_array_equal(
            obs[index], env.get_obs(state, _quiet(params))[index]
        )


def test_readings_round_to_the_instrument_step():
    """Four-tank levels: 1 mm transmitter resolution."""
    env, params = FourTank(), FourTankParams()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    levels = np.asarray(env.get_obs(state, params))[:4]
    np.testing.assert_allclose(levels * 1e3, np.round(levels * 1e3), atol=1e-3)


def test_a_rounded_reading_keeps_its_slope():
    """A gradient tuner differentiates through ``get_obs``. A rounded reading
    is flat almost everywhere, which would cut the loop's feedback out of the
    gradient while the forward pass looks fine, so the slope stays the
    instrument's: one level of reading per level of tank."""
    env, params = FourTank(), FourTankParams()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    slope = jax.grad(lambda h: env.get_obs(state.replace(h1=h), params)[0])(state.h1)
    assert float(slope) == pytest.approx(1.0)


def test_step_returns_a_reading_at_the_params_it_was_given():
    """A params override reaches the observation, not only ``get_obs``."""
    env, params = FourTank(), _quiet(FourTankParams())
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)
    np.testing.assert_array_equal(obs[:4], [state.h1, state.h2, state.h3, state.h4])
    obs, state, *_ = env.step_env(key, state, jnp.zeros(2), params)
    np.testing.assert_array_equal(obs[:4], [state.h1, state.h2, state.h3, state.h4])


# ---------------------------------------------------------------------------
# What stays true
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 1000.0])
def test_noise_never_fires_a_trip_or_changes_reward(scale):
    """h4 sits 0.1 mm above the low-level trip, filling. At scale 1000 a
    reading lands 0.5 m away, far outside the 0.05 to 1.5 m envelope half the
    time, so a trip read from a measured copy would fire within steps."""
    env = FourTank()
    params = FourTankParams()
    key = jax.random.PRNGKey(0)
    noisy = params.replace(sensor_noise_scale=scale)
    _, start = env.reset_env(key, noisy)
    start = start.replace(h4=jnp.float32(params.h_min + 1e-4))
    action = jnp.array([1.0, 1.0])
    charge = params.restart_steps * params.failure_cost

    def run(p):
        step, s, out = jax.jit(env.step_env), start, []
        for t in range(30):
            _, s, reward, done, info = step(jax.random.fold_in(key, t), s, action, p)
            out.append((s, reward, done, info))
        return out

    for a, b in zip(run(noisy), run(_quiet(noisy))):
        jax.tree_util.tree_map(np.testing.assert_array_equal, a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[2], b[2])
        assert not bool(a[3]["tripped"]) and not bool(b[3]["tripped"])
        assert float(a[3]["trip_cost"]) == 0.0
        assert np.isfinite(a[3]["measured_reward"])
        assert float(a[3]["measured_reward"]) > -charge


def test_measured_reward_charges_a_true_trip_and_nothing_else():
    """On the step that trips, the measured reward is the trip charge, as the
    true reward is: ``reward.with_trip`` zeroes every other term."""
    env, params = FourTank(), FourTankParams()
    rows = _rollout(env, params, jax.random.PRNGKey(0), jnp.array([-1.0, -1.0]), 60)
    trips = [(r, info) for _, _, r, _, info in rows[1:] if bool(info["tripped"])]
    assert trips, "pumps off did not trip within 60 steps"
    charge = params.restart_steps * params.failure_cost
    for reward, info in trips:
        assert float(reward) == pytest.approx(-charge, rel=1e-6)
        assert float(info["measured_reward"]) == pytest.approx(-charge, rel=1e-6)


def test_measured_reward_is_the_cost_of_the_reading_the_policy_received():
    """A plant computes reward from the sensors it reports, so the reward and
    the observation of a step must carry one draw. Two tanks, quadratic, in
    units of ``e_floor``."""
    env, params = FourTank(), FourTankParams()
    key = jax.random.PRNGKey(1)
    _, state = env.reset_env(key, params)
    for t in range(5):
        obs, state, _, _, info = env.step_env(
            jax.random.fold_in(key, t), state, jnp.zeros(2), params
        )
        h1, h2, _, _, target_1, target_2 = np.asarray(obs, np.float64)
        expected = ((target_1 - h1) / 1e-3) ** 2 + ((target_2 - h2) / 1e-3) ** 2
        assert float(-info["measured_reward"]) == pytest.approx(expected, rel=1e-4)


def _steady_levels(params: FourTankParams, volts: float):
    """The four levels at which both pumps at ``volts`` hold still, from the
    mass balances: each upper tank drains what its pump sends it, and each
    lower tank drains its upper tank plus its pump's direct share."""
    root = np.sqrt(2.0 * params.g)
    q1, q2 = params.k1 * volts, params.k2 * volts
    h3 = ((1 - params.gamma2) * q2 / (params.a3 * root)) ** 2
    h4 = ((1 - params.gamma1) * q1 / (params.a4 * root)) ** 2
    h1 = (((1 - params.gamma2) * q2 + params.gamma1 * q1) / (params.a1 * root)) ** 2
    h2 = (((1 - params.gamma1) * q1 + params.gamma2 * q2) / (params.a2 * root)) ** 2
    return h1, h2, h3, h4


def _hold_costs(env, params, n_episodes=64, n_steps=500):
    """True and measured cost per step of a plant resting exactly on target."""
    h1, h2, h3, h4 = _steady_levels(params, 5.0)  # raw action 0 is 5 V
    action = jnp.zeros(2)

    def episode(key):
        _, state = env.reset_env(key, params)
        state = state.replace(
            h1=jnp.float32(h1),
            h2=jnp.float32(h2),
            h3=jnp.float32(h3),
            h4=jnp.float32(h4),
            target_h1=jnp.float32(h1),
            target_h2=jnp.float32(h2),
        )

        def step(s, k):
            _, s, reward, _, info = env.step_env(k, s, action, params)
            return s, (-reward, -info["measured_reward"])

        return jax.lax.scan(step, state, jax.random.split(key, n_steps))[1]

    keys = jax.random.split(jax.random.PRNGKey(0), n_episodes)
    true, measured = jax.jit(jax.vmap(episode))(keys)
    return float(true.mean()), float(measured.mean()), true.size


def test_measured_reward_carries_the_proposals_floor():
    """Without rounding, the table's four_tank row: two tanks at
    ``(5e-4 / 1e-3)^2 = 0.25`` each, 0.5 per step.

    Each tank's cost is 0.25 times a chi-squared with one degree of freedom,
    variance ``2 * 0.25^2``, so the sum has standard deviation 0.5 per step and
    0.5 / sqrt(N) over N independent steps. The tolerance is 4 of those.
    """
    env = _unrounded(FourTank())
    true, measured, n = _hold_costs(env, FourTankParams())
    assert measured == pytest.approx(0.5, abs=4 * 0.5 / np.sqrt(n))
    assert true < 1e-2, "the plant did not hold its target"


def test_rounding_adds_its_quantisation_variance_to_the_floor():
    """The shipped reading rounds to 1 mm, which the table's formula leaves
    out: uniform quantisation error adds ``step^2 / 12`` to the variance, so
    each tank costs ``(2.5e-7 + 1e-6 / 12) / 1e-6 = 1 / 3`` and the two
    ``2 / 3``. The formula holds at sigma = step / 2 to about 0.2% (Widrow's
    condition), inside the 4 standard errors allowed here."""
    env = FourTank()
    true, measured, n = _hold_costs(env, FourTankParams())
    assert measured == pytest.approx(2.0 / 3.0, abs=4 * (2.0 / 3.0) / np.sqrt(n))
    assert true < 1e-2, "the plant did not hold its target"


class _HoldStill:
    """The smallest benchmark policy: raw action 0 on every step."""

    name = "still"

    def reset(self, key, obs, task):
        return jnp.zeros((task.n_episodes, task.n_act))

    def act(self, carry, obs, t):
        return carry, carry


#: Shared, so the rollouts below compile once.
_RESTING_PLANT = FourTank()


def _resting_on(offset):
    """Four-tank params whose every episode starts at rest at raw action 0,
    with both targets ``offset`` metres above the levels it rests on."""
    params = FourTankParams()
    h1, h2, h3, h4 = (float(h) for h in _steady_levels(params, 5.0))
    return params.replace(
        initial_h1_range=(h1, h1),
        initial_h2_range=(h2, h2),
        initial_h3_range=(h3, h3),
        initial_h4_range=(h4, h4),
        target_h1_range=(h1 + offset, h1 + offset),
        target_h2_range=(h2 + offset, h2 + offset),
    )


def _hold_rollout(offset):
    params = _resting_on(offset)
    task = task_info("four_tank", _RESTING_PLANT, params, 4)
    key = jax.random.PRNGKey(0)
    return run_policy(_HoldStill(), _RESTING_PLANT, params, task, key, n_steps=200)


def test_a_perfect_hold_counts_as_always_in_band():
    """``time_in_band`` and the settle read the plant, not its instruments.

    The band is ``e_floor``, 1 mm, on each tank, and a reading scatters by
    0.5 mm with 1 mm rounding on top. So a reading alone leaves the band on
    about 5% of steps per tank (2 sigma), and 800 steps read from ``obs`` cannot
    all stay in. The policy still reads its instruments: ``value`` is what it
    saw, and it is off target.
    """
    out = _hold_rollout(0.0)
    assert out["in_band"].all()
    assert time_in_band(episodes_of(out), 0) == 1.0
    assert np.any(out["value"] != out["target"])


def test_per_episode_plants_read_the_band_on_their_own_true_outputs():
    """A study that gives each episode its own params, here two plants on
    target and two off it by 2 mm."""
    on, off = _resting_on(0.0), _resting_on(2e-3)
    params = jax.tree.map(lambda a, b: jnp.stack([a, a, b, b]), on, off)
    task = task_info("four_tank", _RESTING_PLANT, params, 4, per_episode_params=True)
    out = run_policy(
        _HoldStill(),
        _RESTING_PLANT,
        params,
        task,
        jax.random.PRNGKey(0),
        n_steps=50,
        per_episode_params=True,
    )
    assert out["in_band"][:2].all()
    assert not out["in_band"][2:].any()


def test_a_hold_outside_the_band_counts_as_never_in_band():
    """2 mm off a 1 mm band, on the true levels. Noise would carry a reading
    inside on about 2% of steps, and that must not count."""
    out = _hold_rollout(2e-3)
    assert not out["in_band"].any()
    assert time_in_band(episodes_of(out), 0) == 0.0


def test_oracle_planning_params_are_noise_free():
    """Every task, whether or not its readings are declared yet."""
    for spec in REGISTRY.values():
        params = spec.make_test_params().replace(
            sensor_noise_scale=2.0, action_noise=0.01
        )
        planned = plan_params(spec, params)
        assert planned.sensor_noise_scale == 0.0, spec.name
        assert planned.action_noise == 0.0, spec.name


# ---------------------------------------------------------------------------
# Held noise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "delta_t, width",
    [(1.0, 5), (2.5, 2)],
    ids=["every-5-steps", "every-2-steps-at-a-longer-step"],
)
def test_held_noise_stays_fixed_for_an_interval_and_changes_at_the_next(delta_t, width):
    """A 5 s instrument. The value stays current, so the reading follows the
    plant while ``epsilon`` stands still."""
    env = FourTank()
    env.readings = (FieldReading("h1", sigma=5e-4, interval=5.0),)
    params = FourTankParams(delta_t=delta_t)
    rows = _rollout(env, params, jax.random.PRNGKey(0), jnp.zeros(2), 3 * width - 1)
    read = np.array([o[0] for o, *_ in rows])
    true = np.array([s.h1 for _, s, *_ in rows])
    noise = read - true
    blocks = [noise[i * width : (i + 1) * width] for i in range(3)]
    for block in blocks:
        np.testing.assert_allclose(block, block[0], atol=3e-8)  # 2 float32 steps
    assert abs(blocks[0][0] - blocks[1][0]) > 1e-6
    assert abs(blocks[1][0] - blocks[2][0]) > 1e-6
    assert len(set(true.tolist())) == len(true), "the plant did not move"


# ---------------------------------------------------------------------------
# Field and channel readings
# ---------------------------------------------------------------------------


@struct.dataclass
class _Lead:
    z: float


@struct.dataclass
class _Column(EnvState):
    x: jnp.ndarray
    lead: _Lead


def _noise(readings, field, times=2000):
    """``field`` of the measured copy less the true value, over ``times`` steps."""
    env = types.SimpleNamespace(readings=readings)
    params = EnvParams()
    state = _Column(
        time=0,
        x=jnp.zeros(16),
        lead=_Lead(z=jnp.float32(0.0)),
        sensor_key=jax.random.PRNGKey(7),
    )

    def at(t):
        return field(measured_state(env, state.replace(time=t), params))

    return np.asarray(jax.vmap(at)(jnp.arange(times)))


def test_an_array_field_takes_a_sigma_per_index():
    """``distillation``: analysers at the ends of ``x``, a tray sensor at 14,
    nothing else measured. 2000 draws put a sample standard deviation within
    ``4 / sqrt(2 * 2000)``, 9%, of its sigma."""
    noise = _noise(
        (
            FieldReading("x", index=0, sigma=4e-4),
            FieldReading("x", index=14, sigma=3e-3),
            FieldReading("x", index=-1, sigma=4e-4),
        ),
        lambda s: s.x,
    )
    assert noise[:, 0].std() == pytest.approx(4e-4, rel=0.09)
    assert noise[:, 14].std() == pytest.approx(3e-3, rel=0.09)
    assert noise[:, -1].std() == pytest.approx(4e-4, rel=0.09)
    assert np.all(np.delete(noise, [0, 14, 15], axis=1) == 0.0)
    assert abs(np.corrcoef(noise[:, 0], noise[:, -1])[0, 1]) < 0.1


def test_an_index_reading_can_differ_in_interval_from_its_neighbour():
    """A gas chromatograph on one index and a per-step sensor on another."""
    noise = _noise(
        (
            FieldReading("x", index=0, sigma=1.0, interval=4.0),
            FieldReading("x", index=1, sigma=1.0),
        ),
        lambda s: s.x,
        times=40,
    )
    assert len(np.unique(noise[:, 0])) == 10
    assert len(np.unique(noise[:, 1])) == 40


def test_an_array_field_can_take_a_sigma_per_element():
    noise = _noise(
        (FieldReading("x", sigma=tuple(float(i) for i in range(16))),),
        lambda s: s.x,
    )
    assert np.all(noise[:, 0] == 0.0)
    assert noise[:, 3].std() == pytest.approx(3.0, rel=0.09)
    assert noise[:, 15].std() == pytest.approx(15.0, rel=0.09)


def test_a_field_reading_reaches_a_nested_field():
    """``lead.x`` on the patrol tasks."""
    noise = _noise((FieldReading("lead.z", sigma=0.2),), lambda s: s.lead.z)
    assert noise.std() == pytest.approx(0.2, rel=0.09)


def test_a_shared_reading_noises_a_profile_after_its_max():
    """``cement_kiln``'s pyrometer reads ``max(T_solid)`` once. Per-zone noise
    biases that max upward, by about 1.77 sigma on 16 zones; one draw for the
    whole profile leaves it unbiased, with the mean inside 4 standard errors."""
    sigma = 1.6
    apart = _noise((FieldReading("x", sigma=sigma),), lambda s: s.x.max())
    shared = _noise((FieldReading("x", sigma=sigma, shared=True),), lambda s: s.x.max())
    standard_error = sigma / np.sqrt(2000)
    assert apart.mean() > sigma, "independent draws should bias the max up"
    assert abs(shared.mean()) < 4 * standard_error
    assert shared.std() == pytest.approx(sigma, rel=0.09)


def test_an_index_reading_supersedes_a_shared_one_on_its_element():
    """``T_solid[0]`` has its own pyrometer: its noise is that sensor's alone."""
    noise = _noise(
        (
            FieldReading("x", sigma=1.6, shared=True),
            FieldReading("x", index=0, sigma=1.2),
        ),
        lambda s: s.x,
    )
    assert noise[:, 0].std() == pytest.approx(1.2, rel=0.09)
    assert abs(np.corrcoef(noise[:, 0], noise[:, 1])[0, 1]) < 0.1


def test_a_channel_reading_does_not_inherit_the_noise_of_its_fields():
    """``wind_turbine``: noisy torque times speed gives 10.6 kW against a 1 kW
    sensor. Here a field at sigma 1 feeds channel 0, which reads at 1e-3, and
    the channel must scatter at 1e-3."""
    env = FourTank()
    env.readings = (
        FieldReading("h1", sigma=1.0),
        ChannelReading(0, sigma=1e-3),
    )
    params = FourTankParams()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)

    def read(t):
        return env.get_obs(state.replace(time=t), params)[0] - state.h1

    noise = np.asarray(jax.vmap(read)(jnp.arange(2000)))
    assert noise.std() == pytest.approx(1e-3, rel=0.09)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


@noisy_tasks
def test_noise_runs_under_jit_vmap_and_scan(spec):
    """A scanned, vmapped, jitted rollout returns the eager rollout's readings,
    within float32 fusion (the reset's copy compiled inside the step differs
    by a step or so), and nothing in it needs a key but the episode's."""
    env, params = spec.make_env(), spec.make_test_params()
    keys = jax.random.split(jax.random.PRNGKey(4), 3)
    low, high = _bounds(env, params)
    action = 0.5 * (low + high)

    def episode(key):
        obs, state = env.reset_env(key, params)

        stream = jax.random.fold_in(key, 0)

        def step(s, t):
            o, s, _, _, info = env.step_env(
                jax.random.fold_in(stream, t), s, action, params
            )
            return s, (o, info["measured_reward"])

        return obs, jax.lax.scan(step, state, jnp.arange(8))[1]

    first, (obs, measured) = jax.jit(jax.vmap(episode))(keys)
    assert obs.shape == (3, 8, *first.shape[1:])
    assert measured.shape == (3, 8)
    assert np.all(np.isfinite(obs)) and np.all(np.isfinite(measured))

    rows = _rollout(env, params, keys[1], action, steps=8)
    eager = np.stack([np.asarray(o) for o, *_ in rows[1:]])
    # atol: a channel derived from 15 km coordinates has a 1 mm float32 step.
    np.testing.assert_allclose(obs[1], eager, rtol=1e-5, atol=2e-3)
    np.testing.assert_allclose(first[1], rows[0][0], rtol=1e-5, atol=2e-3)
