"""The ten aircraft tasks' readings, against docs/proposals/measurement-noise.md.

Contract, from the proposal's Aircraft table and its refutation notes, not from
the code:

- Each reading scatters by its table sigma and lands on its table step.
- ``theta``, ``phi``, ``power``, ``stick`` and ``aileron`` are rounding only.
- ``gamma``, ``psi`` and ``lead.psi`` are worked out from the velocities read, so
  they carry no noise of their own, and their scatter is ``0.03 m/s / V``.
- Path and slot errors are computed from the readings behind them.
- Setpoints, slot geometry and the turn schedule stay exact.
- ``sensor_noise_scale = 0`` gives the stored state, whatever it holds.
- ``patrol_range_bearing`` reads range at 15 m and azimuth at 0.026 rad. Elevation
  has one altitude path with ``lead.z``, the 25 ft Mode S report, rounded and not
  noised, so it scatters by ``25 ft / sqrt(12) / range``: 0.0088 rad at 250 m.
- A trip stays true: a measured copy is charged the trip it is told about.

Figures are written out here, so a wrong table in the source fails.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg
from target_gym import registry
from target_gym.base import measured_state
from target_gym.formation.marl import FormationMARL
from target_gym.plane3d.env import (
    nearest_point_on_twisted_lemniscate,
    racetrack_guidance,
)
from target_gym.reward import tracking_cost

#: Each task compiles its readings in each worker, seconds apiece at 10 tasks.
pytestmark = pytest.mark.medium

FT, KT, DEG = 0.3048, 1852.0 / 3600.0, np.pi / 180.0

SIGMA = {
    "z": 0.2,
    "x_dot": 0.03,
    "y_dot": 0.03,
    "z_dot": 0.03,
    "theta_dot": 4.3e-6,
    "phi_dot": 4.3e-6,
}
STEP = {
    "z": FT,
    "x_dot": 0.125 * KT,
    "y_dot": 0.125 * KT,
    "z_dot": FT / 60.0,
    "theta": 0.010986 * DEG,
    "phi": 0.010986 * DEG,
    "theta_dot": 0.015625 * DEG,
    "phi_dot": 0.015625 * DEG,
    "power": 0.003,
    "stick": 1.6e-3,
    "aileron": 2.6e-3,
}
ROUNDING_ONLY = ("theta", "phi", "power", "stick", "aileron")


def _seat(offset, power):
    """Where one 3D aircraft's readings sit in the observation."""
    channels = dict(
        x_dot=0,
        y_dot=1,
        z=2,
        z_dot=3,
        theta=4,
        theta_dot=5,
        phi=6,
        phi_dot=7,
        gamma=8,
        psi=9,
        power=power,
        stick=power + 1,
        aileron=power + 2,
    )
    return {k: v + offset for k, v in channels.items()}


_PLANE = dict(x_dot=0, z=1, z_dot=2, theta=3, theta_dot=4, gamma=5, power=7, stick=8)
#: Each task's aircraft, as obs channels. The formation has 22 lead values,
#: then 26 per wingman.
SEATS = {
    "plane": [_PLANE],
    "plane_energy": [_PLANE],
    "plane_sine": [_PLANE],
    "plane3d_heading": [_seat(0, 12)],
    "plane3d_circle": [_seat(0, 14)],
    "plane3d_racetrack": [_seat(0, 15)],
    "plane3d_figure8": [_seat(0, 16)],
    "patrol": [_seat(0, 21)],
    "patrol_range_bearing": [_seat(0, 16)],
    "formation": [_seat(0, 12), _seat(22, 21), _seat(48, 21)],
}
#: Channels that must never move: setpoints, slot geometry, the turn rate and
#: the constant-zero targets.
EXACT = {
    "plane": (6, 9),
    "plane_energy": (6, 9),
    "plane_sine": (6, 9),
    "plane3d_heading": (10, 11),
    "plane3d_circle": (10, 13),
    "plane3d_racetrack": (10, 13, 14),
    "plane3d_figure8": (10, 11, 20),
    "patrol": (16, 17, 18, 25),
    "patrol_range_bearing": (13, 14, 15, 20),
    "formation": (10, 11, 15, 38, 39, 40, 47, 64, 65, 66, 73),
}
#: A relative quantity scatters as both ends' readings, sqrt(2) of one.
ROOT2 = float(np.sqrt(2.0))
PATROL_RELATIVE = {
    10: 0.27 * ROOT2,  # e_back
    11: 0.27 * ROOT2,  # e_right
    12: 0.2 * ROOT2,  # e_up
    15: 0.03 * ROOT2,  # rv_up
}
OTHER_SCATTER = {
    "plane3d_circle": {11: 0.27, 12: 0.27},
    "plane3d_racetrack": {11: 0.27, 12: 0.27},
    "patrol": PATROL_RELATIVE,
    "patrol_range_bearing": {10: 15.0, 11: 0.026},
    "formation": {
        16: 0.27 * ROOT2,  # the first wingman's error, in the lead's seat
        17: 0.27 * ROOT2,
        18: 0.2 * ROOT2,
        **{k + 22: v for k, v in PATROL_RELATIVE.items()},
    },
}
TASKS = list(SEATS)
every_task = pytest.mark.parametrize("name", TASKS)
N_DRAWS = 4000
#: The standard error of a scatter from 4000 draws is 1/sqrt(2 * 4000), 0.9%.
SCATTER_REL = 0.05


def _unrounded(env):
    env.readings = tuple(type(r)(**{**vars(r), "step": 0.0}) for r in env.readings)
    return env


@functools.cache
def _task(name):
    """``(env, params, a stepped state, the true observation)``."""
    spec = registry.get(name)
    env, params = spec.env_factory(), spec.make_test_params()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    space = env.action_space(params)
    action = jnp.zeros(space.shape, jnp.float32)
    state = env.step_env(key, state, action, params)[1]
    quiet = params.replace(sensor_noise_scale=0.0)
    return env, params, state, np.asarray(env.get_obs(state, quiet), np.float64)


def _read(env, state, params, n=N_DRAWS):
    """The observation of one plant state at ``n`` update indices, each a fresh
    draw, since noise comes from the episode key and the update."""
    obs = jax.vmap(lambda t: env.get_obs(state.replace(time=t), params))
    return np.asarray(jax.jit(obs)(jnp.arange(n)), np.float64)


@functools.cache
def _draws(name):
    """Rounded readings, with noise."""
    env, params, state, _ = _task(name)
    return _read(env, state, params)


@functools.cache
def _unrounded_draws(name):
    _, params, state, _ = _task(name)
    env = _unrounded(registry.get(name).env_factory())
    return _read(env, state, params)


def _speed(name, seat):
    env, params, state, true = _task(name)
    v = [true[seat[k]] for k in ("x_dot", "y_dot", "z_dot") if k in seat]
    return float(np.linalg.norm(v))


def _wrapped(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


# ---------------------------------------------------------------------------
# Noise reaches the expected channels
# ---------------------------------------------------------------------------


@every_task
def test_each_reading_scatters_by_its_figure(name):
    true, draws = _task(name)[3], _unrounded_draws(name)
    expected = {
        channel: SIGMA[field]
        for seat in SEATS[name]
        for field, channel in seat.items()
        if field in SIGMA
    }
    expected.update(OTHER_SCATTER.get(name, {}))
    for channel, sigma in expected.items():
        scatter = np.std(draws[:, channel] - true[channel])
        assert scatter == pytest.approx(sigma, rel=SCATTER_REL), f"channel {channel}"


@every_task
def test_each_reading_lands_on_its_step(name):
    draws = _draws(name)
    for seat in SEATS[name]:
        for field, channel in seat.items():
            if field in STEP:
                k = draws[:, channel] / STEP[field]
                # float32 observations, as a fraction of a step.
                tol = 2 * np.spacing(np.abs(draws[:, channel]).max().astype(np.float32))
                assert np.abs(k - np.round(k)).max() <= tol / STEP[field] + 1e-3


@every_task
def test_rounding_only_readings_have_no_scatter(name):
    true, draws = _task(name)[3], _draws(name)
    for seat in SEATS[name]:
        for field in ROUNDING_ONLY:
            if field in seat:
                column = draws[:, seat[field]]
                assert np.ptp(column) == 0.0, field
                assert np.abs(column[0] - true[seat[field]]) <= STEP[field] * 0.51


@every_task
def test_setpoints_stay_exact(name):
    true, draws = _task(name)[3], _draws(name)
    for channel in EXACT[name]:
        np.testing.assert_array_equal(
            draws[:, channel].astype(np.float32),
            np.full(N_DRAWS, true[channel], np.float32),
        )


# ---------------------------------------------------------------------------
# Derived channels inherit the readings behind them
# ---------------------------------------------------------------------------


@every_task
def test_gamma_and_psi_are_the_velocities_read(name):
    """On the same observation, so no separate draw can sit between them."""
    draws = _draws(name)[:500]
    for seat in SEATS[name]:
        vx, vz = draws[:, seat["x_dot"]], draws[:, seat["z_dot"]]
        if "y_dot" in seat:
            vy = draws[:, seat["y_dot"]]
            np.testing.assert_allclose(
                _wrapped(draws[:, seat["psi"]] - np.arctan2(vy, vx)), 0.0, atol=1e-6
            )
            vx = np.hypot(vx, vy)
        np.testing.assert_allclose(
            draws[:, seat["gamma"]], np.arctan2(vz, vx), atol=1e-6
        )


@every_task
def test_angle_scatter_is_velocity_scatter_over_speed(name):
    """1.3e-4 rad at 230 m/s, scaling as 0.03 / V."""
    true, draws = _task(name)[3], _unrounded_draws(name)
    for seat in SEATS[name]:
        for angle in ("gamma", "psi"):
            if angle in seat:
                scatter = np.std(_wrapped(draws[:, seat[angle]] - true[seat[angle]]))
                assert scatter == pytest.approx(0.03 / _speed(name, seat), rel=0.1)


def _measured(name, n=64):
    env, params, state, _ = _task(name)
    return (
        env,
        params,
        [measured_state(env, state.replace(time=t), params) for t in range(n)],
    )


@pytest.mark.parametrize("name", ["plane3d_circle", "plane3d_racetrack"])
def test_a_relative_position_is_the_position_read_less_the_centre(name):
    env, params, copies = _measured(name)
    for t, m in enumerate(copies):
        obs = _draws(name)[t]
        assert obs[11] == pytest.approx(float(m.x - m.target_x), abs=1e-2)
        assert obs[12] == pytest.approx(float(m.y - m.target_y), abs=1e-2)


def test_the_racetrack_guidance_inherits_the_position_read():
    env, params, copies = _measured("plane3d_racetrack", n=8)
    for t, m in enumerate(copies):
        obs = _draws("plane3d_racetrack")[t]
        np.testing.assert_allclose(
            obs[18:21], [float(v) for v in racetrack_guidance(m)], atol=1e-2
        )


def test_the_figure8_distance_inherits_the_position_read():
    env, params, copies = _measured("plane3d_figure8", n=8)
    for t, m in enumerate(copies):
        distance = nearest_point_on_twisted_lemniscate(m, params)[3]
        assert _draws("plane3d_figure8")[t][19] == pytest.approx(float(distance), 1e-5)


def test_the_slot_frame_turns_with_the_lead_velocities_read():
    """The lead's psi is its measured track: at 300 m a stored heading would
    sit 0.04 m away from these (``lead.psi`` in the refutation notes)."""
    env, params, copies = _measured("patrol")
    for t, m in enumerate(copies):
        obs = _draws("patrol")[t]
        f, lead = m.follower, m.lead
        psi = np.arctan2(float(lead.y_dot), float(lead.x_dot))
        dx, dy = float(f.x) - float(lead.x), float(f.y) - float(lead.y)
        back = -(dx * np.cos(psi) + dy * np.sin(psi))
        right = dx * np.sin(psi) - dy * np.cos(psi)
        assert obs[10] == pytest.approx(back - float(m.slot_back), abs=3e-3)
        assert obs[11] == pytest.approx(right - float(m.slot_right), abs=3e-3)
        up = float(f.z) - float(lead.z) - float(m.slot_up)
        assert obs[12] == pytest.approx(up, abs=2e-3)
        own = np.arctan2(float(f.y_dot), float(f.x_dot))
        assert _wrapped(obs[19] - (psi - own)) == pytest.approx(0.0, abs=1e-6)
        speed = np.linalg.norm(
            [float(lead.x_dot), float(lead.y_dot), float(lead.z_dot)]
        )
        assert obs[20] == pytest.approx(speed, abs=1e-4)


def test_the_lead_seat_and_the_wingman_read_one_slot_error():
    """One lead fix goes to every seat."""
    draws = _draws("formation")[:500]
    np.testing.assert_allclose(draws[:, 16:19], draws[:, 32:35], atol=1e-3)
    np.testing.assert_allclose(draws[:, 19:22], draws[:, 58:61], atol=1e-3)


# ---------------------------------------------------------------------------
# Scale 0
# ---------------------------------------------------------------------------


def _set_stored_angles(state, gamma, psi):
    """``state`` with the angles stored apart from what its velocities give."""

    def on(aircraft):
        fields = dict(gamma=gamma)
        if hasattr(aircraft, "psi"):
            fields["psi"] = psi
        return aircraft.replace(**fields)

    if hasattr(state, "follower"):
        return state.replace(follower=on(state.follower))
    if hasattr(state, "wingmen"):
        return state.replace(lead=on(state.lead))
    return on(state)


@every_task
def test_at_scale_zero_the_angles_are_the_stored_ones(name):
    env, params, state, _ = _task(name)
    seat = SEATS[name][0]
    moved = _set_stored_angles(state, 0.25, 1.5)
    quiet = np.asarray(env.get_obs(moved, params.replace(sensor_noise_scale=0.0)))
    assert quiet[seat["gamma"]] == pytest.approx(0.25, abs=1e-6)
    noisy = np.asarray(env.get_obs(moved, params))
    assert noisy[seat["gamma"]] != pytest.approx(0.25, abs=1e-2)
    if "psi" in seat:
        assert quiet[seat["psi"]] == pytest.approx(1.5, abs=1e-6)


# ---------------------------------------------------------------------------
# patrol_range_bearing
# ---------------------------------------------------------------------------

BO = "patrol_range_bearing"
RANGE, AZIMUTH, ELEVATION, TRACKED_RANGE = 10, 11, 12, 19


def test_range_is_one_reading_in_both_its_channels():
    draws = _draws(BO)
    np.testing.assert_array_equal(draws[:, RANGE], draws[:, TRACKED_RANGE])


def test_range_and_azimuth_inherit_no_position_noise():
    """Their whole scatter is the channel's, drawn apart from the fixes: the
    range is uncorrelated with the slot's own altitude reading."""
    draws = _unrounded_draws(BO)
    np.testing.assert_allclose(
        np.corrcoef(draws[:, RANGE], draws[:, 2])[0, 1], 0.0, atol=0.08
    )


def _at_range(metres):
    """Level with the lead, ``metres`` straight behind, in a state whose lead
    altitude sweeps one 25 ft coding step."""
    env, params, state, _ = _task(BO)
    psi = state.lead.psi
    lead = state.lead.replace(z=7000.0)
    follower = state.follower.replace(
        x=lead.x - metres * jnp.cos(psi),
        y=lead.y - metres * jnp.sin(psi),
        z=7000.0,
    )
    return env, params, state.replace(lead=lead, follower=follower)


@pytest.mark.parametrize("metres", [250.0, 500.0])
def test_elevation_scatters_as_the_altitude_coding_over_range(metres):
    """25 ft / sqrt(12) = 2.2 m, so 0.0088 rad at 250 m and half that at 500.
    The follower's own altimeter adds 0.2 m, 0.4% in quadrature."""
    env, params, state = _at_range(metres)
    code = 25 * FT
    offsets = (jnp.arange(2000) + 0.5) / 2000 * code

    def read(offset):
        s = state.replace(lead=state.lead.replace(z=state.lead.z + offset))
        return (
            env.get_obs(s, params)[ELEVATION],
            env.get_obs(s, params.replace(sensor_noise_scale=0.0))[ELEVATION],
        )

    read_, true = jax.jit(jax.vmap(read))(offsets)
    rms = np.sqrt(np.mean((np.asarray(read_) - np.asarray(true)) ** 2))
    assert rms == pytest.approx(0.0088 * 250.0 / metres, rel=0.06)


def test_elevation_agrees_with_the_altitudes_read():
    """One altitude path: what elevation says of the lead is what ``lead.z``
    reads, the 25 ft Mode S report, in the measured reward as well."""
    env, params, copies = _measured(BO)
    for t, m in enumerate(copies):
        obs = _draws(BO)[t]
        dx, dy = float(m.lead.x - m.follower.x), float(m.lead.y - m.follower.y)
        dz = float(m.lead.z) - float(m.follower.z)
        assert obs[ELEVATION] == pytest.approx(
            np.arctan2(dz, np.hypot(dx, dy)), abs=1e-5
        )
        k = float(m.lead.z) / (25 * FT)
        assert abs(k - round(k)) < 1e-3


# ---------------------------------------------------------------------------
# Reward and trips
# ---------------------------------------------------------------------------


def test_the_measured_cost_is_the_cost_of_the_altitude_and_track_read():
    """The heading task's cost, worked from the observation the policy got."""
    env, params = tg.make("plane3d_heading")
    key = jax.random.PRNGKey(1)
    _, state = env.reset_env(key, params)
    obs, new, reward, _, info = env.step_env(key, state, jnp.zeros(3), params)
    altitude = tracking_cost(
        obs[10] - obs[2],
        params.e_floor_altitude,
        params.e_tol_altitude,
        params.tracking_exponent,
    )
    heading = tracking_cost(
        jnp.arctan2(jnp.sin(obs[9] - obs[11]), jnp.cos(obs[9] - obs[11])),
        params.e_floor_heading,
        0.0,
        params.tracking_exponent,
    )
    assert float(info["measured_reward"]) == pytest.approx(
        -float(altitude + heading), rel=1e-4
    )
    assert float(reward) != pytest.approx(float(info["measured_reward"]), rel=1e-5)


def _out_of_the_envelope(state):
    """``state`` with an aircraft below the floor."""
    if hasattr(state, "follower"):
        return state.replace(follower=state.follower.replace(z=-100.0))
    if hasattr(state, "wingmen"):
        return state.replace(lead=state.lead.replace(z=-100.0))
    return state.replace(z=-100.0)


@every_task
def test_a_measured_copy_is_charged_the_trip_it_is_given(name):
    env, params, state, _ = _task(name)
    crashed = _out_of_the_envelope(state)
    charged = env.reward_terms(crashed, params)
    told_not = env.reward_terms(crashed, params, tripped=jnp.array(False))
    told_yes = env.reward_terms(state, params, tripped=jnp.array(True))
    assert float(charged["failure"]) > 0.0
    assert float(told_not["failure"]) == 0.0
    assert float(told_yes["failure"]) == pytest.approx(float(charged["failure"]))


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def test_every_formation_size_has_an_actuator_of_its_own_size():
    from target_gym import Formation

    env = Formation(num_wingmen=4)
    params = env.default_params.replace(actuator_dead_band=0.1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    assert state.actuator_held.shape == (15,)
    # Dead band 10% of a span of 2: f_D = 0.1 (``test_actuators.py``).
    new = env.step_env(key, state, jnp.full(15, 0.5), params)[1]
    np.testing.assert_allclose(new.actuator_held, 0.4, atol=1e-6)


def test_the_multi_agent_view_reads_at_the_params_it_is_given():
    marl = FormationMARL()
    params = marl.default_params.replace(sensor_noise_scale=0.0)
    _, state = marl.reset(jax.random.PRNGKey(0), params)
    obs = marl.get_obs(state, params)
    assert float(obs["lead"][2]) == pytest.approx(float(state.lead.z))


@pytest.mark.large
@pytest.mark.xfail(strict=True, reason="PR 9: the 20 s filter loses a 0.03 rad/s lead")
@pytest.mark.parametrize("seed", range(100, 104))
def test_the_range_bearing_expert_survives_its_range_noise(seed):
    """Differenced raw, 15 m of range noise is 21 m/s of lead speed and the
    expert trips on every training seed. This is the filter's floor, not the
    8.5 m hold of tests/patrol/test_formation_hold.py."""
    env, params = tg.make(BO)
    ctrl = registry.get(BO).make_pid_controller()
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)

    def body(carry, t):
        obs, state, memory = carry
        action, memory = ctrl.step(ctrl.params, memory, obs, jnp)
        obs, state, _, _, info = env.step_env(
            jax.random.fold_in(key, t), state, action, params
        )
        return (obs, state, memory), info["tripped"]

    steps = jnp.arange(int(params.max_steps_in_episode))
    _, tripped = jax.lax.scan(body, (obs, state, ctrl.init_state), steps)
    assert not np.any(tripped)
