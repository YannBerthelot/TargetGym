"""The opt-in actuator stage, against its published models.

Contract:

- ``actuator_dead_band``, ``actuator_stiction`` and ``action_noise`` default to
  0, and at 0 the stage changes nothing (docs/proposals/measurement-noise.md).
- Values are fractions of the action span, the unit of the proposal's table.
- Stiction is He, Wang, Pottmann and Qin (2007), as restated by Bacci di Capaci
  et al. (ADCHEM 2018, eq. 4): with ``e = u[k] - y[k-1]``, ``y[k] = u[k] - f_D``
  if ``e > f_S``, ``u[k] + f_D`` if ``e < -f_S``, else ``y[k-1]``.
- ``actuator_dead_band`` D and ``actuator_stiction`` S set the two bands:
  ``f_D = D / 2`` and ``f_S = D / 2 + S``. Choudhury, Thornhill and Shah's
  (ADCHEM 2003) deadband plus stickband is then ``f_S + f_D = D + S`` and
  their slip jump ``f_S - f_D = S``.
- A command at the end of its range reaches it (Choudhury, Fig. 4).
- ``action_noise`` is the output's scatter, drawn from the state alone.

Examples below use the action range [-1, 1], so a span of 2.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import eval as E
from target_gym.base import EnvParams, actuate
from target_gym.pc_gym.four_tank.env import FourTankParams
from target_gym.pc_gym.four_tank.env_jax import FourTank
from target_gym.registry import REGISTRY


def _declares_readings(spec):
    return bool(getattr(spec.make_env(), "readings", ()))


NOISY = [s for s in REGISTRY.values() if _declares_readings(s)]


def _state(env, params):
    return env.reset_env(jax.random.PRNGKey(0), params)[1]


def _track(params, commands):
    """The output for one command sequence on the first of the two pumps."""
    env = FourTank()
    state = _state(env, params)
    out = []
    for u in commands:
        applied, state = actuate(env, state, jnp.array([u, 0.0]), params)
        out.append(float(applied[0]))
    return out


def test_defects_default_to_off():
    params = EnvParams()
    assert params.actuator_dead_band == 0.0
    assert params.actuator_stiction == 0.0
    assert params.action_noise == 0.0


def test_at_zero_the_stage_changes_no_action_and_records_it():
    """Every action, in range or past it: the plant's own clip decides, and
    ``actuator_held`` keeps what reaches it, for ``info["applied_action"]``."""
    env, params = FourTank(), FourTankParams()
    state = _state(env, params)
    for action in ([0.25, -0.75], [1.0, -1.0], [3.0, -7.0]):
        action = jnp.array(action)
        applied, new = actuate(env, state, action, params)
        np.testing.assert_array_equal(applied, action)
        np.testing.assert_array_equal(new.actuator_held, jnp.clip(action, -1, 1))
        rest = new.replace(actuator_held=state.actuator_held)
        jax.tree_util.tree_map(np.testing.assert_array_equal, rest, state)


def test_a_dead_band_holds_inside_its_width_and_follows_outside():
    """Width 10% of span, 0.2 raw, so f_D = f_S = 0.1. The output lags a ramp
    by 0.1 and holds for a reversal of up to 0.2.

    0.5: e = 0.5 > 0.1, y = 0.4. 0.45 and 0.35: |e| <= 0.1, holds.
    0.25: e = -0.15, y = 0.35. 0.4: e = 0.05, holds. 0.5: e = 0.15, y = 0.4.
    """
    params = FourTankParams(actuator_dead_band=0.1)
    out = _track(params, [0.5, 0.45, 0.35, 0.25, 0.4, 0.5])
    np.testing.assert_allclose(out, [0.4, 0.4, 0.4, 0.35, 0.35, 0.4], atol=1e-6)


def test_stiction_follows_he_models_three_branches():
    """f_S = 0.10 and f_D = 0.04, so D = 0.04 and S = 0.03 of the span of 2.

    0.5: e = 0.5, y = 0.46. 0.54: e = 0.08, holds. 0.57: e = 0.11, y = 0.53.
    0.45: e = -0.08, holds. 0.4: e = -0.13, y = 0.44.
    """
    params = FourTankParams(actuator_dead_band=0.04, actuator_stiction=0.03)
    out = _track(params, [0.5, 0.54, 0.57, 0.45, 0.4])
    np.testing.assert_allclose(out, [0.46, 0.46, 0.53, 0.53, 0.44], atol=1e-6)


def test_a_two_percent_valve_moves_after_two_percent_of_travel():
    """Choudhury's reading of the figure: "a 2 % stiction means that when the
    valve gets stuck it will start moving only after the cumulative change of
    its control signal is greater than or equal to 2%". With no dead band the
    slip lands on the command, their no-offset case. 2% of the span is 0.04."""
    params = FourTankParams(actuator_stiction=0.02)
    out = _track(params, [0.5, 0.53, 0.539, 0.541])
    np.testing.assert_allclose(out, [0.5, 0.5, 0.5, 0.541], atol=1e-6)


def test_a_noisy_command_cannot_walk_away_from_a_stuck_valve():
    """The reason for He's model over Choudhury's flow chart here. Choudhury
    re-sets the stuck point at every reversal of the command, so a command that
    zigzags as it climbs never accumulates a run past the band. The force on a
    real valve follows the command's distance from the stem."""
    params = FourTankParams(actuator_stiction=0.02)
    commands = [0.5]
    for _ in range(12):
        commands += [commands[-1] + 0.02, commands[-1] + 0.01]
    out = _track(params, commands)
    assert commands[-1] > 0.6
    assert commands[-1] - out[-1] <= 0.04, "the valve fell behind its stick band"


def test_a_command_at_the_end_of_its_range_reaches_it():
    """Choudhury's saturation branches: past either limit the valve is at it,
    whatever the bands."""
    params = FourTankParams(actuator_dead_band=0.1, actuator_stiction=0.1)
    out = _track(params, [1.0, 2.0, -3.0, -1.0])
    np.testing.assert_allclose(out, [1.0, 1.0, -1.0, -1.0], atol=1e-6)


def test_each_action_keeps_its_own_held_output():
    env = FourTank()
    params = FourTankParams(actuator_dead_band=0.1)
    state = _state(env, params)
    _, state = actuate(env, state, jnp.array([0.5, -0.5]), params)
    applied, _ = actuate(env, state, jnp.array([0.45, -0.1]), params)
    np.testing.assert_allclose(applied, [0.4, -0.2], atol=1e-6)


def test_action_noise_scatters_the_output_by_its_fraction_of_span():
    """0.15% of a span of 2 is 0.003. 2000 draws put the sample standard
    deviation within 4 / sqrt(2 * 2000), 9%, and the mean within 4 standard
    errors of the command."""
    env = FourTank()
    params = FourTankParams(action_noise=0.0015)
    state = _state(env, params)

    def at(t):
        return actuate(env, state.replace(time=t), jnp.array([0.2, 0.2]), params)[0]

    out = np.asarray(jax.vmap(at)(jnp.arange(2000)))
    assert out.std() == pytest.approx(0.003, rel=0.09)
    assert abs(out.mean() - 0.2) < 4 * 0.003 / np.sqrt(4000)
    assert abs(np.corrcoef(out[:, 0], out[:, 1])[0, 1]) < 0.1


def test_action_noise_does_not_build_up_in_the_held_output():
    """Positioner repeatability scatters each move; it does not move the
    stem's memory."""
    env = FourTank()
    params = FourTankParams(actuator_dead_band=0.1, action_noise=0.01)
    state = _state(env, params)
    state = state.replace(time=jnp.int32(5))
    _, state = actuate(env, state, jnp.array([0.5, 0.5]), params)
    np.testing.assert_allclose(state.actuator_held, [0.4, 0.4], atol=1e-6)


def test_action_noise_stays_inside_the_range():
    env = FourTank()
    params = FourTankParams(action_noise=0.5)
    state = _state(env, params)

    def at(t):
        return actuate(env, state.replace(time=t), jnp.ones(2), params)[0]

    out = np.asarray(jax.vmap(at)(jnp.arange(200)))
    assert out.max() <= 1.0 and out.min() >= -1.0


def test_a_small_command_does_not_reach_the_plant_through_a_stuck_valve():
    """End to end on the pilot: the applied voltage stops moving, under ``jit``
    and ``lax.scan``."""
    env = FourTank()
    params = FourTankParams(actuator_dead_band=0.1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    commands = jnp.array([[0.5, 0.5], [0.45, 0.45], [0.35, 0.35], [0.4, 0.4]])

    def step(s, action):
        _, s, *_ = env.step_env(key, s, action, params)
        return s, s.v1

    _, volts = jax.jit(lambda s: jax.lax.scan(step, s, commands))(state)
    # y = 0.4 throughout, and raw 0.4 is 5 * (1 + 0.4) = 7 V.
    np.testing.assert_allclose(volts, [7.0, 7.0, 7.0, 7.0], atol=1e-5)


def test_a_restart_returns_the_actuator_to_rest():
    """A trip is a cold restart: the memory goes with the state."""
    env = FourTank()
    params = FourTankParams(actuator_dead_band=0.1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    tripped = False
    for t in range(60):
        _, state, _, _, info = env.step_env(
            jax.random.fold_in(key, t), state, jnp.array([-1.0, -1.0]), params
        )
        if bool(info["tripped"]):
            tripped = True
            break
    assert tripped, "pumps off did not trip within 60 steps"
    np.testing.assert_array_equal(state.actuator_held, jnp.zeros(2))


def test_rate_and_lag_default_to_off():
    assert EnvParams().actuator_rate == 0.0
    assert EnvParams().actuator_lag == 0.0


def test_a_rate_limit_moves_the_output_at_most_its_rate():
    """0.1 of the span of 2 per second, at 1 s steps, is 0.2 per step, all the
    way to the end of the range."""
    params = FourTankParams(actuator_rate=0.1)
    assert _track(params, [1.0] * 6) == pytest.approx(
        [0.2, 0.4, 0.6, 0.8, 1.0, 1.0], abs=1e-6
    )


def test_a_lag_closes_a_fraction_of_the_gap_each_step():
    """``1 - exp(-dt / tau)`` of the gap: with ``tau = 1 / ln 2`` at 1 s steps
    that is half."""
    params = FourTankParams(actuator_lag=1.0 / np.log(2.0))
    assert _track(params, [1.0] * 3) == pytest.approx([0.5, 0.75, 0.875], abs=1e-6)


def test_a_rate_limit_caps_what_a_lag_would_move():
    params = FourTankParams(actuator_lag=1.0 / np.log(2.0), actuator_rate=0.1)
    assert _track(params, [1.0] * 2) == pytest.approx([0.2, 0.4], abs=1e-6)


def test_a_rate_limit_bounds_a_step_of_a_task_with_sub_steps():
    """``reactor`` runs 10 s per step, so 0.01 of the span of 2 per second moves
    0.2 per step, not 0.02."""
    spec = REGISTRY["reactor"]
    env = spec.make_env()
    params = spec.make_test_params().replace(actuator_rate=0.01)
    state = _state(env, params)
    applied, _ = actuate(env, state, jnp.array([1.0]), params)
    assert float(applied[0]) == pytest.approx(0.2, abs=1e-6)


@pytest.mark.parametrize("spec", REGISTRY.values(), ids=lambda s: s.name)
def test_a_step_reports_what_the_actuator_applied(spec):
    """Past the box a command does nothing more, so the report is the box."""
    env, params = spec.make_env(), spec.make_test_params()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    shape = env.action_space(params).shape
    info = env.step_env(key, state, jnp.full(shape, 3.0), params)[-1]
    high = env.action_space(params).high
    np.testing.assert_allclose(
        info["applied_action"], np.broadcast_to(high, shape), rtol=1e-6
    )


def test_a_step_reports_the_limited_output_not_the_command():
    env = FourTank()
    params = FourTankParams(actuator_rate=0.1)
    key = jax.random.PRNGKey(0)
    state = _state(env, params)
    info = env.step_env(key, state, jnp.array([1.0, -1.0]), params)[-1]
    np.testing.assert_allclose(info["applied_action"], [0.2, -0.2], rtol=1e-6)


def test_a_step_reports_the_same_scatter_the_plant_got():
    """``action_noise`` is drawn per step, so a report read a step off would
    show another draw than the one that moved the plant."""
    env = FourTank()
    params = FourTankParams(action_noise=0.01)
    key = jax.random.PRNGKey(0)
    state = _state(env, params)
    action = jnp.array([0.2, -0.3])
    applied, _ = actuate(env, state, action, params)
    info = env.step_env(key, state, action, params)[-1]
    np.testing.assert_allclose(info["applied_action"], applied, rtol=1e-6)


def test_the_column_starts_with_its_valves_where_the_plant_is():
    """With a rate limit the first step cannot be a jump from the middle of the
    range to the plant's own flows."""
    spec = REGISTRY["distillation"]
    env, p = spec.make_env(), spec.make_test_params()
    state = _state(env, p)
    span = np.array([p.L_max - p.L_min, p.V_max - p.V_min])
    want = 2.0 * (np.array([state.L, state.V]) - [p.L_min, p.V_min]) / span - 1.0
    np.testing.assert_allclose(state.actuator_held, want, rtol=1e-6)


class _BangBang:
    def __init__(self):
        self.sign = 1.0

    def __call__(self, obs):
        self.sign = -self.sign
        return np.array([self.sign])


def test_an_episodes_travel_is_what_the_actuator_did():
    """0.2 of the span per second at 0.05 s steps is 0.01 strokes per step,
    over the 149 moves of ``first_order``'s 150 steps."""
    spec = REGISTRY["first_order"]
    params = spec.make_test_params().replace(actuator_rate=0.2)
    ep = E.run_episode(spec, params, _BangBang(), seed=0)
    assert E.travel(ep.actions) == pytest.approx(1.49, rel=1e-4)


@pytest.mark.parametrize("spec", NOISY, ids=lambda s: s.name)
def test_every_task_with_readings_runs_the_actuator_stage(spec):
    """A task wired for readings must not ignore a dead-band request: the field
    would look set and do nothing. A command at the top of the range lands on
    it, so this reads the stage's own memory. The plant's state would not show
    it: ``compressor_surge``'s rate-limited actuators move at their limit from
    the reset with or without the stage."""
    env, params = spec.make_env(), spec.make_test_params()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    space = env.action_space(params)
    action = jnp.broadcast_to(space.high, space.shape).astype(jnp.float32)
    params = params.replace(actuator_dead_band=0.01)
    new = env.step_env(key, state, action, params)[1]
    np.testing.assert_array_equal(new.actuator_held, action)
