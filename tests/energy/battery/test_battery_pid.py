"""The battery expert: dispatch feedforward, an inner PI, state-of-charge limits.

Contract, from grid-battery practice (a scheduled setpoint with SoC limits and
an inner PI) and the controller's documented design (not its code):

* The command is the signal, so on target with fresh memory it is
  ``target / power_max`` (positive discharges). On a block's last step it is the
  next block, which the market publishes ahead.
* A PI trims the losses: delivering less than the target commands more.
* Near either SoC limit, 5 % inside the trips (MESA-1 report), the demand fades
  linearly to zero at the limit over the guard margin (0.12 of SoC), only in the
  direction that would breach it.
"""

import jax
import numpy as np
import pytest

import target_gym as tg
from target_gym.registry import REGISTRY


def _command(soc=0.5, delivered=0.4, target=0.4, next_block=None, minutes=30.0):
    expert = tg.make_expert("battery").host()
    expert.reset()
    next_block = target if next_block is None else next_block
    obs = np.array([soc, 3.7, 25.0, delivered, target, next_block, minutes], np.float32)
    return float(np.asarray(expert(obs)).ravel()[0])


def test_on_target_the_command_is_the_dispatch():
    assert _command(target=0.4, delivered=0.4) == pytest.approx(0.4, abs=1e-6)


def test_a_shortfall_commands_more_than_the_dispatch():
    assert _command(target=0.4, delivered=0.35) > 0.4


def test_the_last_step_of_a_block_commands_the_next_block():
    """5 s out, the next step is scored against the next block."""
    assert _command(next_block=-0.2, minutes=5.0 / 60.0) == pytest.approx(
        -0.2, abs=1e-5
    )


def test_earlier_in_a_block_the_next_one_changes_nothing():
    assert _command(next_block=-0.2, minutes=10.0 / 60.0) == pytest.approx(
        0.4, abs=1e-5
    )


def test_the_action_has_a_batch_and_an_action_axis():
    """``(n, 1)``, as every other expert: it was ``(n,)``."""
    controller = REGISTRY["battery"].make_pid_controller()
    obs = np.tile(np.array([0.5, 3.7, 25.0, 0.4, 0.4, 0.4, 30.0], np.float32), (3, 1))
    memory = jax.tree.map(lambda x: np.full((3,), x), controller.init_state)
    action, _ = controller.step(controller.params, memory, obs)
    assert np.asarray(action).shape == (3, 1)


def test_the_integral_stops_while_the_guard_sheds():
    """Shedding leaves a standing error, and integrating it would unwind the guard."""
    controller = REGISTRY["battery"].make_pid_controller()
    obs = np.array([0.16, 3.7, 25.0, 0.1, 0.4, 0.4, 30.0], np.float32)
    _, memory = controller.step(controller.params, controller.init_state, obs)
    assert float(memory.integral) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "soc, target, expected",
    [
        (0.10 + 0.06, 0.4, 0.2),  # half the margin above the floor: half the discharge
        (0.10 + 0.06, -0.4, -0.4),  # charging near the floor is not faded
        (0.90 - 0.03, -0.4, -0.1),  # a quarter of the margin below the ceiling
        (0.90 - 0.03, 0.4, 0.4),  # discharging near the ceiling is not faded
    ],
)
def test_the_soc_guard_fades_only_the_breaching_direction(soc, target, expected):
    assert _command(soc=soc, delivered=target, target=target) == pytest.approx(
        expected, abs=1e-5
    )


HELD_OUT = range(200, 210)


@pytest.mark.large
def test_it_never_trips_on_held_out_seeds():
    spec = REGISTRY["battery"]
    env, p = spec.make_env(), spec.make_test_params()
    step = jax.jit(env.step_env)
    for seed in HELD_OUT:
        expert = tg.make_expert("battery").host()
        expert.reset()
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset_env(key, p)
        for _ in range(int(p.max_steps_in_episode)):
            key, k = jax.random.split(key)
            obs, state, _, _, info = step(k, state, expert(np.asarray(obs)), p)
            assert not bool(info["tripped"]), f"seed {seed} tripped"
