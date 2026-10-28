"""The battery expert: dispatch feedforward, an inner PI, state-of-charge limits.

Contract, from grid-battery practice (a scheduled setpoint with SoC limits and
an inner PI) and the controller's documented design (not its code):

* The command is the dispatch target, so on target with fresh memory it is
  ``target / power_max`` (positive discharges).
* A PI trims the losses: delivering less than the target commands more.
* Near either SoC limit the demand fades linearly to zero at the limit over
  the guard margin (0.12 of SoC), only in the direction that would breach it.
"""

import jax
import numpy as np
import pytest

import target_gym as tg
from target_gym.registry import REGISTRY


def _command(soc=0.5, delivered=0.4, target=0.4):
    expert = tg.make_expert("battery").host()
    expert.reset()
    obs = np.array([soc, 3.7, 25.0, delivered, target], np.float32)
    return float(np.asarray(expert(obs)).ravel()[0])


def test_on_target_the_command_is_the_dispatch():
    assert _command(target=0.4, delivered=0.4) == pytest.approx(0.4, abs=1e-6)


def test_a_shortfall_commands_more_than_the_dispatch():
    assert _command(target=0.4, delivered=0.35) > 0.4


@pytest.mark.parametrize(
    "soc, target, expected",
    [
        (0.05 + 0.06, 0.4, 0.2),  # half the margin above the floor: half the discharge
        (0.05 + 0.06, -0.4, -0.4),  # charging near the floor is not faded
        (0.95 - 0.03, -0.4, -0.1),  # a quarter of the margin below the ceiling
        (0.95 - 0.03, 0.4, 0.4),  # discharging near the ceiling is not faded
    ],
)
def test_the_soc_guard_fades_only_the_breaching_direction(soc, target, expected):
    assert _command(soc=soc, delivered=target, target=target) == pytest.approx(
        expected, abs=1e-5
    )


HELD_OUT = range(200, 210)


@pytest.mark.slow
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
