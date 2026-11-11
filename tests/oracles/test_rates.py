"""Planning through ``base.actuate``'s rate limit, ``oracles.rates``.

Contract, from docs/proposals/oracle-smoothing-protocol.md (Amendment 3) and
the actuator stage's docstring (``base.actuate``): a move v in [-1, 1] is the
actuator's rate as a fraction of its limit. With ``rate = RateLimit.of(env, p)``:

- ``rate.command(held, v)`` is the raw action that moves the actuator by exactly
  ``v`` times its limit in one step: no clip of ``actuate`` binds on it, so the
  plan's slope never dies at the limit;
- ``rate.reached(held, v)`` is where that leaves the actuator;
- ``rate.rows(held, v, steps) >= 0`` says the command stays in the action box on
  every step of a ramp held for ``steps`` steps;
- every actuator path the env allows is such a plan, so at a hold of one step the
  moves lose nothing;
- a plant with a dead band or stiction has no such move, and raises.

Expected travel by hand from the task's figures. boiler_drum, 2 s steps: fuel 0.007 of span/s is 0.028, its 20 s lag closes
1 - exp(-0.1) = 0.0951626 of a gap; feedwater 0.05 is 0.2, its 1 s lag closes
1 - exp(-2) = 0.8646647.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.base import actuate
from target_gym.oracles.controller import enable_x64, float64
from target_gym.oracles.rates import RateLimit
from target_gym.registry import REGISTRY

FIGURES = {
    "boiler_drum": ([0.028, 0.2], [0.0951626, 0.8646647]),
}


def _plant(name):
    spec = REGISTRY[name]
    env, params = spec.make_env(), float64(spec.make_test_params())
    return env, params, RateLimit.of(env, params)


@dataclasses.dataclass
class _Held:
    """The fields of a state ``actuate`` reads."""

    actuator_held: jnp.ndarray
    sensor_key: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.zeros(2, jnp.uint32)
    )
    time: int = 0

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)


def _actuate(env, params, held, action):
    return actuate(env, _Held(jnp.asarray(held)), jnp.asarray(action), params)[0]


@pytest.mark.parametrize("name", FIGURES)
def test_a_full_rate_move_covers_the_travel_of_the_tasks_own_figures(name):
    travel, closes = FIGURES[name]
    with enable_x64():
        _, _, rate = _plant(name)
        np.testing.assert_allclose(rate.travel, travel, rtol=1e-6)
        np.testing.assert_allclose(rate.closes, closes, rtol=1e-6)


@pytest.mark.parametrize("name", FIGURES)
@pytest.mark.parametrize("v", [1.0, -1.0, 0.37, -0.02, 0.0])
def test_the_command_moves_the_actuator_by_exactly_the_planned_rate(name, v):
    """From the middle of the box, inside it on every path below: 1e-12 is far
    over float64 rounding and far under any clip's effect."""
    with enable_x64():
        env, params, rate = _plant(name)
        held = jnp.zeros(rate.travel.shape)
        moves = jnp.full(rate.travel.shape, v)
        applied = _actuate(env, params, held, rate.command(held, moves))
        np.testing.assert_allclose(applied, rate.travel * v, atol=1e-12)
        np.testing.assert_allclose(rate.reached(held, moves), applied, atol=1e-12)


@pytest.mark.parametrize("name", FIGURES)
def test_the_slope_of_a_full_rate_move_stays_alive_past_where_a_command_dies(name):
    """d reached / d v is the travel at v = 1, where a command past the limit
    has none: the gap a planner on commands could not see through."""
    import jax

    with enable_x64():
        _, _, rate = _plant(name)
        held = jnp.zeros(rate.travel.shape)
        slope = jax.jacfwd(lambda v: rate.reached(held, v))(jnp.ones(held.shape))
        np.testing.assert_allclose(jnp.diag(slope), rate.travel, rtol=1e-12)


@pytest.mark.parametrize("name", FIGURES)
@pytest.mark.parametrize("steps", [1, 3, 30])
def test_the_rows_hold_exactly_while_every_command_of_the_ramp_is_in_the_box(
    name, steps
):
    """A ramp from a start near the top: its largest command is the last
    step's, position plus the gap the lag needs. The rows' sign must agree with
    a walk along the ramp, on both sides of the box's edge."""
    with enable_x64():
        _, _, rate = _plant(name)
        rng = np.random.default_rng(0)
        for _ in range(50):
            held = jnp.asarray(rng.uniform(-1.0, 1.0, rate.travel.shape))
            moves = jnp.asarray(rng.uniform(-1.0, 1.0, rate.travel.shape))
            inside, a = True, held
            for _ in range(steps):
                c = rate.command(a, moves)
                inside &= bool(jnp.all((c >= -1.0) & (c <= 1.0)))
                a = rate.reached(a, moves)
            assert bool(jnp.all(rate.rows(held, moves, steps) >= 0.0)) == inside


@pytest.mark.parametrize("name", FIGURES)
def test_any_command_has_the_move_that_lands_where_the_env_lands(name):
    with enable_x64():
        env, params, rate = _plant(name)
        rng = np.random.default_rng(2)
        for _ in range(50):
            held = jnp.asarray(rng.uniform(-1.0, 1.0, rate.travel.shape))
            command = jnp.asarray(rng.uniform(-1.0, 1.0, rate.travel.shape))
            landed = rate.reached(held, rate.move(held, command))
            np.testing.assert_allclose(
                landed, _actuate(env, params, held, command), atol=1e-12
            )


@pytest.mark.parametrize("name", FIGURES)
def test_every_path_the_actuator_can_take_in_a_step_is_a_rate_plan(name):
    """Any command in the box, from any position in it: the move that lands
    where the env lands is in [-1, 1] and its rows hold, so a hold of one step
    forbids nothing the env allows."""
    with enable_x64():
        env, params, rate = _plant(name)
        rng = np.random.default_rng(1)
        for _ in range(200):
            held = jnp.asarray(rng.uniform(-1.0, 1.0, rate.travel.shape))
            command = jnp.asarray(rng.uniform(-1.0, 1.0, rate.travel.shape))
            landed = _actuate(env, params, held, command)
            moves = (landed - held) / rate.travel
            assert bool(jnp.all(jnp.abs(moves) <= 1.0 + 1e-12))
            assert bool(jnp.all(rate.rows(held, moves, 1) >= -1e-12))


def test_a_plant_with_a_dead_band_has_no_rate_plan():
    spec = REGISTRY["boiler_drum"]
    params = spec.make_test_params().replace(actuator_dead_band=0.01)
    with pytest.raises(ValueError, match="dead band"):
        RateLimit.of(spec.make_env(), params)


def test_a_plant_with_no_rate_limit_has_no_rate_plan():
    spec = REGISTRY["battery"]
    with pytest.raises(ValueError, match="no rate limit"):
        RateLimit.of(spec.make_env(), spec.make_test_params())
