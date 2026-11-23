"""The tower's fatigue count: rainflow, charged step by step.

Contract (owner, 2026-11-23: the tower is counted by rainflow), from ASTM E1049-85
sec. 5.4.4, the three-point rainflow rule, not from the code:

* Reduce the record of the base moment to its turning points, the first and the
  last point included. Read them in order; with X the latest range and Y the one
  before it, whenever X >= Y: if Y contains the record's starting point, count Y
  as one half cycle and discard the starting point; otherwise count Y as one
  cycle and discard both its points. At the end, each range left is a half cycle.
* A cycle of range r (MN m) is r^m of damage and a half cycle 0.5 r^m, m the
  Woehler exponent (4 for the tower, PHYSICS.md).
* The plant charges each step the increase of that count over the record so
  far. So the charges add up to the count of the whole record at every step,
  whenever it is read, with no window clock, and no step is charged less than 0.
* The plant keeps 16 open turning points. A 17th closes the innermost pair as a
  cycle: after a ring-down at the tower's 5 to 8 % damping that pair's range is
  under a tenth of the outer one, a ten-thousandth of its damage.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.energy.wind_turbine import env as W
from target_gym.registry import REGISTRY

M = 4.0
MN = 1.0e6


def astm_rainflow(series):
    """Damage of ``series`` (MN m) by ASTM E1049 three-point rainflow, the ranges
    left at the end as half cycles: sum(count x range^m)."""
    points = [series[0]]
    for v in series[1:]:
        if v == points[-1]:
            continue
        if len(points) >= 2 and (points[-1] - points[-2]) * (v - points[-1]) > 0:
            points[-1] = v
        else:
            points.append(v)
    stack, damage = [], 0.0
    for p in points:
        stack.append(p)
        while len(stack) >= 3:
            x = abs(stack[-1] - stack[-2])
            y = abs(stack[-2] - stack[-3])
            if x < y:
                break
            if len(stack) == 3:
                damage += 0.5 * y**M
                stack.pop(0)
            else:
                damage += y**M
                last = stack.pop()
                stack.pop()
                stack.pop()
                stack.append(last)
    return damage + sum(0.5 * abs(b - a) ** M for a, b in zip(stack[:-1], stack[1:]))


@jax.jit
def _charges(series):
    """The plant's charge at every step of ``series`` (N m), from its first point."""
    stack = jnp.zeros(W.RAINFLOW_DEPTH).at[0].set(series[0])
    start = (stack, jnp.int32(1), series[0], jnp.zeros(()))

    def body(carry, new):
        stack, depth, moment, direction = carry
        stack, depth, direction, damage = W.rainflow_step(
            stack, depth, moment, direction, new, M
        )
        return (stack, depth, new, direction), (damage, depth)

    return jax.lax.scan(body, start, series[1:])[1]


def _series(kind, seed, n=1200):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.25
    if kind == "walk":  # a random walk: every scale of range, new extremes often
        return np.cumsum(rng.normal(0.0, 2.0, n))
    if kind == "tower":  # a slow thrust swing under the tower's 0.324 Hz ripple
        slow = 15.0 * np.sin(2 * np.pi * t / 80.0 + rng.uniform(0, 6.28))
        ripple = rng.uniform(1.0, 4.0) * np.sin(2 * np.pi * 0.324 * t)
        return 40.0 + slow + ripple + np.cumsum(rng.normal(0.0, 0.3, n))
    if kind == "noise":  # white noise: a turning point at nearly every step
        return 40.0 + rng.normal(0.0, 5.0, n)
    raise ValueError(kind)


# Float32 on fourth powers of ranges up to 100 MN m: 1e-4 relative, plus 1
# (MN m)^4, the rounding of one such power.
@pytest.mark.parametrize("kind", ["walk", "tower", "noise"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_charges_add_up_to_the_astm_count_of_the_record_so_far(kind, seed):
    series = _series(kind, seed)
    charged = np.cumsum(np.asarray(_charges(jnp.asarray(series * MN))[0], float))
    for n in (1, 2, 3, 7, 50, 333, 800, len(series) - 1):
        expected = astm_rainflow(list(series[: n + 1]))
        assert charged[n - 1] == pytest.approx(expected, rel=1e-4, abs=1.0), n


@pytest.mark.parametrize("kind", ["walk", "tower", "noise"])
def test_no_step_is_charged_less_than_nothing(kind):
    damage = np.asarray(_charges(jnp.asarray(_series(kind, 5) * MN))[0], float)
    assert damage.min() >= 0.0


def test_a_slow_swing_is_counted_whole_under_its_ripples():
    """One 100 s swing of +/- 20 MN m with a +/- 1 MN m ripple at the tower's
    frequency riding on it. The swing alone is 0 -> 20 -> -20 -> 0: half cycles
    of 20, 40 and 20, 0.5 (20^4 + 40^4 + 20^4) = 9 x 20^4 = 1.44e6. The ripple
    widens the extremes by up to 1 each and adds 32 cycles of about 2 MN m, 16
    each: between 9 x 19^4 and 9 x 21^4 + 1000. Counting each range between
    neighbouring turning points instead, as the plant did before, gives under a
    tenth of that."""
    t = np.arange(401) * 0.25
    series = 20.0 * np.sin(2 * np.pi * t / 100.0) + np.sin(2 * np.pi * 0.324 * t)
    total = float(np.sum(np.asarray(_charges(jnp.asarray(series * MN))[0], float)))
    assert 9 * 19.0**4 <= total <= 9 * 21.0**4 + 1000.0


def test_a_long_ring_down_stays_inside_the_sixteen_open_points():
    """60 half cycles each 0.85 of the last, as a tower at 5 % damping rings
    down from 50 MN m: every range nests inside the one before, so none closes
    and the 17th turning point meets a full stack. The count stays within a
    thousandth of ASTM's and the stack within its 16."""
    peaks = 50.0 * 0.85 ** np.arange(60) * (-1.0) ** np.arange(60)
    series = np.concatenate([[0.0], peaks])
    damage, depth = _charges(jnp.asarray(series * MN))
    assert int(np.max(np.asarray(depth))) <= W.RAINFLOW_DEPTH
    total = float(np.sum(np.asarray(damage, float)))
    assert total == pytest.approx(astm_rainflow(list(series)), rel=1e-3)


def test_the_plant_charges_the_rainflow_count_of_its_own_moment():
    """Ten minutes of the turbine in turbulence under its reset's own pitch and
    torque: the tower's charges, at its price, add up to the ASTM count of the
    base moment the run went through."""
    spec = REGISTRY["wind_turbine"]
    env = spec.make_env()
    p = spec.make_test_params(v_mean_range=(18.0, 18.0), sensor_noise_scale=0.0)
    free = p.replace(overspeed_factor=1e9)
    _, s0 = env.reset_env(jax.random.PRNGKey(6), p)
    held = env.initial_action(s0, p)

    def body(s, k):
        s1 = env.step_env(k, s, held, free)[1]
        return s1, (s1.moment, env.reward_terms(s1, free)["tower"])

    moments, tower = jax.lax.scan(
        body, s0, jax.random.split(jax.random.PRNGKey(6), 2400)
    )[1]
    record = [float(s0.moment) / MN, *(np.asarray(moments, float) / MN)]
    price = float(W.tower_cost(p)) / (
        float(p.tower_damage_design) * float(W.life_steps(p))
    )
    assert float(np.sum(np.asarray(tower, float))) == pytest.approx(
        price * astm_rainflow(record), rel=1e-3
    )


# The ceiling plans the tower's count through rows frozen outside the solver
# (oracle.py): rainflow pairs up the turning points of the record to come, and
# every pair's range is charged at its two ends. At the plan the rows were
# counted on they must say what the plant will charge from here: the ASTM count
# of the whole record less the count of the record so far. Float64 here, so
# 1e-9.
@pytest.mark.parametrize("kind", ["walk", "tower", "noise"])
@pytest.mark.parametrize("seed", [0, 1])
def test_the_plans_tower_rows_say_what_the_plant_will_charge(kind, seed):
    from target_gym.energy.wind_turbine import oracle as O

    series = _series(kind, seed, n=400)
    past, plan = series[:150], series[150:]
    stack = np.zeros(W.RAINFLOW_DEPTH)
    stack[0] = past[0] * MN
    depth, moment, direction = 1, past[0] * MN, 0.0
    for new in past[1:] * MN:  # the plant's own count over the past, in float64
        with jax.experimental.enable_x64():
            out = W.rainflow_step(
                jnp.asarray(stack), jnp.asarray(depth), jnp.asarray(moment),
                jnp.asarray(direction), jnp.asarray(new), M,
            )  # fmt: skip
        stack, depth, direction = np.asarray(out[0]), int(out[1]), float(out[2])
        moment = new
    centres, scales, constant = O.rainflow_rows(stack, depth, moment, plan * MN, M)
    assert centres.shape == scales.shape == (len(plan), 2)
    rows = scales * (plan[:, None] * MN - centres) / MN
    planned = float(np.sum(np.abs(rows) ** M) + constant)
    expected = astm_rainflow(list(series)) - astm_rainflow(list(past))
    assert planned == pytest.approx(expected, rel=1e-9, abs=1e-6)
