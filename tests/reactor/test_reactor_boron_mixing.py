"""A trickle of boron flow moves the boron it pays for (review of 2026-11-23).

Contract, from the plant record (PHYSICS.md, section 1), written before the fix:
boron is a feed and bleed of well-mixed coolant,

    d(boron)/dt = Q / M (C_feed - boron)

with Q the command's share of 10 kg/s of dilution (C_feed 0) or of 3 kg/s of
boration (C_feed 7800 ppm), and M the primary's 2.5e5 kg. Nothing in it depends on
how small the flow is. So a command held for 1200 control steps of 30 s, 10 h,
passes ``command x 10 x 30 x 1200`` kg of water, 1.44 primary masses per unit of
command, and the boron closes on its feed by 1 - exp(-1.44 x command):

- dilution at 0.0001 of full flow: 36 kg, 1.43990e-4 of the boron, 0.2433 ppm of
  1690;
- at 0.0002: 72 kg, 2.87959e-4, 0.4867 ppm;
- at 0.001: 360 kg, 1.43896e-3, 2.432 ppm.

Boration passes 0.432 primary masses per unit of command over the same 10 h and
closes on 7800 ppm:

- at 0.0002: 21.6 kg, 8.6396e-5 of the 6110 ppm between the core and the acid,
  0.5279 ppm;
- at 0.001: 108 kg, 4.3191e-4, 2.639 ppm.

The water is paid for by the kilogram whatever it does, so each of these is also
what the day charges: $0.0628 a kg.

A flow this small moves the boron by less than a single-precision number near
1690 ppm can hold in one sub-step (1.2e-4 ppm), which is how the plant came to
give 0.88 ppm for the 72 kg and nothing for the 36 kg.

The plant here is quiet: no load wobble, no sensor noise, no burnup, no debt
carried in, at the start of its cycle, xenon balanced with full load, on full load
all day, its rods held where they are.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.reactor import Reactor, ReactorParams

QUIET = ReactorParams(
    demand_sigma=0.0,
    sensor_noise_scale=0.0,
    burnup_range=(0.0, 0.0),
    carried_days_range=(0.0, 0.0),
    cycle_range=(0.0, 0.0),
    initial_tilt_share=0.0,
    initial_xenon_power_range=(1.0, 1.0),
    ramp_down_range=(1e6, 1e6),
)
STEPS = 1200  # 10 h of the command
LINE = 40  # control steps for the last of it to reach the core (30 at most)
ACID = 7800.0  # ppm


def _after(command):
    """The boron before and after ``command`` held for 10 h, the dollars of
    effluent charged, and whether the plant tripped."""
    env = Reactor()
    _, s0 = env.reset_env(jax.random.PRNGKey(0), QUIET)
    rods = 1.0 - 2.0 * float(s0.rod_in)
    flow = np.where(np.arange(STEPS + LINE) < STEPS, command, 0.0)
    actions = jnp.asarray(
        np.stack([np.full_like(flow, rods), flow, np.ones_like(flow)], axis=1),
        jnp.float32,
    )

    def step(s, a):
        _, s1, _, _, info = env.step_env(jax.random.PRNGKey(1), s, a, QUIET)
        return s1, (info["reward_terms"]["running"], info["tripped"])

    s1, (running, tripped) = jax.jit(lambda s: jax.lax.scan(step, s, actions))(s0)
    return float(s0.boron), float(s1.boron), float(running.sum()), bool(tripped.any())


@pytest.mark.parametrize(
    "command, share, kg",
    [(-0.0001, 1.43990e-4, 36.0), (-0.0002, 2.87959e-4, 72.0), (-0.001, 1.43896e-3, 360.0)],
)  # fmt: skip
def test_a_trickle_of_dilution_removes_the_boron_its_water_carries(command, share, kg):
    before, after, dollars, tripped = _after(command)
    assert not tripped
    assert before == pytest.approx(1690.0, abs=5.0)
    # 1 %: the boron is read to 1.2e-4 ppm, 0.05 % of the smallest drop here.
    assert before - after == pytest.approx(before * share, rel=0.01)
    assert dollars == pytest.approx(0.0628 * kg, rel=0.01)


@pytest.mark.parametrize(
    "command, share, kg", [(0.0002, 8.6396e-5, 21.6), (0.001, 4.3191e-4, 108.0)]
)
def test_a_trickle_of_boration_adds_the_boron_its_acid_carries(command, share, kg):
    before, after, dollars, tripped = _after(command)
    assert not tripped
    assert after - before == pytest.approx((ACID - before) * share, rel=0.01)
    assert dollars == pytest.approx(0.0628 * kg, rel=0.01)


def test_twice_the_trickle_removes_twice_the_boron():
    """The law is linear in a small flow: no flow is too small to count, and none
    is rounded up."""
    before, one, _, _ = _after(-0.0001)
    _, two, _, _ = _after(-0.0002)
    assert before - two == pytest.approx(2.0 * (before - one), rel=0.01)
    assert before - one > 0.2
