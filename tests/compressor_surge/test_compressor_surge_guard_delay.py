"""The guard's decay check sees the valve's dead time (owner, picker 2026-11-25).

Contract, from the owner's answer, not from the code:

- The guard's verdict on decay is the closed loop's slowest rate,
  -ln(max |eigenvalue|) / dt of one step's Jacobian, with every state the next
  step depends on in the state vector. The valve acts on a command issued a
  dead time ago, so the commands still in flight are part of that state.
- Its thresholds do not change.

Hand case: an integrator closed through a two-step delay, x' = x - g u2 with
u = x, u1' = u, u2' = u1. Its characteristic polynomial is z^3 - z^2 + g. With
g = 0.125 it factors as (z - 1/2)(z^2 - z/2 - 1/4), whose roots are 1/2 and
(1 +- sqrt 5) / 4, so the slowest mode has |z| = (1 + sqrt 5) / 4 = cos 36 deg.
Without the delay the loop is x' = (1 - g) x, |z| = 7/8.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.compressor_surge.env import CompressorSurgeParams
from target_gym.compressor_surge.experts import (
    DEFAULT_GAINS,
    closed_loop_decay,
    slowest_decay,
)

DT, G = 0.1, 0.125


def _delayed_integrator(z):
    x, u1, u2 = z
    u = x
    return jnp.stack([x - G * u2, u, u1])


def test_a_loop_through_a_two_step_delay_decays_at_its_hand_derived_rate():
    rate = slowest_decay(_delayed_integrator, np.zeros(3), DT)
    assert rate == pytest.approx(-math.log((1 + math.sqrt(5)) / 4) / DT, rel=1e-9)


def test_the_same_loop_without_its_delay_decays_at_its_own_rate():
    rate = slowest_decay(lambda z: z - G * z, np.zeros(1), DT)
    assert rate == pytest.approx(-math.log(7 / 8) / DT, rel=1e-9)


def test_with_the_dead_time_the_anti_surge_integrator_is_inside_the_loop():
    """Left out of the state, the commands in flight are frozen, so the valve
    does not answer this step's command and the anti-surge integrator is open:
    an eigenvalue of exactly 1, a decay of exactly 0 at every guard point."""
    decay = closed_loop_decay(DEFAULT_GAINS, CompressorSurgeParams())
    assert np.all(np.abs(decay) > 1e-6), decay


def test_without_a_dead_time_the_guard_reads_the_loop_it_always_read():
    """valve_lag 0: no command is in flight, and DEFAULT_GAINS decay at 0.45 /s
    or faster, the figure their comment carries (scripts/
    compressor_surge_numbers.py --section guard)."""
    decay = closed_loop_decay(DEFAULT_GAINS, CompressorSurgeParams(valve_lag=0.0))
    assert np.all(decay >= 0.45), decay
