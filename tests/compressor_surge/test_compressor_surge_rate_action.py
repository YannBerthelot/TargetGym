"""The anti-surge pair's rate action (owner, picker 2026-11-25).

Contract, from REALITY.md and the owner's choice, not from the code:

- Rate action: the valve opens early when flow falls fast, before the control
  line is reached [C3, p. 2]; rate-of-change detection opens the recycle valve
  in time to stop surge in its first cycle [C7].
- As built: while the margin the anti-surge PI reads falls, the control line it
  holds moves right by ``rate_lead`` (s) times the rate of fall (margin per s).
  A steady or rising margin moves nothing. So on a step where the margin falls
  by d over one control step dt, the PI's proportional part opens the valve by
  ``kp_surge * rate_lead * d / dt`` more than without the term; with
  ``rate_lead`` 0 the pair is the PI pair with its backup line.
"""

import numpy as np
import pytest

from target_gym.compressor_surge.env import CompressorSurgeParams, surge_flow_per_speed
from target_gym.compressor_surge.experts import (
    DEFAULT_GAINS,
    HELD_GAINS,
    PairState,
    commands_from_raw,
    load_gains,
    pair_step,
)

P = CompressorSurgeParams()
#: Rated speed, the header at its setpoint, the valve a third open.
SPEED, DP, X = 100.0, 24.0, 0.3


def _obs(margin):
    """An observation whose map margin is ``margin``: m_c = (1 + margin) times
    the map's surge flow at rated speed."""
    m_c = (1.0 + margin) * float(surge_flow_per_speed(P)) * SPEED / 100.0
    return np.array([DP, m_c, 10.0, SPEED, SPEED, 100.0 * X, DP, 0.0])


def _valve(gains, previous, margin):
    """The valve command after one update from memory that last read
    ``previous`` (NaN: no reading yet), the integral at the valve."""
    ps = PairState(I_N=1.0, I_x=X, timer=0.0, margin=previous)
    u, _ = pair_step(gains, ps, _obs(margin), P)
    return float(commands_from_raw(u, P)[1])


def _gains(**changes):
    return {**DEFAULT_GAINS, "Tf_surge": 0.0, **changes}


def test_the_rate_lead_is_a_held_gain_of_the_pair_and_the_shipped_pair_uses_it():
    """The recipe sets it (the fallback keeps it off): the shipped pair leads."""
    assert "rate_lead" in DEFAULT_GAINS and "rate_lead" in HELD_GAINS
    assert load_gains()["rate_lead"] > 0.0


def test_a_falling_margin_opens_the_valve_by_the_lead_times_the_rate():
    """From 0.20 to 0.19 in one 0.1 s step: a fall of 0.1 per s. With a lead of
    0.3 s the line moves right by 0.03, so the valve opens kp_surge x 0.03 more."""
    lead, kp = 0.3, 2.0
    with_rate = _valve(_gains(rate_lead=lead, kp_surge=kp), 0.20, 0.19)
    without = _valve(_gains(rate_lead=0.0, kp_surge=kp), 0.20, 0.19)
    assert with_rate - without == pytest.approx(kp * lead * 0.01 / 0.1, rel=1e-5)


@pytest.mark.parametrize("previous", [0.19, 0.18, np.nan])
def test_a_steady_or_rising_margin_or_a_first_reading_moves_nothing(previous):
    lead = _valve(_gains(rate_lead=0.3), previous, 0.19)
    plain = _valve(_gains(rate_lead=0.0), previous, 0.19)
    # The observation's margin is 0.19 to within float rounding (1e-16), so a
    # steady reading may fall by that much: 1e-9 is far below any rate action.
    assert lead == pytest.approx(plain, abs=1e-9)


def test_the_valve_opens_before_the_control_line_when_the_fall_is_fast():
    """At 0.17, above the 0.15 line, a steady margin closes the valve from its
    integral while a fall of 0.3 per s (line moved right by 0.09) opens it."""
    steady = _valve(_gains(rate_lead=0.3), 0.17, 0.17)
    falling = _valve(_gains(rate_lead=0.3), 0.20, 0.17)
    assert steady < X < falling
