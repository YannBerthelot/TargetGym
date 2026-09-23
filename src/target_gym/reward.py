"""The floor-normalised reward every environment scores through.

``reward = -(tracking_cost + running_cost + failure_cost)``, three additive
terms, each a cost in a unit the plant can defend. See docs/reward-shaping.md
for the argument; this module is the arithmetic.

**Tracking.** ``(max(|e| - e_tol, 0) / e_floor) ** p``. ``e_floor`` is the
achievable floor: the smallest long-run mean |error| any controller can hold
on this plant under its shipped reference and disturbance, computed or bounded
per plant (each PHYSICS.md names the script). An error at the floor costs 1 per
step; the scale is the plant's own irreducible error, not a sensor resolution
or an envelope. ``e_tol`` is a specification tolerance where the plant has one
(a comfort band, a purity spec, a permit limit): the cost is zero inside it,
which is a dead-zone relaxation and forfeits at most the tolerance in long-run
cost (Target-MDP note, Theorem 3). ``p`` is 1 where the owner's cost is linear
in the error (an energy imbalance settled at a price per MWh) and 2 otherwise.
Convex in |e| by construction: a log-scaled tracking term, the previous
choice, is concave and pays a controller that trades rare large excursions
for frequent small ones (Proposition 8 of the note), which is the wrong
preference for a hold.

**Running.** Consumption is charged only above ``c_hold``, what the best shipped
controller (MPC or PID) consumes per step while holding: the part of the fuel, energy, boilup, reagent
or actuator travel that a controller could avoid. Dimensionless form
``w * max(c - c_hold, 0) / c_hold``, with ``w`` documented per plant as "one
floor-width of tracking error is worth w times the hold-phase consumption";
priced form ``price * quantity`` where the owner's tariff is known, with the
tracking cost then in the same currency. Additive, never multiplicative: a
product ``tracking * (1 - w * running)`` charges nothing for consumption
exactly where tracking is worst, and a cost that vanishes when it should
bind is not a cost.

**Failure.** Leaving the operating envelope trips the plant. A tripped plant
is down at ``failure_cost`` per step -- above the largest tracking cost the
envelope can produce, with tracking and running cost zeroed -- for
``restart_steps`` steps, then restarts (``base.failure_kernel``); a plant
that cannot restart stays down to the end of the window. That is the
absorbing state of the Target-MDP note lived through inside the window, so a
policy which fails with any probability has the worst long-run cost (safety
enters through the gain, Proposition 5), and under a finite restart the gain
decomposes as ``rho_hold + p * B + lambda * (restart_steps * failure_cost +
B_restart)`` with ``lambda`` the trip rate. No episode ends at a trip.

The best achievable per-step reward is then about -1 (tracking at the floor,
nothing avoidable consumed), not 0 or 1: cross-plant comparability comes from
the normalised expert advantage ``(PID - x) / (PID - floor)`` and from
reporting the two costs separately (``target_gym.eval``), not from a cap.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np


def tracking_cost(error, e_floor, e_tol=0.0, p=2.0, xp=jnp):
    """Floor-normalised tracking cost, zero inside ``e_tol``, 1 at the floor."""
    excess = xp.maximum(xp.abs(error) - e_tol, 0.0)
    return (excess / e_floor) ** p


def running_cost(quantity, c_hold, weight=1.0, xp=jnp):
    """Avoidable consumption as a fraction of the hold-phase consumption.

    ``c_hold`` must be positive: it is a measured MPC consumption, and a plant
    whose controller consumes nothing while holding has no running cost to
    charge.
    """
    return weight * xp.maximum(quantity - c_hold, 0.0) / c_hold


def priced_cost(quantity, price, xp=jnp):
    """A consumption in the owner's currency per step: ``price`` per unit."""
    return price * quantity


def trip_cost(params):
    """The cost of a trip, charged once on the step that leaves the envelope:
    the plant's restart time priced at the per-step failure cost."""
    return params.restart_steps * params.failure_cost


def _erfc_np(z):
    """erfc(z / sqrt 2) on numpy arrays, for the xp=np scoring path."""
    from scipy.special import erfc

    return erfc(np.asarray(z) / math.sqrt(2.0))


def proximity_cost(margin, sigma, cost, reaction_steps=1.0, risk=1.0, xp=jnp):
    """The trip's cost charged by EXPECTATION instead of by realisation:
    ``cost * Phi(-margin / (sigma sqrt(reaction_steps)))``, the probability
    that the disturbance carries the state past the envelope before the
    controller can arrest it.

    ``margin``: the distance from the current state to the envelope, in the
    envelope's own units - positive inside it, NEGATIVE outside, where the
    charge saturates at the whole lump. On a deterministic rollout (a planner's
    model, which has no noise of its own) the charge is then a smoothed step:
    ~0 far inside, half at the boundary, ~1 lump well outside, and it carries a
    gradient across the crossing where the step function has none. ``sigma``: the plant's one-step
    disturbance scale in the same units - ``e_floor sqrt(pi/2)`` where the
    floor is the irreducible error of a driven plant, since
    ``e_floor = sd sqrt(2/pi)`` is how that floor is computed. ``cost``:
    :func:`trip_cost`.

    WHY EXPECTATION. Charging the realised trip makes the cost discontinuous
    at the envelope - zero derivative on both sides and a jump between - so no
    gradient method can see the cliff coming, and two controllers that spend
    the same time at the edge score differently according to whether one was
    pushed over. The conditional expectation of the same lump is smooth, has a
    gradient wherever the margin does, and is the Rao-Blackwellisation of the
    indicator: at ``reaction_steps = risk = 1`` it has the SAME mean as the
    realised charge and strictly lower variance. Report the realised cost as
    the KPI and optimise this one, and the two agree in expectation.

    RISK AVERSION is the other two arguments, and both change the objective
    rather than the estimator - state them wherever a number produced with
    them is reported. ``reaction_steps``: judge the margin against the
    disturbance accumulated over the time the loop needs to arrest an
    excursion (dead time plus a dominant time constant), not over one step -
    the random-walk bound ``sigma sqrt(h)``, conservative where the
    disturbance is mean-reverting. This is the physical form of "hold a
    margin": a plant that cannot be arrested quickly is charged more for the
    same distance, which is why it bites hardest on long dead times and
    non-minimum-phase plants. ``risk``: a bare multiplier on the charge; a
    poor instrument on its own, since a Gaussian tail is steep enough that
    100x moves the point where the charge overtakes tracking by about one
    sigma."""
    from jax.scipy.stats import norm

    scale = xp.maximum(sigma, 1e-12) * xp.sqrt(xp.maximum(reaction_steps, 1.0))
    # the margin is NOT clamped at 0: a NEGATIVE margin is a state already past
    # the envelope, and the charge must go to the whole lump there, not stay at
    # the half it takes at the boundary. Clamping made this a smoothed step
    # that saturated at 1/2, so a planner rolling a trajectory deep outside the
    # envelope was charged half of what crossing costs and took the trip
    # (measured on 8 lag plants: trips 568 -> 1143 with the clamp in)
    z = margin / scale
    tail = norm.cdf(-z) if xp is jnp else 0.5 * _erfc_np(z)
    return risk * cost * tail


def with_trip(terms: dict, tripped, cost, xp=jnp) -> dict:
    """The terms of a state outside the envelope: every cost zeroed and the
    trip cost charged. The kernel restarts the plant on the same step
    (``base.failure_kernel``), so this is paid once per trip."""
    out = {k: xp.where(tripped, 0.0, v) for k, v in terms.items()}
    out["failure"] = xp.where(tripped, cost, 0.0)
    return out


def total(terms: dict, xp=jnp):
    """Sum a ``{name: cost}`` breakdown into the scalar reward, ``-sum``."""
    out = 0.0
    for v in terms.values():
        out = out + v
    return -out


def select(version, legacy, current, xp=jnp):
    """The reward for ``params.reward_version``: 1 is the log-scaled, capped
    reward the v1 environments shipped with, kept so a number published against
    a v1 environment can be reproduced; anything else is the floor-normalised
    cost above."""
    return xp.where(version == 1, legacy, current)
