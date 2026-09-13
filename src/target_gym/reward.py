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

**Running.** Consumption is charged only above ``c_hold``, what the shipped MPC
consumes per step while holding: the part of the fuel, energy, boilup, reagent
or actuator travel that a controller could avoid. Dimensionless form
``w * max(c - c_hold, 0) / c_hold``, with ``w`` documented per plant as "one
floor-width of tracking error is worth w times the hold-phase consumption";
priced form ``price * quantity`` where the owner's tariff is known, with the
tracking cost then in the same currency. Additive, never multiplicative: a
product ``tracking * (1 - w * running)`` charges nothing for consumption
exactly where tracking is worst, and a cost that vanishes when it should
bind is not a cost.

**Failure.** A terminal state costs, per step, more than the largest tracking
cost the operating envelope can produce, so that a policy which fails with any
probability has the worst possible long-run cost (safety enters through the
gain, Proposition 5 of the note).

The best achievable per-step reward is then about -1 (tracking at the floor,
nothing avoidable consumed), not 0 or 1: cross-plant comparability comes from
the normalised expert advantage ``(PID - x) / (PID - floor)`` and from
reporting the two costs separately (``target_gym.eval``), not from a cap.
"""

from __future__ import annotations

import jax.numpy as jnp


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


def failure_cost(terminated, per_step, xp=jnp):
    """Per-step cost of being in an absorbing failure state."""
    return xp.where(terminated, per_step, 0.0)


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
