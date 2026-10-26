"""The floor-normalised reward every environment scores through.

``reward = -(tracking_cost + running_cost + failure_cost)``: three
non-negative costs, added. docs/reward-shaping.md gives the argument; this
module is the arithmetic.

- **Tracking**: ``(max(|e| - e_tol, 0) / e_floor) ** p``. ``e_floor`` is the
  achievable floor (the smallest long-run mean |error| any controller holds on
  the plant, never finer than its instrument resolves), so an error at the
  floor costs 1 per step. ``e_tol`` is a specification tolerance where the plant
  has one; ``p`` is 1 where the owner's cost is linear in the error, else 2.
- **Running**: consumption above ``c_hold``, what the best shipped controller
  consumes while holding: ``w * max(c - c_hold, 0) / c_hold``, or
  ``price * quantity`` where the owner's tariff is known.
- **Failure**: the step that leaves the operating envelope is charged
  ``restart_steps * failure_cost`` and the plant restarts at once
  (``base.failure_kernel``). By default no episode ends at a trip.

The best per-step reward is about -1 (tracking at the floor, nothing
avoidable consumed). Compare across plants with the normalised expert
advantage and the two costs reported separately (``target_gym.eval``).
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
    """The trip cost charged by expectation instead of by realisation:
    ``risk * cost * Phi(-margin / (sigma sqrt(reaction_steps)))``, the
    probability that the disturbance carries the state past the envelope
    before the controller can arrest it, times the cost. Smooth where the
    realised charge is a step, so a planner gets a gradient toward the edge;
    at ``reaction_steps = risk = 1`` it has the realised charge's mean.

    ``margin``: distance to the envelope in its own units, positive inside and
    negative outside, where the charge saturates at the whole ``cost``.
    ``sigma``: the one-step disturbance scale in the same units.
    ``reaction_steps``: the steps the loop needs to arrest an excursion; the
    margin is judged against ``sigma * sqrt(reaction_steps)``. It and ``risk``
    (a plain multiplier) change the objective, so state them with any number
    they produce."""
    from jax.scipy.stats import norm

    scale = xp.maximum(sigma, 1e-12) * xp.sqrt(xp.maximum(reaction_steps, 1.0))
    # Not clamped at 0: past the envelope the charge must reach the whole
    # cost, or a planner rolling deep outside is charged half and trips.
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
    """``legacy`` (the version-1 reward) where ``version == 1``, else
    ``current``. Version 1 is kept so v1 results can be reproduced."""
    return xp.where(version == 1, legacy, current)
