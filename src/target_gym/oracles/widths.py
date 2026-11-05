"""The smoothing width of a kink site, by rule K5, and a limit's back-off,
its two terms by rule K11.

docs/proposals/oracle-smoothing-protocol.md, K5 (accepted as S1): the
softplus width whose worst one-step effect on every tracked output stays
under half that output's resolution. Each task's sites, scales and probe
states come with its planner model (PRs C and D); this is the generic rule.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["kink_width", "limit_backoff", "noise_backoff"]


def kink_width(site, scale: float, probes, resolutions, evaluations: int = 1):
    """The largest w in {s/10, s/100, ..., s/1e8} with, at every probe,
    w log 2 sum_i |dy / d delta_i| <= q_y / 2 for every tracked output y.

    ``site(probe, delta) -> y`` steps the planner's model once from ``probe``
    with ``delta[i]`` added to the site's output at its ``i``-th evaluation
    in the step, and returns the tracked outputs; ``resolutions`` are their
    precision floors q_y. ``None`` if no width on the ladder passes.
    """
    q = np.atleast_1d(np.asarray(resolutions, float))
    worst = np.zeros_like(q)
    for probe in probes:
        jac = jax.jacfwd(lambda d, probe=probe: site(probe, d))(jnp.zeros(evaluations))
        gain = np.abs(np.asarray(jac, float)).reshape(len(q), evaluations).sum(axis=1)
        worst = np.maximum(worst, gain)
    for rung in range(1, 9):
        w = scale * 10.0**-rung
        if np.all(w * math.log(2.0) * worst <= q / 2):
            return w
    return None


def limit_backoff(site, widths, probes, evaluations: int = 1):
    """K11: for each limited quantity q, the largest over the probes of
    sum_k w_k log 2 sum_i |dq / d delta_ki|, the gap bound K5 puts on a
    tracked output, here on q.

    ``site(probe, deltas) -> q`` steps the planner's model once from
    ``probe`` with ``deltas[k, i]`` added to site k's output at its ``i``-th
    evaluation in the step; ``widths`` are the sites' (0 for an exact one).
    """
    w = np.asarray(widths, float)
    worst = None
    for probe in probes:
        zero = jnp.zeros((len(w), evaluations))
        jac = np.abs(np.asarray(jax.jacfwd(lambda d, p=probe: site(p, d))(zero), float))
        gap = math.log(2.0) * (jac.sum(axis=-1) * w).sum(axis=-1)
        worst = gap if worst is None else np.maximum(worst, gap)
    return worst


def noise_backoff(site, innovation_std: float, probes, risk: float, steps: int):
    """K11's noise term: for each limited quantity q, z s max |dq / d d| over
    the probes, s the standard deviation of one innovation of the plant's
    noise-driven deviation, which the noise-free plan cannot foresee, and z
    the normal quantile of ``risk`` over ``steps`` steps, split evenly
    (Boole's inequality).

    ``site(probe, d) -> q`` steps the planner's model once from ``probe``
    with its deviation ``d``, the probe's own deviation added by the caller.
    """
    z = NormalDist().inv_cdf(1.0 - risk / steps)
    worst = None
    for probe in probes:
        gain = np.abs(np.asarray(jax.jacfwd(lambda d, p=probe: site(p, d))(0.0), float))
        worst = gain if worst is None else np.maximum(worst, gain)
    return z * innovation_std * worst
