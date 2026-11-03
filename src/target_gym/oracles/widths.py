"""The smoothing width of a kink site, by rule K5.

docs/proposals/oracle-smoothing-protocol.md, K5 (accepted as S1): the
softplus width whose worst one-step effect on every tracked output stays
under half that output's resolution. Each task's sites, scales and probe
states come with its planner model (PRs C and D); this is the generic rule.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["kink_width"]


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
