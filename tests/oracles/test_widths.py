"""The K5 width tool (target_gym.oracles.widths).

Contract: docs/proposals/oracle-smoothing-protocol.md K5. The largest w on
{s/10, ..., s/1e8} with w log 2 sum_i |dy / d delta_i| <= q_y / 2 at every
probe and for every output. On a site whose outputs are linear in its value
the bound is closed-form, worked by hand in the comments.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.oracles.widths import kink_width, limit_backoff, noise_backoff

LN2 = math.log(2.0)


def soft_clip(x, w=0.1):
    return x - w * jnp.logaddexp(0.0, (x - 1.0) / w) + w * jnp.logaddexp(0.0, (-x) / w)


def linear_site(gains):
    """One step whose outputs are linear in the site's output:
    y_j = g_j(probe) (soft_clip(probe) + sum_i delta_i), so
    sum_i |dy_j / d delta_i| = evaluations x |g_j|."""
    return lambda probe, delta: jnp.stack(
        [g(probe) * (soft_clip(probe) + delta.sum()) for g in gains]
    )


@pytest.mark.parametrize(
    "gains, probes, q, evaluations, scale, expected",
    [
        # w <= 1e-3 / (2 ln 2 x 2) = 3.6e-4: the rung is 1e-4.
        ([lambda x: 2.0], [0.5], [1e-3], 1, 1.0, 1e-4),
        # Two evaluations double the slope: 1.8e-4, still 1e-4.
        ([lambda x: 2.0], [0.5], [1e-3], 2, 1.0, 1e-4),
        # The finer output wins: 1e-6 / (2 ln 2 x 0.5) = 1.44e-6 -> 1e-6.
        ([lambda x: 2.0, lambda x: 0.5], [0.5], [1e-3, 1e-6], 1, 1.0, 1e-6),
        # The worst probe wins: 1e-2 / (2 ln 2 x 3) = 2.4e-3; on s = 50 the
        # ladder is 5, 0.5, 0.05, 5e-3, 5e-4: 5e-4.
        ([lambda x: x], [1.0, -3.0, 0.2], [1e-2], 1, 50.0, 5e-4),
        # Bounds of 1.2e-4 and 0.9e-4 sit just above and below a rung, so a
        # factor of 2 anywhere in the bound moves the answer.
        ([lambda x: 1.0], [0.5], [2 * LN2 * 1.2e-4], 1, 1.0, 1e-4),
        ([lambda x: 1.0], [0.5], [2 * LN2 * 0.9e-4], 1, 1.0, 1e-5),
        # Two evaluations halve a 1.5e-4 bound to 0.75e-4.
        ([lambda x: 1.0], [0.5], [2 * LN2 * 1.5e-4], 2, 1.0, 1e-5),
        ([lambda x: 0.0], [0.5], [1e-3], 1, 1.0, 0.1),  # no effect: s / 10
        ([lambda x: 1e12], [0.5], [1e-3], 1, 1.0, None),  # no rung passes
    ],
)
def test_kink_width_is_the_largest_rung_under_the_resolution_bound(
    gains, probes, q, evaluations, scale, expected
):
    w = kink_width(linear_site(gains), scale, probes, q, evaluations)
    assert w == (None if expected is None else pytest.approx(expected))


def two_site_limit(gains):
    """A limited quantity linear in two sites' outputs:
    q = g_0 sum_j delta_0j + g_1 sum_j delta_1j, so its back-off is
    log 2 (w_0 |g_0| + w_1 |g_1|) x evaluations."""
    return lambda probe, deltas: jnp.stack(
        [probe * (g[0] * deltas[0].sum() + g[1] * deltas[1].sum()) for g in gains]
    )


@pytest.mark.parametrize(
    "gains, probes, widths, evaluations, expected",
    [
        # ln 2 x (1e-4 x 3 + 1e-3 x 2) = ln 2 x 2.3e-3.
        ([(3.0, 2.0)], [1.0], (1e-4, 1e-3), 1, [LN2 * 2.3e-3]),
        # Ten evaluations of each site: ten times that.
        ([(3.0, 2.0)], [1.0], (1e-4, 1e-3), 10, [LN2 * 2.3e-2]),
        # The worst probe (x 2) and the sign dropped: ln 2 x 2 x 4e-4.
        ([(-4.0, 0.0)], [0.5, -2.0], (1e-4, 1e-3), 1, [LN2 * 8e-4]),
        # Two quantities, each its own; an exact site (width 0) adds nothing.
        ([(1.0, 5.0), (2.0, 0.0)], [1.0], (1e-3, 0.0), 1, [LN2 * 1e-3, LN2 * 2e-3]),
    ],
)
def test_a_limits_back_off_is_the_gap_bound_of_its_quantity(
    gains, probes, widths, evaluations, expected
):
    b = limit_backoff(two_site_limit(gains), widths, probes, evaluations)
    np.testing.assert_allclose(b, expected, rtol=1e-12)


# The normal quantile at 1 - 1e-6 (standard tables): the per-step risk of a
# 1.2% risk over 12000 steps, Boole's inequality split evenly.
Z_1E6 = 4.753424


@pytest.mark.parametrize(
    "gains, probes, std, expected",
    [
        # z x 0.5 x |3|, the sign dropped.
        ([-3.0], [1.0], 0.5, [Z_1E6 * 1.5]),
        # The worst probe (x 2), and each quantity its own.
        ([1.0, 0.0], [0.5, 2.0], 0.1, [Z_1E6 * 0.2, 0.0]),
    ],
)
def test_a_limits_noise_back_off_is_its_quantile_of_one_innovation(
    gains, probes, std, expected
):
    site = lambda probe, d: jnp.stack([probe * g * d for g in gains])  # noqa: E731
    b = noise_backoff(site, std, probes, risk=0.012, steps=12000)
    np.testing.assert_allclose(b, expected, rtol=1e-5)
