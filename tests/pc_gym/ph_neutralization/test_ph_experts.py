"""The pH oracle's planner pH.

The env finds pH by bisection, whose derivative is zero, so the planner's pH
carries the titration curve's slope instead. The closed-loop oracle check went
with the tracking score (charter rule 15): the oracle is rebuilt for the job
score before it is run again.
"""

import jax
import numpy as np
import pytest

from target_gym import registry


def _root(Wa, Wb, p):
    """The charge balance's root in float64 by Brent's method, from the
    equation in env.py's docstring (not from its bisection):
    Wa + 10^(pH-14) - 10^-pH + Wb (1 + 2 10^(pH-pK2)) / (1 + 10^(pK1-pH)
    + 10^(pH-pK2)) = 0."""
    from scipy.optimize import brentq

    def balance(pH):
        carbonate = (1 + 2 * 10 ** (pH - p.pK2)) / (
            1 + 10 ** (p.pK1 - pH) + 10 ** (pH - p.pK2)
        )
        return Wa + 10 ** (pH - 14) - 10**-pH + Wb * carbonate

    return brentq(balance, 0.0, 14.0, xtol=1e-14)


@pytest.mark.parametrize("q3", [12.0, 15.6, 18.0])
def test_the_planners_pH_has_the_titration_curves_slope(q3):
    """The env solves pH by bisection, whose derivative is zero; the
    planner's pH has the env's value and the curve's slope. Both against the
    float64 root of the documented charge balance and its central
    differences. 1e-4 pH is the bisection's resolution; 1e-3 relative on the
    slope allows for float32 against float64."""
    import jax.numpy as jnp

    from target_gym.pc_gym.ph_neutralization.env import steady_state_invariants
    from target_gym.pc_gym.ph_neutralization.oracle import planner_pH

    p = registry.get("ph_neutralization").make_test_params()
    Wa, Wb = (float(v) for v in steady_state_invariants(q3, p.q2_nominal, p))
    value = float(planner_pH(jnp.float32(Wa), jnp.float32(Wb), p))
    assert value == pytest.approx(_root(Wa, Wb, p), abs=1e-4)

    grad = jax.grad(planner_pH, argnums=(0, 1))(jnp.float32(Wa), jnp.float32(Wb), p)
    h = 1e-9
    want = (
        (_root(Wa + h, Wb, p) - _root(Wa - h, Wb, p)) / (2 * h),
        (_root(Wa, Wb + h, p) - _root(Wa, Wb - h, p)) / (2 * h),
    )
    np.testing.assert_allclose([float(g) for g in grad], want, rtol=1e-3)
