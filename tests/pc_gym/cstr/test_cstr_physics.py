"""Physics of the jacketed CSTR at its design point (version 5).

Contract, from S1's design point and the balances: the residence time is S1's
34 min and the 15 s step resolves it; the design operating point is open-loop
stable (PARSEX runs it in closed loop, but nothing says it is unstable [S1],
and the owner's review notes no source shows it unstable); heat removal by the
jacket at the design point equals the heat the balance leaves.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.pc_gym.cstr.env import CSTRParams, compute_velocity, steady_state

P = CSTRParams()


def test_residence_time_and_step_resolution():
    tau = P.V / P.q
    assert tau == pytest.approx(34.0, rel=1e-3)  # [S1, p. 5]
    assert tau / P.delta_t > 100


def test_the_design_point_is_a_rest_point_and_open_loop_stable():
    C_a, T_j, F_w = steady_state(331.15, 297.15, P)
    x = jnp.array([C_a, 331.15, T_j])
    extra = (297.15, P.Caf, P.k0)
    v, _ = compute_velocity(x, action=F_w, params=P, extra=extra)
    np.testing.assert_allclose(np.asarray(v), 0.0, atol=1e-6)
    J = jax.jacfwd(lambda z: compute_velocity(z, F_w, P, extra)[0])(x)
    assert np.all(np.linalg.eigvals(np.asarray(J)).real < 0.0)
