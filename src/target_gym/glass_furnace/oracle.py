"""The oracle (the MPC slot) for the glass furnace task.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``): changing the oracle re-records
only the oracle rows of this package's tasks. The shared machinery
(GradientMPC, NewtonMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every oracle fingerprint.
"""

from target_gym.experts.mpc import NewtonMPC

# ---------------------------------------------------------------------------
# GlassFurnace
# ---------------------------------------------------------------------------


def make_glass_furnace_mpc(env, params, horizon: int = 120, n_iter: int = 10):
    """The oracle: ``NewtonMPC`` on the furnace's own step and reward.

    With delta_t = 30 s, horizon = 120 is one hour of lookahead, about 0.9 of
    the crown's 3960 s (132-step) open-loop time constant. That is enough to
    see each scheduled setpoint change and pre-cool or pre-heat for it, which
    a PID cannot do. It plans on the plant itself, all four regenerator
    nodes, the fuel pipeline and the reversal cycle included, so it needs no
    disturbance estimate: the reduced model the IPOPT oracle carried ran
    0.14 to 0.19 K a step colder than the plant, and that oracle estimated
    the gap online. The pull disturbance is planned on its mean path
    (``plan_params``).
    """
    return NewtonMPC(env, params, horizon=horizon, n_iter=n_iter)
