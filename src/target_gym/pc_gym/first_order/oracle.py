"""The oracle (the MPC slot) for the first-order task.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``): changing the oracle re-records
only the oracle rows of this package's tasks. The shared machinery
(GradientMPC, NewtonMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every oracle fingerprint.
"""

from target_gym.experts.mpc import NewtonMPC


def make_first_order_mpc(env, params, horizon: int = 5, n_iter: int = 10):
    """The oracle: ``NewtonMPC`` on the plant's own step and reward, over
    PC-gym's five-step horizon (0.25 s)."""
    return NewtonMPC(env, params, horizon=horizon, n_iter=n_iter)
