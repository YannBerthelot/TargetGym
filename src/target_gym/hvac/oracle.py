"""The oracle (the MPC slot) for the HVAC task.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``): changing the oracle re-records
only the oracle rows of this package's tasks. The shared machinery
(GradientMPC, NewtonMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every oracle fingerprint.
"""

from target_gym.experts.mpc import NewtonMPC

# ---------------------------------------------------------------------------
# Building HVAC
# ---------------------------------------------------------------------------


def make_hvac_mpc(env, params, horizon: int = 24, n_iter: int = 10):
    """The oracle: ``NewtonMPC`` on the building's own step and reward.

    With delta_t = 900 s, horizon = 24 gives 6 h of lookahead -- enough to see
    the morning setback recovery and the solar peak, which is where the
    anticipation advantage over PID comes from. It reads the true state, the
    thermal mass and the surfaces included, and plans the weather's mean
    path, the env's own OU update with ``T_out_noise_std`` zeroed
    (``plan_params``).
    """
    return NewtonMPC(env, params, horizon=horizon, n_iter=n_iter)
