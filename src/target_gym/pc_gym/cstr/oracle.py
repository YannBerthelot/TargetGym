"""The oracle (the MPC slot) for the CSTR task.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``): changing the oracle re-records
only the oracle rows of this package's tasks. The shared machinery
(GradientMPC, NewtonMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every oracle fingerprint.
"""

from target_gym.experts.mpc import NewtonMPC, protocol_burn_in, window_weights

# ---------------------------------------------------------------------------
# CSTR
# ---------------------------------------------------------------------------


def make_cstr_mpc(
    env,
    params,
    window_start: int | None = None,
    resolve_every: int = 1,
    pre_weight: float = 1e-3,
    horizon: int | None = None,
    n_iter: int = 20,
):
    """The CSTR's oracle: ``NewtonMPC`` over the rest of the episode.

    It re-solves every step on the env's own step and reward, the
    concentration's tracking cost in floor units. ``window_start`` is where
    the protocol starts scoring, read by default from its burn-in
    (``protocol_burn_in``: 12 steps of the registered 150), and the steps
    before it weigh ``pre_weight`` (``window_weights``). The plant is
    deterministic, so the closed loop reaches the plant's optimum to float32
    rounding: the returns of the IPOPT shrinking-horizon NLP it replaced, to
    the unit, at a quarter of its time.
    """
    if window_start is None:
        window_start = protocol_burn_in("cstr", params)
    steps = int(params.max_steps_in_episode)
    horizon = steps if horizon is None else int(horizon)
    return NewtonMPC(
        env,
        params,
        horizon=horizon,
        n_iter=n_iter,
        weights=window_weights(steps, window_start, pre_weight, horizon),
        resolve_every=resolve_every,
    )
