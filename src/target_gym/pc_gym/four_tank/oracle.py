"""The oracle (the MPC slot) for the four-tank task.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``): changing the oracle re-records
only the oracle rows of this package's tasks. The shared machinery
(GradientMPC, NewtonMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every oracle fingerprint.
"""

from target_gym.experts.mpc import NewtonMPC, protocol_burn_in, window_weights

# ---------------------------------------------------------------------------
# FourTank
# ---------------------------------------------------------------------------


def make_four_tank_mpc(
    env,
    params,
    window_start: int | None = None,
    resolve_every: int = 25,
    pre_weight: float = 1e-3,
    horizon: int | None = None,
    n_iter: int = 20,
    n_iter_cold: int = 60,
):
    """The four-tank's oracle: ``NewtonMPC`` over the rest of the episode.

    It re-solves every 25 steps on the env's own step and reward, both lower
    tanks' tracking cost in floor units. A level at a trip limit is charged
    the trip in the planner's own rollout, which no plan here comes near.
    ``window_start`` is where the protocol starts scoring, read by default
    from its burn-in (``protocol_burn_in``: 270 steps of the registered
    1200), and the steps before it weigh ``pre_weight``, so the window weighs
    1000 times more (``window_weights``). That weighting is what matters
    here: the plant is non-minimum-phase (``gamma1 + gamma2`` = 0.4), and the
    exact optimum of an unweighted all-step cost leaves a tail in the scored
    window. The episode's first solve starts cold and runs ``n_iter_cold``
    iterations; later ones start from the previous plan.
    """
    if window_start is None:
        window_start = protocol_burn_in("four_tank", params)
    steps = int(params.max_steps_in_episode)
    horizon = steps if horizon is None else int(horizon)
    return NewtonMPC(
        env,
        params,
        horizon=horizon,
        n_iter=n_iter,
        n_iter_cold=n_iter_cold,
        action_dim=2,
        weights=window_weights(steps, window_start, pre_weight, horizon),
        resolve_every=resolve_every,
    )
