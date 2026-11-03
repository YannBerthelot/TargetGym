"""The oracle (the MPC slot) for the unstable CSTR.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``): changing the oracle re-records
only the oracle rows of this package's tasks. The shared machinery
(GradientMPC, NewtonMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every oracle fingerprint.

The MPC is ``NewtonMPC`` on the env's own step, guided by the cascade PID.
It reads the true state, the drift and the whole schedule, and plans on the
drift's mean path.
"""

import jax
import jax.numpy as jnp

from target_gym.experts.mpc import NewtonMPC
from target_gym.pc_gym.unstable_cstr.experts import (
    make_unstable_cstr_pid_controller,
    pnr_line,
)

# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------

#: The MPC's horizon, in steps of ``delta_t`` (ours), 1.6 min. From 0.01 K
#: inside the point of no return, full cooling turns the excursion at most 21
#: steps later, at a +6 K drift (derived, scripts/unstable_cstr_numbers.py
#: --section pnr). 32 steps is 1.52 times that, so a plan that recovers from
#: near the edge runs past the peak.
MPC_HORIZON = 32
#: The MPC's barrier on T past ``pnr_line(C_a)``: quadratic over its first
#: ``MPC_PNR_ROUNDING_K`` (K, ours), then ``MPC_PNR_PENALTY`` per K in reward
#: units (ours), the tracking cost of a 0.3145 mol/L error. The planner meets
#: the drift's mean where the plant meets a noisy one; without the barrier it
#: plans recoverable paths past the line that a +3 K drift makes a runaway
#: (one trip on the ten recorded seeds), and with it none trips on thirty.
MPC_PNR_ROUNDING_K = 1.0
MPC_PNR_PENALTY = 9.891e6


def unstable_cstr_stage(env, state, action, params):
    """``NewtonMPC``'s stage: the env's step and reward in floor units, and
    the barrier on T past the point-of-no-return line."""
    _, new, reward, _, _ = env.step_env(jax.random.PRNGKey(0), state, action, params)
    past = jnp.maximum(new.T - pnr_line(new.C_a), 0.0)
    d = MPC_PNR_ROUNDING_K
    barrier = jnp.where(past < d, past * past / (2.0 * d), past - d / 2.0)
    return new, (MPC_PNR_PENALTY * barrier - reward) / params.rho_floor_tracking


def make_unstable_cstr_mpc(env, params, horizon: int = MPC_HORIZON, n_iter=10):
    """Registry factory: ``NewtonMPC`` on the env's own step,
    ``unstable_cstr_stage``, guided by the cascade PID.

    Each solve starts from the better of the previous plan, replayed with its
    feedback, and the cascade's rollout from the true state, so the plan is
    never worse than the expert's under the planner's model. The cascade is
    built from the stored gains and held to the PID factory's stability
    guard, so gains the guard refuses stop this factory too.
    """
    return NewtonMPC(
        env,
        params,
        horizon=horizon,
        n_iter=n_iter,
        guide=make_unstable_cstr_pid_controller(),
        stage_fn=unstable_cstr_stage,
    )
