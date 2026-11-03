"""The oracle (the MPC slot) for the reactor task.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``): changing the oracle re-records
only the oracle rows of this package's tasks. The shared machinery
(GradientMPC, NewtonMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every oracle fingerprint.
"""

import jax
import jax.numpy as jnp

from target_gym import reward as R
from target_gym.experts.mpc import NewtonMPC
from target_gym.reactor.env import compute_next_state
from target_gym.reactor.env_jax import CONTROL_PERIOD

# ---------------------------------------------------------------------------
# Reactor
# ---------------------------------------------------------------------------

#: Width of the planner's rounded |.|, as a fraction of the scale of the term
#: it rounds (``e_floor`` for the power error, a thousandth of the rod range
#: for the rods' lag behind their demand). The reward's costs are linear in
#: both, and Newton's method stalls at a kink. On seeds 2, 3 and 7 the
#: episode cost fell from 1225 to 1193 per episode as the width rose from
#: 0.03 to 1, and was flat beyond: an error under the floor is then charged
#: quadratically, which is what the floor says it is worth.
PLANNER_ROUNDING = 1.0


def _rounded_abs(x, width):
    return jnp.sqrt(x * x + width * width) - width


def reactor_stage(env, state, action, params):
    """``NewtonMPC``'s stage: the env's step, sub-step by sub-step, and the
    version-2 reward's two costs summed over them as the env sums them, with
    each |.| rounded by ``PLANNER_ROUNDING``; in floor units. A SCRAM is
    charged the trip, as the env charges it, and the plan keeps the state."""
    p = params
    per_s = p.imbalance_multiple * p.spot_price_per_MWh * p.P_electric_GW / 3.6
    rho_scale = jnp.maximum(jnp.abs(p.rho_ext_min), jnp.abs(p.rho_ext_max))

    def sub_step(carry, _):
        s, cost = carry
        s, _ = compute_next_state(action, s, p, env.integration_method)
        error = _rounded_abs(s.target_n - s.n, PLANNER_ROUNDING * p.e_floor)
        tracking = p.e_floor * R.tracking_cost(
            error, p.e_floor, p.e_tol, p.tracking_exponent
        )
        lag = _rounded_abs(
            s.rho_ext_cmd - s.rho_ext, PLANNER_ROUNDING * 1e-3 * rho_scale
        )
        wear = p.rod_wear_weight * p.e_floor * lag / rho_scale
        return (s, cost + per_s * (tracking + wear)), None

    (new, cost), _ = jax.lax.scan(sub_step, (state, 0.0), None, length=CONTROL_PERIOD)
    new = new.replace(time=state.time + 1)
    tripped = env.is_terminated(new, p)
    cost = cost + jnp.where(tripped, p.restart_steps * p.failure_cost, 0.0)
    return new, cost / p.rho_floor_tracking


def make_reactor_mpc(env, params, horizon: int = 20, n_iter: int = 10):
    """The oracle: ``NewtonMPC`` on the plant's own physics, ``reactor_stage``.

    Horizon 20 env steps, 200 s: several of the slow precursor groups'
    time constants and the rods' full travel at their rate limits. It reads
    the true state, xenon included, and plans the demand's mean path, the
    env's own OU update with ``demand_sigma`` zeroed (``plan_params``).
    """
    return NewtonMPC(
        env, params, horizon=horizon, n_iter=n_iter, stage_fn=reactor_stage
    )
