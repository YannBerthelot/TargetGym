"""Reference controller (the MPC slot) for the reactor task.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
NewtonMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from target_gym import reward as R
from target_gym.experts.mpc import NewtonMPC
from target_gym.experts.pid import Controller
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


# ---------------------------------------------------------------------------
# The expert: Westinghouse rod control on average temperature
# ---------------------------------------------------------------------------


@struct.dataclass
class RodControlParams:
    """The Westinghouse rod control program and the plant's heat balance.

    The program is the NRC's (Westinghouse Technology training manual,
    HRTD 8.1 and 5.0): a +-1.5 F deadband with a 1.0 F lockup, 8 steps/min to
    3 F, 32 steps/min per F to 72 steps/min at 5 F; the power mismatch through
    a 40 s rate/lag, a nonlinear gain (0.3 F per % to 1 %, 1.5 F per %
    beyond) and a variable gain (2 to 50 % load, 1 / load above). The heat
    balance gives the Tavg program: the cold leg is the inlet, so Tavg rises
    ``P / (2 m cp)`` from no load to full power. Ours: a speed in steps/min
    moves the rods over their range as a 228-step stroke would.
    """

    T_inlet: float
    dT_full_load: float
    dt: float = 10.0
    deadband_F: float = 1.5
    lockup_F: float = 1.0
    tau_rate: float = 40.0
    stroke_steps: float = 228.0


@struct.dataclass
class RodControlState:
    """The rate/lag's lagged mismatch (% power), whether the rods are moving,
    and whether the lag has seen a first observation."""

    mismatch_lag: float = 0.0
    moving: float = 0.0
    started: float = 0.0


def _mismatch_gain(m, xp):
    """The nonlinear gain unit: deg F for a mismatch rate of ``m`` % power."""
    a = xp.abs(m)
    return xp.sign(m) * xp.where(a <= 1.0, 0.3 * a, 0.3 + 1.5 * (a - 1.0))


def rod_control_step(params, state, obs, xp=jnp):
    """One 10 s step of Westinghouse rod control.

    obs: ``[n, T_coolant, rho_ext_norm, target_n]``; the action is the
    desired rod position on the same [-1, 1] scale.
    """
    p = params
    n, T_hot, rod, load = obs[..., 0], obs[..., 1], obs[..., 2], obs[..., 3]

    # Temperature channel: Tref (programmed on load) minus Tavg, in deg F.
    T_ref = p.T_inlet + p.dT_full_load * load
    T_avg = 0.5 * (p.T_inlet + T_hot)
    error = 1.8 * (T_ref - T_avg)

    # Power-mismatch channel: tau s / (1 + tau s) on (turbine - nuclear), the
    # lag sampled exactly at the 10 s step and started on the first reading.
    x = 100.0 * (load - n)
    alpha = np.exp(-p.dt / p.tau_rate)
    lagged = xp.where(
        state.started > 0.5, alpha * state.mismatch_lag + (1 - alpha) * x, x
    )
    variable = xp.where(load <= 0.5, 2.0, 1.0 / xp.maximum(load, 1e-6))
    error = error + variable * _mismatch_gain(x - lagged, xp)

    # Rod speed program, with the lockup holding moving rods to 1.0 F.
    size = xp.abs(error)
    moving = xp.where(state.moving > 0.5, size > p.lockup_F, size >= p.deadband_F)
    spm = xp.where(moving, xp.clip(8.0 + 32.0 * (size - 3.0), 8.0, 72.0), 0.0)
    move = xp.sign(error) * spm / 60.0 * p.dt / p.stroke_steps * 2.0
    action = xp.clip(rod + move, -1.0, 1.0)
    new_state = RodControlState(
        mismatch_lag=lagged, moving=xp.where(moving, 1.0, 0.0), started=1.0 + 0.0 * x
    )
    return xp.stack([action], axis=-1), new_state


def make_reactor_pid_controller() -> Controller:
    """The expert: Westinghouse rod control as a
    :class:`~target_gym.experts.pid.Controller`. The program is published
    (:class:`RodControlParams`); nothing is tuned on seeds."""
    from target_gym.reactor.env import ReactorParams

    pr = ReactorParams()
    params = RodControlParams(
        T_inlet=float(pr.T_inlet),
        dT_full_load=float(pr.P_thermal_ref) / (2.0 * float(pr.m_dot_cp)),
    )
    return Controller(params, RodControlState(), rod_control_step)
