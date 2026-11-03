"""The expert (the PID slot) for the HVAC task.

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).
"""

import jax.numpy as jnp
import numpy as np
from flax import struct

from target_gym.experts.pid import Controller, pid_update

# ---------------------------------------------------------------------------
# The expert: ASHRAE Guideline 36 zone heating with optimal start
# ---------------------------------------------------------------------------


@struct.dataclass
class G36Params:
    """The published constants, and the schedule a BMS is programmed with.

    The loop is G36's zone heating PI as LBL's Modelica Buildings library
    implements it (``G36/ThermalZones/ControlLoops.mo``: ``kHeaCon = 0.1``
    per K, ``TiHeaCon = 900`` s, output 0-1). Optimal start is LBL's
    ``OBC/Utilities/OptimalStart`` with its defaults: ``tOptMax`` 3 h, three
    days of slopes, ``uLow`` 0 K and ``uHigh`` 0.5 K, ``thrOptOn`` 60 s, and
    its floors of 1 K/h on the mean slope and 120 s on a learning interval.
    """

    kp: float = 0.1
    Ti: float = 900.0
    dt: float = 900.0
    occupied_start_h: float = 7.0
    occupied_end_h: float = 22.0
    setpoint_design: float = 21.0
    t_opt_max: float = 10800.0
    n_day: int = struct.field(pytree_node=False, default=3)
    u_low: float = 0.0
    u_high: float = 0.5
    thr_opt_on: float = 60.0
    slope_min: float = 1.0 / 3600.0
    t_learn_min: float = 120.0


@struct.dataclass
class G36State:
    """The PI's integral and optimal start's memory."""

    integral: float = 0.0
    #: The occupied setpoint last seen, the warm-up's aim.
    setpoint_occupied: float = 21.0
    #: The last ``n_day`` learned slopes (K/s) and how many there are.
    slopes: tuple = (0.0, 0.0, 0.0)
    n_slopes: int = 0
    #: This morning's warm-up time (s), set ``t_opt_max`` before occupancy.
    t_opt: float = 0.0
    #: A warm-up being timed: 1 while learning, its start deficit and age.
    learning: float = 0.0
    dT_start: float = 0.0
    t_learning: float = 0.0
    #: A slope learned today, pushed at midnight (``has_pending`` 1).
    pending: float = 0.0
    has_pending: float = 0.0
    #: The previous step's hour and time to occupancy, to see crossings.
    prev_hour: float = 0.0
    prev_to_occ: float = 1e9
    prev_opt_on: float = 0.0


def g36_step(params, state, obs, xp=jnp):
    """One step of the G36 zone heating loop with optimal start.

    obs: ``[T_air, T_out, heat_pct, solar_norm, sin_h, cos_h, target_T]``.
    The clock is read from ``sin_h, cos_h``, as a BMS knows the time.
    """
    p = params
    T_air, target = obs[..., 0], obs[..., 6]
    hour = xp.mod(xp.arctan2(obs[..., 4], obs[..., 5]) * 24.0 / (2.0 * np.pi), 24.0)
    occupied = (hour >= p.occupied_start_h) & (hour < p.occupied_end_h)
    to_occ = xp.mod(p.occupied_start_h - hour, 24.0) * 3600.0
    sp_occ = xp.where(occupied, target, state.setpoint_occupied)

    # Midnight: push the slope learned that morning into the moving mean.
    midnight = (hour < state.prev_hour) & (state.has_pending > 0.5)
    slopes = xp.asarray(state.slopes)
    pushed = xp.concatenate([slopes[1:], xp.asarray(state.pending)[..., None]], -1)
    slopes = xp.where(midnight, pushed, slopes)
    n_slopes = xp.where(
        midnight, xp.minimum(state.n_slopes + 1, p.n_day), state.n_slopes
    )
    has_pending = xp.where(midnight, 0.0, state.has_pending)

    # ``t_opt_max`` before occupancy: sample the zone, set the warm-up time.
    sample = (~occupied) & (to_occ <= p.t_opt_max) & (state.prev_to_occ > p.t_opt_max)
    # The moving mean over the slopes learned so far (newest last); none
    # learned yet reads 0, which the floor turns into 1 K/h.
    learned = xp.arange(p.n_day) >= p.n_day - n_slopes
    mean = xp.sum(xp.where(learned, slopes, 0.0), -1) / xp.maximum(n_slopes, 1)
    slope = xp.maximum(mean, p.slope_min)
    deficit = sp_occ - T_air
    t_new = xp.where(deficit > p.u_high, xp.minimum(deficit / slope, p.t_opt_max), 0.0)
    t_opt = xp.where(sample, t_new, state.t_opt)
    opt_on = (~occupied) & (t_opt > p.thr_opt_on) & (to_occ < t_opt)

    # Learning: time each warm-up from its start until the zone reaches the
    # occupied setpoint (or ``t_opt_max`` has passed).
    # It starts when the warm-up does, on the rising edge of ``opt_on``.
    start = opt_on & (state.prev_opt_on < 0.5) & (state.learning < 0.5)
    learning = xp.where(start, 1.0, state.learning)
    dT_start = xp.where(start, deficit, state.dT_start)
    t_learning = xp.where(start, 0.0, state.t_learning + p.dt)
    done = (learning > 0.5) & (
        (T_air >= sp_occ - p.u_low) | (t_learning >= p.t_opt_max)
    )
    pending = xp.where(
        done, dT_start / xp.maximum(t_learning, p.t_learn_min), state.pending
    )
    has_pending = xp.where(done, 1.0, has_pending)
    learning = xp.where(done, 0.0, learning)

    # The heating PI on the setpoint in force.
    setpoint = xp.where(opt_on, sp_occ, target)
    e = setpoint - T_air
    heat, integral, _ = pid_update(
        e, state.integral, 0.0, p.kp, p.kp / p.Ti, 0.0, p.dt, 0.0, 1.0, xp
    )
    new_state = G36State(
        integral=integral,
        setpoint_occupied=sp_occ,
        slopes=slopes,
        n_slopes=n_slopes,
        t_opt=t_opt,
        learning=learning,
        dT_start=dT_start,
        t_learning=t_learning,
        pending=pending,
        has_pending=has_pending,
        prev_hour=hour,
        prev_to_occ=to_occ,
        prev_opt_on=xp.where(opt_on, 1.0, 0.0),
    )
    return xp.stack([2.0 * heat - 1.0], axis=-1), new_state


def make_hvac_pid_controller() -> Controller:
    """The expert: G36 zone heating with optimal start, as a
    :class:`~target_gym.experts.pid.Controller`. Every constant is the
    published one (:class:`G36Params`); nothing is tuned on seeds."""
    from target_gym.hvac.env import HVACParams

    pr = HVACParams()
    params = G36Params(
        dt=float(pr.delta_t),
        occupied_start_h=float(pr.occupied_start_h),
        occupied_end_h=float(pr.occupied_end_h),
        setpoint_design=float(pr.setpoint_occupied),
    )
    state = G36State(
        setpoint_occupied=params.setpoint_design,
        slopes=np.zeros(params.n_day, np.float32),
    )
    return Controller(params, state, g36_step)
