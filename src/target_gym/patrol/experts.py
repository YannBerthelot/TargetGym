"""The patrol tasks' expert (the PID slot).

The expert is a formation hold over the follower's own autopilot
(:func:`make_patrol_pid_controller`), as flight-tested formation autopilots
fly it; the range-and-bearing task's estimates the lead first
(:func:`make_patrol_range_bearing_pid_controller`).

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).
"""

import jax
import jax.numpy as jnp
from flax import struct

from target_gym.experts.pid import (
    DERIVATIVE_FILTER_RATIO,
    Controller,
    _wrap_angle,
    filtered_derivative,
)
from target_gym.plane.dynamics import BANK_DEMAND_MAX, PITCH_DEMAND_MAX
from target_gym.plane.experts_tecs import AIRLINER, TECSParams
from target_gym.plane3d import hornet
from target_gym.plane3d.experts import (
    Plane3DGuidanceParams,
    Plane3DGuidanceState,
    _fly,
)

# ---------------------------------------------------------------------------
# The PID slot: a formation hold over the follower's own autopilot
# ---------------------------------------------------------------------------

# Indices into the full patrol observation (patrol.env.get_obs_full).
_PSI, _E_BACK, _E_RIGHT, _E_UP = 9, 10, 11, 12
_RV_FWD, _RV_RIGHT, _SLOT_BACK, _SLOT_RIGHT = 13, 14, 16, 17
_RV_UP, _POWER = 15, 21
_REL_HEADING = 19
# Indices into the range-and-bearing observation; 0-9 are as above.
_RANGE, _AZ, _EL = 10, 11, 12
_RB_SLOT_BACK, _RB_SLOT_RIGHT, _RB_SLOT_UP = 13, 14, 15
_RB_POWER = 16


#: The Hornet's TECS: AIRLINER's gains with the Hornet's own flight test at the
#: AFF point, full military power climbing 25 m/s and idle sinking 15 m/s
#: (the model's excess thrust, plane3d/hornet.py).
HORNET_TECS = AIRLINER.replace(climb_max=25.0, sink_min=15.0)


@struct.dataclass
class HornetGuidanceParams(Plane3DGuidanceParams):
    """The 3D autopilot on the Hornet: AFF's 250 KCAS [AFF], a speed floor of
    1.2 times its 61.9 m/s clean stall, and its TECS."""

    cruise_cas: float = 250.0 * 1852.0 / 3600.0
    cruise_mach: float = 0.70
    v_min_eas: float = 74.0
    tecs: TECSParams = HORNET_TECS


@struct.dataclass
class FormationHoldParams(HornetGuidanceParams):
    """A formation hold's gains, over the 3D tasks' autopilot.

    The hold is the one formation-flight research flew (AFIT's formation
    hold, NASA's F/A-18 Autonomous Formation Flight trials): the
    along-track error sets a speed demand, the cross-track error a course,
    the vertical error a height, each a PI on top of the lead's own speed,
    heading and turn rate fed forward. The autopilot underneath is
    :func:`target_gym.plane3d.experts._fly` (TECS, a pitch loop, a damped
    45 deg bank loop and stall prevention), the one the 3D tasks' experts fly.

    Along track the closure is ``min(k_x |e|, sqrt(2 a_x |e|))``, the speed
    from which ``a_x`` of thrust margin stops at the slot, plus ``k_xi`` of
    the error's integral. Across track the course is the slot's velocity plus
    ``k_y`` of the error and ``k_yi`` of its integral, at most
    ``max_intercept`` off it. Both integrals run only within ``x_band`` and
    ``y_band`` of the slot, and to at most ``int_max``. ``k_omega`` filters
    the lead's turn rate, ``k_ref`` the demands' rates the autopilot is fed
    forward, and ``k_eng`` closes the throttle on the engine's power state.
    ``t_rate`` is the filter time on the range-and-bearing estimator's lead
    velocity, 0 for the raw difference.
    """

    # Tuned on training seeds 100-109, one set for every follower (both
    # patrol tasks and the formation's wingmen); the step of
    # scripts/tune_formation_hold.py that searches each is in brackets.
    k_x: float = 0.08  # 1/s [slot loops]
    a_x: float = 0.4  # m/s^2 [slot loops]
    k_xi: float = 0.001  # 1/s^2 [integrals and filter]
    x_band: float = 30.0  # m [integrals and filter, with y_band]
    k_y: float = 0.1  # 1/s [slot loops]
    k_yi: float = 0.001  # 1/s^2 [integrals and filter]
    y_band: float = 30.0  # m [integrals and filter, with x_band]
    k_omega: float = 0.3  # per step [integrals and filter]
    k_ref: float = 0.3  # per step [engine and energy]
    # Provisional: lowest mean RMS slot error of 15, 20, 30, 40 and 60 s on the
    # noisy training seeds, ahead of the expert protocol on noisy plants.
    t_rate: float = 20.0  # s
    # Set, not tuned: limits for a join from far out, beyond the 40 m spawn
    # noise. The closure reaches 20 m/s 500 m out, about 9% of cruise. A correction of 0.5 rad off the
    # slot's track, inside L1's 45 deg capture limit, binds 1.3 km out, short
    # of the 1.5 km slot loss. The integrals' authority is held to 2 m/s
    # (k_xi or k_yi times int_max), one step of TECS's speed slew.
    dv_max: float = 20.0  # m/s
    max_intercept: float = 0.5  # rad
    int_max: float = 2000.0  # m s
    # ---- The fighter's direct hold (``law`` 1, ``_direct_hold``) ----
    #: 0, the guidance cascade above over TECS (the airliner's, ``wake_surf``);
    #: 1, NASA AFF's structure on the Hornet: each axis a PID on the slot
    #: error, its rate from the relative velocity, straight to the demand.
    law: float = 0.0
    #: Placed from the plant's own response at the AFF point, a damping of 0.8
    #: on every axis. Along track at 0.3 rad/s: a unit of power gives 36.9 kN
    #: / 15.1 t = 2.44 m/s^2, so kp = 0.09 / 2.44 per m and kd = 0.48 / 2.44
    #: per m/s. Across at 0.4 rad/s, the bank loop being faster than the
    #: engine: a radian of bank gives g, so a lateral demand of 0.16 per m and
    #: 0.64 per m/s. Vertically a climb demand of 0.3 per m. The integrals at
    #: a tenth of the loop's rate. The engine's 2.6 s and the bank loop's 1 s
    #: keep each loop's third-order Routh condition (lag x kp < kd) with room.
    #: Checked on training seeds 100-103 [slot loops]: along track at 0.45
    #: and across at 0.5 each diverge on a seed, 0.3 and 0.4 hold every one.
    kp_x: float = 0.037  # power per m
    kd_x: float = 0.197  # power per m/s
    ki_x: float = 0.0011  # power per m s
    kp_y: float = 0.16  # m/s^2 per m
    kd_y: float = 0.64  # m/s^2 per m/s
    ki_y: float = 0.0064  # m/s^2 per m s
    kp_z: float = 0.3  # m/s per m
    ki_z: float = 0.009  # m/s per m s
    #: The largest climb or descent the vertical loop asks, m/s.
    climb_max: float = 10.0


@struct.dataclass
class FormationHoldState(Plane3DGuidanceState):
    """The autopilot's memory, the two integrals, the lead's last heading and
    filtered turn rate, the last demands and their filtered rates, and for
    the range-and-bearing estimator the filtered relative position and the last own
    velocity (``started`` is set from the second step on)."""

    started: float = 0.0
    int_back: float = 0.0
    int_right: float = 0.0
    psi_lead_prev: float = 0.0
    omega: float = 0.0
    h_dem_prev: float = 0.0
    v_dem_prev: float = 0.0
    h_rate: float = 0.0
    v_rate: float = 0.0
    dx_prev: float = 0.0
    dy_prev: float = 0.0
    vx_prev: float = 0.0
    vy_prev: float = 0.0
    int_up: float = 0.0


def _lead_axes(psi_lead, xp):
    """The lead's forward and right unit vectors in the world x-y plane."""
    fwd = xp.stack([xp.cos(psi_lead), xp.sin(psi_lead)])
    right = xp.stack([xp.sin(psi_lead), -xp.cos(psi_lead)])
    return fwd, right


def _hold(p, s, obs, xp, slot_err, v_lead, psi_lead, slot, **carry):
    """The formation hold on the slot error ``(back, right, up)``, given the
    lead's velocity and heading and the slot's ``(back, right)`` offsets.

    In a turn at rate w the slot swings with the lead, so its velocity is the
    lead's plus ``w * right_offset`` along the lead's track and ``w * back``
    across it. The demands go to the autopilot exactly: TECS's slewed height
    and speed references are set each step so that its slew lands on them,
    with only their rates, filtered by ``k_ref``, fed forward. Unfiltered,
    every step-to-step wobble of the lead's track reaches the pitch loop as a
    rate demand.
    """
    e_back, e_right, e_up = slot_err
    started = s.started > 0.0
    omega_now = _wrap_angle(psi_lead - s.psi_lead_prev, xp) / p.dt
    omega = xp.where(started, s.omega + p.k_omega * (omega_now - s.omega), 0.0)
    fwd, right = _lead_axes(psi_lead, xp)
    v_slot = v_lead + omega * slot[1] * fwd + omega * slot[0] * right
    slot_fwd, slot_right = v_slot @ fwd, v_slot @ right

    int_back = xp.where(
        xp.abs(e_back) < p.x_band, s.int_back + e_back * p.dt, s.int_back
    )
    int_right = xp.where(
        xp.abs(e_right) < p.y_band, s.int_right + e_right * p.dt, s.int_right
    )
    int_back = xp.clip(int_back, -p.int_max, p.int_max)
    int_right = xp.clip(int_right, -p.int_max, p.int_max)
    closure = xp.minimum(p.k_x * xp.abs(e_back), xp.sqrt(2.0 * p.a_x * xp.abs(e_back)))
    speed_fwd = slot_fwd + xp.clip(
        xp.sign(e_back) * closure + p.k_xi * int_back, -p.dv_max, p.dv_max
    )
    limit = speed_fwd * xp.tan(p.max_intercept)
    speed_right = slot_right - xp.clip(
        p.k_y * e_right + p.k_yi * int_right, -limit, limit
    )
    course = xp.arctan2(
        speed_fwd * fwd[1] + speed_right * right[1],
        speed_fwd * fwd[0] + speed_right * right[0],
    )
    track = xp.arctan2(obs[1], obs[0])
    v_ground = xp.sqrt(obs[0] ** 2 + obs[1] ** 2 + 1e-6)
    turn = omega + _wrap_angle(course - track, xp) / p.tau_hdg

    v_dem = xp.sqrt(speed_fwd**2 + speed_right**2)
    h_dem = obs[2] - e_up
    t = p.tecs
    h_rate = xp.where(
        started, s.h_rate + p.k_ref * ((h_dem - s.h_dem_prev) / p.dt - s.h_rate), 0.0
    )
    v_rate = xp.where(
        started, s.v_rate + p.k_ref * ((v_dem - s.v_dem_prev) / p.dt - s.v_rate), 0.0
    )
    h_from = h_dem - xp.clip(h_rate * t.dt, -t.sink_max * t.dt, t.climb_max * t.dt)
    v_from = v_dem - xp.clip(v_rate * t.dt, -t.accel_max * t.dt, t.accel_max * t.dt)
    tecs = s.tecs.replace(
        h_ref=xp.where(started, h_from, s.tecs.h_ref),
        v_ref=xp.where(started, v_from, s.tecs.v_ref),
    )
    return _fly(
        p,
        s.replace(tecs=tecs),
        obs,
        xp,
        h_dem,
        v_ground * turn,
        v_dem,
        int_back=int_back,
        int_right=int_right,
        psi_lead_prev=psi_lead,
        omega=omega,
        h_dem_prev=h_dem,
        v_dem_prev=v_dem,
        h_rate=h_rate,
        v_rate=v_rate,
        started=xp.ones_like(e_up),
        **carry,
    )


def _direct_hold(p, s, obs, xp, slot_err, v_lead, psi_lead, slot, power, up_rate):
    """NASA AFF's formation autopilot on the Hornet (NASA/TM-2002-210729,
    figure 3): on each axis a PID on the slot error, its rate the relative
    velocity, straight to the demand the inner loop holds. Along track the
    throttle, its integral the power trim (taken up bumpless from ``power``,
    the engine's reading, on the first step); vertically the flight path, a
    climb of ``kp_z`` per metre about the lead's level, with the extra pitch
    of a banked turn (``k_bank_comp`` per g); across track the bank
    of the lateral acceleration the lead's turn needs plus the correction.
    A positive bank turns toward increasing heading, to the left."""
    e_back, e_right, e_up = slot_err
    started = s.started > 0.0
    dt = p.dt
    omega_now = _wrap_angle(psi_lead - s.psi_lead_prev, xp) / dt
    omega = xp.where(started, s.omega + p.k_omega * (omega_now - s.omega), 0.0)
    fwd, right = _lead_axes(psi_lead, xp)
    v_slot = v_lead + omega * slot[1] * fwd + omega * slot[0] * right
    dv = obs[0:2] - v_slot
    back_rate, right_rate = -(dv @ fwd), dv @ right

    trim = xp.where(started, s.int_back + p.ki_x * e_back * dt, power)
    trim = xp.clip(trim, 0.0, 1.0)
    throttle = xp.clip(trim + p.kp_x * e_back + p.kd_x * back_rate, 0.0, 1.0)

    speed = xp.sqrt(obs[0] ** 2 + obs[1] ** 2 + obs[3] ** 2 + 1e-6)
    int_up = xp.where(started, s.int_up + e_up * dt, 0.0)
    int_up = xp.clip(int_up, -p.int_max, p.int_max)
    climb = xp.clip(-(p.kp_z * e_up + p.ki_z * int_up), -p.climb_max, p.climb_max)
    gamma = obs[8]
    pitch = obs[4] + (xp.arcsin(xp.clip(climb / speed, -0.5, 0.5)) - gamma)
    # The extra pitch a banked turn needs, as the 3D autopilot adds it.
    load_factor = 1.0 / xp.maximum(xp.cos(obs[6]), 0.35)
    pitch = pitch + p.k_bank_comp * (load_factor - 1.0)
    del up_rate  # the climb demand is the rate loop; kept for the estimator's form

    int_right = xp.where(started, s.int_right + e_right * dt, 0.0)
    int_right = xp.clip(int_right, -p.int_max, p.int_max)
    lateral = speed * omega + p.kp_y * e_right + p.kd_y * right_rate
    lateral = lateral + p.ki_y * int_right
    bank = xp.clip(xp.arctan(lateral / p.gravity), -p.max_bank_rad, p.max_bank_rad)
    action = xp.stack(
        [2.0 * throttle - 1.0, pitch / PITCH_DEMAND_MAX, bank / BANK_DEMAND_MAX]
    )
    new = s.replace(
        started=xp.ones_like(e_up),
        int_back=trim,
        int_right=int_right,
        int_up=int_up,
        psi_lead_prev=psi_lead,
        omega=omega,
    )
    return action, new


def _either_hold(
    p, s, obs, xp, slot_err, v_lead, psi_lead, slot, power, up_rate, **carry
):
    """``law`` 1's direct hold, or the guidance cascade (``_hold``)."""
    direct, direct_state = _direct_hold(
        p, s, obs, xp, slot_err, v_lead, psi_lead, slot, power, up_rate
    )
    direct_state = direct_state.replace(**carry)
    cascade, cascade_state = _hold(
        p, s, obs, xp, slot_err, v_lead, psi_lead, slot, **carry
    )
    pick = p.law > 0.5
    action = xp.where(pick, direct, cascade)
    state = jax.tree_util.tree_map(
        lambda a, b: xp.where(pick, a, b), direct_state, cascade_state
    )
    return action, state


def patrol_hold_step(params, state, obs, xp=jnp):
    """The full-observation hold: the slot error, the lead's heading and its
    velocity (own velocity less the relative one) are all observed."""
    psi_lead = obs[_PSI] + obs[_REL_HEADING]
    fwd, right = _lead_axes(psi_lead, xp)
    v_lead = obs[0:2] - obs[_RV_FWD] * fwd - obs[_RV_RIGHT] * right
    return _either_hold(
        params,
        state,
        obs,
        xp,
        (obs[_E_BACK], obs[_E_RIGHT], obs[_E_UP]),
        v_lead,
        psi_lead,
        (obs[_SLOT_BACK], obs[_SLOT_RIGHT]),
        obs[_POWER],
        obs[_RV_UP],
    )


def patrol_range_bearing_hold_step(params, state, obs, xp=jnp):
    """The same hold behind an estimator of the lead from range and bearings.

    Range, azimuth and elevation fix where the lead is; its velocity is the
    filtered rate of that position, taken with the follower's own motion over
    the last step added back. Both aircraft fly in much the same air, so the lead's heading is its
    track less the crab the follower itself flies at. The slot error follows
    from the relative position in that lead frame.
    """
    psi = obs[_PSI]
    rng, az, el = obs[_RANGE], obs[_AZ], obs[_EL]
    horiz = rng * xp.cos(el)
    ahead, aside = horiz * xp.cos(az), horiz * xp.sin(az)
    # Lead minus follower in the world frame.
    dx = ahead * xp.cos(psi) + aside * xp.sin(psi)
    dy = ahead * xp.sin(psi) - aside * xp.cos(psi)
    dz = rng * xp.sin(el)
    # The lead's velocity is the filtered rate of its position, which the
    # follower knows relative to itself, so the filter's memory slides back
    # with the follower. It starts as if the lead flew the follower's velocity.
    started = state.started > 0.0
    v_own = obs[0:2]
    v_prev = xp.where(started, xp.stack([state.vx_prev, state.vy_prev]), v_own)
    v_mean, dt = 0.5 * (v_own + v_prev), params.dt
    slide = xp.where(started, dt, params.t_rate + dt)
    memory = xp.where(
        started, xp.stack([state.dx_prev, state.dy_prev]), xp.stack([dx, dy])
    )
    Kd_rate = DERIVATIVE_FILTER_RATIO * params.t_rate
    v_lead, filtered = filtered_derivative(
        xp.stack([dx, dy]), memory - slide * v_mean, 1.0, Kd_rate, dt
    )
    crab = _wrap_angle(xp.arctan2(obs[1], obs[0]) - psi, xp)
    psi_lead = xp.arctan2(v_lead[1], v_lead[0]) - crab
    fwd, right = _lead_axes(psi_lead, xp)
    slot_back, slot_right = obs[_RB_SLOT_BACK], obs[_RB_SLOT_RIGHT]
    slot_err = (
        dx * fwd[0] + dy * fwd[1] - slot_back,
        -(dx * right[0] + dy * right[1]) - slot_right,
        -dz - obs[_RB_SLOT_UP],
    )
    return _either_hold(
        params,
        state,
        obs,
        xp,
        slot_err,
        v_lead,
        psi_lead,
        (slot_back, slot_right),
        obs[_RB_POWER],
        obs[3],
        dx_prev=filtered[0],
        dy_prev=filtered[1],
        vx_prev=obs[0],
        vy_prev=obs[1],
    )


#: The hold's tuning, on training seeds 100-109 of every follower, checked
#: on held-out seeds 200-209, never on the scored seeds 0-9
#: (scripts/tune_formation_hold.py). The course hold is tighter than the
#: heading task's (3 s against 10 s, [slot loops]), since the slot moves
#: with the lead. Every other autopilot value is the 3D tasks'
#: (scripts/tune_aircraft_guidance.py). TECS weighs height and speed evenly in the pitch channel
#: (``spd_weight`` 0.5, between the airliner's 0, which leaves the speed to
#: the throttle alone, and ArduPilot's default 1) [engine and energy].
HOLD = FormationHoldParams(
    tau_hdg=3.0,
    tecs=HORNET_TECS.replace(spd_weight=0.5),
    law=1.0,
    # The range from vision, not TCAS's 15 m, so the lead's velocity is the
    # raw difference: the lowest mean RMS slot error of 0, 0.25, 0.5, 1, 2, 3,
    # 5, 10 and 20 s on training seeds 100-103 (2026-11-10).
    t_rate=0.0,
    # The Hornet's own extra pitch per g in a turn (plane3d/hornet.py), in
    # place of the A320's 0.15: a turn's offset halves on training seeds.
    k_bank_comp=hornet.PITCH_PER_G,
)


#: The wake-surfing follower's hold: the formation hold over the A320's own
#: autopilot (the 3D tasks' cruise schedule, speed floor and TECS). Its slot
#: gains are the patrol's, not tuned for a station 3 km in trail.
WAKE_SURF_HOLD = HOLD.replace(
    law=0.0,
    k_bank_comp=0.15,  # the A320's (plane3d.experts)
    cruise_cas=290.0 * 1852.0 / 3600.0,
    cruise_mach=0.78,
    v_min_eas=96.0,
    # Step 1 of scripts/tune_formation_hold.py --wake-surf, on training seeds
    # 100-109 (2026-11-10): the worst seed's RMS 156 m, from 215 m, no trips.
    tecs=AIRLINER.replace(spd_weight=1.0, thr_damp=6.0, time_const=5.0),
    k_ref=1.0,
)


def make_wake_surf_pid_controller() -> Controller:
    """``wake_surf``'s expert: the formation hold on the A320, at its station."""
    return Controller(WAKE_SURF_HOLD, FormationHoldState(), patrol_hold_step)


def make_patrol_pid_controller() -> Controller:
    """The patrol's expert: the formation hold on the full observation."""
    return Controller(HOLD, FormationHoldState(), patrol_hold_step)


def make_patrol_range_bearing_pid_controller() -> Controller:
    """The range-and-bearing patrol's expert: the same hold, behind the estimator."""
    return Controller(HOLD, FormationHoldState(), patrol_range_bearing_hold_step)
