"""
A lead and two wingmen in a V, every aircraft flown by the learner: the lead
flies a patrol route while each wingman holds its slot on the lead, and the
team shares one cost.

The patrol (``target_gym.patrol``) scripts its lead. Here the lead is flown by
the policy too: it must fly the patrol's route (an altitude and a heading that
turns onto a new rate every leg) while ``num_wingmen`` wingmen (two in the
scored task) hold their slots on it, in a V spread evenly over both sides. Every aircraft is the patrol's
3D airframe (:func:`compute_next_state_3d`), all in one air mass, so one
turbulence gust moves the whole formation.

The task is cooperative: one team cost, the lead's tracking cost plus each
wingman's slot and heading-alignment cost, each in the units of its own
instrument. A crash, a collision between any two aircraft or a wingman out of
formation trips the whole formation, which restarts (``base.failure_kernel``).

The registered environment (``env_jax.Formation``) takes one joint action and
returns the observations concatenated, lead first; ``marl.FormationMARL``
gives each aircraft its own observation and action. Each wingman's observation
is the patrol's, so a patrol policy can fly a wingman seat unchanged.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct

from target_gym import reward as R
from target_gym.base import EnvState
from target_gym.patrol.env import (
    PatrolParams,
    decode_action,
    get_obs_full,
    heading_alignment,
    lead_leg,
    slot_error,
    slot_error_vector,
)
from target_gym.plane.dynamics import advance_gust, step_key
from target_gym.plane3d.env import (
    PlaneState3D,
    compute_next_state_3d,
    get_obs_heading,
    wrap_angle,
)
from target_gym.utils import log_scaled_reward

#: The most wingmen a formation takes (five aircraft).
MAX_WINGMEN = 4

#: Values in the lead's observation before its wingmen's slot errors.
LEAD_OBS_BASE = 16
#: Values in each wingman's observation (``patrol.env.get_obs_full``).
WINGMAN_OBS = 26

# ─── State & parameters ─────────────────────────────────


@struct.dataclass
class FormationState(EnvState):
    """The lead, the wingmen and the slots, plus the lead's route.

    Every leaf of ``wingmen`` has a leading axis of ``num_wingmen``, as do the
    slot offsets, which are expressed in the lead's body frame as in the
    patrol. ``lead.target_heading`` is the heading the lead is scored against;
    it turns at ``lead_turn_rate`` per step, the rate of the current leg of
    ``lead_turn_schedule``, whatever the lead does.
    """

    lead: PlaneState3D
    wingmen: PlaneState3D
    slot_back: jnp.ndarray
    slot_right: jnp.ndarray
    slot_up: jnp.ndarray
    lead_turn_rate: float
    lead_turn_schedule: jnp.ndarray
    # One turbulence gust for the whole formation (m/s): same air mass.
    gust_x: float = 0.0
    gust_y: float = 0.0
    gust_z: float = 0.0


@struct.dataclass
class FormationParams(PatrolParams):
    """The patrol's parameters, plus the V's lateral spacing and the team cost.

    The route (``lead_turn_rate_range``, ``lead_leg_seconds``), the slot
    depth and height (``slot_back_range``, ``slot_up_range``), the spawn noise
    and the trip thresholds are the patrol's. ``slot_right_range`` is not
    read: the V puts its wingmen on both sides, ``slot_lateral_range`` apart
    from the lead per rank.
    """

    #: Lateral offset of each rank of the V from the lead, in metres, drawn
    #: once per episode. At least 60 m, so the two wingmen of a rank start
    #: 120 m apart, well clear of ``min_separation``.
    slot_lateral_range: Tuple[float, float] = (60.0, 150.0)

    # ---- Reward (docs/reward-shaping.md; version 2) ----
    # The lead's altitude against the 1 m altimeter resolution and its heading
    # against the 0.0087 rad AHRS resolution, as in the 3D heading task; each
    # wingman's slot against the 3 m relative-GPS resolution and its heading
    # alignment against the AHRS, as in the patrol.
    e_floor_altitude: float = 1.0
    #: Twice the largest team tracking cost of the scored formation (a lead
    #: and two wingmen): the lead at the altitude envelope and half a turn off
    #: heading, each wingman at the slot-loss bound and half a turn off. It
    #: still covers four wingmen.
    failure_cost: float = 2.0 * (
        (12192.0 / 1.0) ** 2
        + (3.14159 / 0.0087) ** 2
        + 2 * ((1500.0 / 3.0) ** 2 + (3.14159 / 0.0087) ** 2)
    )
    #: The NEA reference: the oracle's lowest per-seed holds
    #: (scripts/measure_hold.py) in the reward's units, the lead's altitude
    #: and heading plus each wingman's slot and alignment.
    rho_floor_tracking: float = (
        (0.8187 / 1.0) ** 2
        + (1.125e-3 / 0.0087) ** 2
        + (0.2388 / 3.0) ** 2
        + (1.096e-3 / 0.0087) ** 2
        + (0.2743 / 3.0) ** 2
        + (1.165e-3 / 0.0087) ** 2
    )
    rho_floor: float = (
        (0.8187 / 1.0) ** 2
        + (1.125e-3 / 0.0087) ** 2
        + (0.2388 / 3.0) ** 2
        + (1.096e-3 / 0.0087) ** 2
        + (0.2743 / 3.0) ** 2
        + (1.165e-3 / 0.0087) ** 2
    )

    # ---- Version 1 (the team reward the unregistered module shipped) ----
    #: Weight of the wingmen's mean slot-keeping in the version-1 reward.
    w_track: float = 0.6
    #: Weight of the lead's altitude and heading tracking in it.
    w_lead: float = 0.4


#: Units of the FormationParams fields, for the generated environment pages.
PARAM_UNITS = {
    "slot_lateral_range": "m",
    "e_floor_altitude": "m",
    "failure_cost": "reward",
    "rho_floor_tracking": "reward",
    "rho_floor": "reward",
    "w_track": "-",
    "w_lead": "-",
}


# ─── Slot geometry ──────────────────────────────────────


def formation_slots(back, lateral, up, num_wingmen: int):
    """The (back, right, up) slot offsets of a V of ``num_wingmen``.

    Wingman ``i`` flies on the right when ``i`` is even and on the left when it
    is odd, at rank ``i // 2 + 1``: each rank sits ``back``, ``lateral`` and
    ``up`` further from the lead than the one before.
    """
    side = jnp.array([1.0 if i % 2 == 0 else -1.0 for i in range(num_wingmen)])
    rank = jnp.array([i // 2 + 1 for i in range(num_wingmen)], dtype=jnp.float32)
    return rank * back, side * rank * lateral, rank * up


@struct.dataclass
class SlotView:
    """One wingman and the lead, shaped as the patrol's geometry and
    observation functions read a :class:`PatrolState`."""

    follower: PlaneState3D
    lead: PlaneState3D
    slot_back: float
    slot_right: float
    slot_up: float


def _over_wingmen(fn, state: FormationState):
    """``fn(view)`` for every wingman, stacked along a leading axis."""
    return jax.vmap(
        lambda w, b, r, u: fn(SlotView(w, state.lead, b, r, u)),
    )(state.wingmen, state.slot_back, state.slot_right, state.slot_up)


def slot_errors(state: FormationState):
    """Each wingman's distance from its slot (metres), shape ``(num_wingmen,)``."""
    return _over_wingmen(slot_error, state)


def min_separation(state: FormationState):
    """The smallest 3D distance between any two aircraft (metres)."""
    w = state.wingmen
    pos = jnp.concatenate(
        [
            jnp.stack([state.lead.x, state.lead.y, state.lead.z])[None],
            jnp.stack([w.x, w.y, w.z], axis=-1),
        ]
    )
    diff = pos[:, None, :] - pos[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
    n = pos.shape[0]
    return jnp.min(jnp.where(jnp.eye(n, dtype=bool), jnp.inf, dist))


# ─── Termination & reward ───────────────────────────────


def check_is_terminal(state: FormationState, params: FormationParams, xp=jnp):
    """The formation trips when any aircraft leaves the altitude band, any two
    aircraft come within ``min_separation`` (25 m), or any wingman is
    ``max_slot_error`` (1500 m) from its slot. The whole formation restarts.

    Returns (terminated, truncated).
    """
    z = xp.concatenate([xp.reshape(state.lead.z, (1,)), state.wingmen.z])
    crash = xp.any((z <= params.min_alt) | (z >= params.max_alt))
    collision = min_separation(state) <= params.min_separation
    lost = xp.max(slot_errors(state)) >= params.max_slot_error
    terminated = crash | collision | lost
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def aircraft_costs(state: FormationState, params: FormationParams, xp=jnp):
    """Each aircraft's tracking cost, before any trip.

    ``{"lead": altitude + heading against the route, "wingmen": slot + heading
    alignment for each wingman}``; the team's tracking cost is their sum.
    """
    p = params
    lead = state.lead
    lead_cost = R.tracking_cost(
        lead.target_altitude - lead.z,
        p.e_floor_altitude,
        p.e_tol_altitude,
        p.tracking_exponent,
        xp,
    ) + R.tracking_cost(
        wrap_angle(lead.psi - lead.target_heading),
        p.e_floor_heading,
        0.0,
        p.tracking_exponent,
        xp,
    )
    wingmen = R.tracking_cost(
        slot_errors(state), p.e_floor_slot, p.e_tol_slot, p.tracking_exponent, xp
    ) + R.tracking_cost(
        wrap_angle(state.wingmen.psi - lead.psi),
        p.e_floor_heading,
        0.0,
        p.tracking_exponent,
        xp,
    )
    return {"lead": lead_cost, "wingmen": wingmen}


def compute_reward_terms(state: FormationState, params: FormationParams, xp=jnp):
    """The reward's additive cost terms, each >= 0 (``target_gym.reward``)."""
    costs = aircraft_costs(state, params, xp)
    terminated, _ = check_is_terminal(state, params, xp)
    terms = {"tracking": costs["lead"] + xp.sum(costs["wingmen"])}
    return R.with_trip(terms, terminated, R.trip_cost(params), xp)


def compute_reward_v1(state: FormationState, params: FormationParams, xp=jnp):
    """Version 1: the team reward the unregistered formation module shipped,
    in [0, 1], and ``-restart_steps`` on a tripped step.

    ``w_lead`` times the lead's log-scaled altitude and heading tracking plus
    ``w_track`` times the wingmen's mean log-scaled slot-keeping, each times
    its heading alignment. That module ended the episode on a trip; under the
    restart (``base.failure_kernel``) a 0 on the tripped step would make a
    trip nearly free, since the formation restarts in its slots, so the
    downtime is charged at version 1's best per-step score, as
    ``unstable_cstr`` and ``compressor_surge`` do.
    """
    terminated, _ = check_is_terminal(state, params, xp)
    lead = state.lead
    lead_r = log_scaled_reward(
        xp.abs(lead.target_altitude - lead.z),
        params.precision_floor,
        params.max_alt - params.min_alt,
        xp,
    ) * log_scaled_reward(
        xp.abs(wrap_angle(lead.psi - lead.target_heading)),
        params.heading_precision_floor,
        jnp.pi,
        xp,
    )
    track = _over_wingmen(
        lambda v: log_scaled_reward(
            slot_error(v), params.slot_precision_floor, params.max_slot_error, xp
        )
        * heading_alignment(v, params, xp),
        state,
    )
    team = params.w_track * xp.mean(track) + params.w_lead * lead_r
    return xp.where(terminated, -1.0 * params.restart_steps, team)


def compute_reward(state: FormationState, params: FormationParams, xp=jnp):
    """One team cost: the lead's squared altitude and heading errors against
    its route, plus each wingman's squared slot error and heading misalignment
    with the lead, each in units of its instrument's resolution (1 m, 0.5 deg,
    3 m, 0.5 deg). Every aircraft receives the same reward.

    ``reward_version = 1`` gives :func:`compute_reward_v1` instead.
    """
    return R.select(
        params.reward_version,
        compute_reward_v1(state, params, xp),
        R.total(compute_reward_terms(state, params, xp), xp),
        xp,
    )


# ─── Observations ───────────────────────────────────────


def commanded_heading(state: FormationState):
    """The heading the lead is scored against after this step."""
    return wrap_angle(state.lead.target_heading + state.lead_turn_rate)


def get_obs_lead(state: FormationState, xp=jnp):
    """The lead's observation (16 + 3 per wingman values).

    Layout::

        [ 3D heading observation, its target the heading commanded this step,  # 0-14
          lead_turn_rate,                                                      # 15
          e_back, e_right, e_up of each wingman ]                              # 16-

    The first 15 are what the patrol's lead autopilot reads, so the same
    autopilot flies the lead seat.
    """
    cmd = state.lead.replace(target_heading=commanded_heading(state))
    errors = _over_wingmen(lambda v: jnp.stack(slot_error_vector(v)), state)
    return xp.concatenate(
        [
            get_obs_heading(cmd, xp),
            xp.reshape(state.lead_turn_rate, (1,)),
            xp.reshape(errors, (-1,)),
        ]
    )


def get_obs_wingmen(state: FormationState):
    """Each wingman's observation, the patrol's (26 values), stacked."""
    return _over_wingmen(get_obs_full, state)


def get_obs_formation(state: FormationState, xp=jnp):
    """The joint observation: the lead's, then each wingman's in order."""
    return xp.concatenate(
        [get_obs_lead(state, xp), xp.reshape(get_obs_wingmen(state), (-1,))]
    )


# ─── Transition ─────────────────────────────────────────


def compute_next_state_formation(
    action: jnp.ndarray,
    state: FormationState,
    params: FormationParams,
    integration_method: str = "rk4_2",
    key=None,
):
    """One step: every aircraft moves on its own three controls.

    ``action`` is ``[power, stick, aileron]`` for the lead, then for each
    wingman, in [-1, 1]. One gust, advanced from the step clock as in the
    patrol, is folded into the wind every aircraft flies in; each still gets
    its own altitude-dependent shear. The lead's reference moves on along its
    route whatever the lead does. Returns ``(new_state, lead_metrics)``.
    """
    gust = advance_gust(
        jnp.array([state.gust_x, state.gust_y, state.gust_z]),
        params.turbulence_theta,
        params.turbulence_sigma,
        params.delta_t,
        step_key(key, state.time),
    )
    eff_params = params.replace(
        wind_x=params.wind_x + gust[0],
        wind_y=params.wind_y + gust[1],
        wind_z=params.wind_z + gust[2],
    )
    controls = jnp.reshape(action, (-1, 3))

    lead_cmd = state.lead.replace(target_heading=commanded_heading(state))
    new_lead, metrics = compute_next_state_3d(
        *decode_action(controls[0]),
        lead_cmd,
        eff_params,
        integration_method=integration_method,
    )

    def fly(control, wingman):
        new, _ = compute_next_state_3d(
            *decode_action(control),
            wingman,
            eff_params,
            integration_method=integration_method,
        )
        return new

    new_wingmen = jax.vmap(fly)(controls[1:], state.wingmen)

    new_state = state.replace(
        lead=new_lead,
        wingmen=new_wingmen,
        # The leg live at the next step, as in the patrol.
        lead_turn_rate=state.lead_turn_schedule[lead_leg(state.time + 1, params)],
        time=state.time + 1,
        gust_x=gust[0],
        gust_y=gust[1],
        gust_z=gust[2],
    )
    return new_state, metrics
