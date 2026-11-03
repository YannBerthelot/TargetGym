"""The formation's expert (the PID slot).

The PID (:func:`make_formation_pid_controller`) flies the lead seat as a
formation lead flies: the 3D heading task's course hold and pitch on height,
aimed at the heading the route commands, at a set power. Each wingman seat
flies the patrol's formation hold.

The oracle in the MPC slot (``FormationOracle``) is in ``oracle.py``.
"""

import jax.numpy as jnp
from flax import struct

from target_gym.experts.pid import (
    Controller,
    _wrap_angle,
)
from target_gym.formation.env import (
    LEAD_OBS_BASE,
    WINGMAN_OBS,
)
from target_gym.patrol.experts import (
    HOLD,
    FormationHoldParams,
    FormationHoldState,
    patrol_hold_step,
)
from target_gym.plane.experts_tecs import AIRLINER
from target_gym.plane3d.experts import (
    Plane3DGuidanceParams,
    Plane3DGuidanceState,
    _fly,
)

# ---------------------------------------------------------------------------
# The PID team: a steady lead and the patrol's formation hold in each seat
# ---------------------------------------------------------------------------


@struct.dataclass
class FormationLeadParams(Plane3DGuidanceParams):
    """The lead's autopilot: the heading task's course hold and pitch on
    height, with the throttle at a set power.

    A formation lead flies smooth and predictable and leaves the speed
    changes to its wingmen. Its throttle stays at ``power_set`` (engine
    power fraction), trimmed by ``k_speed`` per m/s off the speed it engaged
    at. Under TECS's throttle the airliner's 20 s engine lag sets up a slow
    throttle cycle, and the wingmen, feeding the lead's speed forward, chase
    it.
    """

    # Tuned on the formation's training seeds 100-109 for its wingmen flying
    # HOLD ([lead power] in scripts/tune_formation_hold.py).
    power_set: float = 0.8  # engine power fraction
    k_speed: float = 0.005  # power per m/s


@struct.dataclass
class FormationLeadState(Plane3DGuidanceState):
    """The autopilot's memory, the speed the set power engaged at, and
    whether it has (``started``, set from the second step on)."""

    started: float = 0.0
    v_set: float = 0.0


def formation_lead_step(params, state, obs, xp=jnp):
    """The lead seat: course hold on the route's commanded heading with its
    turn rate (``obs[15]``, rad per step) fed forward, TECS's pitch on the
    altitude, and the set power."""
    v_ground = xp.sqrt(obs[0] ** 2 + obs[1] ** 2 + 1e-6)
    turn = obs[15] / params.dt + _wrap_angle(obs[11] - obs[9], xp) / params.tau_hdg
    action, new = _fly(
        params, state, obs, xp, obs[10], v_ground * turn, params.target_speed
    )
    v = xp.sqrt(obs[0] ** 2 + obs[1] ** 2 + obs[3] ** 2)
    v_set = xp.where(state.started > 0.0, state.v_set, v)
    power = xp.clip(params.power_set + params.k_speed * (v_set - v), 0.0, 1.0)
    new = new.replace(v_set=v_set, started=xp.ones_like(v))
    return xp.stack([2.0 * power - 1.0, action[1], action[2]]), new


@struct.dataclass
class FormationTeamParams:
    lead: FormationLeadParams
    wingman: FormationHoldParams


@struct.dataclass
class FormationTeamState:
    """The lead's memory and one formation hold's per wingman."""

    lead: FormationLeadState
    wingmen: tuple


def formation_team_step(params, state, obs, xp=jnp):
    """The lead seat on the lead's observation, then the patrol's formation
    hold on each wingman's 26 values (the patrol's full observation)."""
    k = len(state.wingmen)
    lead_dim = LEAD_OBS_BASE + 3 * k
    action, lead = formation_lead_step(params.lead, state.lead, obs, xp)
    actions, wingmen = [action], []
    for i, memory in enumerate(state.wingmen):
        seat = obs[lead_dim + WINGMAN_OBS * i : lead_dim + WINGMAN_OBS * (i + 1)]
        action, memory = patrol_hold_step(params.wingman, memory, seat, xp)
        actions.append(action)
        wingmen.append(memory)
    return xp.concatenate(actions), FormationTeamState(
        lead=lead, wingmen=tuple(wingmen)
    )


#: The lead's tuning, on the formation's training seeds 100-109 with the
#: wingmen flying :data:`HOLD`, checked on held-out seeds 200-209
#: (scripts/tune_formation_hold.py, [lead pitch]). With the throttle set,
#: TECS flies the pitch on height alone (the airliner's ``spd_weight`` 0), a
#: little faster and better damped than the heading task's (``time_const``
#: 1.5 against 2, ``Kd_theta`` 1.0 against 0.3). Every other autopilot value
#: is the 3D heading task's (scripts/tune_aircraft_guidance.py).
LEAD = FormationLeadParams(Kd_theta=1.0, tecs=AIRLINER.replace(time_const=1.5))


def make_formation_pid_controller(num_wingmen: int = 2) -> Controller:
    """The formation's expert as a :class:`Controller`: the steady lead
    (:data:`LEAD`) and the patrol's formation hold (:data:`HOLD`) in every
    wingman seat."""
    state = FormationTeamState(
        lead=FormationLeadState(),
        wingmen=tuple(FormationHoldState() for _ in range(num_wingmen)),
    )
    return Controller(
        FormationTeamParams(lead=LEAD, wingman=HOLD), state, formation_team_step
    )
