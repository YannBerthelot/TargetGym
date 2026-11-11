"""The formation's expert (the PID slot).

The PID (:func:`make_formation_pid_controller`) flies the lead seat as a
formation lead flies: the 3D heading task's course hold and pitch on height,
aimed at the heading the route commands. Each wingman seat
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
    HORNET_TECS,
    FormationHoldParams,
    FormationHoldState,
    HornetGuidanceParams,
    patrol_hold_step,
)
from target_gym.plane3d.experts import (
    Plane3DGuidanceState,
    _cruise_speed,
    _fly,
)

# ---------------------------------------------------------------------------
# The PID team: a steady lead and the patrol's formation hold in each seat
# ---------------------------------------------------------------------------


@struct.dataclass
class FormationLeadParams(HornetGuidanceParams):
    """The lead's autopilot: the heading task's course hold and TECS on
    height and the 3D tasks' speed schedule. A formation lead flies smooth and
    predictable, and with the engine's 1.7 s lag TECS's throttle does.
    ``turn_lag`` is the time, s, over which it eases onto a new leg's turn
    rate, so its wingmen, who see the turn only once it starts, can follow."""

    turn_lag: float = 0.0


@struct.dataclass
class FormationLeadState(Plane3DGuidanceState):
    """The autopilot's memory, and the turn rate it is easing onto."""

    turn_rate: float = 0.0


def formation_lead_step(params, state, obs, xp=jnp):
    """The lead seat: course hold on the route's commanded heading with its
    turn rate (``obs[15]``, rad per step) fed forward, eased over ``turn_lag``
    from the last, and TECS."""
    v_ground = xp.sqrt(obs[0] ** 2 + obs[1] ** 2 + 1e-6)
    ease = params.dt / (params.turn_lag + params.dt)
    rate = state.turn_rate + ease * (obs[15] / params.dt - state.turn_rate)
    turn = rate + _wrap_angle(obs[11] - obs[9], xp) / params.tau_hdg
    target = _cruise_speed(params, obs, xp)
    return _fly(
        params, state, obs, xp, obs[10], v_ground * turn, target, turn_rate=rate
    )


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
#: (scripts/tune_formation_hold.py, [lead pitch]). TECS flies the pitch on
#: height alone (the airliner's ``spd_weight`` 0), a little faster than the
#: heading task's (``time_const`` 1.5 against 2). It eases onto a new turn over
#: 2 s (2026-11-10): taken at once, a reversal rolled it 60 deg in a second and
#: its wingmen, a step behind, tripped on two of the ten seeds; 2 s is the
#: shortest of 2, 4 and 8 with no trip, and keeps the lead nearest its heading.
#: Every other autopilot value is the 3D heading task's
#: (scripts/tune_aircraft_guidance.py).
LEAD = FormationLeadParams(tecs=HORNET_TECS.replace(time_const=1.5), turn_lag=2.0)


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
