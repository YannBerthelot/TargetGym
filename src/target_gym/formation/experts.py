"""The formation's baselines.

The PID (:func:`make_formation_pid_controller`) flies the lead seat as a
formation lead flies: the 3D heading task's course hold and pitch on height,
aimed at the heading the route commands, at a set power. Each wingman seat
flies the patrol's formation hold.

The oracle in the MPC slot (:class:`FormationOracle`) is the patrol oracle's
design applied to the whole formation: each seat flies a law (the patrol's
lead autopilot in the lead seat, the patrol's twin-autopilot law in each
wingman seat), and one receding-horizon planner adds a residual to every seat,
descending the true team cost. Unlike a split into one planner per aircraft it
can coordinate: the lead may ease a turn for its wingmen when that lowers the
team's cost.
"""

import jax
import jax.numpy as jnp
from flax import struct

from target_gym.experts.pid import (
    Controller,
    _wrap_angle,
    make_plane3d_heading_pid,
    plane3d_heading_pid_step,
)
from target_gym.formation.env import (
    LEAD_OBS_BASE,
    WINGMAN_OBS,
    commanded_heading,
)
from target_gym.patrol.env import PatrolState, get_obs_heading
from target_gym.patrol.experts import (
    HOLD,
    TWIN_GAINS,
    FormationHoldParams,
    FormationHoldState,
    PatrolTwinOracle,
    patrol_hold_step,
    twin_law_init,
    twin_law_step,
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


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------


class FormationOracle(PatrolTwinOracle):
    """Each seat's law with one residual planner over the whole formation.

    The laws: the lead flies the patrol's lead autopilot, aimed at the heading
    the route commands this step; each wingman flies the patrol's
    twin-autopilot law on its patrol view of the state (the lead's own
    autopilot on its own state, aimed at its slot). At every control step,
    from the true state and the laws' memory, the planner optimises a residual
    ``delta[0:H]`` for every seat on the summed team cost of the
    environment's own ``step_env`` (a trip is charged its trip cost), with
    the turbulence zeroed in its model. A wingman's residual is added on top
    of the lead's: with its own at zero, a wingman in its slot copies the
    lead's correction as it copies the lead's autopilot, so the correction,
    like the shared gust, drops out of the relative position. Adam, ``n_iter``
    iterations at step ``lr``, each residual within +-``max_residual``; the
    best iterate is kept (:meth:`PatrolTwinOracle._solve`) and the plan is
    shifted one step as the next warm start.

    ``horizon=0`` or ``n_iter=0`` gives the laws alone. Reads ``state``,
    never ``obs``, as every MPC slot does.
    """

    def __init__(
        self,
        env,
        params,
        horizon: int = 30,
        n_iter: int = 40,
        lr: float = 0.02,
        max_residual: float = 1.0,
        gains: dict | None = None,
    ):
        self.env = env
        self.params = params
        self.num_wingmen = env.num_wingmen
        self.action_dim = 3 * (self.num_wingmen + 1)
        self.horizon = int(horizon)
        self.n_iter = int(n_iter)
        self.lr = float(lr)
        self.max_residual = float(max_residual)
        self.gains = dict(TWIN_GAINS, **(gains or {}))
        self.plans = self.horizon > 0 and self.n_iter > 0
        self._model = params.replace(turbulence_sigma=0.0)
        # As the patrol oracle's: with the turbulence zeroed the key only
        # seeds the restart draw after a trip in a proposal.
        self._key = jax.random.PRNGKey(0)
        self._lead_params, lead_state = make_plane3d_heading_pid()
        self._lead_zero = jax.tree.map(
            lambda x: jnp.asarray(x, jnp.float32), lead_state
        )
        self._jit_step = jax.jit(self._step_fn)
        self.reset()

    def _law(self, carry, state):
        """Every seat's law: ``(lead action, wingmen actions (k, 3), carry)``."""
        lead_pid, seats = carry
        cmd = state.lead.replace(target_heading=commanded_heading(state))
        lead, new_pid = plane3d_heading_pid_step(
            self._lead_params, lead_pid, get_obs_heading(cmd)
        )
        # Keep the memory's types fixed, so the jitted step compiles once.
        new_pid = new_pid.replace(lead_psi_prev=lead_pid.lead_psi_prev)
        gains, lead_params = self.gains, self._lead_params

        def seat(memory, wingman, back, right, up):
            view = PatrolState(
                follower=wingman,
                lead=state.lead,
                lead_pid=lead_pid,
                slot_back=back,
                slot_right=right,
                slot_up=up,
                lead_turn_rate=state.lead_turn_rate,
                lead_turn_schedule=state.lead_turn_schedule,
                time=state.time,
                gust_x=state.gust_x,
                gust_y=state.gust_y,
                gust_z=state.gust_z,
            )
            return twin_law_step(lead_params, gains, memory, view)

        wingmen, seats = jax.vmap(seat)(
            seats, state.wingmen, state.slot_back, state.slot_right, state.slot_up
        )
        return lead, wingmen, (new_pid, seats)

    @staticmethod
    def _compose(lead, wingmen, delta):
        """The joint action: each law plus its residual, a wingman's on top of
        the lead's (``delta`` is ``(k + 1, 3)``, the lead's row first)."""
        lead = jnp.clip(lead + delta[0], -1.0, 1.0)
        wingmen = jnp.clip(wingmen + delta[0][None] + delta[1:], -1.0, 1.0)
        return jnp.concatenate([lead, wingmen.reshape(-1)])

    def _plan_cost(self, delta, state, carry):
        env, model, key = self.env, self._model, self._key

        def body(c, d):
            s, cc = c
            lead, wingmen, cc = self._law(cc, s)
            _, s2, r, _, _ = env.step_env(
                key, s, self._compose(lead, wingmen, d), model
            )
            return (s2, cc), -r

        _, costs = jax.lax.scan(body, (state, carry), delta)
        return costs.sum()

    def _step_fn(self, carry, plan, state):
        if self.plans:
            warm = jnp.concatenate([plan[1:], plan[-1:]], axis=0)
            delta = self._solve(state, carry, warm)
            first = delta[0]
        else:
            delta = plan
            first = jnp.zeros((self.num_wingmen + 1, 3))
        lead, wingmen, carry = self._law(carry, state)
        return self._compose(lead, wingmen, first), carry, delta

    def reset(self):
        """Clear every seat's memory and the warm start (a zero residual)."""
        k = self.num_wingmen
        seats = jax.tree.map(lambda x: jnp.stack([x] * k), twin_law_init())
        self._carry = (self._lead_zero, seats)
        self._plan = jnp.zeros((max(self.horizon, 1), k + 1, 3), jnp.float32)


def make_formation_mpc(
    env,
    params,
    horizon: int | None = None,
    n_iter: int | None = None,
    lr: float | None = None,
    n_tail: int | None = None,
    max_residual: float = 1.0,
    gains: dict | None = None,
):
    """The formation oracle, :class:`FormationOracle`: each seat's law with
    one residual planner over the formation (defaults: 30-step horizon, 40
    Adam iterations at 0.02, residual within +-1). ``horizon=0`` or
    ``n_iter=0`` gives the laws alone. ``n_tail`` is accepted for
    compatibility and ignored. The same oracle serves both reward versions.

    The patrol oracle's other settings, with the residual bound at the full
    action range rather than its +-0.3: the lead has to be pulled out of the
    climb its autopilot lets it into from the untrimmed start. Measured on
    training seeds 100-103 (never the protocol's 0-2), over the second half
    of the test episode: at +-1 the team costs 0.82 to 1.45 per step and the
    lead holds 0.67 to 0.91 m of altitude error; at +-0.3 the lead is still
    about 50 m off and the team costs about 2800."""
    return FormationOracle(
        env,
        params,
        horizon=30 if horizon is None else horizon,
        n_iter=40 if n_iter is None else n_iter,
        lr=0.02 if lr is None else lr,
        max_residual=max_residual,
        gains=gains,
    )
