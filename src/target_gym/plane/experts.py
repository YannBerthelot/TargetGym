"""Reference controller (the MPC slot) for
the 2D aircraft tasks (plane, plane_sine, plane_energy).

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.experts import pid as _pid_mod
from target_gym.experts.mpc import (
    GradientMPC,
    _done_value,
    _is_v1,
    _pid_rollout_plan,
    _v2_objective,
)

_PLANE_STALL_MARGIN = 1.3  # multiples of stall speed at which the barrier starts
_PLANE_BARRIER_WEIGHT = 10.0

# Stand-in for "the task's own default" in the factory's keyword arguments,
# so an explicit ``lr_end=None`` (a fixed step) stays expressible.
_TASK_DEFAULT = object()

#: Solver settings per task, keyed by ``params.target_pattern`` (0 hold:
#: ``plane``; 3 sinusoid: ``plane_sine``; 1 steps: ``plane_energy``). The
#: three tasks share this factory and the planner only receives the params,
#: so the pattern is what tells them apart. The table lives here, not in
#: the registry, because this file is in the three tasks' baseline
#: fingerprint and the registry is not: a setting passed from the registry
#: would change the oracle without staling its recorded scores.
#:
#: ``speed_loop``: the cascaded PID's airspeed loop drives the throttle, in
#: the plant and inside the planner's rollout, and the planner optimises the
#: elevator alone (see ``PlaneMPC``). ``handover``: the shipped planner flies
#: the capture of the altitude and the speed-loop planner the hold
#: (``PlaneHandoverMPC``); ``n_iter`` and ``lr_end`` are then the hold
#: planner's. ``lr_end``: the step decays geometrically from ``lr`` (0.05)
#: to ``lr_end`` over each solve, so the descent resolves the elevator finer
#: than its first step. ``gains_key``: the entry of ``data/pid_gains.json``
#: the oracle's copy of the cascaded PID is built from (its speed loop, the
#: initial plan and the guide). The baseline fingerprint collects the gain
#: keys that start with the task's name, so each task reads a key its own
#: fingerprint covers: ``plane`` reads ``plane_cascaded``, the others a copy
#: of it under their own name. ``objective_scale``: the planner's objective
#: is the reward divided by this fixed number, so tracking at the floor
#: costs about 1 per step. The shared objective divides by
#: ``rho_floor_tracking`` instead, and that is the NEA reference these holds
#: set, so each re-measured floor changed the oracle that measured it, and
#: the iteration did not settle (oracle audit, 2026-10). On
#: ``plane_energy``, raising it from 0.997 to 1.29 moved seed 1's hold from
#: 1.45 to 1.18 m, so the holds gave 1.29 and then 0.96; on ``plane``, 0.237
#: gave 0.242 and 0.242 gave 0.232. Fixed, the reference no longer feeds
#: back. The values are the ones each oracle was measured with: 0.84^2 on
#: ``plane``, its reference at the time, and 1 on the other two.
#:
#: Measured on the protocol (seeds 0-2, gain, lower is better; the oracle
#: audit, 2026-10). ``plane``: 3.45 shipped, 1.07 with the speed loop, 0.755
#: with the speed loop and the decay, better on every seed. The decay alone
#: tightens the altitude but leaves the throttle frozen (5.51 on seed 1
#: against 5.84). Clipping the gradient per actuator inside the planner lets
#: the throttle move, too slowly to matter (4.81 on seed 1, 4.74 with the
#: decay, against 0.92 with the speed loop). ``plane_sine``: 2.99 shipped,
#: 2.11 with the speed loop (worse on seed 2), 0.628 with both, better on
#: every seed. The decay alone: 2.76 on seed 0 against 4.09, at full throttle
#: and 27 m/s fast.
#:
#: Flown from the first step, though, the speed-loop planner captured the
#: initial altitude offset more slowly than the shipped one, most likely
#: because its decaying steps add up to about a third of the fixed step's
#: travel per solve (on ``plane`` seed 0 both sat at full throttle through
#: the climb). Over the ten recorded seeds, whose cost is mostly that
#: capture, it was worse on 9 of 10 ``plane`` seeds and all 10 ``plane_sine``
#: seeds, by 4 to 29% per seed and whatever the start offset (3721 per step
#: against 3276 on ``plane``, 1440 against 1251 on ``plane_sine``). So the
#: shipped planner flies the capture and hands over once the aircraft has
#: settled (``PlaneHandoverMPC``).
#: Probed on the protocol seeds with the speed loop's integrator started at
#: zero: 0.902 on ``plane`` with 50 hold iterations (seed 0 at 1.48 against
#: 1.06 for the speed-loop planner alone, seeds 1 and 2 level) and 0.629 on
#: ``plane_sine``; 0.594 and 0.571 with 100. Seeding the integrator to
#: match the capture's last throttle exactly cost 3.98 on ``plane`` seed 0
#: against 0.71: the capture ends a hair inside full throttle, 48 m/s slow,
#: and the integral term then cancelled the proportional one. Where the
#: throttle was off its stops (seed 2, 4 m/s fast at 0.03) it gave 0.33
#: against 0.59. Hence the seed closest to zero in ``_bumpless_integral``.
#: Recorded with it: 0.516 on ``plane`` and 0.546 on ``plane_sine``, zero
#: trips, and over ten seeds 3273 per step against the shipped planner's
#: 3276 on ``plane`` and 1249 against 1251 on ``plane_sine``, ahead on every
#: seed of both (by 0.02 to 3.0%); the reach cost is the shipped planner's
#: to within 0.01%.
#:
#: ``plane_energy`` keeps the shipped planner. Its gain is the ladder's
#: transients and nothing measured moves them, under its earlier 4.55 m
#: floor: 38.78 shipped; 38.38 with the speed loop (1.8% worse on seed 0) and
#: 38.76 with the decay as well (2.7% worse on seed 0), both by trading 7-8%
#: more altitude cost for less airspeed cost; 40.20 with the decay alone.
#: Rescoring the same trajectories with its altitude floor at 1.20 m
#: (``plane_energy-v3``) puts the shipped planner ahead of both by 6.6% and
#: 7.7%. Patterns without a registered task (ramp, chirp) also keep the
#: shipped planner, at the objective scale they had before the audit.
_TASK_SETTINGS = {
    0: dict(
        speed_loop=True,
        handover=True,
        n_iter=100,
        lr_end=0.002,
        gains_key="plane_cascaded",
        objective_scale=0.84**2,
    ),
    3: dict(
        speed_loop=True,
        handover=True,
        n_iter=100,
        lr_end=0.002,
        gains_key="plane_sine_cascaded",
        objective_scale=1.0,
    ),
    1: dict(
        speed_loop=False,
        handover=False,
        n_iter=50,
        lr_end=None,
        gains_key="plane_energy_cascaded",
        objective_scale=1.0,
    ),
}
_SHIPPED_SETTINGS = dict(
    speed_loop=False,
    handover=False,
    n_iter=50,
    lr_end=None,
    gains_key="plane_cascaded",
    objective_scale=0.84**2,
)

#: The capture planner: the shipped two-actuator planner, as it was before
#: the oracle audit (50 iterations at a fixed step).
_CAPTURE_SETTINGS = dict(n_iter=50, lr_end=None)

#: The handover (``PlaneHandoverMPC``). The aircraft counts as settled after
#: ``_HANDOVER_STEPS`` consecutive steps within ``_HANDOVER_E_ON`` metres of
#: the commanded altitude, and the capture planner takes back over beyond
#: ``_HANDOVER_E_OFF`` metres. ``_BUMPLESS_TOL`` is how far, in raw throttle
#: units, the speed loop's first command may land from the capture planner's
#: last one.
_HANDOVER_E_ON = 3.0
_HANDOVER_STEPS = 5
_HANDOVER_E_OFF = 20.0
_BUMPLESS_TOL = 0.01


def task_settings(params) -> dict:
    """The solver settings ``make_plane_mpc`` uses for ``params``' task."""
    pattern = int(getattr(params, "target_pattern", 0))
    return dict(_TASK_SETTINGS.get(pattern, _SHIPPED_SETTINGS))


def cascaded_pid_factory(gains_key: str):
    """A factory for the cascaded altitude-hold PID on the gains stored under
    ``gains_key`` in ``data/pid_gains.json`` (its ``"note"`` is skipped).

    ``make_plane_cascaded_pid`` reads ``plane_cascaded``, which only
    ``plane``'s fingerprint collects. Reads ``_load_gains()`` at call time,
    so a tuner that patches the cache is seen. A missing key raises rather
    than falling back to the constructor's defaults, which would fly a
    different autopilot without saying so.
    """

    def make():
        stored = _pid_mod._load_gains()
        if gains_key not in stored:
            raise KeyError(f"no {gains_key!r} entry in data/pid_gains.json")
        return _pid_mod.StatefulCascadedAltitudePID(
            **{
                k: float(v)
                for k, v in stored[gains_key].items()
                if not isinstance(v, (str, dict, list))
            }
        )

    return make


def _plane_objective(state, params):
    """The aircraft's own reward, less a barrier on flying into the stall.

    The altitude reward scores one thing and the aircraft has two actuators, so
    the planner is free to buy altitude with airspeed. Over a 90 s window that
    is the *best* thing it can do when the target is thousands of metres away:
    a zoom climb converts kinetic energy to potential energy far faster than
    the engines can supply it. Measured on ``plane_steps`` seed 0, that is
    exactly what the plan did -- airspeed fell monotonically from 201 to 30 m/s
    over 90 s while the aircraft climbed 3300 m, touched the commanded altitude
    with nothing left, departed at 91 deg angle of attack and hit the ground at
    t=372. It happened on ``plane`` too; there the phugoid that followed
    happened to damp out, which is luck rather than control.

    Angle of attack is the wrong thing to fence, even though it is what
    actually stalls: through that whole manoeuvre it sat at 4-8 deg and only
    crossed 15 deg at t=90, one step from the departure and at the very end of
    the planning window. Airspeed decays through the entire climb, so a floor
    under it is a constraint the planner can see coming and descend.

    The floor is the stall speed at this mass and altitude rather than a fixed
    number, since both move: ``sqrt(2 m g / (rho S CL_max))`` is where the wing
    can no longer carry the weight, and the barrier switches on at 1.3 times
    it, the usual approach margin. Bounded in [0, 1] by construction, which is
    what lets ``done_value`` stay below the worst step the planner can plan, so
    flying into the ground cannot look better than flying slowly.

    Measured on ``plane_steps`` seed 0 over 400 s: return 304 against 83 for
    the unprotected planner, settled error 0.1 m, and airspeed held above
    128 m/s with a worst angle of attack of 7.7 deg. On ``plane``, 226 against
    189. Like the turbine's barrier, the constants barely matter -- weight 30
    scores 299 and a 1.15 margin 299 -- which is the signature of a term
    shaping the approach rather than trading against the objective.

    Two alternatives, both measured and both rejected. Doubling the tail to
    ``n_tail=120`` prices more of the aftermath and does fix seed 0 (274), but
    it costs two thirds again in compute and only moves the horizon at which
    the same trade becomes profitable. Making the planner hold airspeed
    outright, by planning against ``speed_weight=1.0``, flies beautifully for
    400 s (166, angle of attack 2.1 deg) and then crashes anyway at t=1260 on
    the full episode and at t=386 on seed 2: it changes which trim the planner
    settles into without ever putting a floor under the trade.
    """
    from target_gym.plane.env import compute_reward

    reward = compute_reward(state, params)
    return reward - _PLANE_BARRIER_WEIGHT * _plane_stall_barrier(state, params)


def _plane_stall_barrier(state, params):
    """Squared shortfall of airspeed below 1.3x the stall speed, in [0, 1]."""
    speed = jnp.sqrt(state.x_dot**2 + state.z_dot**2)
    v_stall = jnp.sqrt(
        2.0
        * state.m
        * params.gravity
        / (state.rho * params.wings_surface * params.CL_max)
    )
    margin = speed / (_PLANE_STALL_MARGIN * v_stall)
    return jnp.maximum(1.0 - margin, 0.0) ** 2


def _speed_loop_gains(pid) -> tuple[float, float, float, float, float]:
    """The airspeed loop's constants, read off a live cascaded PID."""
    return (
        float(pid.target_speed),
        float(pid.cruise_power),
        float(pid.Kp_speed),
        float(pid.Ki_speed),
        float(pid.dt),
    )


def _speed_loop(integ, x_dot, gains):
    """One step of ``StatefulCascadedAltitudePID``'s throttle branch, in JAX:
    the PI on the airspeed error with its anti-windup (the step that
    saturates is not integrated). Returns the throttle and the integrator."""
    v_target, cruise, kp, ki, dt = gains
    err = v_target - x_dot
    integ_n = integ + err * dt
    power = cruise + kp * err + ki * integ_n
    power_c = jnp.clip(power, -1.0, 1.0)
    integ_n = jnp.where(power != power_c, integ_n - err * dt, integ_n)
    return power_c, integ_n


def _bumpless_integral(throttle, x_dot, gains, tol=_BUMPLESS_TOL):
    """The speed loop's integrator to start from, so that its first command
    lands within ``tol`` of ``throttle``: the value closest to zero that does.

    Zero (a cold start) when the cold loop already lands there, which is the
    case whenever both sit at full throttle. Otherwise the integral term takes
    up the difference, so the throttle does not jump at the handover.
    """
    v_target, cruise, kp, ki, dt = gains
    err = v_target - x_dot
    cold = cruise + kp * err + ki * err * dt  # the first command from zero
    if abs(float(np.clip(cold, -1.0, 1.0)) - throttle) <= tol:
        return 0.0
    want = throttle - tol if cold < throttle else throttle + tol
    return (want - cold) / ki


class PlaneMPC(GradientMPC):
    """The 2D aircraft's gradient MPC, with the throttle on the PID's loop.

    ``speed_loop=True``: the throttle is the cascaded PID's airspeed loop,
    closed loop, and the planner optimises the elevator alone. The shared
    planner never moved its throttle after its first plan (the oracle audit,
    2026-10): over the scored window of ``plane`` the command was constant
    to three decimals on every protocol seed while the airspeed sat 10 to
    27 m/s off cruise on average, which was the whole of its running cost.
    Neither 200 iterations nor a terminal airspeed cost moved it. The likely
    mechanism, inferred rather than instrumented: the whole-plan gradient
    clip is dominated by the elevator, the warm start repeats the last
    throttle, and the guide swap only fires on a whole-plan win.

    The loop runs twice, consistently. In the plant, a live copy of the PID
    (``make_pid``: the task's own gains, see ``_TASK_SETTINGS``) computes the
    throttle from the observation every step; the loop reads only the
    airspeed ``x_dot``, which the observation carries unchanged from the
    state. Inside the planner's rollout, a JAX port of the same loop (same
    gains, same anti-windup) computes the throttle from each predicted
    state, its integrator seeded from the live PID's, so the elevator plan
    is optimised against the throttle the plant will actually get. The
    integrator is controller state, like the warm start, so the oracle
    stays causal: it reads the state and the mean gust and nothing else.
    The plan keeps two columns so the guide and the initial plan (the PID's
    rollout) keep their shape; the throttle column is inert, its gradient
    is exactly zero.

    With ``speed_loop=False`` this is ``GradientMPC`` unchanged. On
    ``plane`` and ``plane_sine`` the oracle is a ``PlaneHandoverMPC`` over
    two of these: one with ``speed_loop=False`` for the capture of the
    altitude, one with ``speed_loop=True`` for the hold.
    """

    def __init__(
        self,
        env,
        params,
        *,
        speed_loop: bool = False,
        make_pid=None,
        **kwargs,
    ):
        self.speed_loop = bool(speed_loop)
        self._pid = None
        if self.speed_loop:
            if make_pid is None:
                make_pid = cascaded_pid_factory("plane_cascaded")
            self._pid = make_pid()
            self._pid.reset()
            if int(kwargs.get("n_tail", 0)):
                raise ValueError("speed_loop plans without an open-loop tail")
        super().__init__(env, params, **kwargs)

    # -- the throttle loop -------------------------------------------------

    def _rollout(self, actions, state):
        if not self.speed_loop:
            return super()._rollout(actions, state)
        state, integ0 = state  # the plant state and the live PID's integrator
        gains = _speed_loop_gains(self._pid)
        key = jax.random.PRNGKey(0)

        def step_fn(carry, u):
            s, done, integ = carry
            power_c, integ_n = _speed_loop(integ, s.x_dot, gains)
            act = jnp.stack([power_c, u[1]])
            _, new_s, r, terminated, _ = self.env.step_env(key, s, act, self.params)
            if self.objective_fn is not None:
                r = self.objective_fn(new_s, self.params)
            r = jnp.where(done, self.done_value, r)
            return (new_s, jnp.logical_or(done, terminated), integ_n), r

        init = (state, jnp.zeros((), dtype=bool), jnp.asarray(integ0, jnp.float32))
        _, rewards = jax.lax.scan(step_fn, init, actions)
        return jnp.sum(rewards)

    # -- the receding horizon ----------------------------------------------

    def step(self, obs, state):
        if not self.speed_loop:
            return super().step(obs, state)
        from target_gym.plane.env import get_obs

        o = (
            np.asarray(obs)
            if obs is not None
            else np.asarray(get_obs(state, self.params))
        )
        integ0 = float(np.asarray(self._pid._speed_integral))  # before this step
        throttle = float(np.asarray(self._pid.step(o), dtype=np.float32)[0])
        aug = (state, jnp.asarray(integ0, jnp.float32))
        if self._fresh and self.initial_plan_fn is not None:
            self._actions = jnp.asarray(
                self.initial_plan_fn(state, self.params), dtype=jnp.float32
            )
        self._fresh = False
        actions_init = jnp.concatenate([self._actions[1:], self._actions[-1:]], axis=0)
        self._actions = self._jit_optimize(actions_init, aug)
        if self.guide_plan_fn is not None:
            guide = jnp.asarray(
                self.guide_plan_fn(state, self.params), dtype=jnp.float32
            )
            if self._score(guide, aug) > self._score(self._actions, aug):
                self._actions = guide
        first = self._actions[0]
        self._u_prev = first
        return np.array([throttle, float(first[1])], dtype=np.float32)

    def reset(self):
        super().reset()
        if self._pid is not None:
            self._pid.reset()


class PlaneHandoverMPC:
    """The shipped planner for the capture, then the speed-loop planner.

    The two planners are good at different things. Planning both actuators
    with a fixed step, the shipped planner captures the initial altitude
    offset fast, at full throttle, but then holds its throttle where the
    capture left it. Planning the elevator alone with a decaying step, over
    the PID's airspeed loop, the other holds about four times tighter but
    captures more slowly, most likely because its decaying steps travel less
    far per solve. So the capture planner (``PlaneMPC`` with
    ``speed_loop=False``, the oracle the tasks had before the oracle audit:
    on ``plane`` it reproduces that oracle's ten recorded returns to the last
    digit) flies until the aircraft has been within ``e_on`` metres of the
    commanded altitude for ``settle_steps`` consecutive steps, and the hold
    planner flies from there. If the error ever grows past ``e_off`` the
    capture planner takes back over, from the hold planner's plan with the
    throttle column set to the last throttle applied, and the count starts
    again.

    At the handover the hold planner warm-starts from the capture planner's
    plan, and the speed loop's integrator is seeded so that its first command
    lands within ``bumpless_tol`` of the capture planner's last throttle
    (``_bumpless_integral``). The decision reads the true state, like both
    planners, so the oracle stays causal.
    """

    action_dim = 2

    def __init__(
        self,
        capture: PlaneMPC,
        hold: PlaneMPC,
        *,
        e_on: float = _HANDOVER_E_ON,
        settle_steps: int = _HANDOVER_STEPS,
        e_off: float = _HANDOVER_E_OFF,
        bumpless_tol: float = _BUMPLESS_TOL,
    ):
        if capture.speed_loop or not hold.speed_loop:
            raise ValueError("capture plans both actuators, hold the elevator")
        self.capture, self.hold = capture, hold
        self.env, self.params = hold.env, hold.params
        self.horizon = hold.horizon
        self.e_on, self.settle_steps = float(e_on), int(settle_steps)
        self.e_off, self.bumpless_tol = float(e_off), float(bumpless_tol)
        self.reset()

    @property
    def planner(self) -> PlaneMPC:
        """The planner flying this step."""
        return self.hold if self.mode == "hold" else self.capture

    @property
    def _actions(self):
        return self.planner._actions

    def step(self, obs, state):
        error = abs(float(state.target_altitude) - float(state.z))
        if self.mode == "capture":
            self._settled = self._settled + 1 if error < self.e_on else 0
            if self._settled >= self.settle_steps:
                self._to_hold(state)
        elif error > self.e_off:
            self._to_capture(state)
        action = np.asarray(self.planner.step(obs, state), dtype=np.float32)
        self._last = action
        return action

    def _to_hold(self, state):
        capture, hold = self.capture, self.hold
        hold.reset()
        if self._last is not None:
            hold._pid._speed_integral = _bumpless_integral(
                float(self._last[0]),
                float(state.x_dot),
                _speed_loop_gains(hold._pid),
                self.bumpless_tol,
            )
        hold._actions = capture._actions
        hold._u_prev = capture._u_prev
        hold._fresh = False
        self.mode = "hold"
        self.handovers.append((int(state.time), "hold"))

    def _to_capture(self, state):
        plan = np.array(self.hold._actions, dtype=np.float32)
        plan[:, 0] = self._last[0]
        self.capture._actions = jnp.asarray(plan)
        self.capture._u_prev = jnp.asarray(self._last)
        self.capture._fresh = False
        self.mode = "capture"
        self._settled = 0
        self.handovers.append((int(state.time), "capture"))

    def solver_report(self) -> dict:
        return {}

    def reset(self):
        self.capture.reset()
        self.hold.reset()
        self.mode = "capture"
        self._settled = 0
        self._last = None
        self.handovers = []


def make_plane_mpc(
    env,
    params,
    horizon: int = 30,
    n_iter=_TASK_DEFAULT,
    lr: float = 0.05,
    n_tail: int = 60,
    objective_fn=_plane_objective,
    lr_end=_TASK_DEFAULT,
    speed_loop=_TASK_DEFAULT,
    handover=_TASK_DEFAULT,
):
    """Gradient MPC for Airplane2D: optimises power and stick in [-1, 1], or
    the stick alone with the throttle on the PID's airspeed loop, or the
    first until the aircraft has captured its altitude and the second from
    there (``PlaneHandoverMPC``, on ``plane`` and ``plane_sine``).

    Uses gradient-based MPC because the Plane has 9 coupled nonlinear ODEs
    including aerodynamic coefficients that are not expressible in CasADi
    without a full symbolic re-implementation.  dt=1.0 s; horizon=30.

    Two things make it fly. The objective carries a stall barrier, without
    which the planner trades all its airspeed for altitude on every
    acquisition and departs; see ``_plane_objective``, which is where that is
    measured. And ``n_tail=60``: optimising 30 s of flight and being charged
    for nothing beyond it, the plan climbed hard and left the aircraft outside
    the altitude envelope just past the horizon, settling 654x worse than the
    PID and crashing in one episode of two. Simulating 60 further seconds on
    the held action -- which for this aircraft is close to trim -- prices that
    ending into the objective. Measured over 600-step episodes, settled
    tracking error went from 2949 m to 0.083 m, with no terminations. Sixty is
    the knee: 120 is no better on a fixed setpoint and costs twice.

    The tail alone was not enough, which is why the barrier is here. It fixes
    the ending the plan can *see*; the zoom climb is an ending the plan likes,
    and no affordable horizon changes that.

    ``done_value`` sits below the worst step the objective can score, so a plan
    that reaches the ground is charged for the rest of the horizon rather than
    scoring the 0.0 that a barrier-free positive reward could rely on.

    ``n_iter``, ``lr_end``, ``speed_loop`` and ``handover`` default to the
    task's entry in ``_TASK_SETTINGS`` under version 2; pass them to
    override (``lr_end=None`` is the fixed step, ``speed_loop=False`` the
    planner-driven throttle the tasks had before the oracle audit,
    ``handover=False`` the speed-loop planner from the first step). With the
    handover, ``n_iter`` and ``lr_end`` set the hold planner; the capture
    planner keeps the shipped settings (``_CAPTURE_SETTINGS``). Under
    version 2 the objective is the reward over the task's fixed
    ``objective_scale``, not over ``rho_floor_tracking``, so the NEA
    reference does not change the oracle. Version 1, and a caller's own
    ``objective_fn``, keep the planner they had: 50 iterations at a fixed
    step, both actuators planned.
    """
    initial_plan = None
    if not _is_v1(params) and objective_fn is _plane_objective:
        settings = task_settings(params)
        if n_iter is _TASK_DEFAULT:
            n_iter = settings["n_iter"]
        if lr_end is _TASK_DEFAULT:
            lr_end = settings["lr_end"]
        if speed_loop is _TASK_DEFAULT:
            speed_loop = settings["speed_loop"]
        if handover is _TASK_DEFAULT:
            handover = settings["handover"] and speed_loop
        make_pid = cascaded_pid_factory(settings["gains_key"])
        from target_gym.plane.env import compute_reward

        # The shared objective divides by ``rho_floor_tracking``, the NEA
        # reference that this oracle's own holds set; ``units`` pins that
        # divisor to the task's fixed scale instead (see ``_TASK_SETTINGS``).
        # compute_reward does not read it, so only the scale changes.
        scale = settings["objective_scale"]
        units = params if scale is None else params.replace(rho_floor_tracking=scale)
        v2 = _v2_objective(compute_reward, _plane_stall_barrier)

        def objective_fn(state, _params):
            return v2(state, units)

        done_value = _done_value(units)
        # Twenty seconds of horizon, not thirty. In turbulence the 30-step
        # plan never converged within the budget and the planner chattered
        # between its plan and the guide (stick moving 0.17 per step): it
        # held 4.6 m off with a 4.4 m bias where the PID held 2.5 m. Shorter
        # is myopic the other way -- airspeed answers the throttle slowly, so
        # at 15 steps the plan buys altitude with speed it will not see
        # itself pay for (23 m/s off cruise). Measured on seed 0 over the
        # hold, cost per step: 30 steps 4.6 (before the fix below, 10 at 25),
        # 15 steps 4.6, 20 steps 2.7 against the PID's 9; stick 0.009/step.
        if horizon == 30:
            horizon = 20
        # From a constant plan the planner parks at the edge of the altitude
        # tolerance 55 m/s below cruise: raising thrust alone pitches the
        # aircraft out of the band before the linear speed saving pays, and
        # the coordinated thrust-and-elevator move is out of reach of the
        # normalised steps. The cascaded PID's plan is a stabilising start.
        initial_plan = _pid_rollout_plan(env, make_pid, horizon, 2)
        guide_plan = initial_plan
        # No open-loop tail under version 2. Holding the last action for 60 s
        # lets the phugoid carry the aircraft far from the band, and under an
        # unbounded quadratic cost that tail dominated the objective (1e5
        # per solve against 26 per step realised): the planner optimised
        # what the held action did later, parked at the edge of the
        # tolerance and flew 55 m/s slow. Without it, and guided by the
        # PID's plan, it holds altitude to the metre at cruise speed.
        n_tail = 0
    else:
        done_value = -(_PLANE_BARRIER_WEIGHT + 1.0)
        guide_plan = None
        make_pid = None
        # Version 1 (or a caller's own objective) keeps the planner it had.
        if n_iter is _TASK_DEFAULT:
            n_iter = 50
        if lr_end is _TASK_DEFAULT:
            lr_end = None
        if speed_loop is _TASK_DEFAULT:
            speed_loop = False
        if handover is _TASK_DEFAULT:
            handover = False
    if handover and not speed_loop:
        raise ValueError("the handover is to the speed loop: pass speed_loop=True")

    def planner(speed_loop, n_iter, lr_end):
        return PlaneMPC(
            env,
            params,
            speed_loop=speed_loop,
            make_pid=make_pid,
            action_dim=2,
            action_lb=-1.0,
            action_ub=1.0,
            horizon=horizon,
            n_iter=n_iter,
            lr=lr,
            lr_end=lr_end,
            n_tail=n_tail,
            objective_fn=objective_fn,
            initial_plan_fn=initial_plan,
            guide_plan_fn=guide_plan,
            done_value=done_value,
        )

    if not handover:
        return planner(speed_loop, n_iter, lr_end)
    return PlaneHandoverMPC(
        planner(False, **_CAPTURE_SETTINGS), planner(True, n_iter, lr_end)
    )
