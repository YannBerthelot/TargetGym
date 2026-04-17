"""
JAX-compatible PID expert controllers for target_gym environments.

Design mirrors gymnax: explicit state structs + pure functions so everything
is jit/vmap/scan-able.

Usage pattern (single episode with jax.lax.scan):

    from target_gym.experts.pid import make_cstr_pid, pid_step

    params, state0 = make_cstr_pid()

    def step_fn(carry, obs):
        pid_state = carry
        action, new_pid_state = pid_step(params, pid_state, obs)
        return new_pid_state, action

    final_state, actions = jax.lax.scan(step_fn, state0, obs_sequence)

Observation layouts (matches each env's get_obs):
    CSTR       : [Ca, T, target_Ca]
    FirstOrder : [x, target_x]
    Nonsmooth  : [x1, x2, target_x1]
    FourTank   : [h1, h2, h3, h4, target_h1, target_h2]
    Car        : [velocity, target_velocity, ...]  (uses only first two)
    Plane      : uses [z, target_altitude] (minimal obs in runner)

All actions are raw-normalized in [-1, 1] (what step_env expects).

Tuning methodology
------------------
Gains are optimised by gradient descent (Adam) on an ITAE loss averaged over
many uniformly-spaced setpoints, via JAX autodiff through the full closed-loop
rollout.  See ``target_gym.experts.pid_tuning`` to re-run the optimisation.

Notes per environment:
- CSTR       : L2 regularisation prevents Kp runaway in the saturation regime.
- Nonsmooth  : targets (1.0–2.0) lie outside the actuator's steady-state range
               (u∈[−1,1] → x1_ss∈[−0.5, 0.5]); gains cannot solve this.
- FourTank   : cross-coupling (v1↔h2, v2↔h1) limits independent-loop PID; the
               asymmetric gains (pid2 stronger) reflect the coupling structure.
"""

import json
import pathlib

import jax.numpy as jnp
import numpy as np
from flax import struct

# ---------------------------------------------------------------------------
# Persistent gains store
# ---------------------------------------------------------------------------

# Resolved relative to this file so it works regardless of cwd.
_GAINS_FILE = pathlib.Path(__file__).resolve().parents[3] / "data" / "pid_gains.json"
_gains_cache: dict | None = None


def _load_gains() -> dict:
    """Load gains from data/pid_gains.json (lazy, cached until process restart)."""
    global _gains_cache
    if _gains_cache is None:
        if _GAINS_FILE.exists():
            with open(_GAINS_FILE) as f:
                _gains_cache = json.load(f)
        else:
            _gains_cache = {}
    return _gains_cache


def _g(env: str, key: str, default: float) -> float:
    """Return gain ``key`` for ``env`` from the JSON store, or ``default``."""
    return float(_load_gains().get(env, {}).get(key, default))


def _load_gain_schedule(env: str) -> dict | None:
    """Return the ``gain_schedule`` sub-dict for *env*, or None if absent."""
    return _load_gains().get(env, {}).get("gain_schedule", None)


# ---------------------------------------------------------------------------
# JAX state and params structs (jit/vmap/scan compatible)
# ---------------------------------------------------------------------------


@struct.dataclass
class PIDState:
    """Carried state for one SISO PID loop."""

    integral: float
    prev_error: float


@struct.dataclass
class PIDParams:
    """Static configuration for one SISO PID loop."""

    Kp: float
    Ki: float
    Kd: float
    dt: float
    state_index: int
    setpoint_index: int
    action_min: float
    action_max: float


@struct.dataclass
class GainSchedulePIDParams:
    """PID with gains interpolated from a table of operating points.

    At each step, the current setpoint is read from the observation and
    (Kp, Ki, Kd) are looked up via ``jnp.interp``. This makes the
    controller adapt to the operating regime — equivalent to classical
    gain scheduling used in industrial process control.

    All array fields must have the same length N (number of grid points).
    ``operating_points`` must be sorted in ascending order.
    """

    operating_points: jnp.ndarray  # (N,) — sorted setpoint values
    Kp_table: jnp.ndarray  # (N,)
    Ki_table: jnp.ndarray  # (N,)
    Kd_table: jnp.ndarray  # (N,)
    dt: float
    state_index: int
    setpoint_index: int
    action_min: float = -1.0
    action_max: float = 1.0


@struct.dataclass
class MIMOPIDState:
    """Carried state for a 2×2 independent-loop PID (e.g. Four Tank)."""

    state1: PIDState
    state2: PIDState


@struct.dataclass
class MIMOPIDParams:
    """Static configuration for a 2×2 independent-loop PID."""

    pid1: PIDParams
    pid2: PIDParams


# ---------------------------------------------------------------------------
# Pure JAX step functions — jit/vmap/scan compatible
# ---------------------------------------------------------------------------


def pid_reset(params: PIDParams) -> PIDState:  # noqa: ARG001
    """Return a zeroed initial state for a SISO PID."""
    return PIDState(integral=jnp.zeros(()), prev_error=jnp.zeros(()))


def pid_step(
    params: PIDParams,
    state: PIDState,
    obs: jnp.ndarray,
) -> tuple[jnp.ndarray, PIDState]:
    """
    One discrete PID update step.

    Returns (action [shape (1,)], new_state).
    Anti-windup: the integral is not updated when the output is saturated.
    """
    e = obs[params.setpoint_index] - obs[params.state_index]

    new_integral = state.integral + e * params.dt
    derivative = (e - state.prev_error) / params.dt
    u = params.Kp * e + params.Ki * new_integral + params.Kd * derivative
    u_clipped = jnp.clip(u, params.action_min, params.action_max)

    # Anti-windup: undo integral accumulation on saturation
    new_integral = jnp.where(u == u_clipped, new_integral, state.integral)

    new_state = PIDState(integral=new_integral, prev_error=e)
    return jnp.array([u_clipped]), new_state


def gain_scheduled_pid_step(
    params: GainSchedulePIDParams,
    state: PIDState,
    obs: jnp.ndarray,
) -> tuple[jnp.ndarray, PIDState]:
    """One PID update with gains interpolated from the operating-point table.

    Identical to ``pid_step`` except Kp/Ki/Kd are looked up via
    ``jnp.interp`` based on the current setpoint in the observation.
    Fully JIT-compatible.
    """
    sp = obs[params.setpoint_index]
    Kp = jnp.interp(sp, params.operating_points, params.Kp_table)
    Ki = jnp.interp(sp, params.operating_points, params.Ki_table)
    Kd = jnp.interp(sp, params.operating_points, params.Kd_table)

    e = sp - obs[params.state_index]
    new_integral = state.integral + e * params.dt
    derivative = (e - state.prev_error) / params.dt
    u = Kp * e + Ki * new_integral + Kd * derivative
    u_clipped = jnp.clip(u, params.action_min, params.action_max)
    new_integral = jnp.where(u == u_clipped, new_integral, state.integral)
    return jnp.array([u_clipped]), PIDState(integral=new_integral, prev_error=e)


@struct.dataclass
class MIMOGainSchedulePIDParams:
    """Two independent gain-scheduled PID loops (e.g. Four Tank, Plane)."""

    pid1: GainSchedulePIDParams
    pid2: GainSchedulePIDParams


def mimo_gain_scheduled_pid_step(
    params: MIMOGainSchedulePIDParams,
    state: MIMOPIDState,
    obs: jnp.ndarray,
) -> tuple[jnp.ndarray, MIMOPIDState]:
    """One step of two independent gain-scheduled PIDs. Returns (action [2,], new_state)."""
    u1, s1 = gain_scheduled_pid_step(params.pid1, state.state1, obs)
    u2, s2 = gain_scheduled_pid_step(params.pid2, state.state2, obs)
    return jnp.concatenate([u1, u2]), MIMOPIDState(state1=s1, state2=s2)


def mimo_pid_reset(params: MIMOPIDParams) -> MIMOPIDState:
    """Return a zeroed initial state for a 2×2 MIMO PID."""
    return MIMOPIDState(
        state1=pid_reset(params.pid1),
        state2=pid_reset(params.pid2),
    )


def mimo_pid_step(
    params: MIMOPIDParams,
    state: MIMOPIDState,
    obs: jnp.ndarray,
) -> tuple[jnp.ndarray, MIMOPIDState]:
    """
    One step of two independent SISO PIDs. Returns (action [shape (2,)], new_state).
    """
    u1, new_state1 = pid_step(params.pid1, state.state1, obs)
    u2, new_state2 = pid_step(params.pid2, state.state2, obs)
    new_state = MIMOPIDState(state1=new_state1, state2=new_state2)
    return jnp.concatenate([u1, u2]), new_state


# ---------------------------------------------------------------------------
# Python stateful PIDs — for video rollouts (not JAX-traced)
# ---------------------------------------------------------------------------


class StatefulPID:
    """
    Python stateful PID controller, suitable for select_action closures in
    save_video (which runs in a Python while-loop, not under JAX tracing).

    If fixed_setpoint is given, the setpoint is a constant rather than read
    from obs[setpoint_index].  This is useful when the target is a fixed constant (e.g. 0).
    """

    def __init__(
        self,
        Kp: float,
        Ki: float,
        Kd: float,
        dt: float,
        state_index: int,
        action_min: float = -1.0,
        action_max: float = 1.0,
        setpoint_index: int | None = None,
        fixed_setpoint: float | None = None,
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.state_index = state_index
        self.setpoint_index = setpoint_index
        self.fixed_setpoint = fixed_setpoint
        self.action_min = action_min
        self.action_max = action_max
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, obs) -> float:
        state_val = float(obs[self.state_index])
        if self.fixed_setpoint is not None:
            sp = self.fixed_setpoint
        else:
            sp = float(obs[self.setpoint_index])
        e = sp - state_val
        self.integral += e * self.dt
        derivative = (e - self.prev_error) / self.dt
        u = self.Kp * e + self.Ki * self.integral + self.Kd * derivative
        u_clipped = float(np.clip(u, self.action_min, self.action_max))
        if u != u_clipped:
            self.integral -= e * self.dt  # anti-windup
        self.prev_error = e
        return u_clipped


class StatefulGainScheduledPID:
    """Gain-scheduled PID for video/gymnasium rollouts (not JAX-traced).

    At each step, interpolates (Kp, Ki, Kd) from a table of operating
    points based on the current setpoint.
    """

    def __init__(
        self,
        operating_points: np.ndarray,
        Kp_table: np.ndarray,
        Ki_table: np.ndarray,
        Kd_table: np.ndarray,
        dt: float,
        state_index: int,
        setpoint_index: int,
        action_min: float = -1.0,
        action_max: float = 1.0,
    ):
        self.operating_points = np.asarray(operating_points)
        self.Kp_table = np.asarray(Kp_table)
        self.Ki_table = np.asarray(Ki_table)
        self.Kd_table = np.asarray(Kd_table)
        self.dt = dt
        self.state_index = state_index
        self.setpoint_index = setpoint_index
        self.action_min = action_min
        self.action_max = action_max
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, obs) -> float:
        sp = float(obs[self.setpoint_index])
        Kp = float(np.interp(sp, self.operating_points, self.Kp_table))
        Ki = float(np.interp(sp, self.operating_points, self.Ki_table))
        Kd = float(np.interp(sp, self.operating_points, self.Kd_table))

        e = sp - float(obs[self.state_index])
        self.integral += e * self.dt
        derivative = (e - self.prev_error) / self.dt
        u = Kp * e + Ki * self.integral + Kd * derivative
        u_clipped = float(np.clip(u, self.action_min, self.action_max))
        if u != u_clipped:
            self.integral -= e * self.dt
        self.prev_error = e
        return u_clipped


class StatefulMIMOPID:
    """Two independent SISO PIDs packaged as a MIMO controller (e.g. FourTank)."""

    def __init__(self, pid1: StatefulPID, pid2: StatefulPID):
        self.pid1 = pid1
        self.pid2 = pid2

    def reset(self):
        self.pid1.reset()
        self.pid2.reset()

    def step(self, obs) -> np.ndarray:
        return np.array([self.pid1.step(obs), self.pid2.step(obs)])


# ---------------------------------------------------------------------------
# Per-environment JAX factories  →  (PIDParams, PIDState)
# ---------------------------------------------------------------------------


def make_cstr_pid(
    Kp: float | None = None,
    Ki: float | None = None,
    Kd: float | None = None,
) -> tuple[PIDParams, PIDState]:
    """
    PID for CSTR — tracks Ca by controlling Tc (coolant temperature).

    Observation : [Ca, T, target_Ca]
    Action      : raw Tc in [-1, 1] → physical [295, 302] K inside the env

    Sign: Ca increases when Tc decreases (more cooling → lower T → slower
    reaction → higher Ca), so all gains are negative.
    dt = env delta_t = 0.25 (matches PC-gym: tsim=25s, N=100).

    Gains are read from data/pid_gains.json if present; keyword arguments
    override the file (useful for ablation / manual testing).
    """
    Kp = Kp if Kp is not None else _g("cstr", "Kp", -103.6)
    Ki = Ki if Ki is not None else _g("cstr", "Ki", -1.86)
    Kd = Kd if Kd is not None else _g("cstr", "Kd", -26.87)
    params = PIDParams(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        dt=0.25,
        state_index=0,  # Ca
        setpoint_index=2,  # target_Ca
        action_min=-1.0,
        action_max=1.0,
    )
    return params, pid_reset(params)


def make_first_order_pid(
    Kp: float | None = None,
    Ki: float | None = None,
    Kd: float | None = None,
) -> tuple[PIDParams, PIDState]:
    """
    PID for FirstOrderSystem — tracks x by controlling u.

    Observation : [x, target_x]
    Action      : raw u in [-1, 1] → physical [-2, 2] inside the env

    Steady state: x_ss = K * u_actual = u_actual (K=1),
    u_actual = 2 * u_raw, so u_raw_ss = x_target / 2.
    dt = env delta_t = 0.05.
    """
    Kp = Kp if Kp is not None else _g("first_order", "Kp", 2.11)
    Ki = Ki if Ki is not None else _g("first_order", "Ki", 9.72)
    Kd = Kd if Kd is not None else _g("first_order", "Kd", 0.0039)
    params = PIDParams(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        dt=0.05,
        state_index=0,  # x
        setpoint_index=1,  # target_x
        action_min=-1.0,
        action_max=1.0,
    )
    return params, pid_reset(params)


def make_four_tank_pid(
    Kp1: float | None = None,
    Ki1: float | None = None,
    Kd1: float | None = None,
    Kp2: float | None = None,
    Ki2: float | None = None,
    Kd2: float | None = None,
) -> tuple[MIMOPIDParams, MIMOPIDState]:
    """
    Two independent PID loops for FourTank — h1 via v1, h2 via v2.

    Observation : [h1, h2, h3, h4, target_h1, target_h2]
    Action      : [raw_v1, raw_v2] each in [-1, 1] → physical [0, 10] V

    Cross-coupling (v1→h4→h2, v2→h3→h1) is ignored; with γ1=γ2=0.2
    (80 % of each pump goes to the upper cross-tank) independent loops
    are still a reasonable expert baseline.
    dt = env delta_t = 1.0.
    """
    _ft = _load_gains().get("four_tank", {})
    _p1 = _ft.get("pid1", {})
    _p2 = _ft.get("pid2", {})
    Kp1 = Kp1 if Kp1 is not None else float(_p1.get("Kp", 6.74))
    Ki1 = Ki1 if Ki1 is not None else float(_p1.get("Ki", 1.34))
    Kd1 = Kd1 if Kd1 is not None else float(_p1.get("Kd", 0.0))
    Kp2 = Kp2 if Kp2 is not None else float(_p2.get("Kp", 15.29))
    Ki2 = Ki2 if Ki2 is not None else float(_p2.get("Ki", 2.59))
    Kd2 = Kd2 if Kd2 is not None else float(_p2.get("Kd", 0.0))
    params = MIMOPIDParams(
        pid1=PIDParams(
            Kp=Kp1,
            Ki=Ki1,
            Kd=Kd1,
            dt=1.0,
            state_index=0,  # h1
            setpoint_index=4,  # target_h1
            action_min=-1.0,
            action_max=1.0,
        ),
        pid2=PIDParams(
            Kp=Kp2,
            Ki=Ki2,
            Kd=Kd2,
            dt=1.0,
            state_index=1,  # h2
            setpoint_index=5,  # target_h2
            action_min=-1.0,
            action_max=1.0,
        ),
    )
    return params, mimo_pid_reset(params)


def make_glass_furnace_pid(
    Kp: float | None = None,
    Ki: float | None = None,
    Kd: float | None = None,
) -> tuple[PIDParams, PIDState]:
    """
    PID for GlassFurnace — tracks T_crown by controlling fuel flow.

    Observation : [T_crown, fuel_flow, target_T_crown]
    Action      : raw fuel in [-1, 1] → physical [fuel_min, fuel_max] kg/s

    Sign: more fuel → more heat → higher T_crown → Kp > 0.
    dt = env delta_t = 30.0 s.
    """
    Kp = Kp if Kp is not None else _g("glass_furnace", "Kp", 0.01)
    Ki = Ki if Ki is not None else _g("glass_furnace", "Ki", 0.001)
    Kd = Kd if Kd is not None else _g("glass_furnace", "Kd", 0.0)
    params = PIDParams(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        dt=30.0,
        state_index=0,  # T_crown
        setpoint_index=2,  # target_T_crown
        action_min=-1.0,
        action_max=1.0,
    )
    return params, pid_reset(params)


def make_reactor_pid(
    Kp: float | None = None,
    Ki: float | None = None,
    Kd: float | None = None,
) -> tuple[PIDParams, PIDState]:
    """
    PID for Reactor — tracks normalised neutron density ``n`` by moving the
    control rod (raw action → rho_ext).

    Observation : [n, T_coolant, rho_ext_norm, target_n]
    Action      : raw rho_ext in [-1, 1] → physical rho_ext in
                  [rho_ext_min, rho_ext_max] ([-0.010, +0.003] by default).

    Sign: withdrawing rods (positive rho_ext) raises dn/dt → Kp > 0.
    The asymmetric action range means the closed-loop gain is larger
    when inserting rods than when withdrawing, which is physically
    correct (rod insertion is always authorised, withdrawal is capped
    below prompt-critical).
    dt = env delta_t = 0.5 s.
    """
    Kp = Kp if Kp is not None else _g("reactor", "Kp", 5.0)
    Ki = Ki if Ki is not None else _g("reactor", "Ki", 0.5)
    Kd = Kd if Kd is not None else _g("reactor", "Kd", 0.0)
    params = PIDParams(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        dt=0.5,
        state_index=0,  # n
        setpoint_index=3,  # target_n
        action_min=-1.0,
        action_max=1.0,
    )
    return params, pid_reset(params)


def make_plane_pid(
    Kp1: float | None = None,
    Ki1: float | None = None,
    Kd1: float | None = None,
    Kp2: float | None = None,
    Ki2: float | None = None,
    Kd2: float | None = None,
) -> tuple[MIMOPIDParams, MIMOPIDState]:
    """
    MIMO PID for Airplane2D — controls both power and stick to track altitude.

    Observation (full get_obs): [x_dot, z, z_dot, theta, theta_dot, gamma,
                                  target_altitude, power, stick]
    Action: (power, stick) both in [-1, 1].

    Both loops respond to the altitude error (target_altitude - z):
    - pid1: altitude error → power  (coarse, slow)
    - pid2: altitude error → stick  (fine, fast pitch correction)

    Sign: both positive — higher altitude error → more power / nose-up stick.
    dt = env delta_t = 1.0.
    Gains read from data/pid_gains.json; keyword arguments override.
    """
    _pl = _load_gains().get("plane", {})
    _p1 = _pl.get("pid1", {})
    _p2 = _pl.get("pid2", {})
    Kp1 = Kp1 if Kp1 is not None else float(_p1.get("Kp", 0.0002))
    Ki1 = Ki1 if Ki1 is not None else float(_p1.get("Ki", 0.000005))
    Kd1 = Kd1 if Kd1 is not None else float(_p1.get("Kd", 0.0))
    Kp2 = Kp2 if Kp2 is not None else float(_p2.get("Kp", 0.0005))
    Ki2 = Ki2 if Ki2 is not None else float(_p2.get("Ki", 0.00001))
    Kd2 = Kd2 if Kd2 is not None else float(_p2.get("Kd", 0.001))
    params = MIMOPIDParams(
        pid1=PIDParams(
            Kp=Kp1,
            Ki=Ki1,
            Kd=Kd1,
            dt=1.0,
            state_index=1,  # z (altitude)
            setpoint_index=6,  # target_altitude
            action_min=-1.0,
            action_max=1.0,
        ),
        pid2=PIDParams(
            Kp=Kp2,
            Ki=Ki2,
            Kd=Kd2,
            dt=1.0,
            state_index=1,  # z (altitude)
            setpoint_index=6,  # target_altitude
            action_min=-1.0,
            action_max=1.0,
        ),
    )
    return params, mimo_pid_reset(params)


# ---------------------------------------------------------------------------
# Per-environment Python stateful factories  →  StatefulPID / StatefulMIMOPID
# (for video / gymnasium rollouts)
# ---------------------------------------------------------------------------


def make_cstr_stateful_pid() -> StatefulPID:
    """obs: [Ca, T, target_Ca]  (full get_obs layout). Gains from data/pid_gains.json."""
    return StatefulPID(
        Kp=_g("cstr", "Kp", -103.6),
        Ki=_g("cstr", "Ki", -1.86),
        Kd=_g("cstr", "Kd", -26.87),
        dt=0.25,
        state_index=0,
        setpoint_index=2,
    )


def make_first_order_stateful_pid() -> StatefulPID:
    """obs: [x, target_x]  (full get_obs layout). Gains from data/pid_gains.json."""
    return StatefulPID(
        Kp=_g("first_order", "Kp", 2.11),
        Ki=_g("first_order", "Ki", 9.72),
        Kd=_g("first_order", "Kd", 0.0039),
        dt=0.05,
        state_index=0,
        setpoint_index=1,
    )


def make_four_tank_stateful_pid() -> StatefulMIMOPID:
    """obs: [h1, h2, h3, h4, target_h1, target_h2]  (full get_obs layout). Gains from data/pid_gains.json."""
    _ft = _load_gains().get("four_tank", {})
    _p1 = _ft.get("pid1", {})
    _p2 = _ft.get("pid2", {})
    pid1 = StatefulPID(
        Kp=float(_p1.get("Kp", 6.74)),
        Ki=float(_p1.get("Ki", 1.34)),
        Kd=float(_p1.get("Kd", 0.0)),
        dt=1.0,
        state_index=0,
        setpoint_index=4,
    )
    pid2 = StatefulPID(
        Kp=float(_p2.get("Kp", 15.29)),
        Ki=float(_p2.get("Ki", 2.59)),
        Kd=float(_p2.get("Kd", 0.0)),
        dt=1.0,
        state_index=1,
        setpoint_index=5,
    )
    return StatefulMIMOPID(pid1, pid2)


def make_glass_furnace_stateful_pid() -> StatefulPID:
    """obs: [T_crown, fuel_flow, target_T_crown]  (full get_obs layout). Gains from data/pid_gains.json."""
    return StatefulPID(
        Kp=_g("glass_furnace", "Kp", 0.01),
        Ki=_g("glass_furnace", "Ki", 0.001),
        Kd=_g("glass_furnace", "Kd", 0.0),
        dt=30.0,
        state_index=0,
        setpoint_index=2,
    )


def make_reactor_stateful_pid() -> StatefulPID:
    """obs: [n, T_coolant, rho_ext_norm, target_n]  (full get_obs layout). Gains from data/pid_gains.json."""
    return StatefulPID(
        Kp=_g("reactor", "Kp", 5.0),
        Ki=_g("reactor", "Ki", 0.5),
        Kd=_g("reactor", "Kd", 0.0),
        dt=0.5,
        state_index=0,
        setpoint_index=3,
    )


def make_plane_stateful_pid() -> StatefulMIMOPID:
    """
    obs: [x_dot, z, z_dot, theta, theta_dot, gamma, target_altitude, power, stick]
    MIMO PID: both power (pid1) and stick (pid2) track altitude error.
    Gains read from data/pid_gains.json.
    """
    _pl = _load_gains().get("plane", {})
    _p1 = _pl.get("pid1", {})
    _p2 = _pl.get("pid2", {})
    pid1 = StatefulPID(
        Kp=float(_p1.get("Kp", 0.0002)),
        Ki=float(_p1.get("Ki", 0.000005)),
        Kd=float(_p1.get("Kd", 0.0)),
        dt=1.0,
        state_index=1,
        setpoint_index=6,
    )
    pid2 = StatefulPID(
        Kp=float(_p2.get("Kp", 0.0005)),
        Ki=float(_p2.get("Ki", 0.00001)),
        Kd=float(_p2.get("Kd", 0.001)),
        dt=1.0,
        state_index=1,
        setpoint_index=6,
    )
    return StatefulMIMOPID(pid1, pid2)


# ---------------------------------------------------------------------------
# Per-environment gain-scheduled factories (relay autotuning)
# ---------------------------------------------------------------------------
#
# These read from the ``gain_schedule`` sub-dict in data/pid_gains.json.
# If no gain schedule is available, they fall back to the flat-gain factories
# above (wrapping them in a 1-point "schedule" for API compatibility).


def _gs_params_from_json(
    env_name: str,
    dt: float,
    state_index: int,
    setpoint_index: int,
    fallback_Kp: float = 0.0,
    fallback_Ki: float = 0.0,
    fallback_Kd: float = 0.0,
) -> GainSchedulePIDParams:
    """Build a GainSchedulePIDParams from the JSON gain_schedule entry."""
    gs = _load_gain_schedule(env_name)
    if gs is not None:
        ops = jnp.array(gs["operating_points"])
        Kp = jnp.array(gs["Kp"])
        Ki = jnp.array(gs["Ki"])
        Kd = jnp.array(gs["Kd"])
    else:
        # Fallback: single-point schedule from flat gains
        ops = jnp.array([0.0])
        Kp = jnp.array([_g(env_name, "Kp", fallback_Kp)])
        Ki = jnp.array([_g(env_name, "Ki", fallback_Ki)])
        Kd = jnp.array([_g(env_name, "Kd", fallback_Kd)])
    return GainSchedulePIDParams(
        operating_points=ops,
        Kp_table=Kp,
        Ki_table=Ki,
        Kd_table=Kd,
        dt=dt,
        state_index=state_index,
        setpoint_index=setpoint_index,
    )


def _stateful_gs_from_json(
    env_name: str,
    dt: float,
    state_index: int,
    setpoint_index: int,
    fallback_Kp: float = 0.0,
    fallback_Ki: float = 0.0,
    fallback_Kd: float = 0.0,
) -> StatefulGainScheduledPID:
    """Build a StatefulGainScheduledPID from the JSON gain_schedule entry."""
    gs = _load_gain_schedule(env_name)
    if gs is not None:
        ops = np.array(gs["operating_points"])
        Kp = np.array(gs["Kp"])
        Ki = np.array(gs["Ki"])
        Kd = np.array(gs["Kd"])
    else:
        ops = np.array([0.0])
        Kp = np.array([_g(env_name, "Kp", fallback_Kp)])
        Ki = np.array([_g(env_name, "Ki", fallback_Ki)])
        Kd = np.array([_g(env_name, "Kd", fallback_Kd)])
    return StatefulGainScheduledPID(
        operating_points=ops,
        Kp_table=Kp,
        Ki_table=Ki,
        Kd_table=Kd,
        dt=dt,
        state_index=state_index,
        setpoint_index=setpoint_index,
    )


def make_cstr_gain_scheduled_pid() -> tuple[GainSchedulePIDParams, PIDState]:
    p = _gs_params_from_json(
        "cstr",
        dt=0.25,
        state_index=0,
        setpoint_index=2,
        fallback_Kp=-103.6,
        fallback_Ki=-1.86,
        fallback_Kd=-26.87,
    )
    return p, PIDState(integral=jnp.zeros(()), prev_error=jnp.zeros(()))


def make_first_order_gain_scheduled_pid() -> tuple[GainSchedulePIDParams, PIDState]:
    p = _gs_params_from_json(
        "first_order",
        dt=0.05,
        state_index=0,
        setpoint_index=1,
        fallback_Kp=2.11,
        fallback_Ki=9.72,
        fallback_Kd=0.0039,
    )
    return p, PIDState(integral=jnp.zeros(()), prev_error=jnp.zeros(()))


def make_reactor_gain_scheduled_pid() -> tuple[GainSchedulePIDParams, PIDState]:
    p = _gs_params_from_json(
        "reactor",
        dt=0.5,
        state_index=0,
        setpoint_index=3,
        fallback_Kp=5.0,
        fallback_Ki=0.5,
        fallback_Kd=0.0,
    )
    return p, PIDState(integral=jnp.zeros(()), prev_error=jnp.zeros(()))


def make_glass_furnace_gain_scheduled_pid() -> tuple[GainSchedulePIDParams, PIDState]:
    p = _gs_params_from_json(
        "glass_furnace",
        dt=30.0,
        state_index=0,
        setpoint_index=2,
        fallback_Kp=0.01,
        fallback_Ki=0.001,
        fallback_Kd=0.0,
    )
    return p, PIDState(integral=jnp.zeros(()), prev_error=jnp.zeros(()))


def make_cstr_stateful_gs_pid() -> StatefulGainScheduledPID:
    return _stateful_gs_from_json(
        "cstr",
        dt=0.25,
        state_index=0,
        setpoint_index=2,
        fallback_Kp=-103.6,
        fallback_Ki=-1.86,
        fallback_Kd=-26.87,
    )


def make_first_order_stateful_gs_pid() -> StatefulGainScheduledPID:
    return _stateful_gs_from_json(
        "first_order",
        dt=0.05,
        state_index=0,
        setpoint_index=1,
        fallback_Kp=2.11,
        fallback_Ki=9.72,
        fallback_Kd=0.0039,
    )


def make_reactor_stateful_gs_pid() -> StatefulGainScheduledPID:
    return _stateful_gs_from_json(
        "reactor",
        dt=0.5,
        state_index=0,
        setpoint_index=3,
        fallback_Kp=5.0,
        fallback_Ki=0.5,
        fallback_Kd=0.0,
    )


def make_glass_furnace_stateful_gs_pid() -> StatefulGainScheduledPID:
    return _stateful_gs_from_json(
        "glass_furnace",
        dt=30.0,
        state_index=0,
        setpoint_index=2,
        fallback_Kp=0.01,
        fallback_Ki=0.001,
        fallback_Kd=0.0,
    )


# ---------------------------------------------------------------------------
# MIMO gain-scheduled factories (four_tank, plane)
# ---------------------------------------------------------------------------
#
# These read from ``gain_schedule_pid1`` / ``gain_schedule_pid2`` sub-dicts.
# If no gain schedule data is present, they fall back to the flat-gain MIMO
# factories above.


def _load_mimo_gain_schedule(env: str, pid_key: str) -> dict | None:
    """Return ``gain_schedule_pid1`` or ``gain_schedule_pid2`` sub-dict, or None."""
    return _load_gains().get(env, {}).get(pid_key, None)


def _stateful_gs_mimo_from_json(
    env_name: str,
    dt: float,
    state_index_1: int,
    setpoint_index_1: int,
    state_index_2: int,
    setpoint_index_2: int,
    fallback_pid1: tuple[float, float, float],
    fallback_pid2: tuple[float, float, float],
    gs_key1: str = "gain_schedule_pid1",
    gs_key2: str = "gain_schedule_pid2",
) -> StatefulMIMOPID:
    """Build a StatefulMIMOPID with gain-scheduled sub-PIDs from JSON."""
    gs1 = _load_mimo_gain_schedule(env_name, gs_key1)
    gs2 = _load_mimo_gain_schedule(env_name, gs_key2)

    if gs1 is not None:
        pid1 = StatefulGainScheduledPID(
            operating_points=np.array(gs1["operating_points"]),
            Kp_table=np.array(gs1["Kp"]),
            Ki_table=np.array(gs1["Ki"]),
            Kd_table=np.array(gs1["Kd"]),
            dt=dt,
            state_index=state_index_1,
            setpoint_index=setpoint_index_1,
        )
    else:
        _env = _load_gains().get(env_name, {})
        _p1 = _env.get("pid1", {})
        pid1 = StatefulPID(
            Kp=float(_p1.get("Kp", fallback_pid1[0])),
            Ki=float(_p1.get("Ki", fallback_pid1[1])),
            Kd=float(_p1.get("Kd", fallback_pid1[2])),
            dt=dt,
            state_index=state_index_1,
            setpoint_index=setpoint_index_1,
        )

    if gs2 is not None:
        pid2 = StatefulGainScheduledPID(
            operating_points=np.array(gs2["operating_points"]),
            Kp_table=np.array(gs2["Kp"]),
            Ki_table=np.array(gs2["Ki"]),
            Kd_table=np.array(gs2["Kd"]),
            dt=dt,
            state_index=state_index_2,
            setpoint_index=setpoint_index_2,
        )
    else:
        _env = _load_gains().get(env_name, {})
        _p2 = _env.get("pid2", {})
        pid2 = StatefulPID(
            Kp=float(_p2.get("Kp", fallback_pid2[0])),
            Ki=float(_p2.get("Ki", fallback_pid2[1])),
            Kd=float(_p2.get("Kd", fallback_pid2[2])),
            dt=dt,
            state_index=state_index_2,
            setpoint_index=setpoint_index_2,
        )

    return StatefulMIMOPID(pid1, pid2)


def make_four_tank_stateful_gs_pid() -> StatefulMIMOPID:
    """Gain-scheduled MIMO PID for FourTank. obs: [h1, h2, h3, h4, target_h1, target_h2]."""
    return _stateful_gs_mimo_from_json(
        "four_tank",
        dt=1.0,
        state_index_1=0,
        setpoint_index_1=4,
        state_index_2=1,
        setpoint_index_2=5,
        fallback_pid1=(6.74, 1.34, 0.0),
        fallback_pid2=(15.29, 2.59, 0.0),
    )


def make_plane_stateful_gs_pid() -> StatefulMIMOPID:
    """Gain-scheduled MIMO PID for Airplane2D. pid1=power, pid2=stick, both on altitude."""
    return _stateful_gs_mimo_from_json(
        "plane",
        dt=1.0,
        state_index_1=1,
        setpoint_index_1=6,  # z, target_altitude
        state_index_2=1,
        setpoint_index_2=6,
        fallback_pid1=(0.0002, 0.000005, 0.0),
        fallback_pid2=(0.0005, 0.00001, 0.001),
    )


# ---------------------------------------------------------------------------
# 3D plane controllers (3 actions: power, stick, aileron)
# ---------------------------------------------------------------------------
#
# All three tasks share an altitude loop (PID on z → stick) and use a
# task-specific lateral loop (heading / circle radius / figure-8 phase →
# desired bank → aileron). Gains are stored under "plane3d_<task>" in
# data/pid_gains.json so each task can be tuned independently.
#
# Structure of the gains dict for each task::
#
#   {
#     "alt":   {"Kp": ..., "Ki": ..., "Kd": ...},   # altitude → stick
#     "lat":   {"Kp": ..., "Ki": ..., "Kd": ...},   # task error → desired bank
#     "bank":  {"Kp": ...},                         # bank error → aileron
#     "power": <float>,                             # fixed cruise throttle
#   }
#
# For the figure-8 task an additional ``period`` field controls the bank
# oscillation period.


def _wrap_angle_np(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _wrap_angle_jnp(a):
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


# ── obs index conventions (see plane3d/env.py) ────────────────────────────
# heading obs: [x_dot, y_dot, z, z_dot, theta, theta_dot, phi, phi_dot,
#               gamma, psi, target_altitude, target_heading, power, stick, aileron]
#                 0     1   2    3      4         5       6      7
#                 8     9       10              11           12     13     14
# circle  obs: [..., target_altitude, rel_x, rel_y, target_radius, ...]
#                                 10     11     12       13
# fig8    obs: [..., target_altitude, rel_x, rel_y, target_radius, dist, ...]
#                                 10     11     12       13         14


class StatefulPlane3DHeadingPID:
    """
    Heading task PID:
      stick   = altitude PID(z, target_altitude)
      aileron = bank-error PD where desired_bank = lateral PID(heading_err)
      power   = fixed cruise throttle
    """

    def __init__(
        self,
        Kp_alt: float,
        Ki_alt: float,
        Kd_alt: float,
        Kp_hdg: float,
        Ki_hdg: float,
        Kd_hdg: float,
        Kp_bank: float,
        power: float = 0.6,
        max_bank_rad: float = np.deg2rad(25.0),
        dt: float = 1.0,
    ):
        self.alt = StatefulPID(
            Kp=Kp_alt,
            Ki=Ki_alt,
            Kd=Kd_alt,
            dt=dt,
            state_index=2,
            setpoint_index=10,
            action_min=-1.0,
            action_max=1.0,
        )
        self.Kp_hdg, self.Ki_hdg, self.Kd_hdg = Kp_hdg, Ki_hdg, Kd_hdg
        self.Kp_bank = Kp_bank
        self.power = power
        self.max_bank_rad = float(max_bank_rad)
        self.dt = dt
        self._hdg_int = 0.0
        self._hdg_prev = 0.0

    def reset(self):
        self.alt.reset()
        self._hdg_int = 0.0
        self._hdg_prev = 0.0

    def step(self, obs) -> np.ndarray:
        stick = self.alt.step(obs)
        psi = float(obs[9])
        target_heading = float(obs[11])
        phi = float(obs[6])

        hdg_err = _wrap_angle_np(target_heading - psi)
        self._hdg_int += hdg_err * self.dt
        deriv = (hdg_err - self._hdg_prev) / self.dt
        desired_bank = (
            self.Kp_hdg * hdg_err + self.Ki_hdg * self._hdg_int + self.Kd_hdg * deriv
        )
        desired_bank = float(
            np.clip(desired_bank, -self.max_bank_rad, self.max_bank_rad)
        )
        bank_err = phi - desired_bank
        aileron = float(np.clip(self.Kp_bank * bank_err, -1.0, 1.0))
        # Anti-windup on heading integrator
        if abs(aileron) >= 1.0:
            self._hdg_int -= hdg_err * self.dt
        self._hdg_prev = hdg_err
        return np.array([self.power, stick, aileron])


class StatefulPlane3DCirclePID:
    """
    Circle task PID:
      stick   = altitude PID
      aileron = bank-error P controller where desired_bank is computed from
                coordinated-turn physics (φ = atan(v²/gr)) plus a PID
                correction on radial error to stay on the target circle.
      power   = fixed cruise throttle
    """

    def __init__(
        self,
        Kp_alt: float,
        Ki_alt: float,
        Kd_alt: float,
        Kp_rad: float,
        Ki_rad: float,
        Kd_rad: float,
        Kp_bank: float,
        power: float = 0.6,
        target_bank_rad: float = np.deg2rad(15.0),
        max_bank_rad: float = np.deg2rad(30.0),
        dt: float = 1.0,
        gravity: float = 9.81,
    ):
        self.alt = StatefulPID(
            Kp=Kp_alt,
            Ki=Ki_alt,
            Kd=Kd_alt,
            dt=dt,
            state_index=2,
            setpoint_index=10,
            action_min=-1.0,
            action_max=1.0,
        )
        self.Kp_rad, self.Ki_rad, self.Kd_rad = Kp_rad, Ki_rad, Kd_rad
        self.Kp_bank = Kp_bank
        self.power = power
        self.max_bank_rad = float(max_bank_rad)
        self.dt = dt
        self.gravity = gravity
        self._rad_int = 0.0
        self._rad_prev = 0.0

    def reset(self):
        self.alt.reset()
        self._rad_int = 0.0
        self._rad_prev = 0.0

    def step(self, obs) -> np.ndarray:
        stick = self.alt.step(obs)
        x_dot = float(obs[0])
        y_dot = float(obs[1])
        rel_x = float(obs[11])
        rel_y = float(obs[12])
        radius = float(obs[13])
        phi = float(obs[6])

        # Physics-based ideal bank for a coordinated turn at current speed
        speed_sq = x_dot**2 + y_dot**2 + 1e-6
        ideal_bank = float(np.arctan2(speed_sq, self.gravity * max(radius, 1.0)))

        # Signed radial error: positive = outside the circle
        dist = float(np.sqrt(rel_x**2 + rel_y**2))
        rad_err = dist - radius

        self._rad_int += rad_err * self.dt
        deriv = (rad_err - self._rad_prev) / self.dt
        # PID on radial error → bank correction (outside → bank more)
        bank_correction = (
            self.Kp_rad * rad_err + self.Ki_rad * self._rad_int + self.Kd_rad * deriv
        )
        desired_bank = ideal_bank + bank_correction
        desired_bank = float(
            np.clip(desired_bank, -self.max_bank_rad, self.max_bank_rad)
        )
        bank_err = phi - desired_bank
        aileron = float(np.clip(self.Kp_bank * bank_err, -1.0, 1.0))
        if abs(aileron) >= 1.0:
            self._rad_int -= rad_err * self.dt
        self._rad_prev = rad_err
        return np.array([self.power, stick, aileron])


class StatefulPlane3DFigureEightPID:
    """
    Figure-8 task PID for the twisted 3D lemniscate:
      stick   = PID on nearest_dz (altitude error to curve)
      aileron = heading PID that blends tangent heading (when on-curve)
                with correction heading (when off-curve), via bank angle
      power   = fixed cruise throttle

    Obs layout (19 values):
      [x_dot, y_dot, z, z_dot, theta, theta_dot, phi, phi_dot,
       gamma, psi, target_altitude, target_radius,
       nearest_dx, nearest_dy, nearest_dz, tangent_heading,
       power, stick, aileron]
    """

    def __init__(
        self,
        Kp_alt: float,
        Ki_alt: float,
        Kd_alt: float,
        Kp_hdg: float,
        Ki_hdg: float,
        Kd_hdg: float,
        Kp_bank: float,
        power: float = 0.6,
        max_bank_rad: float = np.deg2rad(25.0),
        dt: float = 1.0,
    ):
        self.Kp_alt = Kp_alt
        self.Ki_alt = Ki_alt
        self.Kd_alt = Kd_alt
        self.Kp_hdg = Kp_hdg
        self.Ki_hdg = Ki_hdg
        self.Kd_hdg = Kd_hdg
        self.Kp_bank = Kp_bank
        self.power = power
        self.max_bank_rad = float(max_bank_rad)
        self.dt = dt
        self._alt_int = 0.0
        self._alt_prev = 0.0
        self._hdg_int = 0.0
        self._hdg_prev = 0.0

    def reset(self):
        self._alt_int = 0.0
        self._alt_prev = 0.0
        self._hdg_int = 0.0
        self._hdg_prev = 0.0

    def step(self, obs) -> np.ndarray:
        psi = float(obs[9])
        phi = float(obs[6])
        target_radius = float(obs[11])
        nearest_dx = float(obs[12])
        nearest_dy = float(obs[13])
        nearest_dz = float(obs[14])
        tangent_heading = float(obs[15])

        # ── Altitude: PID on nearest_dz (curve_z - aircraft_z) ──
        alt_err = nearest_dz  # positive = curve is above → climb
        self._alt_int += alt_err * self.dt
        alt_d = (alt_err - self._alt_prev) / self.dt
        stick = float(np.clip(
            self.Kp_alt * alt_err + self.Ki_alt * self._alt_int + self.Kd_alt * alt_d,
            -1.0, 1.0,
        ))
        if abs(stick) >= 1.0:
            self._alt_int -= alt_err * self.dt
        self._alt_prev = alt_err

        # ── Heading: blend tangent (on-curve) with correction (off-curve) ──
        lateral_dist = float(np.sqrt(nearest_dx**2 + nearest_dy**2 + 1e-6))
        # blend: 0 = on curve (follow tangent), 1 = far off (chase nearest)
        blend = float(np.clip(lateral_dist / (0.05 * max(target_radius, 1.0)), 0.0, 1.0))
        correction_heading = float(np.arctan2(nearest_dy, nearest_dx))
        # Blend unit vectors to avoid angle wrapping issues
        bx = blend * np.cos(correction_heading) + (1.0 - blend) * np.cos(tangent_heading)
        by = blend * np.sin(correction_heading) + (1.0 - blend) * np.sin(tangent_heading)
        desired_heading = float(np.arctan2(by, bx))

        hdg_err = float(np.arctan2(
            np.sin(desired_heading - psi), np.cos(desired_heading - psi)
        ))
        self._hdg_int += hdg_err * self.dt
        hdg_d = (hdg_err - self._hdg_prev) / self.dt
        desired_bank = (
            self.Kp_hdg * hdg_err
            + self.Ki_hdg * self._hdg_int
            + self.Kd_hdg * hdg_d
        )
        desired_bank = float(
            np.clip(desired_bank, -self.max_bank_rad, self.max_bank_rad)
        )
        bank_err = phi - desired_bank
        aileron = float(np.clip(self.Kp_bank * bank_err, -1.0, 1.0))
        if abs(aileron) >= 1.0:
            self._hdg_int -= hdg_err * self.dt
        self._hdg_prev = hdg_err
        return np.array([self.power, stick, aileron])


def _g3d(task: str, group: str, key: str, default: float) -> float:
    """Read a plane3d gain: gains["plane3d_<task>"][group][key]."""
    return float(
        _load_gains().get(f"plane3d_{task}", {}).get(group, {}).get(key, default)
    )


def make_plane3d_heading_stateful_pid() -> StatefulPlane3DHeadingPID:
    """Heading task PID. Gains read from data/pid_gains.json under "plane3d_heading"."""
    cruise = float(_load_gains().get("plane3d_heading", {}).get("power", 0.6))
    return StatefulPlane3DHeadingPID(
        Kp_alt=_g3d("heading", "alt", "Kp", 0.0005),
        Ki_alt=_g3d("heading", "alt", "Ki", 1e-5),
        Kd_alt=_g3d("heading", "alt", "Kd", 0.001),
        Kp_hdg=_g3d("heading", "hdg", "Kp", 0.5),
        Ki_hdg=_g3d("heading", "hdg", "Ki", 0.0),
        Kd_hdg=_g3d("heading", "hdg", "Kd", 0.0),
        Kp_bank=_g3d("heading", "bank", "Kp", -2.0),
        power=cruise,
    )


def make_plane3d_circle_stateful_pid() -> StatefulPlane3DCirclePID:
    """Circle task PID. Gains read from data/pid_gains.json under "plane3d_circle"."""
    cruise = float(_load_gains().get("plane3d_circle", {}).get("power", 0.6))
    return StatefulPlane3DCirclePID(
        Kp_alt=_g3d("circle", "alt", "Kp", 0.0005),
        Ki_alt=_g3d("circle", "alt", "Ki", 1e-5),
        Kd_alt=_g3d("circle", "alt", "Kd", 0.001),
        Kp_rad=_g3d("circle", "rad", "Kp", 1e-5),
        Ki_rad=_g3d("circle", "rad", "Ki", 0.0),
        Kd_rad=_g3d("circle", "rad", "Kd", 0.0),
        Kp_bank=_g3d("circle", "bank", "Kp", -2.0),
        power=cruise,
    )


def make_plane3d_figure8_stateful_pid() -> StatefulPlane3DFigureEightPID:
    """Figure-8 task PID. Gains read from data/pid_gains.json under "plane3d_figure8".

    Now uses heading-chasing (same structure as heading PID) to follow the
    moving reference point.  Falls back to heading gains if figure8-specific
    gains are absent.
    """
    cruise = float(_load_gains().get("plane3d_figure8", {}).get("power", 0.6))
    return StatefulPlane3DFigureEightPID(
        Kp_alt=_g3d("figure8", "alt", "Kp", 0.0005),
        Ki_alt=_g3d("figure8", "alt", "Ki", 1e-5),
        Kd_alt=_g3d("figure8", "alt", "Kd", 0.001),
        Kp_hdg=_g3d("figure8", "hdg", "Kp", 0.5),
        Ki_hdg=_g3d("figure8", "hdg", "Ki", 0.0),
        Kd_hdg=_g3d("figure8", "hdg", "Kd", 0.0),
        Kp_bank=_g3d("figure8", "bank", "Kp", -2.0),
        power=cruise,
    )
