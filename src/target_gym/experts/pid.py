"""PID controllers: the shared loops, the forms they run in, and the gains store.

A :class:`Controller` is gains, an initial memory and one pure step function,
so it can sit inside ``jit``, ``vmap`` and ``lax.scan``. It runs in two forms,
and docs/using-the-pids.md says which to pick:

- :class:`HostController` (``env.make_pid()``) keeps its own memory; call it
  once per step and ``reset()`` it at each episode start.
- The pure form (``spec.make_pid_controller()``) has you carry the memory;
  :class:`FunctionalExpertPolicy` (``env.expert_policy``) is it batched.

Each environment's PID is built in its package's ``experts.py`` from the loops
here (``siso_pid``, ``mimo_pid``, the aircraft and patrol laws). The gains are
tuned by ``scripts/tune_pid.py`` (relay autotuning, AMIGO rule by default) and
read from ``src/target_gym/data/pid_gains.json``.
"""

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

# ---------------------------------------------------------------------------
# Persistent gains store
# ---------------------------------------------------------------------------

# Resolved relative to this file so it works regardless of cwd.
# ``parents[1]`` is the package, so this resolves inside it whether the code
# is run from a checkout or an installed wheel. It was ``parents[3]``, the
# repository root from a checkout and the interpreter's lib directory from
# site-packages, so an installed target-gym announced "No PID gains file
# found at /usr/local/lib/python3.13/data/pid_gains.json" and silently
# re-ran gradient tuning for minutes on every user's first call.
_GAINS_FILE = pathlib.Path(__file__).resolve().parents[1] / "data" / "pid_gains.json"
_gains_cache: dict | None = None


def _load_gains() -> dict:
    """Load gains from src/target_gym/data/pid_gains.json, running tuning first if the
    file is absent.
    """
    global _gains_cache
    if _gains_cache is None:
        if not _GAINS_FILE.exists():
            print(
                f"[target_gym] No PID gains file found at {_GAINS_FILE}. "
                "Running gradient-based tuning — this may take a few minutes..."
            )
            from target_gym.experts.pid_tuning import tune_all_and_save

            tune_all_and_save(verbose=True)
        with open(_GAINS_FILE) as f:
            _gains_cache = json.load(f)
    return _gains_cache


def _g(env: str, key: str, default: float) -> float:
    """Return gain ``key`` for ``env`` from the JSON store, or ``default``."""
    return float(_load_gains().get(env, {}).get(key, default))


# ---------------------------------------------------------------------------
# One controller, two runtimes
# ---------------------------------------------------------------------------


@struct.dataclass
class Controller:
    """A controller as data: one pure step function and what it carries.

    ``step(params, state, obs, xp=jnp) -> (action, new_state)`` is written once
    against an array module. Under ``jnp`` it traces, so it can sit inside
    ``jit``, ``vmap`` and ``lax.scan`` and run on GPU; under ``numpy`` it is the
    host controller every baseline is scored with. Running both from the same
    function is what keeps the trainable controller and the scored one equal.

    ``params`` and ``init_state`` are pytrees. Keep their floats as Python
    floats: NumPy treats those as weakly typed, so the host arithmetic stays in
    the observation's float32 exactly as the scored controller ran it.
    """

    params: object
    init_state: object
    step: object = struct.field(pytree_node=False)

    def act(self, state, obs):
        """One traced step for a single environment; vmap it for a batch.

        Pure, not memoryless: the PID's memory is ``state``, which the caller
        passes in and keeps. Start from ``init_state`` and reset with
        ``reset_where_done`` at episode ends.
        """
        return self.step(self.params, state, obs, jnp)

    def host(self) -> "HostController":
        """The stateful NumPy form, as ``make_pid()`` returns it."""
        return HostController(self)

    def batched(self) -> "FunctionalExpertPolicy":
        """The batched JAX form, as ``env.expert_policy`` returns it."""
        return FunctionalExpertPolicy(self.params, self.init_state, self.step)


class HostController:
    """Runs a :class:`Controller` on the host, holding its memory.

    ``reset()`` at the start of every episode, then ``pid(obs) -> action``.
    Every call reads and replaces ``self.state``, so call it exactly once per
    environment step.
    """

    def __init__(self, controller: Controller):
        self.controller = controller
        self.reset()

    def reset(self):
        self.state = self.controller.init_state

    def __getattr__(self, name):
        # Gains read as attributes (``pid.Kp_pitch``), as callers expect.
        if name == "controller":
            raise AttributeError(name)
        return getattr(self.controller.params, name)

    def step(self, obs):
        ctrl = self.controller
        action, self.state = ctrl.step(ctrl.params, self.state, obs, np)
        return action

    __call__ = step


def siso_pid(
    Kp: float,
    Ki: float,
    Kd: float,
    dt: float,
    state_index: int,
    setpoint_index: int,
    action_min: float = -1.0,
    action_max: float = 1.0,
) -> Controller:
    """One PID loop on ``obs[setpoint_index] - obs[state_index]``."""
    params = PIDParams(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        dt=dt,
        state_index=state_index,
        setpoint_index=setpoint_index,
        action_min=action_min,
        action_max=action_max,
    )
    return Controller(params, PIDState(integral=0.0, prev_error=0.0), pid_step)


def mimo_pid(loop1: Controller, loop2: Controller) -> Controller:
    """Two independent loops from :func:`siso_pid`, one per action."""
    return Controller(
        MIMOPIDParams(pid1=loop1.params, pid2=loop2.params),
        MIMOPIDState(state1=loop1.init_state, state2=loop2.init_state),
        mimo_pid_step,
    )


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


def pid_update(e, integral, prev_error, Kp, Ki, Kd, dt, action_min, action_max, xp):
    """The PID recurrence itself, over whichever array module is passed.

    This exists because it was once written twice, in ``pid_step`` for tracing
    and in a host-side class for the Python loop, with the anti-windup
    spelled two different but equivalent ways. That is the duplication check 12
    of the model review checklist is about, and the environments already avoid
    it -- ``compute_reward(state, params, xp=jnp)`` is one function read by
    ``numpy`` on the host and by ``jax`` under trace.

    Anti-windup: on saturation the integral is left where it was, so a
    controller pinned at its limit does not wind up a term it cannot spend.

    Returns ``(u_clipped, new_integral, new_prev_error)``.
    """
    new_integral = integral + e * dt
    derivative = (e - prev_error) / dt
    u = Kp * e + Ki * new_integral + Kd * derivative
    u_clipped = xp.clip(u, action_min, action_max)
    new_integral = xp.where(u == u_clipped, new_integral, integral)
    return u_clipped, new_integral, e


def pid_reset(params: PIDParams) -> PIDState:  # noqa: ARG001
    """Return a zeroed initial state for a SISO PID."""
    return PIDState(integral=jnp.zeros(()), prev_error=jnp.zeros(()))


def pid_step(
    params: PIDParams,
    state: PIDState,
    obs: jnp.ndarray,
    xp=jnp,
) -> tuple[jnp.ndarray, PIDState]:
    """
    One discrete PID update step.

    Returns (action [shape (1,)], new_state).
    Anti-windup: the integral is not updated when the output is saturated.
    """
    e = obs[params.setpoint_index] - obs[params.state_index]
    u_clipped, new_integral, new_prev = pid_update(
        e,
        state.integral,
        state.prev_error,
        params.Kp,
        params.Ki,
        params.Kd,
        params.dt,
        params.action_min,
        params.action_max,
        xp,
    )
    return xp.array([u_clipped]), PIDState(integral=new_integral, prev_error=new_prev)


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
    xp=jnp,
) -> tuple[jnp.ndarray, MIMOPIDState]:
    """
    One step of two independent SISO PIDs. Returns (action [shape (2,)], new_state).
    """
    u1, new_state1 = pid_step(params.pid1, state.state1, obs, xp)
    u2, new_state2 = pid_step(params.pid2, state.state2, obs, xp)
    new_state = MIMOPIDState(state1=new_state1, state2=new_state2)
    return xp.concatenate([u1, u2]), new_state


# ---------------------------------------------------------------------------
# JAX-functional 3D plane PIDs  (jit/vmap/scan compatible)
# ---------------------------------------------------------------------------


@struct.dataclass
class Plane3DPIDState:
    """Carry state shared by all 3D plane PID variants."""

    alt_integral: float
    alt_prev: float
    track_integral: (
        float  # heading integral (heading/fig-8) or radial integral (circle)
    )
    track_prev: float
    # Separate integrator for the power loop (heading task MIMO altitude control)
    power_integral: float
    power_prev: float
    # Previous lead heading, for the patrol follower's turn-rate feedforward.
    # Unused by the other variants, and defaulted so their constructors are
    # untouched.
    lead_psi_prev: float = 0.0


@struct.dataclass
class Plane3DHeadingPIDParams:
    """Static params for the heading and figure-8 task PIDs (identical structure).

    The altitude loop is MIMO: altitude error drives BOTH stick (fast pitch) and
    power (slow thrust), mirroring the 2D Airplane2D PID. `power` is a cruise
    bias added on top of the power PID output.
    """

    Kp_alt: float
    Ki_alt: float
    Kd_alt: float
    Kp_hdg: float
    Ki_hdg: float
    Kd_hdg: float
    Kp_bank: float
    power: float  # cruise throttle bias, added to Kp_power·err + ...
    max_bank_rad: float
    dt: float
    Kp_power: float
    Ki_power: float
    Kd_power: float
    Kd_bank: float = 1.5  # roll-rate (phi_dot) damping — kills the bank wobble


def plane3d_pid_reset(params) -> Plane3DPIDState:  # noqa: ARG001
    z = jnp.zeros(())
    return Plane3DPIDState(
        alt_integral=z,
        alt_prev=z,
        track_integral=z,
        track_prev=z,
        power_integral=z,
        power_prev=z,
    )


def plane3d_heading_pid_step(
    params: Plane3DHeadingPIDParams,
    state: Plane3DPIDState,
    obs: jnp.ndarray,
) -> tuple[jnp.ndarray, Plane3DPIDState]:
    """One step of the heading-task PID. obs must be 1-D (vmap handles batching).

    MIMO altitude control: alt_err drives both stick (fast pitch) and power
    (slow throttle, on top of a cruise bias). Heading controlled via bank.
    """
    alt_err = obs[10] - obs[2]
    new_alt_int = state.alt_integral + alt_err * params.dt
    alt_d = (alt_err - state.alt_prev) / params.dt
    stick_u = (
        params.Kp_alt * alt_err + params.Ki_alt * new_alt_int + params.Kd_alt * alt_d
    )
    stick = jnp.clip(stick_u, -1.0, 1.0)
    new_alt_int = jnp.where(stick_u == stick, new_alt_int, state.alt_integral)

    # Power loop: separate integrator, same altitude error signal.
    new_power_int = state.power_integral + alt_err * params.dt
    power_d = (alt_err - state.power_prev) / params.dt
    power_u = (
        params.power
        + params.Kp_power * alt_err
        + params.Ki_power * new_power_int
        + params.Kd_power * power_d
    )
    power = jnp.clip(power_u, -1.0, 1.0)
    new_power_int = jnp.where(power_u == power, new_power_int, state.power_integral)

    psi = obs[9]
    target_heading = obs[11]
    phi = obs[6]
    phi_dot = obs[7]  # roll rate
    hdg_err = _wrap_angle(target_heading - psi)
    new_hdg_int = state.track_integral + hdg_err * params.dt
    hdg_d = (hdg_err - state.track_prev) / params.dt
    desired_bank = jnp.clip(
        params.Kp_hdg * hdg_err + params.Ki_hdg * new_hdg_int + params.Kd_hdg * hdg_d,
        -params.max_bank_rad,
        params.max_bank_rad,
    )
    bank_err = phi - desired_bank
    aileron = jnp.clip(params.Kp_bank * bank_err - params.Kd_bank * phi_dot, -1.0, 1.0)
    new_hdg_int = jnp.where(jnp.abs(aileron) >= 1.0, state.track_integral, new_hdg_int)

    new_state = Plane3DPIDState(
        alt_integral=new_alt_int,
        alt_prev=alt_err,
        track_integral=new_hdg_int,
        track_prev=hdg_err,
        power_integral=new_power_int,
        power_prev=alt_err,
    )
    return jnp.array([power, stick, aileron]), new_state


# ---------------------------------------------------------------------------
# Close-patrol (formation-keeping) pursuit-guidance PID
# ---------------------------------------------------------------------------


@struct.dataclass
class PatrolPIDParams:
    """Static params for the close-patrol follower PID (full-obs layout).

    Pursuit guidance drives the follower into the slot:
      - vertical   : slot up-error -> stick (pitch)
      - along-track: slot back-error -> power (throttle, on a cruise bias)
      - horizontal : slot in-plane error -> desired heading -> bank -> aileron

    The horizontal loop is a *pursuit* controller (position error steers a
    desired heading, not the bank directly), mirroring the circle/figure-8
    experts — direct position->bank bleeds energy in turns and destabilises
    formation-keeping.  When the follower is far from the slot it points at
    it; as it closes (within ``blend_dist``) it blends onto the lead heading
    so the two fly parallel.

    Reads the full-obs vector from
    :func:`target_gym.patrol.env.get_obs_full` (indices documented there).
    """

    Kp_alt: float
    Ki_alt: float
    Kd_alt: float
    Kp_power: float
    Ki_power: float
    Kd_power: float
    cruise: float  # throttle bias (raw [-1, 1])
    Kp_hdg: float
    Ki_hdg: float
    Kd_hdg: float
    Kp_bank: float
    Kd_bank: float  # roll-rate (phi_dot) damping — kills the bank wobble
    max_bank_rad: float
    blend_dist: float  # in-plane distance (m) over which to blend to lead heading
    #: Fraction of the lead's estimated turn rate fed forward. 1.0 is the
    #: physically right value: in a steady turn the follower has to turn at the
    #: lead's rate whatever its position error happens to be. Exposed as a gain
    #: only so it can be swept, and so a tuner can back it off if the estimate
    #: ever gets noisy.
    Kff_lead: float
    #: Clamp on the estimated lead turn rate (rad/s). The estimate is a
    #: one-step difference of the lead heading, and on the very first call
    #: there is no previous sample, so it is bounded to something the lead can
    #: physically fly rather than special-cased.
    lead_rate_max: float
    dt: float


_PATROL_NOMINAL_SPEED = 230.0  # m/s, the follower's cruise

# Standard gravity, for the coordinated-turn relation tan(phi) = v^2 / (g R).
# The circle and patrol lateral laws already take this as a constructor
# argument; the patrol turn-rate conversion below spelled it three times as a
# literal. Named here so the coupling to the plant's own ``gravity`` parameter
# is at least visible: a plant configured with a different one would disagree
# with every controller that hardcodes this.
_G = 9.81


def make_patrol_pid() -> tuple[PatrolPIDParams, Plane3DPIDState]:
    """JAX-functional close-patrol follower PID (pursuit guidance).

    Gains default to hand-set values that hold a trailing slot; they can be
    overridden from ``src/target_gym/data/pid_gains.json`` under the ``patrol`` key.
    """
    _p = _load_gains().get("patrol", {})
    params = PatrolPIDParams(
        Kp_alt=float(_p.get("Kp_alt", 0.02)),
        Ki_alt=float(_p.get("Ki_alt", 1e-4)),
        Kd_alt=float(_p.get("Kd_alt", 0.05)),
        Kp_power=float(_p.get("Kp_power", 0.003)),
        Ki_power=float(_p.get("Ki_power", 3e-5)),
        Kd_power=float(_p.get("Kd_power", 0.15)),
        cruise=float(_p.get("cruise", 0.6)),
        # Re-searched after the heading loop was changed to command turn rate
        # rather than bank. Chosen for survival, not for settled error: they
        # take the follower from 2 of 6 encounters completed to 5 of 6, at a
        # mean slot error of 135 m against the old 130 m. Departing is the
        # worse failure.
        Kp_hdg=float(_p.get("Kp_hdg", 0.284)),
        Ki_hdg=float(_p.get("Ki_hdg", 0.0)),
        Kd_hdg=float(_p.get("Kd_hdg", 1.808)),
        Kp_bank=float(_p.get("Kp_bank", -4.287)),
        Kd_bank=float(_p.get("Kd_bank", 3.124)),
        max_bank_rad=float(np.deg2rad(30.0)),
        blend_dist=float(_p.get("blend_dist", 500.0)),
        Kff_lead=float(_p.get("Kff_lead", 1.0)),
        lead_rate_max=float(_p.get("lead_rate_max", 0.005)),
        dt=1.0,
    )
    return params, plane3d_pid_reset(params)


# ---------------------------------------------------------------------------
# FunctionalExpertPolicy — JAX-compatible, stateless expert wrapper
# ---------------------------------------------------------------------------


# Registry: which PID param fields are learnable gains for each step_fn.
# Other fields (power, max_bank_rad, dt, gravity) are treated as structural
# constants. The order here is the canonical ordering used by the gain-policy
# action pipeline and by anchor_gains.
_LEARNABLE_GAINS_BY_STEP_FN: dict[str, tuple[str, ...]] = {}


def register_learnable_gains(step_fn, fields: tuple[str, ...]) -> None:
    _LEARNABLE_GAINS_BY_STEP_FN[step_fn.__qualname__] = fields


class FunctionalExpertPolicy:
    """Wraps a functional (params, step_fn) PID pair as a JAX-compatible expert policy.

    Interface::

        policy = FunctionalExpertPolicy(params, zero_state, step_fn)
        pid_state = policy.init_state(num_envs)         # batched initial state
        actions, pid_state = policy(pid_state, obs)     # (num_envs, action_dim)

    The step_fn is vmapped over the env batch dimension automatically.
    Always pass the state: the one-argument form ``policy(obs)`` starts every
    call from a fresh state, so it has no integral or derivative memory and is
    not the tuned PID.
    Pass as ``eval_expert_policy`` to Ajax agents; thread ``pid_state``
    through the while-loop carry in ``step_environment_expert``.

    Gain-policy interface (used when SAC's actor outputs PID gains):

        expert.learnable_fields        # tuple of gain field names
        expert.anchor_gains            # jnp.ndarray of expert gain values
        expert.step_with_gains(state, obs, gains)  # per-env gains
    """

    def __init__(self, params, zero_state, step_fn):
        self._params = params
        self._zero_state = zero_state
        self._step_fn = step_fn
        self._vmapped_step = jax.vmap(step_fn, in_axes=(None, 0, 0))
        # vmap over (params, state, obs) for per-env gain overrides.
        self._vmapped_step_per_env_params = jax.vmap(step_fn, in_axes=(0, 0, 0))

    @property
    def learnable_fields(self) -> tuple[str, ...]:
        key = self._step_fn.__qualname__
        if key not in _LEARNABLE_GAINS_BY_STEP_FN:
            raise ValueError(
                f"No learnable gain registry entry for step_fn {key}. "
                "Register with register_learnable_gains(step_fn, fields)."
            )
        return _LEARNABLE_GAINS_BY_STEP_FN[key]

    @property
    def anchor_gains(self) -> jnp.ndarray:
        return jnp.array(
            [float(getattr(self._params, f)) for f in self.learnable_fields]
        )

    def step_with_gains(self, state, obs, gains):
        """Step the PID with per-env learnable gains.

        state: pytree batched (num_envs, ...)
        obs:   (num_envs, obs_dim)
        gains: (num_envs, len(learnable_fields))
        Returns (actions, new_state).
        """
        fields = self.learnable_fields
        n = len(fields)
        # Build per-env params by broadcasting self._params and overriding learnable fields.
        num_envs = gains.shape[0]
        base = jax.tree.map(
            lambda x: jnp.broadcast_to(jnp.asarray(x), (num_envs,) + jnp.shape(x)),
            self._params,
        )
        overrides = {fields[i]: gains[:, i] for i in range(n)}
        per_env_params = base.replace(**overrides)
        return self._vmapped_step_per_env_params(per_env_params, state, obs)

    def init_state(self, num_envs: int):
        """Return zero-initialised state with a leading batch dimension."""
        return jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_envs,) + jnp.shape(x)),
            self._zero_state,
        )

    def __call__(self, *args):
        # Two-arg form: (state, obs) -> (action, new_state). State threaded by caller.
        # One-arg form: (obs,) -> action. Stateless per call (fresh zero state),
        # matching the pre-state-threading interface still used across Ajax agents.
        # It has no memory, so it is not the tuned PID: no integral, and a
        # derivative against a zero last error on every call.
        if len(args) == 2:
            state, obs = args
            return self._vmapped_step(self._params, state, obs)
        (obs,) = args
        batch_shape = obs.shape[:-1]
        flat_batch = 1
        for d in batch_shape:
            flat_batch *= d
        obs_flat = obs.reshape((flat_batch, obs.shape[-1]))
        zero = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (flat_batch,) + jnp.shape(x)),
            self._zero_state,
        )
        action, _ = self._vmapped_step(self._params, zero, obs_flat)
        return action.reshape(batch_shape + action.shape[1:])

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


_PLANE3D_HEADING_LEARNABLE = (
    "Kp_alt",
    "Ki_alt",
    "Kd_alt",
    "Kp_hdg",
    "Ki_hdg",
    "Kd_hdg",
    "Kp_bank",
)
register_learnable_gains(plane3d_heading_pid_step, _PLANE3D_HEADING_LEARNABLE)


def make_plane3d_heading_pid() -> tuple[Plane3DHeadingPIDParams, Plane3DPIDState]:
    """JAX-functional heading-task PID. Gains from src/target_gym/data/pid_gains.json.

    Altitude is MIMO-controlled: alt_err drives both stick (pitch) and power
    (throttle, added on top of a cruise bias), matching the 2D Airplane2D PID.
    """
    params = Plane3DHeadingPIDParams(
        Kp_alt=_g3d("heading", "alt", "Kp", 0.0005),
        Ki_alt=_g3d("heading", "alt", "Ki", 1e-5),
        Kd_alt=_g3d("heading", "alt", "Kd", 0.001),
        Kp_hdg=_g3d("heading", "hdg", "Kp", 0.5),
        Ki_hdg=_g3d("heading", "hdg", "Ki", 0.0),
        Kd_hdg=_g3d("heading", "hdg", "Kd", 0.0),
        Kp_bank=_g3d("heading", "bank", "Kp", -2.0),
        power=float(_load_gains().get("plane3d_heading", {}).get("power", 0.6)),
        max_bank_rad=float(np.deg2rad(25.0)),
        dt=1.0,
        Kp_power=_g3d("heading", "power_pid", "Kp", 2e-4),
        Ki_power=_g3d("heading", "power_pid", "Ki", 5e-6),
        Kd_power=_g3d("heading", "power_pid", "Kd", 0.0),
    )
    return params, plane3d_pid_reset(params)


# ---------------------------------------------------------------------------
# 3D plane controllers (3 actions: power, stick, aileron)
# ---------------------------------------------------------------------------
#
# All three tasks share an altitude loop (PID on z → stick) and use a
# task-specific lateral loop (heading / circle radius / figure-8 phase →
# desired bank → aileron). Gains are stored under "plane3d_<task>" in
# src/target_gym/data/pid_gains.json so each task can be tuned independently.
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


def _wrap_angle(a, xp=jnp):
    """Wrap an angle to (-pi, pi], over whichever array module is passed.

    This was written twice, and the two were not the same function. The traced
    form used ``arctan2(sin, cos)`` and the host form ``(a + pi) % (2 pi) - pi``,
    which agree everywhere except at exactly ``a = pi``, where the first returns
    ``+pi`` and the second ``-pi``. That is a heading error of half a turn
    resolving to opposite signs in the two implementations -- one controller
    turning left where the other turns right -- in a file that keeps a traced
    and a host copy of the same control laws. See check 12 in
    docs/model-review-checklist.md.

    ``arctan2(sin, cos)`` is kept because it is the numerically better of the
    two near the wrap, and it works unchanged under numpy.
    """
    return xp.arctan2(xp.sin(a), xp.cos(a))


def _g3d(task: str, group: str, key: str, default: float) -> float:
    """Read a plane3d gain: gains["plane3d_<task>"][group][key]."""
    return float(
        _load_gains().get(f"plane3d_{task}", {}).get(group, {}).get(key, default)
    )
