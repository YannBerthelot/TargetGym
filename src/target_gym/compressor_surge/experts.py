"""The compressor's expert: the classic performance and anti-surge pair and
the guard its factory applies. The reference NMPC is in ``oracle.py``.

See ``PHYSICS.md`` in this directory, section 7. Everything here is outside
the version stamp and inside the expert fingerprint: the env never imports
this module at load time, so editing a controller never moves
``compressor_surge-v2``. The oracle reads the pair (its cold guess and its
fallback), so this module is in the oracle fingerprint too.

The pair is the arrangement industrial anti-surge systems use. A pressure PI
on the drive's speed command tracks the header setpoint. An anti-surge PI on
the recycle valve holds the compressor's flow-coefficient margin on a control
line 15 % right of the surge line, with external-reset anti-windup that keeps
its integrator within 0.05 of the observed valve, and a full-opening override
below 7.5 % held for 2 s. The only model knowledge it uses is the surge line
(``surge_flow_per_speed``), which a vendor gives every anti-surge controller.

Its factory refuses gains whose linearised closed loop decays too slowly at
five control-line points, or whose stress battery comes too close to the
surge line or leaves the valve in a limit cycle (``check_pair_gains``). That
keeps the tuner's search away from a controller that rings or trips, since
the return sees the distance to the surge line only once a trip happens.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from scipy.optimize import brentq

from target_gym.compressor_surge.env import (
    KPA,
    N_DEMAND_BLOCKS,
    N_SETPOINT_BLOCKS,
    CompressorSurgeParams,
    CompressorSurgeState,
    characteristic,
    check_is_terminal,
    compute_next_state,
    consumer_flow,
    get_obs,
    recycle_flow,
    suction_density,
    surge_flow_per_speed,
)
from target_gym.experts import pid as _pid_mod

try:
    # Newer JAX; 0.10 removed the experimental one. The declared range starts
    # at JAX 0.6, which has only that.
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

# ---------------------------------------------------------------------------
# Gains
# ---------------------------------------------------------------------------

#: The tuner's starting point and the factory's fallback. Every tuned gain is
#: nonzero: the coordinate search multiplies gains, so a zero would never move.
DEFAULT_GAINS: dict = {
    # Speed command (fraction of rated) per kPa of header-pressure error (ours).
    "kp_speed": 0.08,
    # Per kPa s (ours).
    "ki_speed": 0.04,
    # Recycle opening per unit of flow-coefficient margin (ours). Above the
    # nominal gain below 1 that Mirsky et al. (2015) p.13 call typical, which
    # relies on open-loop steps that are modelled here only as the override.
    # At 3.0, after the battery's demand drops, the command swings faster
    # than the valve closes and the valve ratchets against its closing limit
    # in a limit cycle, which the guard's travel check refuses
    # (``GUARD_MAX_TRAVEL``; measured, scripts/compressor_surge_numbers.py
    # --section guard).
    "kp_surge": 2.0,
    # Per unit margin s (ours). At 0.3 the linearised closed loop keeps a mode
    # decaying at 0.083 /s, under ``GUARD_MIN_DECAY`` (derived,
    # scripts/compressor_surge_numbers.py --section guard).
    "ki_surge": 3.0,
    # Control line, as a flow-coefficient margin over the surge line (ours).
    "surge_line": 0.15,
    # Override line (read-consistent): half the control margin, where Mirsky
    # et al. (2015) p.13 place the open-loop line.
    "override_line": 0.075,
    # How long the override holds the valve fully open, in s (ours).
    "override_hold": 2.0,
    # External reset band on the anti-surge integrator, in opening (ours).
    "reset_band": 0.05,
}
#: The gains the tuner searches.
TUNED_GAINS = ("kp_speed", "ki_speed", "kp_surge", "ki_surge")
#: The gains it holds. Their right value is set by the distance to the surge
#: line, which the return sees only once a trip happens.
HELD_GAINS = ("surge_line", "override_line", "override_hold", "reset_band")
#: The pair's control line, which rendering draws.
SURGE_CONTROL_LINE = DEFAULT_GAINS["surge_line"]


def surge_margin_from_obs(obs, params: CompressorSurgeParams):
    """Flow-coefficient margin over the surge line from an observation:
    ``obs[1] / (surge_flow_per_speed * obs[3] / 100) - 1``. The speed channel
    is in percent of rated. Plain arithmetic, so it takes NumPy or JAX."""
    return obs[1] / (surge_flow_per_speed(params) * obs[3] / 100.0) - 1.0


def raw_from_commands(N_cmd, x_cmd, params: CompressorSurgeParams, xp=np):
    """The raw action that commands speed ``N_cmd`` (fraction of rated) and
    recycle opening ``x_cmd``: the inverse of the env's action map."""
    return xp.stack(
        [
            2.0 * (N_cmd - params.N_min) / (params.N_max - params.N_min) - 1.0,
            2.0 * x_cmd - 1.0,
        ]
    )


def commands_from_raw(u_raw, params: CompressorSurgeParams, xp=np):
    """``[N_cmd, x_cmd]`` of raw action ``u_raw``, clipped to [-1, 1] first
    as the env clips it."""
    u = xp.clip(xp.asarray(u_raw), -1.0, 1.0)
    return xp.stack(
        [
            params.N_min + 0.5 * (u[0] + 1.0) * (params.N_max - params.N_min),
            0.5 * (u[1] + 1.0),
        ]
    )


# ---------------------------------------------------------------------------
# The pair: functional core
# ---------------------------------------------------------------------------


class PairState(NamedTuple):
    """The pair's memory. Read by attribute, never unpacked."""

    #: Speed loop's integral term, a speed command (fraction of rated).
    I_N: float
    #: Anti-surge loop's integral term, a recycle opening.
    I_x: float
    #: Time the override still holds the valve open (s).
    timer: float


def pair_init(obs, xp=np) -> PairState:
    """Fresh memory for observation ``obs``: the speed integral at the drive
    setpoint (``obs[4] / 100``), the anti-surge integral at the valve
    (``obs[5] / 100``) and no override. It reads neither gains nor params.

    The speed loop therefore starts bumpless from any state. The valve loop
    does only on its control line, since its first command is
    ``x + kp_surge es``, or full opening below the override line.
    """
    return PairState(
        I_N=obs[4] / 100.0,
        I_x=obs[5] / 100.0,
        timer=xp.zeros_like(obs[0]),
    )


def pair_step(gains, ps: PairState, obs, params: CompressorSurgeParams, xp=np):
    """One update of the pair. Returns ``(u_raw (2,), new memory)``.

    ``obs`` is the env's observation (``env.get_obs``); pressure channels are
    in kPa, speed and valve in percent. With ``xp=jnp`` it traces, so it scans
    and vmaps inside the env's own rollouts.
    """
    dt = params.delta_t

    # Performance PI: header pressure on the speed command. The integral
    # moves unless the command is saturated in the direction of the error.
    e = obs[6] - obs[0]  # kPa
    N_raw = ps.I_N + gains["kp_speed"] * e
    N_cmd = xp.clip(N_raw, params.N_min, params.N_max)
    N_held = ((N_raw >= params.N_max) & (e > 0.0)) | (
        (N_raw <= params.N_min) & (e < 0.0)
    )
    I_N = xp.where(N_held, ps.I_N, ps.I_N + gains["ki_speed"] * e * dt)

    # Anti-surge PI: the margin on the recycle, reverse acting (a margin below
    # the line opens the valve). External reset: the integral is kept within
    # ``reset_band`` of the observed valve, so it cannot wind up while the
    # valve travels at its rate limit.
    x_obs = obs[5] / 100.0
    band = gains["reset_band"]
    margin = surge_margin_from_obs(obs, params)
    es = gains["surge_line"] - margin
    I_x = xp.clip(ps.I_x, x_obs - band, x_obs + band)
    x_raw = I_x + gains["kp_surge"] * es
    x_held = ((x_raw <= 0.0) & (es < 0.0)) | ((x_raw >= 1.0) & (es > 0.0))
    I_x = xp.where(x_held, I_x, I_x + gains["ki_surge"] * es * dt)

    # Override: below the override line the valve opens fully and stays open
    # for ``override_hold`` after the margin recovers. The comparison with
    # half a step rounds the hold to whole steps, so float rounding of the
    # countdown cannot add one.
    timer = xp.where(
        margin < gains["override_line"],
        gains["override_hold"] + 0.0 * ps.timer,
        xp.maximum(ps.timer - dt, 0.0),
    )
    x_cmd = xp.where(timer > 0.5 * dt, 1.0, xp.clip(x_raw, 0.0, 1.0))

    u_raw = raw_from_commands(N_cmd, x_cmd, params, xp)
    return u_raw, PairState(I_N=I_N, I_x=I_x, timer=timer)


def pair_track(
    gains, ps: PairState, obs, u_raw, params: CompressorSurgeParams, xp=np
) -> PairState:
    """Bumpless transfer: the memory that keeps the pair ready to take over
    from a controller that applied ``u_raw`` at ``obs``.

    The speed integral is the one at which the performance PI gives the
    applied speed command, so the speed loop's first command after a
    handover continues it. The anti-surge integral is the one that gives the
    applied recycle command, clipped to the reset band around the observed
    valve; its first command continues the applied one only within
    ``reset_band / kp_surge`` of the control line. The override is cleared.
    """
    e = obs[6] - obs[0]
    N_applied, x_applied = commands_from_raw(u_raw, params, xp)
    x_obs = obs[5] / 100.0
    band = gains["reset_band"]
    es = gains["surge_line"] - surge_margin_from_obs(obs, params)
    return PairState(
        I_N=N_applied - gains["kp_speed"] * e,
        I_x=xp.clip(x_applied - gains["kp_surge"] * es, x_obs - band, x_obs + band),
        timer=xp.zeros_like(ps.timer),
    )


# ---------------------------------------------------------------------------
# The pair: stateful controller and factory
# ---------------------------------------------------------------------------


class CompressorSurgePair:
    """Pressure PI on speed and anti-surge PI on the recycle.

    Observation layout (``env.get_obs``)::

        [dp (kPa), m_c (kg/s), m_d (kg/s), N (%), N_ramp (%), x (%), setpoint (kPa)]

    The action is raw in [-1, 1] on both actuators. The memory persists
    through a trip (the suite's convention). ``track`` is what a supervising
    controller calls on the steps it controls, so that a fallback to this
    pair continues the speed command last applied.
    """

    def __init__(self, gains=None, params: CompressorSurgeParams | None = None):
        gains = dict(DEFAULT_GAINS if gains is None else gains)
        unknown = sorted(set(gains) - set(DEFAULT_GAINS))
        missing = sorted(set(DEFAULT_GAINS) - set(gains))
        if unknown or missing:
            raise ValueError(
                f"pair gains: unknown {unknown}, missing {missing}; "
                f"expected exactly {sorted(DEFAULT_GAINS)}"
            )
        self.gains = {k: float(v) for k, v in gains.items()}
        self.params = CompressorSurgeParams() if params is None else params
        self.reset()

    def reset(self):
        """Clear the memory. The next ``step`` or ``track`` starts it afresh."""
        self._ps = None

    def _memory(self, obs) -> PairState:
        if self._ps is None:
            self._ps = pair_init(obs)
        return self._ps

    def step(self, obs) -> np.ndarray:
        obs = np.asarray(obs, float)
        u_raw, self._ps = pair_step(self.gains, self._memory(obs), obs, self.params)
        return np.asarray(u_raw, float)

    def track(self, obs, u_raw) -> None:
        obs = np.asarray(obs, float)
        self._ps = pair_track(
            self.gains,
            self._memory(obs),
            obs,
            np.asarray(u_raw, float).reshape(2),
            self.params,
        )

    __call__ = step


def load_gains() -> dict:
    """``DEFAULT_GAINS`` overlaid with the numeric entries stored under
    ``"compressor_surge"`` in ``data/pid_gains.json`` (its ``"note"`` is
    skipped).

    Reads ``target_gym.experts.pid.load_gains()`` at call time, so a tuner
    that patches ``_gains_cache`` is seen.
    """
    stored = _pid_mod.load_gains().get("compressor_surge", {})
    return {
        **DEFAULT_GAINS,
        **{
            k: float(v)
            for k, v in stored.items()
            if not isinstance(v, (str, dict, list))
        },
    }


def make_compressor_surge_pid() -> CompressorSurgePair:
    """Registry factory: the stored gains, refused unless the guard passes."""
    gains = load_gains()
    check_pair_gains(gains)
    return CompressorSurgePair(gains)


@struct.dataclass
class CompressorSurgePIDParams:
    """The pair's gains (the ``DEFAULT_GAINS`` keys) and the plant it reads."""

    kp_speed: float
    ki_speed: float
    kp_surge: float
    ki_surge: float
    surge_line: float
    override_line: float
    override_hold: float
    reset_band: float
    plant: CompressorSurgeParams


@struct.dataclass
class CompressorSurgePIDState:
    """``PairState`` plus whether it has been started from an observation."""

    I_N: float
    I_x: float
    timer: float
    started: bool


def compressor_surge_pid_step(params, state, obs, xp=jnp):
    """``CompressorSurgePair.step`` as a pure step: the first call starts the
    memory from ``obs`` (``pair_init``), as the class does lazily."""
    obs = xp.asarray(obs, float)
    gains = {k: getattr(params, k) for k in DEFAULT_GAINS}
    fresh = pair_init(obs, xp)
    ps = PairState(
        I_N=xp.where(state.started, state.I_N, fresh.I_N),
        I_x=xp.where(state.started, state.I_x, fresh.I_x),
        timer=xp.where(state.started, state.timer, fresh.timer),
    )
    u_raw, ps = pair_step(gains, ps, obs, params.plant, xp)
    return u_raw, CompressorSurgePIDState(
        I_N=ps.I_N, I_x=ps.I_x, timer=ps.timer, started=True
    )


def make_compressor_surge_pid_controller() -> _pid_mod.Controller:
    """``make_compressor_surge_pid`` as a ``Controller``: the same stored
    gains, the same guard and the same checks on the gain names."""
    pair = make_compressor_surge_pid()
    return _pid_mod.Controller(
        CompressorSurgePIDParams(**pair.gains, plant=pair.params),
        CompressorSurgePIDState(I_N=0.0, I_x=0.0, timer=0.0, started=False),
        compressor_surge_pid_step,
    )


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

#: Slowest closed-loop decay the factory accepts at the guard points, /s
#: (ours). ``DEFAULT_GAINS`` decay at 0.45 /s or faster (derived,
#: scripts/compressor_surge_numbers.py --section guard).
GUARD_MIN_DECAY = 0.2
#: Smallest margin over the surge line the stress battery may reach (derived:
#: the battery minimum of ``DEFAULT_GAINS``, 0.1133, less 0.015 (ours) and
#: rounded down to 0.001, never below the override line 0.075;
#: scripts/compressor_surge_numbers.py --section guard). Moving the start
#: pressures by 1e-7 to 1e-5 relative leaves that minimum at 0.1133
#: (measured, --section guard). The 0.015 leaves the tuner room to trade
#: margin for tracking. Frozen with the physics: moving it re-opens the
#: tuning.
GUARD_MIN_MARGIN = 0.098
#: Largest valve-command travel, in openings, a battery run may show over its
#: last ``GUARD_TRAVEL_STEPS`` steps, 30 to 40 s after the move (ours: the
#: level --section guard uses to say the valve has stopped ringing). A loop
#: that passes the decay check can still ring once the valve's rate limits
#: act, which the linearisation leaves out: the valve then ratchets open at
#: up to 0.05 and shut at up to 0.01 per step in a limit cycle. There
#: ``DEFAULT_GAINS`` travel under 1e-4 and the anti-surge set (3, 3) up to
#: 2.52 (measured, --section guard).
GUARD_MAX_TRAVEL = 0.01
#: Steps at the end of each battery run over which the travel is summed: the
#: last 10 s.
GUARD_TRAVEL_STEPS = 100
#: (header setpoint in Pa, consumer opening) points where the anti-surge loop
#: is active and the guard linearises the closed loop (ours): the setpoint
#: range at the lowest drawable demand, and two higher demands.
GUARD_POINTS = (
    (20.0e3, 0.35),
    (24.0e3, 0.35),
    (28.0e3, 0.35),
    (24.0e3, 0.50),
    (28.0e3, 0.60),
)
#: The stress battery (ours): (setpoint before, setpoint after, opening
#: before, opening after). Demand drops from 0.95 put the anti-surge loop to
#: work from a shut valve; two setpoint-only moves at low demand; and the
#: corner of the demand clip, 0.20, below the drawable levels on purpose. For
#: ``DEFAULT_GAINS`` the clip corner comes closest to the surge line, the
#: drops next and the setpoint-only moves least (measured, --section guard).
BATTERY = (
    (20.0e3, 20.0e3, 0.95, 0.35),
    (24.0e3, 24.0e3, 0.95, 0.35),
    (28.0e3, 28.0e3, 0.95, 0.35),
    (28.0e3, 20.0e3, 0.95, 0.35),
    (20.0e3, 28.0e3, 0.95, 0.35),
    (28.0e3, 20.0e3, 0.35, 0.35),
    (20.0e3, 28.0e3, 0.35, 0.35),
    (24.0e3, 24.0e3, 0.95, 0.20),
)
#: Schedule clock a battery run starts at (ours), 10 steps before the 60 s
#: boundary where the setpoint steps (block 2) and the demand starts its ramp
#: (block 3).
BATTERY_START_CLOCK = 590
#: Steps before the move.
BATTERY_LEAD = 10
#: Steps per battery run (ours): the lead and 40 s after the move.
BATTERY_STEPS = 410


class UnsafePairGains(ValueError):
    """Pair gains the guard refuses."""


def settled_point(p_ref, u, gains, params: CompressorSurgeParams) -> np.ndarray:
    """The closed loop's steady state at setpoint ``p_ref`` (Pa) and constant
    consumer opening ``u``, in closed form with the env's own functions:
    ``[m_c, dp, N, N_ramp, x, I_N, I_x]``, float64.

    ``dp = p_ref``. Where the anti-surge loop is active the compressor sits on
    the control line, Phi = 2W (1 + surge_line), the speed is the one whose
    characteristic gives ``p_ref`` there, and ``x`` closes the plenum's flow
    balance. Where that ``x`` would be negative the valve is shut and the
    speed is the root, by bisection, of the equilibrium pressure at the
    consumers' flow. The drive setpoint equals the speed, there is no
    deviation, and the integrators are ``pair_init``'s. The pair's commands
    there reproduce the state, so it is a fixed point of the closed loop.
    """
    rho = suction_density(params)
    flow = rho * params.A_c * params.U_r  # m_c per unit Phi at rated speed
    head = rho * params.U_r**2  # dp per unit Psi at rated speed
    phi = 2.0 * params.W * (1.0 + gains["surge_line"])
    N = float(np.sqrt(p_ref / (head * float(characteristic(phi, params, np)))))
    m_c = flow * N * phi
    m_d = float(consumer_flow(p_ref, u, params, np))
    x = (m_c - m_d) / float(recycle_flow(p_ref, 1.0, params, np))
    if x < 0.0:
        x, m_c = 0.0, m_d

        def pressure_gap(n):
            return (
                head * n**2 * float(characteristic(m_d / (flow * n), params, np))
                - p_ref
            )

        # Pressure rises with speed on the right branch: below at the
        # control-line speed, above where the flow reaches the surge line.
        N = brentq(pressure_gap, N, m_d / surge_flow_per_speed(params), xtol=1e-14)
    return np.array([m_c, p_ref, N, N, x, N, x], dtype=float)


def _state_at(z, setpoints, levels, block_clock, params: CompressorSurgeParams):
    """A state at the first five entries of the closed-loop vector ``z``, no
    deviation, with the given schedule and clock."""
    zero = jnp.zeros_like(z[0])
    return CompressorSurgeState(
        time=jnp.asarray(0, jnp.int32),
        m_c=z[0],
        dp=z[1],
        N=z[2],
        N_ramp=z[3],
        x=z[4],
        demand_dev=zero,
        phi_min=z[0] / (suction_density(params) * params.A_c * params.U_r * z[2]),
        setpoint_levels=setpoints,
        demand_levels=levels,
        block_clock=jnp.asarray(block_clock, jnp.int32),
    )


@functools.lru_cache(maxsize=8)
def _guard_functions(params: CompressorSurgeParams):
    """The guard's two jitted computations for ``params`` (closed over, as
    Python constants, the way the env runs), with the gains traced so a new
    gain set does not recompile."""
    quiet = params.replace(demand_sigma=0.0)
    two_w = 2.0 * params.W

    def closed_loop_step(z, p_ref, u, gains):
        state = _state_at(
            z,
            jnp.full(N_SETPOINT_BLOCKS, p_ref),
            jnp.full(N_DEMAND_BLOCKS, u),
            0,
            quiet,
        )
        ps = PairState(I_N=z[5], I_x=z[6], timer=jnp.zeros_like(z[0]))
        u_raw, ps = pair_step(gains, ps, get_obs(state, quiet), quiet, xp=jnp)
        new = compute_next_state(u_raw, state, quiet, jax.random.PRNGKey(0))[0]
        return jnp.stack([new.m_c, new.dp, new.N, new.N_ramp, new.x, ps.I_N, ps.I_x])

    def jacobian(z, p_ref, u, gains):
        return jax.jacfwd(closed_loop_step)(z, p_ref, u, gains)

    def battery_run(z0, setpoints, levels, gains):
        state = _state_at(z0, setpoints, levels, BATTERY_START_CLOCK, quiet)
        ps = pair_init(get_obs(state, quiet), xp=jnp)

        def body(carry, _):
            s, ps = carry
            u_raw, ps = pair_step(gains, ps, get_obs(s, quiet), quiet, xp=jnp)
            s = compute_next_state(u_raw, s, quiet, jax.random.PRNGKey(0))[0]
            x_cmd = commands_from_raw(u_raw, quiet, jnp)[1]
            return (s, ps), (s.phi_min, check_is_terminal(s, quiet)[0], x_cmd)

        _, (phi_min, tripped, x_cmd) = jax.lax.scan(
            body, (state, ps), None, length=BATTERY_STEPS
        )
        margin = phi_min / two_w - 1.0
        travel = jnp.abs(jnp.diff(x_cmd[-(GUARD_TRAVEL_STEPS + 1) :])).sum()
        return margin[BATTERY_LEAD:].min(), tripped.any(), travel

    return (
        jax.jit(jax.vmap(jacobian, in_axes=(0, 0, 0, None))),
        jax.jit(jax.vmap(battery_run, in_axes=(0, 0, 0, None))),
    )


def _traced_gains(gains, dtype):
    return {k: jnp.asarray(float(gains[k]), dtype) for k in DEFAULT_GAINS}


def closed_loop_decay(
    gains, params: CompressorSurgeParams | None = None, points=GUARD_POINTS
) -> np.ndarray:
    """Per guard point, the closed loop's slowest decay rate,
    -ln(max |eig J|) / delta_t in /s: positive decays, negative grows.

    J is the Jacobian of one closed-loop step (``pair_step``, then the env's
    own ``compute_next_state`` with the deviation off) in
    ``(m_c, dp, N, N_ramp, x, I_N, I_x)`` at ``settled_point``, where the
    rate limits, the override and the reset band are inactive. Computed in
    float64, so the verdict does not depend on the caller's precision.
    """
    params = CompressorSurgeParams() if params is None else params
    z = np.stack([settled_point(p, u, gains, params) for p, u in points])
    jacobians = _guard_functions(params)[0]
    with enable_x64():
        J = np.asarray(
            jacobians(
                jnp.asarray(z, jnp.float64),
                jnp.asarray([p for p, _ in points], jnp.float64),
                jnp.asarray([u for _, u in points], jnp.float64),
                _traced_gains(gains, jnp.float64),
            )
        )
    radius = np.abs(np.linalg.eigvals(J)).max(axis=-1)
    return -np.log(radius) / params.delta_t


def battery_report(gains, params: CompressorSurgeParams | None = None):
    """Per ``BATTERY`` scenario: the smallest margin ``phi_min / 2W - 1``
    over the 400 steps after the move, whether the run tripped, and the valve
    command's travel (sum of |step-to-step change|, in openings) over the
    run's last ``GUARD_TRAVEL_STEPS`` steps.

    Each run starts from ``settled_point`` at the scenario's first setpoint
    and opening, with the schedule ``[p0, p0, p1, p1]`` and
    ``[u0, u0, u0, u1, u1, u1]`` at clock ``BATTERY_START_CLOCK``, so after
    ``BATTERY_LEAD`` steps the setpoint steps and the demand ramps together.
    The pair runs on the env's ``compute_next_state`` with the deviation off,
    in float32 as the env ships; a trip does not restart the plant here.
    """
    params = CompressorSurgeParams() if params is None else params
    z0 = np.stack([settled_point(p0, u0, gains, params) for p0, _, u0, _ in BATTERY])
    setpoints = np.array([[p0, p0, p1, p1] for p0, p1, _, _ in BATTERY])
    levels = np.array([[u0] * 3 + [u1] * 3 for _, _, u0, u1 in BATTERY])
    run = _guard_functions(params)[1]
    margins, tripped, travel = run(
        jnp.asarray(z0, jnp.float32),
        jnp.asarray(setpoints, jnp.float32),
        jnp.asarray(levels, jnp.float32),
        _traced_gains(gains, jnp.float32),
    )
    return (
        np.asarray(margins, float),
        np.asarray(tripped, bool),
        np.asarray(travel, float),
    )


@functools.lru_cache(maxsize=256)
def _guard_problems(gain_items, params):
    gains = dict(gain_items)
    decay = closed_loop_decay(gains, params)
    margins, tripped, travel = battery_report(gains, params)
    problems = [
        f"closed loop at {p / KPA:.0f} kPa, opening {u:.2f} decays at "
        f"{d:+.3f} /s, slower than {GUARD_MIN_DECAY} /s"
        for (p, u), d in zip(GUARD_POINTS, decay)
        if not d >= GUARD_MIN_DECAY
    ]
    for (p0, p1, u0, u1), m, t, v in zip(BATTERY, margins, tripped, travel):
        name = f"{p0 / KPA:.0f} to {p1 / KPA:.0f} kPa, opening {u0:.2f} to {u1:.2f}"
        if t:
            problems.append(f"battery {name} trips")
            continue
        if not m >= GUARD_MIN_MARGIN:
            problems.append(
                f"battery {name} comes within {m:.4f} of the surge line, "
                f"under {GUARD_MIN_MARGIN}"
            )
        if not v <= GUARD_MAX_TRAVEL:
            problems.append(
                f"battery {name}: the valve command still travels {v:.3f} in "
                f"the run's last {GUARD_TRAVEL_STEPS} steps, over "
                f"{GUARD_MAX_TRAVEL} (a limit cycle)"
            )
    return tuple(problems)


def check_pair_gains(gains, params: CompressorSurgeParams | None = None) -> None:
    """Raise ``UnsafePairGains`` unless the closed loop decays at
    ``GUARD_MIN_DECAY`` or faster at every guard point, and every battery run
    never trips, keeps at least ``GUARD_MIN_MARGIN`` and ends with its valve
    command travelling at most ``GUARD_MAX_TRAVEL`` over its last
    ``GUARD_TRAVEL_STEPS`` steps.

    Cached on the gain values, and both computations compile once per process
    and params, so the tuner pays the compile once.
    """
    params = CompressorSurgeParams() if params is None else params
    problems = _guard_problems(
        tuple((k, float(gains[k])) for k in DEFAULT_GAINS), params
    )
    if problems:
        raise UnsafePairGains("; ".join(problems))
