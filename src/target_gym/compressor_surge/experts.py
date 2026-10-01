"""Controllers for the compressor: the classic performance and anti-surge
pair, the guard its factory applies, and the reference NMPC.

See ``PHYSICS.md`` in this directory, section 7. Everything here is outside
the version stamp and inside the baseline fingerprint: the env never imports
this module at load time, so editing a controller never moves
``compressor_surge-v2``.

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

The reference controller is a CasADi/IPOPT NMPC whose model is the env's
own discrete step restated in CasADi (``CompressorSurgeMPC``). It reads the
true state, both schedules and the deviation the coming step uses, keeps
the trip variable of every planned step ``MPC_SURGE_MARGIN`` right of the
surge line through a soft constraint priced at a trip's cost, and hands any
step whose solve fails to the pair, whose memory it keeps tracked to the
actions it applies.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import enable_x64
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
    demand_opening,
    get_obs,
    recycle_flow,
    sound_speed,
    suction_density,
    surge_flow_per_speed,
)
from target_gym.experts import pid as _pid_mod
from target_gym.experts.mpc import CasadiMPC

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

    Reads ``target_gym.experts.pid._load_gains()`` at call time, so a tuner
    that patches ``_gains_cache`` is seen.
    """
    stored = _pid_mod._load_gains().get("compressor_surge", {})
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


# ---------------------------------------------------------------------------
# The reference controller: a CasADi/IPOPT NMPC on the env's own step
# ---------------------------------------------------------------------------

#: Horizon in steps of ``delta_t`` (ours), 6 s. A full 20 to 28 kPa move on
#: the 15 % control line needs 14.7 % of rated speed, 4.9 s of drive travel
#: at 3 %/s (derived, scripts/compressor_surge_numbers.py --section steady)
#: plus the 1 s lag, and closing the recycle from its reset opening takes
#: 3.5 s, so 6 s covers the slowest actuator move a block asks for.
MPC_HORIZON = 60
#: The flow-coefficient margin over the surge line that every planned step's
#: ``phi_min`` keeps (TUNED); ``phi_min`` is the env's trip variable, the soft
#: minimum of Phi over the step's substeps. scripts/compressor_surge_numbers.py
#: --mpc-gate sizes the smallest candidate from the margin a seen deviation
#: innovation costs and the model gap, 2 x (1.019 + 0.020) = 2.08 points, so
#: 0.03, and accepts 0.03 and 0.05 on 128 episodes and the stress battery at
#: the same cost, 1.373e5 per episode (measured at the provisional ``c_hold``,
#: 49.6 kW), so the wider one ships.
MPC_SURGE_MARGIN = 0.05
#: Header-pressure error that costs 1 per stage, kPa (ours). It only sets the
#: objective's units, so IPOPT sees numbers of order one.
MPC_ERROR_SCALE_KPA = 1.0
#: Weight on the squared move of each command between stages (ours).
MPC_MOVE_WEIGHT = 1e-3
#: Terminal cost, in multiples of the last stage's tracking term (ours).
MPC_TERMINAL_WEIGHT = 20.0
#: Recycle power above which the objective charges energy, W (ours, frozen):
#: the provisional ``c_hold`` the NMPC was built and accepted with. The NMPC
#: never reads ``params.c_hold``, so scripts/measure_hold.py could set it from
#: the NMPC's own hold, 62.27 kW (measured), without changing the controller
#: it measured.
MPC_POWER_REF = 49.6e3
#: Energy weight per stage (derived, frozen): v2's ``running_weight`` 1 in
#: tracking units of (0.0275 kPa / ``MPC_ERROR_SCALE_KPA``)^2, 0.0275 kPa being
#: the precision floor ``e_floor`` is clamped at. While the clamp binds, the
#: stage cost is v2's per-step cost times (e_floor / error scale)^2, apart
#: from the power reference and the smoothing below. The NMPC never reads
#: ``params.e_floor``.
MPC_ENERGY_WEIGHT = 1.0 * (0.0275 / MPC_ERROR_SCALE_KPA) ** 2
#: Width of the smoothed max(., 0) on the power excess, as a fraction of
#: ``MPC_POWER_REF`` (ours; the form of ``target_gym.experts.mpc``'s smoothed
#: max, (y + sqrt(y^2 + k^2)) / 2).
MPC_POWER_SMOOTHING = 0.05
#: Width of the smoothed substep rate limits, as a fraction of one substep's
#: actuator travel (ours). The env's rate limits make the next state a
#: piecewise smooth function of the command, with a kink wherever the command
#: is a whole number of substeps' travel away. Over the first 12 steps of
#: four episodes (seeds 1000 to 1003), IPOPT's 150-iteration cap stopped 4 of
#: the 48 solves with the exact clip (3 of them an episode's first step) and
#: none with the clip rounded at 0.01 or 0.1. A solve took 10.2 iterations on
#: average at 0.1, against 18.5 at 0.01 and 29.4 with the exact clip. Against
#: the exact clip, the rounding at 0.1 moves the next header pressure by at
#: most 0.555 Pa and a substep's Phi by at most 6.7e-6, over a 41 x 41 grid
#: of commands spanning the reach of both actuators at each of those 48
#: states (measured, scripts/compressor_surge_numbers.py --section mpc-width).
#: test_mpc_predicts_one_step_like_the_plant holds the one-step error under
#: 0.05 ``e_floor``; test_mpc_model_is_the_env_step checks the restatement
#: with the exact clip.
MPC_CLIP_SMOOTHING = 0.1
#: A surge slack above this, in Phi, counts as an active constraint (ours).
#: IPOPT leaves an inactive slack at about -1e-8.
MPC_SLACK_ACTIVE = 1e-6
#: Solve-time budget per step, s (ours): median and 99th percentile, which
#: --mpc-gate measures. The oracle need not run in real time at 0.1 s steps.
MPC_SOLVE_BUDGET_S = (0.15, 1.5)
#: IPOPT options for a warm-started step (ours), added to the suite's
#: iteration and time caps: start from the shifted plan and multipliers
#: without pushing them off their bounds, with the adaptive barrier. Without
#: them IPOPT's bound push moves each surge slack off zero, where the
#: trip-sized penalty (``surge_penalty``, 7.70e6 per unit of slack in Phi,
#: derived) makes the starting point expensive, and the warm start is lost.
#: A cold start (the first step, the first after a trip, and the one after a
#: failed cold solve) goes to a second IPOPT instance with IPOPT's default
#: options. On the first steps of seeds 1000 to 1003, from the cold guess,
#: IPOPT's defaults took 32 to 51 iterations, and these options took 44 and
#: 71 on two of them and hit the 150-iteration cap on the other two
#: (measured, scripts/compressor_surge_numbers.py --section mpc-width).
MPC_IPOPT_WARM_START = {
    "ipopt.warm_start_init_point": "yes",
    "ipopt.warm_start_bound_push": 1e-9,
    "ipopt.warm_start_bound_frac": 1e-9,
    "ipopt.warm_start_slack_bound_push": 1e-9,
    "ipopt.warm_start_slack_bound_frac": 1e-9,
    "ipopt.warm_start_mult_bound_push": 1e-9,
    "ipopt.mu_init": 1e-4,
    "ipopt.mu_strategy": "adaptive",
}
#: Substeps of the restated step: the env's ``"rk4_10"``.
MPC_SUBSTEPS = 10
#: The model's physical states, in the order of ``rhs_function``'s ``x``. The
#: NLP adds ``phi_min`` as a sixth.
MPC_STATES = ("m_c", "dp", "N", "N_ramp", "x")
#: Floor under the check valve's square root in the NLP. The env returns 0
#: where its softplus underflows; a positive floor keeps the derivative finite
#: there and moves the flow by at most k sqrt(1e-300).
_SQRT_FLOOR = 1e-300


def stage_openings(
    demand_levels, block_clock, dev, params: CompressorSurgeParams, n_sub=MPC_SUBSTEPS
) -> np.ndarray:
    """The consumers' opening at the 2 ``n_sub`` + 1 RK4 stage times of the
    step that starts at schedule clock ``block_clock`` with deviation ``dev``.

    The env's RK4 reads the demand at the start, the middle (twice) and the
    end of each substep of h = delta_t / n_sub, so at tau = block_clock
    delta_t + i h / 2 for i = 0 to 2 n_sub. The env's own ``demand_opening``
    on NumPy, clipped here, so the clip never enters the NLP. ``block_clock``
    and ``dev`` may be arrays of equal shape; the stage times are the last
    axis.
    """
    block_clock = np.asarray(block_clock, float)[..., None]
    dev = np.asarray(dev, float)[..., None]
    h = params.delta_t / n_sub
    tau = block_clock * params.delta_t + 0.5 * h * np.arange(2 * n_sub + 1)
    return demand_opening(np.asarray(demand_levels, float), tau, dev, params, xp=np)


def _softplus(y, ca):
    """log(1 + exp(y)), max-shifted so it never overflows."""
    return ca.fmax(y, 0.0) + ca.log1p(ca.exp(-ca.fabs(y)))


def _smooth_clip(z, lo, hi, width, ca):
    """clip(z, lo, hi) for lo < 0 < hi.

    With ``width`` > 0 each edge is rounded by a softplus whose width is
    ``width`` times that edge's limit, so the map is twice differentiable for
    IPOPT. At a distance d inside an edge of width w it moves z by w
    exp(-d / w) or less, and on the edge by w ln 2. ``width`` 0 is the exact
    clip, with ``fmin`` and ``fmax``.
    """
    if width == 0.0:
        return ca.fmin(ca.fmax(z, lo), hi)
    w_hi, w_lo = width * hi, -width * lo
    return (
        z
        - w_hi * _softplus((z - hi) / w_hi, ca)
        + w_lo * _softplus((lo - z) / w_lo, ca)
    )


def _casadi_characteristic(phi, params: CompressorSurgeParams, ca):
    """``env.characteristic`` for CasADi: the cubic, and its tangent line past
    ``phi_join`` through ``if_else``."""
    p = params
    y = phi / p.W - 1.0
    cubic = p.psi_c0 + p.H * (1.0 + 1.5 * y - 0.5 * y**3)
    y_j = p.phi_join / p.W - 1.0
    psi_j = p.psi_c0 + p.H * (1.0 + 1.5 * y_j - 0.5 * y_j**3)
    slope_j = 1.5 * p.H / p.W * (1.0 - y_j**2)
    return ca.if_else(phi > p.phi_join, psi_j + slope_j * (phi - p.phi_join), cubic)


def _casadi_check_valve_sqrt(z, params: CompressorSurgeParams, ca):
    """``env.check_valve_sqrt`` for CasADi: sqrt(s softplus(z / s))."""
    s = params.check_width
    return ca.sqrt(ca.fmax(s * _softplus(z / s, ca), _SQRT_FLOOR))


def _casadi_recycle_power(dp, x, params: CompressorSurgeParams, ca):
    """``env.recycle_power`` for CasADi, W."""
    return (
        params.k_r
        * x
        * _casadi_check_valve_sqrt(dp, params, ca)
        * dp
        / suction_density(params)
    )


def _casadi_step(
    params: CompressorSurgeParams,
    state,
    command,
    opening,
    smooth_width,
    ca,
    n_sub=MPC_SUBSTEPS,
):
    """One env step restated for CasADi: ``(next state, the n_sub substep
    Phis)``, for a command within one step's reach of the actuators.

    ``state`` is (m_c, dp, N, N_ramp, x), ``command`` (N_cmd, x_cmd) in
    physical units and ``opening`` the consumers' opening at the RK4 stage
    times (``stage_openings``). The env's ``compute_next_state`` and
    ``compute_velocity`` run on ``jnp``, which CasADi cannot trace, so this is
    where the dynamics are written a second time, in the env's order: per
    substep, the drive setpoint and the valve move toward their commands,
    then one RK4 step of the duct, plenum and speed equations, then Phi.
    test_mpc_model_is_the_env_step and
    test_mpc_predicts_one_step_like_the_plant hold it to the env.

    The env moves each actuator by at most one substep's travel a per
    substep, which puts it at start + clip(d, -j a, j a) after j substeps,
    with d the command less the start. That closed form is used here. After
    the last substep it is the command itself, since the NLP bounds every
    command to one step's travel, so no clip edge sits where an actuator runs
    at full rate, which is where the NLP's solution often lies. Substeps 1 to
    n_sub - 1 use ``_smooth_clip`` rounded over ``smooth_width`` of one
    substep's travel (0 is exact; ``MPC_CLIP_SMOOTHING`` says why the NLP
    uses 0.1).
    """
    p = params
    m_c, dp, N, N_ramp, x = state
    N_cmd, x_cmd = command
    h = p.delta_t / n_sub
    rho = suction_density(p)
    plenum = sound_speed(p, np) ** 2 / p.V_p

    def velocity(y, u, N_ramp, x):
        m, P, n = y
        U = n * p.U_r
        phi = m / (rho * p.A_c * U)
        dp_c = rho * U**2 * _casadi_characteristic(phi, p, ca)
        outflow = p.k_d * u * _casadi_check_valve_sqrt(
            P - p.dp_out, p, ca
        ) + p.k_r * x * _casadi_check_valve_sqrt(P, p, ca)
        return (
            (p.A_c / p.L_c) * (dp_c - P),
            plenum * (m - outflow),
            (N_ramp - n) / p.tau_N,
        )

    def moved(y, k, c):
        return tuple(a + c * b for a, b in zip(y, k))

    N_start, x_start = N_ramp, x
    d_N, d_x = N_cmd - N_ramp, x_cmd - x
    phis = []
    for j in range(n_sub):
        if j == n_sub - 1:
            N_ramp, x = N_cmd, x_cmd
        else:
            r_N = (j + 1) * p.drive_rate * h
            N_ramp = N_start + _smooth_clip(d_N, -r_N, r_N, smooth_width / (j + 1), ca)
            x = x_start + _smooth_clip(
                d_x,
                -(j + 1) * p.valve_close_rate * h,
                (j + 1) * p.valve_open_rate * h,
                smooth_width / (j + 1),
                ca,
            )
        y = (m_c, dp, N)
        k1 = velocity(y, opening[2 * j], N_ramp, x)
        k2 = velocity(moved(y, k1, 0.5 * h), opening[2 * j + 1], N_ramp, x)
        k3 = velocity(moved(y, k2, 0.5 * h), opening[2 * j + 1], N_ramp, x)
        k4 = velocity(moved(y, k3, h), opening[2 * j + 2], N_ramp, x)
        m_c, dp, N = (
            a + (h / 6.0) * (b1 + 2.0 * b2 + 2.0 * b3 + b4)
            for a, b1, b2, b3, b4 in zip(y, k1, k2, k3, k4)
        )
        phis.append(m_c / (rho * p.A_c * p.U_r * N))
    return (m_c, dp, N, N_ramp, x), phis


def _casadi_soft_min(phis, temperature, ca):
    """The env's trip variable for CasADi: -T logsumexp(-Phi / T) over the
    substep values, shifted by their minimum so no exponent overflows. The
    shift cancels in the value and in the derivatives."""
    v = ca.vertcat(*phis)
    low = ca.mmin(v)
    return low - temperature * ca.log(ca.sum1(ca.exp(-(v - low) / temperature)))


def soft_min(phis, temperature):
    """The env's trip variable over substep values, -T logsumexp(-Phi / T),
    shifted by the minimum, on NumPy. The last axis is the substeps."""
    phis = np.asarray(phis, float)
    low = phis.min(axis=-1)
    return low - temperature * np.log(
        np.exp(-(phis - low[..., None]) / temperature).sum(axis=-1)
    )


class CompressorSurgeMPC(CasadiMPC):
    """CasADi/IPOPT NMPC for the compressor, falling back to the PID pair.

    States : [m_c, dp, N, N_ramp, x, phi_min]
    Inputs : [N_cmd, x_cmd], the speed and recycle commands in physical units
    Stage k: tvps ``opening`` (the consumers' opening at the 21 RK4 stage
             times of the step from state k), ``p_ref`` (the setpoint the env
             scores state k against) and the stage's weights on tracking and
             energy (``preview``)

    The model is the env's discrete step itself, restated in CasADi
    (``_casadi_step``): ten substeps, each moving the drive setpoint and the
    valve toward their commands at their rate limits, then one RK4 step.
    ``phi_min`` is the env's own trip variable, the soft minimum of Phi over
    the step's substeps, carried as a state as the env carries it. Nothing
    is reduced, so the model gap is floating point and the smoothing of the
    substep rate limits. Multiple shooting over ``horizon`` steps.

    Objective: per stage k < horizon, ((p_ref - dp) / error scale)^2 in kPa
    plus ``MPC_ENERGY_WEIGHT`` times the smoothed excess of the recycle power
    over ``MPC_POWER_REF``, relative to it; ``terminal_weight`` times the
    tracking term of the last state; ``move_weight`` on each command's move.
    Surge: every planned step's ``phi_min`` at least 2W (1 + surge_margin),
    soft, with a penalty per unit of slack equal to the trip's cost in the
    objective's units, restart_steps x 2 (dp_error_max / error scale)^2,
    which does not depend on ``e_floor``. Rate reach: each command lies
    within one step's actuator travel of the actuator, which removes
    redundant freedom and costs no authority, since a command beyond it acts
    exactly like one at it.

    do-mpc evaluates constraints on the state a stage starts from, so a
    step's ``phi_min`` is constrained at the next stage. The NLP therefore
    has one stage past the horizon, whose step carries no cost and no
    constraint; its only role is to let the last planned step's ``phi_min``
    be constrained. Stage 0's ``phi_min`` is the measured one, which no
    command changes.

    Like every MPC in the suite it is an oracle ceiling. It reads the true
    state, both schedules with the clock, and the deviation the coming step
    integrates with. Stage k plans with a^k dev, the deviation's conditional
    mean, which is the env's own update with ``demand_sigma`` zeroed
    (``plan_params``), so the first predicted step is exact.

    A step whose solve fails, or reports success with a non-finite action, is
    counted in ``solve_failures`` and ``fallback_steps`` and handed to the
    pair, whose memory is tracked to every action applied (``pair_track``).
    The pair's speed loop then continues the last applied speed. Its valve
    loop does so only near its 15 % control line; at the margins this
    controller plans at, below the pair's 7.5 % override line, the pair's
    first command opens the recycle fully and holds it 2 s. That jump is the
    accepted response to a lost solve next to the surge line. Holding the
    last action, as ``CasadiMPC`` does, can walk the plant into surge during
    a demand drop.

    A solve that an IPOPT cap stopped is applied, as ``CasadiMPC`` applies
    it. The base counts it in ``solve_failures`` and ``solve_capped``; this
    class counts it in ``capped_steps``, or in ``fallback_steps`` when its
    action is non-finite.
    """

    SCALING = {
        "_x": {
            "m_c": 10.0,
            "dp": 2.0e4,
            "N": 1.0,
            "N_ramp": 1.0,
            "x": 0.3,
            "phi_min": 0.5,
        }
    }

    def __init__(
        self,
        env,
        params: CompressorSurgeParams,
        horizon: int = MPC_HORIZON,
        surge_margin: float = MPC_SURGE_MARGIN,
        error_scale_kPa: float = MPC_ERROR_SCALE_KPA,
        move_weight: float = MPC_MOVE_WEIGHT,
        terminal_weight: float = MPC_TERMINAL_WEIGHT,
        fallback: CompressorSurgePair | None = None,
    ):
        # Everything _build_mpc reads, before CasadiMPC.__init__ calls it.
        self.surge_margin = float(surge_margin)
        self.error_scale_kPa = float(error_scale_kPa)
        self.move_weight = float(move_weight)
        self.terminal_weight = float(terminal_weight)
        #: Penalty per unit of surge slack in Phi: a trip's cost in the
        #: objective's units (restart_steps x failure_cost x (e_floor / error
        #: scale)^2, in which e_floor cancels).
        self.surge_penalty = float(
            params.restart_steps
            * 2.0
            * (params.dp_error_max / self.error_scale_kPa) ** 2
        )
        self._pid = (
            CompressorSurgePair(load_gains(), params) if fallback is None else fallback
        )
        self._preview: dict = {}
        self._last_clock: int | None = None
        #: Set by a failed cold solve, so the next step starts cold again
        #: rather than warm from the fallback's restored cold guess.
        self._needs_cold = False
        self._rhs_cache: dict = {}
        #: Steps whose plan used the surge slack (``MPC_SLACK_ACTIVE``).
        self.slack_steps = 0
        #: Steps handed to the pair: the solve failed, or gave a non-finite
        #: action. The base's ``solve_failures`` also counts capped solves,
        #: which are applied.
        self.fallback_steps = 0
        #: Steps whose solve an IPOPT cap stopped and whose iterate was
        #: applied, as ``CasadiMPC`` applies a capped solve.
        self.capped_steps = 0
        #: The last successful plan's first-step margin over the surge line,
        #: phi_min / 2W - 1 with the env's soft minimum; NaN after a fallback.
        self.last_planned_margin = float("nan")
        #: The last successful plan's largest surge slack over its steps, in
        #: Phi; NaN after a fallback.
        self.last_slack = float("nan")
        super().__init__(env, params, horizon=horizon)

    # ------------------------------------------------------------------
    # The NLP
    # ------------------------------------------------------------------

    def _build_mpc(self):
        import casadi
        import do_mpc

        p = self.params
        model = do_mpc.model.Model("discrete")
        state = [model.set_variable("_x", name) for name in MPC_STATES]
        model.set_variable("_x", "phi_min")
        N_cmd = model.set_variable("_u", "N_cmd")
        x_cmd = model.set_variable("_u", "x_cmd")
        opening = model.set_variable("_tvp", "opening", shape=(2 * MPC_SUBSTEPS + 1, 1))
        p_ref = model.set_variable("_tvp", "p_ref")
        w_track = model.set_variable("_tvp", "w_track")
        w_energy = model.set_variable("_tvp", "w_energy")
        nxt, phis = _casadi_step(
            p, state, (N_cmd, x_cmd), opening, MPC_CLIP_SMOOTHING, casadi
        )
        for name, expr in zip(MPC_STATES, nxt):
            model.set_rhs(name, expr)
        model.set_rhs("phi_min", _casadi_soft_min(phis, p.soft_min_temperature, casadi))
        model.setup()
        # ``setup`` swaps the variables set_variable returned for new symbols,
        # so everything below is built from those after it.
        phi_min = model.x["phi_min"]

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=self.horizon + 1,
            t_step=self.mpc_dt,
            n_robust=0,
            store_full_solution=False,
        )

        dp, x, N_ramp = state[1], state[4], state[3]
        tracking = ((p_ref - dp) / KPA / self.error_scale_kPa) ** 2
        excess = (
            _casadi_recycle_power(dp, x, p, casadi) - MPC_POWER_REF
        ) / MPC_POWER_REF
        k = MPC_POWER_SMOOTHING
        energy = MPC_ENERGY_WEIGHT * 0.5 * (excess + casadi.sqrt(excess**2 + k * k))
        # The stage weights put the terminal cost on the last planned state
        # and nothing on the stage past the horizon (``preview``).
        mpc.set_objective(
            lterm=w_track * tracking + w_energy * energy, mterm=casadi.DM(0.0)
        )
        mpc.set_rterm(N_cmd=self.move_weight, x_cmd=self.move_weight)
        mpc.bounds["lower", "_u", "N_cmd"] = p.N_min
        mpc.bounds["upper", "_u", "N_cmd"] = p.N_max
        mpc.bounds["lower", "_u", "x_cmd"] = 0.0
        mpc.bounds["upper", "_u", "x_cmd"] = 1.0

        # One step's actuator travel from the stage's actuator positions.
        mpc.set_nl_cons(
            "reach",
            casadi.vertcat(N_cmd - N_ramp, N_ramp - N_cmd, x_cmd - x, x - x_cmd),
            ub=p.delta_t
            * np.array(
                [p.drive_rate, p.drive_rate, p.valve_open_rate, p.valve_close_rate]
            ),
        )
        mpc.set_nl_cons(
            "surge",
            2.0 * p.W * (1.0 + self.surge_margin) - phi_min,
            ub=0.0,
            soft_constraint=True,
            penalty_term_cons=self.surge_penalty,
        )

        self._preview = self.preview(None)
        tvp_tpl = mpc.get_tvp_template()

        def tvp_fun(_t):
            # Per stage, in the template's order: the openings, the
            # setpoint and the two weights.
            tvp_tpl.master = casadi.DM(
                np.column_stack(
                    [
                        self._preview["opening"],
                        self._preview["p_ref"],
                        self._preview["w_track"],
                        self._preview["w_energy"],
                    ]
                ).ravel()
            )
            return tvp_tpl

        mpc.set_tvp_fun(tvp_fun)
        self._apply_scaling(mpc)
        mpc.set_param(nlpsol_opts={**self._quiet_ipopt(), **MPC_IPOPT_WARM_START})
        mpc.setup()

        # Flat positions of each stage's variables and constraints, for the
        # shifted warm start of the decision vector and the multipliers.
        stages = self.horizon + 1
        opt = mpc.opt_x_num
        self._x_index = np.array([opt.f["_x", j, 0, -1] for j in range(stages + 1)])
        self._u_index = np.array([opt.f["_u", j, 0] for j in range(stages)])
        self._eps_index = np.array([opt.f["_eps", j, 0] for j in range(stages)])
        n_x, n_g = len(self._x_index[0]), int(mpc.nlp_cons.shape[0])
        per_stage = (n_g - n_x) // stages
        assert n_x + stages * per_stage == n_g, (n_x, stages, n_g)
        self._g_index = n_x + np.arange(stages * per_stage).reshape(stages, per_stage)
        self._x_scale = np.array(mpc._x_scaling.cat, float).ravel()
        self._step_fn = self.rhs_function()
        # The same NLP under IPOPT's default barrier, for cold starts.
        self._solvers = {
            "warm": mpc.S,
            "cold": casadi.nlpsol(
                "compressor_cold", "ipopt", mpc.nlp, self._quiet_ipopt()
            ),
        }
        return mpc

    def rhs_function(self, smooth_width: float | None = None):
        """The model's step as a ``casadi.Function``: inputs ``x`` (the five
        states), ``u`` ([N_cmd, x_cmd]) and ``opening`` (21 stage openings);
        outputs ``x_next`` and ``phi`` (the ten substep Phis).

        ``None`` is the shipped clip smoothing, ``MPC_CLIP_SMOOTHING``; 0.0 is
        the exact clip, with which test_mpc_model_is_the_env_step holds the
        restatement to the env.
        """
        import casadi

        width = MPC_CLIP_SMOOTHING if smooth_width is None else float(smooth_width)
        if width not in self._rhs_cache:
            x = casadi.SX.sym("x", len(MPC_STATES))
            u = casadi.SX.sym("u", 2)
            o = casadi.SX.sym("opening", 2 * MPC_SUBSTEPS + 1)
            nxt, phis = _casadi_step(
                self.params,
                [x[i] for i in range(len(MPC_STATES))],
                (u[0], u[1]),
                o,
                width,
                casadi,
            )
            self._rhs_cache[width] = casadi.Function(
                "compressor_step",
                [x, u, o],
                [casadi.vertcat(*nxt), casadi.vertcat(*phis)],
                ["x", "u", "opening"],
                ["x_next", "phi"],
            )
        return self._rhs_cache[width]

    # ------------------------------------------------------------------
    # What the NLP reads
    # ------------------------------------------------------------------

    def preview(self, state) -> dict:
        """The NLP's time-varying parameters at ``state``, over its stages
        k = 0 to ``horizon`` + 1 (the stage past the horizon included):

        - ``p_ref`` (Pa), the level the env scores the state k steps ahead
          against, ``setpoint_levels[min((block_clock + k) //
          setpoint_block_steps, 3)]``;
        - ``dev``, the deviation the step from stage k integrates with,
          a^k demand_dev with a = exp(-demand_theta delta_t);
        - ``opening``, (horizon + 2, 21), that step's consumer opening at its
          RK4 stage times (``stage_openings``);
        - ``w_track`` and ``w_energy``, the stage's weights: 1 and 1 before
          the horizon, ``terminal_weight`` and 0 at it, 0 past it.

        ``None`` gives the same arrays with zeros for the state's values.
        """
        p = self.params
        k = np.arange(self.horizon + 2)
        w_track = np.where(k < self.horizon, 1.0, 0.0)
        w_track[self.horizon] = self.terminal_weight
        w_energy = np.where(k < self.horizon, 1.0, 0.0)
        if state is None:
            zeros = np.zeros(k.size)
            return {
                "p_ref": zeros,
                "dev": zeros,
                "opening": np.zeros((k.size, 2 * MPC_SUBSTEPS + 1)),
                "w_track": w_track,
                "w_energy": w_energy,
            }
        clock = int(state.block_clock) + k
        block = np.minimum(clock // p.setpoint_block_steps, N_SETPOINT_BLOCKS - 1)
        a = np.exp(-p.demand_theta * p.delta_t)
        dev = float(state.demand_dev) * a**k
        return {
            "p_ref": np.asarray(state.setpoint_levels, float)[block],
            "dev": dev,
            "opening": stage_openings(state.demand_levels, clock, dev, p),
            "w_track": w_track,
            "w_energy": w_energy,
        }

    def _update_setpoint(self, state):
        self._preview = self.preview(state)

    def _extract_x0(self, state):
        return np.array(
            [float(getattr(state, name)) for name in MPC_STATES + ("phi_min",)]
        )

    # ------------------------------------------------------------------
    # Warm start
    # ------------------------------------------------------------------

    def _rollout(self, x0, command):
        """The model rolled out from ``x0`` over every stage, with
        ``command(k, state)`` the command of stage ``k``: the states (stages
        + 1, with ``phi_min``) and the commands (stages)."""
        xs, us = [np.asarray(x0, float)], []
        for k in range(self.horizon + 1):
            u = np.asarray(command(k, xs[-1]), float)
            nxt, phis = self._step_fn(xs[-1][:5], u, self._preview["opening"][k])
            phi_min = soft_min(
                np.asarray(phis).ravel(), self.params.soft_min_temperature
            )
            xs.append(np.append(np.asarray(nxt, float).ravel(), phi_min))
            us.append(u)
        return np.array(xs), np.array(us)

    def _pair_command(self, pair, k, s):
        """The command a fresh pair gives at model state ``s`` of stage
        ``k``, clipped to one step's reach of the actuators, where the env
        acts on it exactly as on the unclipped command."""
        p = self.params
        m_d = consumer_flow(s[1], self._preview["opening"][k][0], p, np)
        obs = np.array(
            [
                s[1] / KPA,
                s[0],
                m_d,
                100.0 * s[2],
                100.0 * s[3],
                100.0 * s[4],
                self._preview["p_ref"][k] / KPA,
            ]
        )
        N_cmd, x_cmd = commands_from_raw(pair.step(obs), p)
        dt = p.delta_t
        return (
            np.clip(N_cmd, s[3] - p.drive_rate * dt, s[3] + p.drive_rate * dt),
            np.clip(
                x_cmd, s[4] - p.valve_close_rate * dt, s[4] + p.valve_open_rate * dt
            ),
        )

    def _initialise(self, x0) -> None:
        """Cold start: the model rolled out from ``x0`` with both commands
        holding the actuators where they are, those commands, zero surge
        slacks and no multipliers. Where that rollout takes some planned
        step's ``phi_min`` below the planned margin, the rollout under a
        fresh copy of the fallback pair, with its commands, instead.

        The hold rollout runs into surge when a demand drop lies inside the
        horizon and the valve is shut, as in the stress battery's demand
        drops, where it stays below the surge line for most of the horizon.
        IPOPT started there converges, however many iterations it is given,
        to a plan that surges, and the warm start carries that plan forward
        until the plant surges on it. With the hold rollout as the only cold
        guess, the stress battery of scripts/compressor_surge_numbers.py
        --mpc-gate tripped in 12 of its 16 runs at both candidate margins,
        with the slack active on 688 to 692 steps; from this guess it trips
        in none and never uses the slack (measured, --mpc-gate). From the
        pair's rollout the same NLP finds a plan with no slack, as
        test_mpc_cold_start_avoids_a_surging_guess checks. The acceptance
        run's 128 episodes gave the same figures with either guess (measured,
        --mpc-gate). The guess depends on ``x0`` and the preview alone."""
        import casadi

        m = self._mpc
        hold = x0[[3, 4]]
        xs, us = self._rollout(x0, lambda k, s: hold)
        if not np.all(xs[1:, 5] >= 2.0 * self.params.W * (1.0 + self.surge_margin)):
            pair = CompressorSurgePair(self._pid.gains, self.params)
            xs, us = self._rollout(x0, functools.partial(self._pair_command, pair))
        m.x0 = np.asarray(x0, float)
        m.u0 = hold
        m.set_initial_guess()
        v = np.array(m.opt_x_num.master, float).ravel()
        v[self._x_index] = xs / self._x_scale
        v[self._u_index] = us / np.array(m._u_scaling.cat, float).ravel()
        # do-mpc's set_initial_guess resets the states and inputs only. The
        # surge slacks go back to zero, their value in a newly built NLP, so
        # a cold solve does not depend on the plan the controller held before.
        v[self._eps_index] = 0.0
        m.opt_x_num.master = casadi.DM(v)
        # A trip's restart must not start from the old plan's multipliers.
        m.lam_x_num = casadi.DM.zeros(v.size)
        m.lam_g_num = casadi.DM.zeros(int(m.nlp_cons.shape[0]))
        self._initialized = True

    def _shift_guess(self) -> None:
        """Warm start: the last plan, and its multipliers, moved one stage
        earlier, the last stage repeated."""
        import casadi

        m = self._mpc
        for attr in ("opt_x_num", "lam_x_num", "lam_g_num"):
            held = getattr(m, attr, None)
            if held is None:
                continue
            v = np.array(held.master if attr == "opt_x_num" else held, float).ravel()
            indices = (
                (self._g_index,)
                if attr == "lam_g_num"
                else (self._x_index, self._u_index, self._eps_index)
            )
            for index in indices:
                v[index[:-1]] = v[index[1:]]
            if attr == "opt_x_num":
                m.opt_x_num.master = casadi.DM(v)
            else:
                setattr(m, attr, casadi.DM(v))

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def step(self, obs, state):
        """The raw action for ``state``; ``obs`` is what the fallback pair
        reads.

        A full override of ``CasadiMPC.step``, as the unstable CSTR's is: a
        solve that reports success with a non-finite action counts as a
        failed solve, and a failed solve hands the step to the pair. A trip
        restarts the schedule, and a falling block clock starts the warm
        start afresh from the new state. A cold step whose solve fails leaves
        the next step cold too, so the warm solver never starts from a cold
        guess without multipliers.
        """
        self._update_setpoint(state)
        x0 = self._extract_x0(state)
        clock = int(state.block_clock)
        cold = (
            not self._initialized
            or self._needs_cold
            or (self._last_clock is not None and clock < self._last_clock)
        )
        self._needs_cold = False
        if cold:
            self._initialise(x0)
        else:
            self._shift_guess()
        self._mpc.S = self._solvers["cold" if cold else "warm"]
        self._last_clock = clock
        # do-mpc appends every solve to its history with np.append, which
        # copies the whole history each step. Nothing here reads it, so it
        # is cleared to keep a late step as cheap as an early one.
        self._mpc.data.init_storage()
        guess = self._save_guess()
        failures, capped = self.solve_failures, self.solve_capped
        u = np.array(self._mpc.make_step(x0), dtype=float).flatten()
        finite = bool(np.all(np.isfinite(u)))
        ok = self._record_solve() and finite
        if not finite:
            # Counted once: _record_solve has already counted a solve that
            # reported failure.
            if self.solve_failures == failures:
                self.solve_failures += 1
            self.last_return_status = "non-finite action"
        if ok:
            raw = np.clip(raw_from_commands(u[0], u[1], self.params), -1.0, 1.0)
            self._pid.track(obs, raw)
            self._record_plan(x0, u)
            self.capped_steps += int(self.solve_capped > capped)
        else:
            self.fallback_steps += 1
            raw = self._fallback(guess, u, obs, cold)
        self._last_u = raw
        return np.clip(raw, -1.0, 1.0)

    def _record_plan(self, x0, u) -> None:
        """The plan's first-step margin (the env's soft minimum over the
        substeps the model predicts for the command applied) and its largest
        surge slack over the planned steps. Stage 0's slack, on the measured
        ``phi_min``, is left out: no command changes it."""
        _, phis = self._step_fn(x0[:5], u, self._preview["opening"][0])
        phi_min = soft_min(
            np.asarray(phis, float).ravel(), self.params.soft_min_temperature
        )
        self.last_planned_margin = float(phi_min / (2.0 * self.params.W) - 1.0)
        eps = np.array(self._mpc.opt_x_num.master, float).ravel()[self._eps_index]
        self.last_slack = float(eps[1:].max())
        if self.last_slack > MPC_SLACK_ACTIVE:
            self.slack_steps += 1

    def _fallback(self, guess, u, obs, cold=False):
        """Restore the warm start the failed solve began from and hand this
        step to the pair, stepped on ``obs``.

        The restore is ``CasadiMPC._fallback``'s; the action it would hold is
        not used. do-mpc keeps the failed iterate's first move as the previous
        move, which the next solve's move penalty reads, so it is replaced by
        the commands applied. After a failed ``cold`` step the restored guess
        is the cold rollout with no multipliers, so the next step goes to the
        cold solver again. The base restores a NumPy copy of the decision
        vector, which do-mpc's ``set_initial_guess`` on that cold step cannot
        index into, so it is handed back as a ``casadi.DM``.
        """
        import casadi

        super()._fallback(guess, u)
        self._mpc.opt_x_num.master = casadi.DM(self._mpc.opt_x_num.master)
        self._needs_cold = bool(cold)
        raw = np.asarray(self._pid.step(obs), dtype=float)
        self._mpc.u0 = commands_from_raw(raw, self.params)
        self.last_planned_margin = float("nan")
        self.last_slack = float("nan")
        return raw

    def reset(self):
        """Clear the warm start, do-mpc's stored history and the pair's
        memory."""
        super().reset()
        self._mpc.reset_history()
        self._pid.reset()
        self._last_clock = None
        self._needs_cold = False

    def solver_report(self) -> dict:
        """The base counters, the steps whose plan used the surge slack, the
        steps handed to the pair and the capped solves applied."""
        return {
            **super().solver_report(),
            "surge_slack_steps": self.slack_steps,
            "fallback_steps": self.fallback_steps,
            "capped_applied_steps": self.capped_steps,
        }


def make_compressor_surge_mpc(env, params, **kwargs) -> CompressorSurgeMPC:
    """Registry factory: the CasADi/IPOPT NMPC on the env's own step, falling
    back to the PID pair.

    The fallback is built from the stored gains and held to the PID factory's
    guard, so gains the guard refuses stop this factory too. ``kwargs`` go to
    ``CompressorSurgeMPC`` (``horizon``, ``surge_margin``,
    ``error_scale_kPa``, ``move_weight``, ``terminal_weight``); any other name
    raises ``TypeError``, which ``test_mpc_baselines._cheap_mpc`` relies on.
    """
    gains = load_gains()
    check_pair_gains(gains)
    return CompressorSurgeMPC(
        env, params, fallback=CompressorSurgePair(gains, params), **kwargs
    )
