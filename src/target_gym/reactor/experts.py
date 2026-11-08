"""The expert (the PID slot) for the reactor task.

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).
"""

import jax.numpy as jnp
import numpy as np
from flax import struct

from target_gym.experts.pid import Controller

# ---------------------------------------------------------------------------
# The published program: Westinghouse rod control on average temperature
# ---------------------------------------------------------------------------


@struct.dataclass
class RodControlParams:
    """The Westinghouse rod control program.

    The program is the NRC's (Westinghouse Technology training manual,
    HRTD 8.1 and 5.0): a +-1.5 F deadband with a 1.0 F lockup, 8 steps/min to
    3 F, 32 steps/min per F to 72 steps/min at 5 F; the power mismatch through
    a 40 s rate/lag, a nonlinear gain (0.3 F per % to 1 %, 1.5 F per %
    beyond) and a variable gain (2 to 50 % load, 1 / load above). A speed in
    steps/min moves the rods over the plant's 228-step stroke.
    """

    dt: float = 30.0
    deadband_F: float = 1.5
    lockup_F: float = 1.0
    tau_rate: float = 40.0
    stroke_steps: float = 228.0


@struct.dataclass
class RodControlState:
    """The rate/lag's lagged mismatch (% power), whether the rods are moving,
    and whether the lag has seen a first observation."""

    mismatch_lag: float = 0.0
    moving: float = 0.0
    started: float = 0.0


def _mismatch_gain(m, xp):
    """The nonlinear gain unit: deg F for a mismatch rate of ``m`` % power."""
    a = xp.abs(m)
    return xp.sign(m) * xp.where(a <= 1.0, 0.3 * a, 0.3 + 1.5 * (a - 1.0))


def rod_control_step(params, state, obs, xp=jnp):
    """One 30 s step of Westinghouse rod control.

    obs: ``[n, T_hot, rho_ext_norm, load, T_avg, T_ref]``; the action is the
    desired rod position on the same [-1, 1] scale.
    """
    p = params
    n, rod, load = obs[..., 0], obs[..., 2], obs[..., 3]

    # Temperature channel: Tref (programmed on load) minus Tavg, in deg F.
    error = 1.8 * (obs[..., 5] - obs[..., 4])

    # Power-mismatch channel: tau s / (1 + tau s) on (turbine - nuclear), the
    # lag sampled exactly at the control step and started on the first reading.
    x = 100.0 * (load - n)
    alpha = np.exp(-p.dt / p.tau_rate)
    lagged = xp.where(
        state.started > 0.5, alpha * state.mismatch_lag + (1 - alpha) * x, x
    )
    variable = xp.where(load <= 0.5, 2.0, 1.0 / xp.maximum(load, 1e-6))
    error = error + variable * _mismatch_gain(x - lagged, xp)

    # Rod speed program, with the lockup holding moving rods to 1.0 F.
    size = xp.abs(error)
    moving = xp.where(state.moving > 0.5, size > p.lockup_F, size >= p.deadband_F)
    spm = xp.where(moving, xp.clip(8.0 + 32.0 * (size - 3.0), 8.0, 72.0), 0.0)
    move = xp.sign(error) * spm / 60.0 * p.dt / p.stroke_steps * 2.0
    action = xp.clip(rod + move, -1.0, 1.0)
    new_state = RodControlState(
        mismatch_lag=lagged, moving=xp.where(moving, 1.0, 0.0), started=1.0 + 0.0 * x
    )
    return xp.stack([action], axis=-1), new_state


# ---------------------------------------------------------------------------
# The expert: the ladder's best rung
# ---------------------------------------------------------------------------


@struct.dataclass
class ReferenceParams:
    """The reference of ``scripts/reactor_ladder.py``, EDF's Mode A: the published
    rod program on Tavg (:func:`rod_control_step`), and the operator's boron from the
    ladder's rung 5. Boron steers the axial offset through the rods, as a cascade
    (the offset sets the rods' target, boron carries them there). It is fed forward
    from the day's plan for the power defect, led by the boron line's expected delay,
    and takes Tavg while the rods sit at their top stop with Tavg still low past
    1.6 C (IAEA NP-T-3.23, annex II-5.1). Every boron gain is SIMC on a bump test at
    full power and at the half-power hold, the design asking the least gain kept.
    Nothing here was set from a cost (docs/proposals/reactor-axial.md)."""

    dt: float = 30.0
    dead_band: float = 1.6
    rods: RodControlParams = struct.field(default_factory=RodControlParams)
    ff: float = -113.648191
    ao_Kc: float = 0.1942105
    ao_Ti: float = 765.0
    inner_Kc: float = 1.2258107
    inner_Ti: float = 8760.0
    t_Kc: float = -0.0480110
    t_Ti: float = 4680.0
    ao_ref: float = 0.0
    lead: float = 20.0 * 30.0  # s, the middle of the line's 10 to 30 steps
    park: float = 1.0 - 2.0 * 3.0 / 228.0
    reach: float = 30.0 * 72.0 / 60.0 / 228.0 * 2.0


@struct.dataclass
class ReferenceState:
    """The rod program's state, the boron loops' outputs and last errors, the plan
    and the clock."""

    rods: RodControlState = struct.field(default_factory=RodControlState)
    boron: float = 0.0
    target: float = 1.0 - 2.0 * 3.0 / 228.0
    last_e: float = 0.0
    last_ao: float = 0.0
    last_r: float = 0.0
    clock: float = 0.0
    down: float = 0.0
    up: float = 0.0
    override: float = 0.0


def reference_step(params, state, obs, xp=jnp):
    """One 30 s step: ``obs = [n, T_hot, rod, load, T_avg, T_ref, delta_I, boron,
    to_down, to_up]``, the action ``[rod demand, boron flow]``."""
    from target_gym.reactor.env import ReactorParams, scheduled_load

    p, plant = params, ReactorParams()
    n, rod, T_avg, T_ref = obs[..., 0], obs[..., 2], obs[..., 4], obs[..., 5]
    di, to_down, to_up = obs[..., 6], obs[..., 8], obs[..., 9]
    first = state.clock == 0.0
    down = xp.where(first, to_down * 3600.0, state.down)
    up = xp.where(first, to_up * 3600.0, state.up)

    error = T_ref - T_avg
    e = xp.sign(error) * xp.maximum(xp.abs(error) - p.dead_band, 0.0)
    demand, rod_state = rod_control_step(p.rods, state.rods, obs[..., :6], xp)
    rods = demand[..., 0]

    ao = p.ao_ref - di / xp.maximum(n, 0.05)
    target = state.target + p.ao_Kc * ((ao - state.last_ao) + p.dt / p.ao_Ti * ao)
    target = xp.clip(target, -1.0, p.park)
    r = target - rod
    boron = state.boron + p.inner_Kc * ((r - state.last_r) + p.dt / p.inner_Ti * r)
    boron = xp.clip(boron, -1.0, 1.0)

    t = state.clock + p.lead
    planned = scheduled_load(t + p.dt, down, up, plant, xp=xp)
    planned = planned - scheduled_load(t, down, up, plant, xp=xp)
    stuck = (rod >= p.park - 1e-3) & (e > 0.0)
    step = p.t_Kc * ((e - state.last_e) + p.dt / p.t_Ti * e)
    override = xp.where(stuck, state.override + step, 0.0)
    out = xp.clip(boron + p.ff * planned + override, -1.0, 1.0)
    new = ReferenceState(
        rods=rod_state, boron=boron, target=target, last_e=e, last_ao=ao, last_r=r,
        clock=state.clock + p.dt, down=down, up=up, override=override,
    )  # fmt: skip
    return xp.stack([rods, out], axis=-1), new


def make_reactor_pid_controller() -> Controller:
    """The expert: the ladder's best rung as a
    :class:`~target_gym.experts.pid.Controller`. :func:`rod_control_step`, the
    published rod program, is a row in the ladder's table beside it."""
    return Controller(ReferenceParams(), ReferenceState(), reference_step)
