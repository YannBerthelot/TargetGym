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
# The expert: the ladder's reference
# ---------------------------------------------------------------------------


@struct.dataclass
class ReferenceParams:
    """The reference of ``scripts/reactor_ladder.py``: the Westinghouse procedure
    (Krsko, NENE 2022, section 3; KNS 18A-534), which the ladder kept over EDF's Mode
    A (PHYSICS.md, D8). Bank D holds the axial offset, never deeper than its insertion
    limit; boron trims Tavg past the rods' 1.6 C dead band (IAEA NP-T-3.23, annex
    II-5.1), fed forward from the day's plan, led by the boron line's expected delay,
    and from xenon's balance (:func:`xenon_rate`, with :func:`xenon_play`). Both
    loops are SIMC PIs on bump tests at full power and at the half-power hold, the
    design asking the least gain kept. Nothing here was set from a cost (docs/proposals/reactor-axial.md).
    """

    dt: float = 30.0
    dead_band: float = 1.6
    # The recipe's bump tests on reactor-v8 (scripts/reactor_ladder.py).
    wd_Kc: float = 0.12757098960052432  # rod action per unit of offset
    wd_Ti: float = 30.0
    t_Kc: float = -0.02094281018869827  # boron per K of Tavg past the dead band
    t_Ti: float = 4920.0
    ff: float = -97.2559113703026
    ao_ref: float = 0.0
    # The rods' action band: no error while the offset is within 0.025 of its
    # reference, half the 0.05 target band (owner, 2026-11-08; PHYSICS.md section 6).
    action_band: float = 0.025
    lead: float = 20.0 * 30.0  # s, the middle of the line's 10 to 30 steps
    park: float = 1.0 - 2.0 * 3.0 / 228.0
    reach: float = 30.0 * 72.0 / 60.0 / 228.0 * 2.0


@struct.dataclass
class ReferenceState:
    """The two loops' outputs and last errors, the plan, the clock and the xenon
    tool's iodine and xenon. The rods' loop starts from where the rods are read on
    the first step, so every field starts at 0."""

    rods: float = 0.0
    boron: float = 0.0
    last_ao: float = 0.0
    last_e: float = 0.0
    clock: float = 0.0
    down: float = 0.0
    up: float = 0.0
    I_hat: float = 0.0
    Xe_hat: float = 0.0
    debt: float = 0.0  # ppm the xenon tool has not moved (its play)
    # The rod program's state, used through a ramp boron can't follow.
    rc_lag: float = 0.0
    rc_moving: float = 0.0
    rc_started: float = 0.0


def xenon_feedforward(I_hat, Xe_hat, n, boron, first, plant, dt, xp=jnp, history=None):
    """The boron flow that cancels xenon's reactivity rate in full, and the next
    ``I_hat, Xe_hat``: :func:`xenon_rate` with no play."""
    rate, I_hat, Xe_hat = xenon_rate(I_hat, Xe_hat, n, first, plant, dt, xp, history)
    return ppm_flow(rate, boron, plant, xp), I_hat, Xe_hat


def xenon_rate(I_hat, Xe_hat, n, first, plant, dt, xp=jnp, history=None):
    """The boron rate, ppm/s, that cancels xenon's reactivity rate, and the next
    ``I_hat, Xe_hat``.

    The operator's xenon tool (OAPS, section 1): the textbook iodine and xenon
    equations (the same point form as the plant's means, PHYSICS.md) run on the
    observed power, started in balance with the recent power the observation
    reports (``history``; the first reading when it is not given). Xenon's worth
    ``rho_Xe_full (Xe_hat - 1)`` is met by boron at its worth. No gain is tuned."""
    from target_gym.reactor.env import LAMBDA_IODINE, LAMBDA_XENON

    p = plant
    s, r = p.sigma_phi0, p.gamma_ratio
    a = (LAMBDA_XENON + s) / (1.0 + r)
    h = n if history is None else history
    balance = (LAMBDA_XENON + s) * h / (LAMBDA_XENON + s * h)
    I_hat = xp.where(first, h, I_hat)
    Xe_hat = xp.where(first, balance, Xe_hat)
    dI = LAMBDA_IODINE * (n - I_hat)
    dXe = a * (r * n + I_hat) - (LAMBDA_XENON + s * n) * Xe_hat
    ppm_rate = p.rho_Xe_full * dXe / p.boron_worth  # ppm/s, negative: dilute
    return ppm_rate, I_hat + dt * dI, Xe_hat + dt * dXe


def boron_can_follow(load_rate, boron, plant, xp=jnp):
    """Whether the boron line can carry a planned ramp of ``load_rate`` per second
    (PHYSICS.md section 6): the ppm a second the ramp needs, the power defect per
    unit of load on the program times the rate, against full dilution or full
    boration at ``boron``."""
    from target_gym.reactor.env import rod_balance, steady_state_temperatures

    p = plant

    def needed(load):
        T_fuel, T_hot, T_cold, _ = steady_state_temperatures(load, p)
        return rod_balance(T_fuel, 0.5 * (T_hot + T_cold), 1.0, p)

    ppm_per_load = (needed(1.0) - needed(0.5)) / 0.5 / p.boron_worth
    need = ppm_per_load * load_rate  # ppm/s, negative: dilute
    dilute = p.dilution_flow / p.primary_mass * xp.maximum(boron, 1.0)
    borate = p.boration_flow / p.primary_mass * (p.boric_acid_ppm - boron)
    return xp.where(need < 0.0, -need <= dilute, need <= borate)


def ppm_flow(ppm_rate, boron, plant, xp=jnp):
    """The boron flow that changes the core's boron by ``ppm_rate`` ppm/s: dilution
    removes ``dilution_flow / primary_mass * boron`` ppm/s at full flow, boration
    adds ``boration_flow / primary_mass * (acid - boron)``."""
    p = plant
    dilute = p.dilution_flow / p.primary_mass * xp.maximum(boron, 1.0)
    borate = p.boration_flow / p.primary_mass * (p.boric_acid_ppm - boron)
    return xp.where(ppm_rate < 0.0, ppm_rate / dilute, ppm_rate / borate)


def xenon_play(debt, ppm_rate, plant, dt, lead, dead_band, xp=jnp):
    """The xenon tool with play (owner, 2026-11-10; PHYSICS.md section 6): the ppm
    to move this step and the debt left.

    ``debt`` is the boron the tool did not move, in ppm; at steady power it leaves
    Tavg shifted by ``debt * boron_worth / (alpha_coolant + alpha_fuel)``. Tavg's
    free zone is its dead band on either side of the program, each edge pulled in
    by the shift xenon makes over ``lead``. Inside it nothing moves; past it, only
    the excess."""
    p = plant
    # K of Tavg per ppm of debt: at a fixed power the fuel cools with the coolant.
    k = p.boron_worth / (p.alpha_coolant + p.alpha_fuel)
    debt = debt + ppm_rate * dt
    pull = xp.abs(ppm_rate) * lead
    edge = xp.maximum(dead_band / k - pull, 0.0)
    left = xp.clip(debt, -edge, edge)
    return debt - left, left


def reference_step(params, state, obs, xp=jnp):
    """One 30 s step: ``obs = [n, T_hot, rod, load, T_avg, T_ref, delta_I, boron
    (20 min late), to_down, to_up, xenon's power history]``, the action ``[rod
    demand, boron flow]``."""
    from target_gym.reactor.env import (
        ReactorParams,
        rod_insertion_limit,
        scheduled_load,
    )

    p, plant = params, ReactorParams()
    n, rod, T_avg, T_ref = obs[..., 0], obs[..., 2], obs[..., 4], obs[..., 5]
    di, to_down, to_up = obs[..., 6], obs[..., 8], obs[..., 9]
    first = state.clock == 0.0
    down = xp.where(first, to_down * 3600.0, state.down)
    up = xp.where(first, to_up * 3600.0, state.up)

    off = p.ao_ref - di / xp.maximum(n, 0.05)
    ao = xp.sign(off) * xp.maximum(xp.abs(off) - p.action_band, 0.0)
    held = xp.where(first, rod, state.rods)
    rods = held + p.wd_Kc * ((ao - state.last_ao) + p.dt / p.wd_Ti * ao)
    # Through a ramp boron can't follow, the rod program takes Tavg; the offset
    # loop picks up from where it leaves the rods (PHYSICS.md section 6).
    now = scheduled_load(state.clock + p.dt, down, up, plant, xp=xp)
    now = (now - scheduled_load(state.clock, down, up, plant, xp=xp)) / p.dt
    fast = ~boron_can_follow(now, obs[..., 7], plant, xp)
    rc_state = RodControlState(state.rc_lag, state.rc_moving, state.rc_started)
    rc_action, rc_state = rod_control_step(
        RodControlParams(), rc_state, obs[..., :6], xp
    )
    rods = xp.where(fast, rc_action[..., 0], rods)
    rods = xp.clip(rods, rod - p.reach, rod + p.reach)
    # Above the insertion limit where the plan puts the power by the step's end:
    # on a ramp up the limit tightens as the power rises.
    after = xp.maximum(
        n, now * p.dt + scheduled_load(state.clock, down, up, plant, xp=xp)
    )
    rods = xp.clip(rods, 1.0 - 2.0 * rod_insertion_limit(after, plant), p.park)

    error = T_ref - T_avg
    e = xp.sign(error) * xp.maximum(xp.abs(error) - p.dead_band, 0.0)
    boron = state.boron + p.t_Kc * ((e - state.last_e) + p.dt / p.t_Ti * e)
    boron = xp.clip(boron, -1.0, 1.0)

    t = state.clock + p.lead
    planned = scheduled_load(t + p.dt, down, up, plant, xp=xp)
    planned = planned - scheduled_load(t, down, up, plant, xp=xp)
    rate, I_hat, Xe_hat = xenon_rate(
        state.I_hat, state.Xe_hat, n, first, plant, p.dt, xp, obs[..., 10]
    )
    moved, debt = xenon_play(state.debt, rate, plant, p.dt, p.lead, p.dead_band, xp)
    xenon = ppm_flow(moved / p.dt, obs[..., 7], plant, xp)
    out = xp.clip(boron + p.ff * planned + xenon, -1.0, 1.0)
    new = ReferenceState(
        rods=rods, boron=boron, last_ao=ao, last_e=e, clock=state.clock + p.dt,
        down=down, up=up, I_hat=I_hat, Xe_hat=Xe_hat, debt=debt,
        rc_lag=rc_state.mismatch_lag, rc_moving=rc_state.moving,
        rc_started=rc_state.started,
    )  # fmt: skip
    return xp.stack([rods, out], axis=-1), new


def make_reactor_pid_controller() -> Controller:
    """The expert: the ladder's reference, the Westinghouse procedure, as a
    :class:`~target_gym.experts.pid.Controller`. :func:`rod_control_step`, the
    published rod program, is a row in the ladder's table beside it."""
    return Controller(ReferenceParams(), ReferenceState(), reference_step)
