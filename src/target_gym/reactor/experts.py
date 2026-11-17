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
# The expert: the Westinghouse procedure, from the operator's curve book
# ---------------------------------------------------------------------------


def curve_book(cycle, plant, xp=jnp):
    """The operator's curve book: the plant's parameters with the core's own at
    ``cycle``, the fraction of its cycle burnt (observation 13). The same straight
    lines as ``env.core_at``, written against ``xp`` so the host controller stays
    in NumPy."""
    c = xp.clip(cycle, 0.0, 1.0)
    return plant.replace(
        boron_ref=plant.boron_ref * (1.0 - c),
        alpha_coolant=plant.alpha_coolant
        + (plant.alpha_coolant_end - plant.alpha_coolant) * c,
        boron_worth=plant.boron_worth + (plant.boron_worth_end - plant.boron_worth) * c,
    )


def ppm_per_load(plant):
    """The ppm of boron the program asks per unit of load: the power defect on the
    program, from ``rod_balance`` at full and half load, over the boron's worth.
    Negative: more load, less boron."""
    from target_gym.reactor.env import rod_balance, steady_state_temperatures

    def needed(load):
        T_fuel, T_hot, T_cold, _ = steady_state_temperatures(load, plant)
        return rod_balance(T_fuel, 0.5 * (T_hot + T_cold), 1.0, plant)

    return (needed(1.0) - needed(0.5)) / 0.5 / plant.boron_worth


def tavg_per_ppm(plant):
    """K of Tavg per ppm of boron at a fixed load: the fuel sits a fixed step above
    the coolant there, so both feedbacks take up the boron's worth. Negative."""
    return -plant.boron_worth / (plant.alpha_coolant + plant.alpha_fuel)


def tavg_spread(plant, xp=jnp):
    """The standard deviation of Tavg's error under the load's frequency-control
    wobble, in K, with the rods and boron still. A unit of load runs the fuel
    hotter by ``P (0.5 / m_dot_cp + 1 / UA)`` at a fixed Tavg, which Tavg gives
    back through both feedbacks, and moves the program by its span."""
    p = plant
    wobble = p.demand_sigma / xp.sqrt(2.0 * p.demand_theta)
    swing = p.P_thermal_ref * (0.5 / p.m_dot_cp + 1.0 / p.UA)
    span = p.T_avg_full_load - p.T_avg_no_load
    return wobble * (p.alpha_fuel * swing / (p.alpha_coolant + p.alpha_fuel) + span)


@struct.dataclass
class ReferenceParams:
    """The reference: the Westinghouse procedure (Krsko, NENE 2022, section 3; KNS
    18A-534), kept over EDF's Mode A (PHYSICS.md, D8), run from the operator's
    curve book at the day's cycle point (PHYSICS.md section 6; owner, 2026-11-15).

    Bank D holds the axial offset, never deeper than its insertion limit. Boron
    trims Tavg past its free band, fed forward from the day's plan, led by the
    boron line's expected delay, and from the operator's tool for xenon's balance
    and the fuel's burnup (:func:`xenon_rate`, with :func:`xenon_play`). The
    turbine loads at the rate the boron line carries.

    Two numbers come from the recipe's bump tests (scripts/reactor_ladder.py): the
    rods' loop on the offset, and the boron line's dead time to Tavg. The rest is
    the curve book's and the plant record's. Nothing here was set from a cost.
    """

    dt: float = 30.0
    # Bump tests: bank D on the offset, a SIMC PI keeping the least gain of full
    # power and the half-power hold.
    wd_Kc: float = 0.12757098960052432  # rod action per unit of offset
    wd_Ti: float = 30.0
    # Bump tests: boron to Tavg is an integrator with this dead time, the line's
    # 5 to 15 min and the plant's lag. SIMC with tc = theta gives the boron loop
    # ``Kc = 1 / (k 2 theta)`` and ``Ti = 8 theta``, ``k`` the curve book's K of
    # Tavg per ppm.
    theta: float = 615.0  # s
    ao_ref: float = 0.0
    # The rods' action band: no error while the flux difference is within half its
    # 5 % target band (owner, 2026-11-08; PHYSICS.md section 6).
    action_band: float = 0.025  # of rated delta I
    # The zone's back-off (charter J1): the operator's tool and the turbine act
    # this many standard deviations of Tavg, under the load's wobble, inside the
    # edge of its free band, since a Tavg that close predicts a charge. The Tavg
    # loop itself acts only past the edge, on what is charged (owner, 2026-11-10).
    backoff: float = 2.0
    lead: float = 20.0 * 30.0  # s, the middle of the line's 10 to 30 steps
    # The tool's play pulls each edge in by what xenon moves Tavg before boron sent
    # now reaches the core: the line's longest delay, since an edge is a limit.
    longest: float = 30.0 * 30.0  # s
    # The turbine reloads after a trip's restart when its load reads this far under
    # the plan (owner, 2026-11-12): ours, 10 times the frequency wobble's 0.5 % sd.
    reload_band: float = 0.05
    # A restart shows as the load falling faster than any plan or turbine moves it,
    # 2.5 % a step: ours, between that and the 35 % a restart from the day's lowest
    # plan drops.
    restart_drop: float = 0.04
    # The turbine counts as back on its plan within the wobble's two sd: ours.
    on_plan: float = 0.01
    park: float = 1.0 - 2.0 * 3.0 / 228.0
    reach: float = 30.0 * 72.0 / 60.0 / 228.0 * 2.0


@struct.dataclass
class ReferenceState:
    """The two loops' outputs and last errors, the plan, the clock, the xenon
    tool's iodine and xenon, and what the turbine was doing. The rods' loop starts
    from where the rods are read on the first step, so every field starts at 0."""

    rods: float = 0.0
    trim: float = 0.0  # ppm/s, the Tavg loop's output
    last_ao: float = 0.0
    last_e: float = 0.0
    clock: float = 0.0
    down: float = 0.0
    up: float = 0.0
    rate: float = 0.0  # the day held: its ramp rate (1/s) and depth, as dispatched
    depth: float = 0.0
    I_hat: float = 0.0
    Xe_hat: float = 0.0
    debt: float = 0.0  # ppm the xenon tool has not moved (its play)
    # The rod program's state, used through a ramp boron can't follow.
    rc_lag: float = 0.0
    rc_moving: float = 0.0
    rc_started: float = 0.0
    load: float = 0.0  # the turbine's load as last read
    paced: float = 0.0  # 1 while the reference holds the turbine under its plan
    cycle: float = 0.0  # the core's cycle point as last read
    owed: float = 0.0  # ppm of the plan's boron still to send
    on_tavg: float = 0.0  # s the rods stay on Tavg after the last boration owed


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
    p = plant
    need = ppm_per_load(p) * load_rate  # ppm/s, negative: dilute
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


def reload_from(load, seconds, plant, xp=jnp):
    """The reloading turbine's load ``seconds`` after it read ``load``: the
    plant's conditioning ramps, ``reload_fast_rate`` up to ``reload_knee`` and
    ``reload_rate`` above (``ReactorParams.restart_load``), which the operators run
    and so know."""
    p = plant
    to_knee = xp.maximum(p.reload_knee - load, 0.0) / p.reload_fast_rate
    below = load + p.reload_fast_rate * seconds
    above = xp.maximum(load, p.reload_knee) + p.reload_rate * xp.maximum(
        seconds - to_knee, 0.0
    )
    return xp.where(seconds <= to_knee, below, above)


def tavg_dead_band(params, plant, xp=jnp):
    """Where the operator's tool and the turbine start acting on Tavg, in K off its
    program: the free band's edge (``e_tol``) less the back-off, ``backoff``
    standard deviations of Tavg under the load's wobble (:func:`tavg_spread`), and
    never under 0."""
    return xp.maximum(plant.e_tol - params.backoff * tavg_spread(plant, xp), 0.0)


def carried_rate(boron, xenon_ppm_rate, plant, xp=jnp):
    """The loading rate the boron line carries, of rated load a second: full
    dilution at ``boron``, less what xenon's rate takes of it (or plus what its
    burnout gives), over the ppm the program asks per unit of load. Never under 0.
    """
    p = plant
    dilute = p.dilution_flow / p.primary_mass * xp.maximum(boron, 1.0)
    return xp.maximum(dilute + xenon_ppm_rate, 0.0) / -ppm_per_load(p)


def turbine_demand(error, carried, dead, lead, plant, xp=jnp):
    """The turbine's loading rate as an action in [0, 1], for a Tavg ``error``
    (Tref - Tavg, K) and the rate boron ``carried``.

    Tavg's free band is reactivity too: every kelvin Tavg sits warmer than the cold
    side of the backed-off band (``dead``) is load the core can pick up, a kelvin
    over ``tavg_per_ppm x ppm_per_load`` of rated load, and the turbine takes it
    over the boron line's ``lead``. So the rate is the carried one with Tavg at
    that cold side and faster the warmer Tavg is; it tapers to a stop as Tavg
    cools from there to the free band's cold edge, and to the turbine's own rate
    as it warms past the hot side of the backed-off band to the hot edge."""
    p = plant
    width = xp.maximum(p.e_tol - dead, 1e-6)
    cold = xp.clip((p.e_tol - error) / width, 0.0, 1.0)  # 1 inside, 0 at the edge
    hot = xp.clip((-error - dead) / width, 0.0, 1.0)  # 0 inside, 1 at the edge
    spare = xp.maximum(dead - error, 0.0) / (tavg_per_ppm(p) * ppm_per_load(p) * lead)
    base = xp.clip((carried + spare) / p.turbine_rate, 0.0, 1.0) * cold
    return base + (1.0 - base) * hot


def reference_step(params, state, obs, xp=jnp):
    """One 30 s step: ``obs = [n, T_hot, rod, load, T_avg, T_ref, delta_I, boron
    (20 min late), to_down, to_up, xenon's power history, the day's ramp rate
    (%/min), its minimum load, the cycle point, the plan's load, the offset's
    penalty time]``, the action ``[rod demand, boron flow, turbine loading rate]``.
    """
    from target_gym.reactor.env import ReactorParams, rod_insertion_limit

    p, book = params, ReactorParams()
    cycle = obs[..., 13]
    plant = curve_book(cycle, book, xp)
    n, rod, load, T_avg, T_ref = (obs[..., k] for k in (0, 2, 3, 4, 5))
    di, boron, to_down, to_up = (obs[..., k] for k in (6, 7, 8, 9))
    planned_now = obs[..., 14]
    first = state.clock == 0.0
    # Each day's dispatch is read as it comes (re-record pass, step 2): on the first
    # step, and when the hours ahead show a new day once the day held has ended,
    # with the day's own ramp rate and minimum (owner, 2026-11-12). A ramp the
    # dispatch reads 0 for has already run: it is placed a ramp before now.
    held_span = state.depth / xp.maximum(state.rate, 1e-12)
    ended = state.clock >= state.up + held_span
    read = first | ((to_down > 0.0) & ended)
    day_rate = xp.where(read, obs[..., 11] / 6000.0, state.rate)
    day_depth = xp.where(read, 1.0 - obs[..., 12], state.depth)
    span = day_depth / xp.maximum(day_rate, 1e-12)

    def start(hours):
        return xp.where(hours > 0.0, state.clock + hours * 3600.0, state.clock - span)

    down = xp.where(read, start(to_down), state.down)
    up = xp.where(read, start(to_up), state.up)

    def plan(t):
        fall = xp.minimum(day_rate * xp.maximum(t - down, 0.0), day_depth)
        rise = xp.minimum(day_rate * xp.maximum(t - up, 0.0), day_depth)
        return 1.0 - fall + rise

    # The turbine is on its plan, held under it by this reference, or reloading
    # after a trip's restart. A restart shows as the load falling faster than any
    # plan moves it; the turbine then reads well under the plan without having been
    # held, and climbs the plant's conditioning ramps (owner, 2026-11-12).
    behind = planned_now - load
    restarted = xp.logical_not(first) & (state.load - load > p.restart_drop)
    paced = xp.where(first | restarted, 0.0, state.paced)
    reloading = (behind > p.reload_band) & (paced < 0.5)

    def taken(t):
        planned = plan(t)
        climbed = reload_from(load, t - state.clock, plant, xp)
        return xp.where(reloading, xp.minimum(planned, climbed), planned)

    error = T_ref - T_avg  # K, positive when cold
    dead = tavg_dead_band(p, plant, xp)
    per_load = ppm_per_load(plant)  # ppm per unit of load, negative
    dilute = plant.dilution_flow / plant.primary_mass * xp.maximum(boron, 1.0)
    borate = plant.boration_flow / plant.primary_mass * (plant.boric_acid_ppm - boron)
    xenon, I_hat, Xe_hat = xenon_rate(
        state.I_hat, state.Xe_hat, n, first, plant, p.dt, xp, obs[..., 10]
    )

    # ── The turbine: the rate boron carries, inside Tavg's band ──
    turbine = turbine_demand(
        error, carried_rate(boron, xenon, plant, xp), dead, p.lead, plant, xp
    )
    turbine = xp.where(reloading, 1.0, turbine)
    rising = (plan(state.clock + p.dt) - plan(state.clock)) / p.dt
    held_back = turbine * plant.turbine_rate < rising
    paced_next = xp.where(
        reloading,
        0.0,
        xp.where(held_back | ((paced > 0.5) & (behind > p.on_plan)), 1.0, 0.0),
    )

    # ── Rods: bank D on the offset; the rod program on Tavg down a fast ramp ──
    off = p.ao_ref - di / xp.maximum(n, 0.05)
    band = p.action_band / xp.maximum(n, 0.05)  # of offset
    ao = xp.sign(off) * xp.maximum(xp.abs(off) - band, 0.0)
    held = xp.where(first, rod, state.rods)
    rods = held + p.wd_Kc * ((ao - state.last_ao) + p.dt / p.wd_Ti * ao)
    # The plan's boron is counted, not timed: what the load the turbine will take
    # asks for, a lead ahead, is added to what is still owed, and the line sends
    # what it can of it each step. A restart lines the plant out, so it owes none.
    # While the reference holds the turbine under its plan, xenon's burnout carries
    # load with the dilution (``carried_rate``), so it comes off what is owed, and
    # the tool below does not borate it away.
    t = state.clock + p.lead
    asked = per_load * (taken(t + p.dt) - taken(t))  # ppm, negative: dilute
    helping = xp.where(paced > 0.5, xp.maximum(xenon, 0.0), 0.0)  # ppm/s
    owed = xp.where(first | restarted, 0.0, state.owed) + asked + helping * p.dt
    sent = xp.clip(owed, -dilute * p.dt, borate * p.dt)
    owed = owed - sent
    # While boration is still owed, down a ramp boron can't follow, the rod program
    # takes Tavg, and keeps it until the last of it has reached the core; the
    # offset loop picks up from where it leaves the rods (PHYSICS.md section 6). On
    # the way up the turbine waits for the boron instead.
    on_tavg = xp.where(
        owed > borate * p.dt, p.lead, xp.maximum(state.on_tavg - p.dt, 0.0)
    )
    on_tavg = xp.where(first | restarted, 0.0, on_tavg)
    fast = on_tavg > 0.0
    now = (taken(state.clock + p.dt) - taken(state.clock)) / p.dt
    rc_state = RodControlState(state.rc_lag, state.rc_moving, state.rc_started)
    rc_action, rc_state = rod_control_step(
        RodControlParams(), rc_state, obs[..., :6], xp
    )
    rods = xp.where(fast, rc_action[..., 0], rods)
    rods = xp.clip(rods, rod - p.reach, rod + p.reach)
    # Above the insertion limit where the plan puts the power by the step's end:
    # on a ramp up the limit tightens as the power rises.
    after = xp.maximum(n, now * p.dt + taken(state.clock))
    rods = xp.clip(rods, 1.0 - 2.0 * rod_insertion_limit(after, plant), p.park)

    # ── Boron, in ppm/s: the Tavg loop, the plan's boron, and the tool ──
    # The Tavg loop acts past the free band, on what is charged.
    e = xp.sign(error) * xp.maximum(xp.abs(error) - plant.e_tol, 0.0)
    Kc = 1.0 / (tavg_per_ppm(plant) * 2.0 * p.theta)  # ppm/s per K, negative
    Ti = 8.0 * p.theta
    trim = state.trim + Kc * ((e - state.last_e) + p.dt / Ti * e)
    trim = xp.clip(trim, -dilute, borate)
    # The operator's tool keeps the boron that xenon and the fuel's burnup ask for
    # as a debt, the Tavg it would leave off its program, and moves it only once
    # that Tavg nears the band's edge (the play, with the back-off). Burnup is
    # read off the cycle point: a ppm of boron is ``1 / boron_ref`` of the cycle,
    # and a refuelling's step back owes nothing. The debt starts from where Tavg
    # reads and is corrected toward it at the Tavg loop's own integral time, so
    # what the plan's boron and the line's delay leave behind does not build up
    # unseen.
    burnt = xp.where(first, 0.0, xp.maximum(cycle - state.cycle, 0.0) * book.boron_ref)
    k = plant.boron_worth / (plant.alpha_coolant + plant.alpha_fuel)  # K per ppm
    read = (T_avg - T_ref) / k
    owing = xp.where(first, read, state.debt + p.dt / Ti * (read - state.debt)) - burnt
    moved, debt = xenon_play(owing, xenon - helping, plant, p.dt, p.longest, dead, xp)
    out = ppm_flow(trim + (sent + moved) / p.dt, boron, plant, xp)
    out = xp.clip(out, -1.0, 1.0)

    new = ReferenceState(
        rods=rods, trim=trim, last_ao=ao, last_e=e, clock=state.clock + p.dt,
        down=down, up=up, rate=day_rate, depth=day_depth, I_hat=I_hat,
        Xe_hat=Xe_hat, debt=debt,
        rc_lag=rc_state.mismatch_lag, rc_moving=rc_state.moving,
        rc_started=rc_state.started, load=load + 0.0 * state.load, paced=paced_next,
        cycle=cycle + 0.0 * state.cycle, owed=owed, on_tavg=on_tavg,
    )  # fmt: skip
    return xp.stack([rods, out, turbine + 0.0 * out], axis=-1), new


def make_reactor_pid_controller() -> Controller:
    """The expert: the reference, the Westinghouse procedure run from the curve
    book, as a :class:`~target_gym.experts.pid.Controller`. :func:`rod_control_step`,
    the published rod program, is a row in the ladder's table beside it."""
    return Controller(ReferenceParams(), ReferenceState(), reference_step)
