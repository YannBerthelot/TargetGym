"""Charter rule 3, by hand, for the reactor: the process ladder as EDF runs Mode A.

The pairing is the plant's (docs/proposals/reactor-plant-check.md, item 4). The black
rods hold Tavg automatically, moving only past 1.6 C off program (IAEA NP-T-3.23,
annex II-5.1). The operator steers the axial offset with boron, which reaches it
through the rods: boron moves Tavg, the rods follow, and the rods move the offset
(OAPS, section 2.2). The operator has the day's plan (OAPS, section 1), so boron can
lead its delay.

Gains come from bump tests with no noise and a flat load, read as first order plus
delay or as an integrator, then SIMC with tc = theta. Every loop is tested at each
operating point the day holds, full power and the half-power hold, and keeps the
design asking the least controller gain (``most_sensitive``, PR 170). Nothing here
looks at a cost.

Rungs, each adding to the last:

1. PI: rods on Tavg with the dead band, boron on the offset through the rods.
2. Feedforward from the plan: boron for the power defect of the planned load change,
   led by the boron line's expected delay.
3. Cascade: the offset sets the rods' target position, and boron carries the rods
   there, read as an integrator.
4. Override: past 0.04 of offset, boron runs at full flow the way that closes it.
5. Rung 3 with an override on the rods' saturation instead: at their top stop with
   Tavg still low, boron takes Tavg, the textbook answer to a saturated actuator.

The published rod program is a row beside them twice: with rung 1's boron, and with
rung 5's. The owner chose the second as the reference on 2026-11-08, after the first
was traced to its boron: it never dilutes for xenon or ahead of the return to power.
The published program's nonlinear speed (8 to 72 steps/min) also copes with the rods'
7-fold gain swing, which a linear PI tuned for the worst point cannot. If that pairing
trips a seed, the best clean rung is the reference instead.

    uv run python scripts/reactor_ladder.py
"""

import json
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.env import (
    moderator_tilt,
    rod_balance,
    rod_insertion_limit,
    rod_tilt,
    rod_worth,
    scheduled_load,
    steady_state_precursors,
    steady_state_temperatures,
    tavg_program,
)
from target_gym.reactor.experts import (
    RodControlParams,
    RodControlState,
    boron_can_follow,
    ppm_flow,
    rod_control_step,
    xenon_play,
    xenon_rate,
)

SEEDS = tuple(range(100, 110))
DT = 30.0
PARK = 1.0 - 2.0 * 3.0 / 228.0  # the rod position at 225 steps, on the action's scale
DEAD_BAND = 1.6  # K, IAEA NP-T-3.23 annex II-5.1
BAND_OVERRIDE = 0.04  # of offset, inside the 0.05 band
STEPS_PER_MOVE = DT * 72.0 / 60.0 / 228.0 * 2.0  # the rods' reach in a step
HOLD_LOAD = 0.5  # the day's low point
ROD_BUMP = 30.0  # rod steps, the same at every point, ours
BORON_BUMP = (
    -0.2
)  # of full flow, dilution: it drives the rods in, away from their top stop
#: The boron line's delay the operator plans with: the middle of its drawn range,
#: 10 to 30 steps, ours, since an operator knows the line, not today's draw.
LEAD_STEPS = 20
#: The procedure's action band on the offset: half the 0.05 target band, ours,
#: fixed by the owner on 2026-11-08 before any rerun (PHYSICS.md section 6).
ACTION_BAND = 0.025


def simc(K, tau, theta):
    """``(Kc, Ti)``: SIMC PI at tc = theta. A process that settles inside a step is
    static: a pure integral read as a step of lag and a step of delay."""
    theta = max(theta, DT)
    if tau <= DT:
        return 1.0 / (2.0 * K * 2.0 * DT) * DT, DT
    return tau / (K * 2.0 * theta), min(tau, 8.0 * theta)


def simc_integrating(k, theta):
    theta = max(theta, DT)
    return 1.0 / (k * 2.0 * theta), 8.0 * theta


def most_sensitive(designs):
    """PR 170's rule: of one loop's designs at several points, the one asking the
    smallest controller gain."""
    return min(designs, key=lambda d: abs(d[0]))


def fopdt(y, base, dt=DT):
    """Gain over 1 unit, lag and delay from a step response, by the 2 % and 63 %
    crossings of its final value."""
    y = np.asarray(y) - base
    final = np.mean(y[-20:])
    sign = np.sign(final)
    theta = np.argmax(sign * y > 0.02 * abs(final)) * dt
    t63 = np.argmax(sign * y > 0.632 * abs(final)) * dt
    return final, max(t63 - theta, dt), theta


def integrator(y, dt=DT, limit=np.inf):
    """Slope over the 40 samples before the response first passes ``limit`` (its
    last 40 if it never does), and the delay to 2 % of its change by then."""
    y = np.asarray(y) - y[0]
    end = int(np.argmax(np.abs(y) > limit)) if np.any(np.abs(y) > limit) else len(y)
    end = max(end, 41)
    slope = (y[end - 1] - y[end - 41]) / (40.0 * dt)
    theta = np.argmax(np.abs(y) > 0.02 * abs(y[end - 1])) * dt
    return slope, theta


# ---------------------------------------------------------------------------
# Operating points
# ---------------------------------------------------------------------------


def point(env, p, seed, load):
    """Lined out at ``load`` with no schedule: xenon balanced there, the rods where
    their tilt cancels the coolant's so the offset sits at 0, boron critical."""
    q = p.replace(initial_xenon_power_range=(load, load))
    _, s = env.reset_env(jax.random.PRNGKey(seed), q)
    d, far = s.n.dtype, jnp.asarray(1e8, s.n.dtype)
    T_fuel, T_hot, T_cold, T_sg = steady_state_temperatures(load, q)
    lo, hi = 3.0 / 228.0, 0.5
    target = -moderator_tilt(T_hot, T_cold, q)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rod_tilt(mid, q) < target else (lo, mid)
    y = max(0.5 * (lo + hi), q.rod_park)
    needed = rod_balance(T_fuel, 0.5 * (T_hot + T_cold), s.Xe_hat, q)
    needed = needed - (rod_worth(y, q) - rod_worth(q.rod_park, q))
    level = jnp.asarray(load, d)
    return q, s.replace(
        n=level, C=steady_state_precursors(level, q), T_fuel=jnp.asarray(T_fuel, d),
        T_coolant=jnp.asarray(T_hot, d), T_cold=jnp.asarray(T_cold, d),
        T_sg=jnp.asarray(T_sg, d), target_n=level, T_ref=jnp.asarray(tavg_program(level, q), d),
        boron=jnp.asarray(q.boron_ref + needed / q.boron_worth, d),
        rod_in=jnp.asarray(y, d), ramp_up_at=far,
        # Below full load the ramp down is long past, so the turbine sits at load_min.
        ramp_down_at=jnp.where(load < 1.0, -far, far),
    )  # fmt: skip


def points(env, p, seed):
    return [point(env, p, seed, 1.0), point(env, p, seed, HOLD_LOAD)]


# ---------------------------------------------------------------------------
# The rungs
# ---------------------------------------------------------------------------


@dataclass
class Tune:
    rod_Kc: float
    rod_Ti: float
    boron_Kc: float  # rung 1: boron on the offset
    boron_Ti: float
    ff: float  # boron action per unit of planned load change per step
    ao_Kc: float  # rung 3, outer: rod target per unit of offset
    ao_Ti: float
    inner_Kc: float  # rung 3, inner: boron per unit of rod position off target
    inner_Ti: float
    t_Kc: float = 0.0  # rung 5, override: boron per K of Tavg past the dead band
    t_Ti: float = 1.0
    wd_Kc: float = 0.0  # the procedure: bank D per unit of offset
    wd_Ti: float = 1.0


def xenon_tool(I_x, Xe_x, debt, n, obs, clock, p):
    """The operator's xenon tool with its play (PHYSICS.md section 6), the same on
    every rung: the boron flow, the next iodine and xenon, and the debt."""
    rate, I_x, Xe_x = xenon_rate(I_x, Xe_x, n, clock == 0.0, p, DT, history=obs[10])
    moved, debt = xenon_play(debt, rate, p, DT, LEAD_STEPS * DT, DEAD_BAND)
    return ppm_flow(moved / DT, obs[7], p), I_x, Xe_x, debt


def dead(e):
    return jnp.sign(e) * jnp.maximum(jnp.abs(e) - DEAD_BAND, 0.0)


def make_policy(tune, rung, p):
    """``(act(memory, obs), memory0)`` for ``rung``. ``memory`` holds the two loops'
    outputs and last errors, the rods' target, the clock and the planned ramps."""

    def planned_step(clock, down, up):
        t = clock + LEAD_STEPS * DT
        return scheduled_load(t + DT, down, up, p) - scheduled_load(t, down, up, p)

    def act(m, obs):
        u, b, target, last_e, last_ao, last_r, clock, down, up, bt, I_x, Xe_x, debt = m
        n, rod, T_avg, T_ref = obs[0], obs[2], obs[4], obs[5]
        di, to_down, to_up = obs[6], obs[8], obs[9]
        # The plan, read once at the start: absolute times on the policy's clock.
        down = jnp.where(clock == 0.0, to_down * 3600.0, down)
        up = jnp.where(clock == 0.0, to_up * 3600.0, up)

        e = dead(T_ref - T_avg)
        u = u + tune.rod_Kc * ((e - last_e) + DT / tune.rod_Ti * e)
        u = jnp.clip(jnp.clip(u, rod - STEPS_PER_MOVE, rod + STEPS_PER_MOVE), -1.0, 1.0)

        # Reference minus offset: positive when the offset is low.
        ao = p.ao_ref - di / jnp.maximum(n, 0.05)
        r = last_r
        if rung >= 3:
            target = target + tune.ao_Kc * ((ao - last_ao) + DT / tune.ao_Ti * ao)
            target = jnp.clip(target, -1.0, PARK)
            r = target - rod
            b = b + tune.inner_Kc * ((r - last_r) + DT / tune.inner_Ti * r)
        else:
            b = b + tune.boron_Kc * ((ao - last_ao) + DT / tune.boron_Ti * ao)
        b = jnp.clip(b, -1.0, 1.0)
        out = b_before_override = b
        if rung >= 2:
            out = out + tune.ff * planned_step(clock, down, up)
        if rung == 4:
            # Borating lets the rods out, which raises the offset.
            out = jnp.where(ao > BAND_OVERRIDE, 1.0, out)
            out = jnp.where(ao < -BAND_OVERRIDE, -1.0, out)
        if rung >= 5:
            # Override on the rods' saturation: at their top stop with Tavg still low,
            # boron takes Tavg. Off the stop, what it built moves into the boron
            # loop's integrator, so the handback is bumpless.
            stuck = (rod >= PARK - 1e-3) & (e > 0.0)
            b = jnp.clip(b + jnp.where(stuck, 0.0, bt), -1.0, 1.0)
            bt = jnp.where(
                stuck, bt + tune.t_Kc * ((e - last_e) + DT / tune.t_Ti * e), 0.0
            )
            out = b + bt + (out - b_before_override)
        # Every rung: the operator's xenon tool, with its play.
        xenon, I_x, Xe_x, debt = xenon_tool(I_x, Xe_x, debt, n, obs, clock, p)
        out = out + xenon
        a = jnp.stack([u, jnp.clip(out, -1.0, 1.0)])
        return a, (u, b, target, e, ao, r, clock + DT, down, up, bt, I_x, Xe_x, debt)

    zero = jnp.asarray(0.0)
    memory = (jnp.asarray(PARK), zero, jnp.asarray(PARK)) + (zero,) * 10
    return act, memory


PUBLISHED = "rod program + rung 5 boron"
PROCEDURE = "bank D on delta I, boron on Tavg"


def procedure(p, tune):
    """The Westinghouse procedure (Krsko, NENE 2022, section 3; KNS 18A-534): bank D
    holds the offset, kept above its insertion limit, and boron trims Tavg past the
    rods' dead band. Boron is fed forward from the plan and from xenon, as on every
    rung. Both loops are SIMC PIs on the recipe's bump tests."""

    def planned_step(clock, down, up):
        t = clock + LEAD_STEPS * DT
        return scheduled_load(t + DT, down, up, p) - scheduled_load(t, down, up, p)

    def act(m, obs):
        u, b, last_ao, last_e, clock, down, up, I_x, Xe_x, debt, rc = m
        n, rod, T_avg, T_ref, di = obs[0], obs[2], obs[4], obs[5], obs[6]
        down = jnp.where(clock == 0.0, obs[8] * 3600.0, down)
        up = jnp.where(clock == 0.0, obs[9] * 3600.0, up)
        off = p.ao_ref - di / jnp.maximum(n, 0.05)
        # The action band: no error within ACTION_BAND of the reference.
        ao = jnp.sign(off) * jnp.maximum(jnp.abs(off) - ACTION_BAND, 0.0)
        u = u + tune.wd_Kc * ((ao - last_ao) + DT / tune.wd_Ti * ao)
        # Through a ramp boron can't follow, the rod program takes Tavg; the offset
        # loop picks up from where it leaves the rods (PHYSICS.md section 6).
        now = (
            scheduled_load(clock + DT, down, up, p) - scheduled_load(clock, down, up, p)
        ) / DT
        fast = ~boron_can_follow(now, obs[7], p)
        rc_a, rc = rod_control_step(RodControlParams(), rc, obs[:6])
        u = jnp.where(fast, rc_a[0], u)
        u = jnp.clip(u, rod - STEPS_PER_MOVE, rod + STEPS_PER_MOVE)
        # Above the insertion limit where the plan puts the power by the step's
        # end: on a ramp up the limit tightens as the power rises.
        after = jnp.maximum(n, scheduled_load(clock + DT, down, up, p))
        deepest = 1.0 - 2.0 * rod_insertion_limit(after, p)
        u = jnp.clip(u, deepest, PARK)
        e = dead(T_ref - T_avg)
        b = jnp.clip(b + tune.t_Kc * ((e - last_e) + DT / tune.t_Ti * e), -1.0, 1.0)
        xenon, I_x, Xe_x, debt = xenon_tool(I_x, Xe_x, debt, n, obs, clock, p)
        out = b + tune.ff * planned_step(clock, down, up) + xenon
        a = jnp.stack([u, jnp.clip(out, -1.0, 1.0)])
        return a, (u, b, ao, e, clock + DT, down, up, I_x, Xe_x, debt, rc)

    zero = jnp.asarray(0.0)
    rc0 = RodControlState(zero, zero, zero)
    return act, (jnp.asarray(PARK),) + (zero,) * 9 + (rc0,)


def reference(saved, p):
    """``(act, memory)`` of the reference a saved ladder run chose."""
    return reference_policy(
        saved.get("reference") or saved["best_rung"], p, Tune(**saved["tune"])
    )


def reference_policy(name, p, tune):
    if name == PUBLISHED:
        return program_with(p, tune, 5)
    if name == PROCEDURE:
        return procedure(p, tune)
    return make_policy(tune, int(name.split()[1]), p)


def program_with(p, tune, rung):
    """The published rod program on Tavg, with ``rung``'s boron: the operator's part,
    on the plant's own rod control. Rule 3 allows a published controller."""
    rc = RodControlParams()
    boron_act, boron_memory = make_policy(tune, rung, p)

    def act(memory, obs):
        state, bm = memory
        a, state = rod_control_step(rc, state, obs[:6])
        b, bm = boron_act(bm, obs)
        return jnp.stack([a[0], b[1]]), (state, bm)

    zero = jnp.asarray(0.0)
    return act, (RodControlState(zero, zero, zero), boron_memory)


def published(p, tune):
    """The published rod program on Tavg with rung 1's boron loop on the offset."""
    return program_with(p, tune, 1)


# ---------------------------------------------------------------------------
# Bump tests
# ---------------------------------------------------------------------------


def closed(env, q, s, act, memory, n):
    """``n`` steps of ``act`` on ``q`` from ``s``; the states."""

    def step(carry, _):
        s, m = carry
        a, m = act(m, env.get_obs(s, q))
        _, s, _, _, _ = env.step_env(jax.random.PRNGKey(0), s, a, q)
        return (s, m), s

    return jax.lax.scan(step, (s, memory), None, length=n)[1]


def offset(tr):
    return np.asarray(tr.delta_I) / np.asarray(tr.n)


def tavg(s):
    return np.asarray(0.5 * (s.T_coolant + s.T_cold))


def design(fits, rule):
    """The median fit over seeds at each point, and the design asking the least gain
    over the points."""
    per_point = []
    for k in range(len(fits[0])):
        fit = [
            float(np.median([f[k][i] for f in fits])) for i in range(len(fits[0][k]))
        ]
        per_point.append((rule(*fit), fit))
    return most_sensitive([d for d, _ in per_point]), [fit for _, fit in per_point]


#: Where the offset's response stops being read as a line: four times the band, ours.
#: At half power a few rod steps move it by tenths, and past this it bends to its clip.
LINEAR_OFFSET = 0.2


def boron_bump(env, base, read, limit=np.inf):
    """A step of dilution with rung 1's rods closed, at every point and seed."""

    def run(q, s):
        act0, m0 = make_policy(base, 1, q)
        m0 = (jnp.asarray(1.0 - 2.0 * s.rod_in),) + m0[1:]

        def bump(m, o):
            a, m = act0(m, o)
            return a.at[1].set(BORON_BUMP), m

        tr = closed(env, q, s, bump, m0, 240)
        slope, delay = integrator(read(tr), limit=limit)
        return slope / BORON_BUMP, max(delay, float(s.boron_delay) * DT)

    return run


def bumps(env, params):
    """The recipe's bump tests, without noise or wobble."""
    p = params.replace(demand_sigma=0.0, sensor_noise_scale=0.0)
    every = [points(env, p, seed) for seed in SEEDS]
    report = {}

    # Rods on Tavg: ROD_BUMP steps in from where the rods sit, an hour, open loop.
    def rods(q, s):
        du = -2.0 * ROD_BUMP / 228.0
        a = jnp.array([1.0 - 2.0 * s.rod_in + du, 0.0])
        tr = closed(env, q, s, lambda m, o: (a, m), jnp.asarray(0.0), 120)
        k, lag, delay = fopdt(tavg(tr), float(tavg(s)))
        return k / du, lag, delay

    fits = [[rods(q, s) for q, s in pts] for pts in every]
    (rod_Kc, rod_Ti), report["rods_on_Tavg"] = design(fits, simc)
    base = Tune(rod_Kc, rod_Ti, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)

    # Boron on the offset, through the rods: diluting drives them in, the offset falls.
    run = boron_bump(env, base, offset, LINEAR_OFFSET)
    fits = [[run(q, s) for q, s in pts] for pts in every]
    (boron_Kc, boron_Ti), report["boron_on_offset"] = design(fits, simc_integrating)

    # The cascade's inner loop: boron on the rods' position.
    run = boron_bump(env, base, lambda tr: 1.0 - 2.0 * np.asarray(tr.rod_in))
    fits = [[run(q, s) for q, s in pts] for pts in every]
    (inner_Kc, inner_Ti), report["boron_on_rods"] = design(fits, simc_integrating)

    # The outer loop: the inner closed, a step of the rods' target, on the offset.
    inner = Tune(rod_Kc, rod_Ti, 0.0, 1.0, 0.0, 0.0, 1.0, inner_Kc, inner_Ti)
    step_target = -2.0 * ROD_BUMP / 228.0

    def outer(q, s):
        act3, m0 = make_policy(inner, 3, q)
        here = jnp.asarray(1.0 - 2.0 * s.rod_in)
        tr = closed(env, q, s, act3, (here, m0[1], here + step_target) + m0[3:], 360)
        k, lag, delay = fopdt(offset(tr), float(s.delta_I / s.n))
        return k / step_target, lag, delay

    fits = [[outer(q, s) for q, s in pts] for pts in every]
    (ao_Kc, ao_Ti), report["offset_on_rod_target"] = design(fits, simc)

    # Feedforward from the balance: ppm for each unit of load, as action per load per
    # step, at the mean of the two flow rates at the reference boron.
    def needed(load):
        T_fuel, T_hot, T_cold, _ = steady_state_temperatures(load, p)
        return float(rod_balance(T_fuel, 0.5 * (T_hot + T_cold), 1.0, p))

    ppm_per_load = (needed(1.0) - needed(HOLD_LOAD)) / (1.0 - HOLD_LOAD) / p.boron_worth
    rate = 0.5 * (
        p.dilution_flow * p.boron_ref
        + p.boration_flow * (p.boric_acid_ppm - p.boron_ref)
    ) / p.primary_mass  # fmt: skip
    ff = ppm_per_load / (rate * DT)

    # The override's loop: boron on Tavg with the rods held where they sit, read as an
    # integrator past the line's delay.
    def boron_on_tavg(q, s):
        a = jnp.array([1.0 - 2.0 * s.rod_in, BORON_BUMP])
        tr = closed(env, q, s, lambda m, o: (a, m), jnp.asarray(0.0), 120)
        slope, delay = integrator(tavg(tr))
        return slope / BORON_BUMP, max(delay, float(s.boron_delay) * DT)

    fits = [[boron_on_tavg(q, s) for q, s in pts] for pts in every]
    (t_Kc, t_Ti), report["boron_on_Tavg"] = design(fits, simc_integrating)

    # The procedure's rods: ROD_BUMP steps in, boron held, an hour, on the offset.
    def rods_on_offset(q, s):
        du = -2.0 * ROD_BUMP / 228.0
        a = jnp.array([1.0 - 2.0 * s.rod_in + du, 0.0])
        tr = closed(env, q, s, lambda m, o: (a, m), jnp.asarray(0.0), 120)
        k, lag, delay = fopdt(offset(tr), float(s.delta_I / s.n))
        return k / du, lag, delay

    fits = [[rods_on_offset(q, s) for q, s in pts] for pts in every]
    (wd_Kc, wd_Ti), report["rods_on_offset"] = design(fits, simc)
    tune = Tune(
        rod_Kc,
        rod_Ti,
        boron_Kc,
        boron_Ti,
        ff,
        ao_Kc,
        ao_Ti,
        inner_Kc,
        inner_Ti,
        t_Kc,
        t_Ti,
        wd_Kc,
        wd_Ti,
    )
    return tune, report


# ---------------------------------------------------------------------------
# Scoring the rungs
# ---------------------------------------------------------------------------


def episode(env, p, seed, act, memory):
    """One whole episode under ``act``; the true cost, the split and any trip."""
    obs, s = env.reset_env(jax.random.PRNGKey(seed), p)

    def step(carry, key):
        s, obs, memory = carry
        a, memory = act(memory, obs)
        obs, s, r, _, info = env.step_env(key, s, a, p)
        t = info["reward_terms"]
        past_ril = s.rod_in > rod_insertion_limit(s.n, p)
        return (s, obs, memory), (
            -r, t["tracking"], t["running"], info["tripped"], s.delta_I / s.n,
            s.rod_in, s.boron, 0.5 * (s.T_coolant + s.T_cold) - s.T_ref, past_ril,
        )  # fmt: skip

    keys = jax.random.split(jax.random.PRNGKey(seed + 7), p.max_steps_in_episode)
    _, (c, tr, ru, trip, ao, rod, boron, e, ril) = jax.lax.scan(
        step, (s, obs, memory), keys
    )
    return {
        "cost": jnp.sum(c), "tracking": jnp.sum(tr), "running": jnp.sum(ru),
        "trips": jnp.sum(trip),
        "past_band_steps": jnp.sum(jnp.abs(ao - p.ao_ref) > p.ao_band),
        "max_abs_offset": jnp.max(jnp.abs(ao - p.ao_ref)),
        "past_dead_band_steps": jnp.sum(jnp.abs(e) > p.e_tol),
        "deepest_rods": jnp.max(rod) * 228.0,
        "boron_span": jnp.max(boron) - jnp.min(boron),
        "past_ril_steps": jnp.sum(ril),
    }  # fmt: skip


def rows(env, p, act, memory):
    run = jax.jit(jax.vmap(lambda seed: episode(env, p, seed, act, memory)))
    out = jax.tree.map(np.asarray, run(jnp.asarray(SEEDS)))
    return {
        "cost_mean": float(out["cost"].mean()),
        "cost_by_seed": [round(float(x), 1) for x in out["cost"]],
        "tracking": float(out["tracking"].mean()),
        "running": float(out["running"].mean()),
        "seeds_tripped": int(np.sum(out["trips"] > 0)),
        "past_band_steps": float(out["past_band_steps"].mean()),
        "max_abs_offset": float(out["max_abs_offset"].max()),
        "past_dead_band_steps": float(out["past_dead_band_steps"].mean()),
        "deepest_rods_steps": float(out["deepest_rods"].max()),
        "boron_span_ppm": float(out["boron_span"].max()),
        "past_ril_steps": float(out["past_ril_steps"].max()),
    }


def main():
    env = Reactor()
    p = ReactorParams()
    tune, models = bumps(env, p)
    table = {}
    for rung in (1, 2, 3, 4, 5):
        act, memory = make_policy(tune, rung, p)
        table[f"rung {rung}"] = rows(env, p, act, memory)
    act, memory = published(p, tune)
    table["rod program + boron on offset"] = rows(env, p, act, memory)
    act, memory = program_with(p, tune, 5)
    table[PUBLISHED] = rows(env, p, act, memory)
    act, memory = procedure(p, tune)
    table[PROCEDURE] = rows(env, p, act, memory)
    hold = lambda m, o: (jnp.array([PARK, 0.0]), m)  # noqa: E731
    table["null: rods parked, no boron"] = rows(env, p, hold, jnp.asarray(0.0))
    rung_names = [k for k in table if k.startswith("rung")]
    clean = [k for k in rung_names if table[k]["seeds_tripped"] == 0]
    best = min(clean, key=lambda k: table[k]["cost_mean"]) if clean else None
    # The candidates (owner, 2026-11-08): the rungs, the published rod program with
    # the recipe's boron, and the published procedure, if it keeps the insertion
    # limit. The reference is the cheapest that trips no seed.
    candidates = [k for k in rung_names + [PUBLISHED, PROCEDURE]
                  if table[k]["seeds_tripped"] == 0]  # fmt: skip
    reference = (
        min(candidates, key=lambda k: table[k]["cost_mean"]) if candidates else None
    )
    by_scenario = {}
    if reference is not None:
        from target_gym import scenarios
        from target_gym.registry import REGISTRY

        for sc in scenarios.suite(REGISTRY["reactor"]):
            q = ReactorParams(**sc.overrides)
            act, memory = reference_policy(reference, q, tune)
            by_scenario[sc.name] = {
                "reference": rows(env, q, act, memory),
                "null": rows(env, q, hold, jnp.asarray(0.0)),
            }
    print(json.dumps({
        "bump_models": models, "tune": tune.__dict__, "best_rung": best,
        "reference": reference, "table": table, "by_scenario": by_scenario,
    }, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
