"""Charter rule 3, by hand, for the reactor: the process ladder, one written recipe.

The pairing comes from how Mode A runs (OAPS, ICAPP 2025, section 2.1): rods hold
Tavg, and boron brings the rods back so they do not stay in the core. Gains come from
bump tests on nominal seeds 100 to 109, with no noise and a flat load, read as first
order plus delay, then SIMC with tc = theta. Nothing here looks at a cost.

Rungs, each adding to the last:

1. PI: rods on Tavg, boron on the rod position, target parked.
2. Steady-state feedforward: boron for the power defect, from the plant's balance.
3. Cascade: delta I sets the rods' target, boron still carries it there.
4. Override: past 4 % of delta I, the rods may not move the way that widens it.

Those four came first, and none of them ran a clean day: the boron loop chases the
rods through its delay. Boron's gain onto delta I is zero, so that pairing has a
relative gain of 0, and the only well-posed one is the other. So a second set of
rungs follows, written from the gain matrix and tuned the same way, after the first
set's results were seen. The report says so, and keeps both:

5. PI: rods on delta I, boron on Tavg.
6. Steady-state feedforward, as rung 2.
7. Cascade: Tavg sets a boron concentration, and boron holds it, read as a sample.
8. Static decoupler: boron takes back the reactivity the rods' loop puts in.

The rods' gain on delta I is not one number. A bank near the top barely tilts the
flux, and at half power a weak feedback lets the same tilt move the offset further.
The first tune read the gain at full power from the top, and in the half-power hold
the gain ran 2.7 to 4 times higher, past the rod loop's margin, where it locked into a
cycle of two steps. So the rods' gain is read at each point the day holds: full power
with the rods out, and half power with the rods where they centre delta I, by a small
bump, and the tune takes the largest (``hold_bump``). SIMC's margin (about 3 in gain,
Skogestad 2003, table 2) is a margin on the gain it is tuned for, so it is tuned for
the largest. This was a fix after results, for a bug the owner approved fixing.

The script keeps the best rung that trips no seed. The published rod program with
rung 1's boron loop is a row beside it, as rule 3 allows.

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
    rod_tilt,
    rod_worth,
    steady_state_temperatures,
    tavg_program,
)
from target_gym.reactor.experts import (
    RodControlParams,
    RodControlState,
    rod_control_step,
)

SEEDS = tuple(range(100, 110))
DT = 30.0
PARK = 1.0 - 2.0 * 3.0 / 228.0  # the rod position at 225 steps, on the action's scale
BAND_OVERRIDE = 0.04
STEPS_PER_MOVE = DT * 72.0 / 60.0 / 228.0 * 2.0  # the rods' reach in a step


def reset_flat(env, p, seed):
    _, s = env.reset_env(jax.random.PRNGKey(seed), p)
    far = jnp.asarray(1e8, s.n.dtype)
    return s.replace(ramp_down_at=far, ramp_up_at=far)


def simc(K, tau, theta):
    """``(Kc, Ti)``: SIMC PI at tc = theta. A process that settles inside a step is
    static, and SIMC's answer to that is a pure integral, gain 1 / (2 K theta). Read
    as a step of lag and a step of delay, that is a quarter of 1 / K a step, which
    is how ``null_policies._tavg_integrator`` read its hand bump. ``Ti`` of one step
    with this ``Kc`` adds the same again as proportional action, so the loop's
    integral is ``Kc * DT / Ti``."""
    theta = max(theta, DT)
    if tau <= DT:
        return 1.0 / (2.0 * K * 2.0 * DT) * DT, DT
    return tau / (K * 2.0 * theta), min(tau, 8.0 * theta)


def simc_integrating(k, theta):
    theta = max(theta, DT)
    return 1.0 / (k * 2.0 * theta), 8.0 * theta


def fopdt(y, base, dt=DT):
    """Gain over 1 unit, lag and delay from a step response, by the 2 % and 63 %
    crossings of its final value."""
    y = np.asarray(y) - base
    final = np.mean(y[-20:])
    sign = np.sign(final)
    theta = np.argmax(sign * y > 0.02 * abs(final)) * dt
    t63 = np.argmax(sign * y > 0.632 * abs(final)) * dt
    return final, max(t63 - theta, dt), theta


@dataclass
class Tune:
    rod_Kc: float
    rod_Ti: float
    boron_Kc: float
    boron_Ti: float
    ff: float  # boron action per unit of load rate per step
    di_Kc: float
    di_Ti: float


def policy_step(tune, rung, memory, obs):
    """One step of ``rung``. ``memory`` is (rods PI output, boron PI output, rod
    target, last Tavg error, last rod error, last delta I, last load)."""
    u, b, target, last_e, last_r, last_di, last_load = memory
    rod, load, T_avg, T_ref, di = obs[2], obs[3], obs[4], obs[5], obs[6]

    if rung >= 3:
        d_target = -tune.di_Kc * ((di - last_di) + DT / tune.di_Ti * di)
        target = jnp.clip(target + d_target, -1.0, PARK)

    e = T_ref - T_avg
    du = tune.rod_Kc * ((e - last_e) + DT / tune.rod_Ti * e)
    if rung >= 4:
        # Withdrawing raises delta I, inserting lowers it.
        wrong = ((di > BAND_OVERRIDE) & (du > 0)) | ((di < -BAND_OVERRIDE) & (du < 0))
        du = jnp.where(wrong, 0.0, du)
    u = jnp.clip(u + du, rod - STEPS_PER_MOVE, rod + STEPS_PER_MOVE)
    u = jnp.clip(u, -1.0, 1.0)

    # Boron adds where the rods sit too far out, so rods in means dilute.
    r = target - rod
    db = tune.boron_Kc * ((r - last_r) + DT / tune.boron_Ti * r)
    b = jnp.clip(b + db, -1.0, 1.0)
    ff = tune.ff * (load - last_load) if rung >= 2 else 0.0
    return (
        jnp.stack([u, jnp.clip(b + ff, -1.0, 1.0)]),
        (u, b, target, e, r, di, load),
    )


@dataclass
class Tune2:
    rod_Kc: float
    rod_Ti: float
    boron_Kc: float
    boron_Ti: float
    ff: float
    ppm_Kc: float  # outer: ppm of target per K, per sample of Tavg
    ppm_Ti: float
    inner_Kc: float  # inner: boron flow per ppm off target
    decouple: float  # boron action per unit of rod move, to cancel its worth


def policy_step2(tune, rung, memory, obs):
    """One step of rungs 5 to 8. ``memory`` is (rod demand, boron PI output, ppm
    target, last delta I, last Tavg error, last load, last rod)."""
    u, b, target, last_di, last_e, last_load, last_rod = memory
    rod, load, T_avg, T_ref, di, ppm = obs[2], obs[3], obs[4], obs[5], obs[6], obs[7]

    e_di = -di
    du = tune.rod_Kc * ((e_di + last_di) + DT / tune.rod_Ti * e_di)
    u = jnp.clip(u + du, rod - STEPS_PER_MOVE, rod + STEPS_PER_MOVE)
    u = jnp.clip(u, -1.0, 1.0)

    e = T_ref - T_avg
    if rung >= 7:
        target = target + tune.ppm_Kc * ((e - last_e) + DT / tune.ppm_Ti * e)
        out = jnp.clip(tune.inner_Kc * (target - ppm), -1.0, 1.0)
    else:
        b = jnp.clip(b + tune.boron_Kc * ((e - last_e) + DT / tune.boron_Ti * e), -1, 1)
        out = b
    if rung >= 6:
        out = out + tune.ff * (load - last_load)
    if rung >= 8:
        out = out + tune.decouple * (rod - last_rod)
    return (
        jnp.stack([u, jnp.clip(out, -1.0, 1.0)]),
        (u, b, target, di, e, load, rod),
    )


def episode(env, p, seed, act, memory):
    """One whole episode under ``act``; the true cost, the split and any trip."""
    obs, s = env.reset_env(jax.random.PRNGKey(seed), p)

    def step(carry, key):
        s, obs, memory = carry
        a, memory = act(memory, obs)
        obs, s, r, _, info = env.step_env(key, s, a, p)
        t = info["reward_terms"]
        return (s, obs, memory), (
            -r, t["tracking"], t["running"], info["tripped"], s.delta_I, s.rod_in, s.boron
        )  # fmt: skip

    keys = jax.random.split(jax.random.PRNGKey(seed + 7), p.max_steps_in_episode)
    _, (c, tr, ru, trip, di, rod, boron) = jax.lax.scan(step, (s, obs, memory), keys)
    return {
        "cost": jnp.sum(c), "tracking": jnp.sum(tr), "running": jnp.sum(ru),
        "trips": jnp.sum(trip), "past_band_steps": jnp.sum(jnp.abs(di) > p.delta_I_band),
        "max_abs_delta_I": jnp.max(jnp.abs(di)), "deepest_rods": jnp.max(rod) * 228.0,
        "boron_span": jnp.max(boron) - jnp.min(boron),
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
        "max_abs_delta_I": float(out["max_abs_delta_I"].max()),
        "deepest_rods_steps": float(out["deepest_rods"].max()),
        "boron_span_ppm": float(out["boron_span"].max()),
    }


def bumps(env, params):
    """The recipe's bump tests, seed by seed, without noise or wobble; the median."""
    p = params.replace(demand_sigma=0.0, sensor_noise_scale=0.0)

    def rollout(s, action, n):
        def step(s, a):
            _, s, _, _, _ = env.step_env(jax.random.PRNGKey(0), s, a, p)
            return s, s

        return jax.lax.scan(step, s, jnp.tile(action, (n, 1)))[1]

    tavg = lambda tr: np.asarray(0.5 * (tr.T_coolant + tr.T_cold))  # noqa: E731

    # Rods on Tavg: 60 steps in, an hour.
    K, tau, theta = [], [], []
    du = -2.0 * 60.0 / 228.0
    for seed in SEEDS:
        s = reset_flat(env, p, seed)
        base = tavg(rollout(s, jnp.array([PARK, 0.0]), 2))[-1]
        tr = rollout(s, jnp.array([PARK + du, 0.0]), 120)
        k, l, d = fopdt(tavg(tr), base)
        K.append(k / du), tau.append(l), theta.append(d)
    rod = (float(np.median(K)), float(np.median(tau)), float(np.median(theta)))
    rod_Kc, rod_Ti = simc(*rod)

    # Boron on the rods, the rods' loop closed: the slope and delay of the rod
    # position under a step of dilution, which drives the rods in, away from their
    # top stop.
    slope, delay = [], []
    tune = Tune(rod_Kc, rod_Ti, 0.0, 1.0, 0.0, 0.0, 1.0)
    for seed in SEEDS:
        s = reset_flat(env, p, seed)

        def step(carry, _):
            s, u, last_e = carry
            obs = env.get_obs(s, p)
            e = obs[5] - obs[4]
            u = u + tune.rod_Kc * ((e - last_e) + DT / tune.rod_Ti * e)
            u = jnp.clip(u, obs[2] - STEPS_PER_MOVE, obs[2] + STEPS_PER_MOVE)
            a = jnp.stack([jnp.clip(u, -1.0, 1.0), jnp.asarray(-0.2)])
            _, s, _, _, _ = env.step_env(jax.random.PRNGKey(0), s, a, p)
            return (s, u, e), 1.0 - 2.0 * s.rod_in

        init = (s, jnp.asarray(PARK), jnp.asarray(0.0))
        _, y = jax.lax.scan(step, init, None, length=120)
        y = np.asarray(y)
        delay.append(float(s.boron_delay) * DT)
        slope.append(float((y[-1] - y[-41]) / (40.0 * DT)) / -0.2)
    boron = (float(np.median(slope)), float(np.median(delay)))
    boron_Kc, boron_Ti = simc_integrating(*boron)

    # Feedforward from the balance: ppm for each unit of load, as action per
    # load per step, at the mean of the two flow rates at the reference boron.
    def needed(load):
        T_fuel, T_hot, T_cold, _ = steady_state_temperatures(load, p)
        return float(rod_balance(T_fuel, 0.5 * (T_hot + T_cold), 1.0, p))

    ppm_per_load = (needed(1.0) - needed(0.5)) / 0.5 / p.boron_worth
    rate = 0.5 * (
        p.dilution_flow * p.boron_ref
        + p.boration_flow * (p.boric_acid_ppm - p.boron_ref)
    ) / p.primary_mass  # fmt: skip
    ff = ppm_per_load / (rate * DT)

    # The cascade's outer loop: rung 1 closed, a step of the rods' target, on delta I.
    base_tune = Tune(rod_Kc, rod_Ti, boron_Kc, boron_Ti, 0.0, 0.0, 1.0)
    K, tau, theta = [], [], []
    d_target = -2.0 * 30.0 / 228.0
    for seed in SEEDS:
        s = reset_flat(env, p, seed)

        def step(carry, _):
            s, memory = carry
            a, memory = policy_step(base_tune, 1, memory, env.get_obs(s, p))
            _, s, _, _, _ = env.step_env(jax.random.PRNGKey(0), s, a, p)
            return (s, memory), s.delta_I

        zero = jnp.asarray(0.0)
        memory = (jnp.asarray(PARK), zero, jnp.asarray(PARK + d_target), zero,
                  jnp.asarray(d_target), zero, jnp.asarray(1.0))  # fmt: skip
        _, y = jax.lax.scan(step, (s, memory), None, length=240)
        k, l, d = fopdt(y, 0.0)
        K.append(k / d_target), tau.append(l), theta.append(d)
    outer = (float(np.median(K)), float(np.median(tau)), float(np.median(theta)))
    di_Kc, di_Ti = simc(*outer)

    return Tune(rod_Kc, rod_Ti, boron_Kc, boron_Ti, ff, di_Kc, di_Ti), {
        "rods_on_Tavg": rod, "boron_on_rods": boron, "dI_on_rod_target": outer,
    }  # fmt: skip


HOLD_LOAD = 0.5  # the day's low point, ``scheduled_load``
HOLD_BUMP = 10.0  # rod steps in, small: the gain is read where the rods sit


def hold_point(env, p, seed, load):
    """Lined out at ``load`` with no schedule: xenon balanced there, the rods where
    their tilt cancels the coolant's so delta I sits at 0, boron critical."""
    q = p.replace(initial_xenon_power_range=(load, load))
    _, s = env.reset_env(jax.random.PRNGKey(seed), q)
    d, far = s.n.dtype, jnp.asarray(1e8, s.n.dtype)
    T_fuel, T_hot, T_cold, T_sg = steady_state_temperatures(load, q)
    lo, hi = 3.0 / 228.0, 0.5
    target = -moderator_tilt(T_hot, T_cold, q)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rod_tilt(mid, q) < target else (lo, mid)
    y = 0.5 * (lo + hi)
    needed = rod_balance(T_fuel, 0.5 * (T_hot + T_cold), s.Xe_hat, q)
    needed = needed - (rod_worth(y, q) - rod_worth(q.rod_park, q))
    level = jnp.asarray(load, d)
    from target_gym.reactor.env import steady_state_precursors

    s = s.replace(
        n=level, C=steady_state_precursors(level, q), T_fuel=jnp.asarray(T_fuel, d),
        T_coolant=jnp.asarray(T_hot, d), T_cold=jnp.asarray(T_cold, d),
        T_sg=jnp.asarray(T_sg, d), target_n=level, T_ref=jnp.asarray(tavg_program(level, q), d),
        boron=jnp.asarray(q.boron_ref + needed / q.boron_worth, d),
        rod_in=jnp.asarray(y, d), ramp_down_at=jnp.asarray(-1e8, d), ramp_up_at=far,
    )  # fmt: skip
    return q, s, y


def hold_bump(env, params):
    """The rods' gain on delta I at the half-power hold, per unit of demand: a bump of
    ``HOLD_BUMP`` steps in from where the rods centre delta I, read as for the others.
    """
    p = params.replace(demand_sigma=0.0, sensor_noise_scale=0.0)
    du = -2.0 * HOLD_BUMP / 228.0
    K = []
    for seed in SEEDS:
        q, s, y = hold_point(env, p, seed, HOLD_LOAD)

        def step(s, a, q=q):
            _, s, _, _, _ = env.step_env(jax.random.PRNGKey(0), s, a, q)
            return s, s

        a = jnp.array([1.0 - 2.0 * y + du, 0.0])
        run = jax.lax.scan(step, s, jnp.tile(a, (120, 1)))[1]
        K.append(fopdt(np.asarray(run.delta_I), float(s.delta_I))[0] / du)
    return float(np.median(K))


def bumps_rod_on_delta_I(env, params, ff):
    """Rungs 5 to 8: the same bumps on the other pairing."""
    p = params.replace(demand_sigma=0.0, sensor_noise_scale=0.0)

    def rollout(s, action, n):
        def step(s, a):
            _, s, _, _, _ = env.step_env(jax.random.PRNGKey(0), s, a, p)
            return s, s

        return jax.lax.scan(step, s, jnp.tile(action, (n, 1)))[1]

    tavg = lambda tr: np.asarray(0.5 * (tr.T_coolant + tr.T_cold))  # noqa: E731
    du = -2.0 * 60.0 / 228.0
    K, tau, theta, k_b, delay, k_ppm, K_rT = [], [], [], [], [], [], []
    for seed in SEEDS:
        s = reset_flat(env, p, seed)
        rods = rollout(s, jnp.array([PARK + du, 0.0]), 120)
        k, l, d = fopdt(np.asarray(rods.delta_I), float(s.delta_I))
        K.append(k / du), tau.append(l), theta.append(d)
        K_rT.append(float(tavg(rods)[60:].mean() - tavg(rods)[0]) / du)
        # Boron at a fifth of full flow, rods parked: Tavg's slope past the delay.
        run = rollout(s, jnp.array([PARK, 0.2]), 120)
        y = tavg(run)
        k_b.append(float((y[-1] - y[-41]) / (40.0 * DT)) / 0.2)
        delay.append(float(s.boron_delay) * DT)
        ppm = np.asarray(run.boron)
        k_ppm.append(float((ppm[-1] - ppm[-41]) / (40.0 * DT)) / 0.2)
    held = hold_bump(env, params)
    full = float(np.median(K))
    rod = (max(full, held), float(np.median(tau)), float(np.median(theta)))
    boron = (float(np.median(k_b)), float(np.median(delay)))
    inner = (float(np.median(k_ppm)), float(np.median(delay)))
    rod_Kc, rod_Ti = simc(*rod)
    boron_Kc, boron_Ti = simc_integrating(*boron)
    inner_Kc, _ = simc_integrating(*inner)
    # The outer loop's model comes from the two above: Tavg per ppm as a static
    # gain, the rods' lag in the plant and the delay of the inner loop.
    K_ppm = boron[0] / inner[0]
    ppm_Kc, ppm_Ti = simc(K_ppm, 600.0, 2.0 * inner[1])
    decouple = float(np.median(K_rT)) / boron[0] / DT
    tune = Tune2(
        rod_Kc, rod_Ti, boron_Kc, boron_Ti, ff, ppm_Kc, ppm_Ti, inner_Kc, decouple
    )
    return tune, {
        "rods_on_delta_I": rod, "rods_on_delta_I_full_power": full,
        "rods_on_delta_I_half_power_hold": held, "boron_on_Tavg": boron,
        "boron_on_ppm": inner,
    }  # fmt: skip


def published(p, tune):
    """The rod program with rung 1's boron loop, sharing rung 1's memory."""
    rc = RodControlParams()

    def act(memory, obs):
        state, last_r, b = memory
        a, state = rod_control_step(rc, state, obs[:6])
        r = PARK - obs[2]
        b = b + tune.boron_Kc * ((r - last_r) + DT / tune.boron_Ti * r)
        b = jnp.clip(b, -1.0, 1.0)
        return jnp.stack([a[0], b]), (state, r, b)

    zero = jnp.asarray(0.0)
    return act, (RodControlState(zero, zero, zero), zero, zero)


def main():
    env = Reactor()
    p = ReactorParams()
    tune, models = bumps(env, p)
    zero = jnp.asarray(0.0)
    memory = (
        jnp.asarray(PARK),
        zero,
        jnp.asarray(PARK),
        zero,
        zero,
        zero,
        jnp.asarray(1.0),
    )
    table = {}
    for rung in (1, 2, 3, 4):
        act = lambda m, o, rung=rung: policy_step(tune, rung, m, o)  # noqa: E731
        table[f"rung {rung}"] = rows(env, p, act, memory)
    tune2, models2 = bumps_rod_on_delta_I(env, p, tune.ff)
    models.update(models2)
    for rung in (5, 6, 7, 8):
        s = env.reset_env(jax.random.PRNGKey(SEEDS[0]), p)[1]
        memory2 = (
            jnp.asarray(PARK), zero, s.boron, zero, zero, jnp.asarray(1.0), jnp.asarray(PARK),
        )  # fmt: skip
        act = lambda m, o, rung=rung: policy_step2(tune2, rung, m, o)  # noqa: E731
        table[f"rung {rung}"] = rows(env, p, act, memory2)
    act, mem = published(p, tune)
    table["rod program + boron trim"] = rows(env, p, act, mem)
    hold = lambda m, o: (jnp.array([PARK, 0.0]), m)  # noqa: E731
    table["null: rods parked, no boron"] = rows(env, p, hold, zero)
    rung_names = [k for k in table if k.startswith("rung")]
    clean = [k for k in rung_names if table[k]["seeds_tripped"] == 0]
    best = min(clean, key=lambda k: table[k]["cost_mean"]) if clean else None
    print(json.dumps({
        "bump_models_K_tau_theta": models, "tune": tune.__dict__, "tune2": tune2.__dict__, "best_rung": best, "table": table,
    }, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
