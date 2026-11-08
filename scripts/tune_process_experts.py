"""Tune the process experts by their recipes, from single-run bump tests.

Run with ``uv run python scripts/tune_process_experts.py [--envs ...]
[--level 2] [--write] [--check]``. Without ``--write`` it prints each test and
the settings; ``--write`` stores them in ``src/target_gym/data/pid_gains.json``,
or that level's file (``--level``, the sensor noise it tunes on); ``--check``
scores the stored expert on the held-out seeds, which only report.

The recipes are docs/proposals/expert-protocols-process.md as amended by
expert-protocols-noise.md and industrial-fixes.md, written down in section 9 of
the first. In short, for each loop:

1. A test starts from a plant no expert holds: in manual at the rest its seed's
   target needs, or, where nobody can leave it in manual, under a fixed loop that
   no stored gain enters. So a run does not depend on what it replaces.
2. One run per training seed (100-109): hold for an episode, record a baseline,
   step, and read the step response through ``identification`` (N1 to N4), the
   loads off where the task has any. Tests read the instruments the expert has.
3. The model is the fit C5 picks, with the half sample of C4.
4. SIMC with tau_c = theta (C6, C7), then the filter and the gain (N5 to N7) on
   valves, whose dead band is 1 % of span. Pumps and drives have no sourced
   figure, so they keep SIMC's values. N7 stops at Kc,min = u_d / NB, ours: the
   noise a loop passes need not fall below the move its largest load takes.

Seeds 0-9 are scored and are never run here.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from target_gym import eval as E  # noqa: E402
from target_gym.experts.pid import Controller  # noqa: E402
from target_gym.identification import (  # noqa: E402
    DEAD_BAND,
    Instrument,
    Loop,
    Plant,
    noise_tune,
    pick_fit,
    rga,
    single_run,
)
from target_gym.provenance import GAIN_LEVELS, gains_path  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402
from target_gym.scoring import ceiling  # noqa: E402

TRAIN = range(100, 110)
HELD_OUT = range(200, 210)
SOURCE = (
    "docs/proposals/expert-protocols-process.md, section 9: SIMC (Skogestad and "
    "Grimholt 2012) on single-run bump tests, seeds 100-109"
)
#: The sensor noise a run tunes on and the level it writes, set by ``--level``.
LEVEL = 1.0
#: Valve dead band on a raw action, which spans 2.
RAW_BAND = DEAD_BAND * 2.0
#: The steps of a test's hold before its baseline, and the length ``Plant`` keys
#: its noise on (section 9, R7). A bump test belongs to the recipe, so these do not
#: follow the task's episode. They are the lengths the recipes first ran on.
TEST_STEPS = {
    "first_order": 150,
    "cstr": 150,
    "ph_neutralization": 300,
    "four_tank": 1200,
    "distillation": 200,
    "unstable_cstr": 1200,
    "compressor_surge": 1200,
}
FREE = (-np.inf, np.inf)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _params(name, **changes):
    return REGISTRY[name].make_test_params(
        sensor_noise_scale=LEVEL, max_steps_in_episode=TEST_STEPS[name], **changes
    )


def manual(hold):
    """A loop in manual: ``hold(obs, xp)`` is the raw action, no memory."""
    return Controller(None, 0.0, lambda _p, s, obs, xp=jnp: (hold(obs, xp), s))


def read_loop(
    plant,
    name,
    mv,
    cv,
    du,
    window,
    inst,
    *,
    room,
    model="fopdt",
    seeds=None,
    loops=(),
    at=None,
    delay=None,
):
    """N1 to N4 and C4, C5 on one loop: a dict with the model and how it read.
    ``seeds`` are the plant's seed indices to test, by default all, ``loops``
    hold through the test where nobody holds the output (N1), ``at`` places the
    step in a plant's own cycle (``Plant.bump``) and ``delay`` is a datasheet's,
    held in place of the band's read."""
    seeds = range(plant.n_seeds) if seeds is None else seeds
    read = single_run(
        lambda d: [plant.bump(s, mv, cv, d, window, inst, loops, at) for s in seeds],
        du,
        room,
        inst,
        model,
        delay,
    )
    best, spread, explained = pick_fit(read.fits)
    out = {
        "k": best["k"],
        "tau1": best["tau" if model == "fopdt" else "tau2"],
        "theta": best["theta"] + inst.half_sample,
        "delay": best["theta"],
        "band": read.nb,
        "step": read.step,
        "bands": read.ratio,
        "spread": spread,
        "fit": float(np.median([f["fit"] for f in read.fits])),
        "tests": len(read.fits),
        "integrating": model != "fopdt",
    }
    print(
        f"  {name}: k {out['k']:.4g}, lag {out['tau1']:.4g}, theta {out['theta']:.4g};"
        f" band {out['band']:.3g}, step {out['step']:.3g} moves {out['bands']:.1f}"
        f" bands; spread {spread:.3g} (noise {explained:.3g}), fit"
        f" {out['fit']:.0f} %, {out['tests']} of {len(seeds)} tests"
    )
    return out


def rescaled(loop, per_unit):
    """``loop`` with its gain per MV unit, ``per_unit`` raw to the unit."""
    return {**loop, "k": loop["k"] * per_unit}


def tuned(loop, h, db=None, u_d=0.0, rule=None):
    """N5 to N7 on C6 and C7. ``db`` is the valve's dead band in the output's
    units, None on a pump or a drive. ``u_d`` is the move the largest load takes
    in those units, so Kc,min is ``u_d`` over the band. ``tau1`` None is an
    integrating loop. ``rule`` replaces C7 (``noise_tune``)."""
    out = noise_tune(
        loop["k"],
        None if loop["integrating"] else loop["tau1"],
        loop["theta"],
        loop["band"],
        h,
        db,
        u_d / loop["band"] if loop["band"] else 0.0,
        rule,
    )
    print(
        f"    Kc {out['Kc']:.4g}, Ti {out['Ti']:.4g}, Tf {out['Tf']:.3g},"
        f" tau_c {out['tau_c']:.4g}"
    )
    return out


def record(loop, out):
    """C10: what a store entry keeps of one loop's test."""
    kept = ("k", "tau1", "theta", "band", "step", "bands", "spread", "tests")
    return {
        **{k: round(float(loop[k]), 6) for k in kept},
        "tau_c": round(float(out.get("tau_c", loop["theta"])), 6),
    }


# ---------------------------------------------------------------------------
# The plants
# ---------------------------------------------------------------------------


def tune_first_order() -> dict:
    """The whole noise protocol on a lone valve, no load."""
    name = "first_order"
    spec, p = REGISTRY[name], _params(name)
    env = spec.make_env()
    # In manual at the target, from the plant's unit gain per raw of 2.
    rest = manual(lambda o, xp: xp.reshape(o[1] / 2.0, (1,)))
    plant = Plant(spec, p, [lambda o, s, q: o[0]], [(None, None)], controller=rest)
    inst = Instrument.of(env, p, "x")
    loop = read_loop(plant, "x on u", 0, 0, 0.1, 60, inst, room=plant.room(0))
    out = tuned(loop, inst.h, RAW_BAND)
    return {
        "Kp": out["Kc"],
        "Ki": out["Kc"] / out["Ti"],
        "Kd": 0.0,
        "Tf": out["Tf"],
        "noise_band": loop["band"],
        "model": {"x": record(loop, out)},
    }


def _cstr_rest(p, C_a, xp=np, Ti=None, Caf=None):
    """T and raw coolant at which ``C_a`` is steady, from the two balances."""
    Ti, Caf = (p.Ti if Ti is None else Ti), (p.Caf if Caf is None else Caf)
    qV = p.q / p.V
    rate = qV * (Caf - C_a)
    T = -p.EA_over_R / xp.log(rate / (p.k0 * C_a))
    heat = qV * (Ti - T) + (-p.deltaHr) * rate / (p.rho * p.C)
    T_c = T - heat * p.rho * p.C * p.V / p.UA
    return T, 2 * (T_c - p.T_c_min) / (p.T_c_max - p.T_c_min) - 1


def tune_cstr() -> dict:
    """Inner loop on T, then the outer on the analyser with the inner closed."""
    name = "cstr"
    spec, p = REGISTRY[name], _params(name, load_scale=0.0)
    env = spec.make_env()
    dt = float(p.delta_t)
    rest = manual(lambda o, xp: xp.reshape(_cstr_rest(p, o[2], xp)[1], (1,)))
    cvs = [lambda o, s, q: o[1], lambda o, s, q: o[0]]
    limits = [(None, None)] * 2

    # What the largest load takes (ours: both limits at once, at the band's middle).
    C_mid = float(sum(p.target_CA_range) / 2)
    T_mid, raw_mid = _cstr_rest(p, C_mid)
    T_far, raw_far = _cstr_rest(p, C_mid, Ti=p.Ti - p.Ti_load, Caf=p.Caf + p.Caf_load)
    T_near, _ = _cstr_rest(p, C_mid, Caf=p.Caf - p.Caf_load)

    inner = Plant(spec, p, cvs, limits, controller=rest)
    inst_T = Instrument.of(env, p, "T")
    T_loop = read_loop(inner, "T on coolant", 0, 0, 0.1, 60, inst_T, room=inner.room(0))
    T_out = tuned(T_loop, inst_T.h, RAW_BAND, abs(raw_far - raw_mid))
    Kp_T, Ti_T, Tf_T = T_out["Kc"], T_out["Ti"], T_out["Tf"]
    a = -np.expm1(-dt / Tf_T) if Tf_T else 1.0

    def closed(T_sp, y, m):
        """The inner PI, as the expert runs it."""
        T_f, integral = (y[0], 0.0) if m is None else m
        T_f = T_f + a * (y[0] - T_f)
        e = T_sp[0] - T_f
        u = Kp_T * (e + (integral + e * dt) / Ti_T)
        if -1.0 <= u <= 1.0:
            integral += e * dt
        return np.clip([u], -1.0, 1.0), (T_f, integral)

    # From here on the MV is the setpoint, at the temperature each seed rests at.
    outer = Plant(
        spec,
        p,
        cvs,
        limits,
        controller=rest,
        start=lambda a, m: inner._y[:, [0]],
        actuate=closed,
        mv_range=FREE,
    )
    inst_C = Instrument.of(env, p, "C_a")
    # The move the coolant's room allows, through the steady gain on T.
    room = abs(T_loop["k"]) * float(1 - np.max(np.abs(inner.u0)))
    C_loop = read_loop(outer, "C_a on T_sp", 0, 1, 0.5, 80, inst_C, room=room)
    C_out = tuned(
        C_loop, inst_C.h, RAW_BAND / abs(Kp_T), max(T_near - T_mid, T_mid - T_far)
    )
    return {
        "Kp": C_out["Kc"],
        "Ti": C_out["Ti"],
        "Kp_T": Kp_T,
        "Ti_T": Ti_T,
        "Tf_C": C_out["Tf"],
        "Tf_T": Tf_T,
        "model": {"T": record(T_loop, T_out), "C_a": record(C_loop, C_out)},
    }


def tune_ph_neutralization() -> dict:
    """C5 asks for the steepest point of the curve, and the amendment places it:
    the buffer stream off, q2 = 0, the plant's lower limit."""
    from target_gym.pc_gym.ph_neutralization.experts import (
        TitrationCurve,
        reagent_demand,
    )

    name = "ph_neutralization"
    spec = REGISTRY[name]
    env, nominal = spec.make_env(), _params(name)
    p = _params(name, q2_nominal=float(nominal.q2_min), q2_noise_std=0.0)
    curve, lab = TitrationCurve.of(nominal), TitrationCurve.of(p)
    span = float(p.q3_max - p.q3_min)

    def rest(o, xp):
        return xp.reshape(
            2 * (reagent_demand(o[2], lab, xp) - p.q3_min) / span - 1, (1,)
        )

    def demand(o, s, q):
        return reagent_demand(o[0], curve, jnp)

    plant = Plant(spec, p, [demand], [(None, None)], controller=manual(rest))
    inst = Instrument.of(env, p, "pH")
    raw_per_flow = 2 / span
    loop = read_loop(
        plant, "demand on q3", 0, 0, 0.1 * raw_per_flow, 120, inst, room=plant.room(0)
    )
    loop = rescaled(loop, raw_per_flow)
    target = float(sum(p.target_pH_range) / 2)
    # The largest load is the buffer's whole span of the demand at the target.
    u_d = abs(
        float(reagent_demand(target, curve, np))
        - float(reagent_demand(target, lab, np))
    )
    out = tuned(loop, inst.h, DEAD_BAND * span, u_d)
    return {
        "Kp": out["Kc"],
        "Ti": out["Ti"],
        "Tf": out["Tf"],
        "model": {"demand": record(loop, out)},
    }


def tune_four_tank() -> dict:
    """Steady gains and the RGA, then SIMC on each decoupled loop. Pumps, so no
    filter or gain from the noise band."""
    from target_gym.pc_gym.four_tank.experts import (
        four_tank_decoupled_pi,
        steady_voltages,
    )

    name = "four_tank"
    # The rig left alone (v5): no setpoint moves, extra outlet or cup.
    spec, p = REGISTRY[name], _params(
        name, setpoint_step=0.0, leak_fraction=0.0, cup_volume=0.0
    )
    env = spec.make_env()
    prm = four_tank_decoupled_pi(1.0, 1.0, 1.0, 1.0, np.eye(2), p).params
    raw_per_volt = 2 / float(p.v_max - p.v_min)

    def rest(o, xp):
        return (
            raw_per_volt * (xp.stack(steady_voltages(o[4], o[5], prm, xp)) - p.v_min)
            - 1
        )

    cvs = [lambda o, s, q: o[0], lambda o, s, q: o[1]]
    limits = [(None, None)] * 2
    plant = Plant(spec, p, cvs, limits, controller=manual(rest))
    inst = Instrument.of(env, p, "h1")

    # The steady-state gains: each pump stepped 0.3 V, the last 200 s of 2000.
    gains = []
    for s in range(plant.n_seeds):
        G = np.zeros((2, 2))
        for j in range(2):
            for i in range(2):
                b = plant.bump(s, j, i, 0.3 * raw_per_volt, 2000, inst)
                G[i, j] = b.dy[-200:].mean() / (b.step / raw_per_volt)
        gains.append(G)
    G = np.median(np.asarray(gains), axis=0)
    pairing = rga(G)
    D = np.linalg.solve(G, np.diag([G[0, 1], G[1, 0]]))
    print(f"  G(0) {np.round(G, 5).tolist()} m/V; RGA {np.round(pairing, 3).tolist()}")
    print(f"  decoupler {np.round(D, 4).tolist()}")

    # Each decoupled loop stepped in its own w: the MVs are v in D's coordinates.
    Dr, Di = D, np.linalg.inv(D)
    decoupled = Plant(
        spec,
        p,
        cvs,
        limits,
        controller=manual(rest),
        start=lambda a, _: a[:, -20:].mean(axis=1) @ Di.T,
        actuate=lambda u, y, m: (Dr @ u, m),
        mv_range=FREE,
    )
    # What the pumps allow: the nearer limit, through the decoupler's larger row.
    room = float(1 - np.abs(plant.u0).max()) / float(np.abs(D).sum(axis=1).max())
    out, loops = {}, {}
    for j, key in enumerate("12"):
        loop = read_loop(
            decoupled,
            f"h{key} on w{key}",
            j,
            j,
            0.3 * raw_per_volt,
            1500,
            inst,
            room=room,
        )
        loop = rescaled(loop, raw_per_volt)
        res = tuned(loop, inst.h)
        out[f"Kp_{key}"], out[f"Ti_{key}"] = res["Kc"], res["Ti"]
        loops[f"h{key}"] = record(loop, res)
    return {
        **out,
        "D11": D[0, 0],
        "D12": D[0, 1],
        "D21": D[1, 0],
        "D22": D[1, 1],
        "model": {**loops, "rga": round(float(pairing[0, 0]), 4)},
    }


#: The column's P loop on the tray while it is tested (ours, a commissioning
#: gain): nobody leaves a column in manual, and a fit reads the input it recorded,
#: so its value enters only as the input's shape. kmol/min per K.
COLUMN_HOLD = 0.16
#: Reflux and boilup at the published operating point (colamod.m).
COLUMN_REST = (2.70629, 3.20629)


def _column_gains(p, n=3000):
    """(yD, xB, T) on (L, V) at the nominal point, per kmol/min: the steady
    response with the feed and the sensors quiet, in float64, since float32 stalls
    a 194 min mode about 2e-4 short of rest."""
    from target_gym.pc_gym.distillation.env import tray_temperature
    from target_gym.pc_gym.unstable_cstr.experts import enable_x64

    spec = REGISTRY["distillation"]
    env = spec.make_env()
    quiet = spec.make_test_params(sensor_noise_scale=0.0, feed_step=0.0, zF_step=0.0)
    L0, V0 = COLUMN_REST

    def settle(L, V):
        _, state = env.reset_env(jax.random.PRNGKey(0), quiet)
        state = jax.tree_util.tree_map(
            lambda x: x.astype(float) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            state.replace(zF=jnp.float32(quiet.zF_nominal)),
        )
        a = (
            2
            * (jnp.array([L, V]) - jnp.array([p.L_min, p.V_min]))
            / (jnp.array([p.L_max - p.L_min, p.V_max - p.V_min]))
            - 1
        )

        def body(s, _):
            return env.step_env(jax.random.PRNGKey(0), s, a, quiet)[1], None

        state, _ = jax.lax.scan(body, state, None, length=n)
        return jnp.array([state.x[-1], state.x[0], tray_temperature(state)])

    # Small enough to read the slope: at 0.002 the high-gain direction already
    # bends, and dyD/dL reads 0.74 for 0.88.
    d = 1e-4
    with enable_x64():
        run = jax.jit(settle)
        return np.stack(
            [
                (np.asarray(run(L0 + d, V0)) - np.asarray(run(L0 - d, V0))) / (2 * d),
                (np.asarray(run(L0, V0 + d)) - np.asarray(run(L0, V0 - d))) / (2 * d),
            ],
            axis=1,
        )


def tune_distillation() -> dict:
    """A tight temperature loop, then two analyser loops above it with it closed.
    The pairing is read first: yD on L, the tray on V, xB on the tray's setpoint."""
    name = "distillation"
    # The feed blended steady for the tests: the loads off, as on every task.
    spec, p = REGISTRY[name], _params(name, feed_step=0.0, zF_step=0.0)
    env = spec.make_env()
    dt = float(p.delta_t)
    L_span, V_span = float(p.L_max - p.L_min), float(p.V_max - p.V_min)
    per_L, per_V = 2 / L_span, 2 / V_span
    raw_L = lambda L: per_L * (L - p.L_min) - 1  # noqa: E731
    raw_V = lambda V: per_V * (V - p.V_min) - 1  # noqa: E731

    G = _column_gains(p)
    GT = G[[0, 2]]  # yD and the tray on L and V
    # With the tray closed, V follows L and the setpoint: dV = (dT - G_TL dL) / G_TV.
    closed_G = np.array(
        [[G[i, 0] - G[i, 1] * G[2, 0] / G[2, 1], G[i, 1] / G[2, 1]] for i in (0, 1)]
    )
    pairing = {
        "yD, xB on L, V": float(rga(G[:2])[0, 0]),
        "yD, T on L, V": float(rga(GT)[0, 0]),
        "yD, xB on L, T_sp": float(rga(closed_G)[0, 0]),
    }
    print(f"  steady gains {np.round(G, 4).tolist()}; RGA(1,1) {pairing}")

    def hold_step(_p, m, obs, xp=jnp):
        first = m[0] > 0
        T0 = xp.where(first, m[1], obs[6])
        L0 = xp.where(first, m[2], p.L_min + obs[2] * L_span)
        V0 = xp.where(first, m[3], p.V_min + obs[3] * V_span)
        V = V0 + COLUMN_HOLD * (T0 - obs[6])
        action = xp.stack([raw_L(L0), raw_V(xp.clip(V, p.V_min, p.V_max))])
        return action, (xp.ones_like(T0), T0, L0, V0)

    hold = Controller(None, (0.0, 0.0, 0.0, 0.0), hold_step)
    cvs = [lambda o, s, q: o[6], lambda o, s, q: o[0], lambda o, s, q: o[1]]
    limits = [(None, None)] * 3
    settled = Plant(spec, p, cvs, limits, controller=hold)
    inst_T = Instrument(dt)
    held = (Loop(0, 1, COLUMN_HOLD * per_V, 0),)
    T_loop = read_loop(
        settled,
        "T on V",
        1,
        0,
        0.005 * per_V,
        20,
        inst_T,
        room=settled.room(1),
        model="integrating_lag",
        loops=held,
    )
    T_loop = rescaled(T_loop, per_V)
    # The feed's sourced composition step (distillation v5, which replaced the
    # 3 sd of the retired drift) moves distillate, and so boilup at fixed
    # reflux, F dz (mass balance over the column at fixed purities).
    u_d = float(_params(name).zF_step) * float(p.F)
    T_out = tuned(T_loop, inst_T.h, DEAD_BAND * V_span, u_d)
    Kp_T, Ti_T, Tf_T = T_out["Kc"], T_out["Ti"], T_out["Tf"]
    a = -np.expm1(-dt / Tf_T) if Tf_T else 1.0

    def closed(u, y, m):
        """MVs reflux, tray setpoint and the boilup at rest, the loop as run."""
        T_f, integral = (y[0], 0.0) if m is None else m
        T_f = T_f + a * (y[0] - T_f)
        e = u[1] - T_f
        V = u[2] + Kp_T * (e + (integral + e * dt) / Ti_T)
        if p.V_min <= V <= p.V_max:
            integral += e * dt
        return np.array([raw_L(u[0]), raw_V(np.clip(V, p.V_min, p.V_max))]), (
            T_f,
            integral,
        )

    def at_rest(acts, _):
        raw = acts[:, -20:].mean(axis=1)
        return np.stack(
            [
                p.L_min + (raw[:, 0] + 1) / per_L,
                settled._y[:, 0],
                p.V_min + (raw[:, 1] + 1) / per_V,
            ],
            axis=1,
        )

    outer = Plant(
        spec,
        p,
        cvs,
        limits,
        controller=hold,
        start=at_rest,
        actuate=closed,
        applied=lambda u, a: u[:2],
        mv_range=FREE,
        direction=lambda u, size: size,
    )
    inst = Instrument.of(env, p, "x")
    # C2's 5 % of the flow's span, and no further than keeps a product in its
    # target box: 0.005 from the nominal 0.99 and 0.01, through the steady gains.
    room_L = 0.005 / abs(closed_G[0, 0])
    room_T = 0.005 / abs(closed_G[1, 1])
    du_L, du_T = 0.05 * L_span, -0.05 * room_T / 0.25
    top = read_loop(outer, "yD on L", 0, 1, du_L, 600, inst, room=room_L)
    bottom = read_loop(outer, "xB on T_sp", 1, 2, du_T, 600, inst, room=room_T)
    # The gate takes the noise out, so these keep SIMC's values (N8).
    top_out, bottom_out = tuned(top, inst.h), tuned(bottom, inst.h)
    wait = inst.gate(top["band"], max(top["delay"], bottom["delay"]))["wait"]
    # The steepest ramp the tray loop follows without using up boilup's room: a
    # ramp lags by rate x (tau_c + theta) (expert-protocols-process.md, item 6).
    V_room = float(min(COLUMN_REST[1] - p.V_min, p.V_max - COLUMN_REST[1]))
    sp_rate = V_room / abs(Kp_T) / (T_out["tau_c"] + T_loop["theta"])
    print(f"  wait {wait} updates, setpoint ramp {sp_rate:.3g} K/min")
    return {
        "Kp_T": Kp_T,
        "Ti_T": Ti_T,
        "Tf_T": Tf_T,
        "Kp_D": top_out["Kc"],
        "Ti_D": top_out["Ti"],
        "Kp_B": bottom_out["Kc"],
        "Ti_B": bottom_out["Ti"],
        "band_D": top["band"],
        "band_B": bottom["band"],
        "every": inst.every,
        "wait": wait,
        "sp_rate": sp_rate,
        "model": {
            "T": record(T_loop, T_out),
            "yD": record(top, top_out),
            "xB": record(bottom, bottom_out),
            "rga": pairing,
        },
    }


#: The commissioning ladder of expert-protocols-process.md, D3, the first rung
#: Mirsky's 10 %. The schedule check climbs it.
SURGE_LADDER = (0.10, 0.125, 0.15)


def _compressor_pair(line: float) -> dict:
    """The pair at control line ``line``, from bumps at each seed's rest there."""
    from target_gym.compressor_surge import experts as X

    name = "compressor_surge"
    spec, p = REGISTRY[name], _params(name)
    env = spec.make_env()
    start = [env.reset_env(jax.random.PRNGKey(s), p)[1] for s in TRAIN]
    # The schedule holds its last levels after an episode, which is where a test
    # stands, and the drive and the valve hold the pair's rest there.
    rests = [
        X.settled_point(
            float(st.setpoint_levels[-1]),
            float(st.demand_levels[-1]),
            {"surge_line": line},
            p,
        )
        for st in start
    ]
    held = np.stack([X.raw_from_commands(r[2], r[4], p) for r in rests])
    reset = np.asarray(X.raw_from_commands(p.N_reset, p.x_reset, p))
    plant = Plant(
        spec,
        p,
        [lambda o, s, q: o[0], lambda o, s, q: X.surge_margin_from_obs(o, p)],
        [(None, None)] * 2,
        controller=manual(lambda o, xp: xp.asarray(reset)),
        start=lambda a, m: held,
    )
    inst, span = Instrument(float(p.delta_t)), float(p.N_max - p.N_min)
    per_speed = 2 / span
    speed = read_loop(
        plant, "dp on N", 0, 0, 0.05 * span * per_speed, 100, inst, room=plant.room(0)
    )
    speed = rescaled(speed, per_speed)
    speed_out = tuned(speed, inst.h)  # a drive: no sourced dead band
    active = [i for i, r in enumerate(rests) if r[4] > 0.02]
    valve = read_loop(
        plant, "margin on x", 1, 1, 0.1, 100, inst, room=plant.room(1), seeds=active
    )
    valve = rescaled(valve, 2.0)  # raw 2 is one full opening
    # The largest load in hold, the demand's 3 sd, as a recycle opening.
    u_d = (
        float(X.consumer_flow(25e3, 0.6 + 3 * p.demand_sigma, p, np))
        - float(X.consumer_flow(25e3, 0.6, p, np))
    ) / float(X.recycle_flow(25e3, 1.0, p, np))
    valve_out = tuned(valve, inst.h, DEAD_BAND, u_d)
    kp_surge = min(abs(valve_out["Kc"]), 1.0)  # Mirsky's "less than 1"
    Tt = np.sqrt(valve_out["Ti"] * (valve_out["tau_c"] + valve["theta"]))  # C8
    return {
        "kp_speed": speed_out["Kc"],
        "ki_speed": speed_out["Kc"] / speed_out["Ti"],
        "kp_surge": kp_surge,
        "ki_surge": kp_surge / valve_out["Ti"],
        "Tf_surge": valve_out["Tf"],
        "surge_line": line,
        "override_line": line / 2,
        "override_hold": 1.0 / float(p.valve_open_rate),
        "reset_band": kp_surge * max(1 - Tt / valve_out["Ti"], 0.0) * line,
        "model": {"speed": record(speed, speed_out), "valve": record(valve, valve_out)},
    }


def _compressor_rga(line: float) -> float:
    """lambda(dp, N), the median over the rests with the valve open: both loops
    stepped small, the demand and the sensors quiet."""
    from target_gym.compressor_surge import experts as X

    name = "compressor_surge"
    spec = REGISTRY[name]
    env = spec.make_env()
    p = spec.make_test_params(
        sensor_noise_scale=0.0,
        demand_sigma=0.0,
        max_steps_in_episode=TEST_STEPS[name],
    )
    rests = []
    for seed in TRAIN:
        st = env.reset_env(jax.random.PRNGKey(seed), p)[1]
        rests.append((float(st.setpoint_levels[-1]), float(st.demand_levels[-1])))
    points = [X.settled_point(*r, {"surge_line": line}, p) for r in rests]
    held = np.stack([X.raw_from_commands(r[2], r[4], p) for r in points])
    plant = Plant(
        spec,
        p,
        [lambda o, s, q: o[0], lambda o, s, q: X.surge_margin_from_obs(o, p)],
        [(None, None)] * 2,
        controller=manual(
            lambda o, xp: xp.asarray(X.raw_from_commands(p.N_reset, p.x_reset, p))
        ),
        start=lambda a, m: held,
    )
    inst, lambdas = Instrument(float(p.delta_t)), []
    for s in (i for i, r in enumerate(points) if r[4] > 0.02):
        G = np.zeros((2, 2))
        for j in range(2):
            for i in range(2):
                b = plant.bump(s, j, i, 0.02, 400, inst)
                G[i, j] = b.dy[-50:].mean() / b.step
        lambdas.append(rga(G)[0, 0])
    return float(np.median(lambdas))


def _schedule_check(gains) -> tuple:
    """The worst margin over the map's line on the training schedules, and the
    trips: what a commissioning run of the task's own moves shows."""
    from target_gym.compressor_surge import experts as X

    name = "compressor_surge"
    spec, p = REGISTRY[name], _params(name)
    env = spec.make_env()
    pair = X.CompressorSurgePair(
        {k: float(gains[k]) for k in X.DEFAULT_GAINS}, spec.make_test_params()
    )
    worst, trips = np.inf, 0
    step = jax.jit(env.step_env)
    for seed in TRAIN:
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset_env(key, p)
        pair.reset()
        for t in range(int(p.max_steps_in_episode)):
            action = jnp.asarray(pair(np.asarray(obs)), jnp.float32)
            obs, state, _, _, info = step(jax.random.fold_in(key, t), state, action, p)
            worst = min(worst, float(state.phi_min) / (2 * p.W) - 1)
            trips += int(info["tripped"])
    return worst, trips


def tune_compressor_surge() -> dict:
    """The recipe at each rung of the ladder, up to the first that passes the
    schedule check: no trip, and the margin never under the override line. A
    ladder that runs out ships its last rung (C9)."""
    from target_gym.compressor_surge import experts as X

    for line in SURGE_LADDER:
        print(f"  control line {line}")
        gains = _compressor_pair(line)
        worst, trips = _schedule_check(gains)
        passed = trips == 0 and worst >= gains["override_line"]
        print(f"  schedule: worst margin {worst:.4f}, {trips} trips, passed {passed}")
        if passed:
            break
    gains["model"]["rga"] = _compressor_rga(gains["surge_line"])
    print(f"  RGA(dp on N) {gains['model']['rga']:.3f}")
    numeric = {k: float(gains[k]) for k in X.DEFAULT_GAINS}
    margins, tripped, travel = X.battery_report(numeric)
    print(f"  battery: least margin {margins.min():.4f}, {int(tripped.sum())} trips,")
    print(
        f"  worst end travel {travel.max():.4f}; decay {X.closed_loop_decay(numeric)}"
    )
    gains["model"]["schedule"] = {"least margin": worst, "trips": trips}
    return gains


def _datasheet_pd(p):
    """The inner PD, SIMC on the datasheet: nobody holds this output, and its pole
    of 2.4 /min leaves too short a window to read it. A closed test over 36 s
    reads 12 K/min/K at a fit of 60 %, for 2.09 from the energy balance, since a
    gain and a lag give one curve there. The lag is the thermowell's, and the half
    rule (5.12) splits the jacket's between it and the delay, with the valve and
    half a step. N5 and N7 leave a loop that stabilises the plant alone (R10)."""
    slope = float(p.UA / (p.rho * p.C * p.V))
    well = float(REGISTRY["unstable_cstr"].make_env().readings[1].lag)
    jacket = float(p.tau_j)
    lag = well + jacket / 2
    theta = jacket / 2 + float(p.actuator_lag) + float(p.delta_t) / 2
    Kp = 1 / (slope * 2 * theta)
    return Kp, Kp * lag


def _pole_vector(p, level=0.55) -> tuple:
    """One coolant, so no pairing to choose: the check is which reading closes
    the inner loop. The unstable mode's size on C_a and T, in noise bands."""
    from target_gym.pc_gym.unstable_cstr.env import steady_temperature

    T = float(steady_temperature(level, p, np))
    r = float(p.k0 * np.exp(-p.EA_over_R / T))
    slope = level * r * p.EA_over_R / T**2  # of the rate, per K
    heat = -p.deltaHr / (p.rho * p.C)
    qV, ua = p.q / p.V, p.UA / (p.rho * p.C * p.V)
    A = np.array([[-qV - r, -slope], [heat * r, heat * slope - qV - ua]])
    values, vectors = np.linalg.eig(A)
    mode = vectors[:, int(np.argmax(values.real))]
    env = REGISTRY["unstable_cstr"].make_env()
    sd = float(env.readings[0].sigma), float(env.readings[1].sigma)
    return (
        float(values.real.max()),
        abs(mode[0]) / (6 * sd[0]),
        abs(mode[1]) / (6 * sd[1]),
    )


def tune_unstable_cstr() -> dict:
    """Inner PD closed from the datasheet's, outer PI with it closed, and the
    limits from the steady map (expert-protocols-process.md section 4)."""
    from target_gym.pc_gym.unstable_cstr import experts as X
    from target_gym.pc_gym.unstable_cstr.env import steady_coolant, steady_temperature

    name = "unstable_cstr"
    spec, p = REGISTRY[name], _params(name)
    env = spec.make_env()
    dt, span = float(p.delta_t), float(p.T_c_max - p.T_c_min)
    per_K = 2 / span
    Kp_T, Kd_T = _datasheet_pd(p)
    print(f"  inner PD from the datasheet: Kp {Kp_T:.4g}, Kd {Kd_T:.4g}")
    pole, on_C, on_T = _pole_vector(p)
    print(f"  pole {pole:.3g} /min; its mode reads {on_C:.3g} on C_a, {on_T:.3g} on T")

    def raw(T_c):
        return per_K * (T_c - p.T_c_min) - 1

    def settle_step(_p, m, obs, xp=jnp):
        T_sp = steady_temperature(obs[3], p, xp)
        bias = steady_coolant(obs[3], 0.0, p, xp)
        rate = xp.where(m[0] > 0, (obs[1] - m[1]) / dt, 0.0)
        T_c = bias + Kp_T * (T_sp - obs[1]) - Kd_T * rate
        return xp.reshape(raw(T_c), (1,)), (xp.ones_like(T_sp), obs[1], T_sp, bias)

    def closed(u, y, m):
        """The inner PD as the expert runs it. ``u`` is the setpoint and the
        bias, ``y`` T first."""
        last = y[0] if m is None else m
        T_c = u[1] + Kp_T * (u[0] - y[0]) - Kd_T * (y[0] - last) / dt
        return np.reshape(raw(np.clip(T_c, p.T_c_min, p.T_c_max)), (1,)), y[0]

    levels = np.linspace(*p.target_CA_range, 21)
    rest = np.array(
        [[float(steady_coolant(L, d, p, np)) for d in (-6.0, 0.0, 6.0)] for L in levels]
    )
    # Item 2: the coolant at the band's middle, no drift.
    bias = float(steady_coolant(0.55, 0.0, p, np))

    # Outer loop: C_a on the setpoint, the inner PD closed around that bias.
    def at_rest(a, m):
        T_sp = np.asarray(m[2])[:, -1] + (np.asarray(m[3])[:, -1] - bias) / Kp_T
        return np.stack([T_sp, np.full_like(T_sp, bias)], axis=1)

    test = Plant(
        spec,
        p,
        [lambda o, s, q: o[1], lambda o, s, q: o[0]],
        [(None, None)] * 2,
        controller=Controller(None, (0.0, 0.0, 0.0, 0.0), settle_step),
        start=at_rest,
        actuate=closed,
        applied=lambda u, a: u[:1],
        mv_range=FREE,
        direction=lambda u, size: size,
    )
    inst_C = Instrument.of(env, p, "C_a")
    room = float(min((rest - p.T_c_min).min(), (p.T_c_max - rest).min()))
    C_loop = read_loop(test, "C_a on T_sp", 0, 1, -0.5, 80, inst_C, room=room / Kp_T)
    # Items 3 and 4: the P loop's offset over the steady map.
    e_eq = float(np.abs(rest - bias).max()) / Kp_T
    line_margin = float(np.maximum(bias - rest, 0.0).max()) / Kp_T
    C_out = tuned(C_loop, inst_C.h, DEAD_BAND * span / Kp_T, e_eq)
    Kc, Ti_Ca = -C_out["Kc"], C_out["Ti"]
    # Item 5: the largest scheduled switch never clips the outer move.
    window = abs(Kc) * max(p.switch_sizes) + e_eq
    # Item 6: the steepest ramp the inner loop follows on the coolant's room,
    # (room / Kp_T) / (tau_c + theta), and SIMC's Kp_T is 1 / (k' (tau_c + theta)).
    sp_rate = room * float(p.UA / (p.rho * p.C * p.V))
    settings = {
        "Kp_T": Kp_T,
        "Kd_T": Kd_T,
        "Kc_Ca": Kc,
        "Ti_Ca": Ti_Ca,
        "Tf_Ca": C_out["Tf"],
        "sp_rate": sp_rate,
        "sp_below": window,
        "sp_above": window,
        "sp_line_margin": line_margin,
        "Tc_bias": bias,
    }
    try:
        X.check_cascade_gains({**X.DEFAULT_GAINS, **settings})
        print("  guard: passes")
    except X.UnstableCascadeGains as finding:
        print(f"  guard, a report (D2): {finding}")
    print(f"  slowest rates {X.closed_loop_rates({**X.DEFAULT_GAINS, **settings})}")
    return {
        **settings,
        "model": {
            "C_a": record(C_loop, C_out),
            "pole": {"rate": pole, "on C_a": on_C, "on T": on_T},
        },
    }


TUNERS = {
    "first_order": tune_first_order,
    "cstr": tune_cstr,
    "ph_neutralization": tune_ph_neutralization,
    "four_tank": tune_four_tank,
    "distillation": tune_distillation,
    "unstable_cstr": tune_unstable_cstr,
    "compressor_surge": tune_compressor_surge,
}


# ---------------------------------------------------------------------------
# Writing and checking
# ---------------------------------------------------------------------------


def write(name: str, settings: dict) -> None:
    """Replace ``name``'s entry in this level's gains file with ``settings``,
    the record of its tests (C10) included."""
    path = gains_path(LEVEL)
    gains = json.loads(path.read_text()) if path.exists() else {}
    entry = {
        k: v if isinstance(v, dict) else round(float(v), 6) for k, v in settings.items()
    }
    entry["tuning_rule"] = SOURCE
    gains[name] = entry
    path.write_text(json.dumps(gains, indent=2) + "\n")
    print(f"  wrote {name} to {path.name}: {entry}")


def check(name: str) -> None:
    spec = REGISTRY[name]
    p = spec.make_test_params(sensor_noise_scale=LEVEL)
    eps = [
        E.run_episode(spec, p, spec.make_pid_controller(LEVEL).host(), seed=s)
        for s in HELD_OUT
    ]
    burn_in, settle = E.scored_burn_in(name, p), E.fixed_settle(name)
    hold = [E.hold([ep], burn_in, "tracking", settle)[0] for ep in eps]
    failed = sum(bool(np.any(ep.failed)) for ep in eps)
    print(
        f"  held-out 200-209: hold {np.mean(hold):.3g} (worst {np.max(hold):.3g})"
        f" against a ceiling of {ceiling(name, p):.3g}; settle {settle}, "
        f"{failed} failed"
    )


def main() -> None:
    global LEVEL
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--envs", nargs="+", default=list(TUNERS), choices=TUNERS)
    parser.add_argument("--level", type=float, default=LEVEL, choices=GAIN_LEVELS)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    LEVEL = args.level
    for name in args.envs:
        print(name)
        if not args.check or args.write:
            settings = TUNERS[name]()
            shown = {k: v for k, v in settings.items() if not isinstance(v, dict)}
            print("  settings", {k: round(float(v), 5) for k, v in shown.items()})
            if args.write:
                write(name, settings)
        if args.check:
            check(name)


if __name__ == "__main__":
    main()
