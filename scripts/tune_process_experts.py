"""Tune the process-plant experts by SIMC, from step tests on the simulator.

Run with ``uv run python scripts/tune_process_experts.py [--envs ...]
[--level 2] [--write] [--check]``. Without ``--write`` it prints the step-test
models and the settings; ``--write`` stores the settings in
``src/target_gym/data/pid_gains.json``, or that level's file (``--level``, the
sensor noise it tunes on); ``--check`` scores the stored expert on the
held-out seeds against the ceiling.

The recipe is SIMC (Skogestad and Grimholt, "The SIMC method for smooth PID
controller tuning", chapter 5 of *PID Control in the Third Millennium*,
Springer 2012; equation numbers below are theirs):

1. A step test at the operating point of each training seed (100-109), from
   the steady state there, in one run: hold, record a baseline, step
   (docs/proposals/expert-protocols-noise.md, N1). The response is the reading
   less the baseline's mean, noise and disturbances included, as a plant gives
   it. ``first_order`` runs the rest of that protocol; the other recipes take
   it when PR 9 rewrites them on the fixed plants.
2. A first-order-plus-delay fit (k, tau1, theta), or for a lag-dominant
   unstable loop (tau1 > 8 theta) the integrating-with-lag model read from
   the start of the response. The digital loop adds half a sample to the
   delay (half rule, 5.12).
3. The median of the fitted models, and the SIMC settings with the tight
   choice tau_c = theta (5.31): PI ``Kc = tau1 / (k (tau_c + theta))``,
   ``tau_I = min(tau1, 4 (tau_c + theta))`` (5.24-5.25); integrating with
   lag ``Kc = 1 / (k' (tau_c + theta))``, ``tau_D = tau2`` (Table 5.1).
4. tau_c is the one tuning parameter. Inner loops of a cascade stay tight;
   every other loop takes the lowest training cost per step among
   tau_c = theta, 2 theta and 3 theta, a looser one only if it is more than
   1 % cheaper (ours: practice detunes a loop that cycles). The outer loop of a cascade is tested with its inner loop closed.

Seeds 0-9 are scored and are never run here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from target_gym import eval as E  # noqa: E402
from target_gym.base import measured_state  # noqa: E402
from target_gym.identification import (  # noqa: E402
    BASELINE,
    DEAD_BAND,
    Instrument,
    Plant,
    fit,
    noise_tune,
    pick_fit,
    single_run,
)
from target_gym.provenance import GAIN_LEVELS, GAINS_PATH, gains_path  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402
from target_gym.scoring import ceiling  # noqa: E402

TRAIN = range(100, 110)
HELD_OUT = range(200, 210)
SOURCE = "SIMC (Skogestad and Grimholt 2012) from step tests on seeds 100-109"
#: The sensor noise a run tunes on and the level it writes, set by ``--level``.
LEVEL = 1.0


# ---------------------------------------------------------------------------
# Step tests and the SIMC rules
# ---------------------------------------------------------------------------


def _plant(name):
    spec = REGISTRY[name]
    env, p = spec.make_env(), spec.make_test_params(sensor_noise_scale=LEVEL)
    step = jax.jit(lambda k, s, a: env.step_env(k, s, a, p))
    reset = jax.jit(lambda k: env.reset_env(k, p))
    return env, p, step, reset


def _run(step, state, act, n, key=0, read=lambda s: s):
    """Observations of ``n`` steps from ``state``, ``act(read(state)) -> raw``.
    A loop closed in the test passes ``read=_instruments(env, p)``, so it sees
    the readings its expert will have."""
    k = jax.random.PRNGKey(key)
    out, s = [], state
    for _ in range(n):
        u = np.atleast_1d(act(read(s)))
        obs, s, *_ = step(k, s, jnp.asarray(u, jnp.float32))
        out.append(np.asarray(obs))
        k = jax.random.split(k)[0]
    return np.array(out)


def _instruments(env, p):
    return jax.jit(lambda s: measured_state(env, s, p))


def _then(hold, moved):
    """``hold`` through the baseline, then ``moved``: a run with its step in it."""
    t = itertools.count()
    return lambda s: (moved if next(t) >= BASELINE else hold)(s)


def _bump(step, st, hold, moved, n, key=0, read=lambda s: s):
    """N1: the ``n`` observations after the step, less the baseline's mean."""
    y = _run(step, st, _then(hold, moved), BASELINE + n, key, read)
    return y[BASELINE:] - y[:BASELINE].mean(axis=0)


def fit_fopdt(t, dy, du, guess):
    """Least-squares ``k e^(-theta s) / (tau1 s + 1)`` fit to a step of ``du``."""

    def residual(q):
        k, tau, theta = q
        return k * du * (1 - np.exp(-np.maximum(t - theta, 0.0) / tau)) - dy

    lo, hi = [-np.inf, 1e-6, 0.0], [np.inf, np.inf, np.inf]
    return least_squares(residual, guess, bounds=(lo, hi)).x


def fit_integrating_lag(t, dy, du, guess):
    """``k' e^(-theta s) / (s (tau2 s + 1))`` fit: the start of a lag-dominant
    response (Fig. 5.3), with ``k'`` the slope after the delay."""

    def residual(q):
        kp, tau2, theta = q
        x = np.maximum(t - theta, 0.0)
        return kp * du * (x - tau2 * (1 - np.exp(-x / tau2))) - dy

    lo, hi = [-np.inf, 1e-3, 0.0], [np.inf, 50.0, 10.0]
    return least_squares(residual, guess, bounds=(lo, hi)).x


def tau_c_for_input(k, tau1, theta, ys, umax):
    """The smallest tau_c whose first move on a setpoint step ``ys`` stays
    within ``umax``, never below theta (5.35): ``|Kc ys| <= umax``. ``tau1`` is
    ``None`` for an integrating process, whose ``k`` is the slope ``k'``."""
    if tau1 is None:
        return max(theta, abs(ys) / (abs(k) * umax) - theta)
    return max(theta, tau1 * abs(ys) / (abs(k) * umax) - theta)


def simc_pi(k, tau1, theta, tau_c):
    """SIMC PI (5.24-5.25), returned as ``(Kc, tau_I)``."""
    return tau1 / (k * (tau_c + theta)), min(tau1, 4.0 * (tau_c + theta))


def _median_model(models, dt):
    """The median of the fitted ``(k, tau1, theta)``, with half a sample added
    to the delay (5.12)."""
    m = np.median(np.asarray(models), axis=0)
    return float(m[0]), float(m[1]), float(m[2]) + dt / 2


def _train_cost(name, make):
    spec = REGISTRY[name]
    p = spec.make_test_params(sensor_noise_scale=LEVEL)
    eps = [E.run_episode(spec, p, make().host(), seed=s) for s in TRAIN]
    return E.gain(eps, E.scored_burn_in(name, p))[0]


def _pick_tau_c(name, theta, build):
    """tau_c in (theta, 2 theta, 3 theta) with the lowest training cost; a
    looser one has to beat every tighter one by more than 1 %, so a tie goes
    to SIMC's tight default."""
    costs = {m: _train_cost(name, lambda m=m: build(m * theta)) for m in (1, 2, 3)}
    best = 1
    for m in (2, 3):
        if costs[m] < 0.99 * min(costs[n] for n in range(1, m)):
            best = m
    print(f"  tau_c = n theta, training cost per step: {costs}, n = {best}")
    return best * theta


# ---------------------------------------------------------------------------
# The plants
# ---------------------------------------------------------------------------


def tune_first_order() -> dict:
    """The whole noise protocol (N1 to N7), which the other recipes take in
    PR 9. The record of the fit goes in the entry; no expert reads ``Tf`` yet."""
    spec = REGISTRY["first_order"]
    env, p, *_ = _plant("first_order")
    inst = Instrument.of(env, p, "x")
    plant = Plant(spec, p, [lambda o, s, q: o[0]], [(None, None)])
    read = single_run(
        lambda du: [plant.bump(s, 0, 0, du, 60, inst) for s in range(plant.n_seeds)],
        0.1,
        plant.room(0),
        inst,
    )
    model, spread, explained = pick_fit(read.fits)
    theta = model["theta"] + inst.half_sample
    # The raw action spans 2, so a dead band of 1 % of span is 0.02.
    out = noise_tune(model["k"], model["tau"], theta, read.nb, inst.h, DEAD_BAND * 2)
    print(
        f"  band {read.nb:.3g}, step {read.step:.3g} moves {read.ratio:.1f} bands;"
        f" spread {spread:.3g}, noise explains {explained:.3g}; k {model['k']:.4g}"
        f" per raw, tau1 {model['tau']:.4g} s, theta {theta:.4g} s, Tf {out['Tf']:.3g}"
    )
    return {
        "Kp": out["Kc"],
        "Ki": out["Kc"] / out["Ti"],
        "Kd": 0.0,
        "Tf": out["Tf"],
        "noise_band": read.nb,
        "k": model["k"],
        "tau1": model["tau"],
        "theta": out["theta"],
    }


def _cstr_steady(p, C_a):
    """T and raw coolant at which ``C_a`` is steady, from the two balances."""
    qV = p.q / p.V
    rate = qV * (p.Caf - C_a)
    T = -p.EA_over_R / np.log(rate / (p.k0 * C_a))
    heat = qV * (p.Ti - T) + (-p.deltaHr) * rate / (p.rho * p.C)
    T_c = T - heat * p.rho * p.C * p.V / p.UA
    return T, 2 * (T_c - p.T_c_min) / (p.T_c_max - p.T_c_min) - 1


def tune_cstr() -> dict:
    from target_gym.pc_gym.cstr.experts import cstr_cascade

    env, p, step, reset = _plant("cstr")
    dt = float(p.delta_t)
    points = []
    for s in TRAIN:
        target = float(reset(jax.random.PRNGKey(s))[0][2])
        T, raw = _cstr_steady(p, target)
        st = reset(jax.random.PRNGKey(s))[1]
        points.append((target, T, raw, st.replace(C_a=jnp.float32(target), T=T)))

    # Inner loop: T against the raw coolant, stepped toward mid-range.
    inner = []
    for _, T, raw, st in points:
        du = -0.1 if raw > 0 else 0.1
        dy = _bump(step, st, lambda _s, u=raw: u, lambda _s, u=raw + du: u, 60)[:, 1]
        t = np.arange(1, len(dy) + 1) * dt
        inner.append(fit_fopdt(t, dy, du, (6.0, 0.7, 0.05)))
    k, tau1, theta = _median_model(inner, dt)
    Kp_T, Ti_T = simc_pi(k, tau1, theta, theta)
    print(f"  inner model k {k:.4g} K/raw, tau1 {tau1:.4g}, theta {theta:.4g} min")

    # Outer loop: C_a against the temperature setpoint, inner loop closed.
    outer, read = [], _instruments(env, p)
    for C_a, T, raw, st in points:
        dT = -0.5 if raw > 0.5 else 0.5
        inner_pi = cstr_cascade(0.0, 1.0, Kp_T, Ti_T, p).params
        loop = _InnerLoop(inner_pi, T, T + dT, raw * Ti_T / Kp_T)
        dy = _bump(step, st, loop, loop, 80, read=read)[:, 0]
        t = np.arange(1, len(dy) + 1) * dt
        outer.append(fit_fopdt(t, dy, dT, (-0.01, 1.0, 0.3)))
    k, tau1, theta = _median_model(outer, dt)
    print(f"  outer model k {k:.4g} mol/L/K, tau1 {tau1:.4g}, theta {theta:.4g} min")

    def build(tau_c):
        Kp, Ti = simc_pi(k, tau1, theta, tau_c)
        return cstr_cascade(Kp, Ti, Kp_T, Ti_T, p)

    Kp, Ti = simc_pi(k, tau1, theta, _pick_tau_c("cstr", theta, build))
    return {"Kp": Kp, "Ti": Ti, "Kp_T": Kp_T, "Ti_T": Ti_T}


class _InnerLoop:
    """The cascade's inner PI on its own, holding ``T_sp`` through the baseline
    and ``T_moved`` after it, from an integral preset to the steady coolant so
    the test starts at rest."""

    def __init__(self, prm, T_sp, T_moved, integral):
        self.prm, self.integral = prm, integral
        self.T_sp, self.T_moved, self.t = T_sp, T_moved, 0

    def __call__(self, state):
        e = (self.T_moved if self.t >= BASELINE else self.T_sp) - float(state.T)
        self.t += 1
        integral = self.integral + e * self.prm.dt
        u = self.prm.Kp_T * (e + integral / self.prm.Ti_T)
        if -1.0 <= u <= 1.0:
            self.integral = integral
        return np.clip(u, -1.0, 1.0)


def tune_ph_neutralization() -> dict:
    from target_gym.pc_gym.ph_neutralization.experts import (
        TitrationCurve,
        ph_linearised_pi,
        reagent_demand,
    )

    env, p, step, reset = _plant("ph_neutralization")
    dt, curve = float(p.delta_t), TitrationCurve.of(p)
    span = float(p.q3_max - p.q3_min)
    models = []
    for s in TRAIN:
        st = reset(jax.random.PRNGKey(s))[1]
        raw0 = 2 * (float(st.q3) - p.q3_min) / span - 1
        act = _then(lambda _s: raw0, lambda _s: raw0 + 2 * 0.1 / span)
        y = _run(step, st, act, BASELINE + 120, key=s)[:, 0]
        demand = np.asarray(reagent_demand(y, curve, np))
        dy = demand[BASELINE:] - demand[:BASELINE].mean()
        t = np.arange(1, len(dy) + 1) * dt
        models.append(fit_fopdt(t, dy, 0.1, (1.0, 90.0, 0.0)))
    k, tau1, theta = _median_model(models, dt)
    print(f"  model k {k:.4g}, tau1 {tau1:.4g} s, theta {theta:.4g} s (demand units)")

    def build(tau_c):
        return ph_linearised_pi(*simc_pi(k, tau1, theta, tau_c), p)

    Kp, Ti = simc_pi(k, tau1, theta, _pick_tau_c("ph_neutralization", theta, build))
    return {"Kp": Kp, "Ti": Ti}


def tune_unstable_cstr() -> dict:
    from target_gym.pc_gym.unstable_cstr import experts as X
    from target_gym.pc_gym.unstable_cstr.env import steady_coolant, steady_temperature

    env, p, step, reset = _plant("unstable_cstr")
    dt, bias = float(p.delta_t), X.load_gains()["Tc_bias"]
    levels = sorted(
        {float(x) for s in TRAIN for x in reset(jax.random.PRNGKey(s))[1].target_levels}
    )

    def at_rest(L):
        st = reset(jax.random.PRNGKey(100))[1]
        T, T_c = float(steady_temperature(L, p, np)), float(
            steady_coolant(L, 0.0, p, np)
        )
        return (
            T,
            T_c,
            st.replace(
                C_a=jnp.float32(L),
                T=jnp.float32(T),
                T_j=jnp.float32(T_c),
                Ti_dev=jnp.float32(0.0),
                target_levels=jnp.full_like(st.target_levels, L),
            ),
        )

    # Inner loop: T against the coolant. Nobody holds this output, which runs
    # away (N1), so the stored P gain stays closed, 1 K lands on the coolant
    # and the fit reads the coolant it recorded, for the first 12 steps.
    read, inner = _instruments(env, p), []
    Kp_held = X.load_gains()["Kp_T"]
    for L in levels:
        T, T_c, st = at_rest(L)
        sent = []

        def loop(extra, T=T, T_c=T_c, sent=sent):
            def act(state):
                u = T_c + extra + Kp_held * (T - float(state.T))
                sent.append(np.clip(u, p.T_c_min, p.T_c_max))
                return X.raw_from_coolant(sent[-1], p)

            return act

        dy = _bump(step, st, loop(0.0), loop(1.0), 12, read=read)[:, 1]
        u = np.array(sent, float)
        inner.append(fit("integrating_lag", dt, dy, u[BASELINE:] - u[:BASELINE].mean()))
    kp, tau2, theta = (
        float(np.median([f[q] for f in inner])) for q in ("k", "tau2", "theta")
    )
    theta += dt / 2
    print(f"  inner model k' {kp:.4g} K/min/K, tau2 {tau2:.4g}, theta {theta:.4g} min")
    Kp_T = 1.0 / (kp * 2 * theta)
    Kd_T = Kp_T * tau2

    # Outer loop: C_a against the temperature setpoint, inner PD closed.
    outer = []
    for L in levels[::3]:
        T, T_c, st = at_rest(L)
        T_sp, prev = T + (T_c - bias) / Kp_T, [T]

        def pd(sp, prev=prev):
            def act(state):
                T_now = float(state.T)
                T_c = bias + Kp_T * (sp - T_now) - Kd_T * (T_now - prev[0]) / dt
                prev[0] = T_now
                return X.raw_from_coolant(np.clip(T_c, p.T_c_min, p.T_c_max), p)

            return act

        dy = _bump(step, st, pd(T_sp), pd(T_sp - 0.5), 80, read=read)[:, 0]
        t = np.arange(1, 81) * dt
        outer.append(fit_fopdt(t, dy, -0.5, (-0.02, 0.5, 0.05)))
    k, tau1, theta = _median_model(outer, dt)
    print(f"  outer model k {k:.4g} mol/L/K, tau1 {tau1:.4g}, theta {theta:.4g} min")

    def build(tau_c):
        Kc, Ti = simc_pi(k, tau1, theta, tau_c)
        g = {**X.load_gains(), "Kp_T": Kp_T, "Kd_T": Kd_T, "Kc_Ca": -Kc, "Ti_Ca": Ti}
        return _HostOnly(X.UnstableCSTRCascadePID(g))

    Kc, Ti = simc_pi(k, tau1, theta, _pick_tau_c("unstable_cstr", theta, build))
    settings = {"Kp_T": Kp_T, "Kd_T": Kd_T, "Kc_Ca": -Kc, "Ti_Ca": Ti}
    X.check_cascade_gains({**X.load_gains(), **settings})
    return settings


class _HostOnly:
    """A host controller where ``_train_cost`` expects a Controller."""

    def __init__(self, pid):
        self.pid = pid

    def host(self):
        return self.pid


def tune_four_tank() -> dict:
    from target_gym.pc_gym.four_tank.experts import (
        four_tank_decoupled_pi,
        steady_voltages,
    )

    env, p, step, reset = _plant("four_tank")
    dt = float(p.delta_t)
    prm = four_tank_decoupled_pi(1.0, 1.0, 1.0, 1.0, np.eye(2), p).params
    raw = (
        lambda v: 2 * (np.asarray(v) - p.v_min) / (p.v_max - p.v_min) - 1
    )  # noqa: E731
    g2 = np.sqrt(2 * p.g)
    rests = []
    for s in TRAIN:
        o, st = reset(jax.random.PRNGKey(s))
        t1, t2 = float(o[4]), float(o[5])
        v = np.array(steady_voltages(t1, t2, prm, np), float)
        h3 = ((1 - p.gamma2) * p.k2 * v[1] / (p.a3 * g2)) ** 2
        h4 = ((1 - p.gamma1) * p.k1 * v[0] / (p.a4 * g2)) ** 2
        st = st.replace(
            **{
                f: jnp.float32(h)
                for f, h in zip(("h1", "h2", "h3", "h4"), (t1, t2, h3, h4))
            }
        )
        rests.append((t1, t2, v, st))

    # The steady-state gains: each pump stepped 0.3 V, run to rest.
    gains = []
    for t1, t2, v, st in rests:
        G = np.zeros((2, 2))
        for j in range(2):
            dv = np.eye(2)[j] * 0.3
            hold, moved = (lambda _s, u=raw(x): u for x in (v, v + dv))
            G[:, j] = _bump(step, st, hold, moved, 2000)[-1, :2] / 0.3
        gains.append(G)
    G = np.median(np.asarray(gains), axis=0)
    D = np.linalg.solve(G, np.diag([G[0, 1], G[1, 0]]))
    print(f"  G(0) {np.round(G, 5).tolist()} m/V; decoupler {np.round(D, 4).tolist()}")

    # Each decoupled loop stepped 0.3 V in its own w.
    models = {0: [], 1: []}
    for t1, t2, v, st in rests:
        for j in range(2):
            hold, moved = (
                lambda _s, u=raw(x): u for x in (v, v + D @ (np.eye(2)[j] * 0.3))
            )
            dy = _bump(step, st, hold, moved, 1500)[:, j]
            t = np.arange(1, len(dy) + 1) * dt
            models[j].append(fit_fopdt(t, dy, 0.3, (0.04, 100.0, 10.0)))
    loops = [_median_model(models[j], dt) for j in range(2)]
    for j, (k, tau1, theta) in enumerate(loops):
        print(
            f"  loop {j + 1} model k {k:.4g} m/V, tau1 {tau1:.4g}, theta {theta:.4g} s"
        )

    def build(n):
        (k1, a1, th1), (k2, a2, th2) = loops
        (Kp_1, Ti_1), (Kp_2, Ti_2) = simc_pi(k1, a1, th1, n * th1), simc_pi(
            k2, a2, th2, n * th2
        )
        return four_tank_decoupled_pi(Kp_1, Ti_1, Kp_2, Ti_2, D, p)

    costs = {n: _train_cost("four_tank", lambda n=n: build(n)) for n in (1, 2, 3)}
    best = 1
    for n in (2, 3):
        if costs[n] < 0.99 * min(costs[m] for m in range(1, n)):
            best = n
    print(f"  tau_c = n theta, training cost per step: {costs}, n = {best}")
    c = build(best).params
    return {
        "Kp_1": c.Kp_1,
        "Ti_1": c.Ti_1,
        "Kp_2": c.Kp_2,
        "Ti_2": c.Ti_2,
        "D11": D[0, 0],
        "D12": D[0, 1],
        "D21": D[1, 0],
        "D22": D[1, 1],
    }


def tune_distillation() -> dict:
    from target_gym.pc_gym.distillation.env import tray_temperature

    env, p, step, reset = _plant("distillation")
    dt = float(p.delta_t)
    rL = lambda L: 2 * (L - p.L_min) / (p.L_max - p.L_min) - 1  # noqa: E731
    rV = lambda V: 2 * (V - p.V_min) / (p.V_max - p.V_min) - 1  # noqa: E731
    # The room each flow has from the nominal point to its nearer bound.
    room_L = min(2.706 - p.L_min, p.L_max - 2.706)
    room_V = min(3.206 - p.V_min, p.V_max - 3.206)
    starts = [reset(jax.random.PRNGKey(s)) for s in TRAIN]
    ys_D = max(abs(float(o[4] - o[0])) for o, _ in starts)
    ys_B = max(abs(float(o[5] - o[1])) for o, _ in starts)

    # Temperature loop: boilup stepped 0.005 kmol/min; the first 20 min for the
    # slope, 400 for the steady gains that size the setpoint moves.
    inner, gain_T, gain_B = [], [], []
    for s, (o, st) in zip(TRAIN, starts):
        L, V = float(st.L), float(st.V)
        hold = lambda _s, L=L, V=V: [rL(L), rV(V)]  # noqa: E731
        moved = lambda _s, L=L, V=V: [rL(L), rV(V + 0.005)]  # noqa: E731
        dy = _bump(step, st, hold, moved, 400, key=s)
        t = np.arange(1, 401) * dt
        inner.append(fit_integrating_lag(t[:20], dy[:20, 6], 0.005, (1.0, 1.0, 0.0)))
        gain_T.append(dy[-1, 6] / 0.005)
        gain_B.append(dy[-1, 1] / 0.005)
    kp, tau2, theta = np.median(np.asarray(inner), axis=0)
    theta += dt / 2
    dT_dxB = abs(np.median(gain_T) / np.median(gain_B))
    ys_T = ys_B * dT_dxB
    tau_c = tau_c_for_input(kp, None, theta, ys_T, room_V)
    Kp_T, Ti_T = 1.0 / (kp * (tau_c + theta)), 4.0 * (tau_c + theta)
    print(
        f"  T loop: k' {kp:.4g} K/min per kmol/min, tau2 {tau2:.3g}, theta {theta:.3g}"
        f" min; setpoint moves up to {ys_T:.3g} K, boilup room {room_V:.3g};"
        f" tau_c {tau_c:.3g} min"
    )

    # Composition loops, the temperature loop closed at the reset temperature
    # and moved, with the reflux, after the baseline.
    read = _instruments(env, p)

    def closed(st, L_moved, T_moved, n, key):
        L0, T0, V0 = float(st.L), float(tray_temperature(st)), float(st.V)
        mem = {"I": 0.0, "t": 0}

        def act(state):
            late = mem["t"] >= BASELINE
            mem["t"] += 1
            e = (T_moved if late else T0) - float(tray_temperature(state))
            I = mem["I"] + e * dt
            V = V0 + Kp_T * (e + I / Ti_T)
            if p.V_min <= V <= p.V_max:
                mem["I"] = I
            return [rL(L_moved if late else L0), rV(np.clip(V, p.V_min, p.V_max))]

        y = _run(step, st, act, BASELINE + n, key, read)
        return y[BASELINE:] - y[:BASELINE].mean(axis=0)

    top, bottom = [], []
    for s, (o, st) in zip(TRAIN, starts):
        L, T0 = float(st.L), float(tray_temperature(st))
        t = np.arange(1, 601) * dt
        top.append(
            fit_fopdt(
                t, closed(st, L + 0.005, T0, 600, s)[:, 0], 0.005, (0.02, 10.0, 1.0)
            )
        )
        bottom.append(
            fit_fopdt(
                t, closed(st, L, T0 - 0.2, 600, s)[:, 1], -0.2, (-0.005, 3.0, 1.0)
            )
        )
    out = {"Kp_T": Kp_T, "Ti_T": Ti_T}
    for name, models, ys, room in (("D", top, ys_D, room_L), ("B", bottom, ys_B, ys_T)):
        k, tau1, th = _median_model(models, dt)
        tc = tau_c_for_input(k, tau1, th, ys, room)
        out[f"Kp_{name}"], out[f"Ti_{name}"] = simc_pi(k, tau1, th, tc)
        print(
            f"  {'yD on L' if name == 'D' else 'xB on T_sp'}: k {k:.4g}, tau1"
            f" {tau1:.3g}, theta {th:.3g} min; tau_c {tc:.3g} min"
        )
    return out


TUNERS = {
    "first_order": tune_first_order,
    "cstr": tune_cstr,
    "ph_neutralization": tune_ph_neutralization,
    "unstable_cstr": tune_unstable_cstr,
    "four_tank": tune_four_tank,
    "distillation": tune_distillation,
}


# ---------------------------------------------------------------------------
# Writing and checking
# ---------------------------------------------------------------------------


def write(name: str, settings: dict) -> None:
    """Replace ``name``'s entry in this level's gains file with ``settings``.
    The unstable CSTR's entry also holds its setpoint limits (``HELD_GAINS``),
    which SIMC does not set, so those stay, as shipped."""
    path = gains_path(LEVEL)
    gains = json.loads(path.read_text()) if path.exists() else {}
    shipped = json.loads(GAINS_PATH.read_text())
    kept = shipped.get(name, {}) if name == "unstable_cstr" else {}
    entry = {k: v for k, v in kept.items() if k not in ("note", "tuning_rule")}
    entry.update({k: round(float(v), 6) for k, v in settings.items()})
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
            print("  settings", {k: round(float(v), 5) for k, v in settings.items()})
            if args.write:
                write(name, settings)
        if args.check:
            check(name)


if __name__ == "__main__":
    main()
