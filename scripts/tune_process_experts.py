"""Tune the process-plant experts by SIMC, from step tests on the simulator.

Run with ``uv run python scripts/tune_process_experts.py [--envs ...]
[--write] [--check]``. Without ``--write`` it prints the step-test models and
the settings; ``--write`` stores the settings in
``src/target_gym/data/pid_gains.json``; ``--check`` scores the stored expert
on the held-out seeds against the ceiling.

The recipe is SIMC (Skogestad and Grimholt, "The SIMC method for smooth PID
controller tuning", chapter 5 of *PID Control in the Third Millennium*,
Springer 2012; equation numbers below are theirs):

1. A step test at the operating point of each training seed (100-109), from
   the steady state there. Where the plant has a disturbance the response is
   taken against an unstepped run on the same noise, so only the step is fitted.
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
from target_gym.provenance import GAINS_PATH  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402
from target_gym.scoring import ceiling  # noqa: E402

TRAIN = range(100, 110)
HELD_OUT = range(200, 210)
SOURCE = "SIMC (Skogestad and Grimholt 2012) from step tests on seeds 100-109"


# ---------------------------------------------------------------------------
# Step tests and the SIMC rules
# ---------------------------------------------------------------------------


def _plant(name):
    spec = REGISTRY[name]
    env, p = spec.make_env(), spec.make_test_params()
    step = jax.jit(lambda k, s, a: env.step_env(k, s, a, p))
    reset = jax.jit(lambda k: env.reset_env(k, p))
    return env, p, step, reset


def _run(step, state, act, n, key=0):
    """Observations of ``n`` steps from ``state``, ``act(obs, state) -> raw``."""
    k = jax.random.PRNGKey(key)
    out, s = [], state
    for _ in range(n):
        obs, s, *_ = step(k, s, jnp.asarray(np.atleast_1d(act(s)), jnp.float32))
        out.append(np.asarray(obs))
        k = jax.random.split(k)[0]
    return np.array(out)


def fit_fopdt(t, dy, du, guess):
    """Least-squares ``k e^(-theta s) / (tau1 s + 1)`` fit to a step of ``du``."""

    def residual(q):
        k, tau, theta = q
        return k * du * (1 - np.exp(-np.maximum(t - theta, 0.0) / tau)) - dy

    lo, hi = [-np.inf, 1e-6, 0.0], [np.inf, np.inf, np.inf]
    return least_squares(residual, guess, bounds=(lo, hi)).x


def fit_unstable_lag(t, dy, du, guess):
    """``k' e^(-theta s) / ((s - a)(tau2 s + 1))`` fit; ``k'`` is the slope a
    lag-dominant loop is tuned on, whatever the sign of its pole."""

    def residual(q):
        kp, a, tau2, theta = q
        tt, b = np.maximum(t - theta, 0.0), 1.0 / tau2
        y = (
            kp
            / tau2
            * (
                -1 / (a * b)
                + np.exp(a * tt) / (a * (a + b))
                + np.exp(-b * tt) / (b * (a + b))
            )
        )
        return y * du - dy

    lo, hi = [0.0, 1e-3, 1e-3, 0.0], [np.inf, 20.0, 2.0, 0.2]
    return least_squares(residual, guess, bounds=(lo, hi)).x


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
    p = spec.make_test_params()
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
    from target_gym.experts.pid import siso_pid

    env, p, step, reset = _plant("first_order")
    dt = float(p.delta_t)
    models = []
    for s in TRAIN:
        target = float(reset(jax.random.PRNGKey(s))[0][1])
        raw0 = 2 * (target / p.K - p.u_min) / (p.u_max - p.u_min) - 1
        st = reset(jax.random.PRNGKey(s))[1].replace(x=jnp.float32(target))
        y = _run(step, st, lambda _s: raw0 + 0.1, 60)[:, 0]
        t = np.arange(1, len(y) + 1) * dt
        models.append(fit_fopdt(t, y - target, 0.1, (2.0, 0.5, 0.0)))
    k, tau1, theta = _median_model(models, dt)
    print(f"  model k {k:.4g} per raw, tau1 {tau1:.4g} s, theta {theta:.4g} s")

    def build(tau_c):
        Kc, Ti = simc_pi(k, tau1, theta, tau_c)
        return siso_pid(Kc, Kc / Ti, 0.0, dt, 0, 1)

    Kc, Ti = simc_pi(k, tau1, theta, _pick_tau_c("first_order", theta, build))
    return {"Kp": Kc, "Ki": Kc / Ti, "Kd": 0.0}


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
        y = _run(step, st, lambda _s, u=raw + du: u, 60)[:, 1]
        t = np.arange(1, len(y) + 1) * dt
        inner.append(fit_fopdt(t, y - T, du, (6.0, 0.7, 0.05)))
    k, tau1, theta = _median_model(inner, dt)
    Kp_T, Ti_T = simc_pi(k, tau1, theta, theta)
    print(f"  inner model k {k:.4g} K/raw, tau1 {tau1:.4g}, theta {theta:.4g} min")

    # Outer loop: C_a against the temperature setpoint, inner loop closed.
    outer = []
    for C_a, T, raw, st in points:
        dT = -0.5 if raw > 0.5 else 0.5
        inner_pi = cstr_cascade(0.0, 1.0, Kp_T, Ti_T, p).params
        loop = _InnerLoop(inner_pi, T + dT, raw * Ti_T / Kp_T)
        y = _run(step, st, loop, 80)[:, 0]
        t = np.arange(1, len(y) + 1) * dt
        outer.append(fit_fopdt(t, y - C_a, dT, (-0.01, 1.0, 0.3)))
    k, tau1, theta = _median_model(outer, dt)
    print(f"  outer model k {k:.4g} mol/L/K, tau1 {tau1:.4g}, theta {theta:.4g} min")

    def build(tau_c):
        Kp, Ti = simc_pi(k, tau1, theta, tau_c)
        return cstr_cascade(Kp, Ti, Kp_T, Ti_T, p)

    Kp, Ti = simc_pi(k, tau1, theta, _pick_tau_c("cstr", theta, build))
    return {"Kp": Kp, "Ti": Ti, "Kp_T": Kp_T, "Ti_T": Ti_T}


class _InnerLoop:
    """The cascade's inner PI on its own, holding ``T_sp``, from an integral
    preset to the steady coolant so the test starts at rest."""

    def __init__(self, prm, T_sp, integral):
        self.prm, self.T_sp, self.integral = prm, T_sp, integral

    def __call__(self, state):
        e = self.T_sp - float(state.T)
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
        base = _run(step, st, lambda _s: raw0, 120, key=s)[:, 0]
        moved = _run(step, st, lambda _s: raw0 + 2 * 0.1 / span, 120, key=s)[:, 0]
        dy = np.asarray(reagent_demand(moved, curve, np)) - np.asarray(
            reagent_demand(base, curve, np)
        )
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

    # Inner loop: T against the coolant, 1 K step, the first 12 steps.
    inner = []
    for L in levels:
        T, T_c, st = at_rest(L)
        base = _run(step, st, lambda _s: X.raw_from_coolant(T_c, p), 12)[:, 1]
        moved = _run(step, st, lambda _s: X.raw_from_coolant(T_c + 1.0, p), 12)[:, 1]
        t = np.arange(1, 13) * dt
        inner.append(fit_unstable_lag(t, moved - base, 1.0, (5.0, 2.0, 0.1, 0.0)))
    kp, a, tau2, theta = np.median(np.asarray(inner), axis=0)
    theta += dt / 2
    print(
        f"  inner model k' {kp:.4g} K/min/K, pole {a:.4g} /min, tau2 {tau2:.4g},"
        f" theta {theta:.4g} min; tau1/theta {1 / a / theta:.3g} (> 8: lag-dominant)"
    )
    Kp_T = 1.0 / (kp * 2 * theta)
    Kd_T = Kp_T * tau2

    # Outer loop: C_a against the temperature setpoint, inner PD closed.
    outer = []
    for L in levels[::3]:
        T, T_c, st = at_rest(L)
        T_sp = T + (T_c - bias) / Kp_T
        runs = []
        for sp in (T_sp, T_sp - 0.5):
            prev = [T]

            def pd(state, sp=sp, prev=prev):
                T_now = float(state.T)
                T_c = bias + Kp_T * (sp - T_now) - Kd_T * (T_now - prev[0]) / dt
                prev[0] = T_now
                return X.raw_from_coolant(np.clip(T_c, p.T_c_min, p.T_c_max), p)

            runs.append(_run(step, st, pd, 80)[:, 0])
        t = np.arange(1, 81) * dt
        outer.append(fit_fopdt(t, runs[1] - runs[0], -0.5, (-0.02, 0.5, 0.05)))
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


TUNERS = {
    "first_order": tune_first_order,
    "cstr": tune_cstr,
    "ph_neutralization": tune_ph_neutralization,
    "unstable_cstr": tune_unstable_cstr,
}


# ---------------------------------------------------------------------------
# Writing and checking
# ---------------------------------------------------------------------------


def write(name: str, settings: dict) -> None:
    """Replace ``name``'s entry in pid_gains.json with ``settings``. The
    unstable CSTR's entry also holds its setpoint limits (``HELD_GAINS``),
    which SIMC does not set, so those stay."""
    gains = json.loads(GAINS_PATH.read_text())
    kept = gains.get(name, {}) if name == "unstable_cstr" else {}
    entry = {k: v for k, v in kept.items() if k not in ("note", "tuning_rule")}
    entry.update({k: round(float(v), 6) for k, v in settings.items()})
    entry["tuning_rule"] = SOURCE
    gains[name] = entry
    GAINS_PATH.write_text(json.dumps(gains, indent=2) + "\n")
    print(f"  wrote {name}: {entry}")


def check(name: str) -> None:
    from target_gym.runners.runners import baseline_policy

    spec = REGISTRY[name]
    p = spec.make_test_params()
    eps = [
        E.run_episode(spec, p, baseline_policy(spec, "pid", p), seed=s)
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--envs", nargs="+", default=list(TUNERS), choices=TUNERS)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
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
