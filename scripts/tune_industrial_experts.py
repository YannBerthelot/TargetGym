"""Tune the industrial experts by their field's recipe, from step tests on the
simulator's training seeds.

Run with ``uv run python scripts/tune_industrial_experts.py [--envs ...]
[--write] [--check]``. Without ``--write`` it prints the step-test models and
the settings; ``--write`` stores them in ``src/target_gym/data/pid_gains.json``;
``--check`` scores the stored expert on the held-out seeds against the ceiling.

* ``boiler_drum``: three-element control, both loops lambda-tuned. A step on
  the feedwater trim (pressure loop in auto) and one on the firing (level loop
  in auto). Drum level and pressure both integrate their input, so each is
  fitted as ``k' e^(-theta s) / s`` and lambda-tuned for an integrating
  process: ``Kc = (2 lam + theta) / (k' (lam + theta)^2)``,
  ``Ti = 2 lam + theta``. The level's inverse response (shrink) is in its
  fitted delay.
* ``glass_furnace``: the crown-temperature PI by SIMC. The crown responds on
  two time scales and is still rising hours after a firing step, so it is
  lag-dominant and SIMC tunes it on the integrating approximation of its
  first hour: ``Kc = 1 / (k' (tau_c + theta))``, ``tau_I = 4 (tau_c + theta)``.
* ``compressor_surge``: the vendor anti-surge pair keeps its structure and its
  held gains; the guarded coordinate search of ``scripts/tune_pid.py`` runs
  on the training seeds, from the package's ``DEFAULT_GAINS``.

A step test runs from the steady state each training seed (100-109) reaches
under the shipped expert, and its response is taken against an unstepped run
on the same noise. The digital loop adds half a sample to the delay (SIMC's
half rule). The closed-loop time constant (``lam`` or ``tau_c``) is the one
tuning choice: ``theta``, ``2 theta`` or ``3 theta``, the looser one only if it
is more than 1 % cheaper over the training seeds (the rule the process-plant
experts use, ``scripts/tune_process_experts.py``). Seeds 0-9 are scored and
are never run here.
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

import target_gym.experts.pid as pid_mod  # noqa: E402
from target_gym import eval as E  # noqa: E402
from target_gym.provenance import GAINS_PATH  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402
from target_gym.scoring import ceiling  # noqa: E402

TRAIN = range(100, 110)
HELD_OUT = range(200, 210)


# ---------------------------------------------------------------------------
# Step tests, fits and rules
# ---------------------------------------------------------------------------


def step_response(name, seed, act, n_settle, n_test, delta):
    """``act(expert, obs, t, delta) -> raw`` drives the plant; the response of
    observation channel ``act.channel`` to ``delta`` against ``0``."""
    spec = REGISTRY[name]
    env = spec.make_env()
    p = spec.make_test_params(max_steps_in_episode=n_settle + n_test + 5)
    step = jax.jit(env.step_env)

    def run(d):
        expert = spec.make_pid_controller().host()
        expert.reset()
        act.reset()
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset_env(key, p)
        ys = []
        for t in range(n_settle + n_test):
            a = act(expert, np.asarray(obs), t - n_settle, d)
            obs, state, *_ = step(jax.random.fold_in(key, t), state, jnp.asarray(a), p)
            if t >= n_settle:
                ys.append(float(np.asarray(obs)[act.channel]))
        return np.array(ys)

    t = np.arange(1, n_test + 1) * float(p.delta_t)
    return t, run(delta) - run(0.0), float(p.delta_t)


def fit_integrating(t, dy, du):
    """Least-squares ``k' e^(-theta s) / s``: ``dy = k' du max(t - theta, 0)``."""

    def residual(q):
        return q[0] * du * np.maximum(t - q[1], 0.0) - dy

    slope = (dy[-1] - dy[len(dy) // 2]) / (t[-1] - t[len(t) // 2]) / du
    return least_squares(
        residual, [slope, t[1]], bounds=([-np.inf, 0], [np.inf, t[-1]])
    ).x


def fit_fopdt(t, dy, du):
    """Least-squares ``k e^(-theta s) / (tau s + 1)`` fit to a step of ``du``."""

    def residual(q):
        k, tau, theta = q
        return k * du * (1 - np.exp(-np.maximum(t - theta, 0.0) / tau)) - dy

    guess = [dy[-1] / du, t[len(t) // 4], 0.0]
    return least_squares(residual, guess, bounds=([-np.inf, 1e-6, 0], np.inf)).x


def simc_pi(k, tau, theta, tau_c):
    """SIMC PI (Skogestad 2003): ``(Kc, Ti)``."""
    return tau / (k * (tau_c + theta)), min(tau, 4.0 * (tau_c + theta))


def lambda_integrating(k, theta, lam):
    """Lambda tuning of an integrating process: ``(Kc, Ti)``."""
    return (2 * lam + theta) / (k * (lam + theta) ** 2), 2 * lam + theta


def simc_integrating(k, theta, tau_c):
    """SIMC PI on an integrating process (Skogestad 2003, Table 1)."""
    return 1.0 / (k * (tau_c + theta)), 4.0 * (tau_c + theta)


def train_cost(name, settings):
    """Mean cost per step over the training seeds with ``settings`` stored."""
    spec = REGISTRY[name]
    p = spec.make_test_params()
    stored = dict(pid_mod.load_gains())
    pid_mod._gains_cache = {**stored, name: {**stored.get(name, {}), **settings}}
    try:
        episodes = [
            E.run_episode(spec, p, spec.make_pid_controller().host(), seed=s)
            for s in TRAIN
        ]
        return float(np.mean([ep.cost.mean() for ep in episodes]))
    finally:
        pid_mod._gains_cache = stored


def pick(name, theta, build):
    """``theta``, ``2 theta`` or ``3 theta``: a looser one only if more than
    1 % cheaper than every tighter one on the training seeds."""
    costs = {m: train_cost(name, build(m * theta)) for m in (1, 2, 3)}
    best = 1
    for m in (2, 3):
        if costs[m] < 0.99 * min(costs[n] for n in range(1, m)):
            best = m
    print(f"  n theta -> training cost per step {costs}; n = {best}")
    return best


# ---------------------------------------------------------------------------
# The plants
# ---------------------------------------------------------------------------


class _OpenLoop:
    """The expert in auto, but for one output, which is put in manual at the
    average of its last ``window`` commands (and observations) and stepped:
    ``hold_fn(actions, observations) -> (obs, delta) -> raw``."""

    def __init__(self, channel, index, hold_fn, window=20):
        self.channel, self.index = channel, index
        self.hold_fn, self.window = hold_fn, window

    def reset(self):
        self.held, self.history = None, []

    def __call__(self, expert, obs, t, d):
        a = np.array(expert(obs), float)
        if t < 0:
            self.history = (self.history + [(a, obs)])[-self.window :]
            return a
        if self.held is None:
            acts, obss = (np.array(x) for x in zip(*self.history))
            self.held = self.hold_fn(acts.mean(0), obss.mean(0))
        a[self.index] = self.held(obs, d)
        return a


def tune_boiler_drum() -> dict:
    from target_gym.boiler_drum.env import BoilerDrumParams

    q_max = float(BoilerDrumParams().q_feed_max)

    def feed_trim(a, obs):  # feed = steam + trim, the level loop opened
        trim = 0.5 * (a[1] + 1.0) * q_max - obs[2]
        return lambda o, d: 2.0 * np.clip(o[2] + trim + d, 0.0, q_max) / q_max - 1.0

    def firing(a, obs):  # the firing held, the pressure loop opened
        u0 = float(a[0])
        return lambda o, d: float(np.clip(u0 + d, -1.0, 1.0))

    out = {}
    for loop, channel, index, hold, du in (
        ("pressure", 1, 0, firing, 0.02),
        ("level", 0, 1, feed_trim, 2.0),
    ):
        models = []
        for s in TRAIN:
            t, dy, dt = step_response(
                "boiler_drum", s, _OpenLoop(channel, index, hold), 150, 120, du
            )
            models.append(fit_integrating(t, dy, du))
        k, theta = np.median(models, axis=0)
        theta = float(theta) + dt / 2
        print(f"{loop}: k' {k:.4g} per s, theta {theta:.3g} s (with half a sample)")
        # The firing PI adds Q_max (Kp e + Ki int e) to the heat, and the
        # action is 2 Q / Q_max - 1, so its gain on the action is 2 Kp.
        scale = 0.5 if loop == "pressure" else 1.0

        def build(lam, k=k, theta=theta, loop=loop, scale=scale):
            Kc, Ti = lambda_integrating(k, theta, lam)
            return {f"Kp_{loop}": scale * Kc, f"Ki_{loop}": scale * Kc / Ti}

        m = pick("boiler_drum", theta, build)
        out.update(build(m * theta))
        out[f"model_{loop}"] = {"k": float(k), "theta": theta, "lam": m * theta}
        pid_mod._gains_cache = {
            **pid_mod.load_gains(),
            "boiler_drum": {**pid_mod.load_gains()["boiler_drum"], **out},
        }
    out["Kd_pressure"] = 0.0
    out["tuning_rule"] = "lambda_integrating"
    out["note"] = (
        "Three-element: feedwater follows measured steam flow, a level PI trims "
        "it; firing is an enthalpy feedforward plus a pressure PI. Both loops "
        "lambda-tuned for an integrating process from step tests on seeds "
        "100-109 (scripts/tune_industrial_experts.py)."
    )
    return out


def tune_glass_furnace() -> dict:
    from target_gym.glass_furnace.env import GlassFurnaceParams

    # Firing stops while the regenerators reverse, and the PI waits at full
    # demand through it: the manual output is its average over one cycle.
    gp = GlassFurnaceParams()
    WINDOW = int(round(float(gp.reversal_period) / float(gp.delta_t)))

    def hold(a, obs):  # the firing held at its last command, the loop opened
        u0 = float(a[0])
        return lambda o, d: float(np.clip(u0 + d, -1.0, 1.0))

    models = []
    for s in TRAIN:
        t, dy, dt = step_response(
            "glass_furnace", s, _OpenLoop(0, 0, hold, window=WINDOW), 400, 120, 0.05
        )
        models.append(fit_integrating(t, dy, 0.05))
    k, theta = np.median(models, axis=0)
    theta = float(theta) + dt / 2
    print(f"crown: k' {k:.4g} K/s per unit, theta {theta:.3g} s (first hour)")

    def build(tau_c):
        Kc, Ti = simc_integrating(k, theta, tau_c)
        return {"Kp": Kc, "Ki": Kc / Ti, "Kd": 0.0}

    m = pick("glass_furnace", theta, build)
    out = build(m * theta)
    out["model"] = {"k": float(k), "theta": theta, "tau_c": m * theta}
    out["tuning_rule"] = "simc_integrating"
    out["note"] = (
        "Crown-temperature PI by SIMC on the integrating approximation of a "
        "firing step's first hour, from step tests on seeds 100-109 "
        "(scripts/tune_industrial_experts.py). Practice runs an MPC over this "
        "loop; this is the base layer under it."
    )
    return out


def tune_compressor_surge() -> dict:
    """The guarded coordinate search of ``scripts/tune_pid.py``, on the
    training seeds, keeping the held gains (the surge and override lines, the
    override hold and the reset band)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import tune_pid

    from target_gym.compressor_surge.experts import DEFAULT_GAINS

    # Start from the package's documented starting point, not from the gains
    # stored before, which were tuned on seeds 0-23 (scored ones included).
    stored = dict(pid_mod.load_gains())
    pid_mod._gains_cache = {**stored, "compressor_surge": dict(DEFAULT_GAINS)}
    try:
        tuner, _ = tune_pid.TUNERS["compressor_surge"]
        return tuner()
    finally:
        pid_mod._gains_cache = stored


def tune_cement_kiln() -> dict:
    """The burning-zone loop by SIMC from a fuel step (the supervisor's
    setpoint frozen), tight as the inner loop of a cascade; then the rule
    base's two scaling factors on the training seeds."""
    from target_gym.cement_kiln.env import CementKilnParams

    kp = CementKilnParams()
    span = float(kp.fuel_max - kp.fuel_min)

    def hold(a, obs):  # fuel in manual at its average command
        u0 = float(a[0])
        return lambda o, d: float(np.clip(u0 + d, -1.0, 1.0))

    models = []
    for s in TRAIN:
        t, dy, dt = step_response("cement_kiln", s, _OpenLoop(1, 0, hold), 180, 60, 0.1)
        models.append(fit_fopdt(t, dy, 0.1 * span / 2.0))  # per kg/s of fuel
    k, tau, theta = np.median(models, axis=0)
    theta = float(theta) + dt / 2
    print(f"burning zone: k {k:.4g} K per kg/s, tau {tau:.4g} s, theta {theta:.3g} s")
    Kc, Ti = simc_pi(k, tau, theta, theta)
    out = {"Kp_inner": Kc, "Ki_inner": Kc / Ti}
    out["model_inner"] = {
        "k": float(k),
        "tau": float(tau),
        "theta": theta,
        "tau_c": theta,
    }

    costs = {}
    for b in (0.001, 0.002, 0.004):
        for u in (0.25, 0.5, 1.0, 2.0):
            costs[(b, u)] = train_cost(
                "cement_kiln", {**out, "lime_halfwidth": b, "step": u}
            )
            print(
                f"  lime half-width {100 * b:.1f} pp, step {u} K: {costs[(b, u)]:.4g}"
            )
    b, u = min(costs, key=costs.get)
    out.update({"lime_halfwidth": b, "step": u, "T_bz_base": 1765.0})
    out["tuning_rule"] = "simc_inner_grid_scaling"
    out["note"] = (
        "Swain and Subudhi (1996) rule base on free lime and the kiln-speed "
        "trend, moving the burning-zone setpoint, over a fuel PI on the "
        "burning zone (SIMC from a fuel step, tight); kiln speed follows feed. "
        "The two scaling factors are the lowest training cost of a 3 x 4 grid "
        "on seeds 100-109 (scripts/tune_industrial_experts.py)."
    )
    return out


TUNERS = {
    "boiler_drum": tune_boiler_drum,
    "glass_furnace": tune_glass_furnace,
    "compressor_surge": tune_compressor_surge,
    "cement_kiln": tune_cement_kiln,
}


def write(name: str, settings: dict) -> None:
    stored = json.loads(GAINS_PATH.read_text())
    stored[name] = settings
    GAINS_PATH.write_text(json.dumps(stored, indent=2) + "\n")
    pid_mod._gains_cache = None
    print(f"wrote {name} to {GAINS_PATH.relative_to(ROOT)}")


def check(name: str) -> None:
    """The stored expert's hold on the held-out seeds, against the ceiling."""
    spec = REGISTRY[name]
    p = spec.make_test_params()
    eps = [
        E.run_episode(spec, p, spec.make_pid_controller().host(), seed=s)
        for s in HELD_OUT
    ]
    holds = np.array(
        [
            E.hold([ep], E.scored_burn_in(name, p), settle=E.fixed_settle(name))[0]
            for ep in eps
        ]
    )
    c = ceiling(name, p)
    b = float(holds.mean())
    print(
        f"{name}: held-out hold {b:.4g} (median {np.median(holds):.4g}, max "
        f"{holds.max():.4g}), ceiling {c:.4g}, room {np.log10(max(b, c) / c):.2f} "
        f"decades, trips {int(sum(ep.failed.sum() for ep in eps))}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--envs", nargs="+", default=list(TUNERS))
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name in args.envs:
        if name in TUNERS and not (args.check and not args.write):
            settings = TUNERS[name]()
            print(json.dumps(settings, indent=2))
            if args.write:
                write(name, settings)
        if args.check:
            check(name)


if __name__ == "__main__":
    main()
