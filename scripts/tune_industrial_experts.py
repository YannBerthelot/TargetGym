"""Tune the industrial experts by their recipes, from single-run bump tests.

Run with ``uv run python scripts/tune_industrial_experts.py [--envs ...]
[--level 2] [--write] [--check]``. Without ``--write`` it prints each test and the
settings; ``--write`` stores them in ``src/target_gym/data/pid_gains.json``, or
that level's file (``--level``, the sensor noise it tunes on); ``--check`` scores
the stored expert on the held-out seeds, which only report.

The recipes for ``battery``, ``boiler_drum``, ``glass_furnace`` and ``hvac`` are
section 10 of docs/proposals/expert-protocols-process.md, on the machinery of
``tune_process_experts`` (a test starts from no stored gain, one run per training
seed, the fit C5 picks, SIMC with tau_c = theta, then N5 to N7 on valves). The rod
program on ``reactor`` is published, so it has no recipe, and ``wind_turbine`` has
its own script, ``wind_turbine_gains.py``. ``cement_kiln`` is set aside and still
runs on the shipped expert, with the 3 x 4 grid of its old protocol.

Seeds 0-9 are scored and are never run here.
"""

from __future__ import annotations

import argparse
import json
import math
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

sys.path.insert(0, str(ROOT / "scripts"))

from tune_process_experts import manual, read_loop, record, tuned  # noqa: E402

from target_gym import eval as E  # noqa: E402
from target_gym.identification import (  # noqa: E402
    BASELINE,
    DEAD_BAND,
    Instrument,
    Loop,
    Plant,
    rga,
)
from target_gym.provenance import GAIN_LEVELS, gains_path  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402
from target_gym.scoring import ceiling  # noqa: E402

TRAIN = range(100, 110)
HELD_OUT = range(200, 210)
#: The sensor noise a run tunes on and the level it writes, set by ``--level``.
LEVEL = 1.0
#: Valve dead band on a raw action, which spans 2.
RAW_BAND = DEAD_BAND * 2.0
#: A test's hold, and the length ``Plant`` keys its noise on (section 10, R8).
TEST_STEPS = {"battery": 120, "boiler_drum": 400, "glass_furnace": 400, "hvac": 720}
SOURCE = (
    "docs/proposals/expert-protocols-process.md, section 10: single-run bump tests, "
    "seeds 100-109"
)


# ---------------------------------------------------------------------------
# Step tests, fits and rules
# ---------------------------------------------------------------------------


def step_response(name, seed, act, n_settle, n_test, delta):
    """One run (N1): the shipped expert settles, ``act(expert, obs, t, delta) ->
    raw`` holds through the baseline and then steps. The response is observation
    channel ``act.channel`` less the baseline's mean."""
    spec = REGISTRY[name]
    env = spec.make_env()
    n = n_settle + BASELINE + n_test
    p = spec.make_test_params(sensor_noise_scale=LEVEL, max_steps_in_episode=n + 5)
    step = jax.jit(env.step_env)
    expert = spec.make_pid_controller().host()
    expert.reset()
    act.reset()
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, p)
    ys = []
    for t in range(n):
        d = delta if t >= n_settle + BASELINE else 0.0
        a = act(expert, np.asarray(obs), t - n_settle, d)
        obs, state, *_ = step(jax.random.fold_in(key, t), state, jnp.asarray(a), p)
        if t >= n_settle:
            ys.append(float(np.asarray(obs)[act.channel]))
    ys = np.array(ys)
    t = np.arange(1, n_test + 1) * float(p.delta_t)
    return t, ys[BASELINE:] - ys[:BASELINE].mean(), float(p.delta_t)


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


def train_cost(name, settings):
    """Mean cost per step over the training seeds with ``settings`` stored."""
    spec = REGISTRY[name]
    p = spec.make_test_params(sensor_noise_scale=LEVEL)
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


class _Closed:
    """The expert stays in auto and ``d`` lands on output ``index``, whose
    moves it records: nobody holds a drum level open loop (N1)."""

    def __init__(self, channel, index):
        self.channel, self.index = channel, index

    def reset(self):
        self.sent = []

    def __call__(self, expert, obs, t, d):
        a = np.array(expert(obs), float)
        a[self.index] = np.clip(a[self.index] + d, -1.0, 1.0)
        if t >= 0:
            self.sent.append(a[self.index])
        return a


def _params(name, **changes):
    return REGISTRY[name].make_test_params(
        sensor_noise_scale=LEVEL, max_steps_in_episode=TEST_STEPS[name], **changes
    )


def tune_battery() -> dict:
    """A power bump in manual, the signal off (R2), then SIMC and R9's derate."""
    name = "battery"
    spec, p = REGISTRY[name], _params(name, dispatch_noise_std=0.0)
    env = spec.make_env()
    # In manual at the block's level: a raw action is MW on this pack.
    rest = manual(lambda o, xp: xp.reshape(o[4], (1,)))
    plant = Plant(spec, p, [lambda o, s, q: o[3]], [(None, None)], controller=rest)
    inst = Instrument.of(env, p, "power")
    loop = read_loop(plant, "P on inverter", 0, 0, 0.1, 20, inst, room=plant.room(0))
    out = tuned(loop, inst.h)
    return {
        "Kp": out["Kc"],
        "Ki": out["Kc"] / out["Ti"],
        "guard_margin": float(p.power_max * p.regulation_tau / p.energy_nominal),
        "model": {"power": record(loop, out)},
    }


def _lambda_rule(loop):
    """D7's rule on a loop's model, ``(theta, lam) -> (Kc, Ti)``. On a first-order
    loop it is C7's, with its cap on a long lag."""
    k, tau1 = loop["k"], loop["tau1"]
    if loop["integrating"]:
        return lambda th, lam: ((2 * lam + th) / (k * (lam + th) ** 2), 2 * lam + th)
    return lambda th, lam: (tau1 / (k * (lam + th)), min(tau1, 4 * (lam + th)))


def _boiler_pairing(plant, inst_p, inst_l) -> np.ndarray:
    """The RGA of (pressure, level) on (firing, feed), read with no loop closed
    from the moves over a test, since level has no steady gain."""
    G = np.zeros((2, 2))
    for j in range(2):
        for cv, inst in enumerate((inst_p, inst_l)):
            moves = [
                plant.bump(s, j, cv, 0.02, 120, inst) for s in range(plant.n_seeds)
            ]
            G[cv, j] = np.median([b.dy[-20:].mean() / b.step for b in moves])
    return rga(G)


def tune_boiler_drum() -> dict:
    """Pressure, then level, under R12's P loop on level. Lambda (D7) on each."""
    name = "boiler_drum"
    spec, p = REGISTRY[name], _params(name, q_steam_noise_std=0.0)
    env = spec.make_env()
    q_max, Q_max = float(p.q_feed_max), float(p.Q_max)
    per_kg = 2.0 / q_max  # raw per kg/s
    level_gain = q_max / (2.0 * float(p.level_trip))  # R12, 400 kg/s per m
    h_steam = 2.75e6  # J/kg, the expert's feedforward

    def rest(o, xp):
        feed = xp.clip(o[2] + level_gain * (o[5] - o[0]), 0.0, q_max)
        heat = o[2] * (h_steam - float(p.h_feedwater))
        return xp.stack([2 * heat / Q_max - 1, per_kg * feed - 1])

    cvs = [lambda o, s, q: o[1], lambda o, s, q: o[0]]
    plant = Plant(spec, p, cvs, [(None, None)] * 2, controller=manual(rest))
    closed = (Loop(1, 1, level_gain * per_kg, 0),)
    inst_p, inst_l = Instrument.of(env, p, "pressure"), Instrument.of(env, p, "level")

    # Steam follows pressure, so pressure stays first order at every level.
    pressure = read_loop(
        plant, "pressure on firing", 0, 0, 0.02, 120, inst_p, room=1.0, loops=closed
    )
    level = read_loop(
        plant,
        "level on feed",
        1,
        1,
        2.0 * per_kg,
        120,
        inst_l,
        room=plant.room(1),
        model="integrating_lag",
        loops=closed,
    )
    # The feed's gain per kg/s, the lag counted as delay (section 10).
    level = {
        **level,
        "k": level["k"] * per_kg,
        "theta": level["theta"] + level["tau1"],
    }
    # N7's floor: the steam demand's 3 sd, as feed and as firing.
    swing = 3 * float(REGISTRY[name].make_test_params().q_steam_noise_std)
    firing = 2 * swing * (h_steam - float(p.h_feedwater)) / Q_max
    p_out = tuned(pressure, inst_p.h, RAW_BAND, firing, _lambda_rule(pressure))
    l_out = tuned(level, inst_l.h, DEAD_BAND * q_max, swing, _lambda_rule(level))
    print("  RGA, pressure and level on firing and feed:")
    print(np.round(_boiler_pairing(plant, inst_p, inst_l), 3))

    def kept(loop, out):
        exact = {"k": loop["k"], "tau1": loop["tau1"], "theta": out["theta"]}
        return {**record(loop, out), **exact, "lam": out["tau_c"]}

    # The firing PI adds Q_max (Kp e + Ki int e) to the heat, and the action is
    # 2 Q / Q_max - 1, so its gain on the action is 2 Kp.
    return {
        "Kp_pressure": 0.5 * p_out["Kc"],
        "Ki_pressure": 0.5 * p_out["Kc"] / p_out["Ti"],
        "Kd_pressure": 0.0,
        "Tf_pressure": p_out["Tf"],
        "Kp_level": l_out["Kc"],
        "Ki_level": l_out["Kc"] / l_out["Ti"],
        "Tf_level": l_out["Tf"],
        "model_pressure": kept(pressure, p_out),
        "model_level": kept(level, l_out),
    }


def tune_glass_furnace() -> dict:
    """A firing step two steps after an interruption, ramped at the slew, read
    against the crown's own cycle with the datasheet's delay (R13, R14), then a
    PI on the half rule (R15). Held in manual the crown repeats every three
    reversals, 75 minutes, on a far slower drift."""
    from target_gym.glass_furnace.env_jax import FUEL_DEAD_TIME_STEPS, settled_point

    name = "glass_furnace"
    spec = REGISTRY[name]
    nominal = spec.make_test_params()
    p = _params(name, m_pull_noise_std=0.0, lhv_noise_std=0.0, target_T_crown_trim=0.0)
    env = spec.make_env()
    dt = float(p.delta_t)
    slew = 2.0 * float(p.actuator_rate) * dt  # raw per step

    def ramp(u, y, held):
        held = u.copy() if held is None else held
        held = held + np.clip(u - held, -slew, slew)
        return held, held

    rest = manual(lambda o, xp: xp.reshape(settled_point(o[4])[0], (1,)))
    plant = Plant(
        spec,
        p,
        [lambda o, s, q: o[0]],
        [(None, None)],
        controller=rest,
        actuate=ramp,
        applied=lambda u, a: a,
    )
    inst = Instrument.of(env, p, "T_crown_meas")
    room = plant.room(0)
    loop = read_loop(
        plant,
        "crown on firing",
        0,
        0,
        0.05,
        40,
        inst,
        room=room,
        model="integrating_lag",
        at=(3 * round(float(p.reversal_period) / dt), 2),
        delay=FUEL_DEAD_TIME_STEPS * dt,
    )
    # The half rule (section 10, R15).
    loop = {**loop, "theta": loop["theta"] + loop["tau1"] / 2}
    # The pull's 3 sd as its share of the fuel, in raw, capped at the room.
    fuel = 0.5 * float(p.fuel_min + p.fuel_max)
    share = 3 * float(nominal.m_pull_noise_std) / float(p.m_pull)
    span = float(p.fuel_max - p.fuel_min)
    out = tuned(loop, inst.h, RAW_BAND, min(room, 2 * share * fuel / span))
    return {
        "Kp": out["Kc"],
        "Ki": out["Kc"] / out["Ti"],
        "Kd": 0.0,
        "Tf": out["Tf"],
        "model": {
            **record(loop, out),
            "k": loop["k"],
            "theta": out["theta"],
            "tau_c": out["tau_c"],
            "tau2": loop["tau1"],
        },
    }


def tune_hvac() -> dict:
    """The room's PI on a quiet building, on the emitter's five lags (R16). The
    curve itself reads the building's datasheet, in ``experts.py``."""
    from target_gym.hvac.env import total_heat_loss_coefficient

    name = "hvac"
    spec = REGISTRY[name]
    quiet = spec.make_test_params()
    design = float(quiet.setpoint_occupied)
    p = _params(
        name,
        T_out_amplitude=0.0,
        T_out_noise_std=0.0,
        solar_peak=0.0,
        gain_occupied=float(quiet.gain_unoccupied),
        setpoint_setback=design,
        setpoint_occupied_range=(design, design),
    )
    env = spec.make_env()
    H = float(total_heat_loss_coefficient(p))
    # From the balance at the design setpoint, with the one gain left.
    load = H * (design - float(p.T_out_mean)) - float(p.gain_unoccupied * p.A_floor)
    raw = 2.0 * load / float(p.Q_heat_max) - 1.0
    rest = manual(lambda o, xp: xp.full((1,), raw))
    plant = Plant(spec, p, [lambda o, s, q: o[0]], [(None, None)], controller=rest)
    inst = Instrument.of(env, p, "T_air")
    window = math.ceil((5 * float(p.emitter_tau) + inst.h / 2) / inst.h)
    loop = read_loop(
        plant, "T_air on heat", 0, 0, 0.1, window, inst, room=plant.room(0)
    )
    out = tuned(loop, inst.h)
    # The loop's output is 0 to 1 of the heat, and a raw action spans 2.
    return {
        "kp": 0.5 * out["Kc"],
        "Ti": out["Ti"],
        "model": {"T_air": record(loop, out)},
    }


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
    "battery": tune_battery,
    "boiler_drum": tune_boiler_drum,
    "glass_furnace": tune_glass_furnace,
    "hvac": tune_hvac,
    "cement_kiln": tune_cement_kiln,
}


def write(name: str, settings: dict) -> None:
    path = gains_path(LEVEL)
    stored = json.loads(path.read_text()) if path.exists() else {}
    stored[name] = {"tuning_rule": SOURCE, **settings}
    path.write_text(json.dumps(stored, indent=2) + "\n")
    pid_mod._gains_cache = None
    print(f"wrote {name} to {path.relative_to(ROOT)}")


def check(name: str) -> None:
    """The stored expert's hold on the held-out seeds, against the ceiling."""
    spec = REGISTRY[name]
    p = spec.make_test_params(sensor_noise_scale=LEVEL)
    eps = [
        E.run_episode(spec, p, spec.make_pid_controller(LEVEL).host(), seed=s)
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
    global LEVEL
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--envs", nargs="+", default=list(TUNERS))
    parser.add_argument("--level", type=float, default=LEVEL, choices=GAIN_LEVELS)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    LEVEL = args.level
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
