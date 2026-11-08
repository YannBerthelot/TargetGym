"""The reference controllers: one fixed ladder, run on every process, thermal and
energy task (docs/proposals/task-charter.md, rule 3).

Run with ``uv run python scripts/reference_ladder.py [--tasks ...] [--out DIR]``.
For each task it bump-tests the plant on nominal seeds 100 to 109, builds every rung
that applies, runs each on those seeds at sensor noise scale 1, and keeps the cheapest
rung that trips no seed. Nothing here reads a score to set a number: the gains come
from the bump tests, and the rows hold only what the plant's datasheet and balances
say, plus each test's step size and length.

A rung a task doesn't have is reported with the reason from its row.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from target_gym.benchmark import episodes_of, run_policy, task_info  # noqa: E402
from target_gym.experts.ladder import (  # noqa: E402
    Cascade,
    Override,
    Pi,
    Rung,
    best_rung,
    decoupler,
    ladder_controller,
    most_sensitive,
)
from target_gym.experts.pid import Controller  # noqa: E402
from target_gym.identification import (  # noqa: E402
    DEAD_BAND,
    Instrument,
    Loop,
    Plant,
    noise_tune,
    pick_fit,
    single_run,
)
from target_gym.registry import REGISTRY  # noqa: E402

SEEDS = tuple(range(100, 110))
NOISE = 1.0
RAW_BAND = DEAD_BAND * 2.0  # a valve's dead band on an action spanning 2
FREE = (-np.inf, np.inf)


@dataclasses.dataclass(frozen=True)
class Cv:
    """A loop: ``read`` on MV ``mv``, to ``target``, each ``f(obs, xp)``. ``du`` and
    ``window`` size its bump test, on the reading named ``inst``. ``valve`` says the
    MV has a valve's dead band. ``room`` replaces the plant's own limit."""

    name: str
    mv: int
    read: object
    target: object
    inst: str
    du: float
    window: int
    model: str = "fopdt"
    valve: bool = True
    room: float | None = None


@dataclasses.dataclass(frozen=True)
class Inner:
    """A cascade under loop ``loop``: ``read`` rests at ``rest``, or where it settled
    in the test if None. ``du`` and ``window`` size the inner test, ``du_outer`` and ``window_outer`` the outer one, in the inner
    reading's units."""

    loop: int
    name: str
    read: object
    rest: object
    inst: str
    du: float
    window: int
    du_outer: float
    window_outer: int
    model: str = "fopdt"
    valve: bool = True


@dataclasses.dataclass(frozen=True)
class Limit:
    """An override keeping ``read`` past ``limit``. ``slope`` and ``theta`` are the
    datasheet's where a bump can't read it; otherwise ``du`` and ``window`` size one."""

    name: str
    read: object
    limit: float
    floor: bool
    mv: int
    inst: str = ""
    du: float = 0.0
    window: int = 0
    slope: float | None = None
    theta: float | None = None
    seeds: object = None


@dataclasses.dataclass
class Row:
    """What the ladder needs of one task. ``rest`` holds the plant for its tests, on
    the ``test`` parameters. ``points`` are the other operating points the episode
    visits, each a change to ``test``: the ends of its drawn targets and loads. Every
    loop is bump-tested at each and tuned where the plant is most sensitive.
    ``balance`` is rung 2's bias, None without one. ``why`` gives the reason for each
    rung left out, and ``points`` for a task with one operating point. ``rest_at``
    gives the rest at a point's parameters where ``rest`` can't read it."""

    steps: int
    rest: object
    loops: tuple = ()
    test: dict = dataclasses.field(default_factory=dict)
    points: tuple = ()
    rest_at: object = None
    score: dict = dataclasses.field(default_factory=dict)
    balance: object = None
    cascades: tuple = ()
    limits: tuple = ()
    decouple: bool = False
    holds: tuple = ()
    start: object = None
    published: object = None
    why: dict = dataclasses.field(default_factory=dict)


def corners(**ranges) -> tuple:
    """The operating points at the ends of drawn ranges, every combination of ends,
    each range pinned to one end: ``corners(a=(0, 1))`` is ``a=(0, 0)``, ``a=(1, 1)``.
    """
    points = [{}]
    for name, (low, high) in ranges.items():
        points = [{**q, name: (end, end)} for q in points for end in (low, high)]
    return tuple(points)


def manual(hold):
    return Controller(None, 0.0, lambda _p, s, obs, xp=jnp: (hold(obs, xp), s))


def read_loop(
    plant, label, mv, cv, du, window, inst, room, model, loops=(), seeds=None
):
    """One loop's model from single-run bump tests on every seed."""
    seeds = range(plant.n_seeds) if seeds is None else seeds
    read = single_run(
        lambda d: [plant.bump(s, mv, cv, d, window, inst, loops) for s in seeds],
        du,
        room,
        inst,
        model,
        None,
    )
    best, _, _ = pick_fit(read.fits)
    integrating = model != "fopdt"
    out = {
        "k": float(best["k"]),
        "tau": None if integrating else float(best["tau"]),
        "theta": float(
            best["theta"] + inst.half_sample + (best["tau2"] if integrating else 0.0)
        ),
        "band": float(read.nb),
    }
    print(f"  {label}: {out}", flush=True)
    return out


def design(model, inst, db) -> dict:
    """SIMC with ``tc = theta``, with the reading's filter (N5, N6). The noise never
    lowers the gain: ``kc_min`` infinite stops N7."""
    return noise_tune(
        model["k"], model["tau"], model["theta"], model["band"], inst.h, db, math.inf
    )


def pi_of(i, out, dt) -> Pi:
    tf = out["Tf"]
    return Pi(i, out["Kc"], out["Ti"], -math.expm1(-dt / tf) if tf else 1.0)


def wrap(read):
    return lambda o, s, q: read(o, jnp)


def closed_inner(pis, mvs, n_cv, dt):
    """The inner PIs as the law runs them, for a test whose MV is their setpoint."""

    def actuate(u, y, m):
        m = [(y[n_cv + i], 0.0) for i in range(len(mvs))] if m is None else m
        u, out = np.array(u, float), []
        for i, (mv, p) in enumerate(zip(mvs, pis)):
            f, integral = m[i]
            f = f + p.alpha * (y[n_cv + i] - f)
            e = u[mv] - f
            v = p.kc * (e + (integral + e * dt) / p.ti)
            if -1.0 <= v <= 1.0:
                integral += e * dt
            u[mv] = np.clip(v, -1.0, 1.0)
            out.append((f, integral))
        return u, out

    return actuate


def gain_matrix(plant, row, inst, n):
    """Steady gains, CVs by MVs, each read at the end of the longest window."""
    window = max(c.window for c in row.loops)
    G = np.zeros((n, n))
    for i in range(n):
        for j, c in enumerate(row.loops):
            room = plant.room(c.mv)
            gains = []
            for s in range(plant.n_seeds):
                b = plant.bump(s, c.mv, i, min(c.du, room), window, inst)
                tail = b.dy[-max(len(b.dy) // 10, 1) :]
                gains.append(float(np.mean(tail)) / b.step)
            G[i, j] = np.median(gains)
    return G


def build(name: str, row: Row) -> dict:
    """Every rung of ``row``, as ``{rung: Controller}``."""
    spec = REGISTRY[name]
    env = spec.make_env()
    p = spec.make_test_params(
        sensor_noise_scale=NOISE, max_steps_in_episode=row.steps, **row.test
    )
    rungs = {}
    if row.published:
        rungs["published"] = row.published(spec)
    if not row.loops:
        return rungs

    n, nc = len(row.loops), len(row.cascades)
    cvs = [wrap(c.read) for c in row.loops] + [wrap(c.read) for c in row.cascades]
    cvs += [wrap(o.read) for o in row.limits]

    def params_at(change):
        return spec.make_test_params(
            sensor_noise_scale=NOISE,
            max_steps_in_episode=row.steps,
            **{**row.test, **change},
        )

    def rest_of(q):
        return row.rest_at(q) if row.rest_at else row.rest

    def plant_at(q):
        return Plant(
            spec,
            q,
            cvs,
            [(None, None)] * len(cvs),
            controller=manual(rest_of(q)),
            start=row.start,
        )

    at_points = [params_at(change) for change in ({}, *row.points)]
    plants = [plant_at(q) for q in at_points]
    plant = plants[0]
    dt = plant.dt

    def inst_of(tag):
        return Instrument.of(env, p, tag)

    def settled(cv):
        value = float(np.median(plant._y[:, cv]))
        return lambda o, xp: value

    def tune(label, mv, cv, du, window, inst, db, model="fopdt", room=None, on=None):
        """SIMC at every operating point, kept where the plant is most sensitive."""
        designs = []
        for k, at in enumerate(on or plants):
            fit = read_loop(
                at,
                f"{label} at point {k}",
                mv,
                cv,
                du,
                window,
                inst,
                room or at.room(mv),
                model,
                row.holds,
            )
            designs.append(design(fit, inst, db))
        out = most_sensitive(designs)
        print(f"    {label}: {out}", flush=True)
        return out

    def valve(c):
        return RAW_BAND if c.valve else None

    pis = []
    for i, c in enumerate(row.loops):
        out = tune(
            c.name, c.mv, i, c.du, c.window, inst_of(c.inst), valve(c), c.model, c.room
        )
        pis.append(pi_of(i, out, dt))

    m = env.action_space(p).shape[0]
    mix = np.zeros((m, n))
    for i, c in enumerate(row.loops):
        mix[c.mv, i] = 1.0
    const = np.median(plant.u0, axis=0)
    flat = lambda o, xp: xp.asarray(const, dtype=xp.float32)  # noqa: E731

    def read(o, xp):
        return xp.stack([c.read(o, xp) for c in row.loops])

    def target(o, xp):
        return xp.stack([c.target(o, xp) for c in row.loops])

    def make(rung, bias):
        return ladder_controller(rung, read, target, bias, dt)

    loops = tuple(pis)
    rungs["pi"] = make(Rung("pi", loops, mix), flat)
    bias = flat
    if row.balance:
        bias = row.balance
        rungs["feedforward"] = make(Rung("feedforward", loops, mix), bias)

    cascades = ()
    if row.cascades:
        inner = []
        for k, c in enumerate(row.cascades):
            mv = row.loops[c.loop].mv
            out = tune(
                c.name, mv, n + k, c.du, c.window, inst_of(c.inst), valve(c), c.model
            )
            inner.append(pi_of(c.loop, out, dt))
        outer = [
            Plant(
                spec,
                q,
                cvs,
                [(None, None)] * len(cvs),
                controller=manual(rest_of(q)),
                start=_inner_start(at, row, n),
                actuate=closed_inner(
                    inner, [row.loops[c.loop].mv for c in row.cascades], n, dt
                ),
                mv_range=FREE,
            )
            for at, q in zip(plants, at_points)
        ]
        tuned = list(pis)
        for k, c in enumerate(row.cascades):
            base = row.loops[c.loop]
            label = f"{base.name} over {c.name}"
            room = 1.0 / abs(inner[k].kc)  # what the inner loop's output can answer
            db = RAW_BAND / abs(inner[k].kc)
            out = tune(
                label,
                base.mv,
                c.loop,
                c.du_outer,
                c.window_outer,
                inst_of(base.inst),
                db,
                base.model,
                room,
                outer,
            )
            tuned[c.loop] = pi_of(c.loop, out, dt)
        cascades = tuple(
            Cascade(c.read, c.rest or settled(n + k), q)
            for k, (c, q) in enumerate(zip(row.cascades, inner))
        )
        loops = tuple(tuned)
        rungs["cascade"] = make(Rung("cascade", loops, mix, cascades), bias)

    if row.decouple:
        G = gain_matrix(plant, row, inst_of(row.loops[0].inst), n)
        print(f"    gains: {np.round(G, 4).tolist()}", flush=True)
        D = np.zeros((m, n))
        D[np.ix_([c.mv for c in row.loops], range(n))] = decoupler(G, range(n))
        rungs["decoupler"] = make(Rung("decoupler", tuple(pis), D), bias)

    if row.limits:
        over = []
        for k, o in enumerate(row.limits):
            if o.slope is not None:
                model = {"k": o.slope, "tau": None, "theta": o.theta, "band": 0.0}
                out = design(model, Instrument(dt), None)
                print(f"    {o.name}: {out}", flush=True)
            else:
                designs = []
                for j, at in enumerate(plants):
                    model = read_loop(
                        at,
                        f"{o.name} at point {j}",
                        o.mv,
                        n + nc + k,
                        o.du,
                        o.window,
                        inst_of(o.inst),
                        at.room(o.mv),
                        "fopdt",
                        row.holds,
                        o.seeds,
                    )
                    designs.append((design(model, Instrument(dt), None), model))
                out, model = min(designs, key=lambda d: abs(d[0]["Kc"]))
                print(f"    {o.name}: {out}", flush=True)
            over.append(
                Override(
                    o.read,
                    o.limit,
                    o.floor,
                    o.mv,
                    float(np.sign(model["k"])),
                    out["Kc"],
                    out["Ti"],
                )
            )
        rungs["override"] = make(
            Rung("override", loops, mix, cascades, tuple(over)), bias
        )
    return rungs


def _inner_start(plant, row, n):
    """The test's MVs at rest: each cascade's stands at where its reading settled."""

    def start(a, m):
        u = plant.u0.copy()
        for k, c in enumerate(row.cascades):
            u[:, row.loops[c.loop].mv] = plant._y[:, n + k]
        return u

    return start


class Reference:
    """A pure controller as a batched ``benchmark.Policy``."""

    oracle = False

    def __init__(self, name, controller):
        self.name, self.c = name, controller
        self._step = jax.jit(
            jax.vmap(lambda m, o: controller.step(controller.params, m, o, jnp))
        )

    def reset(self, key, obs, task):
        return jax.tree.map(
            lambda x: jnp.repeat(jnp.asarray(x)[None], task.n_episodes, 0),
            self.c.init_state,
        )

    def act(self, carry, obs, t):
        action, carry = self._step(carry, jnp.asarray(obs))
        return np.asarray(action, float), carry


def run_rung(name, row, label, controller) -> dict:
    """``controller`` on the ladder's seeds: per-seed costs and tripped seeds."""
    spec = REGISTRY[name]
    env = spec.make_env()
    params = spec.make_test_params(sensor_noise_scale=NOISE, **row.score)
    task = task_info(name, env, params, len(SEEDS))
    episodes = episodes_of(
        run_policy(Reference(label, controller), env, params, task, seeds=SEEDS)
    )
    return {
        "costs": [float(e.cost.sum()) for e in episodes],
        "trips": int(sum(bool(np.any(e.failed)) for e in episodes)),
    }


def climb(name: str) -> dict:
    """The ladder on ``name``: each rung's result and the one kept."""
    print(name, flush=True)
    row = ROWS[name]()
    results = {}
    for label, controller in build(name, row).items():
        results[label] = run_rung(name, row, label, controller)
        r = results[label]
        print(f"  {label}: median {np.median(r['costs']):.4g}, {r['trips']} trips")
    order = [k for k in _ORDER if k in results]
    kept = best_rung({k: results[k] for k in order})
    print(f"  kept: {kept}", flush=True)
    return {"rungs": results, "kept": kept, "left_out": row.why}


_ORDER = ("pi", "feedforward", "cascade", "decoupler", "override", "published")


# ---------------------------------------------------------------------------
# The rows. Each holds what the plant's datasheet and balances say, and nothing
# read from a score.
# ---------------------------------------------------------------------------


def row_first_order() -> Row:
    def half_target(o, xp):
        return xp.reshape(o[1] / 2.0, (1,))  # the plant's unit gain per raw of 2

    x = Cv("x on u", 0, lambda o, xp: o[0], lambda o, xp: o[1], "x", 0.1, 60)
    return Row(
        150,
        half_target,
        (x,),
        balance=half_target,
        points=corners(target_x_range=(0.5, 1.5)),
        why={"cascade": "one state", "decoupler": "one loop", "override": "no limit"},
    )


def row_ph_neutralization() -> Row:
    from target_gym.pc_gym.ph_neutralization.experts import (
        TitrationCurve,
        reagent_demand,
    )

    spec = REGISTRY["ph_neutralization"]
    nominal = spec.make_test_params()
    test = {"q2_nominal": float(nominal.q2_min), "q2_noise_std": 0.0}
    curve = TitrationCurve.of(nominal)
    lab = TitrationCurve.of(spec.make_test_params(**test))
    low, span = float(nominal.q3_min), float(nominal.q3_max - nominal.q3_min)
    top = 2 * float(nominal.q3_max)

    def raw(q3, xp):
        return xp.reshape(2 * (q3 - low) / span - 1, (1,))

    # The loop runs in reagent demand, the titration curve's linearisation.
    demand = Cv(
        "demand on q3",
        0,
        lambda o, xp: xp.clip(reagent_demand(o[0], curve, xp), 0.0, top),
        lambda o, xp: reagent_demand(o[2], curve, xp),
        "pH",
        0.1 * 2 / span,
        120,
    )
    return Row(
        300,
        lambda o, xp: raw(reagent_demand(o[2], lab, xp), xp),
        (demand,),
        test=test,
        points=corners(target_pH_range=tuple(nominal.target_pH_range)),
        balance=lambda o, xp: raw(reagent_demand(o[2], curve, xp), xp),
        why={
            "cascade": "pH is the only reading",
            "decoupler": "one loop",
            "override": "no reachable limit",
        },
    )


def row_four_tank() -> Row:
    from target_gym.pc_gym.four_tank.experts import (
        four_tank_decoupled_pi,
        steady_voltages,
    )

    p = REGISTRY["four_tank"].make_test_params()
    prm = four_tank_decoupled_pi(1.0, 1.0, 1.0, 1.0, np.eye(2), p).params
    v_max = float(p.v_max)
    per_volt = 2 / (v_max - float(p.v_min))

    def balance(o, xp):
        v = steady_voltages(o[4], o[5], prm, xp)
        return xp.stack([per_volt * v[0] - 1, per_volt * v[1] - 1])

    def upper(i, gamma, k, a):
        """The upper tank's level at rest on its pump, ``a sqrt(2 g h) = (1 - gamma)
        k v``."""

        def rest(o, xp):
            v = steady_voltages(o[4], o[5], prm, xp)[i]
            return ((1 - gamma) * k * v / a) ** 2 / (2 * p.g)

        return rest

    du = 0.3 * per_volt
    h1 = Cv("h1 on v2", 1, lambda o, xp: o[0], lambda o, xp: o[4], "h1", du, 1500)
    h1 = dataclasses.replace(h1, valve=False)
    h2 = Cv("h2 on v1", 0, lambda o, xp: o[1], lambda o, xp: o[5], "h1", du, 1500)
    h2 = dataclasses.replace(h2, valve=False)
    return Row(
        1200,
        balance,
        (h1, h2),
        test={"load_scale": 0.0},
        points=corners(
            target_h1_range=tuple(p.target_h1_range),
            target_h2_range=tuple(p.target_h2_range),
        ),
        balance=balance,
        cascades=(
            Inner(
                0,
                "h3 on v2",
                lambda o, xp: o[2],
                upper(1, p.gamma2, p.k2, p.a3),
                "h1",
                du,
                600,
                0.005,
                1500,
                valve=False,
            ),
            Inner(
                1,
                "h4 on v1",
                lambda o, xp: o[3],
                upper(0, p.gamma1, p.k1, p.a4),
                "h1",
                du,
                600,
                0.005,
                1500,
                valve=False,
            ),
        ),
        decouple=True,
        why={
            "override": "low trip has 30 mm of margin at worst load",
            "points": "only the target corners: the lower tanks' loads are steps the "
            "episode switches during a run, and no parameter holds one, so the tests "
            "run at no load",
        },
    )


def _cstr_rest(p, C_a, xp):
    """T and raw coolant at which ``C_a`` is steady, from the two balances."""
    qV = p.q / p.V
    rate = qV * (p.Caf - C_a)
    T = -p.EA_over_R / xp.log(rate / (p.k0 * C_a))
    heat = qV * (p.Ti - T) + (-p.deltaHr) * rate / (p.rho * p.C)
    T_c = T - heat * p.rho * p.C * p.V / p.UA
    return T, 2 * (T_c - p.T_c_min) / (p.T_c_max - p.T_c_min) - 1


def row_cstr() -> Row:
    p = REGISTRY["cstr"].make_test_params()

    def balance(o, xp):
        return xp.reshape(_cstr_rest(p, o[2], xp)[1], (1,))

    C_a = Cv(
        "C_a on coolant", 0, lambda o, xp: o[0], lambda o, xp: o[2], "C_a", 0.1, 80
    )
    T = Inner(
        0,
        "T on coolant",
        lambda o, xp: o[1],
        lambda o, xp: _cstr_rest(p, o[2], xp)[0],
        "T",
        0.1,
        60,
        0.5,
        80,
    )
    return Row(
        150,
        balance,
        (C_a,),
        test={"load_scale": 0.0},
        # The feed's drawn loads, held: Ti and Caf at each end of their steps.
        points=tuple(
            {**q, "Ti": float(p.Ti) + dT, "Caf": float(p.Caf) + dC}
            for q in corners(target_CA_range=tuple(p.target_CA_range))
            for dT in (-float(p.Ti_load), float(p.Ti_load))
            for dC in (-float(p.Caf_load), float(p.Caf_load))
        ),
        rest_at=lambda q: lambda o, xp: xp.reshape(_cstr_rest(q, o[2], xp)[1], (1,)),
        balance=balance,
        cascades=(T,),
        why={"decoupler": "one loop", "override": "no reachable limit"},
    )


def row_distillation() -> Row:
    p = REGISTRY["distillation"].make_test_params()
    L0, V0 = 2.70629, 3.20629  # the column's rest, from its tray equations
    per_L, per_V = 2 / float(p.L_max - p.L_min), 2 / float(p.V_max - p.V_min)
    raw = np.array(
        [per_L * (L0 - float(p.L_min)) - 1, per_V * (V0 - float(p.V_min)) - 1]
    )

    def rest(o, xp):
        return xp.asarray(raw, dtype=xp.float32)

    yD = Cv("yD on L", 0, lambda o, xp: o[0], lambda o, xp: o[4], "x", 0.05 * 2, 600)
    xB = Cv("xB on V", 1, lambda o, xp: o[1], lambda o, xp: o[5], "x", 0.05 * 2, 600)
    tray = Inner(
        1,
        "T_15 on V",
        lambda o, xp: o[6],
        None,
        "",
        0.005 * per_V,
        20,
        0.002,
        600,
        model="integrating_lag",
    )
    return Row(
        600,
        rest,
        (yD, xB),
        test={"zF_noise_std": 0.0},
        cascades=(tray,),
        decouple=True,
        why={
            "feedforward": "no closed-form balance: L and V need the tray model, and "
            "no feed reading exists",
            "override": "the trip reads through a 5 min analyser",
            "points": "the rest at another purity needs the tray model solved, which "
            "the row doesn't have: the column is tested at its one rest",
        },
    )


def row_boiler_drum() -> Row:
    p = REGISTRY["boiler_drum"].make_test_params()
    q_max, Q_max = float(p.q_feed_max), float(p.Q_max)
    per_kg = 2.0 / q_max
    level_gain = q_max / (2.0 * float(p.level_trip))
    h = 2.75e6 - float(p.h_feedwater)

    def balance(o, xp):
        return xp.stack([2 * o[2] * h / Q_max - 1, per_kg * o[2] - 1])

    def rest(o, xp):
        feed = xp.clip(o[2] + level_gain * (o[5] - o[0]), 0.0, q_max)
        return xp.stack([2 * o[2] * h / Q_max - 1, per_kg * feed - 1])

    pressure = Cv(
        "pressure on firing",
        0,
        lambda o, xp: o[1],
        lambda o, xp: o[6],
        "pressure",
        0.02,
        120,
        room=1.0,
    )
    level = Cv(
        "level on feed",
        1,
        lambda o, xp: o[0],
        lambda o, xp: o[5],
        "level",
        2.0 * per_kg,
        120,
        model="integrating_lag",
    )
    return Row(
        400,
        rest,
        (pressure, level),
        test={"q_steam_noise_std": 0.0},
        # The episode draws its steam load from a normal: its ends taken at 2 sigma,
        # 95 % of draws (ours). The rest follows the steam reading.
        points=tuple(
            {"q_steam_nominal": float(p.q_steam_nominal) + d}
            for d in (-2 * float(p.q_steam_noise_std), 2 * float(p.q_steam_noise_std))
        ),
        balance=balance,
        holds=(Loop(1, 1, level_gain * per_kg, 0),),
        why={
            "cascade": "riser and inventory are hidden; only actuator positions read",
            "decoupler": "RGA is 1.03 on the diagonal",
            "override": "no limit a select could hold: the MVs are spent on the CVs",
        },
    )


def row_hvac() -> Row:
    from target_gym.hvac.env import total_heat_loss_coefficient

    spec = REGISTRY["hvac"]
    quiet = spec.make_test_params()
    design_T = float(quiet.setpoint_occupied)
    test = {
        "T_out_amplitude": 0.0,
        "T_out_noise_std": 0.0,
        "solar_peak": 0.0,
        "gain_occupied": float(quiet.gain_unoccupied),
        "setpoint_setback": design_T,
        "setpoint_occupied_range": (design_T, design_T),
    }
    tp = spec.make_test_params(**test)
    H, Q_max = float(total_heat_loss_coefficient(tp)), float(quiet.Q_heat_max)
    gains = float(quiet.gain_unoccupied * quiet.A_floor)
    solar = float(quiet.solar_peak * quiet.A_window * quiet.g_window)

    def load(o, xp):
        return H * (o[6] - o[1]) - gains - o[3] * solar

    def balance(o, xp):
        return xp.reshape(xp.clip(2 * load(o, xp) / Q_max - 1, -1.0, 1.0), (1,))

    def raw_at(T):
        return 2 * (H * (T - float(tp.T_out_mean)) - gains) / Q_max - 1

    def rest_at(q):
        raw = raw_at(float(q.setpoint_occupied_range[0]))
        return lambda o, xp: xp.full((1,), raw)

    ends = tuple(quiet.setpoint_occupied_range)
    points = tuple(
        {"setpoint_occupied_range": (T, T), "setpoint_setback": T} for T in ends
    )
    raw = raw_at(design_T)
    window = math.ceil((5 * float(tp.emitter_tau) + 450.0) / 900.0)
    air = Cv(
        "T_air on heat",
        0,
        lambda o, xp: o[0],
        lambda o, xp: o[6],
        "T_air",
        0.1,
        window,
        valve=False,
    )
    heat = Inner(
        0,
        "emitter on heat",
        lambda o, xp: o[2],
        lambda o, xp: 100.0 * load(o, xp) / Q_max,
        "Q_emitter",
        0.1,
        window,
        5.0,
        window,
        valve=False,
    )

    def published(s):
        return s.make_pid_controller()

    return Row(
        720,
        lambda o, xp: xp.full((1,), raw),
        (air,),
        test=test,
        points=points,
        rest_at=rest_at,
        balance=balance,
        cascades=(heat,),
        published=published,
        why={"decoupler": "one loop", "override": "the comfort band is not a trip"},
    )


def row_battery() -> Row:
    p = REGISTRY["battery"].make_test_params()
    power = float(p.power_max)
    meter = Cv(
        "P on inverter",
        0,
        lambda o, xp: o[3],
        lambda o, xp: o[4],
        "power",
        0.1,
        20,
        valve=False,
    )

    def rest(o, xp):
        return xp.reshape(o[4], (1,))

    def balance(o, xp):
        return xp.reshape((o[4] * 1e6 + float(p.aux_load)) / power, (1,))

    # Discharging lowers the charge: d soc / dt = -P / E, per raw of the inverter.
    slope = -power / float(p.energy_nominal)
    soc = dict(read=lambda o, xp: o[0], mv=0, slope=slope, theta=float(p.delta_t))
    low, high = 0.10, 0.90  # 5 % inside the pack's trips, MESA-1's operators' margin
    return Row(
        120,
        rest,
        (meter,),
        test={"dispatch_noise_std": 0.0},
        balance=balance,
        limits=(
            Limit("charge floor", limit=low, floor=True, **soc),
            Limit("charge ceiling", limit=high, floor=False, **soc),
        ),
        why={
            "cascade": "meter power is algebraic on the inverter",
            "decoupler": "one loop",
            "points": "meter power is the inverter's, scaled: one gain at any dispatch",
        },
    )


def row_wind_turbine() -> Row:
    from target_gym.energy.wind_turbine.env import balance_pitch

    spec = REGISTRY["wind_turbine"]
    p = spec.make_test_params()
    v = 0.5 * sum(p.v_mean_range)
    test = {"turbulence_std": 0.0, "v_mean_range": (v, v)}
    rpm = float(p.omega_rated_rpm)
    rated = rpm * 2.0 * np.pi / 60.0
    tau_max, power = float(p.torque_max), float(p.target_power_range[1])
    pitch = float(balance_pitch(rated, v, power / float(p.eta_gen), p))
    pitch_max = float(p.pitch_max)
    raw_pitch = 2 * pitch / pitch_max - 1

    def torque(o, xp):
        omega = o[0] * 2 * np.pi / 60
        tau = o[4] * 1e6 / (float(p.eta_gen) * float(p.N_gear) * omega)
        return 2 * xp.clip(tau, 0.0, tau_max) / tau_max - 1

    def balance(o, xp):
        return xp.stack([xp.asarray(raw_pitch, dtype=xp.float32), torque(o, xp)])

    def rest_at(q):
        v, P = float(q.v_mean_range[0]), float(q.target_power_range[1])
        pitch = float(balance_pitch(rated, v, P / float(q.eta_gen), q))
        held = 2 * pitch / pitch_max - 1

        def rest(o, xp):
            return xp.stack([xp.asarray(held, dtype=xp.float32), torque(o, xp)])

        return rest

    speed = Cv(
        "speed on pitch",
        0,
        lambda o, xp: o[0],
        lambda o, xp: xp.asarray(rpm, dtype=xp.float32),
        "omega",
        0.02,
        80,
        valve=False,
    )
    grid = Cv(
        "power on torque",
        1,
        lambda o, xp: o[3],
        lambda o, xp: o[4],
        "power",
        0.02,
        40,
        valve=False,
    )

    def published(s):
        return s.make_pid_controller()

    return Row(
        100,
        balance,
        (speed, grid),
        test=test,
        points=corners(
            v_mean_range=tuple(p.v_mean_range),
            target_power_range=tuple(p.target_power_range),
        ),
        rest_at=rest_at,
        balance=balance,
        published=published,
        why={
            "cascade": "the actuators are faster than a step",
            "decoupler": "the coupling moves with wind, 0.07 to 0.82 on the diagonal",
            "override": "the MVs are spent on the CVs",
        },
    )


def row_compressor_surge() -> Row:
    from target_gym.compressor_surge import experts as X

    spec = REGISTRY["compressor_surge"]
    p = spec.make_test_params()
    line = 0.10
    env = spec.make_env()
    full = spec.make_test_params(sensor_noise_scale=NOISE, max_steps_in_episode=1200)
    starts = [env.reset_env(jax.random.PRNGKey(s), full)[1] for s in SEEDS]
    rests = [
        X.settled_point(
            float(s.setpoint_levels[-1]),
            float(s.demand_levels[-1]),
            {"surge_line": line},
            full,
        )
        for s in starts
    ]
    held = np.stack([X.raw_from_commands(r[2], r[4], full) for r in rests])
    reset = np.asarray(X.raw_from_commands(p.N_reset, p.x_reset, p))
    open_seeds = [i for i, r in enumerate(rests) if r[4] > 0.02]
    span = float(p.N_max - p.N_min)
    du = 0.05 * span * 2 / span
    pressure = Cv(
        "dp on speed",
        0,
        lambda o, xp: o[0],
        lambda o, xp: o[6],
        "",
        du,
        100,
        valve=False,
    )
    margin = Limit(
        "margin on recycle",
        lambda o, xp: X.surge_margin_from_obs(o, p),
        line,
        True,
        1,
        "",
        0.1,
        100,
        seeds=open_seeds,
    )
    published = lambda s: s.make_pid_controller()  # noqa: E731
    return Row(
        1200,
        lambda o, xp: xp.asarray(reset, dtype=xp.float32),
        (pressure,),
        limits=(margin,),
        start=lambda a, m: held,
        published=published,
        why={
            "feedforward": "the consumer's opening is unread and the map's speed needs it",
            "cascade": "the drive is the slowest element on the way to pressure",
            "decoupler": "the second MV serves a limit, not a target",
            "points": "each seed starts held at its own drawn setpoint and demand, "
            "inside the ranges, not at their ends: the start isn't a parameter to pin",
        },
    )


def _lee_cascade(p):
    """The inner PID by Lee, Lee and Park (2000), lambda = theta, on the unstable
    pole at the band's fastest end, under an outer SIMC PI. From plant numbers only.
    The form is derived, not copied: the paper itself was not read."""
    from target_gym.pc_gym.unstable_cstr.env import steady_coolant, steady_temperature

    def poles(L):
        T = steady_temperature(L, p, np)
        k = (p.q / p.V) * (p.Caf - L) / L
        kT = k * p.EA_over_R / T**2
        A, b, q = -p.deltaHr / (p.rho * p.C), p.UA / (p.rho * p.C * p.V), p.q / p.V
        J = np.array([[-q - k, -kT * L], [A * k, -q - b + A * kT * L]])
        return np.linalg.eigvals(J).real.max(), k, kT, b

    band = np.linspace(*p.target_CA_range, 21)
    Ld = band[int(np.argmax([poles(L)[0] for L in band]))]
    pole, k, kT, b = poles(Ld)
    well = 10.0 / 60.0
    t2, th = well + p.tau_j / 2, p.tau_j / 2 + p.actuator_lag + p.delta_t / 2
    lam = th
    tau, K = 1 / pole, b / pole
    beta = tau * ((lam / tau + 1) ** 2 * np.exp(th / tau) - 1)
    d = [2 * lam - beta + th, lam**2 - th**2 / 2 + beta * th]
    d.append(th**3 / 6 - beta * th**2 / 2)
    n = [-1 / K, (tau - beta) / K, tau * beta / K]
    f0 = n[0] / d[0]
    f1 = (n[1] - f0 * d[1]) / d[0]
    f2 = (n[2] - f0 * d[2] - f1 * d[1]) / d[0]
    Ki, Kp, Kd = f0, f1 + f0 * t2, f2 + f1 * t2
    tho = 2 * lam + th
    t1 = 1 / (p.q / p.V + k)
    ko = kT * Ld * t1
    Kco, Tio = t1 / (ko * 2 * tho), min(t1, 8 * tho)
    a = -np.expm1(-p.delta_t / beta)
    b_f = -np.expm1(-p.delta_t * 10 * Kp / Kd)
    span = float(p.T_c_max - p.T_c_min)
    low = float(p.T_c_min)
    dt = float(p.delta_t)

    def step(_, m, obs, xp=jnp):
        Ca, T, L = obs[0], obs[1], obs[3]
        e = L - Ca
        io, ii, sp, f, started = m
        first = started > 0
        sp = xp.where(first, sp, T)
        f_new = xp.where(first, f + b_f * (T - f), T)
        rate = (f_new - xp.where(first, f, T)) / dt
        raw = steady_temperature(L, p, xp) - Kco * (e + io / Tio)
        sp = sp + a * (raw - sp)
        ei = sp - T
        coolant = steady_coolant(L, 0.0, p, xp) + Kp * ei + ii - Kd * rate
        held = xp.clip(coolant, low, low + span)
        free = held == coolant
        io = xp.where(free, io + e * dt, io)
        ii = xp.where(free, ii + Ki * ei * dt, ii)
        action = xp.reshape(2 * (held - low) / span - 1, (1,))
        return action, (io, ii, sp, f_new, xp.ones_like(started))

    zero = (0.0, 0.0, 0.0, 0.0, 0.0)
    return Controller(None, tuple(np.float32(x) for x in zero), step)


def row_unstable_cstr() -> Row:
    band = {"target_CA_range": (0.50, 0.70)}
    p = REGISTRY["unstable_cstr"].make_test_params(**band)
    return Row(
        1200,
        None,
        score=band,
        published=lambda s: _lee_cascade(p),
        why={
            "pi": "open-loop unstable: no bump test without an inner loop",
            "feedforward": "as pi",
            "cascade": "the published rule is the cascade",
            "decoupler": "one MV",
            "override": "the published rule holds the trip on its nominal seeds, or fails",
        },
    )


ROWS = {
    "first_order": row_first_order,
    "ph_neutralization": row_ph_neutralization,
    "four_tank": row_four_tank,
    "cstr": row_cstr,
    "distillation": row_distillation,
    "boiler_drum": row_boiler_drum,
    "hvac": row_hvac,
    "battery": row_battery,
    "wind_turbine": row_wind_turbine,
    "compressor_surge": row_compressor_surge,
    "unstable_cstr": row_unstable_cstr,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", nargs="*", default=list(ROWS))
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()
    for name in args.tasks:
        result = climb(name)
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{name}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
