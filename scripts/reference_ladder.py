"""The reference controllers: one fixed ladder, run on every process, thermal and
energy task (docs/proposals/task-charter.md, J1).

Run with ``uv run python scripts/reference_ladder.py [--tasks ...] [--out DIR]``.
For each task it bump-tests the plant on nominal seeds 100 to 109, builds every rung
that applies, runs each on those seeds at sensor noise scale 1 over the task's scenario
suite, and keeps the cheapest rung that trips no seed. A rung's cost on a seed is the
task's ``J`` (charter J4): each scenario's mean cost per step, weighted by its share of
operating time. Nothing here reads a score to set a number: the gains come from the bump
tests, and the rows hold only what the plant's datasheet and balances say, plus each
test's step size and length.

Each task's row lives in its own file, ``scripts/ladder_rows/<task>.py``, defining
``row()``. A loop on a zone (``Cv.zone``) holds a setpoint ``k = 2`` standard deviations
inside the zone's cheap edge, never past its centre (``ladder.zone_setpoint``). The
deviation is the controlled variable's, with the pi rung holding it on the bump-test
seeds in the scored normal day, over the second half of each episode: first at the
zone's centre, then at each placed setpoint until the setpoint stops moving, since on
a nonlinear plant the spread depends on where the loop holds.
Bumps are read against the same seed's unstepped run (``Plant(paired=True)``), so a
slow walk or drift the two runs share leaves the response.

A rung a task doesn't have is reported with the reason from its row.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
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

from target_gym import scenarios  # noqa: E402
from target_gym.benchmark import (  # noqa: E402
    episodes_of,
    run_policy,
    shipped_policy,
    task_info,
)
from target_gym.experts.ladder import (  # noqa: E402
    Cascade,
    Override,
    Pi,
    Rung,
    best_rung,
    cv_sigma,
    decoupler,
    ladder_controller,
    most_sensitive,
    null_verdict,
    zone_setpoint,
)
from target_gym.experts.pid import Controller  # noqa: E402
from target_gym.identification import (  # noqa: E402
    DEAD_BAND,
    Instrument,
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
    MV has a valve's dead band. ``room`` replaces the plant's own limit. ``zone`` is
    ``(low, high, cheap)`` for a loop kept inside a zone: ``target`` is then unused,
    and the loop holds the setpoint the module docstring describes."""

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
    zone: tuple | None = None


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
    gives the rest at a point's parameters where ``rest`` can't read it. ``build``
    fills ``placed``: each zone loop's measured deviation and setpoint."""

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
    placed: dict = dataclasses.field(default_factory=dict)


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


def build(name: str, row: Row, spec=None) -> dict:
    """Every rung of ``row``, as ``{rung: Controller}``."""
    spec = spec or REGISTRY[name]
    env = spec.make_env()
    p = spec.make_test_params(**_test_params(row))
    rungs = {}
    if row.published:
        rungs["published"] = row.published(spec)
    if not row.loops:
        return rungs

    n, nc = len(row.loops), len(row.cascades)
    cvs = [wrap(c.read) for c in row.loops] + [wrap(c.read) for c in row.cascades]
    cvs += [wrap(o.read) for o in row.limits]

    def params_at(change):
        return spec.make_test_params(**{**_test_params(row), **change})

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
            paired=True,
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

    def stacked(targets):
        return lambda o, xp: xp.stack([t(o, xp) for t in targets])

    targets = [c.target for c in row.loops]
    zoned = [i for i, c in enumerate(row.loops) if c.zone]
    if zoned:
        centres = list(targets)
        for i in zoned:
            low, high, _ = row.loops[i].zone
            centres[i] = _constant(0.5 * (low + high))
        for i in zoned:
            c = row.loops[i]
            setpoint, at = None, list(centres)
            # The spread is the controlled variable's where its loop holds, which on
            # a nonlinear plant moves with the setpoint: start at the centre, then
            # re-read at each placed setpoint, with the row's other loops following
            # it (``row.placed``), until it moves less than PLACE_TOLERANCE sigma.
            for _ in range(PLACE_ROUNDS):
                probe = ladder_controller(
                    Rung("pi", tuple(pis), mix), read, stacked(at), flat, dt
                )
                held = cv_traces(name, row, probe, i, spec)
                sigma = cv_sigma([t[len(t) // 2 :] for t in held])
                last, setpoint = setpoint, zone_setpoint(*c.zone[:2], sigma, c.zone[2])
                row.placed[c.name] = (sigma, setpoint)
                print(
                    f"    {c.name}: sigma {sigma:.4g}, setpoint {setpoint:.4g}",
                    flush=True,
                )
                if last is not None and abs(setpoint - last) <= PLACE_TOLERANCE * sigma:
                    break
                at[i] = _constant(setpoint)
            else:
                print(f"    {c.name}: placement still moving after {PLACE_ROUNDS}")
            targets[i] = _constant(setpoint)
    target = stacked(targets)

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
                paired=True,
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


def _test_params(row) -> dict:
    """The bump tests' parameters: noise scale 1 and the row's episode, then the
    row's own changes, which may quiet a disturbance for clean bumps."""
    return {"sensor_noise_scale": NOISE, "max_steps_in_episode": row.steps, **row.test}


def _constant(value):
    value = float(value)
    return lambda o, xp: xp.asarray(value, dtype=xp.float32)


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


class Recorder(Reference):
    """A ``Reference`` that also keeps ``read(obs)`` for every episode and step."""

    def __init__(self, name, controller, read):
        super().__init__(name, controller)
        self.read, self.values = read, []

    def act(self, carry, obs, t):
        rows = np.asarray(obs)
        self.values.append([float(self.read(o, np)) for o in rows])
        return super().act(carry, obs, t)


def cv_traces(name, row, controller, i, spec=None) -> list:
    """Loop ``i``'s reading under ``controller``, one array per bump-test seed, on
    the scored normal day: the suite's first scenario at the scored parameters, so
    a disturbance a row quiets for its bump tests still moves it."""
    spec = spec or REGISTRY[name]
    env = spec.make_env()
    normal = scenarios.suite(spec)[0]
    params = scenarios.params_for(
        spec, normal, **{"sensor_noise_scale": NOISE, **row.score}
    )
    task = task_info(name, env, params, len(SEEDS))
    policy = Recorder(row.loops[i].name, controller, row.loops[i].read)
    rollout = run_policy(policy, env, params, task, seeds=SEEDS)
    values = np.asarray(policy.values)  # (steps, seeds)
    lengths = np.asarray(rollout["length"], int)
    return [values[:L, k] for k, L in enumerate(lengths)]


def run_rung(name, row, label, controller, spec=None) -> dict:
    """``controller`` on the ladder's seeds over the task's scenarios: each seed's
    ``J``, the seeds that trip in any scenario, and each scenario's own costs."""
    return run_suite(name, row, Reference(label, controller), spec)


def run_suite(name, row, policy, spec=None) -> dict:
    """``run_rung`` for any ``benchmark.Policy``."""
    spec = spec or REGISTRY[name]
    env = spec.make_env()
    weights = scenarios.weights(spec)
    per = {}
    for s in scenarios.suite(spec):
        params = scenarios.params_for(spec, s, sensor_noise_scale=NOISE, **row.score)
        task = task_info(name, env, params, len(SEEDS))
        if hasattr(policy, "bind"):  # a null that reads each episode's reset
            policy.bind(env, params, SEEDS)
        episodes = episodes_of(run_policy(policy, env, params, task, seeds=SEEDS))
        per[s.name] = (
            np.array([float(e.cost.mean()) for e in episodes]),
            np.array([bool(np.any(e.failed)) for e in episodes]),
        )
    costs = sum(weights[n] * c for n, (c, _) in per.items())
    tripped = np.any([t for _, t in per.values()], axis=0)
    return {
        "costs": [float(c) for c in costs],
        "trips": int(tripped.sum()),
        "scenarios": {
            n: {"costs": [float(x) for x in c], "trips": int(t.sum())}
            for n, (c, t) in per.items()
        },
    }


def climb(name: str, nulls: bool = False) -> dict:
    """The ladder on ``name``: each rung's result and the one kept, and with
    ``nulls`` the null policies against the kept rung (``check_nulls``)."""
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
    placed = {k: {"sigma": v[0], "setpoint": v[1]} for k, v in row.placed.items()}
    out = {"rungs": results, "kept": kept, "left_out": row.why, "placed": placed}
    if nulls and kept:
        out["nulls"] = check_nulls(name, row, results[kept])
    return out


#: A zone's placement stops once its setpoint moves under this many sigma, the 2-sigma
#: rule's own resolution at 1 %, or after PLACE_ROUNDS reads.
PLACE_TOLERANCE, PLACE_ROUNDS = 0.01, 8

_ORDER = ("pi", "feedforward", "cascade", "decoupler", "override", "published")


class HoldReset:
    """The do-nothing null: each episode's actuator held where its reset put it,
    the plant's own starting move (rule 17), whatever the plant then does."""

    name, oracle = "hold reset", False

    def bind(self, env, params, seeds):
        import jax

        from target_gym.benchmark import episode_keys

        keys = [episode_keys(jax.random.PRNGKey(s), 1, 1)[0][0] for s in seeds]
        held = [env.reset_env(k, params)[1].actuator_held for k in keys]
        self.u = np.stack([np.asarray(h, float).reshape(-1) for h in held])

    def reset(self, key, obs, task):
        return np.zeros((task.n_episodes, task.n_act))

    def act(self, carry, obs, t):
        return self.u, carry


def null_set(name: str) -> list:
    """Charter J2's nulls on ``name``: the stored ones (``null_policies``) and,
    where the plant holds an actuator, the hold of its reset."""
    import jax

    from target_gym.null_policies import null_policies

    spec = REGISTRY[name]
    state = spec.make_env().reset_env(jax.random.PRNGKey(0), spec.make_test_params())[1]
    held = hasattr(state, "actuator_held")
    return list(null_policies(name)) + ([HoldReset()] if held else [])


def check_nulls(name: str, row, reference: dict) -> dict:
    """Charter J2: every null policy (``null_policies``) over the same scenarios and
    seeds as ``reference``, a ``run_suite`` result. A null whose stored table no
    longer fits the plant is reported with its error, not run. The tables and
    integrator gains come from the old plants' recordings until the re-record
    refits them, so a table null's result is marked stale."""
    runs, broken = {}, {}
    for null in null_set(name):
        try:
            runs[null.name] = run_suite(name, row, null)
        except Exception as err:  # a stale table's shape, or a missing field
            broken[null.name] = f"{type(err).__name__}: {err}"
    verdict = null_verdict(reference["costs"], {k: v["costs"] for k, v in runs.items()})
    stale = sorted(k for k in runs if k in ("table", "I-only"))
    for k, v in runs.items():
        print(
            f"  null {k}: median {np.median(v['costs']):.4g}, wins "
            f"{verdict['wins'][k]} of {verdict['seeds']}, {v['trips']} trips",
            flush=True,
        )
    return {**verdict, "runs": runs, "broken": broken, "stale": stale}


def expert_reference(name: str) -> dict:
    """A task whose reference is its shipped expert (aircraft, reactor): the expert
    over the suite, and the nulls against it."""
    print(name, flush=True)
    row = Row(0, None)
    reference = run_suite(name, row, shipped_policy("pid"))
    print(
        f"  expert: median {np.median(reference['costs']):.4g}, "
        f"{reference['trips']} trips",
        flush=True,
    )
    return {"reference": reference, "nulls": check_nulls(name, row, reference)}


#: The rows, one file per task in ``scripts/ladder_rows``, each defining ``row()``.
ROWS_DIR = pathlib.Path(__file__).resolve().parent / "ladder_rows"


def _row_of(path: pathlib.Path):
    def make() -> Row:
        # A row file imports this module by name, also when it runs as a script.
        sys.modules.setdefault("reference_ladder", sys.modules[__name__])
        spec = importlib.util.spec_from_file_location(f"ladder_rows.{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.row()

    return make


ROWS = {
    p.stem: _row_of(p) for p in sorted(ROWS_DIR.glob("*.py")) if not p.stem[0] == "_"
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", nargs="*", default=list(ROWS))
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--nulls", action="store_true", help="J2 against the kept rung")
    parser.add_argument(
        "--expert", nargs="*", default=[], help="tasks whose reference is the expert"
    )
    args = parser.parse_args()
    for name in args.expert:
        result = expert_reference(name)
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{name}.json").write_text(json.dumps(result, indent=1))
    for name in [] if args.expert and args.tasks == list(ROWS) else args.tasks:
        result = climb(name, args.nulls)
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{name}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
