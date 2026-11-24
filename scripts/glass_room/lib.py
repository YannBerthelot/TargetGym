"""Glass room study: shared harness. Probes 1 to 10 ran on the plant at 4be0ab9, where
any lost glass was a lost mark; ``trace_of`` reads ``lost`` that way.

The harness: the env's own step on the benchmark's episode
keys, the ladder's kept rung rebuilt from its logged tuning, and traced rollouts."""

import importlib.util
import math
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from target_gym import scenarios  # noqa: E402
from target_gym.experts.ladder import Cascade, Pi, Rung, ladder_controller  # noqa: E402
from target_gym.glass_furnace import env_jax as E  # noqa: E402
from target_gym.glass_furnace import physical as P  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402

SPEC = REGISTRY["glass_furnace"]
ENV = SPEC.make_env()
DT = 120.0
HOUR = 30  # steps

# The ladder's tuning as logged on 2026-11-13 (bob, charter-records, glass_furnace
# reference log): SIMC designs kept at the most sensitive point, and the placed zone
# setpoint.
OUTER = dict(kc=432.32355264307927, ti=194919.07026192037, tf=9188.738091876485)
INNER = dict(kc=0.01833860575096497, ti=12116.316784758288, tf=494.53959809478573)
PLACED = 0.5523
CROWN_REST = 1827.82  # K, the crown couple at the design state


def scenario_params(name, **extra):
    s = {x.name: x for x in scenarios.suite(SPEC)}[name]
    return scenarios.params_for(SPEC, s, **{"sensor_noise_scale": 1.0, **extra})


def ladder_row():
    path = ROOT / "scripts" / "ladder_rows" / "glass_furnace.py"
    spec = importlib.util.spec_from_file_location("glass_row", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.row()


def _alpha(tf):
    return -math.expm1(-DT / tf) if tf else 1.0


def cascade(setpoint=PLACED, outer=True, bias=None, crown_rest=CROWN_REST):
    """The ladder's cascade rung: a crown PI on the valve under a fining PI on its
    setpoint, around the balance feedforward. ``outer`` False freezes the crown
    setpoint at ``crown_rest`` (no quality feedback)."""
    row = ladder_row() if bias is None else None
    bias = bias or row.balance
    kc = OUTER["kc"] if outer else 0.0
    loops = (Pi(0, kc, OUTER["ti"], _alpha(OUTER["tf"])),)
    inner = Pi(0, INNER["kc"], INNER["ti"], _alpha(INNER["tf"]))
    cas = Cascade(lambda o, xp: o[0], lambda o, xp: crown_rest, inner)
    rung = Rung("cascade", loops, np.ones((1, 1)), (cas,))
    read = lambda o, xp: xp.stack([o[4]])  # noqa: E731
    target = lambda o, xp: xp.stack(
        [xp.asarray(setpoint, dtype=xp.float32)]
    )  # noqa: E731
    return ladder_controller(rung, read, target, bias, DT)


def episode_keys(seed, n):
    k_reset, k_steps = jax.random.split(jax.random.PRNGKey(seed))
    return jax.random.split(k_reset, 1)[0], jax.random.split(k_steps, n)


def trace_of(s1, r, params):
    read = P.sensors(s1.plant, params.grid, params.plant)
    terms = E.compute_reward_terms(s1, params)
    return dict(
        cost=-r,
        fuel_cost=terms["running"],
        lost=terms["tracking"] > 0.0,
        X=read["fining"],
        hot=read["hot"],
        crown=read["crown"],
        canal=read["canal"],
        floor=read["floor"],
        valve=s1.fuel,
        cooling=s1.cooling,
        pull=s1.pull,
        gas=E.gas_factor(s1.walk, s1.time, s1, params),
        walk=s1.walk,
        line=E.batch_line(s1.plant, params),
    )


def run_controller(controller, params, seed, n=None):
    """One episode of a pure ladder controller: the traces, per step."""
    n = int(params.max_steps_in_episode) if n is None else n
    k_reset, keys = episode_keys(seed, n)
    obs, st = ENV.reset_env(k_reset, params)
    mem = jax.tree.map(jnp.asarray, controller.init_state)

    def f(c, k):
        s, m, o = c
        a, m = controller.step(controller.params, m, o, jnp)
        o1, s1, r, _, _ = ENV.step_env(k, s, a, params)
        tr = trace_of(s1, r, params)
        tr["reading"] = o1[4]
        tr["asked"] = E.to_fuel(a[0], params)
        return (s1, m, o1), tr

    return jax.lax.scan(f, (st, mem, obs), keys)[1]


def run_valve(valve, params, seed):
    """One episode of a valve plan, kg/s per step."""
    n = int(valve.shape[0])
    k_reset, keys = episode_keys(seed, n)
    _, st = ENV.reset_env(k_reset, params)

    def f(s, x):
        k, u = x
        _, s1, r, _, _ = ENV.step_env(k, s, E.to_action(u, params), params)
        tr = trace_of(s1, r, params)
        tr["asked"] = u
        return s1, tr

    return jax.lax.scan(f, st, (keys, valve))[1]


def summary(tr, half=None):
    """Numbers of one traced episode (numpy)."""
    t = {k: np.asarray(v) for k, v in tr.items()}
    n = len(t["cost"])
    h = n // 2 if half is None else half
    return dict(
        cost=float(t["cost"].mean()),
        fuel=float(t["fuel_cost"].mean()),
        lost_h=float(t["lost"].sum() / HOUR),
        X_mean=float(t["X"].mean()),
        X_min=float(t["X"].min()),
        X_sd=float(t["X"].std()),
        X_sd_2nd=float(t["X"][h:].std()),
        X_mean_2nd=float(t["X"][h:].mean()),
        reading_sd_2nd=float(t["reading"][h:].std()) if "reading" in t else None,
        hot_max_C=float(t["hot"].max() - 273.15),
        hot_min_C=float(t["hot"].min() - 273.15),
        hot_sd=float(t["hot"].std()),
        capped_h=float((t["asked"] - t["valve"] > 1e-6).sum() / HOUR),
        valve_min=float(t["valve"].min()),
        valve_max=float(t["valve"].max()),
        valve_move=float(np.abs(np.diff(t["valve"])).sum()),
    )


def start(controller, params, seed):
    """The carry a rollout starts from: ``(state, memory, obs)``."""
    k_reset, _ = episode_keys(seed, 1)
    obs, st = ENV.reset_env(k_reset, params)
    return st, jax.tree.map(jnp.asarray, controller.init_state), obs


def roll(controller, params, carry, keys):
    """``controller`` from ``carry`` over ``keys``: ``(carry, traces)``."""

    def f(c, k):
        s, m, o = c
        a, m = controller.step(controller.params, m, o, jnp)
        o1, s1, r, _, _ = ENV.step_env(k, s, a, params)
        tr = trace_of(s1, r, params)
        tr["reading"] = o1[4]
        tr["asked"] = E.to_fuel(a[0], params)
        return (s1, m, o1), tr

    return jax.lax.scan(f, carry, keys)
