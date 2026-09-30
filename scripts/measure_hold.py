"""Measure what the shipped controllers hold, plant by plant.

Run with ``uv run python scripts/measure_hold.py [--envs ...] [--seeds N]``.
Writes ``src/target_gym/data/hold_measurements.json``, one row per
(environment, controller): the long-run mean tracking error after a burn-in, the
running-cost quantity consumed per step while holding, and the time the
controller takes to settle after a target change. These are the inputs to the
floor-normalised reward (docs/reward-shaping.md):

``e_hold``  mean |error| per tracked output after ``burn_in`` steps and, within
            each target cycle, after ``settle`` steps. The MPC's value is an
            *upper bound* on the achievable floor ``e_floor`` -- it is a real
            controller, so no floor may lie above it -- and the sanity check every
            floor has to pass.
``c_hold``  the running-cost quantity (fuel, energy, boilup, reagent, actuator
            travel) per step, same window. Only consumption above the best
            shipped controller's ``c_hold`` is charged by the reward.
``settle``  median steps after a target change until |error| first falls within
            twice ``e_hold``; the evaluation protocol drops that many steps from
            the start of each cycle before scoring the hold.

``burn_in`` is three times the slowest *cost-bearing* time constant of the plant,
in env steps, from each plant's PHYSICS.md (the table ``PLANTS`` below carries
the source). It is not the actuator-to-output response time the protocol's
``measure_time_constants.py`` reports -- a thermal mass or a demand process can
be far slower than the loop that tracks through it -- and it is checked here:
the row records the hold error over the first and second halves of the window,
which should agree if the burn-in was long enough.

Episodes are run longer than ``EnvSpec.test_params`` where the burn-in needs
it, on the shipped reference and disturbance processes. The aircraft keep their
test length: their targets are scheduled against the episode, so lengthening
one changes the task.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from target_gym.registry import REGISTRY, control_step_seconds  # noqa: E402
from target_gym.runners.runners import (  # noqa: E402
    _as_tuple,
    _wants_state,
    baseline_policy,
)

OUT = ROOT / "src" / "target_gym" / "data" / "hold_measurements.json"


# ---------------------------------------------------------------------------
# Per-plant accessors. ``errors`` returns the tracked errors the reward scores
# (one entry per output), ``consumption`` the running-cost quantity per step or
# None. ``tau_cost`` is the slowest cost-bearing time constant in env steps
# with its source; ``hold`` the hold window to score after burn-in.
# ---------------------------------------------------------------------------


def _obs_errors(env):
    vi, ti = list(_as_tuple(env.obs_value_index)), list(_as_tuple(env.obs_target_index))

    def f(obs, state, params):
        o = np.asarray(obs)
        return np.abs(o[vi] - o[ti])

    return f


def _plane3d_errors(kind):
    from target_gym.plane3d.env import (
        distance_to_circle,
        distance_to_racetrack,
        nearest_point_on_twisted_lemniscate,
        wrap_angle,
    )

    def f(obs, state, params):
        alt = abs(float(state.target_altitude - state.z))
        if kind == "heading":
            return np.array(
                [alt, abs(float(wrap_angle(state.psi - state.target_heading)))]
            )
        if kind == "circle":
            return np.array([alt, abs(float(distance_to_circle(state)))])
        if kind == "racetrack":
            return np.array([alt, abs(float(distance_to_racetrack(state)))])
        _, _, _, dist, _ = nearest_point_on_twisted_lemniscate(state, params)
        return np.array([float(dist)])

    return f


def _patrol_errors(obs, state, params):
    from target_gym.patrol.env import slot_error, wrap_angle

    return np.array(
        [
            float(slot_error(state)),
            abs(float(wrap_angle(state.follower.psi - state.lead.psi))),
        ]
    )


def _plane_speed(obs, state, params):
    return abs(float(np.sqrt(state.x_dot**2 + state.z_dot**2) - params.target_speed))


def _battery_fade(obs, state, params):
    from target_gym.energy.battery.env import degradation_rate

    return float(degradation_rate(state.current, state.T_cell, params) * params.delta_t)


def _boiler_errors(obs, state, params):
    return np.array(
        [abs(float(state.level)), abs(float(state.pressure - state.target_pressure))]
    )


def _kiln_errors(obs, state, params):
    from target_gym.cement_kiln.env import discharge_lime

    return np.array([abs(float(discharge_lime(state) - state.target_lime))])


def _wind_errors(obs, state, params):
    from target_gym.energy.wind_turbine.env import electrical_power

    return np.array(
        [
            abs(
                float(
                    state.target_power
                    - electrical_power(state.omega, state.torque, params)
                )
            )
        ]
    )


PLANTS = {
    # name: (errors or None for obs-based, consumption, tau_cost steps, source, hold steps)
    "plane": (None, _plane_speed, 70, "phugoid ~1 min at 1 s steps", None),
    "plane_energy": (None, _plane_speed, 70, "phugoid ~1 min", None),
    "plane_sine": (None, _plane_speed, 70, "phugoid ~1 min", None),
    "plane3d_heading": (_plane3d_errors("heading"), None, 70, "phugoid ~1 min", None),
    "plane3d_circle": (_plane3d_errors("circle"), None, 70, "phugoid ~1 min", None),
    "plane3d_racetrack": (
        _plane3d_errors("racetrack"),
        None,
        70,
        "phugoid ~1 min",
        None,
    ),
    "plane3d_figure8": (_plane3d_errors("figure8"), None, 70, "phugoid ~1 min", None),
    "patrol": (_patrol_errors, None, 70, "phugoid ~1 min", None),
    "cstr": (None, None, 4, "residence ~1 min at 15 s steps", 100),
    "first_order": (None, None, 10, "tau = 0.5 s at 0.05 s steps", 100),
    "four_tank": (None, None, 90, "tank tau ~1.5 min at 1 s steps", 300),
    "ph_neutralization": (
        None,
        lambda o, s, p: float(s.q3),
        36,
        "residence ~3 min at 5 s steps",
        300,
    ),
    "distillation": (
        lambda o, s, p: np.array(
            [abs(float(s.target_yD - s.x[-1])), abs(float(s.target_xB - s.x[0]))]
        ),
        lambda o, s, p: float(s.V),
        194,
        "dominant tau ~194 min at 1 min steps",
        400,
    ),
    "glass_furnace": (
        None,
        lambda o, s, p: float(s.fuel_flow),
        3600,
        "T_melt, T_work ~30 h at 30 s steps",
        3600,
    ),
    "reactor": (
        None,
        lambda o, s, p: abs(float(s.rho_ext_cmd - s.rho_ext)),
        667,
        "OU demand 1/theta = 6667 s at 10 s steps (xenon constrains reach only)",
        2000,
    ),
    "hvac": (
        None,
        lambda o, s, p: float(s.Q_emitter),
        172,
        "thermal mass 43 h at 15 min steps",
        720,
    ),
    "cement_kiln": (
        _kiln_errors,
        lambda o, s, p: float(s.fuel),
        60,
        "22 min residence + transport delay at 30 s steps",
        500,
    ),
    "boiler_drum": (
        _boiler_errors,
        lambda o, s, p: float(s.Q_fuel),
        20,
        "riser 18 s at 2 s steps; drum pressure slower",
        400,
    ),
    "wind_turbine": (
        _wind_errors,
        lambda o, s, p: abs(float(s.pitch_cmd - s.pitch)) / float(p.pitch_max),
        100,
        "rotor inertia, tens of seconds at 0.25 s steps",
        400,
    ),
    # The battery's dispatch schedule covers the 60 min test episode and holds
    # its last block forever after, draining the pack; so it keeps the test
    # length and no burn-in -- the cell's 50 min thermal constant moves the fade
    # cost, not the tracking error, and the hold is scored per dispatch block
    # after settling.
    "battery": (
        lambda o, s, p: np.array([abs(float(s.target_power - s.power))]),
        _battery_fade,
        0,
        "per dispatch block; thermal 50 min affects fade only",
        None,
    ),
}


def _episode(spec, env, params, kind, seed, errors, consumption):
    """One long episode; per-step errors, consumption, target vector, termination."""
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)
    step = jax.jit(env.step_env)
    policy = baseline_policy(spec, kind, params)
    ti = list(_as_tuple(env.obs_target_index))
    E, C, T = [], [], []
    trips = 0  # a trip never ends the window (``base.failure_kernel``)
    while int(state.time) < int(params.max_steps_in_episode):
        o = np.asarray(obs)
        a = policy(o, state) if _wants_state(policy) else policy(o)
        obs, state, _r, _term, info = step(
            key, state, jnp.atleast_1d(jnp.asarray(a)), params
        )
        o2 = np.asarray(obs)
        E.append(errors(o2, state, params))
        C.append(consumption(o2, state, params) if consumption else np.nan)
        T.append(o2[ti])
        trips += int(bool(info.get("tripped", False)))
    return np.array(E), np.array(C, dtype=float), np.array(T), trips


def _cycles(targets, rel=0.05):
    """Cycle starts: steps where the target jumps by more than ``rel`` of its
    range. A continuously drifting target (OU demand, a moving lead) never jumps
    and is one cycle; a dispatch block or a ladder gives one cycle per level."""
    span = np.ptp(targets, axis=0)
    span = np.where(span > 0, span, 1.0)
    jump = np.abs(np.diff(targets, axis=0)) / span
    starts = np.flatnonzero((jump > rel).any(axis=1)) + 1
    return np.concatenate([[0], starts, [len(targets)]])


def _hold_mask(n, burn_in, bounds, settle):
    m = np.zeros(n, bool)
    for a, b in zip(bounds[:-1], bounds[1:]):
        m[a + settle : b] = True
    m[:burn_in] = False
    return m


def _settle(errors, bounds, level):
    """Median steps per cycle until every output is within 2x its hold level."""
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = errors[a:b]
        ok = (seg <= 2.0 * level[None, :]).all(axis=1)
        idx = np.flatnonzero(ok)
        out.append(int(idx[0]) if len(idx) else b - a)
    return float(np.median(out)) if out else float("nan")


def measure(name, seeds):
    spec = REGISTRY[name]
    env = spec.make_env()
    p = spec.make_test_params()
    err_fn, cons_fn, tau, source, hold = PLANTS[name]
    if err_fn is None:
        err_fn = _obs_errors(env)
    burn_in = 3 * tau
    n_test = int(p.max_steps_in_episode)
    if hold is None:
        steps = n_test  # aircraft: scheduled targets, keep the task
    else:
        steps = max(n_test, burn_in + hold)
    params = p.replace(max_steps_in_episode=steps)
    row = {
        "steps": steps,
        "step_seconds": control_step_seconds(env, p),
        "burn_in": burn_in,
        "tau_cost_steps": tau,
        "tau_cost_source": source,
        "seeds": seeds,
    }
    kinds = ["pid"] + (["mpc"] if spec.make_mpc is not None else [])
    for kind in kinds:
        t0 = time.time()
        E, C, B, term, half1, half2 = [], [], [], 0, [], []
        for seed in range(seeds):
            e, c, tg, ended = _episode(spec, env, params, kind, seed, err_fn, cons_fn)
            term += int(ended)
            bounds = _cycles(tg)
            # Provisional hold level from the burn-in window alone, then settle,
            # then the hold mask that drops the settle from every cycle.
            m0 = _hold_mask(len(e), burn_in, bounds, 0)
            if m0.sum() < 10:
                m0 = np.ones(len(e), bool)
                m0[: len(e) // 2] = False
            lvl = e[m0].mean(axis=0)
            settle = _settle(e, bounds, lvl)
            m = _hold_mask(
                len(e), burn_in, bounds, int(min(settle, (len(e) - burn_in) // 4))
            )
            if m.sum() < 10:
                m = m0
            E.append(e[m])
            C.append(c[m])
            B.append(settle)
            idx = np.flatnonzero(m)
            h = len(idx) // 2
            half1.append(e[idx[:h]].mean(axis=0))
            half2.append(e[idx[h:]].mean(axis=0))
        per_seed = [e.mean(axis=0).tolist() for e in E]
        E = np.concatenate(E)
        C = np.concatenate(C)
        row[kind] = {
            "e_hold": E.mean(axis=0).tolist(),
            # Per seed, so a floor can be the lowest hold the controller
            # demonstrated rather than a mean it sits below on some seeds.
            "e_hold_per_seed": per_seed,
            "e_hold_min": np.min(per_seed, axis=0).tolist(),
            "e_hold_rms": np.sqrt((E**2).mean(axis=0)).tolist(),
            "e_hold_first_half": np.mean(half1, axis=0).tolist(),
            "e_hold_second_half": np.mean(half2, axis=0).tolist(),
            "c_hold": None if np.isnan(C).all() else float(np.nanmean(C)),
            "settle": float(np.median(B)),
            "cycles_per_episode": len(bounds) - 1,
            "trips": term,
            "hold_steps_scored": int(len(E)),
            "seconds": round(time.time() - t0, 1),
        }
        print(
            f"  {name:20s} {kind:3s} e_hold={np.round(E.mean(axis=0), 6).tolist()} "
            f"c_hold={row[kind]['c_hold']} settle={row[kind]['settle']:.0f} "
            f"term={term} [{row[kind]['seconds']}s]",
            flush=True,
        )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=str(OUT), help="merge rows into this JSON")
    args = ap.parse_args()
    names = args.envs or list(PLANTS)
    out_path = pathlib.Path(args.out)
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names:
        seeds = args.seeds
        if name.startswith(("plane", "patrol")):
            seeds = min(seeds, 2)  # deterministic plants, expensive planners
        out[name] = measure(name, seeds)
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
