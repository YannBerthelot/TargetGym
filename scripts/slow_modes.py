"""Measure each task's slow mode ``t_s`` (docs/proposals/long-strata.md, R1).

    python scripts/slow_modes.py --tasks cstr first_order --out slow_modes.json

For each task: ten coupled pairs under the shipped reference (``pid``), on seeds
100 to 109, run B in manual mode for half a window, then ``--windows`` windows of
the reference on both. Prints and writes ``t_s``, the slowest element, the
elements that never forgot (to name in ``IGNORE`` if they are bookkeeping) and the
start-up excess cost. Heavy on the slow plants: run it on bob.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import numpy as np

#: Bookkeeping per task, never plant state: name -> reason. Filled from the first
#: run's ``never_forgot`` lists, each entry checked against the task's env code.
_POSITION = "position along the track; no hold task's cost reads it"
_FUEL = "fuel burnt is history: a monotone burn, not a mode that decays"
_PLANE = {".x": _POSITION, ".m": _FUEL, ".fuel": _FUEL}
_PLANE3D = {**_PLANE, ".y": _POSITION}
_PATROL = {".x": _POSITION, ".y": _POSITION, ".m": _FUEL, ".fuel": _FUEL}
_PHASE = (
    "world-frame heading and velocity on a closed path: a point further along "
    "it is the same task, so they never reconverge; path.* errors stand in"
)
_ON_PATH = {**_PLANE3D, ".psi": _PHASE, ".x_dot": _PHASE, ".y_dot": _PHASE}
IGNORE: dict[str, dict[str, str]] = {
    "plane": _PLANE,
    "plane_energy": _PLANE,
    "plane_sine": _PLANE,
    "plane3d_heading": _PLANE3D,
    **{f"plane3d_{k}": _ON_PATH for k in ("circle", "racetrack", "figure8")},
    "patrol": _PATROL,
    "patrol_range_bearing": _PATROL,
    "formation": {
        **_PATROL,
        ".wake_history": "the lead's past absolute positions; the wake acts only "
        "relative to the wingmen, so a re-formed lead elsewhere is the same wake",
    },
    "compressor_surge": {
        ".surge_times": "a log of past surge times, counted only inside "
        "surge_window (env.py:688), so its effect expires on its own"
    },
    "battery": {
        ".baselines": "declared per period; past entries are history and the "
        "current one acts through soc and period_response"
    },
}

#: Angles compared on the circle, so a whole turn is no gap.
ANGLES = (".psi", ".target_heading")

SEEDS = range(100, 110)

#: Ignored on every task: readings of plant states measured directly, rounded
#: to the instrument's resolution, so two near-identical runs flicker a step apart.
COMMON = {".sensor_history": "rounded readings of states measured directly"}


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _speed_and_height(s):
    speed = float(np.sqrt(s.x_dot**2 + s.y_dot**2 + s.z_dot**2))
    return {"speed": speed, "height_error": float(s.z - s.target_altitude)}


def _circle(s, p):
    dx, dy = float(s.x - s.target_x), float(s.y - s.target_y)
    r = float(np.hypot(dx, dy))
    radial_speed = (dx * float(s.x_dot) + dy * float(s.y_dot)) / max(r, 1e-9)
    return {
        "radius_error": r - float(s.target_radius),
        "radial_speed": radial_speed,
        **_speed_and_height(s),
    }


def _racetrack(s, p):
    from target_gym.plane3d.env import racetrack_guidance

    cross, tangent, _ = racetrack_guidance(s)
    return {
        "cross_track": float(cross),
        "heading_error": float(_wrap(float(s.psi) - float(tangent))),
        **_speed_and_height(s),
    }


def _figure8(s, p):
    from target_gym.plane3d.env import nearest_point_on_twisted_lemniscate

    _, _, dz, dist, tangent = nearest_point_on_twisted_lemniscate(s, p)
    return {
        "path_distance": float(dist),
        "vertical_error": float(dz),
        "heading_error": float(_wrap(float(s.psi) - float(tangent))),
        "speed": _speed_and_height(s)["speed"],
    }


#: Path-relative errors, measured where the world frame never reconverges.
OBSERVE = {
    "plane3d_circle": _circle,
    "plane3d_racetrack": _racetrack,
    "plane3d_figure8": _figure8,
}


def measure(
    name: str, windows: float, offset_share: float, pre: float, bias: bool
) -> dict:
    from target_gym.registry import REGISTRY
    from target_gym.runners.runners import baseline_policy
    from target_gym.slow_mode import coupled_pair, slow_mode

    spec = REGISTRY[name]
    env, params = spec.make_env(), spec.make_test_params()
    space = env.action_space(params)
    low, high = np.asarray(space.low, float), np.asarray(space.high, float)
    n = int(params.max_steps_in_episode)
    pairs = [
        coupled_pair(
            env,
            params,
            lambda p: baseline_policy(spec, "pid", p),
            seed=k,
            pre=int(pre * n),
            length=int(windows * n),
            offset=offset_share * (high - low),
            low=low,
            high=high,
            bias=bias,
            angles=ANGLES,
            observe=OBSERVE.get(name),
        )
        for k in SEEDS
    ]
    ignored = {**COMMON, **IGNORE.get(name, {})}
    result = slow_mode(pairs, ignore=tuple(ignored))
    return {
        "task": name,
        "N": n,
        "t_s": result.t_s,
        "state_t_s": result.state_t_s,
        "slowest": result.slowest,
        "never_forgot": list(result.never_forgot),
        "start_excess": result.start_excess,
        "tripped_pairs": result.tripped_pairs,
        "start_gap": float(result.gap[0]),
        "pre": pre,
        "bias": bias,
        "offset": offset_share,
        "ignored": ignored,
        "windows": windows,
        "seeds": list(SEEDS),
    }


def main() -> None:
    from target_gym.registry import REGISTRY

    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(REGISTRY))
    ap.add_argument("--windows", type=float, default=3.0)
    ap.add_argument("--offset", type=float, default=0.1, help="share of the span")
    ap.add_argument("--pre", type=float, default=0.5, help="share of a window")
    ap.add_argument("--bias", action="store_true", help="closed-loop excursion")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="slow_modes.json")
    args = ap.parse_args()
    jobs = [(t, args.windows, args.offset, args.pre, args.bias) for t in args.tasks]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = []
        for row in pool.starmap(measure, jobs):
            print(json.dumps(row), flush=True)
            rows.append(row)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
