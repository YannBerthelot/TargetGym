"""Tune the aircraft experts the way an autopilot is tuned, off the scored seeds.

The experts are what PX4 and ArduPilot fly: TECS for height and speed over a
pitch loop (``plane.experts_tecs``), and on the 3D tasks a damped bank loop
under a course hold or L1 guidance (``plane3d.experts``). This script
reproduces how their numbers were set:

1. Flight test: at 230 m/s with no turbulence, pitch holding the speed, the
   climb at full throttle, the sink at idle and the trim throttle and pitch,
   at 3500, 5500 and 7500 m. These are TECS's aircraft numbers, measured as
   ArduPilot's tuning guide has a pilot measure them.
2. A bounded search on training seeds 100-109, inner loops first: the pitch
   loop and TECS on ``plane3d_heading``; the roll loop and the L1 period on
   ``plane3d_circle`` and ``plane3d_racetrack``, with L1's damping left at
   ArduPilot's 0.75 as its guide tunes.
3. The figure-8's own longitudinal loops and bank margin, against its
   flight-test spec (height within 2 m, path within 3 m where the lobes are
   flyable): the worst episode over the training patterns, each in a few
   turbulence draws, and the worst when any one gain is 25% off.
4. The held-out check on seeds 200-209, for all seven aircraft tasks.

It prints what it finds; the shipped values are set by hand in the two
modules, with the reasons beside them. The scored seeds 0-9 are never flown.

    uv run python scripts/tune_aircraft_guidance.py            # all, ~20 min
    uv run python scripts/tune_aircraft_guidance.py --check    # step 4 only
"""

from __future__ import annotations

import argparse
import itertools
import sys
import warnings

warnings.filterwarnings("ignore")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

from target_gym import eval as E  # noqa: E402
from target_gym.plane import experts as P2  # noqa: E402
from target_gym.plane3d import experts as P3  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402

TRAIN, HELD_OUT = range(100, 110), range(200, 210)
STEPS = {
    "plane3d_heading": P3.plane3d_heading_step,
    "plane3d_circle": P3.plane3d_circle_step,
    "plane3d_racetrack": P3.plane3d_racetrack_step,
    "plane3d_figure8": P3.plane3d_figure8_step,
}
SHIPPED = {
    "plane3d_heading": P3.make_plane3d_heading_pid_controller().params,
    "plane3d_circle": P3.make_plane3d_circle_pid_controller().params,
    "plane3d_racetrack": P3.make_plane3d_racetrack_pid_controller().params,
    "plane3d_figure8": P3.make_plane3d_figure8_pid_controller().params,
}
_ROLLOUTS = {}


def rollout(name, step, params, seeds, env_params=None):
    """Observations and per-step tracking cost of ``step`` on ``seeds``,
    jitted once per task and controller and vmapped over the seeds."""
    spec = REGISTRY[name]
    p = env_params or spec.make_test_params()
    key = (name, step, repr(p))
    if key not in _ROLLOUTS:
        env = spec.make_env()

        def one(ctrl_params, seed):
            k = jax.random.PRNGKey(seed)
            obs, st = env.reset_env(k, p)
            init = P3.Plane3DGuidanceState() if name in STEPS else P2.PlaneTECSState()

            def body(carry, _):
                obs, st, cs = carry
                a, cs = step(ctrl_params, cs, obs, jnp)
                obs, st, _, _, _ = env.step_env(k, st, a, p)
                return (obs, st, cs), (obs, env.reward_terms(st, p)["tracking"])

            return lax.scan(body, (obs, st, init), None, int(p.max_steps_in_episode))[1]

        _ROLLOUTS[key] = jax.jit(jax.vmap(one, in_axes=(None, 0)))
    obs, track = _ROLLOUTS[key](params, jnp.array(list(seeds)))
    return np.asarray(obs), np.asarray(track)


def hold(name, step, params, seeds):
    """Mean tracking cost over the protocol's hold window: the scored burn-in,
    and the plant's fixed settle after each target change."""
    obs, track = rollout(name, step, params, seeds)
    p = REGISTRY[name].make_test_params()
    burn, settle = E.scored_burn_in(name, p), E.fixed_settle(name)
    ti = 6 if name.startswith("plane") and name not in STEPS else 10
    out = []
    for o, c in zip(obs, track):
        change = np.r_[False, np.abs(np.diff(o[:, ti])) > 0.05 * abs(o[0, ti])]
        ep = E.Episode(c, np.zeros(len(c), bool), change, terms={"tracking": c})
        out.append(c[E.hold_mask(ep, burn, settle)].mean())
    return float(np.mean(out))


def flight_test():
    spec = REGISTRY["plane3d_heading"]
    env = spec.make_env()
    p = spec.make_test_params().replace(turbulence_sigma=0.0)
    step = jax.jit(env.step_env)
    key = jax.random.PRNGKey(0)
    for alt in (3500.0, 5500.0, 7500.0):
        rows = []
        for thr in (-1.0, -0.5, 0.0, 0.5, 1.0):
            _, s = env.reset_env(key, p)
            s = s.replace(z=alt, target_altitude=alt, x_dot=230.0, y_dot=0.0)
            s = s.replace(z_dot=0.0, target_heading=0.0, power=0.5)
            integ, log = 0.0, []
            for _ in range(240):
                v = float(jnp.sqrt(s.x_dot**2 + s.y_dot**2 + s.z_dot**2))
                integ += 230.0 - v
                cmd = np.clip(0.03 - 0.01 * (230.0 - v) - 2e-4 * integ, -0.26, 0.26)
                stick = np.clip(
                    4.0 * (cmd - float(s.theta)) - 3.0 * float(s.theta_dot), -1, 1
                )
                _, s, _, _, _ = step(key, s, jnp.array([thr, stick, 0.0]), p)
                log.append((float(s.z_dot), float(s.theta)))
            rows.append(np.mean(log[-60:], axis=0))
        climb, theta = np.array(rows).T
        trim = np.interp(0.0, climb, [-1.0, -0.5, 0.0, 0.5, 1.0])
        print(
            f"{alt:.0f} m: climb at full {climb[-1]:.1f} m/s, sink at idle "
            f"{-climb[0]:.1f} m/s, trim throttle {trim:.2f}, trim pitch "
            f"{np.interp(trim, [-1.0, -0.5, 0.0, 0.5, 1.0], theta):.4f} rad"
        )


def _with(params, fields):
    for field, value in fields.items():
        if field.startswith("tecs."):
            params = params.replace(tecs=params.tecs.replace(**{field[5:]: value}))
        else:
            params = params.replace(**{field: value})
    return params


def search(label, names, base, grid):
    """Every combination of ``grid`` on ``base``, ranked by the geometric mean
    of the tasks' training holds."""
    rows = []
    for values in itertools.product(*grid.values()):
        params = _with(base, dict(zip(grid, values)))
        holds = [hold(n, STEPS[n], params, TRAIN) for n in names]
        rows.append((float(np.exp(np.mean(np.log(holds)))), values))
    rows.sort()
    print(f"{label}: best holds on training seeds, {list(grid)}")
    for score, values in rows[:5]:
        print(f"  {score:9.3f}  {values}")


_FIG8 = {}


def figure8_spec(params, draws=3):
    """The figure-8's worst episode on the training patterns, each flown in
    ``draws`` turbulence draws (the step keys split as the tests split them):
    ``(height, path where the lobes are flyable, trips)``, mean |error| after
    the scored burn-in, as tests/plane3d/test_plane3d_guidance.py measures."""
    if "fly" not in _FIG8:
        spec = REGISTRY["plane3d_figure8"]
        env, p = spec.make_env(), spec.make_test_params()

        def one(ctrl_params, seed, draw):
            k = jax.random.PRNGKey(seed)
            obs, st = env.reset_env(k, p)

            def body(carry, _):
                obs, st, cs, k = carry
                a, cs = P3.plane3d_figure8_step(ctrl_params, cs, obs, jnp)
                k, sub = jax.random.split(k)
                obs, st, _, _, info = env.step_env(sub, st, a, p)
                return (obs, st, cs, k), (obs, info["tripped"])

            k_steps = jnp.where(draw > 0, jax.random.fold_in(k, draw), k)
            init = (obs, st, P3.Plane3DGuidanceState(), k_steps)
            seen, trips = lax.scan(body, init, None, int(p.max_steps_in_episode))[1]
            return obs, seen, trips

        _FIG8["fly"] = jax.jit(jax.vmap(one, in_axes=(None, 0, 0)))
        _FIG8["burn"] = E.scored_burn_in("plane3d_figure8", p)
        _FIG8["dz"] = p.figure8_altitude_amplitude
    seeds, ds = np.meshgrid(np.array(TRAIN), np.arange(draws))
    first, seen, trips = _FIG8["fly"](params, seeds.ravel(), ds.ravel())
    first, seen = np.asarray(first), np.asarray(seen)[:, _FIG8["burn"] :]
    # The lobe tips (radius a / 3) inside 45 deg of bank at the slowest
    # speed stall prevention allows there, as the test's _lobes_flyable.
    sigma = (1.0 - 2.2558e-5 * (first[:, 10] + _FIG8["dz"])) ** 4.2559
    flyable = (104.0**2 / sigma) / (9.81 * np.cos(np.pi / 4)) <= first[:, 11] / 3.0
    height = np.abs(seen[..., 14]).mean(1)
    path = np.hypot(seen[..., 12], seen[..., 13]).mean(1)[flyable]
    return height.max(), path.max(), int(np.asarray(trips).any(1).sum())


def figure8_search(grid):
    """Every combination of ``grid`` on the default parameters, ranked by the
    worse of the worst height and path in units of their targets, then the
    five best also with each gain in turn 25% off either way."""
    rows = []
    for values in itertools.product(*grid.values()):
        fields = dict(zip(grid, values))
        h, p, trips = figure8_spec(_with(P3.Plane3DGuidanceParams(), fields))
        rows.append((max(h / 2.0, p / 3.0) + trips, h, p, fields))
    rows.sort(key=lambda r: r[0])
    print(f"figure-8: worst height and path on training seeds, {list(grid)}")
    for _, h, p, fields in rows[:5]:
        worst = [h, p]
        for field in grid:
            for factor in (0.75, 1.25):
                off = dict(fields, **{field: fields[field] * factor})
                hh, pp, trips = figure8_spec(_with(P3.Plane3DGuidanceParams(), off))
                worst = [max(worst[0], hh + 100 * trips), max(worst[1], pp)]
        print(
            f"  height {h:5.2f} path {p:5.2f} | gains 25% off: "
            f"{worst[0]:6.2f} {worst[1]:6.2f}  {tuple(fields.values())}"
        )


def check():
    base2 = P2.PlaneTECSParams()
    print("held-out seeds 200-209, mean tracking cost over the hold window")
    for name in STEPS:
        print(f"  {name:18s} {hold(name, STEPS[name], SHIPPED[name], HELD_OUT):9.3f}")
    for name in ("plane", "plane_sine", "plane_energy"):
        print(f"  {name:18s} {hold(name, P2.plane_tecs_step, base2, HELD_OUT):9.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="the held-out check only")
    if ap.parse_args().check:
        check()
        return 0
    flight_test()
    base = P3.Plane3DGuidanceParams()
    search(
        "pitch loop and TECS",
        ["plane3d_heading"],
        base,
        {
            "Kp_theta": (6.0, 8.0, 10.0),
            "Ki_theta": (2.0, 4.0, 8.0),
            "Kd_theta": (0.0, 0.3),
            "tecs.time_const": (2.0, 3.0),
            "tecs.ptch_damp": (0.5, 1.0),
            "tecs.integ_gain": (0.1, 0.2),
        },
    )
    search(
        "roll loop and L1 period",
        ["plane3d_circle", "plane3d_racetrack"],
        base,
        {
            "Kp_roll": (4.0, 6.0, 8.0),
            "Kd_roll": (2.0, 3.0, 4.5),
            "period": (25.0, 30.0, 35.0),
        },
    )
    figure8_search(
        {
            "Kp_theta": (8.0, 10.0),
            "Kd_theta": (0.3, 4.0, 6.0, 8.0),
            "tecs.time_const": (1.25, 1.5, 2.0),
            "tecs.ptch_damp": (1.0, 1.5, 2.0),
            "turn_bank_margin": (0.75, 0.85),
        }
    )
    check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
