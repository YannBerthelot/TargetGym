"""Tune the formation hold off the scored seeds, and check it on held-out ones.

The expert of ``patrol``, ``patrol_bearing_only`` and the formation's wingmen
is a formation hold (``patrol.experts.FormationHoldParams``) over the 3D
tasks' autopilot; the formation's lead flies that autopilot's course hold and
pitch at a set power (``formation.experts.FormationLeadParams``). This script
reproduces how their numbers were set:

1. The engine and energy loops on both patrol tasks: TECS's speed weight and
   throttle damping, the autothrottle's closure on the engine's power state,
   and the filter on the demands' rates.
2. The slot loops on both patrol tasks: along-track gain and closing
   deceleration, cross-track gain and the course hold's time constant.
3. The formation's lead behind wingmen flying the result: its pitch loop's
   damping, TECS's time constant, the set power and its speed trim.
4. The held-out check on seeds 200-209, against the tests' limits
   (tests/patrol/test_formation_hold.py).

Steps 1-3 fly training seeds 100-109 only and rank by the worst seed's RMS
slot error, the quantity the limit is set on. It prints what it finds; the
shipped values are set by hand in the two modules, with the reasons beside
them. The scored seeds 0-9 are never flown.

    uv run python scripts/tune_formation_hold.py            # all, ~10 min
    uv run python scripts/tune_formation_hold.py --check    # step 4 only
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

import target_gym as tg  # noqa: E402
from target_gym.eval import scored_burn_in  # noqa: E402
from target_gym.formation.env import slot_errors  # noqa: E402
from target_gym.patrol.env import slot_error, wrap_angle  # noqa: E402

TRAIN, HELD_OUT = range(100, 110), range(200, 210)
PATROL = ("patrol", "patrol_bearing_only")
SLOT_RMS_M, ALTITUDE_M = 8.5, 2.0
_ROLLOUTS = {}


def rollout(name, params, seeds):
    """Per seed, after the scored burn-in: RMS slot error of each wingman,
    the lead's mean altitude and heading errors (formation only) and whether
    it tripped. Jitted once per task and vmapped over the seeds; the gains
    are an argument, so a search does not recompile. A new key every step."""
    if name not in _ROLLOUTS:
        env, p = tg.make(name)
        ctrl = tg.make_expert(name)

        def measure(state):
            if name != "formation":
                return slot_error(state)[None], jnp.zeros(()), jnp.zeros(())
            lead = state.lead
            return (
                slot_errors(state),
                jnp.abs(lead.target_altitude - lead.z),
                jnp.abs(wrap_angle(lead.psi - lead.target_heading)),
            )

        def one(gains, seed):
            key = jax.random.PRNGKey(seed)
            obs, state = env.reset_env(key, p)

            def body(carry, i):
                obs, state, memory = carry
                action, memory = ctrl.step(gains, memory, obs, jnp)
                obs, state, _, _, info = env.step_env(
                    jax.random.fold_in(key, i), state, action, p
                )
                return (obs, state, memory), (measure(state), info["tripped"])

            init = (obs, state, ctrl.init_state)
            steps = jnp.arange(int(p.max_steps_in_episode))
            return lax.scan(body, init, steps)[1]

        burn = scored_burn_in(name, p)
        _ROLLOUTS[name] = (jax.jit(jax.vmap(one, in_axes=(None, 0))), burn)
    run, burn = _ROLLOUTS[name]
    (slot, alt, hdg), trip = run(params, jnp.array(list(seeds)))
    slot = np.asarray(slot)[:, burn:]
    rms = np.sqrt((slot**2).mean(axis=1))
    lead = np.asarray(alt)[:, burn:].mean(1), np.asarray(hdg)[:, burn:].mean(1)
    return rms, *lead, np.asarray(trip).any(1)


def _replace(params, field, value):
    """``params`` with ``field`` set, ``tecs.x`` reaching into TECS's."""
    if field.startswith("tecs."):
        return params.replace(tecs=params.tecs.replace(**{field[5:]: value}))
    return params.replace(**{field: value})


def search(label, names, base, grid, build=lambda p: p):
    """Every combination of ``grid`` on ``base``, ranked by the worst
    training seed's RMS slot error over ``names``, trips first."""
    rows = []
    for values in itertools.product(*grid.values()):
        gains = base
        for field, value in zip(grid, values):
            gains = _replace(gains, field, value)
        worst, trips = 0.0, 0
        for name in names:
            rms, _, _, trip = rollout(name, build(gains), TRAIN)
            worst, trips = max(worst, rms.max()), trips + int(trip.sum())
        rows.append((trips, worst, values))
    rows.sort()
    print(f"{label}: best on training seeds, {list(grid)}")
    for trips, worst, values in rows[:5]:
        print(f"  worst seed {worst:6.2f} m, {trips} trips  {values}")


def check():
    hold = tg.make_expert("patrol").params
    print("held-out seeds 200-209: worst seed, median, trips")
    for name in PATROL:
        rms, _, _, trip = rollout(name, hold, HELD_OUT)
        print(
            f"  {name:20s} slot {rms.max():5.2f} m, {np.median(rms):5.2f} m "
            f"(limit {SLOT_RMS_M}), {trip.sum()} trips"
        )
    team = tg.make_expert("formation").params
    rms, alt, hdg, trip = rollout("formation", team, HELD_OUT)
    print(
        f"  {'formation':20s} slot {rms.max():5.2f} m, {np.median(rms):5.2f} m, "
        f"{trip.sum()} trips\n  {'':20s} lead altitude {alt.max():.2f} m "
        f"(limit {ALTITUDE_M}), heading {np.rad2deg(hdg.max()):.3f} deg (limit 0.5)"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="the held-out check only")
    if ap.parse_args().check:
        check()
        return 0
    hold = tg.make_expert("patrol").params
    search(
        "engine and energy",
        PATROL,
        hold,
        {
            "tecs.spd_weight": (0.0, 0.5, 1.0),
            "tecs.thr_damp": (0.5, 6.0),
            "k_eng": (0.0, 1.0),
            "k_ref": (0.3, 1.0),
        },
    )
    search(
        "slot loops",
        PATROL,
        hold,
        {
            "k_x": (0.05, 0.08, 0.12),
            "a_x": (0.2, 0.4),
            "k_y": (0.05, 0.1, 0.2),
            "tau_hdg": (3.0, 5.0),
        },
    )
    team = tg.make_expert("formation").params
    search(
        "formation lead",
        ("formation",),
        team.lead,
        {
            "Kd_theta": (0.3, 1.0, 3.0),
            "tecs.time_const": (1.5, 2.0),
            "power_set": (0.75, 0.8),
            "k_speed": (0.002, 0.005, 0.01),
        },
        build=lambda lead: team.replace(lead=lead),
    )
    check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
