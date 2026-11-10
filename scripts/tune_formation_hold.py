"""Tune the formation hold off the scored seeds, and check it on held-out ones.

The expert of ``patrol``, ``patrol_range_bearing`` and the formation's wingmen
is a formation hold (``patrol.experts.FormationHoldParams``) over the 3D
tasks' autopilot; the formation's lead flies that autopilot's course hold and
pitch at a set power (``formation.experts.FormationLeadParams``). Every value
either module tunes is searched here, in the step its comment names:

1. [engine and energy]: TECS's speed weight and throttle damping, the
   autothrottle's closure on the engine's power state, and the filter on the
   demands' rates.
2. [slot loops]: along-track gain and closing deceleration, cross-track gain
   and the course hold's time constant.
3. [integrals and filter]: the two integral gains, the band they integrate
   within, and the filter on the lead's turn rate.
4. [lead pitch]: the formation lead's pitch-loop damping and TECS's time
   constant, on its altitude.
5. [lead power]: its set power and speed trim, on its wingmen's slots.
6. The held-out check on seeds 200-209, flown as the tests fly it
   (tests/patrol/test_formation_hold.py: the host form, a split key chain),
   against their limits.

Steps 1-5 fly training seeds 100-109 only, each around the shipped values,
and rank by the worst seed, trips first. Steps 1-3 rank one gain set for
every follower (both patrol tasks and the formation's wingmen behind the
shipped lead) on RMS slot error, the quantity the limit is set on. It prints
what it finds; the shipped values are set by hand in the two modules. The
scored seeds 0-9 are never flown.

    uv run python scripts/tune_formation_hold.py            # all, ~10 min
    uv run python scripts/tune_formation_hold.py --check    # step 6 only
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
from target_gym import registry  # noqa: E402
from target_gym.eval import scored_burn_in  # noqa: E402
from target_gym.formation.env import slot_errors  # noqa: E402
from target_gym.patrol.env import slot_error, wrap_angle  # noqa: E402

TRAIN, HELD_OUT = range(100, 110), range(200, 210)
PATROL = ("patrol", "patrol_range_bearing")
#: AFF's 10 ft one-sigma on the Hornet [AFF]; the lead's altitude, 2 m.
SLOT_RMS_M, ALTITUDE_M = 10.0 * 0.3048, 2.0
_ROLLOUTS = {}


def rollout(name, params, seeds):
    """Per seed, after the scored burn-in: RMS slot error of each wingman,
    the lead's mean altitude and heading errors (formation only) and whether
    it tripped. Jitted once per task and vmapped over the seeds; the gains
    are an argument, so a search does not recompile. A new key every step,
    folded from the seed's, as the values were tuned."""
    if name not in _ROLLOUTS:
        env, p = tg.make(name)
        ctrl = registry.get(name).make_pid_controller()  # gains as an argument

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
    """``params`` with ``field`` set, ``tecs.x`` reaching into TECS's and
    ``a+b`` setting both."""
    for name in field.split("+"):
        if name.startswith("tecs."):
            params = params.replace(tecs=params.tecs.replace(**{name[5:]: value}))
        else:
            params = params.replace(**{name: value})
    return params


def search(label, base, grid, flown, measure="slot"):
    """Every combination of ``grid`` on ``base``, each flown on the tasks of
    ``flown`` (task name to the controller params a candidate makes there),
    ranked by the worst training seed's RMS slot error (``"slot"``) or the
    lead's mean altitude error (``"altitude"``), trips first."""
    rows = []
    for values in itertools.product(*grid.values()):
        gains = base
        for field, value in zip(grid, values):
            gains = _replace(gains, field, value)
        worst, trips = 0.0, 0
        for name, build in flown.items():
            rms, alt, _, trip = rollout(name, build(gains), TRAIN)
            worst = max(worst, (rms if measure == "slot" else alt).max())
            trips += int(trip.sum())
        rows.append((trips, worst, values))
    rows.sort()
    print(f"{label}: best on training seeds, worst {measure} (m), {list(grid)}")
    for trips, worst, values in rows[:5]:
        print(f"  {worst:6.2f}  {trips} trips  {values}")


def _flown(name, seed):
    """One held-out episode of the shipped expert as the tests fly it: the
    host form, and a key split off a chain every step. Returns what
    :func:`rollout` returns for one seed."""
    env, p = tg.make(name)
    expert, step = tg.make_expert(name).host(), jax.jit(env.step_env)
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, p)
    rows, tripped = [], False
    for _ in range(int(p.max_steps_in_episode)):
        key, sub = jax.random.split(key)
        obs, state, _, _, info = step(sub, state, expert(np.asarray(obs)), p)
        tripped = tripped or bool(info["tripped"])
        if name == "formation":
            lead = state.lead
            rows.append(
                (
                    np.asarray(slot_errors(state)),
                    abs(float(lead.target_altitude - lead.z)),
                    abs(float(wrap_angle(lead.psi - lead.target_heading))),
                )
            )
        else:
            rows.append((np.atleast_1d(np.asarray(slot_error(state))), 0.0, 0.0))
    slot, alt, hdg = (np.array(x)[scored_burn_in(name, p) :] for x in zip(*rows))
    return np.sqrt((slot**2).mean(axis=0)), alt.mean(), hdg.mean(), tripped


def check(names=(*PATROL, "formation")):
    """The held-out check, flown as tests/patrol/test_formation_hold.py
    flies it, so it prints what the tests assert."""
    print("held-out seeds 200-209: worst seed (which), median, trips")
    for name in names:
        runs = [_flown(name, s) for s in HELD_OUT]
        rms = np.array([r[0].max() for r in runs])
        worst = HELD_OUT[int(rms.argmax())]
        print(
            f"  {name:20s} slot {rms.max():5.2f} m (seed {worst}), "
            f"{np.median(rms):5.2f} m, limit {SLOT_RMS_M}, "
            f"{sum(r[3] for r in runs)} trips"
        )
    if "formation" not in names:
        return
    alt, hdg = np.array([r[1] for r in runs]), np.array([r[2] for r in runs])
    print(
        f"  {'formation lead':20s} altitude {alt.max():.2f} m (limit {ALTITUDE_M}), "
        f"heading {np.rad2deg(hdg.max()):.3f} deg (limit 0.5)"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="the held-out check only")
    ap.add_argument(
        "--wake-surf",
        action="store_true",
        help="wake_surf's hold alone: the A320's own set, from its shipped values",
    )
    args = ap.parse_args()
    if args.check:
        check()
        return 0
    if args.wake_surf:
        return tune_wake_surf()
    hold = tg.make_expert("patrol").params
    team = tg.make_expert("formation").params
    followers = {
        **{name: lambda g: g for name in PATROL},
        "formation": lambda g: team.replace(wingman=g),
    }
    search(
        "engine and energy",
        hold,
        {
            "tecs.spd_weight": (0.0, 0.5, 1.0),
            "tecs.thr_damp": (0.5, 6.0),
            "k_ref": (0.3, 1.0),
        },
        followers,
    )
    search(
        "slot loops",
        hold,
        {
            "k_x": (0.05, 0.08, 0.12),
            "a_x": (0.2, 0.4),
            "k_y": (0.05, 0.1, 0.2),
            "tau_hdg": (3.0, 5.0),
        },
        followers,
    )
    search(
        "integrals and filter",
        hold,
        {
            "k_xi": (0.0, 0.001, 0.003),
            "k_yi": (0.0, 0.001, 0.003),
            "x_band+y_band": (15.0, 30.0, 60.0),
            "k_omega": (0.1, 0.3, 1.0),
        },
        followers,
    )
    lead = {"formation": lambda g: team.replace(lead=g)}
    search(
        "lead pitch",
        team.lead,
        {"tecs.time_const": (1.5, 2.0)},
        lead,
        measure="altitude",
    )
    check()
    return 0


def tune_wake_surf() -> int:
    """Steps 1 to 3 on ``wake_surf`` alone, from its shipped values: the A320
    3 km in trail is its own follower, with an engine 6 s slow at cruise, so
    the energy step also searches TECS's time constant. Its limit is the
    station box, not the patrol's 10 ft."""
    hold = tg.make_expert("wake_surf").params
    flown = {"wake_surf": lambda g: g}
    search(
        "engine and energy",
        hold,
        {
            "tecs.spd_weight": (0.0, 0.5, 1.0),
            "tecs.thr_damp": (0.5, 2.0, 6.0),
            "tecs.time_const": (5.0, 8.0, 12.0),
            "k_ref": (0.3, 1.0),
        },
        flown,
    )
    search(
        "slot loops",
        hold,
        {
            "k_x": (0.01, 0.02, 0.05),
            "a_x": (0.05, 0.1, 0.2),
            "k_y": (0.02, 0.05, 0.1),
            "tau_hdg": (3.0, 5.0),
        },
        flown,
    )
    search(
        "integrals and filter",
        hold,
        {
            "k_xi": (0.0, 0.001),
            "k_yi": (0.0, 0.001),
            "x_band+y_band": (30.0, 60.0),
            "k_omega": (0.1, 0.3),
        },
        flown,
    )
    check(("wake_surf",))
    return 0


if __name__ == "__main__":
    sys.exit(main())
