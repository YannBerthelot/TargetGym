"""What the reference pays on ``plane3d_heading`` and ``plane3d_racetrack``, and
what simple moves save in fuel: the measurements of the review of 2026-11-23
(plane3d/PHYSICS.md). Nothing here changes a task.

    python scripts/plane3d_review.py costs plane3d_racetrack
    python scripts/plane3d_review.py fuel plane3d_racetrack \
        --scenarios calm,high_hold_wind --shifts 0,-8,-22 --widens 0,600
    python scripts/plane3d_review.py fuel plane3d_heading --banks 0,-2,-5
    python scripts/plane3d_review.py polar

``costs``: the shipped score's terms on every scenario. ``fuel``: the reference
with its speed target moved by ``shift`` kt CAS, its bank limit by ``bank`` deg
(heading) or its guidance read on a pattern ``widen`` m wider, same centre and
half leg (racetrack), on paired seeds; the first row is the baseline. It prints
the task's cost over the settled second half of the window, which is the steady
figure, beside the whole window's, which also counts the speed shed at the start.
``polar``: the plant's level-flight fuel flow against CAS, wings level.
``backoff``: the reference's margin under the maximum holding airspeed.
``--set name=value`` overrides a float parameter. One process, a few seconds.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from target_gym import scenarios as S
from target_gym.plane import dynamics as D
from target_gym.plane3d import env as E
from target_gym.plane3d import hold as H
from target_gym.registry import REGISTRY

KT = 1852.0 / 3600.0


def runner(name):
    """``run(params, seeds, shift, widen, bank, margin)`` -> per-seed sums and
    extremes. ``margin`` (kt) is the reference's ``cap_margin``; negative keeps
    its own."""
    spec = REGISTRY[name]
    env, ctrl = spec.make_env(), spec.make_pid_controller(None)
    steps = int(spec.make_test_params().max_steps_in_episode)
    hold = name.endswith("racetrack")

    def episode(params, seed, shift, widen, bank, margin):
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset_env(key, params)
        gains = ctrl.params.replace(
            max_bank_rad=ctrl.params.max_bank_rad + jnp.deg2rad(bank),
            cruise_cas=ctrl.params.cruise_cas + shift * KT,
            cap_margin=jnp.where(margin < 0.0, ctrl.params.cap_margin, margin * KT),
        )

        def seen(obs, state):
            if not hold:
                return obs
            assigned = env.assigned_cas(state, params)
            wide = state.replace(target_radius=state.target_radius + widen)
            obs = obs.at[22].mul(1.0 + shift * KT / assigned)
            return obs.at[18:21].set(jnp.stack(E.racetrack_guidance(wide)))

        def body(carry, k):
            obs, state, memory = carry
            action, memory = ctrl.step(gains, memory, seen(obs, state), jnp)
            obs, new, reward, _, info = env.step_env(k, state, action, params)
            terms = env.reward_terms(new, params)
            turning = jnp.abs(new.phi) > jnp.deg2rad(10.0)
            out = dict(
                cost=-reward,
                tracking=terms["tracking"],
                running=terms["running"],
                economy=terms.get("economy", jnp.zeros(())),
                trips=info["tripped"],
                turning=turning,
                turn_flow=jnp.where(turning, state.fuel - new.fuel, 0.0),
                alt=jnp.abs(new.target_altitude - new.z),
                path=jnp.abs(E.distance_to_racetrack(new)) if hold else 0.0 * new.z,
                bank=jnp.abs(new.phi),
                cas=D.calibrated_airspeed(new.z, E.airspeed_3d(new, params)),
                cap=H.holding_cap(new.target_altitude, params),
                fuel=new.fuel,
            )
            return (obs, new, memory), out

        memory = jax.tree.map(jnp.asarray, ctrl.init_state)
        _, o = jax.lax.scan(body, (obs, state, memory), jax.random.split(key, steps))
        n_turn = jnp.maximum(o["turning"].sum(), 1)
        burned = state.fuel - o["fuel"][-1]
        sums = {k: o[k].sum() for k in ("cost", "tracking", "running", "economy")}
        # The settled half: a reference told to fly slower sheds its speed in
        # the first minutes, a one-off that a whole-window sum counts as fuel.
        late = jnp.arange(steps) >= steps // 2
        return dict(
            sums,
            trips=o["trips"].sum(),
            burned=burned,
            alt_max=o["alt"].max(),
            path_max=o["path"].max(),
            bank_max=jnp.rad2deg(o["bank"].max()),
            cas=o["cas"].mean() / KT,
            late_cost=jnp.where(late, o["cost"], 0.0).sum(),
            late_cas=jnp.where(late, o["cas"], 0.0).sum() / late.sum() / KT,
            over_cap=(o["cas"] - o["cap"]).max() / KT,
            turn_share=o["turning"].mean(),
            flow_turn=o["turn_flow"].sum() / n_turn,
            flow_leg=(burned - o["turn_flow"].sum())
            / jnp.maximum(steps - o["turning"].sum(), 1),
        )

    run = jax.jit(
        lambda p, seeds, *moves: jax.vmap(lambda s: episode(p, s, *moves))(seeds)
    )
    return spec, lambda *a: {k: np.asarray(v) for k, v in run(*a).items()}


def scenario_params(spec, args):
    extra = dict(item.split("=") for item in args.set)
    extra = {k: float(v) for k, v in extra.items()}
    chosen = [
        s for s in S.suite(spec) if not args.scenarios or s.name in args.scenarios
    ]
    return [(s, S.params_for(spec, s, **extra)) for s in chosen]


def costs(args):
    spec, run = runner(args.task)
    weights, total = S.weights(spec), 0.0
    for s, p in scenario_params(spec, args):
        r = run(p, jnp.arange(args.seeds), 0.0, 0.0, 0.0, -1.0)
        total += weights[s.name] * r["cost"].mean()
        print(f"\n{args.task} / {s.name} (weight {weights[s.name]:.4f})")
        for k in ("cost", "tracking", "running", "economy", "burned", "alt_max",
                  "path_max", "bank_max", "trips"):  # fmt: skip
            print(f"  {k:9s}", r[k])
    print(f"\nweighted cost per episode: {total:.3f}")


def fuel(args):
    spec, run = runner(args.task)
    seeds = jnp.arange(args.seeds)
    for s, p in scenario_params(spec, args):
        print(f"\n{args.task} / {s.name}, {args.seeds} paired seeds")
        ref = None
        for widen in args.widens:
            for bank in args.banks:
                for shift in args.shifts:
                    r = run(p, seeds, shift, widen, bank, -1.0)
                    ref = r if ref is None else ref
                    d = 100.0 * (r["late_cost"] / ref["late_cost"] - 1.0)
                    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
                    whole = 100.0 * (r["cost"].mean() / ref["cost"].mean() - 1.0)
                    print(
                        f"  {shift:+4.0f} kt {bank:+3.0f} deg {widen:5.0f} m |"
                        f" CAS {r['late_cas'].mean():5.1f} | settled half"
                        f" {r['late_cost'].mean():6.1f}, {d.mean():+5.2f} %"
                        f" (se {se:.2f}, {d.min():+.2f} to {d.max():+.2f})"
                        f" | whole window {r['cost'].mean():7.2f}, {whole:+5.2f} %"
                        f" | turning {100 * r['turn_share'].mean():2.0f} %,"
                        f" {r['flow_turn'].mean():.3f} against"
                        f" {r['flow_leg'].mean():.3f} kg/s"
                        f" | path {r['path_max'].max():4.0f} m"
                    )


def backoff(args):
    """The reference's margin under the maximum holding airspeed, by the rule
    for a reference resting on a priced limit (owner, 2026-11-23): on a fixed
    grid of 1 kt, the smallest margin at which it passes the limit on none of
    the nominal seeds, 100 to 109, in the normal day. Flown where the limit
    binds: the low holds, at the scored mass."""
    spec, run = runner("plane3d_racetrack")
    normal = next(s for s in S.suite(spec) if s.name == "normal")
    low = (3_000.0, 14_000.0 * 0.3048)
    p = S.params_for(spec, normal, target_altitude_range=low)
    for margin in range(0, 9):
        r = run(p, jnp.arange(100, 110), 0.0, 0.0, 0.0, float(margin))
        exits = int((r["over_cap"] > 0.0).sum())
        print(
            f"margin {margin} kt: {exits} of 10 seeds pass the cap, the furthest by"
            f" {r['over_cap'].max():+.2f} kt; cost {r['cost'].mean():.2f} a window,"
            f" of which the cap {r['running'].mean():.3f}"
        )
        if exits == 0:
            print(f"kept: {margin} kt")
            break


def polar(args):
    """Level flight, wings level: where the thrust is least (green dot, the
    best lift-to-drag speed) and where the fuel flow is least, which is slower
    since the A320's consumption per newton rises with speed. Each minimum is
    a parabola's through the 1 kt grid about it; VLS is the lowest selectable
    speed, 1.23 to 1.28 times the 1 g stall [SF24], as a CAS."""
    p = REGISTRY["plane3d_racetrack"].make_test_params()

    def flow(cas, z, mass):
        tas = D.scheduled_airspeed(z, cas * KT, 0.99)
        _, _, power = D.trim_level(tas, z, mass, p)
        thrust = D.compute_thrust_output(
            power=power,
            thrust_output_at_sea_level=p.thrust_output_at_sea_level,
            rho=D.compute_air_density_from_altitude(z),
            M=tas / D.compute_speed_of_sound_from_altitude(z),
        )
        return D.fuel_flow(thrust, tas, z, p) * 3600.0, thrust

    def least(cas, values):
        i = int(np.clip(values.argmin(), 4, len(cas) - 5))
        a, b, c = np.polyfit(cas[i - 4 : i + 5], values[i - 4 : i + 5], 2)
        best = -b / (2.0 * a)
        return best, a * best**2 + b * best + c

    cas = np.arange(170.0, 265.0, 1.0)
    table = jax.jit(jax.vmap(flow, in_axes=(0, None, None)))
    for mass in (60_000.0, 73_500.0, 78_000.0):
        stall = np.sqrt(
            2.0 * mass * p.gravity
            / (p.air_density_at_sea_level * p.wings_surface * p.CL_max)
        )  # fmt: skip
        for z in (3_700.0, 6_500.0):
            kg_h, thrust = (np.asarray(v) for v in table(jnp.asarray(cas), z, mass))
            green_dot, least_thrust = least(cas, thrust)
            slow, least_flow = least(cas, kg_h)
            at_green_dot = float(np.interp(green_dot, cas, kg_h))
            rho = float(D.compute_air_density_from_altitude(z))
            vls = [
                float(
                    D.calibrated_airspeed(
                        z, k * stall * np.sqrt(p.air_density_at_sea_level / rho)
                    )
                )
                / KT
                for k in (1.23, 1.28)
            ]
            print(
                f"{mass / 1000:.1f} t at {z:.0f} m: least thrust at {green_dot:.1f} kt"
                f" (L/D {mass * 9.81 / least_thrust:.1f}, {at_green_dot:.0f} kg/h);"
                f" least fuel {least_flow:.0f} kg/h at {slow:.1f} kt,"
                f" {100 * (least_flow / at_green_dot - 1):+.2f} %;"
                f" VLS {vls[0]:.0f} to {vls[1]:.0f} kt"
            )


if __name__ == "__main__":
    floats = lambda text: [float(x) for x in text.split(",")]  # noqa: E731
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("costs", "fuel", "polar", "backoff"))
    parser.add_argument("task", nargs="?", default="plane3d_racetrack")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--scenarios", type=lambda t: t.split(","), default=[])
    parser.add_argument("--shifts", type=floats, default=[0.0])
    parser.add_argument("--widens", type=floats, default=[0.0])
    parser.add_argument("--banks", type=floats, default=[0.0])
    parser.add_argument("--set", nargs="*", default=[], metavar="name=value")
    arguments = parser.parse_args()
    modes = {"costs": costs, "fuel": fuel, "polar": polar, "backoff": backoff}
    modes[arguments.mode](arguments)
