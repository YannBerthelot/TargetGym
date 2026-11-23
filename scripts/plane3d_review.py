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
half leg (racetrack), on paired seeds; the first row is the baseline.
``polar``: the plant's level-flight fuel flow against CAS, wings level.
``--set name=value`` overrides a float parameter. One process, a few seconds.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from target_gym import scenarios as S
from target_gym.plane import dynamics as D
from target_gym.plane3d import env as E
from target_gym.registry import REGISTRY

KT = 1852.0 / 3600.0


def runner(name):
    """``run(params, seeds, shift, widen, bank)`` -> per-seed sums and extremes."""
    spec = REGISTRY[name]
    env, ctrl = spec.make_env(), spec.make_pid_controller(None)
    steps = int(spec.make_test_params().max_steps_in_episode)
    hold = name.endswith("racetrack")

    def episode(params, seed, shift, widen, bank):
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset_env(key, params)
        gains = ctrl.params.replace(
            max_bank_rad=ctrl.params.max_bank_rad + jnp.deg2rad(bank),
            cruise_cas=ctrl.params.cruise_cas + shift * KT,
        )

        def seen(obs, state):
            if not hold:
                return obs
            assigned = E.assigned_cas_3d(state.target_altitude, params)
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
                fuel=new.fuel,
            )
            return (obs, new, memory), out

        memory = jax.tree.map(jnp.asarray, ctrl.init_state)
        _, o = jax.lax.scan(body, (obs, state, memory), jax.random.split(key, steps))
        n_turn = jnp.maximum(o["turning"].sum(), 1)
        burned = state.fuel - o["fuel"][-1]
        sums = {k: o[k].sum() for k in ("cost", "tracking", "running", "economy")}
        return dict(
            sums,
            trips=o["trips"].sum(),
            burned=burned,
            alt_max=o["alt"].max(),
            path_max=o["path"].max(),
            bank_max=jnp.rad2deg(o["bank"].max()),
            cas=o["cas"].mean() / KT,
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
        r = run(p, jnp.arange(args.seeds), 0.0, 0.0, 0.0)
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
                    r = run(p, seeds, shift, widen, bank)
                    ref = r if ref is None else ref
                    d = 100.0 * (r["burned"] / ref["burned"] - 1.0)
                    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
                    eco = 100.0 * (
                        r["economy"].sum() / max(ref["economy"].sum(), 1e-9) - 1
                    )
                    print(
                        f"  {shift:+4.0f} kt {bank:+3.0f} deg {widen:5.0f} m |"
                        f" CAS {r['cas'].mean():5.1f} | fuel {r['burned'].mean():6.1f} kg"
                        f" {d.mean():+5.2f} % (se {se:.2f}, {d.min():+.2f} to {d.max():+.2f})"
                        f" | economy {eco:+5.2f} % | shipped cost {r['cost'].mean():7.2f}"
                        f" | turning {100 * r['turn_share'].mean():2.0f} %,"
                        f" {r['flow_turn'].mean():.3f} against {r['flow_leg'].mean():.3f} kg/s"
                        f" | path {r['path_max'].max():4.0f} m"
                    )  # fmt: skip


def polar(args):
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
        return p.specific_fuel_consumption * 1e-3 * thrust * 3600.0, thrust

    cas = jnp.arange(170.0, 300.0, 5.0)
    table = jax.jit(jax.vmap(flow, in_axes=(0, None, None)))
    print("CAS, kt", np.asarray(cas).astype(int))
    for mass in (60_000.0, 73_500.0, 78_000.0):
        for z in (3_700.0, 6_500.0):
            kg_h, thrust = (np.asarray(v) for v in table(cas, z, mass))
            i = int(kg_h.argmin())
            print(
                f"{mass / 1000:.1f} t at {z:.0f} m: least {kg_h[i]:.0f} kg/h at"
                f" {float(cas[i]):.0f} kt, L/D {mass * 9.81 / thrust[i]:.1f}"
            )
            print("  kg/h ", kg_h.astype(int))


if __name__ == "__main__":
    floats = lambda text: [float(x) for x in text.split(",")]  # noqa: E731
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("costs", "fuel", "polar"))
    parser.add_argument("task", nargs="?", default="plane3d_racetrack")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--scenarios", type=lambda t: t.split(","), default=[])
    parser.add_argument("--shifts", type=floats, default=[0.0])
    parser.add_argument("--widens", type=floats, default=[0.0])
    parser.add_argument("--banks", type=floats, default=[0.0])
    parser.add_argument("--set", nargs="*", default=[], metavar="name=value")
    arguments = parser.parse_args()
    {"costs": costs, "fuel": fuel, "polar": polar}[arguments.mode](arguments)
