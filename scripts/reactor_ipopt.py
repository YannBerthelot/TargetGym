"""Check the reactor's ceiling with the shared interior-point solver, whole day.

``reactor_foresight.py`` is first-order and did not converge. This solves the same
foresight problem by Newton steps, on the oracle's own multiple-shooting NLP
(``oracles.shooting``, the IPOPT port, exact Hessian), stretched from 5 minutes to
the whole day. The load carries the seed's true wobble, read off a rollout, since the
oracle's planner sees only its mean. The plan is then replayed through ``step_env``,
so the cost reported is the reward as charged.

Seeds are the ones nearest each corner of (hold length, boron delay) among 100 to
109, scaled to their range, picked from the draws and not from any cost: 109 short
and short, 102 short and long, 108 long and short, 100 long and long.

A converged solve is a local optimum of a nonconvex problem, so it is still only an
upper bound on the best cost.

    uv run python scripts/reactor_ipopt.py --seed 100 --tune ladder.json
"""

import argparse
import json
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import reactor_foresight as foresight  # noqa: E402
import reactor_ladder as ladder  # noqa: E402

from target_gym.oracles.controller import enable_x64, float64  # noqa: E402
from target_gym.oracles.shooting import Shooting, StageModel  # noqa: E402
from target_gym.reactor import Reactor, ReactorParams  # noqa: E402
from target_gym.reactor.oracle import reactor_model  # noqa: E402

WIDTH = 0.0010526315789473686  # the rod corner's, data/oracle_widths/reactor.json
SCALE = 0.10526315789473685
BORON_PER_ROD = 1.0 / SCALE  # the boron flow's range over the rod reach


def wobble_path(env, p, seed, actions):
    """The wobble at the start of every control step. It takes no input from the
    plant, so any rollout on the seed's keys gives it."""
    _, s = env.reset_env(jax.random.PRNGKey(seed), p)
    keys = jax.random.split(jax.random.PRNGKey(seed + 7), p.max_steps_in_episode)

    def step(s, x):
        a, key = x
        before = s.load_wobble
        _, s, _, _, _ = env.step_env(key, s, a, p)
        return s, before

    return jax.lax.scan(step, s, (actions, keys))[1]


def build(env, p, steps, block, max_iter):
    """The whole-day NLP. ``reactor_step`` decays the wobble from the solve's start,
    so each step's start value is undone to the same decay."""
    model = reactor_model(env, p)
    decay = 1.0 - p.demand_theta * p.delta_t

    def step(d, x, u, t):
        start, down, up, _, delay = d["task"]
        wobble = d["path"][t] * decay ** (-1.0 * t * env.control_period)
        here = {**d, "task": (start, down, up, wobble, delay)}
        return model.step(here, x, u, t)[:3]

    stage = StageModel(
        step=step, block=block, lo=(-1.0, -1.0), hi=(1.0, 1.0), hinges=model.hinges,
        limits=model.limits,
        move_rows=lambda x, u: model.move_rows(x, u, block),
        move_scale=model.move_scale(block),
        tie=model.tie_budget / (model.nu * (steps // block)),
    )  # fmt: skip
    return model, Shooting(stage, steps // block, model.nx, max_iter=max_iter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tune", required=True, help="ladder JSON holding tune2")
    parser.add_argument("--steps", type=int, default=2880)
    parser.add_argument("--block", type=int, default=1, help="env steps a move holds")
    parser.add_argument("--max-iter", type=int, default=600)
    args = parser.parse_args()

    env, p = Reactor(), ReactorParams()
    tune = ladder.Tune2(**json.load(open(args.tune))["tune2"])
    reference, ref_cost = foresight.reference_actions(env, p, tune, args.seed)
    path = wobble_path(env, p, args.seed, reference)
    _, state = env.reset_env(jax.random.PRNGKey(args.seed), p)

    plans, widths = None, [SCALE * 0.1, SCALE * 0.01, WIDTH]
    with enable_x64():
        q, st = float64(p), float64(state)
        model, shooting = build(env, q, args.steps, args.block, args.max_iter)
        data = {
            "task": model.data(st), "end": args.steps, "path": jnp.asarray(path, float),
        }  # fmt: skip
        x0, hold = model.state_x(st), model.hold(st)
        for width in dict.fromkeys(widths):
            data["widths"] = jnp.asarray([width])
            data["boron_width"] = jnp.asarray(width * BORON_PER_ROD)
            t0 = time.time()
            if plans is None:
                plans = shooting.solve(data, x0, hold)
            else:
                plans = shooting.resolve(data, x0, hold, plans)
            jax.block_until_ready(plans.z)
            c = plans.certificate
            print(json.dumps({
                "width": width, "seconds": round(time.time() - t0, 1),
                "iterations": int(c.iterations), "kkt_error": float(c.error),
                "capped": bool(c.capped), "failed": bool(c.failed),
                "violation": float(c.violation), "planned_cost": float(plans.cost),
            }), flush=True)  # fmt: skip
        moves = np.asarray(shooting.moves(plans))

    held = np.repeat(moves, args.block, axis=0)[: args.steps]
    full = np.array(reference)
    full[: len(held)] = held
    cost, trips = foresight.plan_cost(env, p, args.seed, jnp.asarray(full, jnp.float32))
    print(json.dumps({
        "seed": args.seed, "steps": args.steps, "block": args.block,
        "reference": float(ref_cost), "ipopt": float(cost), "trips": int(trips),
        "error": float(c.error), "capped": bool(c.capped), "failed": bool(c.failed),
    }))  # fmt: skip
    np.save(f"plan_{args.seed}_{args.steps}_{args.block}.npy", full)


if __name__ == "__main__":
    main()
