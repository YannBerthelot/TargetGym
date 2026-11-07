"""The reactor's ceiling: a whole-day plan with the true model, state and future.

The shared oracle's horizon is 5 minutes, shorter than the boron line, so it cannot
see the problem. This one plans every action of the day at once by gradient through
the env itself, from the reference controller's own actions, so it can only lower
that cost. The env is deterministic once the seed fixes the wobble, and the cost is
the true one, so a plan's cost on replay is the reward as scored.

A ceiling is a limit from a cheating oracle. It says nothing about what a policy
that sees only the observation could reach.

    uv run python scripts/reactor_foresight.py --seeds 100 101 --iters 2000
"""

import argparse
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import reactor_ladder as ladder  # noqa: E402

from target_gym.reactor import Reactor, ReactorParams  # noqa: E402


def reference_actions(env, p, tune, seed):
    """The best rung's actions on ``seed``, and its cost."""
    obs, s = env.reset_env(jax.random.PRNGKey(seed), p)
    zero = jnp.asarray(0.0)
    memory = (
        jnp.asarray(ladder.PARK), zero, s.boron, zero, zero, jnp.asarray(1.0),
        jnp.asarray(ladder.PARK),
    )  # fmt: skip

    def step(carry, key):
        s, obs, memory = carry
        a, memory = ladder.policy_step2(tune, 5, memory, obs)
        obs, s, r, _, _ = env.step_env(key, s, a, p)
        return (s, obs, memory), (a, -r)

    keys = jax.random.split(jax.random.PRNGKey(seed + 7), p.max_steps_in_episode)
    _, (actions, cost) = jax.lax.scan(step, (s, obs, memory), keys)
    return actions, jnp.sum(cost)


def plan_cost(env, p, seed, actions):
    """The cost of a plan from the reset of ``seed``, as the reward charges it."""
    _, s = env.reset_env(jax.random.PRNGKey(seed), p)
    keys = jax.random.split(jax.random.PRNGKey(seed + 7), p.max_steps_in_episode)

    def step(s, x):
        a, key = x
        _, s, r, _, info = env.step_env(key, s, a, p)
        return s, (-r, info["tripped"])

    _, (cost, trips) = jax.lax.scan(step, s, (actions, keys))
    return jnp.sum(cost), jnp.sum(trips)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 106)))
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--report", type=int, default=250)
    parser.add_argument("--tune", default=None, help="ladder JSON holding tune2")
    args = parser.parse_args()

    env = Reactor()
    p = ReactorParams()
    tune = ladder.Tune2(**json.load(open(args.tune))["tune2"])
    seeds = jnp.asarray(args.seeds)
    start, reference = jax.jit(
        jax.vmap(lambda s: reference_actions(env, p, tune, s))
    )(seeds)  # fmt: skip

    def loss(plans):
        costs, _ = jax.vmap(lambda s, a: plan_cost(env, p, s, a))(seeds, plans)
        return jnp.sum(costs), costs

    grad = jax.value_and_grad(loss, has_aux=True)
    opt = optax.adam(args.lr)

    @jax.jit
    def update(plans, state):
        (_, costs), g = grad(plans)
        g = jnp.nan_to_num(g)
        moves, state = opt.update(g, state)
        return jnp.clip(optax.apply_updates(plans, moves), -1.0, 1.0), state, costs

    plans, state = start, opt.init(start)
    best = np.full(len(args.seeds), np.inf)
    curve = []
    for i in range(args.iters + 1):
        plans, state, costs = update(plans, state)
        best = np.minimum(best, np.asarray(costs))
        if i % args.report == 0:
            curve.append((i, float(best.mean())))
            print(i, round(float(best.mean()), 1), flush=True)
    print(json.dumps({
        "seeds": args.seeds, "iters": args.iters,
        "reference": [float(x) for x in reference],
        "ceiling": [float(x) for x in best],
        "room": [float(r / c) for r, c in zip(reference, best)],
        "curve": curve,
    }))  # fmt: skip


if __name__ == "__main__":
    main()
