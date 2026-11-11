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


def reference_actions(env, p, ref, seed):
    """The best rung's actions on ``seed``, and its cost. ``ref`` is its
    ``(act, memory)``."""
    act, memory = ref
    obs, s = env.reset_env(jax.random.PRNGKey(seed), p)

    def step(carry, key):
        s, obs, memory = carry
        a, memory = act(memory, obs)
        obs, s, r, _, _ = env.step_env(key, s, a, p)
        return (s, obs, memory), (a, -r)

    keys = jax.random.split(jax.random.PRNGKey(seed + 7), p.max_steps_in_episode)
    _, (actions, cost) = jax.lax.scan(step, (s, obs, memory), keys)
    return actions, jnp.sum(cost)


def plan_cost(env, p, seed, actions, surrogate=False):
    """The cost of a plan from the reset of ``seed``, as the reward charges it.
    With ``surrogate``, also the planner's hinges, which give the timed limits a
    gradient: the band's (``oracle.PLAN_BAND_WEIGHT``) on each step's offset, and
    a step of lost output per K of Tavg above its DNB limit."""
    from target_gym.reactor.oracle import PLAN_BAND_WEIGHT

    _, s = env.reset_env(jax.random.PRNGKey(seed), p)
    keys = jax.random.split(jax.random.PRNGKey(seed + 7), p.max_steps_in_episode)
    per = PLAN_BAND_WEIGHT * 100.0 * 30.0 / 3600.0

    from target_gym.reactor.env import tavg_dnb_limit

    hot_per_K = float(p.failure_cost)  # a step of lost output per K, the plan's

    def step(s, x):
        a, key = x
        _, s, r, _, info = env.step_env(key, s, a, p)
        off = jnp.abs(s.delta_I - s.n * p.ao_ref) - p.ao_band * s.n
        hot = 0.5 * (s.T_coolant + s.T_cold) - tavg_dnb_limit(p)
        hinge = per * jnp.maximum(off, 0.0) + hot_per_K * jnp.maximum(hot, 0.0)
        return s, (-r, info["tripped"], hinge)

    _, (cost, trips, hinge) = jax.lax.scan(step, s, (actions, keys))
    if surrogate:
        return jnp.sum(cost), jnp.sum(trips), jnp.sum(hinge)
    return jnp.sum(cost), jnp.sum(trips)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 106)))
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--report", type=int, default=250)
    parser.add_argument("--tune", default=None, help="a saved ladder run")
    parser.add_argument("--scenario", default=None, help="a name in the suite")
    parser.add_argument("--save", default=None, help="where to save the best plans")
    args = parser.parse_args()

    env = Reactor()
    p = ReactorParams()
    if args.scenario:
        from target_gym import scenarios
        from target_gym.registry import REGISTRY

        (sc,) = [
            s for s in scenarios.suite(REGISTRY["reactor"]) if s.name == args.scenario
        ]
        p = ReactorParams(**sc.overrides)
    ref = ladder.reference(json.load(open(args.tune)), p)
    seeds = jnp.asarray(args.seeds)
    start, reference = jax.jit(
        jax.vmap(lambda s: reference_actions(env, p, ref, s))
    )(seeds)  # fmt: skip

    def loss(plans):
        costs, _, hinge = jax.vmap(lambda s, a: plan_cost(env, p, s, a, True))(
            seeds, plans
        )
        return jnp.sum(costs + hinge), costs

    grad = jax.value_and_grad(loss, has_aux=True)
    opt = optax.adam(optax.cosine_decay_schedule(args.lr, args.iters + 1))

    @jax.jit
    def update(plans, state):
        (_, costs), g = grad(plans)
        g = jnp.nan_to_num(g)
        moves, state = opt.update(g, state)
        return jnp.clip(optax.apply_updates(plans, moves), -1.0, 1.0), state, costs

    plans, state = start, opt.init(start)
    best = np.full(len(args.seeds), np.inf)
    kept = np.asarray(start)
    curve = []
    for i in range(args.iters + 1):
        before = plans
        plans, state, costs = update(plans, state)
        costs = np.asarray(costs)  # of ``before``, the plans the gradient was taken at
        better = costs < best
        kept = np.where(better[:, None, None], np.asarray(before), kept)
        best = np.minimum(best, costs)
        if i % args.report == 0:
            curve.append((i, float(best.mean())))
            print(i, round(float(best.mean()), 1), flush=True)
    if args.save:
        np.save(args.save, kept)
    print(json.dumps({
        "seeds": args.seeds, "iters": args.iters, "scenario": args.scenario,
        "reference": [float(x) for x in reference],
        "ceiling": [float(x) for x in best],
        "room": [float(r / c) for r, c in zip(reference, best)],
        "curve": curve,
    }))  # fmt: skip


if __name__ == "__main__":
    main()
