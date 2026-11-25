"""Glass room study: what the readings know of the furnace's state. The library's planner
without foresight is handed the true state. Here a particle filter stands in for it: the
plant's own model run on many sampled paths of the walks, each weighed by how well its
readings match the ones the agent is given, and resampled. No particle sees the true
state or an episode's draws: it starts from the task's reset law and steps on keys of
its own.

It is run along the library planner's valve path on a normal week (no event to infer).
At each knot two forecasts of the fining share over the next 16 h, under the valve that
was then fired, are set against what came out: the mean path from the true state, which
is the planner's forecast, and the mean of the mean paths from 16 particles. The second
one's error is the spread a planner on the readings alone would have to price.

Ours, all of it a filter's tuning: 128 particles; the couples weighed at 1 K, three
times their scatter, so that 128 particles keep their weights; the batch line at 0.5 m;
resampling when the effective count falls under half. Pull and cullet are read exactly,
so every particle takes them.

Beside the forecasts, at each knot: how far the particles' weighted mean is from the
true hidden walks, from the temperature of the blocks within 50 K of their release
temperature and of all blocks, from the bath's mean temperature and from the fining
share now at the canal.

Run: [PARTICLES=128] [DROP=4,12] PYTHONPATH=src python scripts/glass_room/state_filter.py <lib_dir> <out.json> [seed ...]
"""

import json
import os
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402

from target_gym.experts.mpc import plan_params  # noqa: E402
from target_gym.glass_furnace import oracle as O  # noqa: E402

E, P = L.E, L.P
N, SHOWN = int(os.environ.get("PARTICLES", 128)), 16
KNOT, LEAD = 4 * L.HOUR, 16 * L.HOUR
#: Channels weighed, and the scatter each is weighed at: crown, two floor couples and
#: the canal (K), the fining share, the batch line (m), the defects per plate.
CHANNELS = np.array([0, 1, 2, 3, 4, 11, 12])
SCATTER = np.array([1.0, 1.0, 1.0, 1.0, E.FINING_SCATTER, 0.5, E.FINING_SCATTER])
#: ``DROP=4,12`` leaves channels out of the weights: what the rest know without them.
_KEPT = ~np.isin(CHANNELS, [int(c) for c in os.environ.get("DROP", "").split(",") if c])
CHANNELS, SCATTER = CHANNELS[_KEPT], SCATTER[_KEPT]
PULL, CULLET = E.WALKS.index("pull"), E.WALKS.index("cullet")
LEADS = np.array([4, 8, 12, 16]) * L.HOUR - 1
#: The filter's own key: no episode runs on it.
OWN = jax.random.fold_in(jax.random.PRNGKey(0), 2)


class Filter:
    """A particle filter on the plant itself, under the scenario's params ``p``: ``n``
    copies of the plant on sampled paths of the walks, weighed by their readings. A
    carry is ``(particles, log weights, key)``."""

    def __init__(self, p, n=N):
        self.p, self.n = p, n
        quiet = p.replace(sensor_noise_scale=0.0)  # a particle's readings: no noise

        def one(carry, x):
            states, logw, key = carry
            u, y = x
            key, k_step, k_pick = jax.random.split(key, 3)
            step = lambda k, s: L.ENV.step_env(k, s, E.to_action(u, p), p)[
                1
            ]  # noqa: E731
            states = jax.vmap(step)(jax.random.split(k_step, n), states)
            planned = jax.vmap(
                lambda s: E.planned_pull(E.event_hours(s.time, s, p), s, p)
            )(states)
            walk = states.walk.at[:, PULL].set(y[8] / planned - 1.0)
            walk = walk.at[:, CULLET].set(y[10] - p.cullet)
            states = states.replace(
                walk=walk, pull=jnp.full(n, y[8]), cullet=jnp.full(n, y[10])
            )
            read = jax.vmap(lambda s: L.ENV.get_obs(s, quiet))(states)
            miss = (y[CHANNELS] - read[:, CHANNELS]) / SCATTER
            logw = logw - 0.5 * jnp.sum(miss**2, axis=1)
            w = jax.nn.softmax(logw)
            count = 1.0 / jnp.sum(w**2)
            at = (jax.random.uniform(k_pick) + jnp.arange(n)) / n
            pick = jnp.clip(jnp.searchsorted(jnp.cumsum(w), at), 0, n - 1)
            again = count < n / 2
            states = jax.tree.map(lambda a: jnp.where(again, a[pick], a), states)
            logw = jnp.where(again, jnp.zeros(n), logw)
            return (states, logw, key), count

        def shown(carry, key, k):  # k particles drawn by weight
            states, logw, _ = carry
            at = (jax.random.uniform(key) + jnp.arange(k)) / k
            pick = jnp.searchsorted(jnp.cumsum(jax.nn.softmax(logw)), at)
            return jax.tree.map(lambda a: a[jnp.clip(pick, 0, n - 1)], states)

        def averaged(carry):  # one state: the particles' weighted mean
            states, logw, _ = carry
            w = jax.nn.softmax(logw)
            mean = lambda a: jnp.tensordot(w, a, 1).astype(a.dtype)  # noqa: E731
            whole = lambda a: jnp.issubdtype(a.dtype, jnp.floating)  # noqa: E731
            return jax.tree.map(lambda a: mean(a) if whole(a) else a[0], states)

        #: One step: the valve asked (kg/s) and the reading that came after it.
        self.step = jax.jit(lambda carry, u, y: one(carry, (u, y)))
        #: A run of steps: their valves and their readings.
        self.knot = jax.jit(lambda c, valve, obs: jax.lax.scan(one, c, (valve, obs)))
        self.shown = jax.jit(shown, static_argnums=2)
        self.averaged = jax.jit(averaged)

    def start(self, key, salt=0):
        """Particles drawn from the task's reset law on ``key``, which no episode
        runs on."""
        reset = lambda k: L.ENV.reset_env(k, self.p)[1]  # noqa: E731
        parts = jax.vmap(reset)(jax.random.split(key, self.n))
        return parts, jnp.zeros(self.n), jax.random.fold_in(key, salt)


def main():
    lib_dir, out = sys.argv[1:3]
    seeds = [int(a) for a in sys.argv[3:]] or [100, 101, 102]
    p = L.scenario_params("normal")
    mean_p = plan_params(L.SPEC, p).replace(walk_sd=(0.0,) * len(E.WALKS))
    n = int(p.max_steps_in_episode)
    told_keys = jax.random.split(jax.random.PRNGKey(0), LEAD)
    F = Filter(p, N)

    @jax.jit
    def truth(state, keys, valve):  # a knot of the episode: what the agent reads
        def f(s, x):
            k, u = x
            o, s1, _, _, _ = L.ENV.step_env(k, s, E.to_action(u, p), p)
            return s1, (o, P.sensors(s1.plant, p.grid, p.plant)["fining"], s1.walk)

        return jax.lax.scan(f, state, (keys, valve))

    forecast = jax.jit(lambda s, v: O._scan(L.ENV, mean_p, s, told_keys, v)[1][2])
    walks_p = plan_params(L.SPEC, p)  # the walks on, as a path that may come

    @jax.jit
    def may_come(some, key, v):  # each shown particle on a sampled path of its own
        keys = jax.vmap(lambda k: jax.random.split(k, LEAD))(
            jax.random.split(key, SHOWN)
        )
        path = lambda s, ks: O._scan(L.ENV, walks_p, s, ks, v)[1][2]  # noqa: E731
        return jax.vmap(path)(some, keys)

    fined = lambda s: P.sensors(s.plant, p.grid, p.plant)["fining"]  # noqa: E731
    fined_each = jax.jit(jax.vmap(fined))
    forecasts = jax.jit(jax.vmap(forecast, in_axes=(0, None)))

    res = {}
    for seed in seeds:
        valve = jnp.asarray(
            np.load(f"{lib_dir}/normal_{seed}.npz")["valve"], jnp.float32
        )
        k_reset, keys = L.episode_keys(seed, n)
        _, state = L.ENV.reset_env(k_reset, p)
        carry = F.start(OWN, seed)
        fining, rows = [], []
        for t in range(0, n, KNOT):
            if t + LEAD <= n:  # both forecasts, before the knot is played
                v = valve[t : t + LEAD]
                some = F.shown(carry, jax.random.fold_in(carry[2], t), SHOWN)
                each = np.asarray(forecasts(some, v))
                one_state = np.asarray(forecast(F.averaged(carry), v))
                paths = np.asarray(
                    may_come(some, jax.random.fold_in(carry[2], n + t), v)
                )
                priced = np.sqrt(((paths - one_state) ** 2).mean(0))
                w = np.asarray(jax.nn.softmax(carry[1]))
                hid = np.asarray(carry[0].walk)
                block = np.asarray(state.plant.block)
                off = np.tensordot(w, np.asarray(carry[0].plant.block), 1) - block
                hot = block >= float(p.plant.block_release_T) - 50.0
                bath = np.tensordot(w, np.asarray(carry[0].plant.T), 1).mean()
                rows.append(dict(
                    step=t, told=np.asarray(forecast(state, v))[LEADS],
                    read=each.mean(0)[LEADS], among=each.std(0)[LEADS],
                    mean=one_state[LEADS], priced=priced[LEADS],
                    walk=np.asarray(state.walk), walk_read=w @ hid,
                    walk_among=np.sqrt(w @ (hid - w @ hid) ** 2),
                    state=np.array([
                        np.sqrt((off[hot] ** 2).mean()) if hot.any() else 0.0,
                        np.sqrt((off**2).mean()),
                        bath - float(np.asarray(state.plant.T).mean()),
                        float(w @ np.asarray(fined_each(carry[0])) - fined(state)),
                    ]),
                ))  # fmt: skip
            sl = slice(t, t + KNOT)
            state, (obs, f, _) = truth(state, keys[sl], valve[sl])
            carry, count = F.knot(carry, valve[sl], obs)
            fining.append(np.asarray(f))
            print(f"seed {seed} step {t}: fewest effective particles {float(count.min()):.1f}", flush=True)  # fmt: skip
        fining = np.concatenate(fining)
        came = np.array([fining[r["step"] + LEADS] for r in rows])
        off = {
            k: came - np.array([r[k] for r in rows]) for k in ("told", "read", "mean")
        }
        res[seed] = dict(
            off_true=off["told"].tolist(),
            off_read=off["read"].tolist(),
            off_mean=off["mean"].tolist(),
            priced=np.array([r["priced"] for r in rows]).tolist(),
            among=np.array([r["among"] for r in rows]).tolist(),
            walk=np.array([r["walk"] for r in rows]).tolist(),
            walk_read=np.array([r["walk_read"] for r in rows]).tolist(),
            walk_among=np.array([r["walk_among"] for r in rows]).tolist(),
            state=np.array([r["state"] for r in rows]).tolist(),
        )
        rms = {k: np.sqrt((v**2).mean(0)) for k, v in off.items()}
        # past the first day: the filter starts from the reset law, not from readings
        late = slice(6, None)
        miss = np.sqrt(((np.array(res[seed]["walk"]) - np.array(res[seed]["walk_read"]))[late] ** 2).mean(0))  # fmt: skip
        st = np.sqrt((np.array(res[seed]["state"])[late] ** 2).mean(0))
        print(
            f"seed {seed}: forecast error at 4, 8, 12, 16 h from the true state"
            f" {np.round(rms['told'], 5).tolist()}, from the readings"
            f" {np.round(rms['read'], 5).tolist()}; spread among the particles"
            f" {np.round(np.mean(res[seed]['among'], 0), 5).tolist()}; past the first"
            f" day, walks missed by {dict(zip(E.WALKS, np.round(miss, 4).tolist()))},"
            f" hot blocks by {st[0]:.2f} K, all blocks by {st[1]:.2f} K, the bath's"
            f" mean by {st[2]:.2f} K, the fining now by {st[3]:.5f}",
            flush=True,
        )
        pr = np.array(res[seed]["priced"])
        z = off["mean"] / pr
        print(
            f"seed {seed}: from the averaged state {np.round(rms['mean'], 5).tolist()};"
            f" the spread it would price {np.round(np.sqrt((pr**2).mean(0)), 5).tolist()};"
            f" error over that spread, rms {np.round(np.sqrt((z**2).mean(0)), 2).tolist()},"
            f" lowest {np.round(z.min(0), 2).tolist()}",
            flush=True,
        )
        json.dump(res, open(out, "w"), indent=1)
    both = {k: np.concatenate([np.array(r[k]) for r in res.values()]) for k in ("off_true", "off_read", "off_mean")}  # fmt: skip
    print("all seeds:", {k: np.round(np.sqrt((v**2).mean(0)), 5).tolist() for k, v in both.items()}, flush=True)  # fmt: skip
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
