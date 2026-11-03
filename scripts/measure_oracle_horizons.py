"""Measure each oracle's planning horizon by Cooper's DMC recipe.

Run with ``uv run python scripts/measure_oracle_horizons.py [--envs ...]``.
It runs docs/proposals/oracle-horizon-protocol.md on the 14 process and
industrial tasks and writes one file per task,
``src/target_gym/data/oracle_horizons/<task>.json`` (H13). The 10 aircraft
tasks are skipped: their tests fly through ArduPlane 4.0.9's default loops
(decision D6), which PR D builds.

The test (``target_gym.identification.bump_test``), per training seed 100-109:

1. the shipped expert runs one episode from the seed's reset, so the plant
   sits at the operating point it holds (ours: one episode is longer than
   every task's fixed settle);
2. every MV goes to manual at the mean of the expert's last 20 commands, as
   the expert protocols' open-loop tests do (the glass furnace: its last
   reversal cycle, as its expert test);
3. each MV is stepped alone by its C2 step, away from the nearer limit, and
   every CV of H6 is read against the unstepped run for one more episode
   (H8, H9), up to the first trip of either run.

Each response is classed (H10) and fitted (H11); ``target_gym.oracles.horizon``
turns the fits into T_c, N, M and P (H2 to H5). unstable_cstr is measured on
its stabilised plant (section 4.4): the cascade's outer loop in manual, the
inner PD closed, its setpoint stepped 0.5 K down as in the expert test.
"""

from __future__ import annotations

import argparse

import jax.numpy as jnp
import numpy as np

from target_gym import records
from target_gym.identification import TRAIN_SEEDS, bump_test, identify
from target_gym.oracles.horizon import episode_samples, sample_time, task_horizon
from target_gym.registry import REGISTRY


def _s(field):
    return lambda obs, state, p: getattr(state, field)


def _o(index):
    return lambda obs, state, p: obs[index]


def _kiln_lime(obs, state, p):
    from target_gym.cement_kiln.env import discharge_lime

    return discharge_lime(state)


# Per task: the MVs with their C2 steps in raw units (existing steps stay, a
# new one is 0.1, 5 % of [-1, 1]), and the CVs of H6: the tracked outputs,
# then every variable check_is_terminal trips on.
TASKS = {
    "first_order": ({"u": 0.1}, {"x": _s("x")}),
    "cstr": ({"coolant": 0.1}, {"C_a": _s("C_a")}),
    "four_tank": (
        {"pump1": 0.1, "pump2": 0.1},
        {f"h{i}": _s(f"h{i}") for i in range(1, 5)},
    ),
    # +0.1 mL/s on the 10 to 22 mL/s base flow.
    "ph_neutralization": ({"base": 2 * 0.1 / 12.0}, {"pH": _s("pH")}),
    "distillation": (
        {"L": 0.1, "V": 0.1},
        {"yD": lambda o, s, p: s.x[-1], "xB": lambda o, s, p: s.x[0]},
    ),
    "glass_furnace": (
        {"fuel": 0.05},
        {"T_crown": _s("T_crown"), "T_melt": _s("T_melt"), "T_work": _s("T_work")},
    ),
    "reactor": (
        {"rod": 0.1},
        {"n": _s("n"), "T_fuel": _s("T_fuel"), "T_coolant": _s("T_coolant")},
    ),
    "hvac": ({"heating": 0.1}, {"T_air": _s("T_air")}),
    "cement_kiln": (
        {"fuel": 0.1, "speed": 0.1},
        {"lime": _kiln_lime, "T_bz": lambda o, s, p: jnp.max(s.T_solid)},
    ),
    # Firing +0.02 raw; feedwater +2.0 kg/s on 0 to 200 kg/s.
    "boiler_drum": (
        {"fuel": 0.02, "feedwater": 2 * 2.0 / 200.0},
        {"level": _s("level"), "pressure": _s("pressure")},
    ),
    "wind_turbine": ({"pitch": 0.1, "torque": 0.1}, {"P": _o(3), "omega": _s("omega")}),
    "battery": (
        {"power": 0.1},
        {"P": _o(3), "soc": _s("soc"), "V_cell": _o(1), "T_cell": _s("T_cell")},
    ),
    # Speed +1.75 % of rated on 70 to 105 %; recycle +0.05 on 0 to 1.
    "compressor_surge": (
        {"speed": 2 * 0.0175 / 0.35, "recycle": 2 * 0.05 / 1.0},
        {"dp": _s("dp"), "phi_min": _s("phi_min")},
    ),
    "unstable_cstr": ({"T_sp": -0.5}, {"C_a": _s("C_a"), "T": _s("T")}),
}


def _stabilised(p):
    """The unstable_cstr base loop: the expert's inner PD on T, its setpoint
    held at the mean of the cascade's last 20 outputs and stepped as given."""
    from target_gym.pc_gym.unstable_cstr import experts as X

    g = X.load_gains()

    def pd(state, T_prev, T_sp):
        T_c = (
            g["Tc_bias"]
            + g["Kp_T"] * (T_sp[0] - state.T)
            - g["Kd_T"] * (state.T - T_prev) / p.delta_t
        )
        T_c = jnp.clip(T_c, p.T_c_min, p.T_c_max)
        return jnp.reshape(X.raw_from_coolant(T_c, p), (1,)), state.T

    return {
        "policy": pd,
        "memory": lambda state: state.T,
        "start": lambda acts, mems: np.asarray(mems.T_sp_prev)[:, -20:, None].mean(1),
        "direction": lambda u, size: size,
    }


def measure(name: str) -> dict:
    spec = REGISTRY[name]
    n_ep = int(spec.make_test_params().max_steps_in_episode)
    p = spec.make_test_params(max_steps_in_episode=2 * n_ep + 1)
    steps, cvs = TASKS[name]
    mvs, dt = list(steps), float(p.delta_t)
    extra = _stabilised(p) if name == "unstable_cstr" else {}
    if name == "glass_furnace":
        extra["window"] = int(round(float(p.reversal_period) / dt))
    responses = bump_test(
        spec, p, cvs, list(steps.values()), n_settle=n_ep, n_window=n_ep, **extra
    )
    fits = {
        (mvs[j], cv, seed): identify(r)
        for (j, cv), rs in responses.items()
        for seed, r in zip(TRAIN_SEEDS, rs)
    }
    fitted = {key: f for key, f in fits.items() if "model" in f}
    tc, rows, top = task_horizon(fitted, dt)
    fastest = min(fitted, key=lambda key: sample_time(fitted[key]))
    n_samples = episode_samples(n_ep, tc)
    N, M, P = rows[top]
    classes = {}
    for (mv, cv, seed), f in fits.items():
        classes.setdefault(cv, []).append(f["class"])
    at = ("mv", "cv", "seed")
    return {
        "protocol": "docs/proposals/oracle-horizon-protocol.md",
        "dt": dt,
        "episode_steps": n_ep,
        "episode_samples": n_samples,
        "T_c_steps": tc,
        "T_c_seconds": tc * dt,
        "T_c_set_by": dict(zip(at, fastest)),
        "N": N,
        "M": M,
        "P": P,
        "P_steps": P * tc,
        "P_set_by": dict(zip(at, top)),
        "plans_to_episode_end": P >= n_samples,
        "classes": {cv: _verdict(c) for cv, c in classes.items()},
        "test": {
            "seeds": list(TRAIN_SEEDS),
            "settle_steps": n_ep,
            "window_steps": n_ep,
            "steps": steps,
            "hold_window": extra.get("window", 20),
        },
        "fits": [
            {
                **dict(zip(at, key)),
                **f,
                **dict(zip(("N", "M", "P"), rows.get(key, (None,) * 3))),
            }
            for key, f in fits.items()
        ],
    }


def _verdict(found: list[str]) -> str:
    """A CV's class: the one H10 gives every response, else the split."""
    kinds = sorted(set(found))
    if len(kinds) == 1:
        return kinds[0]
    return ", ".join(f"{found.count(k)} {k}" for k in kinds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--envs", nargs="*", default=list(TASKS))
    args = ap.parse_args()
    skipped = sorted(set(REGISTRY) - set(TASKS))
    print(
        f"skipped, {', '.join(skipped)}: their tests fly through ArduPlane "
        "4.0.9's default loops (decision D6), which PR D builds"
    )
    for name in (n for n in args.envs if n in TASKS):
        row = measure(name)
        records.write("oracle_horizons", name, records.stamped(row))
        by, end = row["P_set_by"], row["plans_to_episode_end"]
        print(
            f"{name:18s} T_c {row['T_c_steps']} steps, N {row['N']}, M {row['M']}, "
            f"P {row['P']} ({row['P_steps']} of {row['episode_steps']} steps"
            f"{', to the end' if end else ''}), set by {by['mv']} -> {by['cv']} "
            f"on seed {by['seed']}; classes {row['classes']}"
        )


if __name__ == "__main__":
    main()
