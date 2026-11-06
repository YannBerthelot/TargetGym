"""PJM's regulation performance score on the ``battery`` expert, for check 2.

Run with ``uv run python scripts/pjm_regulation_score.py``, on held-out seeds.

The formula is PJM Manual 12, revision 57, section 4.4.5, on 10 s samples, with
30 minute intervals and scoring from 10 minutes in (section 4.4.8). The task has
no regulation assignment, so the mapping is ours, fixed before any score was seen:

- whole signal: the signal is the target, since the task calls all of it the
  regulation signal, and AReg is the pack's ``power_max``;
- movement only: PJM's signal sits around a base point, so this reading takes the
  hourly level out of both sides, with AReg the largest movement in the episode.
"""

from __future__ import annotations

import numpy as np

SAMPLE_S = 10.0  # PSCE collects the signal every 10 seconds
UNSCORED_S = 600.0  # section 4.4.8
INTERVAL_S = 1800.0  # section 4.4.4's 30 minute intervals


def interval_score(response, signal, areg: float) -> float:
    """One interval's score. The manual sets no floor, so it can go negative."""
    denominator = 0.5 * np.mean(np.abs(signal)) + 0.5 * areg
    return float(1.0 - np.mean(np.abs(response - signal) / denominator))


def score(
    response, signal, areg: float, step_s: float, latency: int = 1, skip_s=UNSCORED_S
) -> float:
    """The mean of the interval scores, with ``latency`` samples forgiven."""
    every = max(round(SAMPLE_S / step_s), 1)
    response = np.asarray(response)[::every][round(skip_s / SAMPLE_S) :]
    signal = np.asarray(signal)[::every][round(skip_s / SAMPLE_S) :]
    if latency:
        response, signal = response[latency:], signal[:-latency]
    size = round(INTERVAL_S / SAMPLE_S)
    starts = range(0, len(signal), size)
    return float(
        np.mean(
            [
                interval_score(response[i : i + size], signal[i : i + size], areg)
                for i in starts
            ]
        )
    )


def main() -> None:
    from target_gym import eval as E
    from target_gym.null_policies import HELD_OUT
    from target_gym.registry import REGISTRY
    from target_gym.runners.runners import baseline_policy

    spec = REGISTRY["battery"]
    p = spec.make_test_params()
    record = {
        "power": lambda s, params, tripped: s.power,
        "target": lambda s, params, tripped: s.target_power,
        "regulation": lambda s, params, tripped: s.regulation,
    }
    rows = []
    for seed in HELD_OUT:
        ep = E.run_episode(
            spec, p, baseline_policy(spec, "pid", p), seed=seed, record=record
        )
        # A state holds the next step's target, so the signal in force is one back.
        power = np.asarray(ep.terms["power"], float)[1:]
        target = np.asarray(ep.terms["target"], float)[:-1]
        movement = np.asarray(ep.terms["regulation"], float)[:-1]
        level = target - movement
        areg = float(np.max(np.abs(movement)))
        step = float(p.delta_t)
        rows.append(
            (
                seed,
                score(power, target, float(p.power_max), step),
                score(power, target, float(p.power_max), step, latency=0),
                score(power - level, movement, areg, step),
                score(power - level, movement, areg, step, latency=0),
                areg,
                float(np.mean(np.abs(power - target))),
            )
        )
    print("seed  whole  (no latency)  movement  (no latency)  AReg kW  mean |err| kW")
    for s, a, b, c, d, areg, err in rows:
        print(
            f"{s}  {a:6.3f}  {b:6.3f}  {c:6.3f}  {d:6.3f}  {areg / 1e3:7.1f}  {err / 1e3:6.1f}"
        )
    table = np.array(rows)[:, 1:]
    print("mean ", np.round(table.mean(0), 3), "\nmin  ", np.round(table.min(0), 3))


if __name__ == "__main__":
    main()
