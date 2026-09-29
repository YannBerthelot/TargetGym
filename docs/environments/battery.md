# Battery

<p align="center"><img src="../../videos/battery/pid_output.gif" width="480px"/></p>

Grid battery storage — equivalent-circuit Li-ion pack tracking a dispatch signal.

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((5,))` |
| Tracked variable(s) | delivered power (MW) |
| Episode length | 360 steps (1800 s at 5 s per step) |
| Import | `from target_gym import GridBattery, BatteryParams` |
| Cite as | `battery-v2` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | power | -1 | 1 |

## Observation space

5 values. Indices (3,) carry the tracked variable(s) that the
reward scores.

## Rewards

See the environment's `compute_reward`.

Every environment in this suite scores on one contract: the reward is
`(tracking terms, multiplied) x (1 - weighted costs)`, bounded in
`[0, 1]`, and reaches 1 only while the target is held exactly. See
[Reward shaping](../reward-shaping.md).

## Starting state

`reset` samples the initial condition and the target; state has 9 fields.

## Episode end

**Termination.** See `check_is_terminal`.

**Truncation.** After 360 steps.

## Baselines

A tuned PID ships with this environment.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 5 |
| `max_steps_in_episode` | 360 |
| `energy_nominal` | 7.2e+09 |
| `power_max` | 1e+06 |
| `n_series` | 192 |
| `capacity_As` | 9e+06 |
| `R0` | 0.02 |
| `R1` | 0.01 |
| `C1` | 20000 |
| `ocv_a` | 3 |
| `ocv_b` | 1.15 |
| `ocv_c` | 0.3 |
| `ocv_d` | 12 |
| `ocv_e` | 0.05 |
| … | 35 more, see the params dataclass |

