# Wind Turbine

<p align="center"><img src="../../videos/wind_turbine/pid_output.gif" width="480px"/></p>

Wind turbine — NREL 5 MW reference turbine, collective-pitch power regulation.

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((5,))` |
| Tracked variable(s) | electrical power (MW) |
| Episode length | 400 steps (100 s at 0.25 s per step) |
| Import | `from target_gym import WindTurbine, WindTurbineParams` |
| Cite as | `wind_turbine-v2` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | pitch_raw | -1 | 1 |
| 1 | torque_raw | -1 | 1 |

## Observation space

5 values. Indices (3,) carry the tracked variable(s) that the
reward scores.

## Rewards

See the environment's `compute_reward`.

Every environment in this suite scores on one contract. The reward is
minus the sum of three non-negative costs,
`-(tracking_cost + running_cost + failure_cost)`. The running cost
charges consumption and the failure cost charges trips out of the
operating envelope. The reward is never positive, and its scale differs
by orders of magnitude between plants. Some plants are
priced in dollars or euros and the rest are dimensionless; the
[per-plant summary](../reward-shaping.md#per-plant-summary) says which.
See [Reward shaping](../reward-shaping.md) for how each cost is built.

## Starting state

`reset` samples the initial condition and the target; state has 9 fields.

## Episode end

**Termination.** None. `terminated` is always false, and leaving the operating envelope trips the plant instead. The step that leaves the envelope is charged the trip cost and sets `info["tripped"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 400 steps.

## Baselines

Recorded over 10 seeds of the 400-step episode (see [Baselines](../baselines.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -0.01816 | 4.541e-05 |
| MPC | -0.00551 | 1.378e-05 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 0.25 |
| `max_steps_in_episode` | 400 |
| `R` | 63 |
| `rho` | 1.225 |
| `J_rotor` | 3.87592e+07 |
| `J_gen` | 534.116 |
| `N_gear` | 97 |
| `eta_gen` | 0.944 |
| `P_rated` | 5e+06 |
| `v_rated` | 11.4 |
| `omega_rated_rpm` | 12.1 |
| `v_cut_out` | 25 |
| `cp_c1` | 0.5176 |
| `cp_c2` | 116 |
| … | 30 more, see the params dataclass |

