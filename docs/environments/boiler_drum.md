# Boiler Drum

<p align="center"><img src="../../videos/boiler_drum/pid_output.gif" width="480px"/></p>

Hold the drum level and the steam pressure of a 160 MW natural-circulation drum boiler with fuel and feedwater, while the turbine's steam draw moves. Drum level is the classic non-minimum-phase loop: a pressure drop swells the level while water is leaving, and cold feedwater shrinks it before it rises. The level integrates, the riser voidage behind both effects is hidden, and both a high and a low level trip the boiler.

```python
import target_gym as tg

env, params = tg.make("boiler_drum")
```

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((7,))` |
| Tracked variable(s) | drum level (m), drum pressure (bar) |
| Episode length | 400 steps of 2 s (13.3 min) |
| Cite as | `boiler_drum-v2` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/boiler_drum/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | fuel | -1 | 1 |
| 1 | feedwater | -1 | 1 |

## Observation space

7 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[0]` against `obs[5]`, `obs[1]` against `obs[6]`.

| # | meaning |
|---|---|
| 0 | drum level relative to normal water level (m) |
| 1 | drum pressure (bar) |
| 2 | `q_steam`, steam flow to the turbine (kg/s), the disturbance |
| 3 | fuel, percent of maximum |
| 4 | feedwater, percent of maximum |
| 5 | target level, always 0 (m) |
| 6 | target pressure (bar) |

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

`reset` samples the initial condition and the target; state has 11 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. High level carries water to the turbine; low level uncovers the tubes. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 400 steps.

## Baselines

Recorded over 10 seeds of the 400-step episode (see [Baseline results](../baseline-results.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -2.568e+05 | 641.9 |
| MPC | -6124 | 15.31 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("boiler_drum", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 2 | s |
| `max_steps_in_episode` | 400 | steps |
| `reward_version` | 2 | - |
| `e_floor_level` | 0.00267 | m |
| `e_floor_pressure` | 0.05 | bar |
| `e_tol` | 0 | m or bar |
| `tracking_exponent` | 2 | - |
| `c_hold` | 1.566e+08 | W |
| `running_weight` | 1 | - |
| `failure_cost` | 440734 | reward |
| `restart_steps` | 7200 | steps |
| `rho_floor_tracking` | 1.91944 | reward |
| `rho_floor` | 1.91944 | reward |
| `V_t` | 88 | m^3 |
| `V_d` | 40 | m^3 |
| `V_r` | 37 | m^3 |
| `V_dc` | 11 | m^3 |
| `A_d` | 20 | m^2 |
| `A_dc` | 0.355 | m^2 |
| `L_r` | 11 | m |
| `m_metal` | 300000 | kg |
| `C_metal` | 550 | J/(kg K) |
| `k_friction` | 25 | - |
| `tau_sr` | 8 | s |
| `T_d` | 15 | s |
| `f_carry` | 0.1 | - |
| `f_cd` | 0.25 | - |
| `c_pw` | 5700 | J/(kg K) |
| `gravity` | 9.81 | m/s^2 |
| `h_feedwater` | 1.037e+06 | J/kg |
| `Q_max` | 2.4e+08 | W |
| `q_feed_max` | 200 | kg/s |
| `Q_nominal` | 1.6e+08 | W |
| `q_steam_nominal` | 93.35 | kg/s |
| `p_nominal` | 85 | bar |
| `V_wt_nominal` | 58 | m^3 |
| `q_steam_noise_std` | 8 | kg/s |
| `level_band` | 0.1 | m |
| `pressure_band` | 2 | bar |
| `level_trip` | 0.25 | m |
| `pressure_min` | 65 | bar |
| `level_precision_floor` | 0.001 | m |
| `pressure_precision_floor` | 0.05 | bar |
| `pressure_max` | 105 | bar |
| `fuel_weight` | 0 | - |

