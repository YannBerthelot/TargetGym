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

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 400 steps.

## Baselines

Recorded over 10 seeds of the 400-step episode (see [Baseline results](../baseline-results.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -0.02169 | 5.423e-05 |
| MPC | -0.005915 | 1.479e-05 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

| parameter | default | unit |
|---|---|---|
| `delta_t` | 0.25 | s |
| `max_steps_in_episode` | 400 | steps |
| `R` | 63 | m |
| `rho` | 1.225 | kg/m^3 |
| `J_rotor` | 3.87592e+07 | kg m^2 |
| `J_gen` | 534.116 | kg m^2 |
| `N_gear` | 97 | - |
| `eta_gen` | 0.944 | - |
| `P_rated` | 5e+06 | W |
| `v_rated` | 11.4 | m/s |
| `omega_rated_rpm` | 12.1 | rpm |
| `v_cut_out` | 25 | m/s |
| `cp_c1` | 0.5176 | - |
| `cp_c2` | 116 | - |
| `cp_c3` | 0.4 | 1/deg |
| `cp_c4` | 5 | - |
| `cp_c5` | 21 | - |
| `cp_c6` | 0.0068 | - |
| `pitch_min` | 0 | deg |
| `pitch_max` | 40 | deg |
| `pitch_rate_max` | 8 | deg/s |
| `pitch_tau` | 0.2 | s |
| `torque_max` | 47400 | N m |
| `torque_tau` | 0.1 | s |
| `turbulence_std` | 1.2 | m/s |
| `v_min` | 4 | m/s |
| `v_max` | 28 | m/s |
| `overspeed_factor` | 1.25 | - |
| `underspeed_factor` | 0.4 | - |
| `power_band` | 500000 | W |
| `precision_floor` | 1000 | W |
| `power_envelope` | 5e+06 | W |
| `pitch_activity_weight` | 0.02 | - |
| `reward_version` | 2 | - |
| `e_floor` | 1680 | W |
| `e_tol` | 0 | W |
| `tracking_exponent` | 1 | - |
| `imbalance_price` | 100 | $/MWh |
| `c_hold` | 0.0013 | - |
| `fatigue_weight` | 1 | - |
| `failure_cost` | 0.0972222 | reward |
| `restart_steps` | 2400 | steps |
| `rho_floor_tracking` | 1.00972e-05 | reward |
| `rho_floor` | 1.00972e-05 | reward |

