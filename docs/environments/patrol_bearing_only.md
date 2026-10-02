# Patrol - MARL, bearing-only

<p align="center"><img src="../../videos/patrol_bearing_only/pid_output.gif" width="480px"/></p>

Close-patrol (formation-keeping) environment: state, parameters and transition.

| | |
|---|---|
| Action space | `Box((3,))`, all actions in [-1, 1] |
| Observation space | `Box((21,))` |
| Tracked variable(s) | measured range (m) |
| Episode length | 200 steps (200 s at 1 s per step) |
| Import | `from target_gym import PlanePatrolBearingOnly, PatrolParams` |
| Cite as | `patrol_bearing_only-v3` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | power | -1 | 1 |
| 1 | stick | -1 | 1 |
| 2 | aileron | -1 | 1 |

## Observation space

21 values. Indices (19,) carry the tracked variable(s) that the
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

`reset` samples the initial condition and the target; state has 12 fields.

## Episode end

**Termination.** None. `terminated` is always false, and leaving the operating envelope trips the plant instead. The step that leaves the envelope is charged the trip cost and sets `info["tripped"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 200 steps.

## Baselines

Recorded over 10 seeds of the 200-step episode (see [Baselines](../baselines.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -5.808e+04 | 290.4 |
| MPC | -2845 | 14.22 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

The PID is a lead-state estimator feeding the same pursuit law the full-observation variant uses. Range with azimuth and elevation is a complete relative-position measurement, so the only unobservable quantity is the lead's heading, which the slot needs because it is expressed in the lead's frame; the estimator recovers it by differencing the estimated relative position and filtering. The MPC slot holds patrol's oracle reading the true state, so it is a full-state bound, not a bearing-only controller. On a task defined by what the observation withholds, its NEA measures information and control together: what the hidden lead state is worth plus what a policy leaves on the table. It never reads the observation, so its per-step costs equal patrol's seed for seed (oracle audit, 2026-10).

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 1 |
| `max_steps_in_episode` | 200 |
| `gravity` | 9.81 |
| `initial_mass` | 73500 |
| `thrust_output_at_sea_level` | 240000 |
| `air_density_at_sea_level` | 1.225 |
| `frontal_surface` | 12.6 |
| `wings_surface` | 122.6 |
| `C_x0` | 0.095 |
| `C_z0` | 0.9 |
| `initial_fuel_quantity` | 19088 |
| `specific_fuel_consumption` | 0.0175 |
| `power_response_rate` | 0.05 |
| `stick_response_rate` | 0.9 |
| … | 69 more, see the params dataclass |

