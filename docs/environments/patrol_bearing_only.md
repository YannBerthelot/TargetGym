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
| Cite as | `patrol_bearing_only-v2` |

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

No MPC ships with this environment, so there is no recorded comparison. PID present -- a lead-state estimator feeding the same pursuit law the full-observation variant uses. Range with azimuth and elevation is a complete relative-position measurement, so the only genuinely unobservable quantity is the lead's HEADING, which the commanded slot needs because the slot is expressed in the lead's frame; it is recovered by differencing the estimated relative position and filtering. Measured performance matches the full-observation expert (4 of 8 seeds complete, ~229 m settled slot error vs ~260 m), so the partial observation costs essentially nothing here. No MPC, and the reason is the withheld observation rather than the manoeuvring lead. This note used to blame the lead, on the grounds that an MPC would need its future trajectory as a time-varying parameter. That holds for a CasADi model and not for a gradient planner: `patrol` now ships a GradientMPC that differentiates step_env, and because the lead is scripted and deterministic the plan propagates it for free. What blocks one here is that the planner reads the slot error out of the state, which is precisely what this variant withholds. Handing it the true state anyway would make it an oracle on a task defined by what is hidden, so it needs a planner built on the estimator.

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

