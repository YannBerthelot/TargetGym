# Compressor Surge

<p align="center"><img src="../../videos/compressor_surge/pid_output.gif" width="480px"/></p>

Hold the header pressure of a variable-speed compressor on a ramped setpoint while the consumers' demand ramps between levels, with the drive speed and an anti-surge recycle valve. Flow below the surge line trips the plant, and neither the reward nor any observation reads the surge margin directly, so the policy has to learn to keep it from the flows it does observe.

```python
import target_gym as tg

env, params = tg.make("compressor_surge")
```

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((8,))` |
| Tracked variable(s) | header pressure (kPa) |
| Episode length | 1200 steps of 0.1 s (2 min) |
| Cite as | `compressor_surge-v5` |
| Status | Scored |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/compressor_surge/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | speed command | -1 | 1 |
| 1 | recycle command | -1 | 1 |

## Observation space

8 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[0]` against `obs[6]`.

| # | meaning |
|---|---|
| 0 | header pressure (kPa) |
| 1 | `m_c`, compressor flow (kg/s) |
| 2 | `m_d`, flow delivered to the consumers (kg/s) |
| 3 | drive speed, percent of rated |
| 4 | drive speed setpoint, percent of rated |
| 5 | recycle valve opening (%) |
| 6 | target header pressure (kPa) |
| 7 | surges counted within the last `surge_window` seconds, as the trip's counter reads |

## Rewards

See the environment's `compute_reward`.

Every environment in this suite scores on one contract. The reward is
minus the sum of non-negative costs: a tracking cost where the task
holds a target, a running cost for what the plant consumes or fails to
deliver, and a failure cost for trips out of the operating envelope.
This task's terms are `energy`, `zone`, `surge`, `failure`, reported
each step in `info["reward_terms"]`.
The reward is never positive, and its scale differs
by orders of magnitude between plants. Some plants are
priced in dollars or euros and the rest are dimensionless; the
[per-plant summary](../reward-shaping.md#per-plant-summary) says which.
See [Reward shaping](../reward-shaping.md) for how each cost is built.

## Starting state

`reset` samples the initial condition and the target; state has 20 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. The counted-surge trip [C5; C3, p. 3]: ``surge_trip_count`` surges within ``surge_window``. A NaN fails the comparison, so a non-finite proposal trips. No trip on the line itself: crossing it is the start of a surge, which is counted when the flow reverses. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 1200 steps.

## Reference and ceiling

What an agent is measured against: the expected cost per step over the task's weighted scenarios, on seeds 100 to 109, lower is better. An agent's result is its saving over the reference, `1 - cost_agent / cost_reference`.

| Line | What it is | Cost per step | Trips |
|---|---|---|---|
| Best null policy | the cheapest of the simple policies | not current | |
| Reference | the controller to beat | not current | |
| MPC | plans with the true model and the schedules operators know, without the disturbances ahead | not current | |
| Ceiling | the best known plan with the true model, the true state and the future: an achieved cost, not a bound | not current | |

Recorded on an earlier version of this task, so not shown: the best null policy, reference, MPC and ceiling lines, which show again once re-recorded. The room shows again with the reference and the ceiling.

The ceiling reads what the policy does not observe: the machine's true surge line, up to 5 % off its map; the consumers' demand levels ahead and their random part. Part of the room may be out of reach of a policy that reads only the observations.

## Baselines

The recorded comparison was measured on an earlier version of this task, so it is not shown; it shows again once re-recorded.

The shipped expert fails its own stability check. It is a PI pair (speed holds the pressure, the recycle valve holds the surge margin) with a backup line and rate action, tuned by its recipe. With the valve's 0.15 s dead time the two loops together grow instead of settling, so the valve swings open and shut in a limit cycle. It keeps its surge margin and does not trip the plant, but it spends recycle energy a settled controller would not: read it as a weak expert, not a tuned one (compressor_surge/PHYSICS.md, section 7).

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("compressor_surge", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 0.1 | s |
| `max_steps_in_episode` | 1200 | steps |
| `sensor_noise_scale` | 1 | - |
| `actuator_dead_band` | 0 | fraction of span |
| `actuator_stiction` | 0 | fraction of span |
| `action_noise` | 0 | fraction of span |
| `actuator_rate` | 0 | fraction of span per delta_t unit |
| `actuator_lag` | 0 | delta_t units |
| `P01` | 101325 | Pa |
| `T01` | 288.15 | K |
| `R_gas` | 287.05 | J/(kg K) |
| `gamma` | 1.4 | - |
| `A_c` | 0.1 | m^2 |
| `L_c` | 4 | m |
| `V_p` | 15 | m^3 |
| `U_r` | 200 | m/s |
| `psi_c0` | 0.3 | - |
| `H` | 0.18 | - |
| `W` | 0.25 | - |
| `phi_join` | 0.78 | - |
| `k_d` | 0.12 | kg/(s Pa^0.5) |
| `dp_out` | 10000 | Pa |
| `k_r` | 0.15 | kg/(s Pa^0.5) |
| `check_width` | 50 | Pa |
| `N_min` | 0.7 | - |
| `N_max` | 1.05 | - |
| `drive_rate` | 0.03 | 1/s |
| `tau_N` | 1 | s |
| `valve_open_rate` | 0.5 | 1/s |
| `valve_lag` | 0.15 | s |
| `valve_close_rate` | 0.1 | 1/s |
| `header_setpoint` | 24000 | Pa |
| `header_band` | 1200 | Pa |
| `setpoint_block_steps` | 300 | steps |
| `setpoint_ramp_s` | 10 | s |
| `demand_block_steps` | 200 | steps |
| `demand_ramp_s` | 5 | s |
| `demand_sigma` | 0.02 | - |
| `demand_theta` | 0.2 | 1/s |
| `demand_dev_reset_clip` | 3 | - |
| `surge_trip_count` | 3 | - |
| `surge_window` | 15 | s |
| `reset_margin` | 0.1 | - |
| `line_shift_max` | 0.05 | - |
| `machine_stratum` | -1 | - |
| `N_reset` | 1 | - |
| `x_reset` | 0.35 | - |
| `reset_newton_iters` | 6 | - |
| `soft_min_temperature` | 0.001 | - |
| `time_unit_seconds` | 1 | s |
| `reward_version` | 3 | - |
| `energy_price` | 1 | 1/kJ |
| `zone_price` | 1000 | kJ/s |
| `surge_price` | 2000 | kJ |
| `failure_cost` | 2200 | kJ |
| `restart_steps` | 9000 | steps |
| `e_floor` | 1.2 | kPa |
| `precision_floor` | 0.0275 | kPa |
| `dp_error_max` | 20.68 | kPa |

