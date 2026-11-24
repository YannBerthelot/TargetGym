# Distillation

<p align="center"><img src="../../videos/distillation/pid_output.gif" width="480px"/></p>

Keep both products of a 40-tray binary column on spec with the reflux and the boilup, using as little boilup as possible. The policy reads the two product analysers, one tray temperature, its own two flows, the specs and the feed flow. The feed composition and the column's interior profile are hidden. Each step is charged, in kmol of boilup, the boilup itself, any product outside its spec, and the light component leaving in the bottoms. Nothing trips.

```python
import target_gym as tg

env, params = tg.make("distillation")
```

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((8,))` |
| Tracked variable(s) | yD (mole fraction), xB (mole fraction) |
| Episode length | 400 steps of 60 s (6.67 h) |
| Cite as | `distillation-v9` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/pc_gym/distillation/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | L_raw | -1 | 1 |
| 1 | V_raw | -1 | 1 |

## Observation space

8 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[0]` against `obs[4]`, `obs[1]` against `obs[5]`.

| # | meaning |
|---|---|
| 0 | `yD`, light-component fraction in the distillate |
| 1 | `xB`, light-component fraction in the bottoms |
| 2 | reflux `L`, as a fraction (0 to 1) of its range |
| 3 | boilup `V`, as a fraction (0 to 1) of its range |
| 4 | distillate spec for `yD` |
| 5 | bottoms spec for `xB` |
| 6 | stage-15 tray temperature (K) |
| 7 | feed flow `F` (kmol/min) |

## Rewards

See the environment's `compute_reward`.

Every environment in this suite scores on one contract. The reward is
minus the sum of non-negative costs: a tracking cost where the task
holds a target, a running cost for what the plant consumes or fails to
deliver, and a failure cost for trips out of the operating envelope.
This task's terms are `boilup`, `off_spec`, `light_lost`, `failure`, reported
each step in `info["reward_terms"]`.
The reward is never positive, and its scale differs
by orders of magnitude between plants. Some plants are
priced in dollars or euros and the rest are dimensionless; the
[per-plant summary](../reward-shaping.md#per-plant-summary) says which.
See [Reward shaping](../reward-shaping.md) for how each cost is built.

## Starting state

`reset` samples the initial condition and the target; state has 20 fields.

## Episode end

**Termination.** None: nothing trips this plant. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 400 steps.

## Baselines

A PID and an MPC ship with this environment, but no comparison has been recorded for them yet.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("distillation", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 1 | min |
| `max_steps_in_episode` | 400 | steps |
| `sensor_noise_scale` | 1 | - |
| `actuator_dead_band` | 0 | fraction of span |
| `actuator_stiction` | 0 | fraction of span |
| `action_noise` | 0 | fraction of span |
| `actuator_rate` | 0 | fraction of span per delta_t unit |
| `actuator_lag` | 0 | delta_t units |
| `alpha` | 1.5 | - |
| `M_tray` | 0.5 | kmol |
| `M_reboiler` | 10 | kmol |
| `M_condenser` | 32.1 | kmol |
| `tau_L` | 0.063 | min |
| `L_nominal` | 2.706 | kmol/min |
| `F` | 1 | kmol/min |
| `zF_nominal` | 0.5 | - |
| `qF` | 1 | - |
| `feed_step` | 0.2 | - |
| `zF_step` | 0.1 | - |
| `event_hold` | 1e+09 | min |
| `L_min` | 1.85 | kmol/min |
| `L_max` | 3.11 | kmol/min |
| `V_min` | 2.18 | kmol/min |
| `V_max` | 3.83 | kmol/min |
| `D_min` | 0.1 | kmol/min |
| `D_max` | 0.9 | kmol/min |
| `yD_spec` | 0.985714 | - |
| `xB_spec` | 0.0142857 | - |
| `precision_floor` | 0.0004 | - |
| `boilup_cost_weight` | 0 | - |
| `time_unit_seconds` | 60 | s |
| `reward_version` | 3 | - |
| `boilup_price` | 1 | kmol |
| `reprocess_price` | 3.206 | kmol/kmol |
| `recovery_price` | 100 | kmol/kmol |
| `failure_cost` | 0 | kmol |
| `restart_steps` | 1 | steps |
| `e_floor_top` | 0.0004 | - |
| `e_floor_bottom` | 0.0004 | - |

