# Glass Furnace

<p align="center"><img src="../../videos/glass_furnace/pid_output.gif" width="480px"/></p>

Keep a float furnace's glass fined and at forming temperature where it leaves for the bath, at the least fuel, by setting the fuel valve. Quality is read only at the cold end, about 18 minutes after the glass leaves; the melt answers a valve move over about two days; and gas quality, the batch blanket and the sulfate walk unseen. Backing off too far from the defect limit costs fuel, too little loses glass.

```python
import target_gym as tg

env, params = tg.make("glass_furnace")
```

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((12,))` |
| Tracked variable(s) | fining share read at the cold end |
| Episode length | 5040 steps of 120 s (7 days) |
| Cite as | `glass_furnace-v4` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/glass_furnace/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | fuel valve | -1 | 1 |

## Observation space

12 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[4]` against `obs[5]`.

| # | meaning |
|---|---|
| 0 | crown couple in bay 3, K |
| 1 | floor couple under bay 3, K |
| 2 | floor couple at the melter's end, K |
| 3 | canal, K |
| 4 | fining share read at the cold end, 0.3 h late |
| 5 | fining pass mark, 0.5 |
| 6 | refiner cooling, share of its capacity |
| 7 | valve after the crown override, kg/s |
| 8 | pull charged, kg/s |
| 9 | pull planned `pull_notice_h` ahead, kg/s |
| 10 | cullet ratio charged |
| 11 | batch line, m, every 30 min |

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

`reset` samples the initial condition and the target; state has 13 fields.

## Episode end

**Termination.** None: nothing trips this plant. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 5040 steps.

## Baselines

No MPC ships with this environment, so there is no recorded comparison. The plant was rebuilt (attempt 3: a float melter in two dimensions, scored in money on fuel and lost glass). The PID is the reference ladder's first rung, a PI on the cold-end fining reading two sigma above its pass mark. The old MPC read the retired plant's params, so the MPC slot is empty; the ceiling is scripts/glass_slow_oracle.py, not yet converged (PHYSICS.md, section 2).

!!! warning "The shipped expert is weak here"
    The fining PI costs 335,604 EUR a 7-day episode, mean of the conformance test's two seeds, against 242,995 for the best constant valve (raw 0.0, 0.5 kg/s). Its setpoint sits two sigma above a pass mark the reward prices as a step, so it loses glass where a wide constant margin does not: full back-off is what this reward pays for (PHYSICS.md, cold review item 4). A finding for the reference ladder; not tuned.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("glass_furnace", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 120 | s |
| `max_steps_in_episode` | 5040 | steps |
| `sensor_noise_scale` | 1 | - |
| `actuator_dead_band` | 0 | fraction of span |
| `actuator_stiction` | 0 | fraction of span |
| `action_noise` | 0 | fraction of span |
| `actuator_rate` | 0 | fraction of span per delta_t unit |
| `actuator_lag` | 0 | delta_t units |
| `gas_price` | 60.5 | EUR/MWh |
| `lost_glass_price` | 600 | EUR/t |
| `pull` | 5.79 | kg/s |
| `cullet` | 0.25 | - |
| `thickness` | 0.2 | m |
| `sulfate` | 0.01 | - |
| `canal_target` | 1373.15 | K |
| `fining_pass` | 0.5 | - |
| `e_floor` | 0.005 | - |
| `crown_high` | 1893.15 | K |
| `fuel_min` | 0.25 | kg/s |
| `fuel_max` | 0.75 | kg/s |
| `override_margin` | 5 | K |
| `override_gain` | 0.0101 | kg/s per K |
| `override_ti` | 7200 | s |
| `cooling_max` | 1.54e+06 | W |
| `cooling_above` | 1 | K |
| `cooling_gain` | 261900 | W/K |
| `cooling_ti` | 7680 | s |
| `pull_change` | 0 | - |
| `pull_ramp_h` | 2 | h |
| `pull_notice_h` | 8 | h |
| `gas_swing` | 0 | - |
| `gas_swing_h` | 6 | h |
| `reward_version` | 2 | - |
| `failure_cost` | 0 | reward |
| `restart_steps` | 0 | steps |

