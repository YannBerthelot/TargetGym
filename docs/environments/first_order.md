# First order

<p align="center"><img src="../../videos/first_order/pid_output.gif" width="480px"/></p>

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((2,))` |
| Tracked variable(s) | x |
| Episode length | 100 steps (5 s at 0.05 s per step) |
| Import | `from target_gym import FirstOrderSystem, FirstOrderParams` |
| Cite as | `first_order-v1` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | input | -1 | 1 |

## Observation space

2 values. Indices (0,) carry the tracked variable(s) that the
reward scores.

## Rewards

See the environment's `compute_reward`.

Every environment in this suite scores on one contract: the reward is
`(tracking terms, multiplied) x (1 - weighted costs)`, bounded in
`[0, 1]`, and reaches 1 only while the target is held exactly. See
[Reward shaping](../reward-shaping.md).

## Starting state

`reset` samples the initial condition and the target; state has 4 fields.

## Episode end

**Termination.** See `check_is_terminal`.

**Truncation.** After 100 steps.

## Baselines

A tuned PID ships with this environment.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 0.05 |
| `max_steps_in_episode` | 100 |
| `K` | 1 |
| `tau` | 0.5 |
| `u_min` | -2 |
| `u_max` | 2 |
| `x_min` | -3 |
| `precision_floor` | 0.006 |
| `x_max` | 3 |

