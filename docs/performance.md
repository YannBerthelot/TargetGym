# Throughput

Steps per second for a vmapped, jitted, scanned rollout, the way an RL loop
drives these environments: gymnax's `step` with its auto-reset, on each task's
scored configuration, with a key of its own for every environment and step, a
random action inside the action space, and everything `step` returns summed, so
the compiler cannot drop any of it. Each rate is the best of three timed runs
after the one that compiles. Reproduce with:

```
uv run python -m target_gym.benchmark_speed --batch 4096 --steps 500
uv run python -m target_gym.benchmark_speed --batch 256
```

Measured on 2026-11-28 on one machine: 2 x Intel Xeon Gold 6134 at 3.20 GHz,
Ubuntu 22.04, Python 3.13, JAX 0.6.2 on CPU. Each run was one process pinned to
two cores of one socket, at a 1-minute load average between 1.5 and 5.7 on its 32 CPUs.

**These are not properties of the plants.** They move with the machine, the
batch size and the JAX version, which is why they are measured here rather than
claimed in the eighteen `PHYSICS.md` contracts, where every other number is a
fact about the process being modelled. Nothing in CI checks them, because a
check that fails on somebody else's laptop is worse than no check.

Thousands of steps per second:

| environment | batch 4096 | batch 256 |
| --- | --- | --- |
| `first_order` | 2460 | 1450 |
| `unstable_cstr` | 1720 | 1090 |
| `cstr` | 1210 | 767 |
| `boiler_drum` | 1070 | 562 |
| `four_tank` | 905 | 511 |
| `battery` | 692 | 416 |
| `hvac` | 324 | 263 |
| `ph_neutralization` | 212 | 182 |
| `cement_kiln` | 226 | 130 |
| `reactor` | 235 | 127 |
| `wind_turbine` | 102 | 94.6 |
| `glass_furnace` | 59.4 | 54.0 |
| `distillation` | 50.4 | 29.4 |
| `plane_energy` | 30.9 | 25.3 |
| `plane` | 30.9 | 24.8 |
| `plane_sine` | 30.8 | 24.5 |
| `plane3d_heading` | 29.0 | 23.4 |
| `plane3d_circle` | 28.7 | 23.2 |
| `plane3d_racetrack` | 27.6 | 22.2 |
| `compressor_surge` | 58.0 | 18.7 |
| `plane3d_figure8` | 16.8 | 16.3 |
| `patrol` | 18.3 | 14.5 |
| `patrol_range_bearing` | 18.3 | 14.1 |
| `wake_surf` | 18.0 | 13.4 |
| `formation` | 9.3 | 7.4 |

The table this page carried before 0.8.0 timed a rollout that kept only the
state, with one key and zero actions, so the compiler could drop the
observation and the reward. On this machine, at batch 256, that overstated the
rate 1.6 times on `compressor_surge`, 1.7 on the `reactor`, 1.8 on
`ph_neutralization`, 5.2 on `cstr` and 18 to 19 on `hvac` (two runs). Its
numbers do not compare with these.

Two runs of one environment differed by up to about 12 % (`hvac`, 233 and 263
thousand at batch 256), so read the table to two figures.

## The auto-reset

gymnax's `step` computes a fresh reset on every step and keeps it where an
episode ended, so the reset is paid on every step, not once an episode. On the
scored tasks at batch 256, `step_env` alone, with no reset and its observation,
reward and termination kept, ran 1.4 times (`reactor`) to 2.9 times (`hvac`)
faster than `step`.

## What this means for RL

The [RL protocol](rl-protocol.md)'s largest sample budget is 1e7 environment
steps. At batch 256 on these two cores, that is about 7 seconds of environment
time on `first_order` and 23 minutes on `formation`. On the scored tasks it is
about 40 seconds on `hvac`, 55 on `ph_neutralization`, 80 on the `reactor` and
9 minutes on `compressor_surge`. Whether the network or the environment costs
more depends on the network and the hardware; these figures are the
environment's side of it.

The spread is arithmetic per step:

- Every aircraft flies four steps of its flight-control loop per environment
  step: the loop closes at 4 Hz, around a short period of about 2.5 s. With
  one aircraft a task runs at 22 to 25 thousand steps per second at batch 256
  (16 thousand on the figure eight), with two (`patrol`, `wake_surf`) at about
  14 thousand and with the formation's three at about 7 thousand.
- `compressor_surge` takes 10 RK4 substeps a step, each with an update of the
  actuators' rate limits.
- The distillation column integrates 41 stages through 16 RK4 substeps, and its
  contract shows that 16 is a stability requirement, not a refinement.
- The fastest, `first_order`, `cstr` and `unstable_cstr`, carry one to a few
  states and take one to four RK4 substeps a step.

Every environment ran faster at batch 4096 than at 256, from barely
(`plane3d_figure8`, within the spread of two runs) to 3.1 times
(`compressor_surge`), so the order moves with the batch: `compressor_surge` is
among the slowest at 256 and mid-table at 4096. `formation` is the slowest at
both.
