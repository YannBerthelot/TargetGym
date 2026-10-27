# Baseline results

<!-- Written by scripts/generate_baseline_table.py from
     src/target_gym/data/protocol_results.json and baseline_returns.json.
     Do not edit by hand. -->

Every environment ships two reference controllers, scored on its test
configuration (`spec.make_test_params()`):

- **The expert** (the `"pid"` slot): reads the observation only, as your agent
  does. It is the score to beat.
- **The oracle** (the `"mpc"` slot): reads the true state, including what the
  observation hides. It shows what information and planning are worth.

Costs on this page are per step, lower is better, in each environment's own
units. To score your own policy, see
[Evaluating against it](using-the-pids.md#evaluating-against-it).
How the controllers are built and measured is in
[How the baselines work](baselines.md).

## Room to grow

**Hold** is the tracking cost per step once a controller has settled on
each target, after a settle fixed per environment. The **ceiling** is
tracking every quantity within its instrument's resolution: as good as a
real plant can verify. **Room** is the number of decades from the expert's
hold down to the ceiling; 0 means the expert is already there. The
**oracle score** places the oracle between the expert (0) and the ceiling
(1). Score a policy the same way with `target_gym.scoring.score_policy`.
3 seeds each.

| environment | expert hold | ceiling | room (decades) | oracle hold | oracle score |
| --- | --- | --- | --- | --- | --- |
| **Aircraft** | | | | | |
| [`plane`](environments/plane.md) | 6.04 | 1 | 0.78 | 0.422 | 1.00 |
| [`plane_energy`](environments/plane_energy.md) | 1,778 | 0.694 | 3.41 | 55.1 | 0.44 |
| [`plane_sine`](environments/plane_sine.md) | 1,522 | 0.63 | 3.38 | 0.29 | 1.00 |
| [`plane3d_heading`](environments/plane3d_heading.md) | 19,023 | 2 | 3.98 | 0.726 | 1.00 |
| [`plane3d_circle`](environments/plane3d_circle.md) | 832 | 2 | 2.62 | 1.93 | 1.00 |
| [`plane3d_racetrack`](environments/plane3d_racetrack.md) | 1.35e6 | 2 | 5.83 | 1.07 | 1.00 |
| [`plane3d_figure8`](environments/plane3d_figure8.md) | 5.92e5 | 1 | 5.77 | 0.77 | 1.00 |
| [`patrol`](environments/patrol.md) | 265 | 2 | 2.12 | 0.0371 | 1.00 |
| [`patrol_bearing_only`](environments/patrol_bearing_only.md) | 199 | 2 | 2.00 | 0.0371 | 1.00 |
| [`formation`](environments/formation.md) | 23,239 | 6 | 3.59 | 1.19 | 1.00 |
| **Process Control** | | | | | |
| [`cstr`](environments/cstr.md) | 0.538 | 1 | 0.00 | 4.5e-9 | — |
| [`first_order`](environments/first_order.md) | 2.91e-7 | 1 | 0.00 | 0 | — |
| [`four_tank`](environments/four_tank.md) | 30.3 | 2 | 1.18 | 3.08e-8 | 1.00 |
| [`ph_neutralization`](environments/ph_neutralization.md) | 12.6 | 1 | 1.10 | 0.089 | 1.00 |
| [`distillation`](environments/distillation.md) | 62.6 | 2 | 1.50 | 2.61e-4 | 1.00 |
| [`unstable_cstr`](environments/unstable_cstr.md) | 2.82 | 1 | 0.45 | 0.00686 | 1.00 |
| **Industrial / Energy** | | | | | |
| [`glass_furnace`](environments/glass_furnace.md) | 2.02 | 1 | 0.30 | 0.354 | 1.00 |
| [`reactor`](environments/reactor.md) | 23.3 | 0.00278 | 3.92 | 1.38 | 0.31 |
| [`hvac`](environments/hvac.md) | 0.00982 | 7.5e-5 | 2.12 | 0.00522 | 0.13 |
| [`cement_kiln`](environments/cement_kiln.md) | 4.68 | 1 | 0.67 | 0.0295 | 1.00 |
| [`boiler_drum`](environments/boiler_drum.md) | 524 | 1.14 | 2.66 | 4.62 | 0.77 |
| [`compressor_surge`](environments/compressor_surge.md) | 11.4 | 1 | 1.06 | 0.00677 | 1.00 |
| **Renewable Energy** | | | | | |
| [`wind_turbine`](environments/wind_turbine.md) | 2.43e-5 | 6.94e-6 | 0.54 | 1.27e-5 | 0.52 |
| [`battery`](environments/battery.md) | 7.93e-4 | 1.39e-4 | 0.76 | 2.14e-4 | 0.75 |

## Whole episode

The **gain** is the mean cost per step after the plant's burn-in, moves
between targets included: what the plant pays. **Oracle vs expert** is
the expert's gain divided by the oracle's, so 10 means the oracle pays a
tenth. The **transient** is the summed cost around each target change.

| environment | expert gain | oracle gain | oracle vs expert | expert transient | oracle transient |
| --- | --- | --- | --- | --- | --- |
| **Aircraft** | | | | | |
| [`plane`](environments/plane.md) | 6.21 | 0.516 | 12 | 2.73e6 | 1.19e6 |
| [`plane_energy`](environments/plane_energy.md) | 11,210 | 511 | 21.9 | 1.84e6 | 1.81e5 |
| [`plane_sine`](environments/plane_sine.md) | 1,522 | 0.546 | 2,788 | 2.14e6 | 7.61e5 |
| [`plane3d_heading`](environments/plane3d_heading.md) | 19,023 | 0.726 | 26,193 | 8.86e6 | 2.01e6 |
| [`plane3d_circle`](environments/plane3d_circle.md) | 832 | 1.93 | 431 | 7.04e6 | 1.23e6 |
| [`plane3d_racetrack`](environments/plane3d_racetrack.md) | 1.35e6 | 1.07 | 1.26e6 | 6.89e6 | 1.21e6 |
| [`plane3d_figure8`](environments/plane3d_figure8.md) | 5.92e5 | 0.77 | 7.69e5 | 3.04e8 | 5,320 |
| [`patrol`](environments/patrol.md) | 265 | 0.0371 | 7,130 | 45,322 | 6,041 |
| [`patrol_bearing_only`](environments/patrol_bearing_only.md) | 199 | 0.0371 | 5,366 | 38,481 | 6,041 |
| [`formation`](environments/formation.md) | 23,239 | 1.19 | 19,592 | 5.36e6 | 6,276 |
| **Process Control** | | | | | |
| [`cstr`](environments/cstr.md) | 41.2 | 9.42e-9 | 4.38e9 | 6.6e5 | 6.1e5 |
| [`first_order`](environments/first_order.md) | 3.57e-6 | 0 | — | 1.19e5 | 1.17e5 |
| [`four_tank`](environments/four_tank.md) | 30.3 | 3.08e-8 | 9.84e8 | 6.58e5 | 1.85e5 |
| [`ph_neutralization`](environments/ph_neutralization.md) | 12.6 | 0.096 | 131 | 91,237 | 22,144 |
| [`distillation`](environments/distillation.md) | 62.6 | 0.0125 | 5,015 | 13,342 | 1,396 |
| [`unstable_cstr`](environments/unstable_cstr.md) | 57,426 | 4,035 | 14.2 | 5.84e5 | 2.52e5 |
| **Industrial / Energy** | | | | | |
| [`glass_furnace`](environments/glass_furnace.md) | 2.09 | 0.394 | 5.31 | 1,139 | 247 |
| [`reactor`](environments/reactor.md) | 23.7 | 1.38 | 17.2 | 12,374 | 597 |
| [`hvac`](environments/hvac.md) | 0.0185 | 0.00884 | 2.1 | 0.913 | 0.514 |
| [`cement_kiln`](environments/cement_kiln.md) | 4.75 | 0.0705 | 67.4 | 2,463 | 255 |
| [`boiler_drum`](environments/boiler_drum.md) | 524 | 4.67 | 112 | 38,413 | 2,378 |
| [`compressor_surge`](environments/compressor_surge.md) | 1,781 | 60.8 | 29.3 | 8.94e5 | 1.01e5 |
| **Renewable Energy** | | | | | |
| [`wind_turbine`](environments/wind_turbine.md) | 2.64e-5 | 1.3e-5 | 2.03 | 0.0132 | 0.00298 |
| [`battery`](environments/battery.md) | 0.00346 | 7.82e-4 | 4.43 | 0.0292 | 6.87e-4 |

## Episode returns

The cost per step over the whole test episode (minus the return divided by
the episode length): reaching the targets and holding them, together.
10 seeds, paired: **oracle wins** counts the seeds where the oracle
beat the expert.

| environment | steps | expert cost/step | oracle cost/step | oracle wins |
| --- | --- | --- | --- | --- |
| **Aircraft** | | | | |
| [`plane`](environments/plane.md) | 280 | 10,401 | 3,273 | 10/10 |
| [`plane_energy`](environments/plane_energy.md) | 1200 | 19,627 | 1,013 | 10/10 |
| [`plane_sine`](environments/plane_sine.md) | 480 | 5,184 | 1,249 | 10/10 |
| [`plane3d_heading`](environments/plane3d_heading.md) | 200 | 46,344 | 8,266 | 10/10 |
| [`plane3d_circle`](environments/plane3d_circle.md) | 300 | 19,185 | 3,142 | 10/10 |
| [`plane3d_racetrack`](environments/plane3d_racetrack.md) | 650 | 7.43e5 | 1,414 | 10/10 |
| [`plane3d_figure8`](environments/plane3d_figure8.md) | 400 | 1.11e6 | 14.7 | 10/10 |
| [`patrol`](environments/patrol.md) | 200 | 299 | 14.2 | 10/10 |
| [`patrol_bearing_only`](environments/patrol_bearing_only.md) | 200 | 290 | 14.2 | 10/10 |
| [`formation`](environments/formation.md) | 200 | 46,678 | 31.2 | 10/10 |
| **Process Control** | | | | |
| [`cstr`](environments/cstr.md) | 100 | 6,319 | 5,752 | 10/10 |
| [`first_order`](environments/first_order.md) | 100 | 1,018 | 1,002 | 10/10 |
| [`four_tank`](environments/four_tank.md) | 500 | 1,167 | 336 | 10/10 |
| [`ph_neutralization`](environments/ph_neutralization.md) | 300 | 336 | 91.2 | 10/10 |
| [`distillation`](environments/distillation.md) | 200 | 262 | 29.9 | 10/10 |
| [`unstable_cstr`](environments/unstable_cstr.md) | 1200 | 57,139 | 4,178 | 10/10 |
| **Industrial / Energy** | | | | |
| [`glass_furnace`](environments/glass_furnace.md) | 1600 | 1.35 | 0.274 | 10/10 |
| [`reactor`](environments/reactor.md) | 864 | 24.4 | 1.37 | 10/10 |
| [`hvac`](environments/hvac.md) | 720 | 0.0197 | 0.0106 | 10/10 |
| [`cement_kiln`](environments/cement_kiln.md) | 700 | 6.37 | 0.405 | 10/10 |
| [`boiler_drum`](environments/boiler_drum.md) | 400 | 642 | 15.3 | 10/10 |
| [`compressor_surge`](environments/compressor_surge.md) | 1200 | 1,790 | 161 | 10/10 |
| **Renewable Energy** | | | | |
| [`wind_turbine`](environments/wind_turbine.md) | 400 | 4.54e-5 | 1.38e-5 | 10/10 |
| [`battery`](environments/battery.md) | 360 | 0.00346 | 8.88e-4 | 10/10 |

## Caveats

- [`patrol_bearing_only`](environments/patrol_bearing_only.md): The PID is a lead-state estimator feeding the same pursuit law the full-observation variant uses. Range with azimuth and elevation is a complete relative-position measurement, so the only unobservable quantity is the lead's heading, which the slot needs because it is expressed in the lead's frame; the estimator recovers it by differencing the estimated relative position and filtering. The MPC slot holds patrol's oracle reading the true state, so it is a full-state bound, not a bearing-only controller. On a task defined by what the observation withholds, its score measures information and control together: what the hidden lead state is worth plus what a policy leaves on the table. It never reads the observation, so its per-step costs equal patrol's seed for seed.
- [`formation`](environments/formation.md): The PID is the patrol's, seat for seat: the patrol's lead autopilot in the lead seat, aimed at the heading the route commands, and the patrol PID in each wingman seat. It holds the lead's altitude poorly (about 140 m off), the patrol's scripted lead never needing better. The MPC slot is the patrol oracle's design applied to the whole formation: the patrol's lead autopilot in the lead seat and its twin-autopilot law in each wingman seat, with one residual planner over every seat on the team cost.
