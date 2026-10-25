# Baseline results

<!-- Written by scripts/generate_baseline_table.py from
     src/target_gym/data/protocol_results.json and baseline_returns.json.
     Do not edit by hand. -->

Every environment ships two reference controllers, scored on its test
configuration (`spec.make_test_params()`):

- **PID**: reads the observation only, as your agent does. It is the score to
  beat.
- **MPC**: reads the true state, including what the observation hides, so it is
  an upper bound rather than a peer.

Every reward is a cost, so lower is better. To score your own policy against
these numbers, see [Evaluating against it](using-the-pids.md#evaluating-against-it).
How the controllers are built and measured is in
[How the baselines work](baselines.md).

## Protocol score

The headline of the [RL protocol](rl-protocol.md). The **gain** is the mean
cost per step after the plant's burn-in, and **NEA** places a controller on a
scale where the PID is 0 and the achievable floor is 1:

    NEA = (PID gain - your gain) / (PID gain - floor)

Gains are in each environment's own units, so compare them along a row; NEA
compares across rows. 3 seeds each.

| environment | PID gain | MPC gain | floor | MPC NEA |
| --- | --- | --- | --- | --- |
| **Aircraft** | | | | |
| [`plane`](environments/plane.md) | 6.21 | 0.516 | 0.21 | 0.949 |
| [`plane_energy`](environments/plane_energy.md) | 11,210 | 511 | 0.963 | 0.954 |
| [`plane_sine`](environments/plane_sine.md) | 1,522 | 0.546 | 0.176 | 1.000 |
| [`plane3d_heading`](environments/plane3d_heading.md) | 19,023 | 0.726 | 0.454 | 1.000 |
| [`plane3d_circle`](environments/plane3d_circle.md) | 832 | 1.93 | 0.466 | 0.998 |
| [`plane3d_racetrack`](environments/plane3d_racetrack.md) | 1.35e6 | 1.07 | 0.487 | 1.000 |
| [`plane3d_figure8`](environments/plane3d_figure8.md) | 5.92e5 | 0.77 | 0.184 | 1.000 |
| [`patrol`](environments/patrol.md) | 265 | 0.0371 | 0.0363 | 1.000 |
| [`patrol_bearing_only`](environments/patrol_bearing_only.md) | 199 | 0.0371 | 0.0363 | 1.000 |
| **Process Control** | | | | |
| [`cstr`](environments/cstr.md) | 41.2 | 9.42e-9 | 0 | 1.000 |
| [`first_order`](environments/first_order.md) | 3.57e-6 | 0 | 0 | 1.000 |
| [`four_tank`](environments/four_tank.md) | 30.3 | 3.08e-8 | 0 | 1.000 |
| [`ph_neutralization`](environments/ph_neutralization.md) | 12.6 | 0.096 | 0.0169 | 0.994 |
| [`distillation`](environments/distillation.md) | 62.6 | 0.0125 | 8.53e-5 | 1.000 |
| [`unstable_cstr`](environments/unstable_cstr.md) | 57,426 | 4,035 | 0.00396 | 0.930 |
| **Industrial / Energy** | | | | |
| [`glass_furnace`](environments/glass_furnace.md) | 2.09 | 0.394 | 0.00377 | 0.813 |
| [`reactor`](environments/reactor.md) | 23.7 | 1.38 | 1.25 | 0.994 |
| [`hvac`](environments/hvac.md) | 0.0185 | 0.00884 | 0.001 | 0.553 |
| [`cement_kiln`](environments/cement_kiln.md) | 4.75 | 0.0705 | 0.00956 | 0.987 |
| [`boiler_drum`](environments/boiler_drum.md) | 524 | 4.67 | 1.92 | 0.995 |
| [`compressor_surge`](environments/compressor_surge.md) | 1,781 | 60.8 | 1.4e-5 | 0.966 |
| **Renewable Energy** | | | | |
| [`wind_turbine`](environments/wind_turbine.md) | 2.64e-5 | 1.3e-5 | 1.01e-5 | 0.821 |
| [`battery`](environments/battery.md) | 0.00346 | 7.82e-4 | 2.22e-4 | 0.827 |

## Episode returns

The cost per step over the whole test episode (minus the return divided by
the episode length): reaching the targets and holding them, together.
10 seeds, paired: **MPC wins** counts the seeds where the MPC beat
the PID.

| environment | steps | PID cost/step | MPC cost/step | MPC wins |
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

- [`patrol_bearing_only`](environments/patrol_bearing_only.md): The PID is a lead-state estimator feeding the same pursuit law the full-observation variant uses. Range with azimuth and elevation is a complete relative-position measurement, so the only unobservable quantity is the lead's heading, which the slot needs because it is expressed in the lead's frame; the estimator recovers it by differencing the estimated relative position and filtering. The MPC slot holds patrol's oracle reading the true state, so it is a full-state bound, not a bearing-only controller. On a task defined by what the observation withholds, its NEA measures information and control together: what the hidden lead state is worth plus what a policy leaves on the table. It never reads the observation, so its per-step costs equal patrol's seed for seed (oracle audit, 2026-10).
