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

## At a glance

How many times cheaper the oracle holds each target than the expert,
and where the ceiling sits: tracking every quantity within its
instrument's resolution. Switch the anchor to the ceiling to see how far
each controller is from as good as a real plant can verify. Hover a row
for the numbers, including the whole episode.

<div class="tg-results-chart"></div>
<script type="application/json" id="tg-results-data">[{"name":"plane","group":"Aircraft","expert":3.9494032927921836,"oracle":0.43404277960459386,"ceiling":1.0,"expert_gain":3.9494032927921836,"oracle_gain":0.5286854054734466},{"name":"plane_energy","group":"Aircraft","expert":7944.319651882765,"oracle":55.03556500118432,"ceiling":0.6944444444444445,"expert_gain":18909.907016076206,"oracle_gain":511.4264289681344},{"name":"plane_sine","group":"Aircraft","expert":3.4913954925983313,"oracle":0.2908504340464426,"ceiling":0.6298815822625345,"expert_gain":3.4913954925983313,"oracle_gain":0.5434684810655379},{"name":"plane3d_heading","group":"Aircraft","expert":4.564862690787219,"oracle":0.735469923974953,"ceiling":2.0,"expert_gain":4.564862690787219,"oracle_gain":0.735469923974953},{"name":"plane3d_circle","group":"Aircraft","expert":4.400324082128031,"oracle":1.94241143233298,"ceiling":2.0,"expert_gain":4.400318746195148,"oracle_gain":1.9424177083805099},{"name":"plane3d_racetrack","group":"Aircraft","expert":8.380952791908319,"oracle":1.0637917741973621,"ceiling":2.0,"expert_gain":8.380947574961692,"oracle_gain":1.063795931462234},{"name":"plane3d_figure8","group":"Aircraft","expert":8298.023415406287,"oracle":0.7715008953159365,"ceiling":1.0,"expert_gain":8298.02353023548,"oracle_gain":0.7714991115204368},{"name":"patrol","group":"Aircraft","expert":264.6584235954285,"oracle":0.037119886965956535,"ceiling":2.0,"expert_gain":264.6584248034159,"oracle_gain":0.037119884764930854},{"name":"patrol_bearing_only","group":"Aircraft","expert":199.20231346766153,"oracle":0.037119886965956535,"ceiling":2.0,"expert_gain":199.202312819163,"oracle_gain":0.037119884764930854},{"name":"cstr","group":"Process Control","expert":0.5384376945310133,"oracle":4.4971059225322795e-09,"ceiling":1.0,"expert_gain":41.23786548601277,"oracle_gain":9.420074148334663e-09},{"name":"first_order","group":"Process Control","expert":2.910890690389658e-07,"oracle":0.0,"ceiling":1.0,"expert_gain":3.5713776159847415e-06,"oracle_gain":0.0},{"name":"four_tank","group":"Process Control","expert":30.28834361658494,"oracle":3.077449156044243e-08,"ceiling":2.0,"expert_gain":30.28834361658494,"oracle_gain":3.077449156044243e-08},{"name":"ph_neutralization","group":"Process Control","expert":12.576964106020077,"oracle":0.08895924857341697,"ceiling":1.0,"expert_gain":12.583775967234233,"oracle_gain":0.09600657768150238},{"name":"distillation","group":"Process Control","expert":62.5916948646307,"oracle":0.0002610119988506199,"ceiling":2.0,"expert_gain":62.593887679775555,"oracle_gain":0.012482149792454702},{"name":"glass_furnace","group":"Industrial / Energy","expert":2.016519948622833,"oracle":0.35356583004196485,"ceiling":1.0,"expert_gain":2.0888898671325298,"oracle_gain":0.3935563686528501},{"name":"reactor","group":"Industrial / Energy","expert":23.32241257822808,"oracle":1.378374774031985,"ceiling":0.0027777777777777783,"expert_gain":23.707245022426417,"oracle_gain":1.3783988989199754},{"name":"hvac","group":"Industrial / Energy","expert":0.009820630252304181,"oracle":0.00521838269202782,"ceiling":7.500000000000001e-05,"expert_gain":0.018539571937530596,"oracle_gain":0.008839867491681438},{"name":"cement_kiln","group":"Industrial / Energy","expert":4.67806490569376,"oracle":0.02946235461633244,"ceiling":1.0,"expert_gain":4.751233760789392,"oracle_gain":0.07054527778497986},{"name":"boiler_drum","group":"Industrial / Energy","expert":524.1454528630912,"oracle":4.622581396118014,"ceiling":1.1402740955827686,"expert_gain":524.173030201605,"oracle_gain":4.668878252191159},{"name":"wind_turbine","group":"Renewable Energy","expert":2.4260000647882882e-05,"oracle":1.269647026355347e-05,"ceiling":6.944444444444445e-06,"expert_gain":2.6381257000084683e-05,"oracle_gain":1.3004420578930365e-05},{"name":"battery","group":"Renewable Energy","expert":0.0007932258506096794,"oracle":0.00021392915275035077,"ceiling":0.00013888888888888886,"expert_gain":0.0034647340856809954,"oracle_gain":0.000781631191645189},{"name":"unstable_cstr","group":"Process Control","expert":2.817811035836616,"oracle":0.006860461602341019,"ceiling":1.0,"expert_gain":57426.27775293447,"oracle_gain":4034.878881758566},{"name":"compressor_surge","group":"Industrial / Energy","expert":11.365509030165263,"oracle":0.006769536329358998,"ceiling":1.0,"expert_gain":1781.013958473079,"oracle_gain":60.788684407643096},{"name":"formation","group":"Aircraft","expert":23238.974378255207,"oracle":1.1746055503624182,"ceiling":6.0,"expert_gain":23238.974378255207,"oracle_gain":1.1746055504865944}]</script>

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
| [`plane`](environments/plane.md) | 3.95 | 1 | 0.60 | 0.434 | 1.00 |
| [`plane_energy`](environments/plane_energy.md) | 7,944 | 0.694 | 4.06 | 55 | 0.53 |
| [`plane_sine`](environments/plane_sine.md) | 3.49 | 0.63 | 0.74 | 0.291 | 1.00 |
| [`plane3d_heading`](environments/plane3d_heading.md) | 4.56 | 2 | 0.36 | 0.735 | 1.00 |
| [`plane3d_circle`](environments/plane3d_circle.md) | 4.4 | 2 | 0.34 | 1.94 | 1.00 |
| [`plane3d_racetrack`](environments/plane3d_racetrack.md) | 8.38 | 2 | 0.62 | 1.06 | 1.00 |
| [`plane3d_figure8`](environments/plane3d_figure8.md) | 8,298 | 1 | 3.92 | 0.772 | 1.00 |
| [`patrol`](environments/patrol.md) | 265 | 2 | 2.12 | 0.0371 | 1.00 |
| [`patrol_bearing_only`](environments/patrol_bearing_only.md) | 199 | 2 | 2.00 | 0.0371 | 1.00 |
| [`formation`](environments/formation.md) | 23,239 | 6 | 3.59 | 1.17 | 1.00 |
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
| [`plane`](environments/plane.md) | 3.95 | 0.529 | 7.47 | 4.84e6 | 1.19e6 |
| [`plane_energy`](environments/plane_energy.md) | 18,910 | 511 | 37 | 3.17e6 | 1.81e5 |
| [`plane_sine`](environments/plane_sine.md) | 3.49 | 0.543 | 6.42 | 4.87e6 | 7.61e5 |
| [`plane3d_heading`](environments/plane3d_heading.md) | 4.56 | 0.735 | 6.21 | 6.06e6 | 2.01e6 |
| [`plane3d_circle`](environments/plane3d_circle.md) | 4.4 | 1.94 | 2.27 | 4.84e6 | 1.23e6 |
| [`plane3d_racetrack`](environments/plane3d_racetrack.md) | 8.38 | 1.06 | 7.88 | 4.84e6 | 1.21e6 |
| [`plane3d_figure8`](environments/plane3d_figure8.md) | 8,298 | 0.771 | 10,756 | 4.9e6 | 5,323 |
| [`patrol`](environments/patrol.md) | 265 | 0.0371 | 7,130 | 45,322 | 6,041 |
| [`patrol_bearing_only`](environments/patrol_bearing_only.md) | 199 | 0.0371 | 5,366 | 38,481 | 6,041 |
| [`formation`](environments/formation.md) | 23,239 | 1.17 | 19,784 | 5.36e6 | 6,273 |
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
| [`plane`](environments/plane.md) | 280 | 13,489 | 3,273 | 10/10 |
| [`plane_energy`](environments/plane_energy.md) | 1200 | 19,892 | 1,009 | 10/10 |
| [`plane_sine`](environments/plane_sine.md) | 480 | 9,493 | 1,249 | 10/10 |
| [`plane3d_heading`](environments/plane3d_heading.md) | 200 | 24,895 | 8,267 | 10/10 |
| [`plane3d_circle`](environments/plane3d_circle.md) | 300 | 12,582 | 3,142 | 9/10 |
| [`plane3d_racetrack`](environments/plane3d_racetrack.md) | 650 | 5,818 | 1,414 | 10/10 |
| [`plane3d_figure8`](environments/plane3d_figure8.md) | 400 | 18,260 | 14.7 | 10/10 |
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
