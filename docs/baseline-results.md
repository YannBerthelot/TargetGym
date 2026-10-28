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
<script type="application/json" id="tg-results-data">[{"name":"plane","group":"Aircraft","expert":4.359949897584461,"oracle":0.5030408107099079,"ceiling":1.0,"expert_gain":4.359949897584461,"oracle_gain":0.6612176220065781},{"name":"plane_energy","group":"Aircraft","expert":286.36414842734695,"oracle":1.4346241595002847,"ceiling":0.6944444444444445,"expert_gain":18907.646672841467,"oracle_gain":496.8827473140606},{"name":"plane_sine","group":"Aircraft","expert":3.1900643624749065,"oracle":0.31763202802118234,"ceiling":0.6298815822625345,"expert_gain":3.1900643624749065,"oracle_gain":0.5273488745237559},{"name":"plane3d_heading","group":"Aircraft","expert":3.5197603332312184,"oracle":0.578064745851619,"ceiling":2.0,"expert_gain":3.5197603332312184,"oracle_gain":0.578064745851619},{"name":"plane3d_circle","group":"Aircraft","expert":3.464411363123347,"oracle":3.1829227025310196,"ceiling":2.0,"expert_gain":3.464411363123347,"oracle_gain":3.1829227025310196},{"name":"plane3d_racetrack","group":"Aircraft","expert":12.674099523878512,"oracle":1.0666768992227376,"ceiling":2.0,"expert_gain":12.674099523878512,"oracle_gain":1.0666768992227376},{"name":"plane3d_figure8","group":"Aircraft","expert":204.95170444802383,"oracle":0.5189745273111233,"ceiling":1.0,"expert_gain":204.95170444802383,"oracle_gain":0.5189745273111233},{"name":"patrol","group":"Aircraft","expert":301.8681943766276,"oracle":0.053187578530050815,"ceiling":2.0,"expert_gain":301.8681943766276,"oracle_gain":0.053187578530050815},{"name":"patrol_bearing_only","group":"Aircraft","expert":284.3472917683919,"oracle":0.053187578530050815,"ceiling":2.0,"expert_gain":284.3472917683919,"oracle_gain":0.053187578530050815},{"name":"cstr","group":"Process Control","expert":5.351844440435972e-07,"oracle":7.743093915334425e-08,"ceiling":1.0,"expert_gain":40.574538447454565,"oracle_gain":6.055761952500853e-08},{"name":"first_order","group":"Process Control","expert":4.1001887537476876e-10,"oracle":0.0,"ceiling":1.0,"expert_gain":2.5698207375745137e-06,"oracle_gain":0.0},{"name":"four_tank","group":"Process Control","expert":22.73549642829597,"oracle":1.0823311568404955e-07,"ceiling":2.0,"expert_gain":22.73549642829597,"oracle_gain":1.0823311568404955e-07},{"name":"ph_neutralization","group":"Process Control","expert":176.3924860568255,"oracle":0.08916053595838648,"ceiling":1.0,"expert_gain":176.3959451028495,"oracle_gain":0.09254092571742628},{"name":"distillation","group":"Process Control","expert":390.5597425599148,"oracle":0.022935034238076735,"ceiling":2.0,"expert_gain":390.5597425599148,"oracle_gain":0.022935034238076735},{"name":"glass_furnace","group":"Industrial / Energy","expert":0.966626035515219,"oracle":0.17642213267584642,"ceiling":1.0,"expert_gain":1.0621182994761815,"oracle_gain":0.23432330360491807},{"name":"reactor","group":"Industrial / Energy","expert":20.59371337383355,"oracle":1.4550849451448464,"ceiling":0.0027777777777777783,"expert_gain":20.976558486252657,"oracle_gain":1.4757691727100333},{"name":"hvac","group":"Industrial / Energy","expert":0.02296698320457724,"oracle":0.014208605279949188,"ceiling":7.500000000000001e-05,"expert_gain":0.024219282179737327,"oracle_gain":0.01374337118379892},{"name":"cement_kiln","group":"Industrial / Energy","expert":3.6019709180748767,"oracle":0.02572445288550525,"ceiling":1.0,"expert_gain":3.6806535686426454,"oracle_gain":0.07316629200316038},{"name":"boiler_drum","group":"Industrial / Energy","expert":890.8466335406198,"oracle":5.534708091409822,"ceiling":1.1402740955827686,"expert_gain":890.9114037591526,"oracle_gain":5.615284871186733},{"name":"wind_turbine","group":"Renewable Energy","expert":2.5207256451060072e-05,"oracle":1.1796639865255069e-05,"ceiling":6.944444444444445e-06,"expert_gain":2.789124735509555e-05,"oracle_gain":1.2609169900260989e-05},{"name":"battery","group":"Renewable Energy","expert":0.0006802680312170715,"oracle":0.00022913516181502782,"ceiling":0.00013888888888888886,"expert_gain":0.003327833331628043,"oracle_gain":0.0009253222457801855},{"name":"unstable_cstr","group":"Process Control","expert":3.2054767887849436,"oracle":0.007304711043748235,"ceiling":1.0,"expert_gain":56956.91191127999,"oracle_gain":3799.6714203457345},{"name":"compressor_surge","group":"Industrial / Energy","expert":15.834977883561267,"oracle":0.2390619778123484,"ceiling":1.0,"expert_gain":488.6196853882157,"oracle_gain":23.012493830370108},{"name":"formation","group":"Aircraft","expert":23788.02022542318,"oracle":1.011592126103739,"ceiling":6.0,"expert_gain":23788.02022542318,"oracle_gain":1.011592126103739}]</script>

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
| [`plane`](environments/plane.md) | 4.36 | 1 | 0.64 | 0.503 | 1.00 |
| [`plane_energy`](environments/plane_energy.md) | 286 | 0.694 | 2.62 | 1.43 | 0.88 |
| [`plane_sine`](environments/plane_sine.md) | 3.19 | 0.63 | 0.70 | 0.318 | 1.00 |
| [`plane3d_heading`](environments/plane3d_heading.md) | 3.52 | 2 | 0.25 | 0.578 | 1.00 |
| [`plane3d_circle`](environments/plane3d_circle.md) | 3.46 | 2 | 0.24 | 3.18 | 0.15 |
| [`plane3d_racetrack`](environments/plane3d_racetrack.md) | 12.7 | 2 | 0.80 | 1.07 | 1.00 |
| [`plane3d_figure8`](environments/plane3d_figure8.md) | 205 | 1 | 2.31 | 0.519 | 1.00 |
| [`patrol`](environments/patrol.md) | 302 | 2 | 2.18 | 0.0532 | 1.00 |
| [`patrol_bearing_only`](environments/patrol_bearing_only.md) | 284 | 2 | 2.15 | 0.0532 | 1.00 |
| [`formation`](environments/formation.md) | 23,788 | 6 | 3.60 | 1.01 | 1.00 |
| **Process Control** | | | | | |
| [`cstr`](environments/cstr.md) | 5.35e-7 | 1 | 0.00 | 7.74e-8 | — |
| [`first_order`](environments/first_order.md) | 4.1e-10 | 1 | 0.00 | 0 | — |
| [`four_tank`](environments/four_tank.md) | 22.7 | 2 | 1.06 | 1.08e-7 | 1.00 |
| [`ph_neutralization`](environments/ph_neutralization.md) | 176 | 1 | 2.25 | 0.0892 | 1.00 |
| [`distillation`](environments/distillation.md) | 391 | 2 | 2.29 | 0.0229 | 1.00 |
| [`unstable_cstr`](environments/unstable_cstr.md) | 3.21 | 1 | 0.51 | 0.0073 | 1.00 |
| **Industrial / Energy** | | | | | |
| [`glass_furnace`](environments/glass_furnace.md) | 0.967 | 1 | 0.00 | 0.176 | — |
| [`reactor`](environments/reactor.md) | 20.6 | 0.00278 | 3.87 | 1.46 | 0.30 |
| [`hvac`](environments/hvac.md) | 0.023 | 7.5e-5 | 2.49 | 0.0142 | 0.08 |
| [`cement_kiln`](environments/cement_kiln.md) | 3.6 | 1 | 0.56 | 0.0257 | 1.00 |
| [`boiler_drum`](environments/boiler_drum.md) | 891 | 1.14 | 2.89 | 5.53 | 0.76 |
| [`compressor_surge`](environments/compressor_surge.md) | 15.8 | 1 | 1.20 | 0.239 | 1.00 |
| **Renewable Energy** | | | | | |
| [`wind_turbine`](environments/wind_turbine.md) | 2.52e-5 | 6.94e-6 | 0.56 | 1.18e-5 | 0.59 |
| [`battery`](environments/battery.md) | 6.8e-4 | 1.39e-4 | 0.69 | 2.29e-4 | 0.68 |

## Whole episode

The **gain** is the mean cost per step after the plant's burn-in, moves
between targets included: what the plant pays. **Oracle vs expert** is
the expert's gain divided by the oracle's, so 10 means the oracle pays a
tenth. The **transient** is the summed cost around each target change.

| environment | expert gain | oracle gain | oracle vs expert | expert transient | oracle transient |
| --- | --- | --- | --- | --- | --- |
| **Aircraft** | | | | | |
| [`plane`](environments/plane.md) | 4.36 | 0.661 | 6.59 | 1.64e6 | 3.2e5 |
| [`plane_energy`](environments/plane_energy.md) | 18,908 | 497 | 38.1 | 2.49e6 | 86,392 |
| [`plane_sine`](environments/plane_sine.md) | 3.19 | 0.527 | 6.05 | 2.84e6 | 2.27e5 |
| [`plane3d_heading`](environments/plane3d_heading.md) | 3.52 | 0.578 | 6.09 | 2.9e6 | 9e5 |
| [`plane3d_circle`](environments/plane3d_circle.md) | 3.46 | 3.18 | 1.09 | 1.63e6 | 3.42e5 |
| [`plane3d_racetrack`](environments/plane3d_racetrack.md) | 12.7 | 1.07 | 11.9 | 1.64e6 | 3.26e5 |
| [`plane3d_figure8`](environments/plane3d_figure8.md) | 205 | 0.519 | 395 | 8.41e6 | 6,068 |
| [`patrol`](environments/patrol.md) | 302 | 0.0532 | 5,676 | 30,726 | 3,168 |
| [`patrol_bearing_only`](environments/patrol_bearing_only.md) | 284 | 0.0532 | 5,346 | 28,714 | 3,168 |
| [`formation`](environments/formation.md) | 23,788 | 1.01 | 23,515 | 1.41e7 | 5,121 |
| **Process Control** | | | | | |
| [`cstr`](environments/cstr.md) | 40.6 | 6.06e-8 | 6.7e8 | 3.49e5 | 3.09e5 |
| [`first_order`](environments/first_order.md) | 2.57e-6 | 0 | — | 55,292 | 53,873 |
| [`four_tank`](environments/four_tank.md) | 22.7 | 1.08e-7 | 2.1e8 | 8.75e5 | 5e5 |
| [`ph_neutralization`](environments/ph_neutralization.md) | 176 | 0.0925 | 1,906 | 41,669 | 2,557 |
| [`distillation`](environments/distillation.md) | 391 | 0.0229 | 17,029 | 1.13e5 | 38,594 |
| [`unstable_cstr`](environments/unstable_cstr.md) | 56,957 | 3,800 | 15 | 5.84e5 | 2.48e5 |
| **Industrial / Energy** | | | | | |
| [`glass_furnace`](environments/glass_furnace.md) | 1.06 | 0.234 | 4.53 | 1,164 | 315 |
| [`reactor`](environments/reactor.md) | 21 | 1.48 | 14.2 | 1,261 | 95 |
| [`hvac`](environments/hvac.md) | 0.0242 | 0.0137 | 1.76 | 1.01 | 0.519 |
| [`cement_kiln`](environments/cement_kiln.md) | 3.68 | 0.0732 | 50.3 | 1,410 | 153 |
| [`boiler_drum`](environments/boiler_drum.md) | 891 | 5.62 | 159 | 31,194 | 6,919 |
| [`compressor_surge`](environments/compressor_surge.md) | 489 | 23 | 21.2 | 9.58e5 | 46,463 |
| **Renewable Energy** | | | | | |
| [`wind_turbine`](environments/wind_turbine.md) | 2.79e-5 | 1.26e-5 | 2.21 | 0.0139 | 0.00324 |
| [`battery`](environments/battery.md) | 0.00333 | 9.25e-4 | 3.6 | 0.0585 | 9.08e-4 |

## Episode returns

The cost per step over the whole test episode (minus the return divided by
the episode length): reaching the targets and holding them, together.
10 seeds, paired: **oracle wins** counts the seeds where the oracle
beat the expert.

| environment | steps | expert cost/step | oracle cost/step | oracle wins |
| --- | --- | --- | --- | --- |
| **Aircraft** | | | | |
| [`plane`](environments/plane.md) | 280 | 3,778 | 1,121 | 9/10 |
| [`plane_energy`](environments/plane_energy.md) | 1200 | 18,323 | 659 | 10/10 |
| [`plane_sine`](environments/plane_sine.md) | 480 | 2,241 | 387 | 10/10 |
| [`plane3d_heading`](environments/plane3d_heading.md) | 200 | 8,357 | 3,446 | 9/10 |
| [`plane3d_circle`](environments/plane3d_circle.md) | 300 | 3,535 | 1,123 | 8/10 |
| [`plane3d_racetrack`](environments/plane3d_racetrack.md) | 650 | 1,637 | 495 | 10/10 |
| [`plane3d_figure8`](environments/plane3d_figure8.md) | 400 | 14,513 | 13.9 | 10/10 |
| [`patrol`](environments/patrol.md) | 200 | 305 | 19.5 | 10/10 |
| [`patrol_bearing_only`](environments/patrol_bearing_only.md) | 200 | 301 | 19.5 | 10/10 |
| [`formation`](environments/formation.md) | 200 | 55,472 | 26.6 | 10/10 |
| **Process Control** | | | | |
| [`cstr`](environments/cstr.md) | 100 | 6,252 | 5,987 | 10/10 |
| [`first_order`](environments/first_order.md) | 100 | 426 | 411 | 10/10 |
| [`four_tank`](environments/four_tank.md) | 500 | 832 | 373 | 10/10 |
| [`ph_neutralization`](environments/ph_neutralization.md) | 300 | 168 | 5.31 | 10/10 |
| [`distillation`](environments/distillation.md) | 200 | 8.67e5 | 123 | 10/10 |
| [`unstable_cstr`](environments/unstable_cstr.md) | 1200 | 57,083 | 3,950 | 10/10 |
| **Industrial / Energy** | | | | |
| [`glass_furnace`](environments/glass_furnace.md) | 1600 | 1.28 | 0.312 | 10/10 |
| [`reactor`](environments/reactor.md) | 864 | 22.1 | 1.4 | 10/10 |
| [`hvac`](environments/hvac.md) | 720 | 0.0203 | 0.0113 | 10/10 |
| [`cement_kiln`](environments/cement_kiln.md) | 700 | 5.64 | 0.157 | 10/10 |
| [`boiler_drum`](environments/boiler_drum.md) | 400 | 752 | 22.1 | 10/10 |
| [`compressor_surge`](environments/compressor_surge.md) | 1200 | 1,885 | 102 | 10/10 |
| **Renewable Energy** | | | | |
| [`wind_turbine`](environments/wind_turbine.md) | 400 | 5.42e-5 | 1.48e-5 | 10/10 |
| [`battery`](environments/battery.md) | 360 | 0.00354 | 9.48e-4 | 10/10 |

## Caveats

- [`patrol_bearing_only`](environments/patrol_bearing_only.md): The PID is a lead-state estimator feeding the same pursuit law the full-observation variant uses. Range with azimuth and elevation is a complete relative-position measurement, so the only unobservable quantity is the lead's heading, which the slot needs because it is expressed in the lead's frame; the estimator recovers it by differencing the estimated relative position and filtering. The MPC slot holds patrol's oracle reading the true state, so it is a full-state bound, not a bearing-only controller. On a task defined by what the observation withholds, its score measures information and control together: what the hidden lead state is worth plus what a policy leaves on the table. It never reads the observation, so its per-step costs equal patrol's seed for seed.
- [`formation`](environments/formation.md): The PID is the patrol's, seat for seat: the patrol's lead autopilot in the lead seat, aimed at the heading the route commands, and the patrol PID in each wingman seat. It holds the lead's altitude poorly (about 140 m off), the patrol's scripted lead never needing better. The MPC slot is the patrol oracle's design applied to the whole formation: the patrol's lead autopilot in the lead seat and its twin-autopilot law in each wingman seat, with one residual planner over every seat on the team cost.
