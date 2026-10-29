# Scoring a policy: experts, ceiling and metric

A proposal written on 2026-10-27. It replaces the normalised expert advantage
(NEA) in [the RL protocol](../rl-protocol.md), section 4, and the way the
shipped baselines are built.

## Status

Accepted on 2026-10-27, with every recommended option: `S` and `H` on the hold
(decision 1), consumption as a ratio beside `S` (decision 2), the base layer
now and an observation-only MPC later on the column and the furnace
(decision 3), and a published rule base on the kiln (decision 4). The metric
is the first pull request; the experts follow, one family at a time.

The aircraft copy UAV autopilot practice (PX4, ArduPilot) on every task,
chosen on 2026-10-27 over airliner modes: TECS, L1 guidance, a course hold
and a 45 deg bank limit with stall prevention, on the A320-class airframe,
which flies 45 deg within its limits and the model's validity. Airliner
autopilots cap bank at 25-30 deg for comfort, which would leave the heading
expert mid-turn when scoring starts. The single-aircraft tasks are the second
pull request; the patrol family and formation, a flight-test formation hold
(no deployed product exists), the third.

## What a result has to say

1. **Against the expert:** is the policy better or worse than the solution
   practitioners use today, and by how much?
2. **Across environments:** how does that result compare with results on other
   tasks?
3. **Against the ceiling:** how much room is left before the task's best
   possible performance?

NEA answered the second question badly. It was the linear share of the gap from
the PID to an achievable floor, and on 19 of the 24 tasks the PID costs at
least 100 times the floor (or the floor is 0), so any working policy scored
above 0.99. Part of
that gap is real (the oracle reads the true state and plans), and part of it is
experts that are badly built or tuned on the seeds they are scored on (see
[The experts](#the-experts-plant-by-plant)).

## Three roles

| Role | What it is | What it may use | How it is built |
|---|---|---|---|
| **Expert** | The solution practitioners deploy for this plant today | The observation, plus the plant constants a commissioning engineer has (datasheet values, a surge line) | The field's own structure and tuning recipe, on seeds that are never scored |
| **Ceiling** | As good as possible: tracking within the instrument's resolution | Nothing: it is a number, not a controller | The reward's charge for an error of one resolution |
| **Oracle** | The best controller we can build | The true state and the target schedule | Informative only. It shows what information and planning are worth, and whether the ceiling is reachable |

The expert has to be a faithful copy of practice, because "better than the
expert" is the claim a result makes. A weak expert turns every policy into a
winner. It does not have to be strong: where practice is weak, the task has
room to grow, and the metric should say so.

## The metric

Costs are per step, lower is better: `x` is the policy's, `b` the expert's,
`c` the ceiling's.

| Question | Number | Reads as |
|---|---|---|
| Against the expert | `b / x` per seed, with the win rate and a 95% bootstrap interval on the interquartile mean | "3.2 times cheaper than the expert, on 9 of 10 seeds". Below 1 is worse |
| Across environments | `S = log(b / x) / log(b / c)`, capped at 1 | 0 is the expert, 1 is the ceiling, negative is worse than the expert |
| Room to grow | `H = log10(b / c)` for the task; `log10(x / c)` left for a policy | "the task offers 2 decades beyond the expert; this policy has 1 left" |

For example, with an expert at 100, a ceiling at 1 and a policy at 10, the
policy is 10 times cheaper than the expert, `S = 0.5`, and one of the task's two
decades is left.

Why ratios and logarithms:

- **No units.** A ratio of two costs has none, so a plant priced in euros and a
  plant scored in floor-widths compare.
- **No dependence on the cost's shape.** If tracking costs `k e^p`, every
  logarithm above is multiplied by `p`, which cancels in `S`. Squared or linear
  tracking gives the same `S` for the same errors (exactly for one tracked
  term, closely for several). A linear share changes with `p`, which is why NEA
  collapsed.
- **Bounded at the top.** Nothing scores above 1.

**No room.** Where the expert already tracks within the resolution (`b <= c`),
the task has nothing left to improve and `S` is not reported. `first_order` is
such a task: its PID costs 3.6e-6 per step.

## What is scored

### The hold, with the transients beside it (decision 1)

The ceiling describes holding a target. The protocol's gain also contains the
moves between targets, which every controller pays for: on
`compressor_surge` the oracle's gain is 61 per step against a hold of 0.21.
Scored on the gain, even the oracle would reach `S = 0.45` there and about 1
on `patrol`, so `S` would not compare across tasks.

Proposal: `S` and `H` are computed on the **hold**, the cost after a fixed
settle time per target change. The settle time is the slower of the shipped
controllers' measured settles (`hold_measurements.json`), so the split is
deterministic and the same window for every controller. The transients are scored separately, against the expert only:
the expert's transient cost per target change divided by the policy's. They
have no ceiling, because nothing bounds them cheaply; the oracle's ratio is
shown as context. The verdict against the expert (`b / x`) stays on the total
cost, so a policy cannot win by being slow to reach its targets.

### Consumption (decision 2)

The resolution says nothing about fuel or energy. `S` and `H` use the tracking
terms only, and consumption is reported as the policy's cost divided by the
expert's. The verdict is on the total reward, so a policy that buys tracking
with fuel loses there.

### The ceiling's value

Per tracked term, the ceiling is the reward's charge for an error one
resolution beyond the tolerance band, `(resolution / e_floor) ** p`. That is 1
where `e_floor` is the resolution, and less where `e_floor` was raised to the
oracle's measured hold. On the four priced plants
(reactor, battery, wind turbine, building) it is the price of that error in the
plant's currency. The terms live in `target_gym.scoring`, which no
fingerprint hashes, so setting them needs no re-record.

## The experts, plant by plant

Tuning rules for every expert:

- **Never on scored seeds.** The protocol scores seeds 0-2 and the baselines
  0-9. Today the aircraft PIDs were tuned on exactly 0-2, the racetrack on 0-4,
  and the unstable CSTR and compressor on 0-23. Tuning uses seeds 100 and up,
  and a held-out check uses 200 and up.
- **Process loops:** a step test on the simulator, a first-order-plus-delay
  fit, then SIMC rules (Skogestad), or lambda tuning for integrating loops such
  as drum level. Where a standard publishes defaults (ASHRAE Guideline 36 for
  zone temperature), use them.
- **Aircraft:** inner loops first, as the PX4 and ArduPilot guides do, then the
  guidance law by its published period and damping, with a bounded search on
  training seeds where the field tunes in flight.
- **Sanity targets.** Flight tests of small UAVs in 5 m/s wind hold altitude
  within 1-2 m, a path within 2-3 m (L1 guidance) and a formation slot within
  about 2 m on GPS; NASA's F/A-18 formation trials met 1-sigma spacing of
  10 ft. An aircraft expert tens of metres off in our turbulence is badly
  built, not realistic. These are small-aircraft figures, so compare in
  wingspans.

| Environment | Practice | Ours today | Change |
|---|---|---|---|
| first_order | Academic; PI, SIMC-tuned | PID with in-code defaults | SIMC from a step test |
| cstr | Reactor temperature PID on coolant; concentration analysers are too slow to close on | PID from concentration to coolant | Cascade: concentration trims a temperature loop |
| unstable_cstr | High-gain temperature PI/PID with anti-windup | Cascade PI on concentration, PD on temperature | Keep; retune off the scored seeds |
| four_tank | Academic; decentralised crossed PI | Two crossed PI loops, tuned on separate seeds | Keep |
| distillation | LV control with a tray-temperature loop, and an MPC layer is the norm | Two diagonal composition PI loops | Decision 3 |
| ph_neutralization | PID with titration-curve linearisation and inflow feedforward | Plain PI | Add linearisation and feedforward |
| glass_furnace | PID loops with an MPC layer, common but proprietary | Crown-temperature PID, grid-tuned on seeds 0-2 | Decision 3 |
| cement_kiln | Rule-based expert (fuzzy) control built from operator rules | PID cascade with a lime trim | Decision 4 |
| boiler_drum | Three-element control, lambda-tuned | Three-element, hand-tuned; one seed dominates the mean (524 ± 528) | Lambda retune; find the bad seed |
| compressor_surge | Vendor anti-surge PID on the distance to a surge control line | The same, tuned on seeds 0-23 | Retune off the scored seeds |
| reactor | Rod control on average temperature against a load reference | Flux PID whose internal step (0.5 s) differs from the environment's control step | Check whether the gains absorb it; retune |
| hvac | Guideline 36 PI (kp 0.1/K, Ti 900 s) | PI with in-code defaults | Use the Guideline 36 values |
| battery | Scheduled setpoint, state-of-charge limits, inner PI | Feedforward, PI trim, state-of-charge guard | Keep |
| wind_turbine | `k ω²` torque below rated, gain-scheduled pitch PI above | Torque by plant-model inversion | Replace the torque law with `k ω²`. Corrected on 2026-10-28: the expert's torque was the constant-power law (the inversion is the oracle's), and `k ω²` is below-rated only. The NREL 5 MW baseline was built instead (the turbine's PHYSICS.md, section 5) |
| plane, plane_sine | Altitude-hold cascade with autothrottle | Altitude cascade, tuned on seeds 0-2 | Retune off the scored seeds |
| plane_energy | TECS (Lambregts) | The plane's cascade, not retuned | Implement TECS |
| plane3d_heading | Heading to bank, altitude to pitch cascade | Same structure, but holds 51 m and 1 rad off; roll-rate damping has the opposite sign to the shared PID's | Check the sign; retune |
| plane3d_circle, _figure8, _racetrack | L1 (or NPFG) guidance over the attitude loops | Three different laws; figure-eight and racetrack drift by kilometres | One L1 guidance law for all three |
| patrol | PI formation hold over the wingman's own autopilots | Pursuit guidance, 44-68 m out of the slot | Formation hold, tuned on training seeds |
| patrol_bearing_only | An estimator on range and bearings feeding the same hold | Estimator plus pursuit | Estimator plus formation hold |
| formation | Leader-follower hold; the lead flies its own route guidance | Lead 141-155 m off its altitude; pursuit wingmen | A lead with a working altitude hold; formation-hold wingmen |

### Where practice runs an MPC (decision 3)

On distillation columns an MPC over the temperature loops is the norm, and on
glass furnaces it is common. There the expert should be an MPC that reads only
the observation, which is PR 2 of the experts refactor. Until it exists, the
expert is the base layer practitioners run under it (LV with a tray-temperature
loop on the column, the crown-temperature PID on the furnace), and the
environment page says that practice adds an MPC on top.

### The cement kiln (decision 4)

Kilns are most often run by rule-based expert systems that encode operators'
rules. The choice is to implement a published rule base, or to keep the PID
cascade and document that it is below common practice.

## Order of work

1. **The metric.** A new module, `target_gym.scoring` (`b / x`, `S`, `H`, the
   transient and consumption ratios, each environment's tracked terms), a fixed
   settle per environment in `eval.py`, and the docs and generated tables. NEA
   goes. No fingerprint moves, but the
   protocol results are re-evaluated once for the fixed settle times (about
   3.7 hours serial, run as three parallel groups).
2. **The experts, one family per pull request,** each with its re-record:
   the aircraft guidance (heading fix, L1, TECS, formation hold), then the
   process loops (SIMC retunes, the CSTR cascade, pH, Guideline 36, the wind
   torque law, the reactor step, the boiler), then the plants whose practice
   is an MPC, after the experts refactor's PR 2.

No RL result has been published, so nothing needs migrating.

## Risks

- Some practice claims rest on judgment rather than an opened source: the CSTR
  and unstable CSTR, distillation, pH, drum level, anti-surge, HVAC, battery
  and wind rows. The sources below are the ones that were checked.
- The aircraft sanity targets come from small UAVs and certification
  allowances, not from our aircraft in our turbulence.
- The fixed settle comes from the shipped controllers' measured settles, so it
  moves when an expert is rebuilt, and the holds must be re-scored then.
- On plants whose main lever is consumption, `S` measures only half of what a
  policy does; the consumption ratio carries the rest.

## Sources

Checked by reopening each page:

- Process control practice: [Desborough and Miller, PID prevalence](https://skoge.folk.ntnu.no/prost/proceedings/cpc6-jan2002/desborough.pdf),
  [Qin and Badgwell, MPC survey](https://skoge.folk.ntnu.no/prost/publications_various/mpc_survey_qin.pdf),
  [SIMC tuning](https://skoge.folk.ntnu.no/vgprosessregulering/papers-pensum/PIDbook-chapter5.pdf),
  [distillation control](https://skoge.folk.ntnu.no/publications/2007/skogestad_distillation-dos-and-donts_special-issue-of-cherd/old-submitted/submitted-aug06/dist06-extended.pdf),
  [pH control](https://www.plantservices.com/articles/2006/026),
  [glass furnace control](https://www.irsm.cas.cz/materialy/cs_content/2006/Havel_CS_2006_0000.pdf),
  [cement kiln expert control](https://www.redalyc.org/journal/3442/344272383007/344272383007.pdf),
  [drum level](https://controlguru.com/dynamic-shrinkswell-and-boiler-level-control/),
  [anti-surge](https://web-material3.yokogawa.com/Surge_Prevention_Control_on_Yokogawa_Stardom_Platform.us.pdf),
  [Guideline 36 terminal units](https://simulationresearch.lbl.gov/modelica/releases/v9.0.0/help/Buildings_Controls_OBC_ASHRAE_G36_PR1_TerminalUnits.html),
  [wind turbine control](https://skoge.folk.ntnu.no/prost/proceedings/acc09/data/papers/0628.pdf),
  [PWR load following](https://www.framatome.com/solutions-portfolio//app/uploads/2025/09/a0655-p-ge-g-en-0655-201901-advanced-load-following-control.pdf),
  [quadruple tank](https://strathprints.strath.ac.uk/42140/1/quad_tank_rk.pdf).
- Aircraft practice: [RVSM altitude keeping](https://www.law.cornell.edu/cfr/text/14/appendix-G_to_part_91),
  [TECS patent](https://patents.google.com/patent/US4536843),
  [ArduPilot TECS tuning](https://ardupilot.org/plane/docs/tecs-total-energy-control-system-for-speed-height-tuning-guide.html),
  [PX4 fixed-wing tuning](https://docs.px4.io/main/en/config_fw/position_tuning_guide_fixedwing),
  [L1 guidance flight data](https://ocw.mit.edu/courses/16-886-air-transportation-systems-architecting-spring-2004/dd34235f104c18b23a872509e288aa08_15_guidance.pdf),
  [AFIT formation hold](https://scholar.afit.edu/etd/6427),
  [NASA F/A-18 formation flight](https://ntrs.nasa.gov/citations/20030005820),
  [WVU three-aircraft formation](https://cdn.intechopen.com/pdfs/5974/InTech-Autonomous_formation_flight_design_and_experiments.pdf).
