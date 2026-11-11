# Wind turbine: model, sources, limits and deviations

Version 5, 2026-11-08: the charter's Amendment 3 applied to the version 4 rebuild. The
task represents the family of utility-scale, pitch-regulated, three-bladed upwind
turbines; the NREL 5 MW is the member it is built on. Tags such as [S1] point to
`REALITY.md`. Every number is cited, or marked ours with the family range it sits in.
Nothing is anchored on an earlier version of the task (rule 11).

## 1. The job

Above rated wind, pitch holds rated speed and torque holds rated power, within the
pitch, torque and speed limits, while the tower is damped (REALITY.md sec. 6, [S1 Sec.
7], [S5]). The named property is tower motion: thrust swings the tower, and pitching to
hold speed changes the thrust. Version 5 prices that trade: lost production against
the tower's fatigue and the pitch system's wear, all in dollars.

## 2. Model

A rigid drivetrain on the tower's first fore-aft mode, both driven by the rotor in the
wind it sees, the free wind less the tower top's velocity:

    J dw/dt   = tau_aero - N tau_gen
    m x''     = F_thrust - c x' - k x
    v_rel     = v - x'
    v_eff     = v_rel - (w_i - w_qs)
    tau_aero  = 0.5 rho A v_eff^3 Cp(lambda, beta) / w,   lambda = w R / v_eff
    F_thrust  = 0.5 rho A v_eff^2 Ct(lambda, beta)
    dw_i/dt   = (w_qs - w_i) / tau1,   w_qs = a v_rel,   a from Ct by momentum theory
    tau1      = 1.1 / (1 - 1.3 min(a, 0.5)) R / v_rel                     (Oye, [S22])
    P_elec    = eta N tau_gen w

**Dynamic inflow** (owner, 2026-11-08). The tables hold equilibrium induction, so
the rotor-average induced velocity w_i (the state's `w_induced`) is a state that lags its quasi-steady value
`w_qs` by Oye's tau1, as OpenFAST's DBEMT writes it [S22]. The tables are read at the
wind that gives the rotor plane its lagging inflow, so a wind or pitch step changes
the loads at once and then relaxes as the wake catches up. `a` comes from Ct by
momentum theory up to Ct 0.96 and by Buhl's correction above it, which joins it in
value and slope [S22]. One lumped lag: Oye's second filter tau2 and its radial split
are not modelled. tau1 is 9 s at rated wind (11.4 m/s) and 3 s at 25 m/s (ours from
the formula, family range 3 to 11 s over 11 to 25 m/s for a = 0.1 to 0.33).

The structure is S16's Eq. (1) without the blade's flap mode, with thrust from ROSCO's
Ct table [S18] and torque from its Cp table [S18] read bilinearly and held at their
edges up to 30 deg; past it, to 90 deg, Cp carries on along each row's slope over
its last 5 deg and Ct falls linearly to 0 (ours, D3), so a feathered rotor brakes.
`k = m (2 pi f)^2` places the mode at S1's frequency; `c = 2 zeta sqrt(k m)`.
The relative wind is what gives the tower its aerodynamic damping and couples pitch
to tower motion. Pitch and torque move toward their commands at their rate limits
and stop on them (S1's pitch actuator was set at 30 Hz "to be negligible" [S1 Sec. 7.4];
CART's torque loop is 500 rad/s [S3]). Integration: RK4, two sub-steps per 0.25 s step.

Wind: an Ornstein-Uhlenbeck deviation about the episode's mean with IEC's
sigma1 = Iref (0.75 V + 5.6) [S13] and the Kaimal integral time scale L_u / V, L_u =
8.1 x 42 m [S13, S19] (23 s at 15 m/s; an OU process and the Kaimal spectrum share
their value at zero frequency, 4 sigma^2 T). A scenario may add one extreme operating
gust, with IEC's magnitude and shape over 10.5 s [S13], or a 10 % curtailment, each at a
time the seed draws.

Cost (rule 7): the tower adds two states and a second table lookup per evaluation,
and no sub-stepping: 0.25 s is 12 steps per tower period, and RK4 at a 0.125 s
sub-step has omega h = 0.25 on the mode. Measured with `benchmark_speed`'s
`benchmark_env` (batch 256, CPU, 400 steps, best of 3, on a loaded machine), the
step runs at 0.052 of `cstr`'s rate against 0.057 for version 3, about 1.1 times the
old cost. The reset's balancing pitch is two short scans (0.1 deg, then 0.01 deg),
since every step's restart branch runs it; a single 0.01 deg scan made the step 8
times slower.

## 3. Numbers

| Name | Value | Source |
|---|---|---|
| `R`, `rho` | 63 m, 1.225 kg/m^3 | [S1 Table 1-1] |
| `J_rotor` | 38,759,228 kg m^2 | [S16 App. B] gives 38,759,227 |
| `J_gen`, `N_gear`, `eta_gen` | 534.116 kg m^2, 97, 0.944 | [S1 Table 5-1, p. 19] |
| `P_rated`, `v_rated`, `omega_rated_rpm`, `v_cut_out` | 5 MW, 11.4 m/s, 12.1 rpm, 25 m/s | [S1 Table 1-1] |
| `tower_frequency_hz` | 0.3240 Hz | [S1 Table 9-1], FAST |
| `tower_modal_mass` | 436,865 kg | [S16] rule, 110,000 + 240,000 + 0.25 x 347,460 kg [S1 Table 1-1] |
| `tower_damping_ratio` | 0.01 | [S1 Sec. 6] |
| `tower_height` | 87.6 m | [S1 Table 6-2] |
| Cp, Ct tables | 26 x 31 nodes, 0 to 30 deg | [S18] |
| Cp, Ct past 30 deg | to 90 deg | ours: every family table stops at 30 deg [S18 family] |
| `pitch_min`, `pitch_max` | 0, 90 deg | [S1 p. 26] |
| `pitch_rate_max` | 8 deg/s | [S1 Table 7-2] |
| `torque_max`, `torque_rate_max` | 47,402.91 N m, 15,000 N m/s | [S1 Table 7-2] |
| `v_mean_range` | 13 to 25 m/s | [S2 Table 2], the above-rated load cases |
| `turbulence_iref` | 0.14 | IEC class B [S13], a member of the family (class A, 0.16, is a scenario) |
| `turbulence_lambda1`, `kaimal_factor` | 42 m, 8.1 | [S13], [S19] |
| `v_ref` | 50 m/s | ours, class I of 37.5 to 50 m/s [S13]; the branch it feeds never binds |
| `gust`, `gust_window` | 0; 60 to 580 s | [S13]: a 1-year event; the window is ours (any time a whole gust fits after a minute) |
| `derate_to`, `derate_window` | 1.0; 60 to 300 s | [S11 Fig. ES-8]; the 10 % size of the scenario is S11's AGC de-rate (p. xix); the window is ours |
| `overspeed_factor` | 1.20 | [S17], "approximately 20 %" |
| `rating` | 1.0 | [S1 Sec. 7], [S9 Ex. 29] |
| `delta_t` | 0.25 s | ours, family range 80 Hz [S1] to 100 Hz [S3] for controllers; 12 steps per tower period |
| `max_steps_in_episode` | 2400 | 10 min: a SCADA interval [S7], a load case [S2] |
| `reward_version` | 4 | the v5 score in dollars |
| `energy_price` | 60 $/MWh | ours, family LCOE range 60 to 140 $/MWh [S20 Table 1] (NREL's 39 to 42 $/MWh not read, [S20] note) |
| `capex_per_kw` | 1750 $/kW | ours, family range 1,300 to 2,200 $/kW [S20 Table 1] |
| `turbine_share` | 0.75 | ours, family range 0.65 to 0.84 [S20 Table 4.1] |
| `tower_share` | 0.263 | [S20], REpower MM92 |
| `pitch_share` | 0.0266 | [S20], REpower MM92 |
| `design_life_years` | 20 | [S20] |
| `dynamic_inflow` | 1 | 1 the lag on, 0 the quasi-steady tables |
| `oye_k1`, `oye_k2` | 1.1, 1.3 | [S22], Oye's tau1 |
| `fatigue_exponent` | 4 | cited: the average slope of the two regimes, 3 and 5, of the DNV-RP-C203 type E curve (Eurocode 3 detail category 80) for welded tower joints, FLOAT's default [S23, sec. 3.5.1, 3.7.1]. The bilinear curve itself can't be used: the DEL is on the tower-top deflection's base moment, not on stress, so there is no stress level to place its knee at 1e7 cycles |
| `tower_damage_design` | 2.126e5 (MN m)^4 per step | ours: measured once, S1's controller, published gains, seeds 100 to 107 of the normal scenario |
| `pitch_travel_design` | 0.1754 deg per step | ours: the same runs |
| `restart_steps` | 28800 | ours, 2 h: an overspeed trip waits for a manual reset [S21]; family range from ROSCO's automatic 400 s [S9 Ex. 32] to a site visit (not found) |
| `reset_callout` | 1000 $ | ours: no family figure found |
| `failure_cost` | 0.05556 $ per step | rated production at `energy_price`, 0.02083, plus `reset_callout` over `restart_steps`, 0.03472 |
| `e_floor`, `precision_floor` | 1680 W, 1700 W | not read by the score |
| sensor sigmas | 5.2e-4 rad/s, 0.2 deg, 90 N m, 1 kW | ours, the Odgaard benchmark's inputs (not in REALITY.md) |
| anemometer sigma | 0.36 m/s | [S6] |

Design rates: reproduced by running S1's
baseline controller (`experts.nrel_baseline_step` with Kp 0.01882681, Ki 0.008068634,
theta_K 6.302336) through `compute_next_state` for 2400 steps on seeds 100 to 107 and
averaging `damage` and `pitch_travel`. Choosing them picks the family member whose tower
and pitch system last exactly 20 years at that duty.

## 4. Limits (rule 14) and zones (rule 22)

| Limit | Kind | Model | Source |
|---|---|---|---|
| Overspeed, 1.20 x rated | Trip (safety system) | feather at 8 deg/s whatever the agent sends; restart priced at `restart_steps` x `failure_cost` = $1,600 | [S4], [S17], [S21] |
| Pitch, torque and their rates | Protected (actuator) | saturation, no cost | [S1 Table 7-2] |
| Rated power | regulated set point | shortfall against what the wind makes available is lost production; more is free | [S1 p. 19] |
| Tower fatigue | cost | DEL priced at the tower's cost share | [S5 Sect. 1], [S20] |
| Pitch wear | cost | travel priced at the pitch system's cost share | [S20] |
| Low rotor speed, cut-out, grid frequency, ramp rates | not modelled | no source makes them trips here | REALITY.md sec. 3 |

Zones: the task has no quality or comfort zone. Its one controlled variable with a hard
edge is rotor speed, and its cheap side is high: speed held up keeps the rotor's energy
for lulls and costs no production, while the overspeed trip at 1.20 x rated is the
expensive edge. The reference's speed setpoint under rule 22 is therefore at most
1.20 x rated less 2 sigma of the speed. S1's controller holds sigma at 3.1 to 6.2 % of
rated over the six scenarios (seeds 100 to 103), so the edge less 2 sigma is 1.08 to
1.14 x rated, above S1's rated setpoint: the recipe's rated setpoint already sits inside.

## 5. Observation, scenarios and score

Observation (rule 16): `[omega_rpm, pitch_deg, torque_pct, P_MW, rating_P_MW, a_fa,
v_nacelle]`: S1's controller readings, the current rating's power [S9 Ex. 29], the
nacelle accelerometer [S3, S5] (noise not found, none) and the nacelle anemometer [S6].
No delays (not found). A curtailment shows in channel 4 when it starts: advance notice
of a TSO set point is not sourced.

Episode (rule 17): a 10 min window of established operation; the reset balances the
rotor at rated speed delivering the rating with the tower at its static deflection.

Scenarios (rule 20), `scenarios.py`, each with its source:

| Scenario | Overrides | Source |
|---|---|---|
| `normal` | none | REALITY.md sec. 5, [S1 Sec. 7], [S2 Table 2], [S13] |
| `gusty_class_a` | `turbulence_iref` 0.16 | [S13] |
| `extreme_gust` | `gust` 1, timed by the seed | [S13] |
| `near_rated` | `v_mean_range` 11.4 to 14 m/s | [S1 Table 1-1], [S2 Sects. 5.2, 5.3] |
| `near_cut_out` | `v_mean_range` 21 to 25 m/s | [S1 Table 1-1], [S2 Table 2] |
| `curtailment` | `derate_to` 0.9, timed by the seed | [S10 Art. 13(6)], [S11 Fig. ES-8, p. xix], [S9 Ex. 29] |

Score (rule 15), dollars per step, each term >= 0:
- `lost`: MWh short of the available power, min(rating, the wind's power at the
  table's peak Cp), at `energy_price`. Over-production is free. The wind's own lulls
  cost nothing a controller could save.
- `tower`: the tower's cost (1750 $/kW x 5000 kW x 0.75 x 0.263 = $1,726,031) times
  the step's damage over the design damage of 20 years. Damage closes per half-cycle of
  the tower-base moment, `k x H`, between turning points: 0.5 (range / 1 MN m)^4.
- `pitch`: the pitch system's cost ($174,563) times the step's travel over the design
  travel of 20 years.
- `failure`: a trip, $1,600.

S1's controller, seeds 100 to 103, per 10 min: lost $0.26 to $1.35, tower $0.39 to
$4.36, pitch $0.14 to $0.20, no trips, over all six scenarios. So the tower is the
largest cost at high wind and lost production the largest near rated: rule 23's cost is
present. The floor of the score, in closed form: 0, at the available power with the
tower still and the pitch still; the measured reward adds the meter's scatter on
`lost`, at most E|N(0, 1 kW)| = 798 W, $3.3e-6 per step.

## 6. Behaviour tests (charter step 3, Amendment 3)

`tests/energy/wind_turbine/test_wind_turbine_reality.py`. Tests 2, 4, 5 and 8 were
restated as family ranges or properties, written in the test before the rerun.

| Test | Result | Measured |
|---|---|---|
| 1 Rated torque | pass | |
| 2 Peak Cp in 0.45 to 0.49, at a TSR of 7 to 10.5 (five rotors) | pass | 0.466 at 7.5 |
| 3 Region 3 hold, published gains | pass | |
| 4 Wind-step peak time in 5.5 to 22 s (bandwidths 0.2 to 0.6 rad/s) | fail | 2.25 s with dynamic inflow (2.5 s without); overshoot passes |
| 5 Sensitivity to pitch grows along Region 3, at least doubling | pass | |
| 6 Actuator saturations | pass | |
| 7 Tower mode 0.324 Hz | pass | 0.323 Hz |
| 8 A tower-velocity loop takes power out of the tower band, more with gain, half at 10 % | pass | |
| 9 Turbulence grows with wind | pass | |
| 10 EOG amplitude and duration | pass | |
| 11 Overspeed feathers and the rotor slows | pass | |
| 12 De-rate to 0.9 | pass | |

Test 4 still fails with dynamic inflow (owner, 2026-11-08): the speed peak comes at
2.25 s, against 2.5 s without it. Oye's lag makes the first response to a wind step
sharper, the loads overshooting before the wake catches up, so it cannot slow the
peak. What sets 2.25 s is S1's gains on this rigid drivetrain; the published
bandwidths that give 7.3 to 22 s assume a flexible drivetrain and the blade modes
(D2). A finding, kept failing.

The job tests (`test_wind_turbine_job.py`) cover the trip, the dollar score's terms, the
half-cycle count on a sine, the curtailment's timing, the observation, the episode and
the scenario suite.

## 7. Deviations still in the model

- D1. Dynamic inflow is one lumped Oye lag (tau1 only, no radial split), and the
  oracle's planner keeps the quasi-steady tables (section 9).
- D2. One tower mode: no side-side, second fore-aft, drivetrain or blade modes, no 1p or
  3p loading. So blade-root loads are not priced, and S5's trade of 2.1 % tower DEL for
  1.26 % blade DEL is only half in the score; pitch wear stands in for the rest.
- D3. Past 30 deg the tables are extrapolated (ours), so a feathered rotor brakes to a
  stop rather than idling.
- D4. A single rotor-effective point wind, OU spectrum, no coherence.
- D5. Half-cycles are counted between turning points step by step, not by full
  rainflow: a large slow swing broken by small ones is counted as smaller ranges. On the
  narrow-band tower signal the two agree closely; on wind-driven slow swings this
  undercounts.
- D6. The overspeed level is S17's general figure; the restart time and the callout are
  ours.
- D7. Cut-out at 25 m/s is not modelled.
- D8. The sensor sigmas other than the anemometer's are not in REALITY.md.
- D9. No sensor delays (not found).
- D10. One violent tower half-cycle (about 130 MN m under random bang-bang pitch)
  costs $0.46, more than a step of trip downtime ($0.056). The owner restated the
  trip bound per episode on 2026-11-08 (`tests/test_failure_cost_bound.py`): an
  episode with a trip costs at least the $1,600 restart, more than the worst
  surviving 10 min, so tripping on purpose never pays. It passes.

**Cost (charter J6).** The lag adds one state. On bob, batch 256, 8 cores, in one
run of all 24 tasks: 0.31 M steps/s with it against 0.35 without (13 % slower),
and the median task runs 0.28 M steps/s, so the step is about the median's,
well inside 3 times it.

## 8. What the reference ladder's row needs

`scripts/reference_ladder.py`'s row (not edited here) needs: the dollar score
(`compute_reward_terms`: `lost`, `tower`, `pitch`, `failure`); the six scenarios from
`EnvSpec.scenarios`; bump tests that step the wind or the pitch (the rating is fixed
except in `curtailment`); speed setpoint at rated, inside the rule 22 back-off (section
4); the trip at 1.20 x rated; seven observation channels, 0 to 4 as before. Its cascade
rung can feed the accelerometer (channel 5) to collective pitch through a tower-velocity
loop of S5's form, which is what the score now pays for.

## 9. What the oracle needs

The oracle plans the env's own score: lost production against the available
power, the pitch travel and, for the ceiling, the tower's fatigue. The
half-cycle count reads turning points of the history, which a gradient planner
can't plan through, so the ceiling counts them outside the solver on each plan,
freezes them, and plans each frozen half-cycle as a smooth row of power m,
re-solving until the turning points stop changing (online rainflow counting,
Loew and Bottasso, WES 7, 1605, 2022). The registry's receding slot doesn't
plan the tower. Its widths file still predates v5 (`measure_oracle_widths.py`
marks the wind entry stale).

## Weights (charter J4)

Each scenario's real operating time, in hours a year (`scenarios.py`): normal
1800, near rated 1050, gusty class A 400, curtailment 260, near cut-out 130,
extreme gust 0.17 (one 10 min window a year, the gust being a one-year event
[S13]). The rest are ours, from a Rayleigh wind of mean 8.5 to 10 m/s, with the
family ranges in `scenarios.py`.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `wind_turbine` | 2400 | 0.25 | 10 min | 2 in [-1, 1] | 7 | 24 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->
