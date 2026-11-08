# Wind turbine: model, sources, limits and deviations

Rebuilt on 2026-11-08 under the charter's Amendments 1 and 2 (version 4). Tags such as
[S1] point to the sources in `REALITY.md`; S16 to S19 were added by this rebuild.
Every number is cited or marked Ours with its reason. Nothing here is compared with
an earlier version of the task (rule 11).

## 1. The job

Above rated wind, pitch holds rated speed and torque holds rated power, within the
pitch, torque and speed limits, while the tower and drivetrain are damped (REALITY.md
sec. 6, [S1 Sec. 7], [S5]). The named property (charter, "The tasks") is tower motion:
the rotor's thrust swings the tower, and pitching to hold speed changes that thrust.

## 2. Model

A rigid drivetrain on the tower's first fore-aft mode, both driven by the rotor in the
wind it sees, the free wind less the tower top's velocity:

    J dw/dt   = tau_aero - N tau_gen
    m x''     = F_thrust - c x' - k x
    v_rel     = v - x'
    tau_aero  = 0.5 rho A v_rel^3 Cp(lambda, beta) / w,   lambda = w R / v_rel
    F_thrust  = 0.5 rho A v_rel^2 Ct(lambda, beta)
    P_elec    = eta N tau_gen w

The structure is S16's Eq. (1) without the blade's flap mode, with thrust from ROSCO's
Ct table [S18] and torque from its Cp table [S18] read bilinearly and held at their
edges. `k = m (2 pi f)^2` places the mode at S1's frequency; `c = 2 zeta sqrt(k m)`.
The relative wind is what gives the tower its aerodynamic damping and couples pitch
to tower motion. Pitch and torque move toward their commands at their rate limits
and stop on them (S1's pitch actuator was set at 30 Hz "to be negligible" [S1 Sec. 7.4];
CART's torque loop is 500 rad/s [S3]). Integration: RK4, two sub-steps per 0.25 s step.

Wind: an Ornstein-Uhlenbeck deviation about the episode's mean with IEC's
sigma1 = Iref (0.75 V + 5.6) [S13] and the Kaimal integral time scale L_u / V, L_u =
8.1 x 42 m [S13, S19] (23 s at 15 m/s; an OU process and the Kaimal spectrum share
their value at zero frequency, 4 sigma^2 T). One extreme operating gust can be set
(`gust_start`), with IEC's magnitude and shape over 10.5 s [S13].

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
| `tower_damping_ratio` | 0.01 | [S1 Sec. 6]: 1 % of critical, isolated tower |
| tower stiffness | 1.81e6 N/m | derived, m (2 pi 0.324)^2; S16's beam estimate is 1.98e6 |
| tower damping | 17,800 N s/m | derived, 2 x 0.01 sqrt(k m) |
| Cp, Ct tables | 26 x 31 nodes, 0 to 30 deg | [S18] |
| `pitch_min`, `pitch_max` | 0, 90 deg | [S1 p. 26] |
| `pitch_rate_max` | 8 deg/s | [S1 Table 7-2] |
| `torque_max`, `torque_rate_max` | 47,402.91 N m, 15,000 N m/s | [S1 p. 20, Table 7-2] |
| `v_mean_range` | 13 to 25 m/s | [S2 Table 2], its above-rated load cases |
| `turbulence_iref` | 0.14 | Ours: class B; the turbine's class is not found in S1, owner to choose |
| `turbulence_lambda1`, `kaimal_factor` | 42 m, 8.1 | [S13], [S19] |
| `v_ref` | 50 m/s | Ours: class I [S13]; only the gust's other branch reads it, which never binds |
| `gust_start` | -1 (none) | [S13]: the EOG is a 1-year event, absent from a normal 10 min |
| `overspeed_factor` | 1.20 | [S17], "approximately 20 %" (secondary, approximate) |
| `rating` | 1.0 | [S1 Sec. 7], a normal day at rated; ROSCO's R [S9 Ex. 29] |
| `delta_t` | 0.25 s | Ours: commercial rates not found; S1 runs 80 Hz, CART 100 Hz |
| `max_steps_in_episode` | 2400 | 10 min: a SCADA interval [S7], a load case [S2] |
| `restart_steps` | 1600 | 400 s, ROSCO's start-up demo [S9 Ex. 32] |
| `failure_cost` | 3.4722e-4 MWh | rated power for one 0.25 s step |
| `reward_version` | 3 | the job's score (rule 15) |
| `e_floor`, `precision_floor` | 1680 W, 1700 W | not read by the score; kept for the KPI tables and the oracle's resolution (a class 0.1S meter at 5 MW) |
| sensor sigmas | 5.2e-4 rad/s, 0.2 deg, 90 N m, 1 kW | Ours for this task: the Odgaard benchmark's simulation inputs, not in REALITY.md |
| anemometer sigma | 0.36 m/s | [S6], the transfer function's residual RMSE |

## 4. Limits (rule 14)

| Limit | Kind | Model | Source |
|---|---|---|---|
| Overspeed, 1.20 x rated | Trip (safety system) | past it the pitch feathers at 8 deg/s whatever the controller sends; the plant is lost and restarts, at `restart_steps` x `failure_cost` = 0.556 MWh | [S4], [S17], [S9 Ex. 32] |
| Pitch 0 to 90 deg, 8 deg/s | Protected (actuator) | saturation, no cost | [S1 p. 26, Table 7-2] |
| Torque 47,402.91 N m, 15,000 N m/s | Protected (converter) | saturation, no cost | [S1 Table 7-2] |
| Rated power | regulated set point | the score | [S1 p. 19] |
| Low rotor speed | none | not a trip: no source calls it one | REALITY.md sec. 3 |
| Tower fatigue (DEL) | cost | not modelled: no source prices it | [S5 Sect. 1] |
| Cut-out 25 m/s, grid frequency, ramp rates | not modelled | no grid and no 10 min mean in the task | [S1], [S10], [S12] |

## 5. Observation, episode and score

Observation (rule 16): `[omega_rpm, pitch_deg, torque_pct, P_MW, rating_P_MW, a_fa,
v_nacelle]`. Speed, pitch, torque and power are what S1's controller reads; the
rating's power is a set point known ahead [S9 Ex. 29]; `a_fa` is the nacelle
accelerometer's fore-aft reading in m/s^2 [S3, S5], noise not found so none; the
nacelle anemometer reads the free wind with 0.36 m/s scatter [S6]. Measurement delay:
not found, none modelled.

Episode (rule 17): 10 min of established operation. The mean wind is drawn from 13 to
25 m/s [S2]; the rotor starts at rated speed delivering the rating, at the pitch that
balances it in the starting wind, with the tower at rest at its static deflection.
Only the turbulence moves; the rating holds for the whole episode.

Score (rule 15): the energy delivered off the rating, `abs(P_ref - P) x dt`, in MWh,
above or below alike: rated power is the regulated set point [S1 p. 19], so holding it
is the job; less is lost production and more overloads the generator. A trip charges
the production of a 400 s restart. Rotor speed inside the trip earns nothing. The floor
of the score, in closed form: 0 on the true state; the measured reward adds the meter's
scatter, `E|N(0, 1 kW)|` = 798 W, 5.5e-8 MWh per step.

## 6. Behaviour tests (charter step 3)

`tests/energy/wind_turbine/test_wind_turbine_reality.py`, the audit's twelve with its
tolerances. All twelve failed on the version-3 plant except test 1.

| Test | Result | Measured |
|---|---|---|
| 1 Rated torque | pass | |
| 2 Peak Cp 0.482 at 7.55 | fail | 0.4659 at 7.5: ROSCO's table, not S1's |
| 3 Region 3 hold, published gains | pass | 13, 16, 20, 25 m/s |
| 4 Wind step response | fail | peak 2.5 s against 7.33 s +/- 25 %; overshoot 13.4 % passes |
| 5 Sensitivity doubles at 6.3 deg | fail | theta_K 0.12 deg (D1) |
| 6 Actuator saturations | pass | |
| 7 Tower mode 0.324 Hz | pass | 0.323 Hz |
| 8 Tower loop suppresses the peak | fail | power cut 5.8 times against 10; pitch 0.08 deg RMS passes |
| 9 Turbulence grows with wind | pass | |
| 10 EOG amplitude and duration | pass | |
| 11 Overspeed feathers | fail | pitch climbs at 8 deg/s to 90 deg as required; speed falls from 1.52 to 0.843 rad/s, then wobbles 1.3e-4 rad/s at the feathered idle |
| 12 De-rate to 0.9 | pass | |

Test 8's loop gain is ours (10 % of critical damping added through the thrust's
sensitivity to pitch, fixed before the run): S5 tuned by hand and printed no gain.
Test 11's last clause reads the feathered idle: once at 90 deg the table, held at its
30 deg column, leaves a small positive Cp at low tip speed, so the rotor idles rather
than stops (deviation D3). Test 12 assumes, for want of ROSCO's formula, that the
speed reference stays rated and the torque reference falls with the rating.

The job tests (`test_wind_turbine_job.py`) check the trip level, the trip's price, the
score's units, the observation's channels and the episode's start; they fail on the
version-3 plant (16 of 18) and pass.

## 7. Deviations still in the model

- D1. The steady Cp table's power sensitivity to pitch runs nearly through the origin
  (theta_K 0.12 deg against S1's 6.3 deg), because S1 linearised with the wake frozen.
  The published gains still hold region 3 (test 3), but regulate faster than placed
  (test 4).
- D2. One tower mode: no side-side mode (no lateral force in the model), no second
  fore-aft mode (2.9 Hz, above the 2 Hz Nyquist rate), no drivetrain torsion, no blade
  flap, no 1p or 3p loading.
- D3. The tables stop at 30 deg of pitch and hold their last column beyond it, so a
  feathered rotor idles instead of braking to a stop.
- D4. The turbulence is a single rotor-effective point wind with an OU spectrum (-2
  slope against Kaimal's -5/3), no rotor averaging or coherence.
- D5. Tower fatigue is not in the score: REALITY.md classes it as a cost but no source
  prices it. So the property is in the plant and the readings, and reaches the score
  only through the thrust's coupling to rotor speed and power. The owner may price it.
- D6. The overspeed level is S17's approximate general figure, not this turbine's; and
  ROSCO's start-up (400 s) is a demo, not a commercial restart time.
- D7. Cut-out at 25 m/s is not modelled: an episode at a 25 m/s mean keeps running.
- D8. The sensor sigmas other than the anemometer's are not in REALITY.md.
- D9. No sensor delays (not found).

## 8. What the reference ladder's row needs

`scripts/reference_ladder.py`'s `row_wind_turbine` (not edited here) has to change for
v4: the score is MWh off the rating with no tracking floor and no dollars; the target
is the constant rating, not a dispatch, so the bump tests should step the wind (or the
pitch) rather than the target; the observation has seven channels (indices 0 to 4
keep their meaning); the trip is the feather at 1.20 x rated, with no underspeed trip;
the episode is 2400 steps. Its cascade rung can use the accelerometer (channel 5) as the
available measurement for a tower-velocity loop (S5's form). The oracle's row is
"not measured" until it is run on this plant.
