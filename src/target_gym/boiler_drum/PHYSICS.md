# Boiler drum: model, sources, limits and deviations

Version 4, rebuilt on 2026-11-08 under charter Amendments 1 and 2. The plant
record is `REALITY.md`, and its cold review `COLD_REVIEW.md`. Source tags
[S1] to [S21] are REALITY.md's. Nothing here is compared with an earlier
version of the task (rule 11).

## 1. The job

REALITY.md sec. 6: keep the drum water between the low trip that starves the
tubes and the high trip that floods the turbine while the load moves, using
feedwater and firing. The unit runs boiler-following [S10]: the turbine valve
follows the load plan, and the boiler holds steam pressure for it by firing.

Named property (rule 18): **shrink and swell**, the inverse response of the
level to steam and feedwater moves. It is in the model through the steam
distribution in the risers and under the water surface, Astrom and Bell's
fourth-order model.

## 2. Model

The plant is the four-state model of [S1], ported from the authors' own
simulation code (S1 Appendix B, `sfd302`) with the P16-G16 numbers in it.
States: drum pressure `p`, total water volume `V_wt`, riser outlet quality
`alpha_r`, steam volume under the water surface `V_sd`. The code solves
`e @ dx = b` for the four derivatives:

```
e11 dV_wt + e12 dp                     = q_f - q_s                       (mass)
e21 dV_wt + e22 dp                     = Q + q_f h_f - q_s h_s           (energy)
            e32 dp + e33 dalpha_r      = Q - alpha_r h_c q_dc            (risers)
            e42 dp + e43 dalpha_r + e44 dV_sd
                 = rho_s (V_sd0 - V_sd) / T_d + (h_f - h_w) q_f / h_c    (drum)
q_dc^2 = k_e (rho_w - rho_s) V_r alpha_v,   T_d = 600 / q_s
level  = (V_wd + V_sd) / A_d - 1 m,   V_wd = V_wt - V_dc - (1 - alpha_v) V_r
```

with `alpha_v` the mean void fraction for a linear quality profile, and the
`e` coefficients exactly as S1 Appendix B writes them (eq. (25) in the report's
text for `e42`). The steam through the turbine valve is proportional to drum
pressure, `q_s = q_demand p / p_nominal` (Bell and Astrom 1987, eq. 14), with
`q_demand` the valve's flow at `p_nominal` on the load plan.

Saturated steam properties are quadratic fits to IAPWS values over 60 to 110
bar (residuals under 0.5 %, asserted in `test_boiler_drum_env.py`), in place of
S1's own fits around 10 MPa, which the scan does not print legibly.

## 3. Numbers

| Symbol | Value | Unit | Source |
|---|---|---|---|
| `V_t`, `V_d`, `V_r`, `V_dc` | 85, 37, 37, 11 | m^3 | [S1] Appendix B, P16-G16 |
| `A_d` | 23 | m^2 | [S1] Appendix B |
| `m_metal`, `m_riser`, `m_drum` | 300 000, 100 000, 20 000 | kg | [S1] Appendix B |
| `C_metal` | 650 | J/(kg K) | [S1] Appendix B |
| `k_e` | 200 | - | [S1] Appendix B (`ke`) |
| `V_sd0` | 8 | m^3 | [S1] Appendix B (`Vsd0`) |
| `beta` | 0.3 | - | [S1] Appendix B |
| `T_d_flow` | 600 | kg | [S1] Appendix B, `Td = 600 / qs` |
| `normal_level_height` | 1.0 | m | [S1] Appendix B, `l = lw + ls - 1.0` |
| `p_nominal` | 85 | bar | [S1] Appendix B, the steam transducer's 8.5 MPa |
| property fits | see params | - | IAPWS saturation values, least squares |
| `h_feedwater` | 1.037 | MJ/kg | Ours: feedwater at 240 C; S1 reads it as an input |
| `Q_nominal`, `q_steam_nominal` | 160 MW, 93.35 kg/s | - | Ours: S1's 160 MW rating taken as the heat input; `Q = q_s (h_s - h_f)`. S1's high-load tests ran at 100 to 112 kg/s (Fig. 15), so this unit is the smaller |
| `Q_max` | 240 | MW | Bell and Astrom 1987: the fuel span is about 1.5 times nominal firing |
| `q_feed_max` | 200 | kg/s | Ours: Bell and Astrom's span is 140 kg/s; 200 leaves the feed pump headroom at the 112 kg/s of Fig. 15 |
| `load_range` | 0.40 to 1.0 | of nominal | [S17] Table 1, hard coal minimum load 25 to 40 %; three-element control holds above 20 to 30 % [S12] |
| `ramp_rate_range` | 1 to 2 | %/min | [S10] |
| `ramp_start_range` | 300 to 900 | s | Ours: the slowest full ramp, 60 % at 1 %/min, ends inside the hour |
| `max_steps_in_episode` | 1800 | steps | Ours: one hour around one planned load change; [S3] has setpoints "change several times during every 24 h" |
| `level_trip_low` | -0.330 | m | [S9], "Drum Level Low-low (-330 mm)" |
| `level_trip_high` | +0.254 | m | [S20], "Drum Level HH (+254 mm)" |
| `level_trip_delay` | 10 | s | [S9] items 23 and 24, "for more than ten seconds"; [S21] on why trips are delayed |
| `restart_steps` | 4500 | steps | [S17] Table 1, hard coal hot start 2.5 to 3 h, low end, at 2 s steps |
| `failure_cost` | 0.1867 | t per step | `q_steam_nominal` for one 2 s step: the steam a restart does not make |
| `e_floor_level`, `e_floor_pressure` | 2.67e-3 m, 0.05 bar | - | Ours, read only by the time-in-band KPI tables, never by the score |
| `delta_t` | 2 | s | Ours: S1 logged at 0.1 Hz and shows the swell within one or two samples, so 2 s resolves it |

Actuators (shared `actuate` stage, Bell and Astrom 1987, section 4 and
appendix E): fuel rate 0.007 of span per s, feedwater rate 0.05 of span per s,
feedwater lag 1 s. The firing lag of 20 s is **ours** (burners, furnace gas and
tube walls on oil); S1's fuel step test (Fig. 10, test 7) still shows its rapid
swell through it.

Readings: level 0.5 mm (Rosemount 5300), pressure 0.022 bar (Yokogawa
EJA530E), steam flow 0.047 kg/s (Rosemount 8800D, not checked); none has a dead
time. These are datasheet figures outside REALITY.md. With the score in steam
off the plan, resting on plan costs a learner `0.047 sqrt(2 / pi) * 2e-3 =
7.5e-5` t per step through the steam meter alone.

## 4. Limits (rule 14)

| Limit | Kind | Model |
|---|---|---|
| Low level, -330 mm [S9] | Trip | Past it for more than 10 s [S9]: master fuel trip, a hot restart |
| High level, +254 mm [S20] | Trip | Past it for more than 10 s [S9]: turbine trip against carryover [S12], a hot restart |
| Drum pressure | None | REALITY.md found no sourced trip value or safety valve setting, so none is modelled. Off 60 to 110 bar the property fits extrapolate |
| Firing and feedwater ranges | Actuator range | `Q_max`, `q_feed_max` |

A trip charges `restart_steps * failure_cost = 840 t` of steam, the nominal
steam of a 2.5 h hot start, and the plant restarts at a new planned load
(`base.failure_kernel`).

## 5. Observation, episode and score

**Observation** (rule 16): drum level, drum pressure, steam flow (the three
elements [S5, S12]), the two actuator positions, the level target (zero), the
pressure setpoint, and the load plan: the valve's load now, the load it ramps
to and the seconds until the ramp starts. The plan is known to the unit master
in coordinated operation [S10]. Riser quality and the steam under the surface
are hidden.

**Episode** (rule 17): established operation. The unit holds a planned load
drawn from 40 to 100 % of nominal, balanced at normal water level, and the plan
ramps once, at 1 to 2 %/min [S10], to another load in that range, starting 5 to
15 minutes in. Nothing else moves: REALITY.md shows no unmeasured disturbance
on a utility unit, and runbacks [S10] have no sourced frequency.

**Score** (rule 15), in tonnes of steam: each step charges the steam delivered
off the plan, `|q_s - q_demand| * delta_t / 1000`, which in boiler-following is
the pressure error priced as production; a trip charges 840 t. Level precision
inside the trips earns nothing. The floor of the score, in closed form: steam
on plan costs zero, so the floor of the score is 0; rule 9's room needs an oracle, not run
yet ("not measured").

## 6. Behaviour tests (charter step 3)

`tests/boiler_drum/test_boiler_drum_reality.py` runs the cold review's eight
tests with its tolerances. On the rebuilt plant:

| Test | Source | Result |
|---|---|---|
| 1. Swell on a medium-load steam step | S1 Figs. 12, 16 | Pass |
| 2. Pressure ramp on that step | S1 Fig. 12 | **Fails**: 5.5 bar in 250 s (in 3 to 9), but the second 125 s fall 0.69 of the first, against at least 0.70 |
| 3. Swell per kg/s larger at medium load | S1 p. 26 | Pass |
| 4. Swell for a 10 %-of-load step grows toward low load | S1 p. 20, S2, S11 | **Fails**: 20, 22, 24 mm at 50, 75, 100 % |
| 5. Shrink smaller and shorter than swell | S14 | **Fails**: shrink 27 mm, swell 24 mm |
| 6. Feedwater steps: inverse response, +-0.18 m | S1 Fig. 11 | Pass |
| 7. Rapid swell on a fuel step | S1 Fig. 10 | Pass |
| 8. A 10 %/min ramp roughly halves the step's level deviation | S2 sec. 5.2 | **Fails**: ratio 0.31, against 0.35 to 0.70 |

Tests 2 and 6 differ from the audit's text in one respect, written in the
tests: the audit balances the plant at 52 kg/s and steps around it, while the
figures cycle about the middle of their range. At 52 kg/s test 6's mean feed
exceeds the steam by 8 kg/s, which climbs the level 0.5 m in 1000 s on any drum,
and test 2's step doubles the figure's energy deficit (10.8 bar). The
tolerances are the audit's.

The four failures are on S1's own model with the authors' own numbers, so no
sourced physics is missing that would change them (rule 2 forbids moving a
number). They are known deviations (rule 13):

- Test 4: per kg/s the swell is larger at low load (test 3), but a step that
  scales with the load swells less at low load. Effect on the score: low-load
  ramps are milder than S11 reports, so high-level trips at low load are rarer.
- Test 5: shrink is about 10 % larger than swell. Effect: a load decrease
  brings the level nearer the low trip than practice reports, where the low
  trip has the wider margin (330 against 254 mm).
- Tests 2 and 8: within a few per cent of their bounds; the pressure ramp
  curves slightly more than "roughly linear", and a ramp helps a little more
  than S2's larger unit.

## 7. Deviations still in the model

- The feedwater flow loop is perfect: the action is flow, as a cascade's
  setpoint, with no pump or valve characteristic (cold review 7).
- The level reading is white noise with no delay; S3 fits coloured noise from
  surface boiling and carries a delay whose units it does not state (cold
  review 6).
- No runback, fan trip or feed pump trip (cold review 8): their frequency is
  not found.
- Pressure is held at one setpoint; sliding pressure [S2] is not modelled.
- A restart is instant at a new load; a real one runs through single-element
  start-up with the level lowered for swell [S5, S19].

## 8. What the reference ladder's row needs

- Inputs: the load plan in the observation (channels 7 to 9) as the
  steady-state feedforward rung: steam feedforward to feedwater, `q_s (h_s -
  h_f)` to firing.
- Bump tests at loads across 40 to 100 %, since the plant's gains change with
  load.
- Its cost is now tonnes of steam off plan, with trips at 840 t; a rung that
  trips any nominal seed is dropped.
- Seeds 100 to 109 each carry their own plan; the ladder must not assume
  nominal load.
