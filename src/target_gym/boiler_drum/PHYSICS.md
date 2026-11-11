# Boiler drum: model, sources, limits and deviations

Version 5, rebuilt on 2026-11-08 under charter Amendments 1 to 3. The plant
record is `REALITY.md`, and its cold review `COLD_REVIEW.md`. Source tags
[S1] to [S23] are REALITY.md's. Under Amendment 3 the task represents the drum
boiler family, not one unit: its numbers come from P16-G16 where S1 gives
them, and the rest are labelled "ours" with the family's range. Nothing here is compared with an earlier
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
| `ramp_rate_range` | 1 to 6 | %/min | Family range: 1 to 2 typical [S10], 1.5 to 6 for coal [S17] |
| `event_time_range` | 300 to 900 | s | Ours: inside the window, with an hour after it |
| `runback_load` | 0.50 | of nominal | Ours, family range 0.40 to 0.60: "below the three-mill minimum" [S10], minimum load 25 to 40 % [S17]; inside the 37 to 56 % one plant's ID fan and feed pump derates ran to (REALITY.md [ST1]) |
| `runback_rate_range` | 12 to 25 | %/min | Ours, family range from [S10]: a runback from 75 % load "in just over two minutes" |
| `feed_trip_capacity` | 0.5 | of nominal steam | Ours, family range 0.5 to 0.6: the second of two 50 or 60 % pumps |
| `standby_delay_range` | 30 to 60 | s | Ours, from [S11]: "more than 30 seconds" |
| `event_hold` | 9990 | s (stream, upsets) | 2.775 h, the median of six ID fan and feed pump forced derates at one plant, 0.35 to 67 h (REALITY.md [ST1]): how long a runback holds the unit back before it ramps up at its plan's rate; a window never sets it, so its runback holds to the end |
| `window_ramp` | on | - | The window's one planned ramp; a stream turns it off and plans each load change as an event |
| `steam_rate` | 3.5 | kg/kWh | Ours, family range 2.18 to 4.51 kg/kWh (ideal-cycle examples, [S23]) |
| `imbalance_price` | 80 | GBP/MWh | Ours, family range 30 to 156: 59 % of 2018's periods in 30 to 60, short-system mean 78.84 [S22]; annual means 44.61 (2017 to 2020) and 156.20 (2021 to 2022) [S22b] |
| `steam_resolution` | 0.047 | kg/s | The steam meter's scatter (Readings): off-plan steam it cannot see is not counted |
| `ramp_start_range` | 300 to 900 | s | Ours: the slowest full ramp, 60 % at 1 %/min, ends inside the hour |
| `max_steps_in_episode` | 1800 | steps | Ours: one hour around one planned load change; [S3] has setpoints "change several times during every 24 h" |
| `level_trip_low` | -0.330 | m | [S9], "Drum Level Low-low (-330 mm)" |
| `level_trip_high` | +0.254 | m | [S20], "Drum Level HH (+254 mm)" |
| `level_trip_delay` | 10 | s | [S9] items 23 and 24, "for more than ten seconds"; [S21] on why trips are delayed |
| `restart_steps` | 4500 | steps | [S17] Table 1, hard coal hot start 2.5 to 3 h, low end, at 2 s steps; one plant's trips came back in 0.87 to 11.3 h (REALITY.md [ST1]) |
| `restart_load` | 0.25 | of nominal steam | [S17] Table 1, hard coal minimum load 25 to 40 %, low end: the minimum fire a hot restart relights at (owner, 2026-11-11) |
| `failure_cost` | 4.267 | GBP per step | `q_steam_nominal` for one 2 s step, at `steam_rate` and `imbalance_price`: the generation a restart does not make |
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

A trip charges `restart_steps * failure_cost = GBP 19,203`, the generation of
840 t of nominal steam over a 2.5 h hot start. The unit restarts hot
(`restart_plant`; long-strata R8, owner 2026-11-11): purged and relit at nominal
pressure and normal water level, on minimum fire (`restart_load`, 25 % of
nominal steam), while the plan it tripped on carries on. The turbine then
reloads at the plan's ramp rate: the valve passes the smaller of the plan and
25 % plus the rate times the time since the restart (`pickup`), until it meets
the plan. The steam the valve does not pass is off plan, and the score charges
it; the policy holds pressure and level through the reload. The observation's
load now stays the plan's; the steam flow reading shows the valve's. The
boiler's expert reloads without a second trip from plan loads of 0.41 to 0.96,
its level within -0.09 to +0.12 m. Without the reload (the valve passing the
whole plan at once) the swell reached the high trip level above a 0.9 load.

**Zones and cheap side (rule 22).** The score prices no zone. The level window
between the trips, -330 to +254 mm, is a trip limit with no price inside it, so
the reference holds normal water level (0, inside the window's centre, 38 mm on
the high side) and has no cheap edge to back off from. Steam pressure is a
setpoint whose error is priced, not a zone.

## 5. Observation, episode and score

**Observation** (rule 16): drum level, drum pressure, steam flow (the three
elements [S5, S12]), the two actuator positions, the level target (zero), the
pressure setpoint, and the load plan: the valve's load now, the load it ramps
to and the seconds until the ramp starts. The plan is known to the unit master
in coordinated operation [S10]. Riser quality and the steam under the surface
are hidden.

**Upsets.** The observation adds the firing and feedwater caps (channels 10,
11), as the fan and pump trip alarms tell the control room. An upset is read as
it strikes, not before; from then the plan shows the runback's load.

**Episode** (rule 17) and **scenarios** (rule 20): an hour of established
operation. The unit holds a planned load, balanced at normal water level, and
the plan ramps once to another load, 5 to 15 minutes in. The suite
(`scenarios.py`), each from REALITY.md:

| Scenario | What moves | Source |
|---|---|---|
| `normal` | one ramp at 1 to 6 %/min between 40 and 100 % load | [S10], [S17], [S3] |
| `industrial_swing` | one steam demand swing at 15 to 25 %/min | [S15], [S3] |
| `low_load` | one ramp at 1 to 6 %/min between 25 and 50 % load | [S11], [S1], [S2], [S16] |
| `id_fan_runback` | from 75 to 100 % load, the fans cap firing at the runback load's and the plan runs back at 12 to 25 %/min to 50 % | [S10] Figs. 5, 6 |
| `feed_pump_trip` | from 75 to 100 % load, feed falls to half the nominal flow until the standby pump delivers 30 to 60 s later, and the unit runs back | [S10] Fig. 7, [S11], [S9] |

**Weights (charter J4).** Each scenario's real operating time, in hours a year
(`scenarios.py`): normal 5600, low load 1600, industrial swing 800, ID fan
runback 1.4, feed pump trip 2.8. The normal, low-load and swing hours are ours,
with the family ranges in `scenarios.py`. The bad days are sourced (owner,
2026-11-11): one plant's event log shows 0.5 ID fan runbacks and about 1 feed
pump trip a unit-year, each holding 2.775 h (REALITY.md [ST1]; EPRI's 1978 survey
gives 0.75 to 1.25 feed pump failures a unit-year, [ST2]). One plant is a thin
base. The bad days weigh 0.06 % of the headline together, so it
is mostly a normal day's; the per-scenario savings and the worst scenario
show the rest.

**The stream** (docs/proposals/long-strata.md, S1). As a continuing plant
(`target_gym.stream`) a stream starts on a flat plan with neither of the
window's events, and the scenario's event starts through `start_event`:

- In normal load following (a day on average, four load changes in it) and in
  the industrial swing and low-load regimes (a shift on average, one load
  change a visit), the event is a planned load change, as the window's one
  ramp: announced 300 to 900 s ahead, from where the plan stands to a load in
  the regime's `load_range`, at a rate in its `ramp_rate_range`.
- In the ID fan runback and feed pump trip regimes (an hour on average, one
  upset a visit, as the weights count them), the event is the window's upset.
  It carries its kind and its return in the state, so it runs to its return
  through a switch: the runback holds `event_hold` (1 h), then the fans are
  back and the unit ramps up to its plan at the plan's rate.
- While a runback still holds the load down the unit master takes no new
  event (ours).
- Plan times count from `plan_t0`, the step the plan was last rebased at
  (each event), so float32 keeps them exact however long the plant runs.

[S3]'s "several times during every 24 h" is the only source for any of these
numbers; four a day, the dwells and the hold are ours, set once without
tuning and open for the owner. In a window nothing changes: a scored window is
bit-identical to what it was.

**Score** (rule 15), in GBP: each step charges the steam delivered off the plan
beyond the steam meter's resolution, as generation at `steam_rate`, at the
imbalance price; in boiler-following a pressure error is generation off
schedule [S10]. A trip charges GBP 19,203. Level precision inside the trips
earns nothing. The floor of the score, in closed form: steam within the
meter's resolution of the plan costs zero, so the floor is 0; rule 9's room
needs an oracle, not run yet ("not measured").

**The reference measured per scenario** (shipped three-element expert, seeds
100 to 109, sensor noise on; `scripts` not changed): no trip in any scenario.

| Scenario | Cost, GBP per hour (10 seeds) | Level, min and max |
|---|---|---|
| `normal` | 0.0 to 0.5 on nine seeds, 7.5 on one | -76 to +95 mm |
| `industrial_swing` | 0.1 to 1.3, 13.0 on one | -80 to +156 mm |
| `low_load` | 11.8 to 14.1, 0.2 on one | -104 to +205 mm |
| `id_fan_runback` | 34 to 95 | -76 to +108 mm |
| `feed_pump_trip` | 2.3 to 4.1 | -209 to +237 mm |

The feed pump trip is where shrink and swell reach the trips: the level falls
to 63 % of the low margin and, when the standby pump delivers, rises to 93 % of
the high one (S10's unit fell to 84 % of its low margin). Rule 23: the
reference costs more than zero on 49 of 50 seeds.

**Why the room is small (charter J5).** The ceiling (data/oracle_ceilings,
seeds 100 to 109) costs zero on four of the five scenarios. The reference
costs zero on `normal` and `low_load`. So room on `J` (5.3) rests on
`id_fan_runback`, which is 0.04 % of the year. Two costs this family has and
a level and firing controller sets were looked for, and neither is priced:

- Wear per load follow. [S24] (Table 1-1, 2011 $) prices a "significant" load
  follow, a swing of more than 15 to 20 % of capacity, at a median $1.92 per
  MW of capacity for gas-fired steam units and $3.34 for small subcritical
  coal. An episode holds one ramp of the load plan, and the plan sets whether
  that ramp is significant. Every controller that follows the plan pays the
  same count, so the term would add a constant and no room. What a controller
  does change is how fast it ramps. [S24] gives that as a multiplier of 1.2
  to 6 for ramps 1.1 to 2 times faster than typical. But it marks those
  figures preliminary, and it states no typical rate to measure "faster"
  from, so the multiplier can't be priced without a number of ours.
- The drum's heating-rate limit. EN 12952-3 (formerly TRD 301) sets the
  allowed rate of change of the drum's saturation temperature. The standard's
  formula could not be fetched (it is sold, and the worked examples found
  were not reachable), so no limit is added.

Nothing unsourced is added. The boiler's room stays as measured: this family
is close to solved by a three-element controller on its normal days, and its
cost lies in rare upsets.

## 6. Behaviour tests (charter step 3, Amendment 3)

`tests/boiler_drum/test_boiler_drum_reality.py` runs the cold review's eight
tests. Under Amendment 3, tests 2, 6 and 8 are stated as family properties
with ranges from several units' sources, written in each test before its rerun
(after the earlier results were known, which the tests say). On the rebuilt
plant:

| Test | Source | Result |
|---|---|---|
| 1. Swell on a medium-load steam step | S1 Figs. 12, 16 | Pass |
| 2. Family: pressure ramps down through a dwell, 3 to 9 bar in 250 s, second half at least half the first | S1 Figs. 12, 15; S3 Figs. 3, 5 | Pass: 5.5 bar, ratio 0.69 |
| 3. Swell per kg/s larger at medium load | S1 p. 26 | Pass |
| 4. Swell for a 10 %-of-load step grows toward low load | S1 p. 20, S2, S11 | **Fails**: 20, 22, 24 mm at 50, 75, 100 % |
| 5. Shrink smaller and shorter than swell | S14 | **Fails**: shrink 27 mm, swell 24 mm |
| 6. Family: feedwater steps show the inverse response within +-0.18 m | S1 Fig. 11; S3 Fig. 5 | Pass |
| 7. Rapid swell on a fuel step | S1 Fig. 10 | Pass |
| 8. Family: a ramp moves the level at most 0.7 of a step's | S2 sec. 5.2; [S10] | Pass: 0.31 |

Tests 2 and 6 balance the plant at the middle of the figures' ranges, which
the figures' ramps both ways require; the review's challenger found both
moves sound. At the audit's 52 kg/s test 6's mean feed exceeds the steam by 8
kg/s and any drum climbs 0.5 m in 1000 s.

The two failures are known deviations (rule 13), on S1's own model:

- Test 4 is a family property from three sources: per kg/s the swell is larger
  at low load (test 3), but a step that scales with the load swells less at
  low load. The review measured that sliding pressure [S2] would pass it only
  narrowly (25.0, 24.6, 23.6 mm); it is not added. Effect on the score:
  `low_load` ramps are milder than S11 reports.
- Test 5 rests on one source [S14]: shrink is about 10 % larger than swell.
  Effect: a load decrease or a feed pump trip brings the level nearer the low
  trip than practice reports.

## 7. Deviations still in the model

- A trip's lump still prices a whole 2.5 h hot start [S17], while the restart
  now plays out the reload from minimum fire to the plan. If [S17]'s hot start
  runs to full load, the lump pays twice for part of that reload (long-strata
  R8); the dead time to light-off alone is not found yet. The reload's rate is
  the plan's (ours).
- The feedwater flow loop is perfect: the action is flow, as a cascade's
  setpoint, with no pump or valve characteristic (cold review 7).
- The level reading is white noise with no delay; S3 fits coloured noise from
  surface boiling and carries a delay whose units it does not state (cold
  review 6).
- Runbacks and feed pump trips are scenarios, one per window; their real
  frequency is not found, and the headline weighs each scenario equally
  (rule 21).
- Pressure is held at one setpoint; sliding pressure [S2] is not modelled.
- A restart is instant at a new load; a real one runs through single-element
  start-up with the level lowered for swell [S5, S19].

## 8. What the reference ladder's row needs

- Inputs: the load plan (channels 7 to 9) for the steady-state feedforward
  rung: steam feedforward to feedwater, `q_s (h_s - h_f)` to firing; the caps
  (channels 10, 11), which an override rung reads to stop winding up against a
  capped fan or pump.
- Bump tests at loads across 25 to 100 %, since the gains change with load and
  `low_load` costs the shipped expert most on a normal day.
- No zone to back off from (section 4): the level setpoint is normal water
  level.
- Cost in GBP; a rung that trips any nominal seed in any scenario is dropped.
  Seeds carry their own plan and upset.

## 9. What the oracle needs

- `oracle.py` is ported to v5 (plan, upset caps, GBP cost with the meter's dead
  band smoothed, a resting target for feedwater, G12) and builds; it has not
  run. It knows the upset ahead, as a ceiling may.
- Its horizon must cover a runback and the standby pump's delay (about 3 to 5
  min, 90 to 150 steps), and the trip's 10 s delay is not planned (the limit is
  held on every step, stricter than the plant).
- `scripts/measure_oracle_widths.py`'s boiler row is redone: no smoothed
  sites and no back-offs, since the plant draws nothing after the reset. Both
  the receding slot and the ceiling hold the steam under the water surface at
  or above 0.
