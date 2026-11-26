# Building heating with a TABS slab: model, sources, limits and deviations

Version 6: the plant rebuilt as version 5 on 2026-11-08 under charter
Amendments 1 to 3, with the stream of 2026-11-11 (section 5), and the heat
source's size and the reset that meets the band of 2026-11-23. The plant
record is `REALITY.md`, and its cold review `COLD_REVIEW.md`. Source numbers
[1] to [21] are REALITY.md's. Under Amendment 3 the task represents the
family of slab-heated offices, not one building: numbers outside the record
are labelled "ours" with the family's range. Nothing here is compared with an earlier version
of the task (rule 11).

## 1. The job

REALITY.md sec. 6: choose the water temperature about a day ahead from the
weather, so the room stays inside its comfort band through the coming hours,
because the slab cannot correct within the day [9, p. 3], [7, p. 7].

Named property (rule 18): **slow concrete-slab heating**. Heat reaches the
room only through a concrete core with hours of time constant, and the slab
self-regulates: its output falls as the room warms [7, p. 7], [9, p. 7].

### Task: `hvac`

Heat an office through a concrete slab by choosing the supply water
temperature, so the room stays in its 21 to 26 C comfort band during occupied
hours at the least electricity cost. The slab takes hours to respond, so the
policy has to act ahead: it reads the room, slab and water temperatures, the
outdoor temperature with its last-24 h mean and next-24 h forecast, the clock,
and the tariff with the hours to its next change. Each step is charged, in
euros, the heat pump's electricity at the tariff, each occupied
kelvin-hour of room outside the band or of floor above its limit. The
evaluation also charges, once, the heat the building's store lost across the
scored days, at the cheapest price heat can be stored at. Nothing trips.

| obs | meaning |
|---|---|
| 0 | room air temperature (C) |
| 1 | outdoor air temperature (C) |
| 2 | mean outdoor temperature over the last 24 h (C) |
| 3 | slab core temperature (C) |
| 4 | return water temperature (C) |
| 5 | supply water temperature (C) |
| 6 | sine of the hour of day |
| 7 | cosine of the hour of day |
| 8 | centre of the comfort band (C) |
| 9 | forecast mean outdoor temperature over the next 24 h (C) |
| 10 | electricity price now (EUR/kWh) |
| 11 | hours until the tariff next changes |

## 2. Model

The zone is the ISO 13790 5R1C network (air, surface star node and mass node,
one capacitance), with a TABS slab added as [9]'s first-order model (Fig. 3):
supply water, the fictitious resistance `R_t`, the concrete core's capacity,
then the room.

```
dT_core/dt = (Q_slab + H_slab (T_air - T_core)) / C_slab
Q_slab     = min(H_water max(T_supply - T_core, 0), Q_source)
             H_water = covered area / R_t, Q_source = source_capacity A_floor
H_slab     = A_floor / R_up + A_floor / R_down
R_up       = d / (2 lambda) + R_flooring + 1 / h_floor   (floor face, carpet)
R_down     = d / (2 lambda) + 1 / h_ceiling              (ceiling face)
T_return   = T_made - Q_slab / (m_dot c_w)
T_made     = min(T_supply, T_core + Q_source / H_water)
T_floor    = T_air + (T_core - T_air) (1 / h_floor) / R_up
```

The core couples to the air node, and the air and surface nodes are solved in
closed form each step. Both faces of the slab heat the zone, as in [9]'s
simulated room (on a top storey, with its roof, an open question sent to the
owner on 2026-11-26). The plant heats
only: water below the core gives no heat, since it has no chiller. The slab
takes no more than the heat source gives (section 4): at that size the water
the plant makes, `T_made`, is cooler than the valve asks, and it is what the
supply sensor reads and what the heat pump's COP is taken at.

Measured on the plant: the slab's heat into the room after a 1 K supply step
reaches 63.2 % of its change in 150 min (test 1); at design (-10 C, 21 C) the
supply is 27.7 C and the floor surface 22.2 C (test 8).

## 3. Numbers

| Symbol | Value | Unit | Source |
|---|---|---|---|
| `slab_thickness` | 0.25 | m | [9] Table 2 |
| `tabs_covering` | 0.80 | - | [9] Table 2 |
| `R_t` | 0.08 | m^2 K/W | [9] Table 2, per covered m^2 |
| `water_flow` | 15 | kg/(h m^2) | [9] Table 2 |
| `R_flooring` | 0.125 | m^2 K/W | [9] Table 2 (carpet) |
| `h_floor` | 10.8 | W/(m^2 K) | [5] p. 25 design table: 100 W/m^2 at 29 C and 50 W/m^2 at 24.8 C over a 20 C room, 11.1 and 10.4 |
| `h_ceiling` | 6.5 | W/(m^2 K) | Ours: EN 1264-5 Table A.1 as commonly quoted, not read |
| `concrete_density`, `concrete_heat_capacity`, `concrete_conductivity` | 2400, 1000, 2.0 | kg/m^3, J/(kg K), W/(m K) | Ours: ISO 10456 values as commonly quoted, not read; published ranges are 0.8 to 1.2 kJ/(kg K) and 1.5 to 3.5 W/(m K) |
| `c_water` | 4186 | J/(kg K) | Water |
| `T_supply_max` | 49 | C | The TABS family's documented maximum, Madrid's [14, Table 1]; Muenster runs 16 to 30 C [13]; screed floors are cut at 55 C [17] |
| `T_supply_min` | 15 | C | Ours: the valve's lowest setting, which is off |
| `source_capacity` | 38 | W/m^2 | [13]: Muenster's reversible heat pumps give 530 kW of heat for 14 000 m^2, 37.9 W/m^2. Here 5.7 kW, 1.39 times the design load of 4.1 kW (132.4 W/K over 31 K) |
| `cop_carnot_fraction` | 0.52 | - | Ours, family range 0.50 to 0.57: brine heat pump datasheets give COP 4.4 to 5.0 at B0/W35 and 2.7 to 3.2 at B0/W55 [18]; the model gives 4.58 and 3.10 |
| `T_brine` | 0 | C | Ours, family range -5 to 5 C: the EN 14511 rating point B0 [18] |
| `price_valle`, `price_llano`, `price_punta` | 0.139, 0.184, 0.263 | EUR/kWh | [19]: one Spanish 2.0TD offer, valle 00 to 08 h, punta 10 to 14 and 18 to 22 h; Madrid charges its slabs on the night tariff [14] |
| `productivity_per_K` | 0.02 | 1/K | [20]: 2 % of work per K above 25 C; ours below 21 C, where [20] shows a loss but no per-K figure was read |
| `occupants_per_m2` | 0.10 | 1/m^2 | Ours, family range 0.07 to 0.12 (8 to 15 m^2 a person) |
| `labour_cost` | 31.8 | EUR/h | [21]: EU hourly labour cost, 2023 |
| `occupied_start_h`, `occupied_end_h` | 8, 18 | h | [6, p. 3]: an office's day |
| `frost_on`, `frost_off` | 4.4, 7.2 | C | [1]: below 40 F the zones go to setback until above 45 F |
| `band_low`, `band_high` | 21, 26 | C | [9] p. 5: room setpoints 21 C heating, 26 C cooling, all year |
| `floor_surface_max` | 29 | C | [5] p. 23 (EN 1264), [4] |
| `comfort_tolerance` | 2.5 | K | Half the band, read only by the time-in-band KPI tables |
| `precision_floor` | 0.1 | K | The BMS step, ours (Readings) |
| `max_steps_in_episode` | 1152 | steps | Twelve days: three burned in, seven scored of the two weeks a curve change is judged over [11, p. 1], two of the ceiling's tail (section 5) |
| `comfort_price` | 9.54 | EUR/(K h) | Derived: `productivity_per_K` x `occupants_per_m2` x `A_floor` x `labour_cost`, 0.02 x 15 occupants x 31.8: the work an occupied kelvin-hour outside the band loses |
| `T_out_design` | -10 | C | The curve's design point [12, p. 7], where the stored heat's price is read |
| `restart_steps`, `failure_cost` | 0, 0.0 | - | No limit trips this plant |
| `delta_t` | 900 | s | Ours: the standard building-simulation step |

The building around the slab keeps its ISO 13790 envelope: 150 m^2 floor,
`lambda_at` 4.5, "light" class mass 110 kJ/(K m^2), `h_is` 3.45, `h_ms` 9.1,
wall, roof and window U-values 0.28, 0.18 and 1.30 W/(m^2 K) on 120, 150 and
25 m^2, 0.5 air changes per hour; H = 159.4 W/K. The zone is a top storey,
with its roof, and the slab is a node of its own, so the 5R1C mass beside it
is the "light" class, whose figure counts no massive floor (owner, 2026-11-25
and 2026-11-26, the review's finding 5: a middle storey with no roof failed
the slab time-constant and setback reality tests at the light mass, and the
roof restores both; before, the "medium" class). These, the weather (5 C mean, 5 K daily half swing, 3 K OU deviation
with a 12 h correlation), the solar peak (175 W/m^2, a PVGIS heating-season
mean over four facades) and the internal gains (8 and 2 W/m^2, over the
office day of [6]) are **ours**: REALITY.md has no weather statistics.

Readings: room, outdoor, slab core and return water, each 0.023 K (SHT45
class) rounded to the BMS's 0.1 K step (ours). The 24 h mean is computed from
the true outdoor history, so it carries no noise worth modelling (0.023 K over
96 readings). The supply is the valve's setting, exact.

## 4. Limits (rule 14)

| Limit | Kind | Model |
|---|---|---|
| Supply water 49 C [14] | Protected | The valve's range stops at the TABS family's maximum |
| Heat source 38 W/m^2 [13] | Protected | The slab takes no more than the source gives, 5.7 kW; the water is then cooler than the valve asks, and the electricity is charged at that water's COP |
| Frost, room below 4.4 C [1] | Protected | The plant asks for 49 C, which runs the source flat out, until the room exceeds 7.2 C; the electricity is charged |
| Comfort band 21 to 26 C [9] | Zone | Each occupied kelvin-hour outside costs `comfort_price`, the work it loses [20, 21] |
| Floor surface 29 C [5], [4] | Zone | Each occupied kelvin-hour above costs `comfort_price` |
| ASHRAE 55 drift [4] | Not priced | No source prices it; test 11 checks the plant stays inside it |

Nothing trips. The old 5 C and 35 C trips had no source and are gone.

**Zones and cheap side (rule 22).** The comfort band, 21 to 26 C, is the
zone; its cheap side is the low edge, since heat costs and the band's top is
reached only by the sun. Leaving the band in office hours is charged as lost
work, so each rung of the reference's ladder holds the lowest candidate of the
zone's grid (21 C and 0.25 K steps to the centre, 23.5 C) at which it leaves
the band on none of the ladder's seeds in any scenario, and the ladder keeps
the cheapest rung (charter J1, owner, 2026-11-23). The floor limit never binds
in the heating season (22.2 C at design).

## 5. Observation, episode and score

**Observation** (rule 16): room air [6, 14], outdoor air and its mean over
the last 24 h [9, 13], slab core [7, p. 8], [12, p. 12], return water [12, 14],
the supply, the clock, the band's centre, the next 24 h's forecast mean
outdoor temperature, and the tariff now with the hours to its next change.
The forecast is what the job sentence plans on, "a day ahead from the
weather": the daily cycle plus today's deviation decaying at the weather's
rate, so its error is the weather's unforecast part. The tariff is a schedule
known ahead (rule 16). The building's other mass is hidden.

**Episode** (rule 17) and **scenarios** (rule 20): twelve days from established
operation, scored on days 4 to 10 (`scenarios.hvac_evaluation`). The reset settles the building, slab included, for the last day's
mean weather and gains, with the supply that holds it there, as a heating
curve runs it [7, p. 7], at the level that meets the band: held on that
supply through a night with no unforecast weather, the room is at the 21 C
edge when the office opens (21.01 C in every scenario; the room at the
midnight reset is 21.4 to 21.6 C). The level is found in closed form: the
balances are linear, so two settled buildings give the one whose room opens
at the edge, each taken through the night on its mean weather and gains by
the exact solution of the two slow nodes (`env.drift_over`). No night setback: slab buildings hold a
near-constant slab setpoint [7, p. 7], and [15] advises against setback for
radiant floors. The suite (`scenarios.py`):

| Scenario | Overrides | Source |
|---|---|---|
| `normal` | none, 5 C mean | [7], [9] |
| `cold_snap` | 5 C mean -> -5 C | [15], [11] |
| `shoulder_season` | 12 C mean, around the heating limit | [11], [9] |

**Weights (charter J4).** Each scenario's share of a 13-fortnight heating
season (`scenarios.py`): normal 6.5, shoulder season 5.5, cold snap 1. All are
ours, with family ranges in `scenarios.py`; REALITY.md has no weather statistics.

**The evaluation cut** (owner, 2026-11-24). A scenario, a seed or a simulated
step stays only if removing it would hide a named behaviour or change which
controller looks better. Measured on the 50 seeds of the review (feedforward
reference against the ceiling, 14-day episodes):

- Three behaviours, one scenario each: the night's charge of the slab
  (normal), the heat source at its size (the cold snap, which alone places the
  reference's set point), and the work near the heating limit (the shoulder
  season). A sunny spell (twice the sun, 1.5 fortnights a season) repeated the
  shoulder's limit and low gains (0.5) normal's night charge, so their weights
  fold into those two. Their overrides stay plant parameters.
- The start: day 1 held 46 % of the fortnight's gap between the reference and
  the ceiling, the building warming from its reset, and the gap was gone after
  day 3. Days 1 to 3 burn in, charged in training and read by no evaluation.
- The end: day 14 held 11 % of the gap, the ceiling spending the slab's charge
  on a last night. Its plan charges two days past the scored ones (days 11 and
  12), which no line reads.
- Seeds: every line runs on seeds 100 to 104, and 105 to 109 too in a scenario
  whose paired saving of the reference over the ceiling has a 95 % Student t
  interval that touches 0 (`target_gym.evaluation.line_seeds`, written before any
  result). The placement of the set point keeps the ladder's ten seeds.

**The stream** (docs/proposals/long-strata.md, S1). As a continuing plant
(`target_gym.stream`) a cold snap lasts 5.5 d on average, as western European
cold spells do (`REALITY.md` [ST1]), and the other regimes a fortnight
(`dwell_h` 336), inside the week to a month weather regimes hold ([ST2]). Each
brings no event of its own (`events_per_h` 0): the building's scenarios are
weather and gains, not upsets. A sunny spell's fortnight is ours: blocking highs
last 5.7 to 10 d ([ST3], [ST4]), and no source gives how long a clear spell or an
office's low-gain spell lasts (schools close 1 to 2 weeks, [ST5]). The day
(weather cycle, sun, office hours, tariff) reads a step count wrapped on
`DAY_STEPS`, exact however long the plant runs; the weather's deviation is
already mean-reverting from its long-run law. At a switch the outdoor mean does
not jump: it moves to the new regime's as a first-order lag of `T_out_mean_tau`,
6 h, as a front takes hours to pass (ours, owner 2026-11-11; no source gives it
as a time constant, `REALITY.md` [ST6]). A linear lag keeps the long-run mean and the regimes'
shares. The 24 h forecast includes where the mean is heading, as a forecast sees
a front coming. In a window nothing changes: a scored window is bit-identical to
what it was.

**Score** (rule 15, Amendment 3), in euros: the heat pump's electricity, the
slab's heat over its COP at the supply temperature, at the tariff; and each
occupied kelvin-hour of room outside the band, or of floor above 29 C, at the
work it loses. Every term is a cost, as the shared reward contract asks. The
evaluation adds one charge at the scored window's edges: the store, `C_m T_mass
+ C_slab T_core`, at the valle tariff over the COP of the design supply
(`env.stored_value`, recorded on every run), as it was before the first scored
step less as it is after the last, spread over the scored steps
(`evaluation.Evaluation.edge_charge`), so heat bought
in the unscored burn-in, or run down on the last scored evening, is paid for
at what storing it costs (owner, 2026-11-25, the review's finding 1: a policy
at full supply on the last unscored evening and off on the last scored one had
cut the reference's scored cost 7 to 53 % at no loss of comfort). Night
charging pays twice: the night tariff is 0.53 of the peak's, and the slab
carries the heat into the day. Inside the band being
nearer its middle earns nothing (rule 15; [20]'s loss inside 21 to 26 C is
small and is not counted). The floor of the score, in closed form: a room held
in the band costs only its electricity.

**The published heating curve measured per scenario** (`experts.py`, seeds 100
to 109, sensor noise on; measured on 2026-11-08, before the source had its
size and the reset met the band, and due again with the records), euros per
fortnight. It is the ladder's `published`
rung; the ladder keeps its cascade rung as the reference (charter J1), which
costs less on every scenario and pays no lost work:

| Scenario | Electricity | Lost work | Occupied mean room |
|---|---|---|---|
| `normal` | 13.9 to 18.5 | 0.2 to 9.9 | 22.7 C |
| `cold_snap` | 32.1 to 37.4 | 0.3 to 11.0 | 22.6 C |
| `shoulder_season` | 3.5 to 6.8 | 0.1 to 9.2 | 22.9 C |
| `sunny_spell` | 8.7 to 13.1 | 0.5 to 8.3 | 23.7 C |
| `low_gains` | 16.6 to 21.3 | 1.1 to 17.0 | 22.1 C |

The curve undershoots 21 C on cold mornings (minimum 20.1 C) and never reaches
the band's top. Rule 23: every seed costs more than zero.

## 6. Behaviour tests (charter step 3, Amendment 3)

`tests/hvac/test_hvac_reality.py` runs the cold review's eleven tests. Tests 2
and 6 are restated as family properties, written in the tests before the
rerun:

| Test | Source | Result |
|---|---|---|
| 1. Slab time constant 145 to 462 min, on a step the source can follow (+1 K) | [10], [6] | **Fails, 135 min** since the 2026-11-25 storey and light mass (open finding, sent to the owner) |
| 2. Family: an ordinary change (+5 K supply, the size of a curve shift [11]) felt after 60 min or more | [8], [11] | Pass, 390 min: the source's size cuts the step |
| 3. Curve shift moves the room one for one | [11] | Pass, ratio 0.79 |
| 4. Slab output falls when gains rise | [7], [9] | Pass |
| 5. Setback recovery 1 h or more | [15] | Pass, 7.75 h: 8 h with the water off cools the room, and the source at 1.39 times the design load brings it back slowly |
| 6. Family: the steady supply at -8 C lies in the TABS range, above 20 C and at most 30 C | [13] | Pass, 25.0 C |
| 7. Heating limit 8 to 15 C | [11], [9] | Pass |
| 8. Floor at most 29 C, supply at most 49 C at design | [5], [14] | Pass: 22.0 C and 26.6 C |
| 9. Morning overshoot after reduced mode | [12] | Pass |
| 10. Design supply never recovers a setback | [15] | **Fails, 20.55 C after 15 h** (limit 20.5) since the 2026-11-25 storey and light mass (open finding, sent to the owner) |
| 11. ASHRAE 55 drift under the curve | [4] | Pass: at most 0.24 K in 15 min and 0.43 K in 1 h |

Test 5 failed (45 min) while the source had no size: full heat was 39 kW into
a zone whose design load is 4.9 kW. With Muenster's 38 W/m^2 [13] the plant
recovered in 16.5 h (7.75 h on the 2026-11-25 middle storey), beyond [15]'s 1.0 to 6.9 h, which are for baseboard homes
with boilers at twice the design load; [15] says of a source sized to the load
that it "may never recover". The fix was a missing piece of the plant (charter
B6), not a number moved until the test passed.

## 7. Deviations still in the model

- Heating only; TABS plants are reversible [13, 14]. Overheating above 26 C is
  charged to an actuator that can only stop heating.
- Weather, sun and gains are ours (section 3). Cold snaps are a scenario, not
  events inside one; sunny spells and low-gain days are folded into the normal
  and shoulder fortnights' weights (section 5).
- A free continuous supply every 15 min; practice changes the curve once a day
  and spreads it over 12 h [6].
- One retailer's tariff [19] stands for the family's day-night prices.
- The forecast is the weather model's own conditional mean; real day-ahead
  forecasts carry their own error, not modelled.
- The slab gives all its heat, the radiant share included, to the air node;
  in a room about half of a floor's or a ceiling's exchange is radiant, to the
  surfaces (the owner kept this, 2026-11-25).
- One size for the heat source, whatever the water it makes; a heat pump's
  capacity moves with its supply and brine temperatures, not modelled.
- The COP below W35. The ceiling's plan heats with supply water at 24 to 31 C
  and the cascade mostly below 35 C, under the lowest datasheet point read
  (W35, [18]). Their COPs there, 5.3 to 6.5 over a fortnight, are the fixed
  Carnot share carried down, ours.

## 8. What the reference ladder's row needs

- The move is the supply water temperature (C). A bump test is a supply step;
  the slab's time constant is hours (test 1), so bumps need about a day.
- The zone's set point (section 4): each rung's own lowest candidate of the
  band's grid with no exit, by `ladder.zone_grid` and
  `ladder.first_without_exit`; the row names the score's `comfort` term as the
  exit's charge, and the cheap side "low". Measured on 2026-11-24, of 50
  episodes: the pi rung leaves on 48, 40, 16, 5 and 0 at 21.00 to 22.00 C, the
  feedforward on 48, 36, 16, 5 and 0, the cascade on 49, 49, 46, 23, 5 and 0
  at 21.00 to 22.25 C. So the pi and the feedforward hold 22.00 C and the
  cascade 22.25 C; the feedforward is the cheapest and is kept. Every seed is
  held by some candidate, so none is left out of a placement. Those numbers are
  of 14-day episodes read whole; since the evaluation cut an exit counts only on
  the scored days 4 to 10, so the set points and the kept rung are measured
  again with the records (on the review's runs, pi is the cheaper rung on
  established days).
- The steady-state feedforward rung is the heating curve from the plant's
  balance (hvac/experts.py), read on the 24 h mean or the forecast (channel 9).
- The cascade rung has measured inner variables: the slab core (channel 3) or
  the return water (channel 4) [7, 12].
- The night-charging rung (owner, 2026-11-25, the review's P2): Madrid charges
  its floors at night on the cheap tariff and lets them release heat by day
  [14, p. 14]. The water runs only from 00:00 to 07:00, at a lift above the
  core that puts the day's balance heat in those 7 hours, and the lift is
  trimmed each day on the last day's occupied room mean, 1 K per K [11, p. 2],
  held to 5 K [9, part 2]. It is placed by the exit rule like the PI rungs.
- Cost in euros of electricity at the tariff and occupied lost work, read on
  the scored days 4 to 10 alone, so the reset's warm-up does not enter it.

## 9. What the oracle needs

- `oracle.py` plans the v5 score (COP, tariff, occupied hinges at the
  work-loss price, the stored heat). Its ceiling was first recorded on
  2026-11-08 (`data/oracle_ceilings/hvac.json`): one plan over the episode,
  with the episode's own weather. Since 2026-11-25 the plan, and the receding
  oracle's, charges the scored days 4 to 10 alone, as the lines are read
  (`oracle.scored_window`): the stored heat prices the store at both edges, so
  the ceiling bounds what any line can score there.
- Its horizon must reach across a night to the next occupied day, at least 24
  h (96 steps), for night charging to show; the slab's time constant (150 min
  for its flux) sets how far ahead a supply move pays.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `hvac` | 1152 | 900 | 288.0 h | 1 in [-1, 1] | 12 | 110 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->
