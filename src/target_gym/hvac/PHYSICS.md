# Building heating with a TABS slab: model, sources, limits and deviations

Version 5, rebuilt on 2026-11-08 under charter Amendments 1 to 3. The plant
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

## 2. Model

The zone is the ISO 13790 5R1C network (air, surface star node and mass node,
one capacitance), with a TABS slab added as [9]'s first-order model (Fig. 3):
supply water, the fictitious resistance `R_t`, the concrete core's capacity,
then the room.

```
dT_core/dt = (Q_slab + H_slab (T_air - T_core)) / C_slab
Q_slab     = H_water max(T_supply - T_core, 0)        H_water = covered area / R_t
H_slab     = A_floor / R_up + A_floor / R_down
R_up       = d / (2 lambda) + R_flooring + 1 / h_floor   (floor face, carpet)
R_down     = d / (2 lambda) + 1 / h_ceiling              (ceiling face)
T_return   = T_supply - Q_slab / (m_dot c_w)
T_floor    = T_air + (T_core - T_air) (1 / h_floor) / R_up
```

The core couples to the air node, and the air and surface nodes are solved in
closed form each step. The zone is one storey of a stack of like storeys, so
both faces of the slab heat it, as in [9]'s simulated room. The plant heats
only: water below the core gives no heat, since it has no chiller.

Measured on the plant: the slab's heat into the room after a 5 K supply step
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
| `max_steps_in_episode` | 1344 | steps | Two weeks: a curve change is judged over two weeks [11, p. 1] |
| `comfort_price` | 9.54 | EUR/(K h) | `productivity_per_K` x 15 occupants x `labour_cost`: the work an occupied kelvin-hour outside the band loses |
| `restart_steps`, `failure_cost` | 0, 0.0 | - | No limit trips this plant |
| `delta_t` | 900 | s | Ours: the standard building-simulation step |

The building around the slab keeps its ISO 13790 envelope: 150 m^2 floor,
`lambda_at` 4.5, "medium" class mass 165 kJ/(K m^2), `h_is` 3.45, `h_ms` 9.1,
U-values 0.28 / 0.18 / 1.30 W/(m^2 K), 0.5 air changes per hour; H = 159.4
W/K. These, the weather (5 C mean, 5 K daily half swing, 3 K OU deviation
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
| Frost, room below 4.4 C [1] | Protected | The plant takes the supply to 49 C until the room exceeds 7.2 C; the electricity is charged |
| Comfort band 21 to 26 C [9] | Zone | Each occupied kelvin-hour outside costs `comfort_price`, the work it loses [20, 21] |
| Floor surface 29 C [5], [4] | Zone | Each occupied kelvin-hour above costs `comfort_price` |
| ASHRAE 55 drift [4] | Not priced | No source prices it; test 11 checks the plant stays inside it |

Nothing trips. The old 5 C and 35 C trips had no source and are gone.

**Zones and cheap side (rule 22).** The comfort band, 21 to 26 C, is the
zone; its cheap side is the low edge, since heat costs and the band's top is
reached only by the sun. The reference's room setpoint is 21 C plus two
standard deviations of the occupied room temperature, measured on its bump
seeds. The floor limit never binds in the heating season (22.2 C at design).

## 5. Observation, episode and score

**Observation** (rule 16): room air [6, 14], outdoor air and its mean over
the last 24 h [9, 13], slab core [7, p. 8], [12, p. 12], return water [12, 14],
the supply, the clock, the band's centre, the next 24 h's forecast mean
outdoor temperature, and the tariff now with the hours to its next change.
The forecast is what the job sentence plans on, "a day ahead from the
weather": the daily cycle plus today's deviation decaying at the weather's
rate, so its error is the weather's unforecast part. The tariff is a schedule
known ahead (rule 16). The building's other mass is hidden.

**Episode** (rule 17) and **scenarios** (rule 20): two weeks from established
operation. The reset settles the building, slab included, at 21 C for the last
day's mean weather and gains, with the supply that holds it there, as a
heating curve runs it [7, p. 7]. No night setback: slab buildings hold a
near-constant slab setpoint [7, p. 7], and [15] advises against setback for
radiant floors. The suite (`scenarios.py`):

| Scenario | Overrides | Source |
|---|---|---|
| `normal` | none, 5 C mean | [7], [9] |
| `cold_snap` | 5 C mean -> -5 C | [15], [11] |
| `shoulder_season` | 12 C mean, around the heating limit | [11], [9] |
| `sunny_spell` | twice the season's mean sun | [6], [8] |
| `low_gains` | occupied gains at the unoccupied 2 W/m^2 | [9] part 2 |

**Weights (charter J4).** Each scenario's share of a 13-fortnight heating
season (`scenarios.py`): normal 6, shoulder season 4, sunny spell 1.5, cold
snap 1, low gains 0.5. All are ours, with family ranges in `scenarios.py`;
REALITY.md has no weather statistics.

**The stream** (docs/proposals/long-strata.md, S1). As a continuing plant
(`target_gym.stream`) each regime lasts a fortnight on average (`dwell_h` 336),
the fortnight the suite defines it as, and brings no event of its own
(`events_per_h` 0): the building's scenarios are weather and gains, not upsets.
No source gives how long a cold snap or a sunny spell lasts (REALITY.md), so
the dwell is ours, set once without tuning and open for the owner. The day
(weather cycle, sun, office hours, tariff) reads a step count wrapped on
`DAY_STEPS`, exact however long the plant runs; the weather's deviation is
already mean-reverting from its long-run law. In a window nothing changes: a
scored window is bit-identical to what it was.

**Score** (rule 15, Amendment 3), in euros: the heat pump's electricity, the
slab's heat over its COP at the supply temperature, at the tariff; and each
occupied kelvin-hour of room outside the band, or of floor above 29 C, at the
work it loses. Night charging pays twice: the night tariff is 0.53 of the
peak's, and the slab carries the heat into the day. Inside the band being
nearer its middle earns nothing (rule 15; [20]'s loss inside 21 to 26 C is
small and is not counted). The floor of the score, in closed form: a room held
in the band costs only its electricity.

**The reference measured per scenario** (the heating curve, seeds 100 to 109,
sensor noise on), euros per fortnight:

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
| 1. Slab time constant 145 to 462 min | [10], [6] | Pass, 150 min |
| 2. Family: an ordinary change (+5 K supply, the size of a curve shift [11]) felt after 60 min or more | [8], [11] | Pass, 225 min |
| 3. Curve shift moves the room one for one | [11] | Pass, ratio 0.76 |
| 4. Slab output falls when gains rise | [7], [9] | Pass |
| 5. Setback recovery 1 h or more | [15] | **Fails**: 45 min; 8 h with the water off cools the room only to 20.2 C |
| 6. Family: the steady supply at -8 C lies in the TABS range, above 20 C and at most 30 C | [13] | Pass, 26.0 C |
| 7. Heating limit 8 to 15 C | [11], [9] | Pass |
| 8. Floor at most 29 C, supply at most 49 C at design | [5], [14] | Pass: 22.2 C and 27.7 C |
| 9. Morning overshoot after reduced mode | [12] | Pass |
| 10. Design supply never recovers a setback | [15] | Pass, 20.49 C after 15 h (limit 20.5) |
| 11. ASHRAE 55 drift under the curve | [4] | Pass: at most 0.25 K in 15 min and 0.41 K in 1 h |

Test 5 stays as a finding. [15]'s recoveries are for baseboard homes; a slab
building is not set back (REALITY.md sec. 2), and 8 h off cools this one only
0.8 K, so the full-heat recovery is short. Effect on the score: none, since no
scenario sets the building back. Test 10 passes by 0.01 K for the same reason.

## 7. Deviations still in the model

- Heating only; TABS plants are reversible [13, 14]. Overheating above 26 C is
  charged to an actuator that can only stop heating.
- Weather, sun and gains are ours (section 3). Cold snaps, sunny spells and
  low-gain days are scenarios, a fortnight each, not events inside one.
- A free continuous supply every 15 min; practice changes the curve once a day
  and spreads it over 12 h [6].
- One retailer's tariff [19] stands for the family's day-night prices.
- The forecast is the weather model's own conditional mean; real day-ahead
  forecasts carry their own error, not modelled.
- The 5R1C "medium" mass is kept beside the slab, which a TABS building would
  count in its class.

## 8. What the reference ladder's row needs

- The move is the supply water temperature (C). A bump test is a supply step;
  the slab's time constant is hours (test 1), so bumps need about a day.
- The zone back-off (section 4): room setpoint 21 C plus 2 sigma of the
  occupied room temperature on the bump seeds, with `ladder.zone_setpoint`
  and the cheap side "low".
- The steady-state feedforward rung is the heating curve from the plant's
  balance (hvac/experts.py), read on the 24 h mean or the forecast (channel 9).
- The cascade rung has measured inner variables: the slab core (channel 3) or
  the return water (channel 4) [7, 12].
- Cost in euros of electricity at the tariff and occupied lost work; seeds
  start established, so no warm-up transient enters it.

## 9. What the oracle needs

- `oracle.py` is ported to v5 (COP, tariff, occupied hinges at the work-loss
  price) and builds; it has not run.
- Its horizon must reach across a night to the next occupied day, at least 24
  h (96 steps), for night charging to show; the slab's time constant (150 min
  for its flux) sets how far ahead a supply move pays.
