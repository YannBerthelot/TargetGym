# Building heating with a TABS slab: model, sources, limits and deviations

Version 4, rebuilt on 2026-11-08 under charter Amendments 1 and 2. The plant
record is `REALITY.md`, and its cold review `COLD_REVIEW.md`. Source numbers
[1] to [17] are REALITY.md's. Nothing here is compared with an earlier version
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
| `T_supply_max` | 55 | C | [17]: hard limit at the heating elements for cement screed |
| `T_supply_min` | 15 | C | Ours: the valve's lowest setting, below any core here |
| `frost_on`, `frost_off` | 4.4, 7.2 | C | [1]: below 40 F the zones go to setback until above 45 F |
| `band_low`, `band_high` | 21, 26 | C | [9] p. 5: room setpoints 21 C heating, 26 C cooling, all year |
| `floor_surface_max` | 29 | C | [5] p. 23 (EN 1264), [4] |
| `comfort_tolerance` | 2.5 | K | Half the band, read only by the time-in-band KPI tables |
| `precision_floor` | 0.1 | K | The BMS step, ours (Readings) |
| `max_steps_in_episode` | 1344 | steps | Two weeks: a curve change is judged over two weeks [11, p. 1] |
| `heat_price` | 0.10 | EUR/kWh | Ours: REALITY.md names energy as the cost but no tariff |
| `comfort_price` | 0.03 | EUR/(K h) | Ours: twice the heat that holds this building 1 K warmer for 1 h (159.4 W/K at EUR 0.10/kWh is EUR 0.016) |
| `restart_steps`, `failure_cost` | 0, 0.0 | - | No limit trips this plant |
| `delta_t` | 900 | s | Ours: the standard building-simulation step |

The building around the slab keeps its ISO 13790 envelope: 150 m^2 floor,
`lambda_at` 4.5, "medium" class mass 165 kJ/(K m^2), `h_is` 3.45, `h_ms` 9.1,
U-values 0.28 / 0.18 / 1.30 W/(m^2 K), 0.5 air changes per hour; H = 159.4
W/K. These, the weather (5 C mean, 5 K daily half swing, 3 K OU deviation
with a 12 h correlation), the solar peak (175 W/m^2, a PVGIS heating-season
mean over four facades) and the internal gains (8 and 2 W/m^2, occupied 07:00
to 22:00) are **ours**: REALITY.md has no weather statistics and no dwelling
schedule.

Readings: room, outdoor, slab core and return water, each 0.023 K (SHT45
class) rounded to the BMS's 0.1 K step (ours). The 24 h mean is computed from
the true outdoor history, so it carries no noise worth modelling (0.023 K over
96 readings). The supply is the valve's setting, exact.

## 4. Limits (rule 14)

| Limit | Kind | Model |
|---|---|---|
| Supply water 55 C [17] | Protected | The valve's range stops at 55 C, as the safety device cuts it |
| Frost, room below 4.4 C [1] | Protected | The plant takes the supply to 55 C until the room exceeds 7.2 C; the heat is charged |
| Comfort band 21 to 26 C [9] | Zone | Each kelvin-hour outside costs `comfort_price` |
| Floor surface 29 C [5], [4] | Zone | Each kelvin-hour above costs `comfort_price` |
| ASHRAE 55 drift [4] | Not priced | No source prices it; test 11 checks the plant stays inside it |

Nothing trips. The old 5 C and 35 C trips had no source and are gone.

## 5. Observation, episode and score

**Observation** (rule 16): room air [6, 14], outdoor air and its mean over
the last 24 h [9, 13], slab core [7, p. 8], [12, p. 12], return water [12, 14],
the supply, the clock, and the band's centre. The building's other mass is
hidden. No heat meter or pyranometer: REALITY.md lists neither as read.

**Episode** (rule 17): two weeks from established operation. The reset
settles the building, slab included, at 21 C for the last day's mean weather
and gains, with the supply that holds it there, as a heating curve runs it
[7, p. 7]. No night setback: slab buildings hold a near-constant slab setpoint
[7, p. 7], and [15] advises against setback for radiant floors.

**Score** (rule 15), in euros: the slab's heat at `heat_price`, and each
kelvin-hour of room outside the band or floor above 29 C at `comfort_price`,
counted in Kh as [9] counts discomfort. Inside the band, being nearer its
middle earns nothing. The floor of the score, in closed form: a room held
anywhere in the band costs only its heat.

## 6. Behaviour tests (charter step 3)

`tests/hvac/test_hvac_reality.py` runs the cold review's eleven tests with its
tolerances. On the old plant the module cannot run: it has no slab and its
action is heat power. On the rebuilt plant:

| Test | Source | Result |
|---|---|---|
| 1. Slab time constant 145 to 462 min | [10], [6] | Pass, 150 min |
| 2. Full heat felt only after 60 min or more | [8] | **Fails**: 0.5 K in 45 min at 55 C supply |
| 3. Curve shift moves the room one for one | [11] | Pass, ratio 0.76 |
| 4. Slab output falls when gains rise | [7], [9] | Pass |
| 5. Setback recovery 1 h or more | [15] | **Fails**: 30 min; 8 h with the water off cools the room only to 20.2 C |
| 6. Supply 30 to 35 C at -8 C (+-1 K) | [11] | **Fails**: 26.0 C |
| 7. Heating limit 8 to 15 C | [11], [9] | Pass |
| 8. Floor at most 29 C, supply at most 55 C at design | [5], [17] | Pass: 22.2 C and 27.7 C |
| 9. Morning overshoot after reduced mode | [12] | Pass |
| 10. Design supply never recovers a setback | [15] | Pass, 20.49 C after 15 h (limit 20.5) |
| 11. ASHRAE 55 drift under the curve | [4] | Pass: at most 0.25 K in 15 min and 0.41 K in 1 h |

Known deviations from these (rule 13):

- Test 6 compares a concrete core with [11]'s guidance for screed underfloor
  heating, which has a larger water-to-surface resistance. A TABS with [9]'s
  `R_t` runs 4 to 9 K colder. Effect on the score: none on its structure; the
  supply range an agent uses is lower than underfloor guidance.
- Tests 2 and 5 drive the slab with 55 C water, 27 K over its design supply.
  The screed limit allows it; the TABS buildings in REALITY.md run 30 C
  (Muenster [13]) to 49 C (Madrid [14]). With that much drive the room moves
  within the hour. Effect: an agent can buy fast correction with very hot
  water, at its heat price; the comfort band, not the plant, makes that rare.
- Test 10 passes by 0.01 K, and test 5's setback never reaches its 17 C,
  because the slab holds the room: a slab cannot be set back much, which is
  the sources' reason not to.

## 7. Deviations still in the model

- Heating only; TABS plants are reversible [13, 14]. Overheating above 26 C is
  charged to an actuator that can only stop heating.
- Weather, sun and gains are ours (section 3); no cold snap, sunny spell or
  weekend is drawn as such (cold review 7).
- A free continuous supply every 15 min; practice changes the curve once a day
  and spreads it over 12 h [6].
- Flat heat price; the night tariff that drives slab charging [14] has no
  sourced numbers.
- The 5R1C "medium" mass is kept beside the slab, which a TABS building would
  count in its class.

## 8. What the reference ladder's row needs

- The move is the supply water temperature (C), not heat power. A bump test is
  a supply step; the plant's time constant is hours (test 1), so bumps need
  about a day.
- The steady-state feedforward rung is the heating curve from the plant's
  balance (hvac/experts.py): supply from the 24 h outdoor mean, through the
  design point.
- The cascade rung has a measured inner variable now: the slab core
  (observation channel 3) or the return water (channel 4) [7, 12].
- Cost in euros of heat and Kh outside 21 to 26 C; seeds start established, so
  no warm-up transient enters it.
