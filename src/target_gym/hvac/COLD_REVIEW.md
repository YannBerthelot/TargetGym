# Cold review: `hvac` against REALITY.md

Reviewer had not seen the design. Compared only against `REALITY.md` and its
sources [n]. Files: `env.py`, `env_jax.py`, `PHYSICS.md`, and the `row_hvac`
row of `scripts/reference_ladder.py` on `feat/reference-ladder`. No code changed.

Arithmetic used below (from the defaults in `env.py:65-87`): H_tr_is = 3.45 x
675 = 2329 W/K, H_tr_ms = 9.1 x 375 = 3413 W/K, H_ve = 66 W/K. With the mass
held fixed, heat into the air node sees about 1/(1/2329 + 1/3413) + 33 + 66 =
1480 W/K, so full heat (7.2 kW) lifts the air about 4.9 K above its mass-driven
value within roughly three emitter lags (45 min).

## Mismatches

1. **There is no slab between the water and the room, so slow slab heating is
   absent.** Task: the action is heat power (`env.py:427-431`), delivered
   through a 900 s first-order lag (`env.py:87, 414`) straight into the air
   node balance (`env.py:342, 359`). The only slow state is the whole
   building's 43 h mass (`PHYSICS.md:64`), which the heater reaches only
   through the air. Plant: TABS core time constant 145 to 462 min in heating,
   embedded floors 16 to 161 min, near-surface TABS 7.5 to 22 min [10, p. 7];
   "typically 4 to 8 hours" [6, p. 2]; one to several hours before a setpoint
   change is felt [8, p. 15]. Effect: the agent can move room air by about 5 K
   within the hour (arithmetic above), so intra-day correction works, which
   is exactly what [9, p. 3] says a TABS cannot do ("only day-to-day room
   temperature compensation is promising"). The 15 min lag matches only
   near-surface TABS or light floors. What an agent learns here is a fast
   convective heater on a heavy house, not slab charging a day ahead (the
   operator's job, REALITY section 6).

2. **Actuator is heat power, not supply water temperature, so there is no
   self-regulation.** Task: commanded watts, delivered regardless of room
   temperature (`env.py:414`). Plant: supply temperature via mixing valve on a
   heating curve [11, p. 2], [14, Fig. 15], [12, p. 7]; heat flow falls as the
   room warms because room and slab sit close [7, p. 7], [9, p. 7]. Effect:
   solar and occupancy gains are not absorbed passively; every gain must be
   cancelled by the controller cutting power, which inflates the role of
   anticipation of sun in the score and removes the property that makes real
   TABS tolerable on sunny days. Missing physics: emitter output as a function
   of (supply or mean water temperature minus room or surface temperature).

3. **Night setback of 4 K on a system the sources say should not set back.**
   Task: occupied 20 to 22.5 C, setback 17 C, 22:00 to 07:00
   (`env.py:100-107, 325-329`). Plant: most TABS hold a near constant slab
   setpoint round the clock, few allow setback, and small [7, p. 7]; DOE
   advises no setback for radiant floors [15]; Madrid raises the night
   setpoint to charge the slab [14, p. 14]; Rehau charges outside use hours
   [12, p. 7]. Effect: the task's headline difficulty (recovery from setback,
   `env.py:32-34`, `PHYSICS.md:172-175`) is an operating mode practice avoids
   for slow emitters. The G36 setback (16 C unoccupied [3]) belongs to air
   systems. With the fast heater of item 1 the recovery is also short: the
   expert learns about 5 K/h (`PHYSICS.md:282-284`), against 1.0 to 6.9 h
   recoveries for baseboard homes [15].

4. **Comfort objective far tighter than any source, and no drift term.**
   Task: quadratic cost outside plus or minus 0.5 K of the setpoint while
   occupied (`env.py:150, 540-545`), including overheating. Plant: comfort
   bands of 21 to 26 C [9, p. 5], 22.3 to 26 C operative [6, p. 3]; field
   deadbands 2.8 to 5.6 K, one at 2.2 K had poor comfort [8, pp. 20-21];
   ASHRAE 55 limits operative drift to 1.1 K in 15 min and 2.2 K in 1 h [4].
   Effect: a 1 K band makes the score mostly a count of small solar
   overshoots, which real practice tolerates inside a 5 K band; meanwhile a
   controller that swings the air 3 K in 15 min (possible, item 1) is not
   charged for it.

5. **Observation omits what slab operators read and gives what they do not
   filter.** Task: `[T_air, T_out, heat_pct, solar_norm, sin_h, cos_h,
   target_T]` (`env.py:490-520`), outdoor reading instantaneous with 0.023 K
   noise (`env_jax.py:58-63`). Plant: slab core or return water temperature is
   measured by all eleven designers [7, p. 8], [12, p. 12], [14, p. 15];
   outdoor used as a 24 h mean [9, part 1], [13]; room feedback as the daily
   min or max [9, part 2], [6, pp. 4-5]; outdoor readings "often wrong" (fault
   or sun) [11, p. 1]. Effect: the agent is denied the slab state real
   controllers close the loop on (making the hidden-mass difficulty partly
   artificial) and handed an outdoor sensor cleaner than reported practice.
   A heat meter reading (`heat_pct`) is not among what REALITY lists as read.

6. **Trip at 5 C and 35 C air, where the plant changes mode or has no limit.**
   Task: termination bounds on air (`env.py:110-112, 523-528`) with a priced
   trip and in-place restart (`env.py:157-163`). Plant: below 4.4 C a zone
   puts the group in setback with an alarm until 7.2 C [1], [2, p. 3]; the
   hard limits that exist are on supply water (55 C, safety cut-off even
   without power [17]), floor surface (29 C occupied, EN 1264 [5, p. 23]) and
   condensation in cooling [17]. Effect: the task scores limits the plant does
   not have and omits the ones it does; with a power actuator the 55 C water
   and 29 C surface limits cannot even be expressed.

7. **Every day is the same day; the bad days are missing.** Task: deterministic
   sun with no cloud variation (`env.py:307-314`, `PHYSICS.md:186-191`),
   fixed occupancy and gains every day with no weekend (`env.py:98-101`),
   weather a sinusoid plus OU noise (`env.py:297-304, 433-442`). Plant: the
   hard days are morning start and mode change [12, p. 6], low-gain days and
   weekends [9, part 2], sun above design [6, pp. 4-5], [8, pp. 20-21], and
   cold snaps [15], [11, p. 1]. Effect: a schedule-locked policy can memorise
   the day; robustness to the documented failure days is never tested.

8. **Decision rate and horizon unlike operation.** Task: a free continuous
   action every 15 min (`env.py:134`), 7.5 day episode (`env.py:135`). Plant:
   one mode per day and setpoint changes spread over 12 h [6, pp. 4-5];
   correction limited to plus or minus 5 K per day [9, part 2]; on/off valve or
   PWM pump [6], [9, Fig. 6]; heating to cooling lockout 2 to 24 h [7, p. 9];
   curve tuning judged over two weeks [11, pp. 1, 4]. Effect: the agent can
   chatter at 15 min, which a slab would filter out but this plant does not
   (item 1). Seven days is short of the two weeks a curve change takes to show.

9. **Episode start and end.** Task: starts at midnight, heater off, mass and
   air equal at a uniform 18 to 22 C draw unrelated to the weather or a
   heating history (`env_jax.py:124-161`); ends by truncation with no value on
   the heat left in the mass (`env.py:527`). Plant: a building in continuous
   operation with its slab charged by the curve [7, p. 7]. Effect: the first
   night is a free-float transient, and the last hours reward coasting on
   stored heat, which a continuing building would pay for the next day.

10. **Heating only, while TABS plants are reversible.** Task: no cooling
    (`PHYSICS.md:193-195`), overheating charged but only cut-able
    (`env.py:540`). Plant: reversible heat pumps [13], [14]; changeover
    lockouts [7, p. 9]; shading in buildings that perform well [8]. Effect:
    the upper half of the occupied band is scored against an actuator that
    cannot act on it, so it scores the weather (the reason the setback is
    already one-sided, `env.py:139-141`).

11. **Energy price flat gas.** Task: EUR 0.10/kWh of heat, constant
    (`env.py:154, 546`). Plant: heat pumps [13], [14] and a night tariff that
    drives charging [14, p. 14]; setback saves about 1% per F per 8 h [15].
    Effect: no reason to pre-charge at night, the main economic lever of
    slab operation. Weight is moot while it is not decisive, but the time
    structure of the price is what is missing.

## Unsupported parts

No source in REALITY.md supports the following (the 5R1C envelope parameters
are cited to ISO 13790 in PHYSICS.md, outside this record, and are not judged
here):

- `emitter_tau = 900 s` (`env.py:87`): PHYSICS.md calls it "ours". It falls in
  the near-surface TABS range of [10] but the task describes radiators, for
  which REALITY has no figure.
- Trip bounds `T_air_min = 5 C`, `T_air_max = 35 C`, `failure_cost`,
  `restart_steps` (`env.py:110-112, 157-163`). The nearest source, 4.4 C [1],
  is a mode change, not a trip.
- `comfort_tolerance = 0.5 K` and the quadratic shape (`env.py:150, 156`).
- `setpoint_setback = 17 C` and the per-episode draw 20 to 22.5 C
  (`env.py:104-107`); G36 gives 16 C and 21 C fixed [3].
- Occupancy 07:00 to 22:00 and gains 8 / 2 W/m2 (`env.py:98-101`); the only
  schedule in REALITY is the CBE office, 08:00 to 18:00 [6, p. 3].
- Weather model (mean, amplitude, OU std and rate, `env.py:59, 90-92`) and the
  solar profile shape (`env.py:307-314`): REALITY has no weather statistics.
- `heat_pct` as an observation (heat meter) and the instantaneous outdoor
  reading: REALITY lists filtered outdoor and water or slab sensors instead.
- `gas_price` (`env.py:154`).
- Initial state draw 18 to 22 C, heater off, at midnight (`env.py:131`,
  `env_jax.py:130-161`).

Items REALITY marks "not found" that the task relies on:

- Wall thermostat lag or filter (REALITY 1): the task assumes no dead time
  (`PHYSICS.md:386`).
- A measured slab and room step response (REALITY 4): the task's dynamic
  response has no measured anchor at all.
- Measured overheating on sunny days (REALITY 5): the overheating half of the
  comfort cost, and the floor set by overheating (`PHYSICS.md:329`), rest on
  it.
- Pipe or water freeze limits (REALITY 2): the 5 C trip stands in for one.
- ASHRAE 55 drift values at 0.5, 2 and 4 h (REALITY 3).

## Proposed behaviour tests

Each states the source, the quantity, the experiment and the tolerance fixed
in advance. Tests 1, 3, 6 and 8 need the missing physics of items 1 and 2
(a slab node and a supply water temperature); they are stated so they can be
run once it exists, and against the current task they should fail.

1. **Slab heating time constant.** Source: [10, p. 7, Fig. 3], type E heating
   145 to 462 min (63.2% step response); [6, p. 2] 4 to 8 h. Quantity: heat
   flux from the heated surface into the room. Experiment: steady state at
   constant outdoor 0 C, no sun or gains, then a step in supply temperature
   (today: a step in heat command); time to 63.2% of the flux change.
   Tolerance: inside 145 to 462 min, no margin, since the range already spans
   the 66 cases of the source.

2. **Setpoint change felt only after an hour or more.** Source: [8, p. 15].
   Quantity: time from a step in the heat command (full heat) to a 0.5 K rise
   in room air. Experiment: settled at 20 C, outdoor 0 C, quiet weather,
   constant action then full. Tolerance: at least 60 min; 0.5 K is the
   smallest change the task's own band treats as felt. Current task: about
   15 to 20 min by the arithmetic above.

3. **Heating curve sensitivity, underfloor.** Source: [11, p. 2], lowering the
   curve 2 C lowers the room 2 C for underfloor (5 C gives 2.5 C for
   radiators). Quantity: steady room change over supply curve change.
   Experiment: heating curve run as a fixed recipe, two runs offset 2 K, three
   days each, last 24 h mean compared. Tolerance: ratio 0.75 to 1.25, since
   the source is a rule of thumb given to whole degrees.

4. **Self-regulation.** Source: [7, p. 7], [9, p. 7]. Quantity: emitter heat
   output. Experiment: fixed supply temperature (today: fixed curve recipe
   with no room feedback), add a step of internal gain of 8 W/m2. Tolerance:
   emitter output strictly lower one hour after the step. A sign test, since
   the sources give no magnitude. Fails today by construction.

5. **Setback recovery is hours, not minutes.** Source: [15], recoveries 1.0 h
   (old, uninsulated) to 6.9 h (super insulated) with boilers at twice design
   load. Quantity: time from setback end to air within 0.5 K of the occupied
   setpoint. Experiment: 8 h setback from 21 to 17 C at constant outdoor
   -10 C (the design point of `PHYSICS.md:65`), full heat from the setback
   end. Tolerance: at least 1.0 h, the source's fastest case, for a building
   better insulated than any of them and a heater at 1.5x rather than 2x.

6. **Recommended flow temperature at -8 C.** Source: [11, p. 3], underfloor
   built after 2010, 30 to 35 C. Quantity: steady supply temperature that
   holds 20 C. Experiment: constant outdoor -8 C, no sun, unoccupied gains,
   curve recipe until steady. Tolerance: 30 to 35 C, plus or minus 1 K for
   the steady-state criterion.

7. **Heating limit (balance point).** Source: [11, p. 3], 8 to 15 C outdoor
   for buildings from 1995 to passive; [9, p. 5] 13.1 C. Quantity: outdoor
   temperature at which the free-floating zone holds a 20 C daily mean.
   Experiment: heater off (constant action -1), constant outdoor mean with
   the daily sun and occupancy schedule, no OU noise, five days per point,
   bisect on outdoor mean. Tolerance: 8 to 15 C. Expected to pass today: free
   gains 39.1 kWh/day over 159.4 W/K give about 9.8 C.

8. **Floor surface and water limits at design.** Source: [5, p. 23] 29 C
   occupied surface; [17] 55 C supply cut-off. Quantity: maximum floor
   surface and supply temperatures. Experiment: design outdoor -10 C, heating
   curve recipe holding 21 C. Tolerance: surface at most 29 C and supply at
   most 55 C, no margin, both are stated limits.

9. **Morning overshoot after reduced to normal mode.** Source: [12, p. 6].
   Quantity: peak room air above setpoint in the 4 h after setback end.
   Experiment: heating curve with 50% room influence ([16, p. 80] range) run
   as a fixed recipe, setback 22:00 to 07:00. Tolerance: overshoot greater
   than 0 on a heavy system; a sign test, since the source gives no figure.

10. **No recovery from setback with a heater sized to the load.** Source:
    [15], "may never recover from setback" at design conditions. Quantity:
    air temperature at the end of the occupied period. Experiment: design
    outdoor -10 C, 8 h setback to 17 C, then a constant action equal to the
    steady design load at 21 C (4.8 kW, `PHYSICS.md:66`). Tolerance: still
    more than 0.5 K below setpoint 15 h later; with a 43 h mass the deficit
    should decay by at most about a third over 15 h.

11. **Drift under the curve recipe stays within ASHRAE 55.** Source: [4],
    1.1 K in 15 min, 2.2 K in 1 h. Quantity: largest occupied air change over
    any 15 min and any 1 h window. Experiment: heating curve recipe on the
    held-out weather seeds. Tolerance: the two limits as stated. This checks
    that the plant, run as practice runs it, does not produce changes no
    source allows; a slab plant cannot, the current fast heater can.
