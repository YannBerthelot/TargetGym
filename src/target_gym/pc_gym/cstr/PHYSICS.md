# Jacketed CSTR: physics model, sources, limits and deviations

A PARSEX-like member of the jacketed exothermic CSTR family (charter Amendment
3, rule 19): a continuous stirred tank with a first-order exothermic reaction
A to B, cooled through a jacket by fresh cooling water. `REALITY.md` here is the
family's record, sources [S1] to [S12]; `COLD_REVIEW.md` lists how the task
before version 5 differed.

**Status.** Version 8, by the owner's decision of 2026-11-24: the reactor is run
for what it makes. The task of versions 5 to 7 held a temperature setpoint with
one valve; a PI of the ladder's own form at four times its gain closed most of
its room, since nothing in that plant made gain cost. With the feed as a second
move, the cooling's limit and the interlock come into play, and what practice
does when the cooling saturates, cut the feed [S13], becomes the job. How the
version was prototyped, measured and reviewed is at the end of this page.

**Named property: a cooling limit the production runs against, on a reactor
whose first answer to a feed cut is to heat.** The jacket is in the model, with
its own energy balance and holdup; the feed is colder than the reactor, so less
of it cools less at once and reacts less only later. Fouling is gone: it acts
over weeks and has no measured rate (owner's decision).

---

## 1. Model

```
dC_A/dt = (C_Af - C_A) q / V - k(T) C_A,             k = k0 exp(-E/R / T)
dT/dt   = (T_f - T) q / V + (-dH) k C_A / (rho cp) - UA (T - T_j) / (rho cp V)
dT_j/dt = [F_w cp_w (T_w,in - T_j) + UA (T - T_j)] / (M_j cp_w)
```

PARSEX's reactor recirculates through an external cooler with water on the
shell side [S1, p. 3, Fig. 1]; here the cooler is lumped as a jacket of its own
holdup. Its reaction is first order A to B [S1, p. 4]. The operator moves the
fresh cooling water [S1, p. 4], from S1's 40 g/s floor [S1, Table 1] to the
136 g/s S1 logs [S1, Fig. 2], and the feed, up to S1's 110 g/s [S1, Table 1]:
the feed is what practice cuts when the cooling saturates [S13, p. 436]. A high
reactor temperature trips an interlock that cuts the feed [S3, PDF p. 38].

---

## 2. Numbers: cited, derived or ours

| Symbol | Value | Unit | Cited or ours |
|---|---|---|---|
| `k0` | 1.2e13 | 1/min | cited: 2.0e11 1/s [S1, p. 4] |
| `EA_over_R` | 10700 | K | cited [S1, p. 4] |
| `deltaHr` | -2.0e5 | J/mol | cited: -200 kJ/mol [S1, p. 4] |
| `Caf` | 1.5 | mol/L | cited: 1.5 kmol/m3 [S1, p. 4] |
| `q` | 5.4 | L/min | cited: the design feed, 90 g/s [S1, Tables 3, 4], as water |
| `q_min`, `q_max` | 2.4, 6.6 | L/min | the top cited: 110 g/s [S1, Table 1]; the floor ours, 40 g/s: S1's range starts at 0 and no turndown was found |
| `V` | 183.6 | L | derived: q x the 34 min residence [S1, p. 5] |
| `rho`, `C` | 1000, 4.18 | g/L, J/(g K) | the aqueous feed as water, textbook |
| `UA` | 46717 | J/(min K) | derived: the duty S1's design point leaves (58 C, feed 24 C, CA 0.31 mol/L), 8.56 kW, over the 11 K to the cooler's 47 C [S1, Tables 3, 4] |
| `Fw_min`, `Fw_max` | 2400, 8160 | g/min | cited: the 40 g/s floor [S1, Table 1]; the top is the 136 g/s S1 logs [S1, Fig. 2], since the plant ran above Table 1's 110 g/s: "around 120 g/s" at the design point [S1, p. 6] |
| `Tw_in` | 303.08 | K | derived: the 8.56 kW of `UA`'s row at S1's 120 g/s with the cooler at 47 C [S1, p. 6; Tables 3, 4] is a 17.1 K rise, so water enters at 29.9 C; the design point then needs 112 to 128 g/s over the feed's range |
| `M_j` | 36720 | g | ours, family: 20 % of the reactor's mass, a jacket time constant of 2.2 min (derived), the scale of the 2 min jacket dead time of [S6] |
| `feed_T_range` | 296.25, 298.05 | K | cited: 23.1 and 24.9 C on two days [S1, Tables 3, 4]; the feed at room temperature swings across it |
| `feed_T_period` | 1440 | min | ours: room temperature cycles once a day; the phase is drawn per episode, so an episode sees up to about 0.9 K of drift |
| `T_setpoint` | 331.15 | K | cited: 58 C [S1, Tables 3, 4] |
| `T_band` | 1 | K | ours: product made further from the setpoint is off-spec; no tolerance found (REALITY.md, third and fourth searches); flat inside by the owner's decision (2026-11-24); the sweep draws it from 0.5 to 2 |
| `setpoint_ramp` | 60 | min | derived: S1's closed loop took about an hour to carry its 2 C move to its peak [S1, Fig. 2] |
| `valve_rest` | 0.95 | of span | cited: where [S13]'s example holds the cooling valve, short of its limit [S13, pp. 436, 438] |
| `T_trip` | 348.15 | K | ours: 75 C, 17 K over the setpoint and short of the 81 C adiabatic rise, so a runaway trips; the interlock is cited [S3], its setting for this chemistry is not found |
| `delta_t`, `max_steps_in_episode` | 0.25, 960 | min, steps | ours: 4 h, over twice the 110 min a 2 C step took to settle [S1, Fig. 2] |
| `event_window` | 30, 150 | min | ours |
| `setpoint_step` | 2 | K (scenario) | cited: a 2 C step [S1, Fig. 2] |
| `k0_drop` | 0.4 | - (scenario) | cited: a 40 % drop [S1, Fig. 3] |
| `k0_decay` | 90 | min (scenario) | ours: no source gives a rate inside hours; the slowest linear decay that ends inside the 240 min episode from the latest event, at 150 min |
| `feed_rise` | 0.1 | - (scenario) | ours: feed build-up is sourced [S3], its size is not; with the feed to cut it can be held (D6) |
| (version 7's) | 0.03 | - | derived: feed build-up is sourced [S3], its size is not, so it is the largest whole percent the valve can hold at the design point, 135.5 of its 136 g/s at the middle feed temperature (112 to 128 g/s of design water leaves 1.5 % on the warmest feed and 4.7 % on the coldest) |
| `cooling_loss` | 5 | min (scenario) | ours: cooling failures are sourced [S2; S3], their length is not |
| `event_hold` | 480, 1440, 120 | min (stream, per scenario) | ours: how long a setpoint move (a shift), a catalyst loss (until a catalyst change, a day) and a feed rise (until its batch runs out) last before they return; no source gives these; a window never sets it, so its events hold to the end |
| readings | 0.04 K; lag 13 s; dead time 2 min | - | Pt100 scatter as before; thin-walled well about 13 s [S10, Fig. 2]; jacket outlet about 2 min [S6] |

---

## 3. Limits (rule 14)

| Limit | Kind | Model | Source |
|---|---|---|---|
| High reactor temperature | Trip | The interlock cuts the feed at 75 C; restart priced as 4 h of nameplate production (ours) | [S3, PDF p. 38]; setting ours |
| Temperature band | Zone | Product made more than 1 K from the setpoint is off-spec and counts for nothing; flat inside | ours (no tolerance found); owner, 2026-11-24 |
| Cooling water | Hardware | 40 to 136 g/s; none during a cooling loss | [S1, Table 1, Fig. 2]; [S2; S3] |
| Feed | Hardware | 40 to 110 g/s | top [S1, Table 1]; floor ours |

---

## 4. Observation (rule 16)

`[T, T_j, F_w_pct, q_pct, T_setpoint]`: the reactor temperature through its
thermowell [S4; S10], the jacket outlet temperature, 2 min late [S6; S8,
Fig. 6], the water and the feed flows as logged on PARSEX [S1, Fig. 2; Table 1],
and the setpoint. No composition: an online analyser on such a reactor was not
found, and Bayer sampled for the lab once a day [S3]. Pressure, level and
circulation flow [S3, Fig. 11] are not modelled (D2).

### Task: `cstr`

Run a jacketed exothermic reactor for what it makes, with two moves: the
cooling water and the feed. Each step is charged the production lost against
the reactor's nameplate, and all of it while the temperature is more than 1 K
from its setpoint, where the product is off spec. The policy reads the reactor
and jacket temperatures, its two flows and the setpoint, and never the
composition. A reactor that reaches 75 C trips its interlock, charged as 4 h of
lost production.
The reference keeps the band on the normal day and through a setpoint move,
but not in the three upsets (catalyst loss, a richer feed, a short loss of
cooling): there it leaves the band on every seed, and their rooms are reported
as measured.

| obs | meaning |
|---|---|
| 0 | `T`, reactor temperature (K) |
| 1 | `T_j`, jacket temperature (K), 2 min late |
| 2 | cooling water flow, percent of its 40 to 136 g/s range |
| 3 | feed flow, percent of its 40 to 110 g/s range |
| 4 | reactor temperature setpoint (K) |

---

## 5. Episode and scenarios (rules 17 and 20)

A window starts in established production: the reactor on its 58 C setpoint
for the day's feed temperature, the water valve at 95 % of its span [S13] and
the feed that takes, or, where the feed's top binds, the feed at its top and
the water that takes. `scenarios.py`:

| Scenario | Event | Source |
|---|---|---|
| `normal` | the feed drifts with the room, 23 to 25 C over a day | [S1, Tables 3, 4] |
| `setpoint_move` | setpoint +2 K, as a ramp over an hour, the band with it | [S1, Fig. 2]; the ramp derived |
| `catalyst_loss` | activity -40 % over 90 min (rate ours) | [S1, Fig. 3; p. 6] |
| `feed_buildup` | feed reactant +10 % (ours) | [S3, PDF pp. 36-37, 46] |
| `cooling_loss` | no water for 5 min (ours) | [S2, PDF p. 17]; [S3, Fig. 11] |

Each event comes once, at a minute drawn in 30 to 150.

---

## 6. Score (rule 15)

The job [S13; S5; S6]: make product, inside the temperature band, and never run
away. The score is the production lost against the nameplate, the top feed fully
converted, as a fraction of it per step:

```
reward = -(1 - made / nameplate)             inside the band
       = -1                                  outside it (the product is off-spec)
       = -960 x 1                            on an interlock trip
made = q (C_Af - C_A),   nameplate = q_max C_Af
```

One unit throughout, so no price is needed. Nothing pulls toward the setpoint
inside the band (owner, 2026-11-24).

| parameter | value | source |
| --- | --- | --- |
| `T_band` | 1 K | ours: no tolerance found (REALITY.md, third and fourth searches) |
| `e_floor` | 0.1 K | ours: a reactor probe's resolution, what the band's edge is read to |
| `failure_cost` | 1 | a step of nameplate production |
| `restart_steps` | 960 | ours: 4 h of lost production, seven residence times; no restart time found |
| `rho_floor` | 0 | not measured on this plant |
| `precision_floor` | 0.1 K | version 1 only |

---

## 7. Behaviour tests

`tests/pc_gym/cstr/test_cstr_reality.py`, COLD_REVIEW.md's T1 to T8 with T6
restated on the family (Amendment 3):

| Test | Range and source | Result |
|---|---|---|
| Design point | 74 to 79 % conversion, cooler near 47 C, water from 10 % under 120 g/s to 136 g/s [S1, p. 6, Fig. 2] | pass |
| T1. Rate doubles per 10 C | 1.5 to 3.0 [S5] | pass |
| T2. 40 % activity loss | cools at least 1.7 K [S1, Fig. 3] | pass |
| T3. Jacket delay | at least 60 s to 10 % of the response [S6] | pass |
| T4. Thermowell | 4 to 45 s [S10] | pass |
| T5. Analyser | no composition reading (none found) | pass |
| T6. Loss of circulation | 0.75 to 8 K/min over 3 min, pooled: Morton's 1 to 2 C/min [S4], Bayer's over 2 C/min [S3], the audit's factor 4 on top and 25 % below | pass (0.81) |
| T7. 25 K rise | 3 to 48 min [S2; S3] | pass |
| T8. Open-loop settling | within 5 % of its final value from 1.6 to 6.4 residence times on [S1, Fig. 2], at the feed the reset holds | pass |
| A feed cut first heats, then cools | the balances; the feed is colder than the reactor [S1, Tables 3, 4] | pass |
| The feed build-up is past the cooling and inside a feed cut | [S3]; [S13, p. 436] | pass |
| The score is product inside the band | section 6 | pass |
| A setpoint move is a 2 K ramp the band follows | [S1, Fig. 2] | pass |
| A window starts in established production | [S13, pp. 436, 438] | pass |
| Runaway trips the interlock | [S3] | pass |
| Starts at rest on its setpoint | rule 17 | pass |

**T8, how it is read (owner, 2026-11-23).** Open loop the design point is an
underdamped oscillator: poles at -0.0197 +- 0.0465i per min (a period of
135 min, damping 0.39) and the jacket's at -0.53 per min. A cooling step
overshoots its final value by 25 %, peaks at 58.5 min and stays within 5 % of
its final value from 126.5 min, 3.72 residence times. It first reaches 95 % of
that value at 0.99 residence times, which is a rise time: the test read that
figure until 2026-11-23 and stood as a strict xfail. S1's own figure is a
settling time after an overshoot (a peak near 60 min, settled after about 110
[S1, Fig. 2]), so the plant is now read the same way. The plant and the audit's
range are as the owner kept them on 2026-11-12. The finding of 2026-11-12 (the
plant settling too fast, kept as a strict xfail) is withdrawn by the owner's
decision of 2026-11-24: it was the test's measure, not the plant.

---

## 8. Known deviations (rule 13)

**D1. A lumped jacket.** PARSEX cools through an external recirculating
cooler [S1]; here one well-mixed volume with no transport delay. The 2 min
dead time of [S6] is on the jacket reading, not the heat path.

**D2. No pressure, level or circulation flow.** At Bayer the first alarm acted
on was high pressure [S3, Fig. 11]. Effect: the runaway is seen on temperature
alone.

**D3. Two handles, the water and the feed.** PARSEX also moves feed
concentration [S1, p. 4]; HSE lists inhibitor, quench and dump [S5]. Effect: a
narrower job, with no last resort past a feed cut.

**D4. A simulated reaction.** PARSEX injects steam for the heat a reaction
would release [S1]; the model is that reaction.

**D5. The catalyst loss is a decay, not S1's step.** S1's 40 % loss was
instantaneous, which its authors say "is not likely to occur" in operation,
where activity "is always expected to decay throughout the operation"
[S1, p. 6]. The owner chose the decay (2026-11-09): the same 40 % falls
linearly over 90 min. No source gives a rate inside a 4 h window (real
deactivation runs over a catalyst's life), so the rate is ours under the family
ruling: the slowest that ends inside the episode from the window's latest
event (240 - 150 min). S1 rode its step out on feed concentration, a handle
this task lacks (D3). With S1's 120 g/s design water the steady water after the
loss is 64 to 75 g/s, above the 40 g/s floor; with v5's 24 C water it sat at 50
to 58 g/s and every rung quenched the reactor under the step.

**D6. The feed build-up stops at the valve's top (owner, 2026-11-23).** Until
then the rise was 10 % (ours). Holding 58 C under it needs 179 g/s of water
against the valve's 136, so the reactor sat 5 K high whatever the controller
did: 1272 per step for the foresight ceiling and 1360 for the reference, 97 %
and 69 % of their costs over the suite. Bayer's cooler held its temperature
only while the feed stayed under a limit [S3, PDF pp. 36-37], so the documented
day is a feed at the edge of the cooling, not past it. What practice does past
it is cut the feed (version 7 had no such handle). Version 8 has it, so its
build-up is the 10 % again and the scenario is one a feed cut could
hold; the reference does not (owner, 2026-11-24: reported unheld).
A 2026-11-25 trial of S13's high-temperature override on the feed (a low
select 0.1 K under the band's top, tuned by a bump test and SIMC) held none of
the three upsets: all 30 upset episodes still left the band at every set point.
It cost about 5 % less in this build-up and 43 % more in cooling loss, so the
ladder kept the PI. A feed cut first heats the reactor, as less cold feed
comes in, and acts too slowly for a 1 K band.

**Finding, J2 (v6, decay).** Under S1's step the ladder's cascade lost to the
I-only null on all 10 seeds (median J 403 against 124), almost all of it the
catalyst loss (3045 against 248 per step): SIMC's loops gave the valve's span
away over 16 min while the null, a relay at its gain, floored the valve in one
step. Nothing in the row was wrong (sign, pairing, saturation). With the loss
as a 90 min decay (D5) the cascade's median J is 90.7 and I-only's 88.5; the
null wins 5 of 10 seeds, not more than half, so J2 passes, at its edge. The
catalyst loss now costs the cascade 158 per step against I-only's 17, and the
normal day 1.6 against 22.5. The I-only gain and the null table were fitted on
v5 and are stale.

---

## 9. Expert

Valve position control [S13, section 2.3]: the reference ladder's rung of that
name, written out with its gains as leaves (`experts.py`; a test holds the two
laws equal move for move). A PI holds the reactor's temperature by the cooling
water, and a slow PI holds the water valve at a set point by the feed, never
past the throughput asked for. Both take the valves over where the reset left
them.

Its recipe is `tune_cstr` in `scripts/tune_process_experts.py`, run on version 8
at the three noise levels on 2026-11-24 (owner's word of that day; only cstr's
rows of the three gains files moved):

1. The temperature loop. In manual with both valves held, the water at 80 % of
   its span so a bump has room and the feed's temperature at the middle of its
   day, a water bump read for 110 min [S1, Fig. 2], then the noise protocol's
   SIMC (N5 to N7).
2. The valve position loop. The same plant with that loop closed, a feed bump
   read on the water valve for three hours, and SIMC at 30 dead times, which
   its source asks for at least [S13].
3. The two set points, on the scored plant's normal day. On a grid fixed before
   any run (the valve at 99, 98, 95 and 90 %; the temperature in tenths of a
   kelvin under the band's top, down to its centre), the cheapest at which no
   deciding training seed leaves the band or trips (owner's rule, 2026-11-23).
   The project's own function until the ladder's shared placement is merged.
4. Where no set point keeps the band, the valve position loop's closed-loop
   time doubles (60, then 120 dead times) and step 3 runs again; the last rung
   ships with the seeds it lost if none does (C9). The project's own rule, a
   fixed step at every level (accepted by the owner, 2026-11-24), added when
   the half-noise level failed step 3. The loop's dead time is read off the
   noise band, 1.9 min at half the noise against 2.6 at the scored level, so 30
   of them made a 56 min loop that swung with the temperature loop: the best
   set point still lost the band on 4 of 10 seeds.

| noise | Kp (raw/K) | Ti (min) | Tf (min) | Kv | Tiv (min) | valve loop's time | under the top | valve |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.164 | 17.7 | 0.04 | 0.375 | 36.7 | 60 dead times, 112 min | 0.9 K | 99 % |
| 1 | 0.107 | 15.6 | 0.18 | 0.623 | 45.4 | 30 dead times, 79 min | 0.9 K | 95 % |
| 2 | 0.097 | 14.8 | 0.69 | 0.494 | 46.7 | 30 dead times, 101 min | 1.0 K | 98 % |

No seed was left out at any level. The expert is not the charter's reference,
which is the ladder's rung on its own tests (section 10).

## 10. What the reference ladder row needs

`scripts/ladder_rows/cstr.py`: the temperature on the water by SIMC, and the
water valve held short of its limit by the feed, the ladder's valve position
rung [S13]. Bump tests from a reset with both valves in mid-range, at the
middle and both ends of the feed's temperature range. The rungs start bumpless:
the reset stands on the day's feed temperature, which no reading shows. The
feedforward, the cascade and a second temperature loop on the feed were each
tried under the valve position loop on 2026-11-23 and leave the band at every
set point, the last because a feed cut first heats the reactor.

**The reference does not hold the three upsets** (owner, 2026-11-24: report
them as unheld). On the ladder's seeds every candidate set point leaves the
band on all ten seeds of `catalyst_loss`, `feed_buildup` and `cooling_loss`, on
both rungs, so they do not constrain the placement. The kept rung is the PI,
the feed held where the reset left it, 0.7 K above the setpoint (median J
0.3331 against 0.3334 for valve position). Per-scenario rooms are reported as
measured: about 1.15 on the normal day, 1.21 on the setpoint move, and 1.9 to
2.3 in the upsets, where the reference is close to holding the reset's valves;
the task's room is 1.39 (ceiling not re-solved on the merged tree).

## 11. Oracle needs

No oracle on the shared solver ships with version 8: the planner model of
versions 5 to 7 moved one valve to track a setpoint, and the owner decided on
2026-11-24 that 0.8.0 does not need one for two moves and the production score
(the charter's MPC line reads "not measured" until a later update; room and
saving do not use it). The foresight ceiling's
pieces are in `oracle.py`: the smooth term (the production lost) and, as
margins, the band's two edges and the interlock, priced at a step's product.

---


### Scenario weights (charter J4)

`normal` 0.7, `setpoint_step` 0.1, `catalyst_loss` 0.1, `feed_buildup` 0.05, `cooling_loss` 0.05: shares of operating time for the headline. No source says how often each
window occurs (`REALITY.md`), so each is ours, the normal day in the family
range 0.5 to 0.9 and the rest split among the upsets; the family sweep (B7)
draws them.

### The stream (docs/proposals/long-strata.md, S1)

As a continuing plant (`target_gym.stream`) the normal day lasts 24 h on average
and each upset regime a shift, 8 h, with one event per visit on average
(`events_per_h` 1/8). Each event carries the size it started with, so it runs to
its return through a switch of regime, and the feed's day wraps on a step count.
The normal day is sourced: abnormal situations arise almost daily in large
process plants (`REALITY.md` [ST1]). That an upset regime lasts a shift is
ours; shifts run 8 to 12 h ([ST2], [ST3]). The events per visit and the holds
are ours, searched again on 2026-11-11 without a source (`REALITY.md`). In a window nothing changes:
`event_hold` is never reached and a scored window is bit-identical to v6's.

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `cstr` | 960 | 15 | 4.0 h | 2 in [-1, 1] | 5 | 35 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Readings

The reactor temperature and the jacket outlet through their instruments, the
water and the setpoint exactly. `sensor_noise_scale = 0` removes the noise and
keeps the lag and the dead time.

| reading | sigma | lag | dead time | instrument | source |
| --- | --- | --- | --- | --- | --- |
| `T` | 0.04 K | 13 s | none | Pt100 RTD in a thin-walled thermowell | [S10, Fig. 2] |
| `T_j` | 0.04 K | none | 2 min | Pt100 at the jacket outlet | [S6] |

## Written exception to check 2

The credibility check on a sourced real figure has nothing to read here. The owner
accepted a class exception on 2026-11-06: no published figure could be found or opened
after two searches with a re-fetch check. The plan in docs/proposals/industrial-fixes.md
says what was tried.

## How version 8 was prepared (the production prototype, 2026-11-23)

The owner asked for a version of this task scored on what the reactor makes, to
be prepared to replace the tracking task and brought back to him first. It is
`scripts/prototypes/cstr_production.py`: this plant with the feed as a second
handle, scored in lost nameplate production. It was unregistered when the
figures below were measured; the owner made it version 8 on 2026-11-24 and its
plant moved into this package. No record has been taken on the registered version.

Why: on the tracking task a PI of the ladder's own form at four times its gain
closes most of the room, since nothing in the plant makes gain cost. With the
feed to move, the cooling's limit and the interlock come into play, and what
practice does when the cooling saturates, cut the feed [S13], becomes the job.

Its own numbers, each the project's and none from a source (REALITY.md, third
search, says what was looked for):

| number | value | why it is ours | range the sweep draws |
|---|---|---|---|
| band on the temperature | 1 K either side of the setpoint; product made outside it counts for nothing | no tolerance found in [S1], [S9] or [S12] | 0.5 to 2 K |
| feed's floor | 40 g/s | S1's range starts at 0 [S1, Table 1]; no turndown found | 0 to 40 g/s |
| start of a window | the reactor on its setpoint, the water valve at 95 % of its span and the feed that takes (or the feed at its top and the water that takes) | sourced since the first sweep: [S13]'s example holds the valve at 95 %, short of its limit. The first choice, the valve at its top (ours), left no cooling at the first step, and the reference could not be placed in a band under 0.9 K | none |
| setpoint move | S1's 2 C [S1, Fig. 2] as a ramp over 60 min | derived: S1's closed loop took about an hour to its peak, and a band that jumped 2 K could not be kept | 30 to 120 min |
| feed build-up | 10 % | sourced as an event [S3], its size ours; with a feed to cut it can be held | 3 to 10 % |
| cooling loss | 5 min | as the task above | 2 to 8 min |
| weights | normal 0.7, setpoint move 0.1, catalyst loss 0.1, feed build-up 0.05, cooling loss 0.05 | no source says how often each occurs | normal 0.5 to 0.9, the rest in proportion |
| trip | 4 h of nameplate production | as the task above | none |

Sourced: the feed's top, 110 g/s [S1, Table 1]; the structure of the reference
and its valve position loop's closed-loop time of 30 dead times [S13].

The reference was then the ladder's valve position rung (on the merged tree
the ladder keeps the PI, see the upsets' ruling above)
(`scripts/reference_ladder.py`): the temperature on the water by SIMC, the
water valve held by the feed. Set points are placed on a fixed grid by the
owner's rule of 2026-11-23, the cheapest at which no deciding seed of the
normal day leaves the band.

### What was measured (2026-11-23, the prototype at its 95 % start)

The reference: temperature on the water, Kc -0.119 raw per K and Ti 27.5 min;
the water valve held by the feed, Kc 0.53 and Ti 46.5 min; placed with the
temperature 0.9 K under the band's top and the valve at 99 %, no seed left out.
Cost is lost nameplate production per step, seeds 100 to 109.

| scenario | weight | reference | ceiling | room | reference off-spec, hottest |
|---|---|---|---|---|---|
| normal | 0.70 | 0.2637 | 0.2385 | 1.11 | 0 %, 59.0 C |
| setpoint move | 0.10 | 0.2445 | 0.2174 | 1.13 | 0 %, 60.6 C |
| catalyst loss | 0.10 | 0.6431 | 0.2790 | 2.31 | 51 %, 65.1 C |
| feed build-up | 0.05 | 0.6835 | 0.3191 | 2.14 | 57 %, 67.6 C |
| cooling loss | 0.05 | 0.5288 | 0.2540 | 2.08 | 36 %, 64.1 C |
| J | | 0.3340 | 0.2453 | 1.36 (1.33 to 1.39) | no trip |

The reference costs more than the ceiling on 10 of 10 seeds in every scenario.
In product, 6.59 against 7.47 mol/min on-spec, 13.3 % more (11.9 to 14.8). The
ceiling is never the reference's own moves; its two starts end 0.24 % apart at
the median and 12.9 % at worst.

J2's runs, on 30 seeds hashed from commit ad3b60f: the reference trips on none;
holding the reset's valves costs less on 4 seeds (2 in the catalyst loss, 2 in
the feed build-up), both valves wide open on 1 (cooling loss), the other nulls
on none.

### The sweep (charter B7) and J2's rule

Eight Latin-hypercube draws of the project's own numbers over the ranges above,
seeds 100 to 109. On every draw the ladder was rebuilt and placed with no seed
left out, the reference tripped on no seed and cost less than every null on the
median over seeds (0.276 to 0.397 against the best null's 0.323 to 0.709). On
the two draws at the ends of the band the ceiling cost less than the reference
on the same five seeds: 0.258 against 0.396 at 0.63 K, 0.225 against 0.314 at
1.89 K.

So J2's rule accepts the prototype: no trip in 30 seeds, no null ahead on more
than half of them, the sweep holding. These are the prototype's own runs of the
rule, not the charter's script from a tag, and the sweep ran no behaviour test
of its own (the plant's are the task's above).

Two earlier sweeps failed, and each was a fault found, not a number moved. With
the window starting at the valve's top, no set point held a band under 0.9 K
(the start is now the record's 95 %). With a placement grid that stopped 1 K
under the band's top, none held the two widest bands (the grid now reaches the
band's centre, as it was written to).

### Cold review (charter B8, 2026-11-23)

A fresh reader was given REALITY.md, then the task only. Its 13 findings, and
what each is here:

| finding | what it is here |
|---|---|
| The band is a cliff with no pull to the setpoint inside it; the record's job is the setpoint and it gives no tolerance (major) | open, the owner's: the band is the project's own number, and the score invites riding its upper edge |
| A trip is a fee with an instant restart, where the record's runaways end in rupture (major) | as the task above: the trip's 4 h is ours |
| The cooling loss is a 5 min ride-through, the trip 17 K away (major) | as the task above; here the reference still loses the band 36 % of the window and reaches 64 C |
| The window started with the water valve at its top, no cooling left at the first step, where the record holds it short (major) | fixed from the record: the window now starts at 95 % [S13] |
| Production is paid every step on a composition no analyser reads (unknown) | deviation: the score reads the true state, as every task's does; the agent reads no composition |
| The feed moves in one step, with a fixed volume and a 40 g/s floor against S1's 0 (minor) | deviation: feed valve and level dynamics not found; the floor is ours |
| The normal day rests on a hidden daily feed cycle the record does not give (unknown) | as the task above: the period is ours |
| The setpoint move is a ramp where S1 stepped (minor) | deviation, derived |
| The catalyst loss is a 90 min decay (minor) | as the task above (D5) |
| The feed build-up is a step in feed strength, where Bayer's was unreacted material accumulating (unknown) | as the task above; its size is ours |
| The water's top is 136 g/s where Table 1 says 110 (unknown) | as the task above [S1, Fig. 2] |
| No circulation loss or fouling in the suite (unknown) | as the task above |
| Flows are read exactly; the jacket's holdup is ours (unknown) | as the task above |
