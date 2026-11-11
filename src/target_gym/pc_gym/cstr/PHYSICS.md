# Jacketed CSTR: physics model, sources, limits and deviations

A PARSEX-like member of the jacketed exothermic CSTR family (charter Amendment
3, rule 19): a continuous stirred tank with a first-order exothermic reaction
A to B, cooled through a jacket by fresh cooling water. `REALITY.md` here is the
family's record, sources [S1] to [S12]; `COLD_REVIEW.md` lists how the task
before version 5 differed.

**Status.** Back from Extra by the owner's decision (Amendment 3, 2026-11-08).
The earlier Extra rested on one figure: PARSEX's loss of cooling heats at
0.81 K/min where the audit's T6 asked 2 K/min, from Bayer's different plant.
Under rule 19 that is a family member's figure, and T6 is restated on the
family's pooled range below.

**Named property: the jacket, and the runaway it must stay ahead of.** The
jacket is in the model, with its own energy balance and holdup; a loss of
circulation runs the reactor away until the interlock trips (rule 18). Fouling
is gone: it acts over weeks and has no measured rate (owner's decision).

---

## 1. Model

```
dC_A/dt = (C_Af - C_A) / tau - k(T) C_A,             k = k0 exp(-E/R / T)
dT/dt   = (T_f - T) / tau + (-dH) k C_A / (rho cp) - UA (T - T_j) / (rho cp V)
dT_j/dt = [F_w cp_w (T_w,in - T_j) + UA (T - T_j)] / (M_j cp_w)
```

PARSEX's reactor recirculates through an external cooler with water on the
shell side [S1, p. 3, Fig. 1]; here the cooler is lumped as a jacket of its own
holdup. Its reaction is first order A to B [S1, p. 4]. The operator moves the
fresh cooling water [S1, p. 4], from S1's 40 g/s floor [S1, Table 1] to the
136 g/s S1 logs [S1, Fig. 2]. A high
reactor temperature trips an interlock that cuts the feed [S3, PDF p. 38].

---

## 2. Numbers: cited, derived or ours

| Symbol | Value | Unit | Cited or ours |
|---|---|---|---|
| `k0` | 1.2e13 | 1/min | cited: 2.0e11 1/s [S1, p. 4] |
| `EA_over_R` | 10700 | K | cited [S1, p. 4] |
| `deltaHr` | -2.0e5 | J/mol | cited: -200 kJ/mol [S1, p. 4] |
| `Caf` | 1.5 | mol/L | cited: 1.5 kmol/m3 [S1, p. 4] |
| `q` | 5.4 | L/min | cited: 90 g/s of feed [S1, Tables 3, 4], as water |
| `V` | 183.6 | L | derived: q x the 34 min residence [S1, p. 5] |
| `rho`, `C` | 1000, 4.18 | g/L, J/(g K) | the aqueous feed as water, textbook |
| `UA` | 46717 | J/(min K) | derived: the duty S1's design point leaves (58 C, feed 24 C, CA 0.31 mol/L), 8.56 kW, over the 11 K to the cooler's 47 C [S1, Tables 3, 4] |
| `Fw_min`, `Fw_max` | 2400, 8160 | g/min | cited: the 40 g/s floor [S1, Table 1]; the top is the 136 g/s S1 logs [S1, Fig. 2], since the plant ran above Table 1's 110 g/s: "around 120 g/s" at the design point [S1, p. 6] |
| `Tw_in` | 303.08 | K | derived: the 8.56 kW of `UA`'s row at S1's 120 g/s with the cooler at 47 C [S1, p. 6; Tables 3, 4] is a 17.1 K rise, so water enters at 29.9 C; the design point then needs 112 to 128 g/s over the feed's range |
| `M_j` | 36720 | g | ours, family: 20 % of the reactor's mass, a jacket time constant of 2.2 min (derived), the scale of the 2 min jacket dead time of [S6] |
| `feed_T_range` | 296.25, 298.05 | K | cited: 23.1 and 24.9 C on two days [S1, Tables 3, 4]; the feed at room temperature swings across it |
| `feed_T_period` | 1440 | min | ours: room temperature cycles once a day; the phase is drawn per episode, so an episode sees up to about 0.9 K of drift |
| `T_setpoint` | 331.15 | K | cited: 58 C [S1, Tables 3, 4] |
| `T_trip` | 348.15 | K | ours: 75 C, 17 K over the setpoint and short of the 81 C adiabatic rise, so a runaway trips; the interlock is cited [S3], its setting for this chemistry is not found |
| `delta_t`, `max_steps_in_episode` | 0.25, 960 | min, steps | ours: 4 h, over twice the 110 min a 2 C step took to settle [S1, Fig. 2] |
| `event_window` | 30, 150 | min | ours |
| `setpoint_step` | 2 | K (scenario) | cited: a 2 C step [S1, Fig. 2] |
| `k0_drop` | 0.4 | - (scenario) | cited: a 40 % drop [S1, Fig. 3] |
| `k0_decay` | 90 | min (scenario) | ours: no source gives a rate inside hours; the slowest linear decay that ends inside the 240 min episode from the latest event, at 150 min |
| `feed_rise` | 0.1 | - (scenario) | ours: feed build-up is sourced [S3], its size is not |
| `cooling_loss` | 5 | min (scenario) | ours: cooling failures are sourced [S2; S3], their length is not |
| `event_hold` | 480, 1440, 120 | min (stream, per scenario) | ours: how long a setpoint move (a shift), a catalyst loss (until a catalyst change, a day) and a feed rise (until its batch runs out) last before they return; no source gives these; a window never sets it, so its events hold to the end |
| readings | 0.04 K; lag 13 s; dead time 2 min | - | Pt100 scatter as before; thin-walled well about 13 s [S10, Fig. 2]; jacket outlet about 2 min [S6] |

---

## 3. Limits (rule 14)

| Limit | Kind | Model | Source |
|---|---|---|---|
| High reactor temperature | Trip | The interlock cuts the feed at 75 C; restart priced as 4 h of lost production (ours) | [S3, PDF p. 38]; setting ours |
| Cooling water | Hardware | 40 to 136 g/s; none during a cooling loss | [S1, Table 1, Fig. 2]; [S2; S3] |

---

## 4. Observation (rule 16)

`[T, T_j, F_w_pct, T_setpoint]`: the reactor temperature through its
thermowell [S4; S10], the jacket outlet temperature, 2 min late [S6; S8,
Fig. 6], the water flow as logged on PARSEX [S1, Fig. 2], and the setpoint. No
composition: an online analyser on such a reactor was not found, and Bayer
sampled for the lab once a day [S3]. Pressure, level and circulation flow
[S3, Fig. 11] are not modelled (D2).

---

## 5. Episode and scenarios (rules 17 and 20)

The reactor starts on the steady state that holds 58 C for the day's feed
temperature, with the water where that needs it. `scenarios.py`:

| Scenario | Event | Source |
|---|---|---|
| `normal` | the feed drifts with the room, 23 to 25 C over a day | [S1, Tables 3, 4] |
| `setpoint_step` | setpoint +2 K | [S1, Fig. 2] |
| `catalyst_loss` | activity -40 % over 90 min (rate ours) | [S1, Fig. 3; p. 6] |
| `feed_buildup` | feed reactant +10 % (ours) | [S3, PDF pp. 36-37, 46] |
| `cooling_loss` | no water for 5 min (ours) | [S2, PDF p. 17]; [S3, Fig. 11] |

Each event comes once, at a minute drawn in 30 to 150.

---

## 6. Score (rule 15)

The job [S5; S6]: hold the reactor temperature at its setpoint, and never run
away. Tracking is the real job here, so it is scored:

```
reward = -((T_sp - T) / 0.1 K)^2,   and 960 x 2.45e5 on an interlock trip
```

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 0.1 K | ours: a reactor probe's resolution, the documented minimum the score resolves |
| `e_tol` | 0 | no band found |
| `tracking_exponent` | 2 | ours |
| `failure_cost` | 2.45e5 | ours: twice the tracking cost of 35 K, past any untripped state (a reactor cooled to its feed) |
| `restart_steps` | 960 | ours: 4 h of lost production, seven residence times; no restart time found |
| `rho_floor`, `rho_floor_tracking` | 0 | not measured on this plant |
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
| T8. Open-loop settling | 95 % in 1.6 to 6.4 residence times [S1, Fig. 2] | **fail: 0.99** |
| Runaway trips the interlock | [S3] | pass |
| Starts at rest on its setpoint | rule 17 | pass |

**Finding, T8.** Open loop, a cooling step settles in 0.99 residence times (0.97 on v5),
faster than the 1.6 the audit derived from PARSEX's closed loop (3.2, halved).
The audit called it weak, kept for scale; the reactor's own temperature time
constant is not documented (REALITY.md, "Time scales"). Not moved.

---

## 8. Known deviations (rule 13)

**D1. A lumped jacket.** PARSEX cools through an external recirculating
cooler [S1]; here one well-mixed volume with no transport delay. The 2 min
dead time of [S6] is on the jacket reading, not the heat path.

**D2. No pressure, level or circulation flow.** At Bayer the first alarm acted
on was high pressure [S3, Fig. 11]. Effect: the runaway is seen on temperature
alone.

**D3. One handle.** PARSEX also moves feed concentration [S1, p. 4]; HSE lists
inhibitor, quench and dump [S5]. Effect: a narrower job.

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

`experts.py`: a PI from the reactor temperature to the cooling water, biased
at the water the setpoint needs at the design feed. Gains by SIMC (tau_c =
theta) on one bump test of the water (+0.2 raw) from the design steady state,
noise and events off: k = -10.7 K per raw unit, tau 18.4 min, theta 8.9 min,
so Kp = 0.097 per K and Ti = 18.4 min (`src/target_gym/data/pid_gains.json`).
The same gains stand at noise levels 0.5 and 2, since the test is noise-free,
which the noise-level study does not accept (`test_gain_levels.py`, two
failures for the ladder).

## 10. What the reference ladder row needs

- Tracking of the reactor temperature, no zone; scenarios as section 5.
- Bump tests on the water from the reset steady state with the event off.
- PI by SIMC on T, then feedforward: none measured (feed temperature and
  activity are hidden); then the cascade on the jacket outlet (obs 1, 2 min
  late); the expert above is the PI rung.

## 11. Oracle needs

`oracle.py` plans on the reactor and jacket balances, with the interlock as a
limit and the scenario's event held from the state. It does not see a later
event. Not run.

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
| `cstr` | 960 | 15 | 4.0 h | 1 in [-1, 1] | 4 | 33 |

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
