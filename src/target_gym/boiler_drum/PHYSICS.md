# Boiler drum — physics model, provenance and validation

Reference plant: a **natural-circulation drum boiler**, 160 MW, 85 bar, with
the geometry of the Öresundsverket P16-G16 unit used by Åström & Bell.

Contract for `target_gym.boiler_drum`. Method: `docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 1. Model and provenance

Global mass and energy balances over the whole water/steam circuit, plus two
states saying **where** the steam is:

```
q_s = q_demand · p / p_nominal                      (fixed turbine valve)
d/dt[ρ_s V_st + ρ_w V_wt]                          = q_f − q_s
d/dt[ρ_s h_s V_st + ρ_w h_w V_wt − p V_t + m C t_s] = Q + q_f h_f − q_s h_s
dm_sr/dt = q_ct + q_flash,r − m_sr/τ_sr        (steam in the risers)
dm_sd/dt = f_carry·m_sr/τ_sr + q_flash,d − m_sd/T_d − q_cd   (bubbles in the drum)
```

with `V_st + V_wt = V_t`. The first two are linear in `(dp/dt, dV_wt/dt)` and
are solved as a 2×2 system each step. Level follows from where the water is:

```
level = [ (V_wt − V_dc − (1 − α_v)·V_r) + m_sd/ρ_s − V_ref ] / A_d
```

**Tracking steam as mass rather than quality is what makes the model work.**
The void fraction is `α_v = m_sr/(ρ_s V_r)`, so a falling pressure raises it
*instantly* at constant mass — bubbles expand, water is pushed out of the
risers into the drum, and the level rises. A formulation that relaxes riser
*quality* toward a quasi-steady value on the 18 s riser time constant
suppresses exactly this effect, and produces no swell at all.

**Flashing** is the other half. A falling pressure lowers the saturation
temperature, leaving the entire water inventory superheated so that it boils:
`q_flash = −(V ρ_w c_pw · dt_s/dp / h_c)·dp/dt`. For this boiler a 1 bar drop
flashes about 143 kg of water — 3.1 m³ of steam, roughly 16 cm of level.

**Steam follows pressure.** At a fixed valve the turbine passes steam in
proportion to drum pressure, Bell and Åström's `q_s = (1.1 u2 − 0.19) P` (1987,
eq. 14). The hidden valve state `q_demand` wanders, and the meter reads `q_s`. So
pressure regulates itself: a 1 bar offset at fixed firing decays with a 227 s
time constant, against about 400 s from the report's own `dP/dt` (eq. 3, `u2`
0.7).

Saturated steam properties are quadratic fits to IAPWS values over 60–110 bar
(residuals below 0.5 %, asserted in the tests). Fits rather than steam tables
because the model must stay `jit`/`vmap`-friendly.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Superheater and steam temperature | The drum is the control problem; steam temperature is a separate downstream loop. |
| Furnace-side combustion detail | One lag and one rate limit stand for the burners, the furnace gas and the tube walls (see *Actuators*). |
| Spatial distribution along the risers | One lumped riser; the void fraction is a volume mean. |
| Drum-level-gauge dynamics | Treated as instantaneous. |
| Steam separator detail | Represented only by `f_carry`, the fraction of riser steam passing below the water level. |
| Drum geometry | Level is linear in volume (`A_d` constant), valid for deviations around normal water level. |

**Regime of validity.** 60–110 bar (the property-fit range), near-normal water
level, loads between roughly 50 % and 130 % of nominal. Termination on drum
pressure keeps episodes inside the fit range.

---

## 2. Validation targets

| Quantity | Target | Model | |
|---|---|---|---|
| Steam property fits vs IAPWS, 60–110 bar | < 1 % | **max 0.49 %** | ✅ |
| Saturation temperature at 85 bar | 299.3 °C | 299.27 | ✅ |
| ρ_water, ρ_steam at 85 bar | 711.5, 45.3 kg/m³ | 712.0, 45.5 | ✅ |
| Latent heat at 85 bar | 1410 kJ/kg | 1410.1 | ✅ |
| Circulation ratio | 5–15 (natural circulation) | **8.84** | ✅ |
| Circulation ratio falls with load | required | 10.2 → 8.1 | ✅ |
| Mean riser void fraction | 0.3–0.6 | 0.467 | ✅ |
| Drum volume closure | 40 m³ | **40.00** | ✅ |
| Metal share of pressure inertia | dominant | **58.5 %** | ✅ |
| Swell on a valve step | 25–100 mm typical | 21 mm (5 %) → 89 mm (21 %) | ✅ |
| Shrink on a feedwater step | present, correct sign | −2.4 mm | ⚠️ |
| Level self-regulates | it must **not** | +8.9 cm per 10 min at 2 % bias | ✅ |
| Pressure self-regulates at a fixed valve | about 400 s (report, eq. 3) | 227 s | ✅ |

The drum volume closure is the load-bearing check, because nothing in the
model is fitted to it: the water, bubbles and steam space computed from the
solved inventory add to 40.00 m³ against a drum specified independently as
40 m³. Circulation ratio is what sets `k_friction`, the one free parameter in
the circulation loop.

---

## 3. Parameter table

| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `V_t`, `V_d`, `V_r`, `V_dc` | 88, 40, 37, 11 | m³ | Åström & Bell P16-G16 geometry | ✅ |
| `A_d` | 20 | m² | Drum area at normal water level | ✅ |
| `A_dc`, `L_r` | 0.355 m², 11 m | – | Downcomer area, riser height | ✅ |
| `m_metal`, `C_metal` | 300 t, 550 J/(kg K) | – | Tube and drum steel | ✅ |
| property fit coefficients | see params | – | Least squares on IAPWS saturation values | ✅ |
| `k_friction` | 25 | – | Set by the circulation ratio (validated 5–15) | ⚠️ |
| `tau_sr` | 8.0 | s | Riser steam transit; equals `m_sr/q_steam` at nominal | ⚠️ |
| `T_d` | 15.0 | s | Bubble residence below the water level | ⚠️ |
| `f_carry` | 0.10 | – | TUNED — riser steam passing below the level | ⚠️ |
| `f_cd` | 0.25 | – | TUNED — feedwater heat deficit paid by drum bubbles | ⚠️ |
| `h_feedwater` | 1.037 MJ/kg | – | Feedwater at 240 °C (59 K subcooled) | ✅ |
| `Q_nominal`, `q_steam_nominal` | 160 MW, 93.35 kg/s | – | `Q = q_s(h_s − h_f)`, self-consistent | ✅ |
| `level_trip` | ±0.25 | m | Carryover / dryout; real trips are ±150–300 mm | ✅ |
| `delta_t` | 2.0 | s | ~14 steps to the swell peak, ours as a control interval | ⚠️ |

`f_carry` and `f_cd` are the two genuinely tuned numbers. They were chosen
together against three targets at once — baseline bubble volume, swell
magnitude and shrink magnitude — not fitted to any single one.

---

## 4. Task design

### Task: `boiler_drum`

Hold the drum level and the steam pressure of a 160 MW natural-circulation drum
boiler with fuel and feedwater, while the turbine's steam draw moves. Drum
level is the classic non-minimum-phase loop: a pressure drop swells the level
while water is leaving, and cold feedwater shrinks it before it rises. The
level integrates, the riser voidage behind both effects is hidden, and both a
high and a low level trip the boiler.

| obs | meaning |
|---|---|
| 0 | drum level relative to normal water level (m) |
| 1 | drum pressure (bar) |
| 2 | `q_steam`, steam flow to the turbine (kg/s), which follows the valve and drum pressure |
| 3 | fuel actuator position, percent of maximum |
| 4 | feedwater actuator position, percent of maximum |
| 5 | target level, always 0 (m) |
| 6 | target pressure (bar) |

**Why it is hard.** Drum level is the classic non-minimum-phase control
problem, and nothing else in this suite has that shape.

1. **Swell.** Open the turbine valve and pressure falls; every bubble expands
   and the water inventory flashes, so the level goes **up** while mass is
   leaving. A controller that believes the gauge cuts feedwater exactly when
   it should be adding it, and the level then collapses into a low-level trip.
2. **Shrink.** Subcooled feedwater collapses bubbles, so adding water makes the
   level fall before it rises.
3. **Level is an integrator.** There is no self-regulating steady state — a
   2 % feedwater bias walks the level 9 cm in ten minutes.
4. **Both limits are irrecoverable.** High level carries water into the
   turbine, low level uncovers the tubes; real boilers trip on both.
5. **Voidage is unmeasurable.** `m_sr`, `m_sd` and `V_wt` are hidden. No plant
   instrument reads riser void fraction, and it is exactly the state driving
   the inverse response.

**Observation.** Drum level, drum pressure and steam flow are the three
measurements of classic three-element control, plus the positions of the two
actuators. The turbine valve is hidden.

**Two coupled targets.** Firing and feedwater are not independent: firing
harder raises pressure, which *compresses* bubbles and lowers the level.

**Reward** = ½·level band + ½·pressure band − fuel.

---

## 5. Baselines

| controller | return (400 steps) | mean level error | pressure error |
|---|---|---|---|
| MPC (horizon 30 ≈ 60 s) | **318 – 364** | **0.1 – 0.6 cm** | 0.2 – 0.7 bar |
| Three-element PID | 215 – 296 | 2.7 – 7.3 cm | 0.01 – 0.05 bar |
| best constant action | 5.0 | trips in 5 – 117 steps | — |

Throughput is measured in [docs/performance.md](../../docs/performance.md).

**Three-element control is the right PID baseline**, and its structure is the
point: feedwater tracks *measured steam flow* as a feedforward, closing the
mass balance without going through the level gauge, and the level PI only
trims it slowly. That is what makes it immune to the level lying during a
transient. Single-element level control, a level loop on the feedwater alone,
is the weaker controller precisely because it has no such feedforward.

**The expert's tuning: lambda, from single-run bump tests.** The recipe is section
10 of `docs/proposals/expert-protocols-process.md`. On each training seed (100 to
109) the firing is in manual at the seed's heat balance, a P loop holds the level,
since nobody leaves a drum level in manual, and the steam demand is quiet. Each loop
then takes one step:

| loop | step | model | lambda | gains |
|---|---|---|---|---|
| pressure | firing +0.02, 120 steps | k 60 bar per unit action, tau 146 s, theta 22 s | theta | Kc 0.0554 per bar on the action, Ti 146 s |
| level | feed +2 kg/s through the loop's setpoint, 120 steps | k' 7.96e-5 m/s per kg/s, theta 19.6 s and lag 1.3 s, counted as delay | theta | Kp 451 kg/s per m, Ki 7.18 |

Steam now follows pressure at a fixed turbine valve, so pressure regulates itself and
its model is first order. The firing lag is most of its 22 s, which is why the gains
fall from Kc 1.34, the value the old plant gave, to 0.055. The level's trim has no limit of its
own now: the integral stops only at the feedwater's limits, while the error pushes
past them.

The level's RGA, firing and feed on pressure and level, read from the moves over a
test with the loop open, is 1.03 on the diagonal, so the pairing stands. The level
step above is N1's closed test, and its fit carries the shrink in its delay.

On the held-out seeds 200 to 209 it holds 409 a step (worst 792) against a ceiling of
1.14, with no trip, where the old gains held 1624. Level RMS is 48 mm, 34 to 64 by
seed, and p99 77 to 160 mm, against the +-1 inch of check 2, so it is 1.9 times the
figure on RMS and 3 to 6 on p99. Pressure RMS is 0.24 to 0.45 bar. Spec, a strict
xfail in `tests/boiler_drum/test_boiler_drum_expert.py`: mean level error under 25 mm
on every held-out seed. Measured 26 to 52 mm.

**Most of what remains is the fuel rate.** A 20 s firing lag and a 0.7 %/s slew keep
pressure, and with it swell, from answering the steam swings the feed sees at once.
The review's figures, 0.65 bar and a cost of 1342 at 20 s, came from the plant before
steam followed pressure. Its 0.6 trips per episode at 60 s do not carry over: at 60 s
this plant does not trip, because pressure regulates itself.

**The oracle is the shared solver, on the reward itself (`oracle.py`).** Cooper's
recipe gives P = 237 steps (474 s), which covers the ~35 s swell peak several
times over, so the plan sees the level reverse and keeps feeding through a
swell. Each move is an actuator's rate, sent as the command that makes it, so
neither rate limit nor lag hides a move from the plan (`oracles/rates.py`). The
steam follows the planned pressure, with the turbine valve on its mean path. The
reward has been squared errors in floor units since version 2, so no surrogate
is needed; the fuel's kink at `c_hold` is an epigraph row, and the level and
pressure trips are hard limits, backed off by what one steam-demand innovation
can move them (K11). On the ten test seeds it beats the stored gains by 12 to 67
times, with no trip, no failed or capped solve and a replay gap under 1e-7, in
about 23 iterations a solve. What cost remains sits where the steam demand swings,
since the plan assumes a swing decays on its mean path.

---

## 6. Known deviations

**⚠️ D1 — feedwater shrink is weak.** A 10 kg/s (11 %) feedwater step produces
about 2.4 mm of shrink, at the low end of what is reported for real boilers.
The magnitude is bounded by the feedwater subcooling, which is already at a
realistic 59 K, and the two channels through which it acts (suppressed riser
boiling, collapsed drum bubbles) largely cancel — as conservation requires,
since the heat deficit is fixed however it is split. The **load-change** swell,
which is the effect that actually destabilises level control, is at documented
magnitude.

**⚠️ D2 — no combustion dynamics.** Heat release follows the fuel command
instantly. A real boiler has burner and mill lags of seconds to minutes, which
make pressure control materially harder than modelled here.

**⚠️ D3 — lumped riser.** One volume-mean void fraction stands in for a
distribution that varies along the riser height, and `τ_sr` is constant rather
than varying with circulation.

**⚠️ D4 — `f_carry` and `f_cd` are tuned.** They are not measurable plant
quantities. They were set against three behavioural targets jointly, but they
are the least defensible numbers in the table.

**⚠️ D5 — level is linear in volume.** `A_d` is constant, which is right for
deviations around normal water level and wrong for large excursions in a
cylindrical drum. Trips at ±25 cm keep the model inside the linear region.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `boiler_drum` | 400 | 2 | 13 min | 2 in [-1, 1] | 7 | 14 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2)

`compute_reward = -(tracking + running + failure)`, one tracking term for the level and one for the pressure, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor_level` | 2.67e-3 m | an earlier oracle's lowest per-seed long-run mean \|level\| under the shipped steam-demand disturbance (seeds 2.7 / 17 / 18 mm), measured while it saw the steam noise of that seed. The current oracle, planning on the mean demand, holds 3.4 / 2.5 / 2.7 mm (`scripts/measure_hold.py`, 1200 hold steps after a 60-step burn-in; PID 37-70 mm). Its best seed, 2.45 mm, is within 1.1x of the floor, which is kept as the scale. Upper bound |
| `e_floor_pressure` | 0.05 bar (the MPC holds 0.052) | the pressure transmitter's resolution. The oracle holds 0.066 / 0.052 / 0.053 bar per seed (PID 0.035-0.050), just above it; an earlier oracle's 0.028 bar came from seeing the steam noise. The hold is within 1.5x of the resolution, so the resolution stays the scale (the PID's best seed, 0.035 bar, sits below it, and a hold the instrument cannot see is not a floor). |
| `e_tol` | 0 | no specification band |
| `tracking_exponent` | 2 | quadratic on both |
| `c_hold` | 1.566e8 W | firing rate while holding, MPC (PID 1.568e8) (`scripts/measure_hold.py`) |
| `running_weight` | 1 | sweep 0.5 / 1 / 2 |
| `failure_cost` | 4.4e5 | twice the level-trip plus the reachable pressure excursion's cost, 2 x ((0.25 / 2.67e-3)^2 + (23 / 0.05)^2): the 88 bar target to the 65 bar trip, per trip step |
| `restart_steps` | 7200 (4 h) | restart time priced into a trip, `restart_steps x failure_cost` (a drum trip's restart; provisional); where a plant engineer would get it: the plant's restart procedure |

`rho_floor_tracking` is the oracle's best hold for tracking -- the lowest per-seed
hold cost the reference controller demonstrated, in the reward's units, which
is 1 per term where the floor is that hold and less where the floor is clamped
at the instrument resolution. On this plant neither holds exactly: the level
floor (2.67 mm) is an earlier oracle's hold, so the current oracle's 2.45 mm
costs 0.84, and its 0.0519 bar pressure hold sits just above the 0.05 bar
resolution kept as the scale, so that term costs 1.08 and `rho* = 1.92` -- `rho_floor` the same with consumption
charged in full; `floor_is_documented_minimum` records whether `e_floor` is a
measured/certified floor or a resolution used as a scale, and where it is a
resolution on a deterministic plant both references are 0, since exact hold is
achievable there and the resolution only sets the unit;
`failure_cost` is the per-step cost of a tripped plant, above the largest tracking
cost the envelope can produce, and `restart_steps` the time a restart would take,
so a trip costs `restart_steps x failure_cost` (`reward.trip_cost`). A trip never
ends the window (`base.failure_kernel`): the step that leaves the envelope is
charged the trip cost, with tracking and running cost zeroed, and the plant
restarts at once as `reset_env` would, on the same clock. `terminated` stays
false unless `continuing = False`, which ends the episode on that step instead; `info["tripped"]` marks the event for the evaluator. `reward_version = 1` reconstructs the
capped log-scaled reward of the previous version (`precision_floor` and the
old weights are read only by it).

## Actuators

Fuel and feedwater reach the plant through the shared actuator stage (`actuate`),
which is the block diagram of Bell and Åström's Fig. 1: a first-order lag, a rate
limit, and the range. A reset starts both actuators where the plant is
(`initial_action`).

| figure | value | status | source |
| --- | --- | --- | --- |
| fuel rate | 0.007 of span per s, 143 s full stroke | Read, checked on the page and in the listing | [Bell and Åström 1987](https://lup.lub.lu.se/record/8602296), section 4, and appendix E (`UDH:0.007`) |
| feedwater rate | 0.05 of span per s, 20 s full stroke | Read, checked the same way | same, section 4 and appendix E (`UDH:0.05`) |
| feedwater lag | 1 s | Read in the listing | same, appendix E (`tc1:1`), the `Tc` of Fig. 1 |
| firing lag | 20 s | **Ours, for the owner to accept** | see below |
| control interval | 2 s | **Ours** | the unit's logs were sampled at 0.1 Hz ([Åström and Bell 2000](http://www.dei.unipd.it/~picci/Files/TEACHING/CONTROLLO_DEI_PROCESSI/DrumBoiler.pdf)), which says nothing of a controller's scan |
| dead time of the three readings | none | **Ours** | transmitters with no modelled damping |

These are the actuators of the plant this model takes its geometry from, a 160 MW
oil-fired unit at Öresundsverket. Its fuel span, 0 to 14 kg/s, is about 1.5 times
its nominal firing, as `Q_max` is here. Its feedwater span is 0 to 140 kg/s against
200 kg/s here, so the same stroke moves about twice the share of nominal flow.

**The firing lag is not sourced.** The report's model has no heat release lag, and
`Tc` for fuel is 1 s, the actuator alone. On a plant log sampled every 10 s that
leaves little room for a long lag on oil. A coal unit has one: General Signal's
[US 4909037](https://patents.google.com/patent/US4909037) puts a mill at 100 to
160 s before a change shows at the furnace, as a dead time and a lag. 20 s stands
for burners, furnace gas and tube walls on oil or gas, well under a mill and well
over the valve. No figure for it was found; the IEEE working group's papers
(de Mello 1991) would be the place to look, and they are not open. It is the
lower of the two lags the industrial review tried.

## Readings

The observation reports drum level, drum pressure and steam flow through their
instruments (`docs/proposals/measurement-noise.md`). `fuel_pct` and `feed_pct` are
the actuators' positions, as a plant's drives report them, and they stay exact
with both targets.
`sensor_noise_scale = 0` gives the noise-free task.

| reading | sigma | step | instrument | status | source |
| --- | --- | --- | --- | --- | --- |
| `level` | 0.5 mm | | Rosemount 5300 guided-wave radar | Confirmed | [5300](https://www.emerson.com/documents/automation/73580.pdf) |
| `pressure` | 0.022 bar | | Yokogawa EJA530E pressure transmitter | Confirmed | [EJA530E](https://web-material3.yokogawa.com/GS01C31F01-01EN.pdf) |
| `q_steam` | 0.047 kg/s | | Rosemount 8800D vortex class | Not checked | [8800D](https://core.instrumart.com/assets/Rosemount-8800D-Series-Vortex-Flow-Meter-Datasheet.pdf) |

`level` excludes drum surface turbulence, so on a real drum it is a lower bound.
`pressure` is an upper bound: 0.055 % of an assumed 0 to 120 bar span at a stated
3 sigma, a conformance figure that includes linearity and hysteresis.

**Weak figure.** The refutation pass did not reach `q_steam`. It reads the sheet's
0.1 % repeatability as 2 sigma on 93.35 kg/s, and a main this large may use flow
nozzles instead.

The expert reads these values, and so does every tuner. The oracle plans and acts
on the true state, so it stays a full-state bound. `reward`, trips and the score
read the true state. `info["measured_reward"]` is the same cost on the readings,
with any trip taken from the true state, so resting on target costs a learner
`(0.022 / 0.05)^2 = 0.19` per step in pressure and `(0.5 / 2.67)^2 = 0.035` in
level.
