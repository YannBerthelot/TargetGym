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
q_s = q_d · p / p_nominal                           (fixed turbine valve)
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
eq. 14). The hidden valve state `q_d` wanders, and the meter reads `q_s`. So
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

**The expert's tuning: lambda, from step tests on the training seeds.** Both
loops are lambda-tuned for an integrating process, as drum level is in
practice (`scripts/tune_industrial_experts.py`; the table above predates it).
On each training seed (100 to 109), from the steady state the shipped
controller reaches there, one loop is put in manual at its average output
and stepped, against an unstepped run on the same noise:

| loop | step | fitted `k' e^(-theta s) / s` | lambda | gains |
|---|---|---|---|---|
| level (pressure in auto) | feed trim +2 kg/s | 7.41e-5 m/s per kg/s, theta 13.5 s | theta | Kp 749 kg/s per m, Ki 18.5 |
| pressure (level in auto) | firing +0.02 | 0.359 bar/s per unit action, theta 1.55 s | theta | Kp 0.672, Ki 0.144 per bar (of Q_max) |

The level's inverse response, the shrink before the rise, is in its fitted
delay. Theta includes half a sample. `Kc = (2 lam + theta) / (k' (lam +
theta)^2)` and `Ti = 2 lam + theta`. Lambda is theta, 2 theta or 3 theta,
the looser only if it is more than 1 % cheaper over the training seeds. The
tight choice won on both loops. The hand-tuned gains it replaces were
Kp 60, Ki 0.6 on level and Kp 0.05, Ki 0.002 on pressure.

**The bad seed.** Under the hand tuning, protocol seed 1 cost 1058 a step
against 322 and 192 on seeds 0 and 2. Level carried almost all of it: a mean
level error of 28 to 79 mm against the 2.7 mm floor, with pressure within
0.03 to 0.05 bar. Seed 1's steam demand swings the most (73 to 98 kg/s), and
the slow level trim let the level wander with it. On the held-out seeds 200
to 209 the lambda-tuned expert holds 142 a step (median 140, worst 222), the
hand tuning 635 (median 580, worst 1383), against a ceiling of 1.14: 2.10
decades of room, and no seed dominates.

Spec, a strict xfail in `tests/boiler_drum/test_boiler_drum_expert.py`. Mean
level error under 25 mm (about an inch) on every held-out seed. Measured
19 to 34 mm, 6 of 10 seeds above.

**These gains predate the actuator limits,** and PR 9 of the industrial fixes
retunes them. On the limited plant, held-out seeds 200 to 209, they trip nowhere
and hold 75 mm and 1.03 bar, at 1624 a step. With no lag or rate limit, steam
following pressure alone, the same gains hold 25 mm and 0.032 bar at 122 to 404.
Most of the loss is the fuel rate: 20 s of lag alone holds 0.59 bar at 913 to
1259. The review's figures, 0.65 bar and a cost of 1342 at 20 s, came from the
plant before steam followed pressure. Its 0.6 trips per episode at 60 s do not
carry over: at 60 s this plant does not trip, because pressure regulates itself.

**MPC wins on horizon, not on tuning.** A 30-step horizon covers the ~35 s
swell peak, so the optimiser sees the level reverse and keeps feeding through
a swell. It buys a 10× reduction in level error for a small pressure trade.

Its objective is a **quadratic, not the environment's reward**. The reward's
tracking bands clip flat beyond `level_band`, which is exactly the situation
the controller is called on to fix; optimising it directly leaves no gradient
where one is most needed. The objective shares the reward's *minimiser*, not
its shape — the same lesson recorded for the pH environment.

**The MPC plans on the mean steam demand, with a large descent budget.** Its
model is the simulator itself, rolled with a fixed key. Until the oracle
audit (2026-10-01) the steam-demand noise was not declared in the spec's
`noise_fields`, so the plan followed one fixed noise path: exactly the
plant's on protocol seed 0 and a wrong one on every other seed. Declared, the
plan follows the mean. The descent then needed far more iterations: on the
protocol seeds, 40 iterations scored 19.2, 160 with a decaying step 8.2, 640
5.3 and the shipped 1280 4.67, against 23.8 before (zero trips). Each doubling
still gains about half what the one before it did, so the optimiser, not the
plant, is what limits it. (The table above predates the version-2 reward and
this change.)

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
