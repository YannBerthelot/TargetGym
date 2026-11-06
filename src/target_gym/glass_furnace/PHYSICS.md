# Glass furnace — physics model, provenance and validation

Reference plant: a **regenerative end-port fired float-glass furnace**,
~500 t/day, natural gas.

This document is the contract for `target_gym.glass_furnace`. Every constant is
derived here, cited, or explicitly flagged `TUNED — not sourced`, and each row
of §2 is asserted by a test in `tests/glass_furnace/test_glass_furnace_env.py`.
Method: `docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 1. Model scope

Eleven ODE states, a thermocouple lag and a two-step fuel pipeline, plus one
algebraic variable:

| State | Meaning | Timescale |
|---|---|---|
| `T_crown` | crown refractory — **the measured, controlled variable** | ~15 min |
| `T_melt` | glass in the melting zone (hidden) | ~30 h |
| `T_work` | glass in the working end (hidden) | ~30 h |
| `T_rA[4]`, `T_rB[4]` | regenerator checker nodes, hot end first (hidden) | ~1–2 h |
| `m_batch` | unmelted batch blanket on the melt (hidden) | ~1 h |
| `T_crown_meas` | crown thermocouple reading — **what the observation reports** | 120 s |
| `fuel_pipeline[2]` | fuel commanded but not yet burning (hidden) | 60 s |
| `lhv_deviation` | the gas's heating value off `LHV` (hidden) | 1 day |
| `T_gas` *(algebraic)* | flame / combustion-space gas | ~0.5 s |

**`T_gas` is solved to steady state, not integrated.** Its radiative time
constant `C_gas/(4εσAT³)` is ~0.5 s, sixty times shorter than the 30 s control
step; integrating it explicitly is unstable (eigenvalue ~2 s⁻¹, far outside
RK4's stability region — it produces NaN) and buys nothing. Solving its balance
algebraically each step is the standard singular-perturbation reduction: the
flame stays a distinct, hotter radiating node without adding a stiff state.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Electric boost, bubblers | Common but optional equipment; adds actuators without changing the core control problem. |
| Combustion chemistry / NOx | Fuel is converted at fixed LHV with 10 % excess air. Flue-gas composition is not a controlled variable here. |
| Glass chemistry, seeds, homogeneity | Quality is proxied by temperature; modelling redox and fining is a research problem in itself. |
| Spatial (CFD) resolution | Zonal lumped model. A CFD furnace cannot run at millions of steps/s. |
| Per-port firing distribution | Single lumped firing rate. |

**Regime of validity.** Calibrated for continuous operation near the nominal
crown temperature (1565–1610 °C) at steady pull. It is *not* valid for
cold-start / heat-up, idling at reduced pull, or campaign-end conditions.

---

## 2. Validation targets

Published typical values for a regenerative float furnace, each asserted by a
test at the nominal operating point (`fuel_raw = -0.2`).

| Quantity | Target | Model | |
|---|---|---|---|
| Specific energy consumption | 4–6 GJ/tonne | 5.2 | ✅ |
| Crown temperature | 1550–1620 °C | 1603 | ✅ |
| Regenerator temperature effectiveness | 0.75–0.85 | 0.70–0.79 | ⚠️ |
| Preheated air temperature | 1200–1400 °C | 1148–1389 | ⚠️ |
| Stack temperature after regenerator | 400–600 °C | 430–624 | ⚠️ |
| Glass residence time | 24–30 h | 29.9 | ✅ |
| Batch-to-glass yield | ~0.83 | 0.83 | ✅ |
| Working end cooler than melt | required | yes | ✅ |
| Flame hotter than crown hotter than glass | required | yes | ✅ |

**Specific energy is the headline check.** It is what the regenerator exists to
achieve: without heat recovery the same crown temperature costs roughly twice
the fuel, which is how we know the recovery loop is wired correctly and not
merely plausible-looking.

---

## 3. Parameter table

### Combustion
| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `LHV` | 50.0e6 | J/kg | Natural gas lower heating value | ✅ |
| `AFR` | 17.0 | – | Stoichiometric air/fuel mass ratio for methane | ✅ |
| `excess_air` | 0.10 | – | 10 %, typical for glass furnaces | ✅ |
| `c_p_air` / `c_p_gas` | 1150 / 1200 | J/(kg·K) | Hot air / flue gas | ✅ |
| `fuel_min` / `fuel_max` | 0.513 / 0.698 | kg/s | Sized from the measured 1861 °C·s/kg gain so the action range spans the 1427–1677 °C operating band plus failure margin, then raised 2.6 % so the burners deliver the same time-averaged heat despite the reversal interruption | ✅ |

### Regenerators
| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `N_REGEN_NODES` | 4 | – | Per chamber; enough to develop a hot-to-cold gradient | ⚠️ |
| `C_regen_node` | 3.0e7 | J/K | Thermally active checker layer | ⚠️ |
| `eps_regen_node` | 0.80 | – | Per-node effectiveness; TUNED to hit air-preheat and stack targets jointly | ⚠️ |
| `reversal_period` | 1500 | s | Timing table, *Actuators and timing* | ✅ |
| `reversal_dead_time` | 40 | s | Timing table | ✅ |
| `reversal_firing_floor` | 0.05 | – | Pilots stay lit through the changeover | ⚠️ |
| `tau_thermocouple` | 120 | s | Timing table | ⚠️ |
| `FUEL_DEAD_TIME_STEPS` | 2 | steps | Timing table | ⚠️ |
| `charge_period` / `charge_duty` | 300 / 0.30 | s / – | Reciprocating batch charger cycle | ⚠️ |
| `charge_mass_jitter` | 0.15 | – | Dose-to-dose mass variation | ⚠️ |

### Glass and batch
| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `m_pull` | 5.79 | kg/s | 500 t/day float line | ✅ |
| `C_melt`, `C_work` | 7.8e8 | J/K | ~625 t glass; gives the 30 h residence time | ✅ |
| `c_p_glass_a/b` | 900 / 0.35 | J/(kg·K) | Soda-lime `c_p(T) = 900 + 0.35·T[°C]`: 1040 at 400 °C, 1425 at 1500 °C | ✅ |
| `dH_fusion` | 0.8e6 | J/kg | Latent heat + endothermic batch reactions | ✅ |
| `batch_yield` | 0.83 | – | CO₂ / volatiles loss on melting | ✅ |
| `m_batch_full` | 39 000 | kg | 0.15 m × 200 m² × 1300 kg/m³ blanket | ⚠️ |
| `batch_shield` | 0.85 | – | TUNED — radiation blocked at full coverage | ⚠️ |

### Heat transfer
| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `C_crown` | 2.5e8 | J/K | ~250 t silica crown | ✅ |
| `C_gas` | 2.4e5 | J/K | ~1000 m³ flue gas at 1600 °C (documentary only — `T_gas` is algebraic) | ✅ |
| `eps_rad` | 0.8 | – | Effective flame + crown emissivity | ⚠️ |
| `A_crown` / `A_melt` / `A_work` | 200 / 200 / 10 | m² | Working end largely shielded beyond the throat | ⚠️ |
| `h_conv` | 30 | W/(m²·K) | TUNED — forced convection in the combustion space | ⚠️ |
| `UA_work_cooling` | 3000 | W/K | Conditioning-zone heat extraction: a float working end cools glass from ~1500 °C to the ~1150 °C the tin bath needs, i.e. several MW | ✅ |

---

## 4. Task design

### Task: `glass_furnace`

Hold the crown temperature of a regenerative float-glass furnace on a slowly
walking setpoint by setting the fuel flow. The crown is read through a lagging
thermocouple, the glass and regenerator temperatures are hidden, the firing
side reverses on a fixed cycle, and the furnace responds over hours, so a few
instruments have to steer a large, slow plant.

| obs | meaning |
|---|---|
| 0 | crown temperature as the thermocouple reads it (°C); it lags the true crown temperature the reward scores |
| 1 | combustion air preheat temperature (°C) |
| 2 | fuel flow, percent of maximum |
| 3 | regenerator phase: 0 while regenerator A preheats the air, 1 while B does |
| 4 | target crown temperature (°C) |
| 5 | pull, the glass leaving the furnace (kg/s) |

**Setpoint band 1565–1610 °C.** Operationally realistic — a float furnace is
trimmed ±10–20 °C around nominal. The previous 1500–1650 °C band was
untrackable: the settling time is 140–300 h (measured), so a 150 °C step means
re-heating the whole glass inventory. Worse, it made the *sampling* decide the
score rather than the controller — with five setpoints drawn from a 150 °C
range, a seed whose draws clustered scored 4413 with 6 °C mean error while one
with a 146 °C swing scored 734 with 37 °C. Narrowing the band cut PID return
variance from ±1067 to ±286 and mean tracking error from 26.6 °C to 2.5 °C.

**Setpoint schedule.** Five slots, and the schedule is a *trim walk*: the first
slot is the level the furnace is already being held at, and each later slot is
the previous one plus at most ±6 °C, clipped to the band. Independent draws
from the band, which is what this did before, span about 30 °C on average and
can step 40 °C between consecutive slots — the opposite of the ±10–20 °C trim
the band was chosen to represent. The walk gives an 8 °C median span across an
episode with single trims of 2.8 °C median, 6 °C worst. Reset places the crown
within ±4 °C of the first slot for the same reason: a running furnace is found
at its setpoint, not 18 °C off it.

**Reward.** The floor-normalised cost described in *Reward (version 2)*
below.

**Instrumentation and dead time.** The crown reading in the observation is
``T_crown_meas``, a first-order lag of ``tau_thermocouple`` = 120 s on the true
crown temperature: the element sits in a refractory sheath and reports a
filtered version of what it is in. Fuel carries a pure transport delay of
``FUEL_DEAD_TIME_STEPS`` = 2 steps, 60 s, covering the gas train, the
combustion-air adjustment behind it and flame development. The reward scores
the true crown temperature; the instrument is the controller's problem.

Neither existed before, and their absence was not a detail. Inertia is not dead
time: a first-order plant with no transport delay has no bandwidth limit, so a
PID can be tuned arbitrarily tight against it. One duly held the crown to
0.078 K mean against a 10 K open-loop drift, roughly thirty times better than a
real furnace is held, and the task had no headroom left for anything to beat.
With the lag in place the same PID holds 0.46-0.83 K, and dead time is the
specific thing a predictive controller handles better by planning through it.

**Batch charging is discrete.** The doser fires for ``charge_duty`` = 30 % of
each ``charge_period`` = 300 s, with the dose mass varying +/-15 % cycle to
cycle, and the rate is averaged over the step so the mass is conserved exactly
whatever ``delta_t`` is. It used to be a continuous stream tied to the pull
rate, with the only load variation an AR(1) drift on pull at a 50 min
correlation time -- slower than the plant, and a slow smooth load is precisely
what integral action cancels perfectly. Pulsed charging puts content at the
charger frequency instead, which the loop has to reject rather than integrate
away. Open-loop crown swing went from 10 K to 23 K.

**Reset** starts on the furnace's settled orbit. A furnace that takes days to
equilibrate would otherwise spend the whole episode in an initial transient with
nothing to control, and it did: the reset's offsets dated from before the firing
interruption and the pulsed charger, so with the crown held the melt fell
2.4 K/h over the first reversal cycle (the review measured -0.9 K/h, over a
longer window) and the fuel climbed 10 % over the episode. The air preheat
reading started 240 K too hot.

`settled_point` reads the orbit from two rows, each a constant firing (raw -0.1
and +0.1) run for 2000 h with the disturbances off and read where an episode
starts, just after a full reversal cycle. Over the reset's crown range it
interpolates them, and the middle agrees with a run at raw 0 within 0.03 K and
1 kg of batch. Each regenerator gets its own profile, since the chamber about to preheat the air is
the hot one. Rerun the rows if the plant's balance changes, or the reset silently
stops being settled; `test_reset_sits_on_the_steady_state_of_its_firing` fails
when that happens.

**Pull is measured, gas is not.** A float line sets its ribbon speed and width,
so it knows its pull, and the observation reports it, exact. The disturbance a
crown loop cannot see is the gas: a burner with no calorimeter turns a kilogram
into more or less heat as the supply changes. That is a gain on the firing, and
the air stays ratioed to the mass. A furnace is found running on the gas of the
day, so reset's fuel flow burns the orbit's heat on it. `m_pull_noise_std` stays
0.4 kg/s, 7 % of the pull. **Weak figure:** that is more than a line that is
"steady and known" should wander, and no source for a line's real scatter was
found, so it is left for the owner.

**Observation.** A real furnace has crown and air-preheat thermocouples, the
operator knows the reversal state, and the line knows its pull. Glass
temperatures, checker profile, batch mass and the gas's heating value are hidden.

---

## 5. Known deviations

**⚠️ D1 — regenerator effectiveness sits at the low end.** 0.70–0.79 against a
published 0.75–0.85, with air preheat correspondingly at the bottom of its band
(1148 °C at nominal, target 1200–1400) and stack at the top (624 °C, target
400–600). A four-node lumped stack alternating between the two streams cannot
fully reproduce a continuous counter-flow gradient: at cyclic steady state each
node settles toward the flow-weighted mean of the two stream temperatures it
sees, which partly flattens the profile. More nodes would narrow the gap at
proportional cost. Consequence: specific energy sits at the upper end of its
realistic band rather than the middle.

**⚠️ D2 — reversal is symmetric and lossless.** Real reversal briefly
interrupts firing and causes a measurable crown temperature dip every 25 min.
Here the changeover is instantaneous, so the disturbance it injects is milder
than reality. This is now the largest remaining gap for *control*: it is a
periodic upset at a known frequency, and the phase is observable, so it is
exactly the disturbance a predictive controller can feed forward and a PID
cannot.

**⚠️ D4 — cullet ratio is fixed.** Batch is a mix of raw materials and recycled
glass, and cullet melts with roughly 2.5 % less energy per 10 % of the charge.
Real plants see it move by tens of percent as supply changes, usually without
measuring it well. That is a *gain* disturbance -- it changes how much crown
temperature a kilogram of gas buys -- and gain variation is the thing integral
action does not fix, unlike the additive load disturbances this model does
carry. Omitting it makes the plant more linear than the real one.

**⚠️ D3 — single lumped firing rate.** An end-port furnace fires through
alternating ports with a spatially varying heat release; this model has one
well-stirred combustion space, so it cannot represent port-to-port imbalance or
flame-length effects.

---

## 6. Baselines

| controller | mean return (6 seeds) | mean tracking error |
|---|---|---|
| PID | **4907 ± 286** | **2.51 °C** |
| best constant action | 2481 | — |

(The table predates the version-2 reward and the tuning below.)

**The expert: the base layer, SIMC-tuned.** Practice runs an MPC over the
crown loop, and an observation-only MPC is planned (experts refactor, PR 2).
Until it exists the expert is the base layer under it: a PI on the measured
crown temperature (`scripts/tune_industrial_experts.py`). On each training
seed (100 to 109), from the steady state there, the firing is put in manual
at its average over one regenerator cycle (25 min; firing stops at each
reversal, and the PI sits at full demand through it) and stepped by +0.05.
The crown is still rising hours later, on two time scales, so SIMC tunes it
on the integrating approximation of its first hour: k' 0.00276 K/s per unit
action, theta 15 s (half a sample; the fit leaves no delay of its own).
`Kc = 1 / (k' (tau_c + theta))` = 12.1 per K and `tau_I = 4 (tau_c +
theta)` = 120 s, with tau_c = theta (2 and 3 theta were within 1 % on the
training seeds). No derivative. The grid search it replaces (Kp 2.5,
Ki 0.15, Kd 0.05) ran on the scored seeds 0 to 2. Relay autotuning does *not*
work on this plant: the crown temperature integrates the firing rate, so
under a bang-bang relay it drifts without sustained zero-crossings and
Åström–Hägglund has no ultimate gain or period to extract.

On the held-out seeds 200 to 209 it held 0.739 a step, the grid-searched
gains 0.837, against a ceiling of 1, before the slew limit: the base layer
tracked within the thermocouple's resolution, so the task had no room left for
the hold. These gains predate the limit, and PR 9 of the industrial fixes
retunes them. They demand a relay, saturated on 95 % of steps (the review: 94 %),
because 12 per K on a 0.36 K reading spans the action. The review had the fuel
moving 2.3 % of firing a step; the actuator now moves it 0.49 %, its limit
nearly every step, and the same gains hold 1.12 K with no trips.
Spec, a strict xfail in `tests/glass_furnace/test_glass_furnace_pid.py`: the
true crown within 2 K on at least 95 % of hold steps. Measured 88.8 to
97.0 %, 6 of 10 seeds under, with peaks of 4.3 to 6.9 K. The grid-searched
gains score the same (88.8 to 96.9 %, peaks 4.4 to 7.1 K): the misses are
the firing interruptions at each reversal, which a PI on the crown meets only
after the fact.

**The MPC.** Since November 2026 the oracle is `NewtonMPC` (iterative LQR)
on the plant itself: the environment's own `step_env` with the full state,
all four checker nodes per chamber, the fuel pipeline and the reversal cycle,
over a 120-step (one hour) horizon, with the pull planned on its mean path
(`plan_params`). It plans what the plant does, so it needs no disturbance
estimate and no rule for which input reaches the plant first.

It replaced a CasADi model of the plant (oracle audit, 2026-10-03): two
checker nodes per chamber against the plant's four and the flame solved
algebraically. That reduced model ran 0.14 to 0.19 K a step colder than the
plant, so it corrected itself with a crown heat-rate estimate from the
one-step prediction error, and it had to apply the first input its plan could
move, since the first two intervals burn the pipeline. On the protocol seeds
0-2 it scored 0.168 / 0.845 / 0.167 per step, against 0.300 / 1.783 / 0.326
for the controller before it, and over the ten baseline seeds it cost 2638 s
to record, with IPOPT stopped at its iteration cap 4 times. What is left is
mostly authority: on seed 1 a -5.5 K trim meets a low pull, and the fuel
spends much of the window at its minimum with the crown still above target.

The long-run hold that oracle measured, 0.061 / 0.077 / 0.088 K per seed,
against 0.192 / 0.175 / 0.228 K before it, set `rho_floor`:
`rho_floor = rho_floor_tracking = (0.0614 / 1)^2`, from `(0.175 / 1)^2` (see
the reward table below; `e_floor` stays the 1 K thermocouple resolution).

---

## 7. Performance

9 states integrated with `rk4_2` plus a 6-iteration Newton solve for `T_gas`:

| model | steps/s |
|---|---|
| previous 3-state | 20.2 M |
| **this model** | **~3.9 M** |
| plane (for scale) | 4.2 M |
| reactor (for scale) | 1.5 M |

The refinement costs ~5× throughput and lands beside the plane — comfortably
above the reactor, which the library already ships. Buying this much fidelity
was affordable precisely because the furnace started as the cheapest non-trivial
environment in the library.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `glass_furnace` | 1600 | 30 | 13.3 h | 1 in [-1, 1] | 6 | 29 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2)

`compute_reward = -(tracking + running + failure)`, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 1.0 K (the MPC holds 0.0614) | the lowest per-seed long-run mean \|crown error\| the shipped MPC held under the shipped pull disturbance (`scripts/measure_hold.py`, 3600 hold steps after a 10 800-step burn-in; seeds 0.061 / 0.077 / 0.088 K, PID 0.49-0.52 K). The oracle of the oracle audit (2026-10-03); the one before it held 0.192 / 0.175 / 0.228 K. Per-seed minimum; upper bound Below the instrument resolution the plant's own table cites, and measurement noise is not modelled, so the resolution sets the scale: a hold the instrument cannot see is not a floor. |
| `e_tol` | 0 | the crown temperature target has no band |
| `tracking_exponent` | 2 | quadratic |
| `c_hold` | 0.590 kg/s | fuel flow while holding (`scripts/measure_hold.py`) |
| `running_weight` | 1 | **provisional** (no fuel price supplied); sweep 0.5 / 1 / 2 |
| `failure_cost` | 6.7e4 | twice the reachable crown excursion's cost, 2 x (183 / 1)^2 (1610 C target to the 1427 C trip), per trip step |
| `restart_steps` | 40320 (2 weeks) | restart time priced into a trip, `restart_steps x failure_cost` (refractory damage or glass out of range ends the campaign; a furnace heat-up schedule runs about two weeks, provisional); where a plant engineer would get it: the plant's restart procedure |

`rho_floor_tracking` is the oracle's best hold for tracking -- the lowest per-seed
hold cost the reference controller demonstrated, in the reward's units, which
is 1 per term where the floor is that hold and less where the floor is clamped
at the instrument resolution -- and `rho_floor` the same with consumption
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

## Actuators and timing

| figure | value | status | source |
| --- | --- | --- | --- |
| fuel slew | 1/1800 of the span per s: the full band in 30 min, 1 % of firing per min | **Ours, for the owner to accept** | see below |
| reversal period | 25 min | Read | 20 to 30 min, [Hubert, Lehigh glass lecture 3](https://www.lehigh.edu/imi/teched/GlassProcess/Lectures/Lecture03_Hubert_industglassmeltfurnaces.pdf) |
| firing gap at a reversal | 40 s | Read | about 30 to 60 s with no flames, same lecture, and [this Mustansiriyah lecture](https://uomustansiriyah.edu.iq/media/lectures/5/5_2018_11_25!11_08_27_PM.pdf) |
| fuel transport delay | 60 s | **Ours** | gas train, air adjustment, flame development; no source found |
| crown thermocouple lag | 120 s | **Ours** | a refractory sheath; no source found |
| control interval | 30 s | **Ours** | a supervisory loop's interval; no source found |
| gas heating value | 1 % (1 sigma) | Read for the size | ΔHi 4 % between North Sea and Russian H-gas with Wobbe within 0.4 %, Leipzig 2011, [Leicher, GWI, 2019](https://www.energigas.se/media/2515/1_2-joerg-leicher_gwi-s.pdf), slides 13 and 27; two sigma either side of the mean |
| its correlation time | 1 day | Read for the range, the day is ours | composition varies "usually weekly and sometimes even daily", [Glass Service, GM3 2021](https://www.glassonline.com/wp-content/uploads/2025/01/GLASS-SERVICE-GM3_2021.pdf); the fast end, since a 13 h episode sees little of a week |

**The slew limit is not sourced.** Our judgement is that the gas valve is not what
limits a furnace, and that keeping the air, the regenerators and the glass together
is. Havel et al.
([2006](https://www.irsm.cas.cz/materialy/cs_content/2006/Havel_CS_2006_0000.pdf))
ask that temperature changes in the tank stay below a limit, and give no number.
No published move limit was found for a float furnace. 1 % of firing per minute is
0.5 % a step, under a quarter of the 2.3 % the review saw. It still allows a 6 K
trim, 0.0032 kg/s at the plant's steady gain, in about one step, and answers a 4 %
change of gas in four minutes, so it limits the loop and not the task.

## Readings

The observation reports the crown, the preheated air and the delivered fuel flow
through their instruments (`docs/proposals/measurement-noise.md`), and `fuel_pct`
inherits the fuel meter's noise. The reversal phase, the setpoint and the pull
stay exact.
`sensor_noise_scale = 0` gives the noise-free task.

| reading | sigma | step | instrument | status | source |
| --- | --- | --- | --- | --- | --- |
| `T_crown_meas` | 0.36 K | | Type B thermocouple with E+H iTEMP TMT82 transmitter | Confirmed | [TMT82](https://bdih-download.endress.com/file/2c4c2c5496b6c81881df53d319b2e2f5/TI01010TDE_2725-00.pdf) |
| `T_air_preheat` | 0.32 K | | Type S thermocouple with TMT82 transmitter | Confirmed | [TMT82](https://bdih-download.endress.com/file/2c4c2c5496b6c81881df53d319b2e2f5/TI01010TDE_2725-00.pdf) |
| `fuel_flow` | 3e-4 kg/s | | Rosemount 8800D vortex gas meter | Not checked | [8800D](https://core.instrumart.com/assets/Rosemount-8800D-Series-Vortex-Flow-Meter-Datasheet.pdf) |

Both thermocouple figures come from the transmitter's accuracy at a stated
2 sigma, so they are upper bounds on its scatter, electronics only.

**Weak figure.** The refutation pass did not reach `fuel_flow`, and the meter
class is assumed. It reads the sheet's 0.1 % repeatability, which has no stated
coverage, as 2 sigma on 0.6 kg/s.

The crown's noise goes after the modelled lag, on a measured copy of the state.
`T_crown_meas` in the state stays the lag's own value, so one step's noise never
enters the next reading.

The expert reads these values, and so does every tuner. The oracle plans and acts
on the true state, so it stays a full-state bound. `reward`, trips and the score
read the true crown. `info["measured_reward"]` is the same cost on the readings,
with any trip taken from the true state. Its crown is the thermocouple, lag and
noise together, because that is the only crown a plant can read, so resting on
target costs a learner `(0.36 / 1)^2 = 0.13` per step. At `sensor_noise_scale = 0`
it is the true reward, as on every task, even though the lag remains.
