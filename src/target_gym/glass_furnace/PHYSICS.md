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

**The expert: the base layer, SIMC-tuned.** Practice runs an MPC over the crown
loop. The expert is the base layer under it: a PI on the filtered crown reading. Its
recipe is section 10 of `docs/proposals/expert-protocols-process.md`.

On each training seed (100 to 109) the firing is in manual at the settled point's
fuel, the pull and the gas quiet. The step lands two steps after a firing
interruption and ramps at the fuel's slew, and the crown is read against the same
phase of its own cycle before, three reversals back, because that swing is as large
as the step's whole effect. Held in manual the crown repeats every 75 minutes, and a
reading one reversal back still carries 0.7 K. The noise band is 3.37 K with the
reversal dips in it, so N3's 5 bands cannot be reached, and the step is the fuel's
whole room, one raw unit, moving the crown 1.8 bands in the 20 minutes before the next
interruption. That ships with its ratio, and no band exit can read the delay, so it
is held at the fuel line's 60 s.

The fit, `k' e^(-theta s) / (s (tau2 s + 1))`, reads k' 0.0049 K/s per unit action and
tau2 56 s. A noise-free run reads 0.0045 and under 85 s. With half a sample and the
half rule, theta is 103 s. SIMC gives Kc 1.0, and N5 and N7 take it down: the filter
reaches its cap of 82 s, and the output's noise still passes the valve's dead band,
so Kc falls to the floor of the largest load, 0.297 per K, with Ti 2730 s. The
protocol's derivative (D6) is left out: at this Kc, 100 s of derivative on a 0.36 K
reading at 30 s scatters about 0.5 raw a step.

Kp has fallen from 12 to 0.3, and the action no longer saturates. On the held-out
seeds 200 to 209 it holds 1.15 a step (worst 1.82) against a ceiling of 1, where the
old gains' relay held 1.87, with no trip: crown RMS 1.3 to 1.9 K, p99 4 to 6 K, and
100 % of steps within the +-10 K of check 2. Spec, a strict xfail in
`tests/glass_furnace/test_glass_furnace_pid.py`: the true crown within 2 K on at least
95 % of hold steps. Measured 78 to 91 %. The misses are the firing interruptions at
each reversal, which a PI on the crown meets only after the fact.

**The oracle.** It is the shared solver on the plant itself (`oracle.py`):
all four checker nodes per chamber, the fuel pipeline, the flame's algebraic
solve and the reversal cycle, with the pull and the gas's heating value planned on
their mean paths. Cooper's recipe sets one move per 14.5 minutes (T_c = 29 steps,
from the working end's fits; the crown's own fits fail the noise test, H16), and the
plan runs to the episode's end. Each move is the valve's slew, held for those 14.5
minutes, so the plan sees through the rate limit that stalled the fuel at 8.84 K. It
still cannot answer a single 25-minute reversal: on protocol seed 2 it holds 0.44 K,
against the 0.29 K of a planner that solved every step and the 0.25 K bound.

Before that the oracle was `NewtonMPC` (iterative LQR) on `step_env`, over a
120-step horizon, and before it a CasADi model of the plant (oracle audit,
2026-10-03) with two checker nodes per chamber against the plant's four. That
reduced model ran 0.14 to 0.19 K a step colder than the plant, so it corrected
itself with a crown heat-rate estimate from the one-step prediction error.

The long-run hold `NewtonMPC` measured, 0.061 / 0.077 / 0.088 K per seed,
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

## Rebuilt plant

`rebuilt.py` is the plant the proposal `docs/proposals/glass-furnace-task.md` asked for. It
sits beside the shipped plant, unregistered, and reuses its firing, regenerators, clocks and
readings. Nothing above changes until the owner reads the judging. The sourced tables and
their second-fetch status are in `glass-sources-2.md` in the session notes.

**Roof.** Five brick cells, 15, 30, 60, 120 and 225 mm, hot face first, by conduction. The
score reads the first cell, the owner's choice, so the score is the hot face to 15 mm. The
back loses through an equivalent U on its cold face.

**Bath.** Three zones along the melter, blanket, hot spot and throat side, each a surface
and a deep layer, then the working end. The pull runs through them with two buoyant cells
and the conditioning return. A cell whose far end is hotter runs backward. A floor sees
half flame and half crown.

**Blanket.** Mass from the charger, melted by radiation and convection from above and by the
melt from below, over an area of mass / (density x thickness). Melting heat and thickness
are moved by hidden ramps. There is no runaway, and none was sought.

**Removed by the owner on 2026-11-07:** the batch charge and the batch-line reading. The
owner could not size them. They return if the control properties need them.

| Symbol | Value | Unit | Status | Source |
|---|---|---|---|---|
| crown thickness | 450 | mm | Read | ICG TC11 deck 2022, slide 3; 457 mm on a float brick, Wereszczak et al. 2000 |
| `k_brick` | 2.48 | W/(m K) | Read, a bound | Wereszczak et al. 2000, p. 173: as new 1.24 doubled for service temperature; vendor sheets give 1.44 to 1.74 at 1000 C |
| `rho_brick` | 1870 | kg/m3 | Read | Wereszczak et al. 2000, p. 166 |
| `cp_brick` | 1245 | J/(kg K) | **Ours** | not found; 3R per atom on SiO2 (74.8 J/(mol K) at 60.08 g/mol), the high-temperature limit |
| `U_roof_back` | 1.70 | W/(m2 K) | Derived | 2 kW/m2 (Beerkens 2011 p. 41; Wang et al. 2000 p. 303) at the 1195 to 1210 C cold face (Wereszczak) |
| `tc_depth` | 36 | mm | **Ours** | not found; 10 min by d^2 / 2 alpha, the middle of the proposal's 5, 10, 20 min. A pocket, per the Tempsens catalogue |
| `tc_offset` | see code | K | Derived | the orbit's mean of hot face less thermocouple, so the reading sits on the target's scale. No source; a real plant's setpoint is in thermocouple units |
| `glass_mass` | 625 | t | As shipped | 30 h residence |
| `tank_depth` | 1.35 | m | Derived | 625 t over 200 m2 at 2319 kg/m3; float tanks run 1.2 to 1.5 m |
| `tank_width` | 11 | m | Read | Wang et al. 2000, 455 t/d float |
| three equal zones | 1/3 | | **Ours** | the hot spot's centre at 1/2 of the length, the near end of the 1/2 to 2/3 rule (Materials 17 (2024) 4989) |
| `surface_fraction` | 0.422 | | Read | Poles 1997: the current reverses at z = 0.578 |
| `rho_glass` | 2319 | kg/m3 | Read | Pilon et al. 2002, eq. 10, at 1400 C |
| `beta_glass` | 5.95e-5 | 1/K | Read | Wang and Zhou 1992, Table 1, cubical, float |
| viscosity | VFT | Pa s | Read | DGG X standard glass, eq. 1 |
| k_eff | Pilon | W/(m K) | Read, weak | Pilon et al. 2002, eq. 14. Sources differ threefold; this is the middle one |
| cell flow | 0.00542 g beta (dT/l) h^4 / nu | kg/(s m) | Read, form only | Poles 1997, eq. 2.3.7, integrated to the reversal height. Constant-viscosity, low-Rayleigh solution; a melt is far outside it |
| `current_scale` | 0.05 | | Derived | set once, so the first loop carries the 600 % of pull in Beerkens 2011 p. 12 and Hubert slide 48. The raw formula gave 38 times the pull |
| `back_flow_ratio` | 2.6 | | Read | Wang and Zhou 1992, Fig. 8, float, 1.5 m, no weir |
| `gas_view` | 0.5 | | **Ours** | a flame neither black nor clear. Needed because the shipped form sent a floor both full views and doubled the flux |
| `U_bottom` | 1.0 | W/(m2 K) | Read | as shipped; 1.4 kW/m2, inside Beerkens's 2 kW/m2 |
| `rho_batch` | 1400 | kg/m3 | Read | Pilon et al. 2002 |
| `blanket_thickness` | 0.20 | m | Read, model | Pilon et al. 2002, assumed there. No float measurement was found |
| `blanket_solid` | 0.85 | | Read | Wang et al. 2000, the best match to plant data, as the shipped `batch_shield` |
| `T_batch_top` | 1200 | C | Read | Wang et al. 2000, p. 302, reactions complete at 1473 K |
| `H_melt` | 2.2e6 | J/kg batch | Read | Pilon et al. 2002, at 1450 K |
| `cullet_gain` | 0.352 | per unit ratio | Derived | 2.5 % less fuel per 10 points at held crown (DOE 2017, AGC), through this plant's balance; `scripts/glass_rebuilt_settle.py` |
| `cullet_move`, band | 10, +-5 | points | Band read, size the owner's | 20 to 30 % on float lines (Guardian, Wang et al. 2000, DOE 2017); up to 10 points in 1 h, six per episode |
| `thickness_move`, band | 20, +-20 | % | The owner's | no source; 12.5 to 20 cm across model papers |
| `m_pull_noise_std` | 0.058 | kg/s | Read | +-2 % (US11261117B2) as two sigma; the shipped 0.4 was flagged weak |
| `tc` delay, `tau_thermocouple` | 120 | s | **Ours**, as shipped | the sheath |

**Checks on the settled orbit, crown 1590 C.** No number was moved to reach these.

| Quantity | Published | Model |
|---|---|---|
| Specific energy | 4 to 6 GJ/t | 5.13 |
| Preheated air | 1200 to 1400 C | 1229 |
| Stack | 400 to 600 C | 522 |
| Crown cold face | 1195 to 1210 C | about 1225 |
| Crown loss | under 2 kW/m2 | 2.0 |
| Share of net heat into the blanket | 80 to 90 % | 87 % |
| Hot spot passes | 6 to 8 | 7.6 |
| Batch line | 3 to 6.5 m | 3.5 m |
| Energy balance, fuel against all outflows | closed | 1.2 % |

**Deviations of the rebuilt plant.**

- **Blanket.** Coverage 19.5 % against the 25 to 40 % optimum, and 22 min of residence against
  30 to 60 min. The batch line is inside its band, so sources disagree more than the model does.
- **Longitudinal surface difference.** About 40 K against the 170 to 205 K that sources give
  between a batch point and the peak. A zone is a mean over 6 m, and the zones can't hold a peak.
- **Vertical difference.** 40 to 75 K against the 300 to 340 K of the 1992 model. Two layers can't
  hold a stagnant bottom, so the deep layer is the return current and not the floor. Viscosity in
  the cell flow is therefore read too hot, which `current_scale` absorbs.
- **Hot spot.** It sits between the hot and throat zones, at about 2/3 of the length, the far end
  of the rule. Cell 2 settles near zero. The source's second loop of 200 % does not appear.
- **Working end.** 1475 C against about 1250 C at the end of conditioning. The shipped cooling is
  unchanged, and 1 zone is the whole conditioning end.
- **Hot face swing.** 38 K in the 15 mm cell at each reversal. No source gives it; thermocouple
  figures do not see it. A thinner cell would swing more.

**Judging, 2026-11-07.** One attempt, in the proposal's order; the full tables are in the
proposal's "Results of the judging".

| Step | Result |
|---|---|
| Plain PI, 12 seeds | Reading sd 1.56 K against the real 2.2 K, hot face rms 7.02 K, no trips |
| Recipe | k' 0.00842 K/s per raw, theta 189 s, Kc 0.314, Ti 1513 s, pull feedforward 0.916 raw per kg/s |
| Best rung | Lead of 10 min on the reading, cost 47.98, reading sd 1.44 K |
| Room, 4 seeds, whole episode | 1.61, spread 1.60 to 1.62. Oracle cost 29.0 to 29.8, converged by knots and by start |
| Switch-off | Firing gap and reading lag shrink the gap. Every bath factor leaves the claim |
| Rule 7 | 1.18 times the median step |

## Physical plant

`physical.py` is attempt 2, unregistered, written from the equations. Its design, results and
every open label are in `docs/proposals/glass-furnace-task.md`, section "Attempt 2", and the new
sources with their second-fetch status are in `glass-sources-3a.md` and `glass-sources-3b.md` in
the session notes.

| Quantity | Value | Source |
|---|---|---|
| Geometry | 21.5 + 20.5 m, 11 m wide, six ports | W2000b |
| Glass depth | 1.2 m | Materials 17 (2024) 4989 |
| Gas absorption | 0.08 /m | Hoogendoorn, side-fired |
| Melt conductivity | 1.31 + 5.90e-4 T plus Rosseland; bands 42.8 and 459.2 /m | Liu 2016 eqs 5 to 7, Table III |
| Foam thickness | Eq 19, K = 2905 | Pilon, Fedorov, Viskanta 2001 |
| Foam lifetime | 50 s half life at 1200 C, 58 kJ/mol | Kappel 1987 Tables 2, 3 |
| Foam extinction | 267 /m, 20 % through 2 cm by two-flux | Fedorov and Pilon 2002 |
| Fining band | 1420 to 1500 C | Beerkens review |
| Sulfate | 1 wt % batch, 0.27 wt % SO3 retained | Laimbock via Pilon; Rudolph 1990 |
| Batch | 2.2 MJ/kg to 1450 K, speed from the charger | Pilon II; UV1986 |
| Regenerator casing | 1.4 W/(m^2 K), from 1726 W/m^2 | ICG slides 12 to 14 |
| Refiner crown | 6.8 W/(m^2 K), four times insulated | Ford float, seen once |
| Emissivity, firing shares, couple depth, height | `ours` | None found |

## Attempt 3: the slow plant and its job (unregistered)

`physical.py` with `physical_env.py`, `physical_scenarios.py`, and the plant record
`REALITY.md` (charter Amendments 1 to 3). Design, results and decisions are in
`docs/proposals/glass-furnace-task.md`, from "Attempt 3" on. The cold review is
`COLD_REVIEW.md`. The settled design state is `scripts/glass_slow_design.npz`.

**Job and score (rule 15).** Keep the glass fined and at forming temperature at the canal
at the least fuel (REALITY.md, section 6). Cost per step: fuel at 60.5 EUR/MWh [S21], plus
the charged pull at 600 EUR/t (`ours`, the owner's pick) when the canal is under 1100 C
[S16] or the fining share leaving is under 0.5. The named property: lower variance lets
the fuel drop toward the defect limit, worth 2 to 4 % of fuel in the family [S1; S2; S3].

**Limits by kind (rule 14).** Fining: a zone, priced as lost glass. Canal: a zone, held by
the refiner's cooling loop. Crown at 1620 C [S13]: protected by the plant's override, the
owner's pick and a known deviation (the record finds wear, not protection). The 1400 C
low side is not modelled: it never binds, since the hottest face sits far above it.

### Cold review, item by item (rule 13)

| # | What differs from the plant | Status | Effect on the score |
|---|---|---|---|
| 1 | The agent moves the valve; operators move crown and bottom setpoints, with firing following the crown [S4] | Known deviation, for the owner | The agent must also do a crown loop's fast work against gas swings, which a setpoint action would hand to an inner loop |
| 2 | Quality is read as the fining share, 0.3 h late with 0.005 noise; real lines count defects at the cold end [S38], with no unit or resolution found | Known deviation | The loss variable is nearly observed; only the transit and noise hide the margin |
| 3 | Crown: a protected override on the true hottest face; the record finds wear, unpriced | Known deviation (owner's pick); wear measured as is | Running hot costs only fuel. Hours over 1600 and 1610 C are reported (`glass_slow_eval.py`) |
| 4 | Lost glass at 600 EUR/t, a step on the canal and fining at that moment, on the charged pull | Known deviation; the price is the owner's | The step makes full back-off optimal; the challenge found any price from 300 to 700 EUR/t gives the same back-off |
| 5 | The fining criterion is a flow-weighted mean share of 0.5, where bubbles leave only with complete fining [S12; S31] | Known deviation; fixed before any run, as asked | Looser than the source; defects from the fastest streamlines are averaged away |
| 6 | The canal is held by a cooling loop the task builds; no source shows that loop | Known deviation | The canal never binds; overheating costs fuel only, and cooling is free |
| 7 | Reset draws the walks from the seed but starts from a state settled at nominal | Known deviation | A start transient the agent did not cause, common to all policies |
| 8 | Walks are smooth first-order processes; real pull wander is stepwise | Known deviation | Slow, smooth disturbances favour a constant valve with a margin |
| 9 | No couple lag, drift or reversal ripple; air preheat and stack had no source | Partly fixed: air preheat and stack removed | Readings are cleaner than real |
| 10 | Batch islands as a wedge; no level loop, tweel, waist coolers or boost; forward flow into the waist 2.6 times the pull against 1.2 to 2.5 [S17] | Known deviation | The refiner is slightly more coupled to the melter than the range allows |
| 11 | Unsourced numbers | Listed below with their family ranges | See the list |
| 12 | A constant valve with a margin may do nearly as well as a controller | Measured: the best box null costs 4.1 % more than the oracle's best path, weighted; 0.7 to 3.7 % per scenario. The oracle is not converged, so these are lower bounds on room | Small room, as the family's 2 to 4 % saving says |
| 13 | Crown wear neither priced nor measured | Fixed: measured as is | Hours over 1600 and 1610 C, and the maximum, per policy |
| 14 | Cooling air power, scanner rejects and cullet remelt are not charged; the agent sees the valve after the override | Known deviation; the last is intended | Small |

### Unsourced numbers and their family ranges (rule 19)

| Number | Value | Family range or reason |
|---|---|---|
| Cold-end transit | 0.3 h | 14 to 20 min: a 50 m bath and 100 m lehr at 7.5 to 10.4 m/min [S40; S41; S42], arithmetic |
| Planned pull change | +-10 % | Inside 2 % float wander [S5] to a container plant's 29 to 42 % [S3, slide 8] |
| Gas swing | +-6 % of heating value | 3 MJ/m3 within a day [S20] on a Wobbe index of about 50, arithmetic |
| Pull ramp, notice, gas ramp, event window | 2 h, 8 h, 6 h, 24 to 96 h | Not found; ours |
| Fining pass | 0.5 | Middle of Beerkens' 1420 to 1500 C band [S31] |
| Lost glass price | 600 EUR/t | Not found; the owner's pick near one market figure seen once |
| Crown override margin and gains | 5 K, 0.0101 kg/s per K, 2 h | Ours, a protection's design; SIMC on the fuel step |
| Refiner cooling capacity, setpoint offset, gains | Twice design, 1 K, by SIMC | Not found; ours |
| Walk spreads and memories | Gas 2 %, pull 1 %, cullet 5 points, blanket 10 %, sulfate 17 %; 24 or 12 h | Gas: ENTSOG's 4 to 6 MJ/m3 read as four spreads [S20]; pull: the 2 % wander as two [S5]; cullet and blanket: the owner's +-10 points and +-20 %; sulfate: ours |
| Quality reading noise | 0.005 | Not found; ours |
| Waist and working end widths | 3.5 and 7.9 m | A 1300 t/d furnace's ratios to its melter [S18], scaled |
| Surface emissivity, port shares, couple depth and bay, combustion height | 0.8; ports 1 to 4 even then 0.7, 0.4; 36 mm in bay 3; 2.4 m | Not found; ours |
| Bubble radius | 8.3 mm | Top of the laboratory range [S32] |
| Scenario weights | Normal 0.7; pull up and down 0.05 each; gas rich and lean 0.1 each | Gas swings: none at stable entry points to weekly where LNG blends [S19; S20], each 0 to 0.3. Pull changes: none on a steady float line [S5] to about 20 job changes a month on a container line [S9], each 0 to 0.3. Normal: the rest, 0.4 to 1 |

### The reference ladder's row (rule 3, for `scripts/reference_ladder.py`)

- **Action:** the valve, kg/s, box 0.25 to 0.75. Step 120 s, episode 5040 steps (7 days).
- **Controlled variable:** the fining share read at the cold end, observation 4. Zone 0.5 to 1;
  the cheap side is low (less fuel, lower share), so the setpoint sits 2 sigma above 0.5.
- **Other limits:** the canal (held by the plant's loop) and the crown (held by the plant's
  override) need no rung of the reference.
- **Measured disturbances for feedforward:** pull and planned pull (observations 7 and 8),
  cullet ratio (9), batch line (10).
- **Scenarios:** `physical_scenarios.suite()`: normal, pull_up, pull_down, gas_rich,
  gas_lean.
- **Start:** `physical_env.reset(key, fields, ep, fuel, cooling)` from
  `scripts/glass_slow_design.npz`.

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
