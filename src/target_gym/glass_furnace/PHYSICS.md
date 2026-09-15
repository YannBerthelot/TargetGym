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
| `reversal_period` | 1500 | s | 25 min, typical float furnace | ✅ |
| `reversal_dead_time` | 40 | s | Firing interrupted while the valves change over | ⚠️ |
| `reversal_firing_floor` | 0.05 | – | Pilots stay lit through the changeover | ⚠️ |
| `tau_thermocouple` | 120 | s | Crown element in a refractory sheath | ⚠️ |
| `FUEL_DEAD_TIME_STEPS` | 2 | steps | 60 s: gas train, air adjustment, flame development | ⚠️ |
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

**Reward.** `log_scaled_reward(|err|, precision_floor, 250 °C) -
fuel_cost_weight·fuel_norm`. Every halving of the error is worth the same
increment, from the operating envelope down to the 1 °C the crown thermocouple
resolves, so the reward keeps separating controllers exactly where a good one
operates: 1 °C scores 0.875, 2 °C scores 0.801, 4 °C scores 0.709. It replaced
a clipped square, `clip(1 - |err|/40, 0, 1)²`, which was flat at zero for any
error beyond 40 °C and so gave no gradient where it was most needed. See
`docs/reward-shaping.md`.

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

**Reset** starts from a *consistent operating point* — offsets measured from
the settled steady state at nominal firing — not arbitrary temperatures. A
furnace that takes days to equilibrate would otherwise spend the whole episode
in an initial transient with nothing to control.

**Observation** `[T_crown, T_air_preheat, fuel_pct, reversal_phase, target]` —
a real furnace has crown and air-preheat thermocouples and the operator knows
the reversal state. Glass temperatures, checker profile, batch mass and the
pull-rate disturbance are hidden: **6 of 9 dynamic quantities**.

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

PID gains come from a grid search maximising mean episode reward
(`scripts/tune_pid.py --envs glass_furnace`). Relay autotuning does *not* work
on this plant: the crown temperature integrates the firing rate, so under a
bang-bang relay it drifts without sustained zero-crossings and Åström–Hägglund
has no ultimate gain or period to extract — the same failure mode as FourTank.

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
| `glass_furnace` | 1600 | 30 | 13.3 h | 1 in [-1, 1] | 5 | 26 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are `EnvSpec.test_params`, which is what the recorded
baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2)

`compute_reward = -(tracking + running + failure)`, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 1.0 K (the MPC holds 0.175) | the lowest per-seed long-run mean \|crown error\| the shipped MPC held under the shipped pull disturbance (`scripts/measure_hold.py`, 3600 hold steps after a 10 800-step burn-in; seeds 0.192 / 0.175 / 0.228 K, PID 0.49-0.52 K). Per-seed minimum; upper bound Below the instrument resolution the plant's own table cites, and measurement noise is not modelled, so the resolution sets the scale: a hold the instrument cannot see is not a floor. |
| `e_tol` | 0 | the crown temperature target has no band |
| `tracking_exponent` | 2 | quadratic |
| `c_hold` | 0.590 kg/s | fuel flow while holding (`scripts/measure_hold.py`) |
| `running_weight` | 1 | **provisional** (no fuel price supplied); sweep 0.5 / 1 / 2 |
| `failure_cost` | 6.7e4 | twice the reachable crown excursion's cost, 2 x (183 / 1)^2 (1610 C target to the 1427 C trip), per trip step |
| `restart_steps` | 40320 (2 weeks) | restart time priced into a trip, `restart_steps x failure_cost` (refractory damage or glass out of range ends the campaign; a furnace heat-up schedule runs about two weeks, provisional); where a plant engineer would get it: the plant's restart procedure |

`rho_floor_tracking` is the NEA reference for tracking -- the lowest per-seed
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
restarts at once as `reset_env` would, on the same clock. `terminated` is never
raised; `info["tripped"]` marks the event for the evaluator. `reward_version = 1` reconstructs the
capped log-scaled reward of the previous version (`precision_floor` and the
old weights are read only by it).
