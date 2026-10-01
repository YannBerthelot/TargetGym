# Grid battery — physics model, provenance and validation

Reference system: a **2 MWh / 1 MW lithium-ion grid battery** — a 2-hour
system, the common utility build — modelled as a first-order equivalent
circuit with lumped thermal and capacity-fade dynamics.

Contract for `target_gym.energy.battery`. Method:
`docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 1. Model

No single published benchmark defines a grid BESS the way NREL defines a
turbine, so this is a **first-principles equivalent-circuit model** with
parameters sized to reproduce published *behaviour* — round-trip efficiency,
cell voltage window, thermal rise. Per the project's standard, that is
physically defensible rather than citation-backed, and §2 is where it earns
its keep.

```
V_term     = OCV(soc) − I R0 − v_rc
dsoc/dt    = −I / Q
dv_rc/dt   = I / C1 − v_rc / (R1 C1)
C_th dT/dt = I² R0 + v_rc I − UA (T − T_amb)
dq_loss/dt = calendar(T) + cycle(|I|, T)
```

The controller commands **power**, so current follows from `P = V_term·I` — a
quadratic whose physical (smaller) root is taken. That is not a detail: it
means deliverable power is bounded by `(OCV − v_rc)² / 4R0`, a limit that
tightens as the pack empties.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Cell-to-cell imbalance, balancing circuits | Single lumped pack. |
| Electrochemical (P2D/SPM) dynamics | An equivalent circuit is the standard reduction for control work and is orders of magnitude cheaper. |
| Voltage hysteresis, multiple RC branches | One RC branch; adequate for minute-scale dispatch. |
| Power-electronics switching, converter efficiency curve | Losses are ohmic only. |
| Calendar life beyond the episode | Fade accumulates but never fully depletes the pack in one episode. |

**Regime of validity.** Dispatch operation at up to ~1 C, 10–90 % state of
charge, near ambient temperature. Not valid for fast charging, deep discharge
or thermal-runaway conditions.

---

## 2. Validation targets

| Quantity | Target | Model | |
|---|---|---|---|
| Round-trip efficiency at rated power | 88–95 % (Li-ion grid BESS) | **90.6 %** | ✅ |
| Cell voltage window | 2.7–4.2 V | **2.84–4.19 V** | ✅ |
| OCV monotone in state of charge | required | yes | ✅ |
| Steady thermal rise at rated power | ~10–20 K (actively cooled) | **14.6 K** | ✅ |
| Thermal time constant | tens of minutes | 50 min | ✅ |
| 10→90 % traverse at rated power | ≈ 96 min for a 2 h system | 96 min | ✅ |
| Coulomb counting: ∫I dt = ΔSoC·Q | exact | closes | ✅ |

Round-trip efficiency is the load-bearing check — it is what sizes `R0`, and
it is jointly constrained by pack voltage, capacity and rated power. Two
sizing errors were caught by these targets before any model code ran: an `R0`
of 0.05 Ω gives 79 % round-trip (far below the band), and a passive `UA` of
250 W/K implies a **438 K** temperature rise, which is absurd — grid packs are
actively cooled.

---

## 3. Parameter table

| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `energy_nominal`, `power_max` | 2 MWh, 1 MW | – | 2-hour system, 0.5 C | ✅ |
| `n_series`, `capacity_As` | 192, 2500 Ah | – | Gives ~800 V nominal | ✅ |
| `R0` | 0.02 | Ω | Sized for 90.6 % round-trip | ✅ |
| `R1`, `C1` | 0.01 Ω, 20 kF | – | TUNED — diffusion time constant ~200 s | ⚠️ |
| OCV coefficients | see params | – | NMC-like fit; validated on window and monotonicity | ⚠️ |
| `C_thermal`, `UA_thermal` | 9e6 J/K, 3000 W/K | – | ~10 t pack; active cooling | ⚠️ |
| `k_calendar`, `k_cycle`, `E_activation` | 3e−9, 1.5e−9, 20 kJ/mol | – | TUNED — Arrhenius calendar plus throughput cycling | ⚠️ |
| `delta_t` | 5.0 | s | 360 steps = 30 min | ✅ |

---

## 4. Task design

**Dispatch tracking, paid for in wear.** Following the grid's request moves
charge in and out of the pack, and every MWh moved costs capacity fade, which
the running cost prices at every step. The charge window is real: leaving
0.05-0.95 state of charge trips the pack, and a dispatch held in one direction
long enough would get there. Over a scored episode, though, the window does
not bind on a controller that tracks exactly. The episode is 360 steps of 5 s
(30 minutes, six 300 s dispatch blocks) from an initial state of charge drawn
uniformly in 0.35-0.75. A feedforward that follows the schedule exactly keeps
the state of charge within 0.17-0.85 over 2000 seeds and never trips; over a
60-minute schedule, twice the scored length, it stays within 0.10-0.92, still
with no trip (oracle audit, 2026-10-01). So within a scored episode, tracking
now does not cost the ability to track later. What it costs is wear.

**The dispatch signal is a schedule, not a random walk.** Twelve blocks, each
held for a 300 s market interval, drawn uniformly in ±0.8 MW, with 2 kW of
regulation jitter on top; the scored 30-minute episode spans the first six.
That is what a grid battery is actually handed: a setpoint for a dispatch
interval, then another.

It used to be an Ornstein-Uhlenbeck process, and that made the task
unmeasurable rather than merely unrealistic. Its one-step innovation had a
standard deviation of 63.6 kW against a 150 kW tracking band, so the best
attainable tracking reward was **0.429** — and the shipped PID scored 0.447
while the MPC scored 0.430. Both controllers were pinned on an irreducible
noise floor, and nothing that could be plugged into the environment would have
scored meaningfully better. The error is now the transient after each scheduled
step, which is a property of the controller rather than of the dice.

The whole schedule is in the state, so a predictive controller can see the next
block coming; the observation exposes only the current request, which is what
keeps that a real advantage rather than a free lunch.

**Observation** `[soc, V_cell, T_cell, P_MW, target_P_MW]` — what a battery
management system actually reports. The diffusion voltage `v_rc` and the
accumulated capacity fade `q_loss` are hidden: neither is directly measurable,
and the fade in particular is the cost the controller is implicitly trading
against.

**Efficiency depends on state.** Losses scale with current squared, and the
current needed for a given power depends on state of charge through the OCV
curve, so the same dispatch costs more when the pack is low.

**Reward** = −(dispatch imbalance + degradation + trip cost), in dollars per
step (version 2, below). Version 1 also had a weak pull toward mid charge, to
keep headroom in both directions; version 2 drops it, and over a scored
episode a controller that tracks exactly does not need that headroom.

---

## 5. Baselines

Version-2 reward, over the ten recorded seeds of the scored 360-step episode.
The reward is a cost, so a return closer to zero is better.

| controller | return | mean power error | trips |
|---|---|---|---|
| PID + charge guard | **−1.244** | 0.020 MW | 0 |
| constant −0.5 / 0 / +0.5 | −29.9 / −18.3 / −25.3 | 0.59 / 0.37 / 0.50 MW | 0 |

The PID's return is the recorded baseline, as on the environment page
(`docs/environments/battery.md`, which also carries the oracle's). The other
numbers were measured on the same seeds (2026-10-01), with a replay of the
PID that reproduces its recorded returns to within 5e-6.

The charge guard fades the demand out *only in the direction that would
breach a limit*: discharge is throttled below 0.17 state of charge and charge
above 0.83, within 0.12 of each limit, and the middle is untouched. Within a
scored episode it is insurance that is almost never called on. Over 2000
seeds it acts on 4 of them, 111 of 720,000 steps, and a copy of the PID with
the guard removed never trips either, staying within 0.17-0.85 state of
charge. On those four seeds the guard changes the return by between −0.041
and +0.0005: it can only move the command away from the target, which costs
imbalance and saves a little wear. It stays because the limits are real and a
trip is priced as an hour of downtime, and nothing else in the PID would keep
it off them.

**The oracle is a feedforward, not a planner.** Delivered power equals the
command within the step, and the target is the scheduled level plus white
noise drawn after the action. So the best causal command is the level of the
block the step is scored against, `dispatch_block(t + 1)`, and what is left is
the noise, whose expected absolute value is `e_floor`. The oracle commands
exactly that (`experts.py`, `ScheduleFeedforward`). Measured on the protocol
seeds, its mean error is 1577 W against the closed-form 1596 W, and its
protocol cost is 7.82e-4 $/step against 1.115e-3 for the gradient MPC it
replaced (oracle audit, 2026-10). A controller that knew the noise would
remove only the noise, and shading the command toward zero to save
degradation gains about 0.03% on the protocol seeds. It does not guard the
state of charge. Over 2000 seeds of the scored 30-minute episode the state of
charge stays within 0.17-0.85 and never trips; over the full 60-minute
schedule, within 0.10-0.92, still with no trip. So within a scored episode
the energy budget does not bind on a controller that tracks exactly; what
tracking costs it is degradation, which the running cost prices.

---

## 6. Known deviations

**⚠️ D1 — no published reference system.** Unlike the wind turbine, the
parameters are sized to reproduce published *behaviour* rather than taken from
a specific documented machine. The efficiency, voltage window and thermal rise
are right; the particular cell chemistry is generic NMC-like.

**⚠️ D2 — single RC branch.** Real packs show relaxation over several
timescales. One branch captures the dominant minute-scale polarisation and
misses both faster and slower components.

**⚠️ D3 — simplified ageing.** Calendar plus throughput with an Arrhenius
temperature factor. Real fade depends on depth of discharge, C-rate history and
state-of-charge dwell in ways this does not represent, so the degradation cost
is directionally right but not quantitatively trustworthy.

**⚠️ D4 — no cell imbalance.** A lumped pack cannot represent the
weakest-cell behaviour that actually determines real pack limits.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `battery` | 360 | 5 | 30 min | 1 in [-1, 1] | 5 | 19 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are `EnvSpec.test_params`, which is what the recorded
baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2), in dollars per step

`compute_reward = -(tracking + running + failure)`, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 1596 W | closed form: the dispatch target is a block level plus white noise of sd 2 kW drawn after the action, so no controller holds E\|error\| below sd * sqrt(2/pi). The shipped PID holds 5.6 kW within blocks, and the oracle, a feedforward of the scheduled level, 1.58 kW: the floor itself (`scripts/measure_hold.py`, per dispatch block after settling, 3 seeds). |
| `e_tol` | 0 | none |
| `tracking_exponent` | 1 | linear imbalance |
| `imbalance_price` | 100 $/MWh | dispatch imbalance tariff (as in the audit) |
| `c_hold` | 1.92e-8 | fractional capacity fade per step while holding, PID and oracle alike (`scripts/measure_hold.py`); mostly calendar ageing, which no controller avoids, hence charged only above it |
| `fade_price` | 300 $/kWh | replacement cost of lost capacity |
| `pack_kWh` | 1692 | capacity_As x OCV at 50% SOC / 3.6e6 |
| `failure_cost` | 2 x the imbalance of the reachable 1.8 MW (a 0.8 MW target against the 1 MW power limit) per step | SOC / thermal trip |
| `restart_steps` | 720 (1 h) | restart time priced into a trip, `restart_steps x failure_cost` (a protection trip's reset; provisional); where a plant engineer would get it: the plant's restart procedure |

The version-1 SOC-comfort term is dropped (**provisional**): it has no owner
price, and a pack driven to the edge of its window pays through the dispatch
it can then not follow.

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
