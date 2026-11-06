# Grid battery: physics model, provenance and validation

Reference system: a **1 MW / 1.69 MWh lithium-ion grid battery**, modelled as a
first-order equivalent circuit with lumped thermal and capacity-fade dynamics,
behind an inverter and an auxiliary load.

Contract for `target_gym.energy.battery`. Method:
`docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 0. The rebuilt control problem

A rebuild, written down before its code (`docs/proposals/industrial-fixes.md`, PR 6).

The task asked whether a controller can follow a dispatch signal, and any policy
that copied the target could: delivered power equalled the command, the pack could
not run out inside 30 minutes, and wear cost $4.24/MWh against a $100/MWh imbalance,
so no choice ever arose. An echo of the target beat the expert on 10 of 10 seeds.

The rebuilt task asks **how far to follow a regulation signal from a finite, ageing
pack.** Over a 4 h window of hourly blocks, exact following leaves the charge window
on most seeds, and deep swings cost more wear per MWh than the imbalance they avoid.
Three things make that a decision:

1. **Energy is the limit.** Hourly levels of up to 0.8 MW against a 1.69 MWh pack.
2. **Cycle depth has a convex price.** A half-cycle of depth `d` costs half of
   `Phi(d)` of the pack's life, so the wear on the next MWh grows with `d`. Past a
   depth of about 0.64 the next MWh costs more in wear than in imbalance
   (Xu, Shi, Kirschen and Zhang, 2017).
3. **Delivered power is not the command.** Auxiliary loads sit between the inverter
   and the meter, so a trim has work to do.

The next block is in the observation, so the oracle no longer owns the lookahead.

Every figure is sourced in section 3, or labelled ours for the owner to accept.

---

## 1. Model

No single published benchmark defines a grid BESS the way NREL defines a turbine,
so this is a **first-principles equivalent-circuit model** with parameters sized to
reproduce published *behaviour*: round-trip efficiency, cell voltage window, thermal
rise. Its ageing and its losses behind the inverter come from published fits and a
measured system.

```
P_dc       = P_inv + pcs_loss P_inv^2 / P_max
P_meter    = P_inv - aux_load
V_term     = OCV(soc) - I R0 - v_rc
dsoc/dt    = -I / Q
dv_rc/dt   = I / C1 - v_rc / (R1 C1)
C_th dT/dt = I^2 R0 + v_rc I - UA (T - T_amb)
dq_loss/dt = calendar(T) + ripple(|I|, T) + deep half-cycles
```

The controller commands the **inverter**, and the meter is behind the auxiliary load,
so a command does not arrive whole. Current follows from `P_dc = V_term I`, a
quadratic whose physical (smaller) root is taken.

**Ageing has two parts.** A cycle of depth `u` costs `Phi(u) = 5.24e-4 u^2.03` of the
pack's life. A half-cycle costs `Phi(d) / 2`, charged as its depth grows, so the
marginal wear of a deep swing is high and that of a shallow one low. Swings smaller
than `cycle_hysteresis` are not turning points (a racetrack filter), and count as
ripple, charged by throughput at the stress of a cycle that deep. Full rain-flow
counting needs an unbounded stack, so this keeps two states: where the half-cycle
turned, and how far it has reached.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Cell-to-cell imbalance, balancing circuits | Single lumped pack. |
| Electrochemical (P2D/SPM) dynamics | An equivalent circuit is the standard reduction for control work and is orders of magnitude cheaper. |
| Voltage hysteresis, multiple RC branches | One RC branch; adequate for minute-scale dispatch. |
| Inverter dynamics and transformer | Assumed to settle inside the 5 s step (section 3). |
| Auxiliary load that moves with weather or load | Constant. MESA-1 saw it nearly flat up to 1C, and 5 to 7 kW lower in winter. |
| Calendar life beyond the episode | Fade accumulates but never depletes the pack in one episode. |

**Regime of validity.** Dispatch operation at up to about 1C, 10 to 90 % state of
charge, near ambient temperature. Not valid for fast charging, deep discharge or
thermal-runaway conditions.

---

## 2. Validation targets

| Quantity | Target | Model | |
|---|---|---|---|
| AC round-trip efficiency at rated power | 83 to 88 %, around the 85 % Xu et al. 2017 simulate | **84.4 %** | ✅ |
| Cell voltage window | 2.7 to 4.2 V | **2.84 to 4.19 V** open circuit | ✅ |
| OCV monotone in state of charge | required | yes | ✅ |
| Steady thermal rise at rated power | about 10 to 20 K (actively cooled) | **17.0 K** | ✅ |
| Thermal time constant | tens of minutes | 50 min | ✅ |
| 10 to 90 % traverse at rated power | closes on the cells | **75 min** | ✅ |
| Cycles to end of life at 80 % depth | 3000 | **3002** | ✅ |
| Calendar fade at rest | 2 %/yr | **2.0 %/yr** | ✅ |
| Coulomb counting: integral of I dt = delta SoC Q | exact | closes | ✅ |

Round-trip efficiency is the load-bearing check. It is what sizes `R0`, and it is
jointly constrained by pack voltage, capacity and rated power. Two sizing errors
were caught by these targets before any model code ran: an `R0` of 0.05 ohm falls
far below the band, and a passive `UA` of 250 W/K implies a **438 K** temperature
rise, which is absurd for an actively cooled pack.

The pack is **1.69 MWh at 677 V** (2500 Ah at the OCV of 50 % charge), and the
traverse **75 min**. Earlier versions of this page said 2 MWh, about 800 V and a
validated 96 min, read off the rating rather than the cells, and the 96 min test
passed by reading that same rating.

---

## 3. Parameter table

Status of each source: **read** means the page was opened and the value found there
by two readers, and **ours** means a choice for the owner to accept.

| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `power_max`, `energy_nominal` | 1 MW, 1.69 MWh | – | Closes on the cells | ✅ |
| `n_series`, `capacity_As` | 192, 2500 Ah | – | 677 V at 50 % charge | ✅ |
| `R0` | 0.02 | ohm | Sized for the round-trip band | ✅ |
| `pcs_loss` | 3 % at rated | – | Ours: a search result gave 97.0 % CEC-weighted for a Delta 125 kW inverter, but the datasheet sits behind a terms page and neither reader opened it | ⚠️ |
| `aux_load` | 15 kW | W | Read: [MESA-1 test report](https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-27237.pdf), 30 kW on a 2 x 1 MW system, nearly constant up to 1C, scaled by power | ✅ |
| `R1`, `C1` | 0.01 ohm, 20 kF | – | TUNED: diffusion time constant about 200 s | ⚠️ |
| OCV coefficients | see params | – | NMC-like fit; validated on window and monotonicity | ⚠️ |
| `C_thermal`, `UA_thermal` | 9e6 J/K, 3000 W/K | – | About 10 t pack; active cooling | ⚠️ |
| `stress_alpha`, `stress_beta` | 5.24e-4, 2.03 | – | Read: [Xu et al. 2017](https://arxiv.org/abs/1703.07824) and [Xu et al. 2018](https://arxiv.org/abs/1707.04567), NMC cells, 3000 cycles at 80 % depth | ✅ |
| `life_fade` | 0.2 | – | The usual end of life at 80 % of capacity | ✅ |
| `cycle_hysteresis` | 0.05 | – | Ours: the depth below which a swing is ripple | ⚠️ |
| `k_cycle` | 2.66e-10 | 1/(kA s) | `Phi(0.05) / (2 * 0.05)` of life per capacity moved, in fade: $3.77/MWh, the order of the $2/MWh handbook figure Xu et al. 2018 quote | ⚠️ |
| `k_calendar` | 6.34e-10 | 1/s | Read: Xu et al. 2018, 10 years of shelf life, so 20 % over 10 years | ✅ |
| `E_activation` | 20 | kJ/mol | TUNED: Arrhenius | ⚠️ |
| `fade_price` | 1500 | $/kWh | Read: Xu et al. 2017 and 2018, $300/kWh to replace the cells, over the 20 % fade that ends a pack's life. That is a 2017 price; today's cells are cheaper, which would make following deeper worthwhile | ✅ |
| `dispatch_block_seconds` | 3600 | s | Ours: the hour regulation markets clear in | ⚠️ |
| `dispatch_range` | +-0.8 MW | W | Ours: leaves the regulation movement room inside 1 MW | ⚠️ |
| `dispatch_noise_std`, `regulation_tau` | 10 kW, 900 s | – | Ours: a movement of about 95 kW sd and a small innovation. A real signal comes every 2 to 6 s (Xu et al. 2017), so the 5 s step is one instruction | ⚠️ |
| `max_steps_in_episode` | 2880 | steps | Ours: 4 h, chosen so exact following leaves the window on most seeds (3 h: 51 %, 4 h: 62 %) | ⚠️ |
| `delta_t` | 5.0 | s | One regulation instruction | ✅ |

**Actuator and timing figures.** The inverter has no rate limit or lag here, labelled
ours. MESA-1's response was faster than its 10 s logging could resolve, which is the
only figure read, and an inverter response inside the 5 s step is the assumption. The
meter has no dead time, on the same assumption, because telemetry runs on the signal's
2 to 6 s cycle.

---

## 4. Task design

### Task: `battery`

Follow a regulation signal for 4 hours with a 1 MW / 1.69 MWh lithium-ion battery.
The signal is an hourly level plus a regulation movement, and following it exactly
runs out of energy on most seeds. Leaving the charge or cell-voltage window trips the
pack, and deep cycles cost more wear per MWh than shallow ones, so how far to follow
is a real decision.

| obs | meaning |
|---|---|
| 0 | `soc`, state of charge (0 to 1) |
| 1 | cell voltage (V) |
| 2 | pack temperature (°C) |
| 3 | power at the meter (MW), positive when discharging |
| 4 | target power (MW) |
| 5 | the next block's level (MW) |
| 6 | minutes until it goes live |

**The window binds.** From a state of charge drawn in 0.35 to 0.75, four hourly
levels of up to 0.8 MW move up to 2.8 times the pack's capacity. Following them
exactly left the charge window on 62 % of seeds in calibration, and the cell's
voltage limit comes sooner: at rated current it trips near 8 to 10 % charge, and near
85 % on charge.

**The signal.** Each hourly level is drawn uniformly in +-0.8 MW. On top goes an
AR(1) movement with an innovation of 10 kW and a correlation time of 900 s, drawn
after the action. Levels are published ahead, so a controller reads the next one and
how far off it is. That took away most of the oracle's old lead, which came from
reading the next level from the hidden state.

**Observation.** What a battery management system and a market feed report. The
diffusion voltage, the accumulated fade and the half-cycle under way are hidden.

**Reward** = -(imbalance + wear + trip cost), in dollars per step (version 2, below).
The whole-episode cost of following is mostly wear. A half-cycle's next MWh costs
`R dPhi/dd / 2`, which passes the $100/MWh imbalance at a depth of 0.64: there,
giving up the imbalance is cheaper than the wear. Calendar ageing is left out of the
charge because no controller avoids it.

---

## 5. Baselines

Measured on the held-out seeds 200 to 209 (2026-11-06), before any re-record.

| controller | cost per episode | imbalance | wear | trips |
|---|---|---|---|---|
| expert | $80.8 | $42.4 | $38.4 | 0 |
| stand-in oracle | $79.5 | $38.1 | $41.4 | 0 |

The null policies, on the same seeds (`tests/test_null_policies.py`), as the expert's
cost over theirs:

| policy | ratio | seeds won |
|---|---|---|
| echo of the target | 0.68 | 2 of 10 |
| table of the target | 0.61 | 0 of 10 |
| table with an I-only trim | 0.16 | 0 of 10 |
| constant, middle of the range | 0.47 | 0 of 10 |
| constant, either bound | 0.03 | 0 of 10 |

These nulls have no idea of the pack's energy, so they trip. That is the finding the
rebuild was for: the echo used to win 10 of 10.

**The expert** is what a commercial controller does (`experts.py`): a feedforward of
the signal, a PI trim on the meter, and hard limits on state of charge. On a block's
last step the feedforward is the next block. The guard fades the demand to zero in
the direction that would breach a limit, over 0.12 of charge, at 0.10 and 0.90,
which is 5 % inside the trips as the MESA-1 report advises. The integrator stops
while the guard sheds. Its gains predate the rebuild, and PR 9 writes its protocol
afresh.

**The oracle is a stand-in** until the shared solver replaces it (#141, PR 9). It
reads the true state, commands the signal's next value, and stops following at the
bound of Xu et al. 2017, Algorithm 1, a depth of 0.64 on these prices, or at the
charge and voltage limits. That is an online policy, so it leads by only 2 % and wins
8 of 10 seeds. It does not yet meet the oracle rule's win on every seed.

---

## 6. Known deviations

**⚠️ D1: no published reference system.** Unlike the wind turbine, the parameters
are sized to reproduce published *behaviour* rather than taken from a specific
documented machine. The efficiency, voltage window and thermal rise are right; the
particular cell chemistry is generic NMC-like.

**⚠️ D2: single RC branch.** Real packs show relaxation over several timescales. One
branch captures the dominant minute-scale polarisation and misses both faster and
slower components.

**⚠️ D3: depth counts a half-cycle at a time.** The racetrack filter is not full
rain-flow counting: a large cycle broken by a swing of more than 0.05 counts as
two shallower ones, which understates it. The ripple term also runs on the deep
half-cycles' throughput, which overstates a 0.8 half-cycle by about 6 %.

**⚠️ D4: no cell imbalance.** A lumped pack cannot represent the weakest-cell
behaviour that actually determines real pack limits.

**⚠️ D5: the signal is synthetic.** Its structure is real and its numbers are ours.
A published regulation signal may not be energy-limited: PJM designed RegD to be
energy neutral ([FERC, ER19-1651](https://ferc.gov/sites/default/files/2020-06/20200326134642-ER19-1651-000.pdf)).
This task's hourly levels are what make it energy-limited.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `battery` | 2880 | 5 | 4.0 h | 1 in [-1, 1] | 7 | 18 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2), in dollars per step

`compute_reward = -(tracking + running + failure)`, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 7979 W | closed form: the signal moves after the action by an innovation of sd 10 kW, so no causal controller holds E\|error\| below sd * sqrt(2/pi) |
| `e_tol` | 0 | none |
| `tracking_exponent` | 1 | linear imbalance |
| `imbalance_price` | 100 $/MWh | ours: the one imbalance price of the priced plants, and inside the $10 to $200/MWh mismatch prices of Xu et al. 2017 |
| `c_hold` | 3.17e-9 | `k_calendar * delta_t`: a step's calendar fade, which no controller avoids, hence charged only above it |
| `fade_price` | 1500 $/kWh | $300/kWh of cells over the 20 % fade that ends a pack's life (section 3) |
| `pack_kWh` | 1692 | capacity_As x OCV at 50% SOC / 3.6e6 |
| `failure_cost` | $1.50 per step | twice the dearest step that survives: $0.25 of imbalance at the reachable 1.8 MW, $0.20 of wear at full depth and full power, $0.30 for a half-cycle's turn |
| `restart_steps` | 720 (1 h) | restart time priced into a trip, `restart_steps x failure_cost` (a protection trip's reset; provisional); where a plant engineer would get it: the plant's restart procedure |

The version-1 SOC-comfort term stays dropped: depth now carries that price.

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

## Readings

The observation reports state of charge, cell voltage, temperature and meter power
through their instruments (`docs/proposals/measurement-noise.md`). The signal, the
next block and the time to it stay exact. `sensor_noise_scale = 0` gives the noise-free task,
rounding included.

| reading | sigma | step | instrument | status | source |
| --- | --- | --- | --- | --- | --- |
| `soc` | 4.7e-5 | | BMS estimator on TI BQ79616 cell data | Not checked | [BQ79616](https://www.ti.com/lit/ds/symlink/bq79616.pdf) |
| `V_cell` | 5.5e-5 V | | TI BQ79616 cell monitor | Not checked | [BQ79616](https://www.ti.com/lit/ds/symlink/bq79616.pdf) |
| `T_cell` | 0.03 K | 0.003 K | 10 kOhm NTC on a BQ79616 GPIO input | Not checked | [BQ79616](https://www.ti.com/lit/ds/symlink/bq79616.pdf) |
| `power` | 330 W | | SEL-735 revenue meter | Confirmed | [SEL-735](https://selinc.com/products/735/) |

**Weak figures.**

- The refutation pass reached only `power`. That figure reads the class 0.1S limit
  at 1 MW as 3 sigma, which the page does not state, and it scales with reading,
  so it is an upper bound below full power.
- `soc` is an estimate, not a sensor: voltage scatter over an OCV slope of 1.16 V
  per unit. Real state-of-charge error is a slow bias, left out as all bias is.
- `V_cell` has no noise figure. The sheet's ENOB of 16 at one LSB gives
  LSB/sqrt(12) = 55 uV, which is already a quantisation figure, so no rounding goes
  on top of it: `sigma` holds the 190.73 uV LSB.
- `T_cell` assumes a divider slope of 50 mV/K near 25 C, which also turns the
  sheet's 152.59 uV GPIO step into 0.003 K.

Cell voltage is computed from current, charge and the hidden diffusion voltage, so
it is a channel reading: it inherits nothing from `soc`.

The expert reads these values, and so does every tuner. The oracle plans and acts
on the true state, so it stays a full-state bound. `reward`, trips and the score
read the true state. `info["measured_reward"]` is the same cost on the readings,
with any trip taken from the true state. Trips read the true cell voltage.
