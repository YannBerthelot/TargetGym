# Nuclear reactor: physics model, provenance and validation

Reference plant: a **3 GW thermal pressurised-water reactor**, modelled with point
kinetics, six delayed-neutron groups, xenon-135 poisoning, thermal feedback, a steam
generator driven by the turbine, and rate-limited control rods.

Contract for `target_gym.reactor`. Method: `docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 0. The rebuilt control problem

A rebuild (`docs/proposals/industrial-fixes.md`, PR 6).

The task used to ask a reactor to track a power demand with its rods, on a plant
with no secondary side, a fixed inlet temperature and a fixed flow. One rod integrator
matched the full-state oracle, because power was all there was to hold.

The rebuilt task asks **how to hold Tavg on its program through a 100-50-100 day.**
With a steam generator in the loop, the turbine sets the heat leaving the primary,
and energy balance then fixes steady power at the load wherever the rods are. Rods
200 pcm apart leave `n` at 0.8000 and move Tavg by 6 K. So the rods' job is Tavg,
which is what a PWR's rod control does, and what the tracked output now is. The owner
chose that on 2026-11-06, with the 30 s step in section 3, over keeping `n` scored.

Three things make that a control problem:

1. **Xenon is a large, slow, unmeasured disturbance.** It moves the rods' balance by
   hundreds of pcm through the day.
2. **The bank has a stop.** After a hold of about 3 to 8 h, the return asks for more
   than the 1500 pcm bank has, so Tavg sags.
3. **Rod travel costs money.** Holding tighter costs steps, which is what a dead band
   is for.

---

## 1. Model and provenance

**Point kinetics** with the standard six-group U-235 thermal delayed-neutron data of
Keepin (1965):

```
dn/dt   = ((rho - beta) / Lambda) n + sum_i lambda_i C_i
dC_i/dt = (beta_i / Lambda) n - lambda_i C_i          i = 1..6
```

Reactivity is the sum of the rod, the thermal feedbacks and the xenon term:

```
rho = rho_ext
    + alpha_fuel    (T_fuel - T_fuel_ref)
    + alpha_coolant (T_avg  - T_coolant_ref)
    - rho_Xe_full   (Xe_hat - 1)
```

**Thermal model.** Fuel, a hot-leg node holding the core, a cold-leg node holding the
steam generator tubes, and the steam generator's secondary:

```
C_fuel    dT_fuel/dt = P_ref n - UA (T_fuel - T_hot)
C_coolant dT_hot/dt  = UA (T_fuel - T_hot) - m_dot_cp (T_hot - T_cold)
C_cold    dT_cold/dt = m_dot_cp (T_hot - T_cold) - UA_sg (T_avg - T_sg)
C_sg      dT_sg/dt   = UA_sg (T_avg - T_sg) - P_ref load
T_avg = (T_hot + T_cold) / 2
```

The turbine takes `P_ref load` from the secondary. A cut in load warms the secondary,
then the cold leg, and the moderator feedback brings the reactor down with it, with
no rod moving. That is how a PWR follows its turbine.

**Xenon-135** is tracked through its iodine parent in normalised units, so
equilibrium at full power is `I_hat = Xe_hat = 1`:

```
dI_hat/dt  = lambda_I (n - I_hat)
dXe_hat/dt = a n + b I_hat - (lambda_Xe + sigma_phi0 n) Xe_hat
```

Integration is **TR-BDF2 on the fast block** (neutrons plus precursors) with the
thermal and fission-product states advanced explicitly. One control step is 6
sub-steps of 5 s, each of 4 inner solves. Those 4 are needed: at 2 solves, an inner
step of 2.5 s misses the prompt jump, and inserting rods raises power. Over a full
day of rod program, this agrees with 0.5 s sub-steps to a mean of 0.07 K in Tavg and
to 4 pcm in 3086 pcm of rod travel.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Spatial neutronics (flux tilt, axial offset, xenon oscillations) | Point kinetics is a zero-dimensional reduction. Real load follow is limited by axial offset, which this cannot show. |
| Samarium-149 and other poisons | Xenon dominates the short-term reactivity balance. |
| Burn-up and fuel depletion | Episodes are 12 h; depletion acts over months. |
| Boron / chemical shim | Real PWRs trim reactivity with boron. The EUR asks for load following without adjusting it (Lokhov 2012), and here the rods do everything. |
| Primary-loop transport delay | The two coolant nodes smear it. |
| Turbine, condenser, steam pressure | The turbine is a heat sink at the load. |
| Decay heat | Power follows the neutron flux, so a scram removes all heat instantly. |

**Regime of validity.** Power manoeuvring between 50 % and 100 % of rated, with
reactivity well below prompt critical. Not valid for startup, shutdown transients, or
any accident scenario.

---

## 2. Validation targets

| Quantity | Target | Model | |
|---|---|---|---|
| Total delayed fraction beta | 0.0065 (U-235 thermal) | **0.00650** | ✅ |
| Neutron generation time Lambda | 1e-5 to 1e-4 s (PWR) | 1e-4 | ✅ |
| Doppler coefficient | -2 to -4 pcm/K | **-3.0** | ✅ |
| Moderator coefficient | -10 to -50 pcm/K | **-30** | ⚠️ |
| I-135 half-life | 6.57 h | **6.57 h** | ✅ |
| Xe-135 half-life | 9.14 h | **9.13 h** | ✅ |
| Stable period at +100 pcm (inhour) | tens of seconds | **55 s** | ✅ |
| Maximum rod worth vs prompt critical | must stay < beta | **0.77 beta** | ✅ |
| Steady power at a load, any rod position | the load | **0.8000 at 0.8** | ✅ |
| Tavg shift per rod move at fixed power | `d rho / |alpha_f + alpha_m|` | **3.03 K per 100 pcm** | ✅ |
| Rod speed, either way | 72 steps/min: 7.89 pcm/s | **7.89** | ✅ |
| Coolant rise across the core | 30 to 40 K | **33.3 K** | ✅ |
| Fuel temperature at full power | below the 1473 K trip | **1051 K** | ✅ |
| Xenon peak after a scram | 9 to 11 h | 8.3 h | ⚠️ D2 |
| Peak xenon worth after a scram | about 2000 to 3000 pcm | **-2213 pcm** | ✅ |

The **inhour check** ties Lambda and all twelve delayed-neutron constants together
through the dispersion relation, and a reactor period is the quantity an operator
would recognise.

The **reactivity budget** is the second. With Tavg on program, the rods have to
cancel the power defect and the xenon, and the worth required is an output of the
model:

| load | Tavg | feedback, from full power | rods needed, at full-power xenon |
|---|---|---|---|
| 1.00 | 580.2 K | 0 | -38 pcm |
| 0.75 | 576.4 K | +480 pcm | -518 pcm |
| 0.50 | 572.5 K | +961 pcm | -999 pcm |

The bank spans -1000 to +500 pcm, so the rods sit about 40 pcm in at full power, with
540 above them, and 50 % load fits at the bottom. Xenon moves that through the day.
A return to full power after a hold asks for more than the full-power position by:

| hold | 2 h | 3 h | 4 h | 5 h | 6 h | 7 h | 8 h | 9 h |
|---|---|---|---|---|---|---|---|---|
| extra rods at the return (pcm) | 473 | 576 | 630 | 647 | 638 | 610 | 569 | 519 |

So holds of about 3 to 8 h run out of bank, by up to about 100 pcm or 3 K of Tavg
until the xenon burns off, and shorter or longer ones do not. With only 40 pcm spare
at full power, as this plant once had, every policy sagged 18 K on every return.

---

## 3. Parameter table

Status of each source: **read** means two readers opened the page and found the
value. **earlier** means a previous session read it from the NRC's manual, whose PDFs
now refuse automated readers, so it was not re-read here. **ours** means a choice for
the owner to accept, with what is behind it.

| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `beta_i`, `lambda_i` | 6-group | – | Keepin (1965), U-235 thermal | ✅ |
| `Lambda_gen` | 1e-4 | s | PWR range | ✅ |
| `alpha_fuel` | -3.0e-5 | 1/K | Doppler, PWR range | ✅ |
| `alpha_coolant` | -30e-5 | 1/K | Ours: inside the -10 to -50 pcm/K range this page used, where the old -5 sat outside. Plant limits seen only in search results run near -80 pcm/K at end of cycle | ⚠️ |
| `T_fuel_ref` | 1064 | K | Ours: sets the rods about 40 pcm in at full power, a mechanical-shim position for load follow without boron | ⚠️ |
| `rho_ext_min/max` | -0.010 / +0.005 | – | A 1500 pcm bank, below prompt critical | ⚠️ |
| `rod_speed` | 7.89e-5 | 1/s | Earlier: 72 steps/min over the 228-step stroke (NRC HRTD 8.1, NUREG/CR-5555), one speed both ways. It replaces 40 and 20 pcm/s, which were 5 and 2.5 times that | ✅ |
| `P_thermal_ref` | 3.0 | GW | Large PWR | ✅ |
| `C_fuel`, `UA` | 3.3e7 J/K, 6.6e6 W/K | – | About 100 t of UO2; gives a 5 s fuel time constant | ⚠️ |
| `m_dot_cp` | 9.0e7 | W/K | Gives the 33 K core rise | ✅ |
| `C_coolant`, `C_cold` | 4.7e8, 9.3e8 | J/K | Ours: about 250 t of primary water at 5.6 kJ/kg/K, split a third and two thirds. No open source for the volume was found | ⚠️ |
| `UA_sg` | 1.36e8 | W/K | Ours: 22 K from Tavg to the steam at full load | ⚠️ |
| `C_sg` | 1.0e9 | J/K | Ours: secondary water and tube metal, about 7 s against `UA_sg` | ⚠️ |
| `T_avg_no_load`, `T_avg_full_load` | 564.8, 580.2 | K | Earlier: 557 to 584.7 F, NRC HRTD 5.0 | ✅ |
| `sigma_phi0` | 7.95e-5 | 1/s | sigma_a 2.65e-18 cm2 x Phi of about 3e13 n/cm2/s | ✅ |
| `gamma_ratio` | 0.049 | – | Direct Xe yield / I yield (0.003 / 0.061) | ✅ |
| `rho_Xe_full` | 0.025 | – | Equilibrium xenon worth, 2500 pcm | ✅ |
| `n_max` | 1.09 | – | Ours: the high-flux trip as remembered, 109 %. Neither reader could open a source | ⚠️ |
| `T_avg_min` | 556 | K | Ours: 541 F, the minimum temperature for criticality as remembered. Neither reader could open a source | ⚠️ |
| `load_ramp_rate` | 5 %/min | – | Read: the EPRI URD column of Table 1 in [Chari and Rohr, 1997](https://proceedings.cns-snc.ca/index.php/pcns/article/download/3245/3244/3280). The URD itself is not open, so this is a vendor's quotation of it. The EUR's standard rate is lower, at 3 %/min ([Lokhov, 2012](https://www.neimagazine.com/features/featureload-following-capabilities-of-npps)) | ✅ |
| `load_min` | 0.5 | – | Read: the URD's 100-50-100 daily cycle, same table | ✅ |
| `hold_range` | 3 to 8 h | s | Inside the URD's 2 to 10 h, same paper, so the return falls inside 12 h | ✅ |
| `ramp_down_range` | 0.5 to 2 h | s | Ours | ⚠️ |
| `demand_theta`, `demand_sigma` | 300 s, 0.5 % sd | – | Ours: a small frequency-control wobble | ⚠️ |
| `delta_t` | 5.0 | s | Physics sub-step; 6 per env step, so 1440 env steps = 12 h | ✅ |

**Actuator and timing figures.** The rate limit is the jack speed above, and the
rods have no lag on top. The rods are read back exactly, by counting steps. The RTDs
and ion chambers have no dead time or lag here, labelled ours: their response is
taken to sit inside the 30 s step, with no figure opened.

**Why 30 s and not the plan's 60 s.** The program's slowest speed is 8 steps/min. Over
60 s that is 8 steps, 53 pcm, about 1.8 K, more than the 1.5 F dead band, so the rods
limit-cycle: 172,300 pcm of travel in a day against 2100 at 30 s, 20 s or 10 s.

---

## 4. Task design

### Task: `reactor`

Hold a 3 GW pressurised-water reactor's average coolant temperature on its program
through a day of load follow, using the control rods. The turbine goes from full load
to half and back at 5 %/min, power follows it by itself, and xenon moves the rods'
balance by hundreds of pcm. Rod travel wears the mechanism, and costs are in dollars.

| obs | meaning |
|---|---|
| 0 | `n`, neutron power relative to nominal (1 is full power) |
| 1 | hot-leg temperature (K) |
| 2 | rod position, normalised to [-1, 1] |
| 3 | turbine load, relative to rated |
| 4 | `T_avg`, the mean of the hot and cold legs (K) |
| 5 | `T_ref`, the program at this load (K) |

**The day.** Each episode starts at full load on program, with xenon at the
equilibrium of a recent power of 0.9 to 1.0. The load starts down between 0.5 and 2 h
in, holds at 50 % for 3 to 8 h and comes back, each ramp at 5 %/min, with a small
frequency-control wobble on top. The wobble is two-sided at full load as well, so the
load can pass 1 by a fraction of a percent. Each ramp is one move to the scorer, so a
day has three cycles, and a ramp lasts 20 steps.

**Why it is hard.**

1. **Xenon is hidden and moves by hundreds of pcm.** Nobody measures it, and it
   drifts through the whole hold.
2. **The bank runs out** after a mid-length hold (section 2).
3. **Tight holding costs rod travel,** and the mechanism has a finite life.
4. **Four timescales at once**: prompt neutrons (ms), precursors (s to min),
   thermal (s to min), xenon (hours).
5. **Most of the state is hidden.** The six precursors, fuel temperature, the steam
   generator, iodine and xenon are all unobserved.

**Reward** = -(Tavg off program + rod travel + trip cost), in dollars.

---

## 5. Known deviations

**⚠️ D1: the load goes straight to heat.** The turbine is a heat sink at the load.
Steam pressure, valve limits and the secondary's own control are left out, so a sag in
Tavg costs no megawatts here, only its price in the reward.

**⚠️ D2: the xenon peak is early and shallow.** 8.3 h after a full scram at 1.89 times
equilibrium, against a documented 9 to 11 h. The peak *worth* (-2213 pcm) lands in the
right band, so the reactivity consequence is realistic even though the timing is
roughly an hour early.

**⚠️ D3: rods only.** Real plants use boron for much of a load cycle (Chari and Rohr
1997), and the EUR's rods-only load following relies on grey rods this model does not
have. Here the one bank carries the whole swing, and it has to sit inserted at full
power to have the room.

**⚠️ D4: the coolant is two lumps.** Transport delay around the loop is smeared, so a
fast change at the steam generator reaches the core slightly early.

**❌ D5: no decay heat.** Power follows the neutron flux exactly, so a scram removes
all heat immediately. A real reactor still produces about 7 % of rated power seconds
after shutdown, decaying over hours.

---

## 6. The expert: Westinghouse rod control on average temperature

The expert is a PWR's automatic rod control, as the NRC's Westinghouse Technology
training manual describes it (HRTD 8.1, rod control; HRTD 5.0, the Tavg program and
the rate/lag unit). It reads the observation only (`experts.py`, `rod_control_step`),
and its program is published, so nothing is tuned on seeds.

- **Temperature channel.** Tavg against the program.
- **Power-mismatch channel.** A rate/lag `40 s / (1 + 40 s)` on (load - power) in %
  power, sampled exactly at the 30 s step. A nonlinear gain turns it into deg F: 0.3 F
  per % up to 1 %, 1.5 F per % beyond. A variable gain multiplies it by 2 up to 50 %
  load and by 1 / load above. The channel only acts on changes, so a steady mismatch
  is left to the temperature channel.
- **Rod speed.** On the total error in deg F: no motion inside +-1.5 F from rest;
  moving rods stop only at 1.0 F (lockup); 8 steps/min to 3 F, then 32 steps/min per F
  up to 72 steps/min from 5 F. A positive error withdraws.

Its protocol is PR 9's, written afresh on this plant.

### Results

Measured on the held-out seeds 200 to 209 (2026-11-06), before any re-record.

| policy | ratio to the expert | seeds won | trips | Tavg cost | rod cost |
|---|---|---|---|---|---|
| expert | 1 | | 0 | $1163 | $480 |
| oracle, 4 seeds | 1.8 | 4 of 4 | 0 | $206 | $755 |
| a plain integrator on Tavg, the gate's witness | **1.17** | **10 of 10** | 0 | $798 | $598 |
| table with an I-only trim on Tavg, SIMC gain | 0.72 | 0 of 10 | 0 | $885 | $1407 |
| the review's rod integrator, as written | 0.15 | 0 of 10 | 0 | $10,534 | $712 |
| table of the target | 0.11 | 0 of 10 | 0 | | |
| constant, middle of the range | 0.04 | 0 of 10 | 0 | | |
| the review's law, at its per-second gain | 0.04 | 0 of 10 | 2 | | |

The review's law stopped winning because of the plant: it holds energy, which keeps
Tavg near its starting value and off program at part load. The oracle's row is the
`NewtonMPC` that held this slot, on seeds 200 to 203.

The oracle is now the shared solver (`oracle.py`): the env's own stiff step, sub-step
by sub-step, over the 5 minutes the horizon protocol gives, P = 10 steps with a ramp
of 2. The rod demand stays within the rods' reach, the corner left inside it is
smoothed at the K5 width, each sub-step's Tavg error and a step's rod travel are
epigraph terms, and the trips are hard limits on every sub-step's state. It reads the
true state, xenon and the day's schedule included, and plans the load's mean path.

**An integrator on Tavg beats the rod program.** The dead band costs $365 of Tavg
error to save $119 of rod travel, at the $1 a step labelled ours above. The two break
even at about $3 a step, between $1.9 and $5.8 by seed. Either the rod price is too
low, or a controller tighter than the Westinghouse program is simply better on this
reward. That is for the owner and PR 9, and the gate shows it as a strict xfail with
this witness in place. The price was not moved to make it pass.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `reactor` | 1440 | 30 | 12.0 h | 1 in [-1, 1] | 6 | 23 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2), in dollars per 30 s control step

`compute_reward = -(tracking + running + failure)`; each physics sub-step charges its 5 s and `step_env` sums the six, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 0.05 K | the RTD reading's resolution, about its 0.045 K scatter: a documented minimum, not a certified floor |
| `e_tol` | 0 | none. The rod program's dead band is the controller's, not the reward's |
| `tracking_exponent` | 1 | linear, as lost output is |
| `imbalance_multiple` | 1.25 x 0.0023 per K of `spot_price_per_MWh` on `P_electric_GW` | **ours.** Off program, steam conditions cost output. The relative Carnot slope at 547 K steam and 305 K condenser is 305 / (547 x 242) = 0.23 %/K. At the one imbalance price of the priced plants, $100/MWh (1.25 x the $80 spot), on `P_electric_GW` = 1.0 GW, a third of the 3 GW thermal, that is 100 x 1000 x 0.0023 = $230/h per K, so **$1.917 per K over a 30 s step**. It prices a hot plant as a cold one, which the slope does not say |
| `rod_step_price` | $1 per step | **ours.** Rod travel, as the steps the rods actually moved. Mechanism lives of 6 to 9 million steps turned up in search results, but neither reader could open one |
| `failure_cost` | 87.34 $ per step | twice the dearest step that survives: 40 K off program, $7.67, and 36 rod steps, $36, per 30 s |
| `restart_steps` | 5760 (48 h) | restart time priced into a trip, `restart_steps x failure_cost` (a SCRAM's xenon-limited restart; provisional); where a plant engineer would get it: the plant's restart procedure |

A step at the floor costs `rho_floor_tracking`, $0.0958: the price per K over the 30
s step, times `e_floor`.

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

The observation reports power and the coolant temperatures through their instruments
(`docs/proposals/measurement-noise.md`). Rod position, the load and Tref stay exact:
the rods are the controller's own rate-limited command, known by counting steps.
`sensor_noise_scale = 0` gives the noise-free task. Tavg is the mean of the two RTDs,
so it scatters by 0.045 K over sqrt(2), and shares a draw with the hot-leg channel.

| reading | sigma | step | instrument | status | source |
| --- | --- | --- | --- | --- | --- |
| `n` | 1.9e-4 of rated | | Ex-core power-range ionization chambers | Corrected | [OSTI 7260785](https://www.osti.gov/servlets/purl/7260785) |
| `T_coolant`, `T_cold` | 0.045 K | | Pt100 RTD with TMT82-class transmitter | Corrected | [TMT82](https://bdih-download.endress.com/file/2c4c2c5496b6c81881df53d319b2e2f5/TI01010TDE_2725-00.pdf) |

**Weak figures.** Both are bounds, from opposite sides.

- `n` is an upper bound. The paper's broadband 5.5 and 19 mil are mostly real
  flux common to all channels, driven by rod actions and a feedwater controller,
  so the refuter kept only its 2 to 15 Hz band, at most 1.9 mil at 0.01 % per
  mil. That scatter is a fraction of signal, so it is proportional to `n`. A
  reading's sigma is a constant, so this task uses the figure at full power at
  every power. It overstates the bound by up to 2 times at the lowest load of 0.5,
  and understates it only above full power.
- `T_coolant` is a lower bound. It covers the transmitter alone, at 307 C. Hot-leg
  streaming is real and likely larger, and no figure for it was opened. NRC
  documents on power-range accuracy could not be read.

The expert reads these values, and so does every tuner. The oracle plans and acts
on the true state, so it stays a full-state bound. `reward`, trips and the score
read the true state. `step_env` sums `info["measured_reward"]` over the six
sub-steps as it does `reward`. Each sub-step's true error is shifted by the draw
in the observation that step returns, because a controller gets one reading per
step, and any trip comes from the true state. The cost is linear, so noise has no
floor of the proposal's quadratic form here.
