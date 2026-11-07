# Nuclear reactor: physics model, provenance and validation

Reference plant: a **3 GW thermal pressurised-water reactor** with a two-point core:
iodine and xenon in the top and bottom halves, an axial offset, rods that enter from
the top, soluble boron that reaches the core late, thermal feedback, and a steam
generator driven by the turbine.

Contract for `target_gym.reactor`. Design notes and measured results:
`docs/proposals/reactor-axial.md`.

Each number is **read** (opened and read twice), **derived** (computed from read
numbers, the sum shown) or **ours** (a choice, with its reason).

---

## 0. The control property

**Axial offset.** Rods act in seconds but disturb the power's axial shape, and that
shape drives a xenon oscillation near 29 h long. Boron leaves the shape alone but
reaches the core 5 to 15 minutes late. A plant running Mode A depends on both, which
is why OAPS names the operator's struggle with the offset as the real limit on its
load following.

Tavg is held by reactivity from either actuator. Delta I is moved only by the rods.
So the rods are spent on the offset, and boron carries Tavg through the ramps, late.

---

## 1. Model and provenance

**Total power** is point kinetics with Keepin's six delayed groups:

```
dn/dt   = ((rho - beta) / Lambda) n + sum_i lambda_i C_i
dC_i/dt = (beta_i / Lambda) n - lambda_i C_i          i = 1..6
```

**Reactivity:**

```
rho = rod_worth(y) - rod_worth(y_park) + boron_worth (boron - boron_ref)
    + alpha_fuel (T_fuel - T_fuel_ref) + alpha_coolant (T_avg - T_coolant_ref)
    - rho_Xe_full (Xe_hat - 1)
rod_worth(y) = -1500 pcm (y - sin(2 pi y) / (2 pi))
```

`y` is how far in the rods are, as a fraction of the stroke. The curve is the
integral worth of a cosine flux, so a shallow bank is worth little.

**The two-point core** is the Onega and Kisner model, as Mastroianni's 1979 thesis
prints it (eqs 2.15 to 2.18). Flux, iodine and xenon are each a cosine plus a first
harmonic, `psi = n (cos bz + A sin 2bz)`, with amplitudes found by integrating over
the two halves of the core. `A` is then exactly (top - bottom) / (top + bottom), the
axial offset, and each half is `mean +- tilt`:

```
dI_hat/dt  = lambda_I (n - I_hat)
dI_tilt/dt = lambda_I (n A - I_tilt)
dXe_hat/dt = a (r n + I_hat) - (lambda_Xe + s n (1 + A Xe_tilt / Xe_hat)) Xe_hat
dXe_tilt/dt = a (r n A + I_tilt) - lambda_Xe Xe_tilt - (8 / 3 pi) s n (A Xe_hat + Xe_tilt)

A = -(8 b + 10 eps) / (N + sqrt(N^2 + 64 b (b + 5/4 eps)))
b = S Xe_tilt
N = 45 pi^3 D / (8 H^2) - S Xe_hat + 7 n Phi alpha nu_Sigma_f
```

Here `s` is `sigma_phi0`, `r` is `gamma_ratio`, `a = (lambda_Xe + s) / (1 + r)` and
`S` is the full-power xenon's absorption, `sigma_x X0`. With `eps = 0`, `Xe_hat = 1`
and `n = 1` this is the thesis's `A = v - sqrt(v^2 + 1)`, `v = N / (8 S Xe_tilt)`.

**What we added to the thesis, by its own method.** Projecting the diffusion equation
onto the first harmonic gives the thesis's `v`, down to `D`, `S` and the denominator,
so the same projection places an absorption difference `eps` between the halves. A
term of 0.25 % of absorption, the thesis's stand-in for the rod move, then leaves a
tilt of 0.19 once it ends, which is the plant's reading. Two sources feed `eps`:

```
eps = -0.5 rho_ext_min nu_Sigma_f sin(pi y)^3                     rods from the top
    + alpha_coolant nu_Sigma_f (rise_full - (T_hot - T_cold)) / 4  moderator
```

The rods' weight is their overlap with the harmonic: nothing when out, and nothing
all the way in. The moderator's term is the coolant's rise up the core: the two
node means sit a quarter of the rise either side of Tavg, and a smaller rise at part
load moves power up the core, the direction Krsko describes. Boron enters no `eps`:
it is uniform, and the APR1400 paper finds its effect on axial shape negligible.

**Boron** is a feed and bleed of well-mixed coolant. A command joins a queue, and
reaches the core `boron_delay` control steps later:

```
d(boron)/dt = Q / M (C_feed - boron)
Q = 10 kg/s dilution (C_feed 0), 3 kg/s boration (C_feed boric_acid_ppm)
```

The heat path is unchanged: fuel, hot leg, cold leg, steam generator, with the
turbine taking `P_ref load`. Energy balance fixes steady power at the load. Thermal
states and xenon step explicitly, neutrons by TR-BDF2, 4 solves in each of 6 sub-steps
of 5 s.

**Left out:** the second axial harmonic and radial modes, grey rods, boron's effect
on the moderator coefficient, burn-up, samarium, decay heat, loop transport delay.

---

## 2. Validation targets

| Quantity | Source figure | Model | |
|---|---|---|---|
| Axial mode period, full power | 28.3 h (Ringhals-4), about 28.5 h (Oconee II, fig. III-1) | **29.4 h** | yes |
| Axial mode decay, full power | 0.05 per hour (Ringhals-4) | **0.035 per hour** | near |
| Undamped xenon cross section | 3.02e-18 cm2 (thesis, section 3.2) | 2.92e-18 | yes |
| Tilt after the Oconee pulse | 0.19 (fig. III-1) | **0.19 +- 0.04** | yes |
| Rods 34 steps in at 50 %, xenon at 110 of 96 | delta I near +5 % (Krsko figs 3, 5, 6) | **+10 to +13 %** | sign yes, size twice |
| Rate that full flows allow at 1690 ppm | 1 to 2 %/min (OAPS) | **about 2 %/min** | yes |
| Rods steps for delta I = 0 at 50 % | 34 steps (Krsko fig. 3) | **42 steps** | yes |
| Inhour at +100 pcm | tens of seconds | 55 s | yes |
| Coolant rise | 30 to 40 K | 33.3 K | yes |
| Peak xenon worth after a scram | 1800 to 3200 pcm, test band | **1419 pcm** | **no** |

**The peak xenon miss.** The two-point model's flux, 2.69e13 with its pi / 4, burns
xenon at 5.75e-5 per second, against the 7.95e-5 this page used before. The flux is
tied to the period, so it stayed, and the old test fails on it. Reported, not moved.

---

## 3. Parameter table

| Symbol | Value | Unit | Source | Status |
|---|---|---|---|---|
| `beta_i`, `lambda_i` | 6-group | | Keepin (1965) | earlier |
| `Lambda_gen` | 1e-4 | s | PWR range | earlier |
| `alpha_fuel` | -3.0e-5 | 1/K | Doppler, PWR range | earlier |
| `alpha_coolant` | -30e-5 | 1/K | Inside -10 to -50 pcm/K, no figure opened | ours |
| `T_fuel_ref` | 1051.4 | K | Zero feedback at full power on program | derived |
| `rho_ext_min` | -0.015 | | A 1500 pcm bank, kept from the earlier plant, no figure opened | ours |
| `rod_speed` | 72 / 60 / 228 | 1/s | 72 steps/min over 228 steps, NRC HRTD 8.1, read by an earlier session | earlier |
| `rod_park` | 3 / 228 | | D bank at 225 steps, Krsko fig. 3 | read |
| `boron_ref` | 1690 | ppm | Krsko fig. 3 | read |
| `boron_worth` | -8e-5 | 1/ppm | Two NRC papers gave -6.73 and -8.15 pcm/ppm in search summaries, neither would open | ours |
| `dilution_flow`, `boration_flow` | 10, 3 | kg/s | OAPS, section 2.1 | read |
| `boron_delay_range` | 10 to 30 | steps | 5 to 15 min, OAPS, section 2.1, drawn at reset | read |
| `boric_acid_ppm` | 7800 | ppm | A vendor's recycling concentrate, seen in a search summary, the PDF would not open | ours |
| `primary_mass` | 2.5e5 | kg | The water behind `C_coolant` and `C_cold`. 60 ppm of 1690 then takes 9 t, against Krsko fig. 4's 6 t | ours |
| `sigma_x` | 2.72e-18 | cm2 | Mastroianni, Table III-1 | read |
| `D_diffusion` | 0.405 | cm | Table III-1 | read |
| `alpha_power` | 3.64e-16 | cm2 s | Table III-1 | read |
| `Sigma_f` | 0.14 | 1/cm | Table III-1 | read |
| `gamma_x`, `gamma_I` | 0.0046, 0.061 | | Table III-1 | read |
| `H_core` | 365.8 | cm | Table I-1 | read |
| `nu_Sigma_f` | 2.44 x 0.16 | 1/cm | Derived, see below | derived |
| `Phi_full` | 2.02e13 / 0.75 | 1/cm2/s | Thesis section 3.1, the test's flux at 75 % power | read |
| `sigma_phi0` | 5.75e-5 | 1/s | `sigma_x Phi_full pi / 4` | derived |
| `gamma_ratio` | 0.0754 | | `gamma_x / gamma_I` | derived |
| `rho_Xe_full` | 0.025 | | 2500 pcm, kept | earlier |
| `P_thermal_ref` | 3.0 | GW | Large PWR | earlier |
| `C_fuel`, `UA`, `m_dot_cp` | as in `env.py` | | The 33 K rise, a 5 s fuel lag | earlier |
| `C_coolant`, `C_cold`, `UA_sg`, `C_sg` | as in `env.py` | | No source for the volumes | ours |
| `T_avg_no_load`, `T_avg_full_load` | 564.8, 580.2 | K | NRC HRTD 5.0, read by an earlier session | earlier |
| `n_max`, `T_avg_min` | 1.09, 556 | | Remembered trips, neither opened | ours |
| `load_ramp_rate` | 2 %/min | | Top of OAPS's Mode A range | read |
| `load_min`, `hold_range` | 0.5, 3 to 8 h | | The URD's cycle, Chari and Rohr (1997) | earlier |
| `ramp_down_range` | 0.5 to 2 h | s | | ours |
| `demand_theta`, `demand_sigma` | 300 s, 0.5 % | | A small frequency-control wobble | ours |
| `max_steps_in_episode` | 2880 | steps | A day, so the xenon's swing shows after the return | design |

**`nu_Sigma_f`.** Table I-1 prints Sigma_a 1.53 and nu Sigma_f 1.56, and with those
the thesis's equations are heavily overdamped, which its own results are not.
Sigma_a is not one of the varied parameters. Criticality asks Sigma_a = nu Sigma_f,
and its fission estimate of 0.16 with Chalmers' nu of 2.44 gives 0.39. That value
reproduces its undamped cross section to 3 % and its plant's period to 2 %. It was
chosen on the thesis's printed results, not on anything of this task's.

---

## 4. Task design

Hold Tavg on program and delta I in its band through a 100-50-100 day, with rods and
boron. Each episode starts at full load, rods parked, boron critical, xenon at the
balance of a recent power of 0.9 to 1.0 and no tilt. The load starts down in 0.5 to
2 h, holds at 50 % for 3 to 8 h and comes back, at 2 %/min with a small wobble.

| obs | meaning |
|---|---|
| 0 | `n`, power relative to rated |
| 1 | hot-leg temperature (K) |
| 2 | rod position, +1 fully out, -1 fully in |
| 3 | turbine load |
| 4 | `T_avg` (K) |
| 5 | `T_ref` (K) |
| 6 | `delta_I`, top minus bottom, a fraction of rated |
| 7 | boron in the core (ppm) |

Actions: `[rod demand, boron flow]` in [-1, 1]. Rod demand reads as obs 2. Boron flow
is -1 for full dilution, +1 for full boration. Xenon, iodine, their tilts, the boron
on its way and the delay's length are all hidden.

---

## 5. Known deviations

**D1: the load goes straight to heat.** Steam pressure, valves and the secondary's
controls are left out.

**D2: delta I at part load is about twice Krsko's.** The sensitivity is read and
reproduces Oconee, so the excess is in `eps`, or in `alpha_coolant`, which is ours.
Nothing was moved to close it.

**D3: rods from one side.** A bank reaches only the top, so at full power with rods
out nothing can push power up the core. Xenon tilted to the top can drive delta I
down and past the band, and only time brings it back. Real plants have more than one
bank.

**D4: no scored peaking.** Delta I is scored on its band alone, with no link to local
power or to a trip.

**D5: no decay heat.** A scram removes all heat at once.

---

## 6. The expert

The expert is rung 5 of `scripts/reactor_ladder.py`, rods on delta I and boron on
Tavg, from bump tests and SIMC. The published rod program stays in `experts.py` as
`rod_control_step`, a row in that script's table. Results and room are in
`docs/proposals/reactor-axial.md`.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `reactor` | 2880 | 30 | 24.0 h | 2 in [-1, 1] | 8 | 60 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2), in dollars per 30 s control step

`compute_reward = -(tracking + running + failure)`, with Tavg and delta I past its band in `tracking`, and rod travel and effluent in `running`; each physics sub-step charges its 5 s and `step_env` sums the six, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 0.05 K | the RTD reading's resolution, about its 0.045 K scatter: a documented minimum, not a certified floor |
| `e_tol` | 0 | none |
| `tracking_exponent` | 1 | linear, as lost output is |
| `imbalance_multiple` | 1.25 x 0.0023 per K of `spot_price_per_MWh` on `P_electric_GW` | **ours.** Off program, steam conditions cost output. The relative Carnot slope at 547 K steam and 305 K condenser is 305 / (547 x 242) = 0.23 %/K. At the one imbalance price of the priced plants, $100/MWh (1.25 x the $80 spot), on `P_electric_GW` = 1.0 GW, a third of the 3 GW thermal, that is 100 x 1000 x 0.0023 = $230/h per K, so **$1.917 per K over a 30 s step**. It prices a hot plant as a cold one, which the slope does not say |
| `rod_step_price` | $1 per step | **ours,** kept by the owner on 2026-11-07. Rod travel, as the steps the rods actually moved |
| `delta_I_band` | 0.05 | +-5 % of rated, Krsko section 2. No cost inside it |
| `delta_I_price` | $1000/h per 1 % past the band | **ours.** Past the band a plant must act on its power, so each 1 % is priced as 1 % of 1 GWe derated, at $100/MWh |
| `effluent_price` | $0.0628 per kg injected | **ours.** Every kg injected displaces a kg to treat (OAPS). Boiling it off takes 2.26 MJ/kg, 0.628 kWh, at $100/MWh |
| `failure_cost` | 1708.36 $ per step | twice the dearest step that survives: 40 K off program, $7.67, 36 rod steps, $36, full dilution, $18.84, and all the power in one half, $791.67, per 30 s |
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
(`docs/proposals/measurement-noise.md`). Rod position, the load, Tref and boron stay exact:
the rods are the controller's own rate-limited command, known by counting steps.
`sensor_noise_scale = 0` gives the noise-free task. Tavg is the mean of the two RTDs,
so it scatters by 0.045 K over sqrt(2), and shares a draw with the hot-leg channel.

| reading | sigma | step | instrument | status | source |
| --- | --- | --- | --- | --- | --- |
| `n` | 1.9e-4 of rated | | Ex-core power-range ionization chambers | Corrected | [OSTI 7260785](https://www.osti.gov/servlets/purl/7260785) |
| `delta_I` | 2.7e-4 of rated | | Two ex-core sections, each as noisy as `n`, summed. Ours, from that one figure | ours | as `n` |
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
