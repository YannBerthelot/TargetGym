# CSTR — physics model, provenance and validation

Reference process: a **non-isothermal continuous stirred-tank reactor** running
an irreversible first-order exothermic reaction A → B, cooled through a jacket.
The coolant temperature is the manipulated variable.

Contract for `target_gym.pc_gym.cstr`. Method: `docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 1. Model and provenance

Two states — reactant concentration and reactor temperature — under perfect
mixing:

```
dCa/dt = (q/V)(Caf − Ca) − r
dT/dt  = (q/V)(Ti − T) + (−ΔHr)·r / (ρ c_p) + UA(Tc − T) / (ρ c_p V)

with    r = k0 · exp(−E/RT) · Ca
```

**Provenance: adapted from [PC-gym](https://github.com/MaximilianB2/pc-gym).**
The parameters and both ODEs were compared against PC-gym's `model_classes.py`
term for term and match exactly. PC-gym itself carries no citation for the
model; the parameter set is the standard non-isothermal CSTR used throughout
process-control teaching.

The interesting structure is the feedback between the two states: heat released
raises the temperature, the Arrhenius term raises the rate exponentially, and
that releases more heat. The jacket is the only thing holding it.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Jacket dynamics | Coolant temperature is commanded directly, with no lag, rate limit or flow limit. Ours, for the owner to accept. |
| Reverse reaction, side reactions, catalyst decay | One irreversible first-order step. |
| Level / volume dynamics | Constant holdup, perfectly mixed. |
| Mixing lag | Instantaneous. |

**Feed loads.** After the settle, `Ti` and `Caf` move in steps nothing measures
(`step_load`). Each is a uniform draw held for 10 min, within ±1 K and
±0.02 mol/L, and the first arrives at step 67, the task's fixed settle, so the
whole path falls in the scored hold. The draw comes from the episode's key and
the clock, so every policy on one episode key meets the same loads. All these
figures are ours, for the owner to accept. The limits are the largest round pair
that leaves the ignition fold above `T_c_max` at every corner (§2).
`load_scale = 0` switches them off.

**Regime of validity.** The low-conversion (extinguished) branch, roughly
315–332 K, with coolant between 295 and 302 K. The model is written for the
whole state space but the environment only operates on that branch — see §2.

---

## 2. Validation targets

| Quantity | Target | Model | |
|---|---|---|---|
| Parameters and ODEs vs PC-gym | exact match | **verbatim** | ✅ |
| Residence time V/q | — | **1.00 min** | ✅ |
| Steady-state multiplicity | 3 solutions over a coolant window | **3 for Tc ≈ 299–304 K** | ✅ |
| Operating branch is stable | required | eigenvalues < 0 throughout | ✅ |
| Operating temperature | 315–330 K | 317.7 → 328.7 K | ✅ |
| Conversion on that branch | low | 7.3 % → 16.5 % | ✅ |
| Every target concentration reachable | required | 0.835 ≤ Ca ≤ 0.927 covers (0.87, 0.90) | ✅ |
| Every target holds under every load | 0.5 K left at each coolant stop | needs 295.86 to 301.50 K | ✅ |
| Reaction is exothermic | ΔHr < 0 | −50 kJ/mol | ✅ |
| The coolant cannot ignite the reactor | fold above `T_c_max` at every load | 303.23 K, 302.25 K at the worst load, against 302 K | ✅ |

**Multiplicity is the load-bearing check.** It is not something the parameters
were tuned to produce — it falls out of the Arrhenius feedback, and its
presence is what makes this a genuine reactor rather than a first-order lag
with a nonlinear gain:

| coolant Tc | steady states (T) | reading |
|---|---|---|
| 295 K | 317.7 | extinguished only |
| 300 K | 324.5 · 350.0 · 369.7 | **three** — the middle one is unstable |
| 302 K | 328.7 · 343.5 · 373.6 | three |
| 305 K | 378.1 | ignited only |

**The plant cannot run away, and it has no trip.** The low branch ends at a fold
at Tc = 303.23 K, T = 335.65 K, and `T_c_max` is 302 K, so no action reaches it.
The worst load, feed 1 K hotter and 0.02 mol/L richer, brings the fold down to
302.25 K, and with the coolant at its stop the reactor settles at 331.9 K. Across
the coolant's range and every load it rests between 317 and 332 K, with
C_a from 0.81 to 0.95, inside the 300 / 350 K and 0.7 / 1.0 mol/L limits.
`check_is_terminal` never fires, and those limits only document the envelope.

**The plant also slows as it is pushed.** The eigenvalues run from −1.53 /min
at Tc = 295 to −0.75 /min at Tc = 302, so the closed loop that is comfortable
at the cold end is half as fast at the hot end.

---

## 3. Parameter table

| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `q`, `V` | 100, 100 | L/min, L | PC-gym | ✅ |
| `rho`, `C` | 1000, 0.239 | g/L, J/(g K) | PC-gym | ✅ |
| `deltaHr` | −5.0e4 | J/mol | PC-gym | ✅ |
| `EA_over_R` | 8750 | K | PC-gym | ✅ |
| `k0` | 7.2e10 | 1/min | PC-gym | ✅ |
| `UA` | 5.0e4 | J/(min K) | PC-gym | ✅ |
| `Ti`, `Caf` | 350 K, 1.0 mol/L | – | PC-gym | ✅ |
| `T_c_min`, `T_c_max` | 295, 302 | K | THIS REPO — not in PC-gym, which has no bounds | ⚠️ |
| `T_max` | 350 | K | THIS REPO: an envelope limit, not a trip, and unreachable | ✅ |
| `target_CA_range` | (0.87, 0.90) | mol/L | THIS REPO, see D1 | ✅ |
| `Ti_load`, `Caf_load` | 1, 0.02 | K, mol/L | THIS REPO: the fold's limit, for the owner to accept | ⚠️ |
| `load_start`, `load_period` | 67, 40 | steps | THIS REPO: the fixed settle, and a 10 min block, for the owner to accept | ⚠️ |
| `delta_t` | 0.25 | min | 4 steps per residence time | ✅ |

PC-gym's model class carries **no bounds, targets or termination**. Everything
in the lower half of this table is this repository's own choice, and is
therefore where to look first when something is unreachable — as it was for the
four-tank environment.

---

## 4. Task design

### Task: `cstr`

Hold the concentration of reactant A in a cooled, exothermic stirred-tank
reactor at a target drawn at reset, by setting the coolant temperature. The
coolant reaches the concentration only through the reactor temperature and an
exponential reaction rate, so the authority is narrow and nonlinear. The
analyser reads a minute late, and after the settle the feed moves in steps
nothing measures.

| obs | meaning |
|---|---|
| 0 | `C_a`, concentration of A (mol/L) |
| 1 | `T`, reactor temperature (K) |
| 2 | target `C_a` (mol/L) |

**Why it is interesting.** Concentration is controlled indirectly: the coolant
sets the temperature, and the temperature sets the reaction rate through an
exponential. The gain is roughly −0.011 mol/L per K, so the entire 7 K input
range buys 0.078 mol/L of concentration — a narrow, nonlinear authority.

**Observation.** A real reactor has a thermocouple and a composition analyser.
Here the thermometer is current and the analyser is 1 min, 4 steps, behind
(see Readings), and the feed loads are hidden.

**Reward** — the floor-normalised cost described in *Reward (version 2)*
below.

**Loads.** PC-gym's plant has no disturbance, so a table of the target could
pass for control. The feed loads of §1 are ours, and the hold is scored against
them. They come from the episode's key, not the step's, which is why the registry
declares no `disturbance_fields` for the conformance suite's PRNG-hygiene checks.

**The oracle.** The MPC slot is a problem definition, not a tuned
controller (docs/proposals/oracle-audit.md): the shared solver's oracle
(`target_gym.oracles`). Once every T_c, Cooper's sample time from the
horizon protocol (one step here, `data/oracle_horizons/cstr.json`), it plans
P = 32 moves from the true state and applies the first. Near the end of the
episode the plan shrinks to the steps that are left. The plan maximises the
episode reward over the horizon on the environment's own dynamics with the
noise zeroed, and with the true load, read at each solve and held over the
plan, so it never sees an arrival coming. Two rewrites make that problem smooth without changing it
on any plan in the box: the coolant command's clip is replaced by a bound
on the move, and the tracking term is written as (e / e_floor)^2, which is
the reward itself since e_tol is 0 and p is 2. The trip test is constant,
so no limit enters, and nothing is smoothed. IPOPT's algorithm solves it
with exact Hessians to a scaled KKT error of 1e-8, or stops at 150
iterations. A cold start puts every planned state at the measured one and
holds the coolant the reset records, which clips to full cooling since the
reset carries none, and each later solve starts
from the last plan shifted one step. A warm solve that fails or hits the cap is solved again cold,
and one that still fails applies the last plan's next move. Every solve's certificate (KKT error, iterations, violation, and
the plan replayed on `step_env`) is summed into the record. `e_floor` is the
analyser's resolution and `rho_floor` is 0, both documented minima, so the
oracle sets neither.

---

## 5. Known deviations

**✅ D1, fixed: the target band keeps authority for the loads.** Targets were
sampled from (0.84, 0.91) mol/L, and 0.84 needed the coolant within 0.2 K of its
stop. They now come from (0.87, 0.90) mol/L, 10 to 13 % conversion, against 7 to
16 % across the coolant's range. Under every load each target holds with at least
0.5 K left at each stop, 7 % of the range, and a test keeps it so.

**⚠️ D2 — the jacket is ideal.** Coolant temperature is commanded directly.
A real jacket has a flow actuator, a transport lag and a finite cooling duty.

**✅ D3, fixed: the feed moves.** See the feed loads in §1.

**⚠️ D4: no runaway, by construction.** The coolant's stop and the load limits
keep the reactor below its ignition fold (§2), so the multiplicity is in the model
but only one branch is ever visited, and nothing can trip. A load past the limits
would need a real trip at the same time.

---

## 6. Expert

**Stale until PR 9.** The gains and figures below predate the 1 min analyser and
the loads. On the new plant the outer loop's phase margin is negative, and the
cascade cycles: it holds 1.5e4 per step on seeds 200 to 209, and a constant
action beats it. The same outer loop tuned by SIMC for the longer delay,
Kp = -34.6 K per mol/L, loses to no null policy (measured, not shipped). PR 9
reruns the tuning.

Reactor practice closes a temperature loop on the coolant and lets the
concentration trim its setpoint, because composition analysers are too slow
to close on directly. The expert is that cascade (`make_cstr_pid_controller`,
`cascade_step`), tuned by SIMC (Skogestad and Grimholt, "The SIMC method for
smooth PID controller tuning", 2012) from step tests, and stored by
`scripts/tune_process_experts.py --envs cstr`.

- **Inner loop**, T on the raw coolant command. A step of 0.1 toward mid-range
  from the steady state at each training seed's target (seeds 100 to 109).
  The median first-order fit is k = 6.10 K per unit raw action,
  tau1 = 0.635 min and, with half the 0.25 min sample, theta = 0.194 min
  (measured). Tight SIMC (tau_c = theta) gives Kp_T = 0.268 per K and
  Ti_T = 0.635 min.
- **Outer loop**, C_a on the temperature setpoint, tested with the inner loop
  closed: a 0.5 K setpoint step from the same steady states. The median fit
  is k = -0.00918 mol/L per K, tau1 = 0.861 min and theta = 0.354 min, which
  carries the inner loop's own response. Tight SIMC gives Kp = -132 K per
  mol/L and Ti = 0.861 min; 2 theta and 3 theta cost more on the training
  seeds (36.7 per step against 60.6 and 124, measured).
- **Bias.** The outer PI's output is added to the target's steady temperature,
  from the concentration balance and the design kinetics
  (`steady_temperature`), as the unstable CSTR's cascade does. The reset puts
  the reactor at 330 K, off every steady state on the operating branch.
- **Behaviour.** The first approach overshoots: on training seed 100 the
  concentration error reaches 17 % of its initial value the other way. The
  cascade is inside the analyser's band to stay after 20 to 40 steps on the
  training seeds, 26.5 at the median (measured). The protocol's
  fixed settle, 67 steps, is where it comes within twice the level it holds:
  the plant has no noise, so that level is float32 rounding. The episode is 150
  steps (100 in v2), over twice that settle.
  At the bottom of the target band (D1) the inner loop runs against the
  coolant stop.
- **Held-out check** (seeds 200 to 209). It holds 9.7e-8 per step against a
  ceiling of 1 and never trips (`tests/pc_gym/test_expert_specs.py`).

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `cstr` | 150 | 15 | 38 min | 1 in [-1, 1] | 3 | 12 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2)

`compute_reward = -(tracking + failure)`, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 5e-4 mol/L | documented minimum: the online composition analyser's scatter, the figure in the readings table. It was 1e-4 until 2026-11-06, finer than one reading. Measured with no load, before PR 9 re-measures it: the MPC before the oracle audit (2026-10) held 1e-6 mol/L after settling (`scripts/measure_hold.py`, 283 hold steps after a 12-step burn-in; PID 7e-5), and the NLP that replaced it holds the target to float32 resolution (protocol hold cost 0). |
| `e_tol` | 0 | no product specification band was supplied for the reactor concentration |
| `tracking_exponent` | 2 | quadratic |
| `failure_cost` | 7.2e5 | twice the span's cost, (0.3 / 5e-4)^2. **Never charged**: the plant cannot trip -- across the coolant's range and every load it settles between 317 and 332 K with C_a >= 0.81, inside the 300 / 350 K and 0.7 limits, which are documentation of the envelope |
| `restart_steps` | 240 (1 h) | restart time priced into a trip, `restart_steps x failure_cost` (cool, purge and re-feed after a runaway; provisional; never exercised, see `failure_cost`); where a plant engineer would get it: the plant's restart procedure |

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

The observation reports concentration and temperature through their instruments
and the setpoint exactly (`docs/proposals/measurement-noise.md`).
`sensor_noise_scale = 0` removes the noise and keeps the analyser's age.

| reading | sigma | step | dead time | instrument | status | source |
| --- | --- | --- | --- | --- | --- | --- |
| `C_a` | 5e-4 mol/L | none | 1 min | Online UV-Vis process photometer with flow cell | Provisional | [OMA-300](https://www.ima.co.uk/wp-content/uploads/AA_DS001A_OMA300.pdf) |
| `T` | 0.04 K | none | none | Pt100 RTD with HART head transmitter | Confirmed | [iTEMP TMT82](https://bdih-download.endress.com/file/2c4c2c5496b6c81881df53d319b2e2f5/TI01010TDE_2725-00.pdf) |

The analyser's delay is its sample loop's, not the photometer's, whose sheet gives
a response of 1 to 5 s. Swagelok gives
[a common goal of one minute or less](https://www.swagelok.com/en/blog/4-areas-inspect-measuring-time-delay-sampling-systems)
from the tap to the reading, and
[its distributor's page](https://northerncal.swagelok.com/understanding-time-delay)
repeats it. Both are the same vendor's, and the figure is a goal, so a real loop
may be slower. `T`'s thermowell lag and both readings' sampling are not modelled:
each reads every 15 s step, which is ours, for the owner to accept.

`C_a`, the tracked quantity, rests on a weak figure. The sheet gives only a
sensitivity of 0.1 % of full scale. Reading that as 2 sigma and mapping it to a
1 mol/L full scale are ours. `T` is an upper bound on scatter: the sheet gives ±0.08 °C, stated as 2 sigma and
including non-linearity and repeatability, and the true scatter is unknown and
smaller.

The expert reads these readings, and so does every tuner. The oracle plans and
acts on the true state, so it stays a full-state bound. `reward`, trips and the
score read the true state. `info["measured_reward"]` is the same cost on the
readings, with any trip taken from the true state, and a learner can train on
it.

A plant resting exactly on target pays 1 per step in measured reward,
`(5e-4 / 5e-4)^2`. `e_floor` and `precision_floor` are the analyser's scatter, so
the ceiling is what this analyser can verify.
