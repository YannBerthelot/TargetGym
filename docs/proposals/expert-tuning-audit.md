# Expert tuning audit: does every expert follow a real-life protocol?

Audit of 2026-11-02, read-only, against the owner's rule:

> I want a data agnostic protocol to select this value, not a post-hoc
> selection. The protocol can include collecting data if it is reasonable and
> matches real life protocols, otherwise we stick to what the protocol gives
> us. The protocol has to come from real life practices. The goal is not to
> have the best results by tuning the controller, but to have the performance
> that the best available tuning protocol/heuristic from real life gives us.

Verdicts per parameter:

- **PASS**: a practitioner procedure or published design, run as a fixed
  recipe, on a realistic test (bump test, flight test), not on a score.
- **FAIL**: chosen by searching episode costs, by held-out or scored seeds,
  or adjusted after seeing results.
- **UNCLEAR**: set by hand with no stated source.

The MPC oracles are out of scope: they are the best known controller, not
the expert, so tuning them for performance is their purpose.

## Summary

| family | tasks | all PASS | some FAIL or UNCLEAR |
| --- | --- | --- | --- |
| aircraft | 10 | none | all 10 |
| process | 7 | cstr, first_order, four_tank, distillation | ph_neutralization, compressor_surge, unstable_cstr |
| industrial and energy | 7 | reactor, hvac, wind_turbine, glass_furnace (see note) | cement_kiln, battery, boiler_drum |

Eleven tasks pass as they stand; thirteen do not. Four of the passing tasks
went through a score-based step that happened to land on the rule's own
value (below), so their gains stay but the step must go.

## What fails, and the protocol that replaces it

### Aircraft (all 10 tasks)

The TECS and attitude gains of every aircraft task come from one grid search
on mean tracking cost (`scripts/tune_aircraft_guidance.py` step 2), and the
2D tasks reuse it.

| parameters | today | replace with |
| --- | --- | --- |
| TECS `time_const`, `ptch_damp`, `integ_gain` (2.0, 1.0, 0.1) | grid on plane3d_heading | ArduPilot TECS tuning guide: TIME_CONST from its default, raised by 1 while height oscillates; PTCH_DAMP in small steps, at most about 0.5; INTEG_GAIN default |
| pitch loop `Kp_theta`, `Ki_theta`, `Kd_theta` (8, 2, 0.3) | grid | ArduPilot AUTOTUNE, or its manual PTCH2SRV procedure (P, then D, then I) |
| roll loop `Kp_roll`, `Kd_roll` (6, 3) | grid on circle and racetrack | ArduPilot AUTOTUNE, or the manual RLL2SRV procedure |
| L1 `period` (35) | grid, top edge | ArduPilot navigation tuning: from the default, minus 5 if turns are slow, plus 1 or 2 if it weaves after a turn, on a rectangular test circuit |
| figure-8 overrides (`Kd_theta` 6, `time_const` 1.5, `ptch_damp` 2.0, `turn_bank_margin` 0.75) | grid on worst error, retuned after held-out misses on seeds 200-202 | the same procedures as above; `ptch_damp` 2.0 is four times the guide's cap |
| patrol, bearing-only patrol and formation follower gains (`k_x`, `a_x`, `k_y`, `tau_hdg`, `k_xi`, `k_yi`, bands, `k_omega`, `k_ref`, `k_eng`, `thr_damp` 6.0) | grid on worst-seed slot error (`scripts/tune_formation_hold.py`) | no published gains exist for this structure: loops set from closed-loop step or relay tests on the follower; `thr_damp` per the TECS guide |
| formation lead `power_set`, `k_speed`, `Kd_theta`, `time_const` | grid | trim throttle from level flight; pitch and TECS as above |

Unclear, to be sourced: `thr_trim` 0.2 (the flight test measures 0.15 to
0.36 but the pick is by hand), `sink_max`, `accel_max`, `k_vdot`,
`turn_bank_margin` 0.85, the heading task's `tau_hdg`, `alpha_max`,
`k_bank_comp`, the figure-8's `curvature_lead`, the patrol limits `dv_max`,
`max_intercept`, `int_max`, and `spd_weight` on plane_energy, which scores
both height and speed. `climb_max` 8.0 is the flight test's 7.9 rounded up,
where the guide says about 80% of the measured climb.

Pass: `climb_max` and `sink_min` from the flight test, and `pitch_max`,
`roll_comp`, `thr_damp` (non-patrol), L1 `damping`, `max_bank_rad` and
`v_min_eas` at ArduPilot's defaults or the 1.3 Vs convention.

### Process

| task | parameters | today | replace with |
| --- | --- | --- | --- |
| ph_neutralization | Kp 8.79, Ti 40 s | SIMC with tau_c = 3 theta, picked because tau_c = theta cycles on low-buffer seeds (training cost 140, 4.2, 0.99) | SIMC on the highest-gain step-test model, the steepest point of the curve, with tau_c = theta: tuning for the worst-case gain is the documented robust practice (Skogestad 2012; Shinskey) |
| compressor_surge | `kp_speed`, `ki_speed`, `kp_surge`, `ki_surge` | coordinate descent on episode return | SIMC on a speed-to-pressure step test; anti-surge PI per Mirsky et al. 2015, or SIMC on a valve-to-margin step |
| compressor_surge | `surge_line` 0.15, `override_hold` 2 s, `reset_band` 0.05 | by hand | Mirsky's 10% control line; hold from the valve stroke time; back-calculation tracking time (Astrom and Hagglund 2006) |
| unstable_cstr | held limits `sp_rate` 3 K/min, `sp_below` 8 K, `sp_above` 2 K, `sp_line_margin` 0.3 K | by hand, validated afterwards | DCS PID blocks carry setpoint high, low and rate limits, but no source found gives a rule for their numbers (an earlier draft cited Seborg et al.; that could not be confirmed). See the process spec, section 4 |

### Industrial and energy

| task | parameters | today | replace with |
| --- | --- | --- | --- |
| cement_kiln | free-lime half-width 0.004, setpoint step 0.1 K | lowest training cost on a 3x4 grid, then extended by hand past its edge | fuzzy scaling from the quality-spec window and the operator's setpoint increment, or by PI equivalence to the SIMC loop (Jantzen 1998) |
| cement_kiln | speed half-width, trend window, `sp_trim_max` 120 K, `T_bz_base` 1765 K | by hand, partly from an older searched cascade | the kiln's nominal burning-zone temperature, a clamp from its limits, uniform Mamdani triangles (Jantzen) |
| battery | PI Kp 0.5, Ki 0.02, SoC guard margin 0.12 | by hand, no source | SIMC or IMC on a power-step test; linear BMS power derating over the datasheet window |
| boiler_drum | level-trim clamp 25 kg/s | by hand | drop it: actuator limits plus anti-windup are the practice |

Glass furnace passes on provenance (SIMC on a bump test), but the bump test
fitted a 15 s dead time where the plant's fuel pipeline delays the heat by
whole 30 s steps, so its Kp of 12 is aggressive. Re-running the bump test so
it identifies the delay is a fix to how the protocol was executed, not to
the protocol.

## Cross-cutting fixes

1. **The tau_c menu is a score-based step.** `_pick_tau_c`
   (`scripts/tune_process_experts.py:158`) and `pick()`
   (`scripts/tune_industrial_experts.py:147`) choose tau_c or lambda from
   theta, 2 theta and 3 theta by training cost. They chose theta, SIMC's own
   setting, for first_order, the cstr outer loop, four_tank, the unstable_cstr
   outer loop, glass_furnace and boiler_drum, so those gains stay; pH is the
   one they moved. Replace the menu by the fixed rule.
2. **`scripts/tune_pid.py` is a grid search** and still registers
   glass_furnace (on seeds 0-2) and reactor; running it would overwrite the
   SIMC gains. Retire it for the experts.
3. **Stale records.** `pid_gains.json`'s `_meta` says relay autotuning by
   `tune_pid.py`; its `reactor` entry is read by nothing but enters the
   fingerprint; the in-code `gain()` fallbacks differ from the stored values;
   `docs/using-the-pids.md` says the gains come from `tune_pid.py`.
4. **Judgment calls defined before running.** Practitioner guides lean on
   the operator's eye ("height oscillates", "weaves after a turn", "turns are
   slow"). Each protocol script states its reading of those in numbers
   before it is run, and ships what it yields.
5. **Held-out seeds.** The figure-8 was retuned after its misses on seeds
   200-202 were seen, so 200-209 are no longer clean for it. The redo reports
   on a fresh held-out set.

## Plan

1. Owner reviews this audit and the replacement protocols. Done on
   2026-11-02: both specs (`expert-protocols-aircraft.md`,
   `expert-protocols-process.md`) are accepted with every recommendation,
   and a value no source gives comes from a rule fixed in advance, labelled
   ours.
2. One protocol script per family, replacing the searches, each citing its
   source and stating its judgment calls in numbers, run on training seeds
   only; held-out seeds only report.
3. Process and industrial first (seven tasks), then aircraft (ten tasks,
   the larger piece). Each lands with one re-record.
4. All of it ships with the CasADi removal in 0.8.0. The compressor and
   unstable_cstr oracles fall back to their experts, so their oracle rows
   move with the expert gains: the expert work lands after the CasADi port to
   avoid recording those rows twice.
