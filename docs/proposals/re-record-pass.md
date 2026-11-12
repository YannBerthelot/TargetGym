# The re-record pass after long strata

Long strata (#173) changes every env and scenario file, so every record is redone once,
after it lands. This page lists what that pass must do, in order, from the owner's
decisions of 2026-11-11. Nothing here runs before #173 merges.

1. **Plant tests first.** Run the small tier. The distillation reality tests (boilup
   step against reflux step, internal against external flows) and cstr's open-loop
   settling test failed before #173. Whatever still fails goes to the process
   session, which fixes the plant against `REALITY.md` before anything is recorded.
   Done in #191 (2026-11-12): the distillation tests restated on D1, cstr's kept as
   a strict-xfail finding. Outside the small tier, two boiler_drum reality tests have
   failed since before #173 (`test_swell_grows_toward_low_load`,
   `test_shrink_is_smaller_and_shorter_than_swell`), and the reactor oracle's
   replay test fails at 20 and -3000. Both closed in #195 (2026-11-12, owner): the
   swell test restated on S1's equal steps (p. 16), the shrink test a strict-xfail
   finding (Houtz's circulation disruption, which S1's model lacks), and the oracle's
   delta-I surrogate left out of K8's replay. Step 1 is done.
2. **Stream fixes** (owner, 2026-11-11, from #189's review). None moves a scored
   window before its trip, but each changes what a stream records.
   - **Reactor restart after a hot standby.** The plant comes back at the plan's full
     load with xenon at 1.33 of balance, and the expert re-trips 10 of 10 times, about
     18 min in, as the burnout drives the hot leg past 610 K. Restart at reduced power
     while the xenon burns out, then return to the plan at a ramp, as the boiler's
     restart does and as Doel-3's record shows. Check the expert rides it. Done in
     #193: restart at 15 %, reload at 15 %/h to 50 % and 3 %/h above, the shortfall
     charged, standby lump 6 h. Open: on stream fast_return days the expert still
     trips once on 4 of 5 seeds, 23 to 67 h after a restart, a weakness of its own on
     those days, not of the restart.
   - **Reactor expert reads the plan once.** It reads the load plan on its first step
     only, so in a stream it never sees later days. Make it read each day's plan; its
     windows may move, which this pass re-records anyway. Done in #193, with the
     day's ramp rate and minimum load added as obs 11 and 12.
   - **Absorbed event draws.** A draw while an event of its kind is under way is
     absorbed (boiler, glass, reactor), so fewer events happen than the sourced rate.
     Estimated, not measured: glass gas swings about 0.8 a day against the sourced 1.5
     (each keeps its kind busy about 14 h); reactor about 70 % of EDF's 39 load-follow
     days a year. Measure each task's delivered rate on a long stream, then set
     `events_per_h` so the delivered rate is the sourced one. Done in #192: per-visit
     correction, and `Scenario.pinned_events_per_h` so pinned scoring runs see the
     same rate; boiler upsets reach 0.71 in a pinned run, a documented gap.
   - **Aircraft references on long streams** (owner, 2026-11-11, from #190's E1 run:
     4 seeds of 111 plant hours each). plane's and plane3d_heading's experts fall
     into altitude limit cycles (±430 m, period 20 min; ±600 m, growing) that outlive
     the upset that started them and last to the sortie's end; no window shows them.
     formation's expert trips 387 times per 1 000 h (43 in 111 h). Diagnose each and
     fix it by the practitioner's procedure (tuning by real protocol), then rerun E1.
     Done in #194: TECS on ArduPilot master's pitch loop (E1 cost per step plane 3.0
     to 1.0, plane_energy 129 to 0.55, plane3d_heading 3.8 to 1.4), the figure-8 on
     the shared tune, and formation's lead at the parade roll rate with its bank
     capped at 30 deg (0 collisions per 1 000 h, window cost 28.6 to 42.7).
   Step 2 is done (2026-11-12).
3. **Recipes.** Run `tune_process_experts.py` and `tune_industrial_experts.py` with
   `--write` at levels 1, 0.5 and 2 on every task they cover. The recipes run their
   bump tests at the level's sensor noise, so cstr's and unstable_cstr's gains will
   differ by level, which `test_gain_levels.py` requires; the gains stored today come
   from the old noise-free recipe.
4. **Stamps.** `stamp_env_versions.py`: re-stamp in place with a note where a task's
   behaviour is preserved, otherwise bump its version.
5. **Charter records.** `record_charter.py` per task: reference, nulls, MPC, ceiling,
   J2, anchor. Every charter task has a B7 sweep (`scripts/family_sweep.py` on its
   package's `ours.json`); cstr, distillation, four_tank, ph_neutralization,
   unstable_cstr and glass_furnace have no sourced range for any `ours` number, so
   theirs runs the shipped member alone (owner, 2026-11-12). Ceiling files now carry
   the stamp of the plant they ran on, and none of today's does, so every ceiling
   reads stale and "not measured" until it is re-measured. The pH oracle is not
   rebuilt, so its MPC line reads "not measured".
6. **Old records** (`baseline_returns`, `protocol_results`, `credibility`) only for
   tasks still scored on tracking, the aircraft. Before them, re-measure the oracle's
   best hold (`rho`, `scripts/measure_hold.py`) for `plane3d_circle` and
   `plane3d_figure8`: both still carry the hold measured before they moved to the
   amended plant on 2026-11-10.
