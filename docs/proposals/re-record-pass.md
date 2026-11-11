# The re-record pass after long strata

Long strata (#173) changes every env and scenario file, so every record is redone once,
after it lands. This page lists what that pass must do, in order, from the owner's
decisions of 2026-11-11. Nothing here runs before #173 merges.

1. **Plant tests first.** Run the small tier. The distillation reality tests (boilup
   step against reflux step, internal against external flows) and cstr's open-loop
   settling test failed before #173. Whatever still fails goes to the process
   session, which fixes the plant against `REALITY.md` before anything is recorded.
2. **Stream fixes** (owner, 2026-11-11, from #189's review). None moves a scored
   window before its trip, but each changes what a stream records.
   - **Reactor restart after a hot standby.** The plant comes back at the plan's full
     load with xenon at 1.33 of balance, and the expert re-trips 10 of 10 times, about
     18 min in, as the burnout drives the hot leg past 610 K. Restart at reduced power
     while the xenon burns out, then return to the plan at a ramp, as the boiler's
     restart does and as Doel-3's record shows. Check the expert rides it.
   - **Reactor expert reads the plan once.** It reads the load plan on its first step
     only, so in a stream it never sees later days. Make it read each day's plan; its
     windows may move, which this pass re-records anyway.
   - **Absorbed event draws.** A draw while an event of its kind is under way is
     absorbed (boiler, glass, reactor), so fewer events happen than the sourced rate.
     Estimated, not measured: glass gas swings about 0.8 a day against the sourced 1.5
     (each keeps its kind busy about 14 h); reactor about 70 % of EDF's 39 load-follow
     days a year. Measure each task's delivered rate on a long stream, then set
     `events_per_h` so the delivered rate is the sourced one.
3. **Recipes.** Run `tune_process_experts.py` and `tune_industrial_experts.py` with
   `--write` at levels 1, 0.5 and 2 on every task they cover. The recipes run their
   bump tests at the level's sensor noise, so cstr's and unstable_cstr's gains will
   differ by level, which `test_gain_levels.py` requires; the gains stored today come
   from the old noise-free recipe.
4. **Stamps.** `stamp_env_versions.py`: re-stamp in place with a note where a task's
   behaviour is preserved, otherwise bump its version.
5. **Charter records.** `record_charter.py` per task: reference, nulls, MPC, ceiling,
   J2, anchor. Open gaps: only the reactor has a B7 sweep runner, so the other tasks
   fail J2 on "sweep not measured" until one is written; the ceiling files carry no
   stamp of the plant they ran on; the pH oracle is not rebuilt, so its MPC line reads
   "not measured".
6. **Old records** (`baseline_returns`, `protocol_results`, `credibility`) only for
   tasks still scored on tracking, the aircraft.
