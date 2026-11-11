# The re-record pass after long strata

Long strata (#173) changes every env and scenario file, so every record is redone once,
after it lands. This page lists what that pass must do, in order, from the owner's
decisions of 2026-11-11. Nothing here runs before #173 merges.

1. **Plant tests first.** Run the small tier. The distillation reality tests (boilup
   step against reflux step, internal against external flows) and cstr's open-loop
   settling test failed before #173. Whatever still fails goes to the process
   session, which fixes the plant against `REALITY.md` before anything is recorded.
2. **Recipes.** Run `tune_process_experts.py` and `tune_industrial_experts.py` with
   `--write` at levels 1, 0.5 and 2 on every task they cover. The recipes run their
   bump tests at the level's sensor noise, so cstr's and unstable_cstr's gains will
   differ by level, which `test_gain_levels.py` requires; the gains stored today come
   from the old noise-free recipe.
3. **Stamps.** `stamp_env_versions.py`: re-stamp in place with a note where a task's
   behaviour is preserved, otherwise bump its version.
4. **Charter records.** `record_charter.py` per task: reference, nulls, MPC, ceiling,
   J2, anchor. Open gaps: only the reactor has a B7 sweep runner, so the other tasks
   fail J2 on "sweep not measured" until one is written; the ceiling files carry no
   stamp of the plant they ran on; the pH oracle is not rebuilt, so its MPC line reads
   "not measured".
5. **Old records** (`baseline_returns`, `protocol_results`, `credibility`) only for
   tasks still scored on tracking, the aircraft.
