## What this changes

<!-- What behaviour differs afterwards. If it is a physics change, say what
     moved and by how much. -->

## Checks

- [ ] `make ci` passes (ruff, black, the fast tests)
- [ ] New or changed behaviour is covered by a test

If this touches an environment's dynamics, reward or termination:

- [ ] Its `PHYSICS.md` is updated in the same commit -- parameter sources,
      validation targets, or a numbered deviation
- [ ] Any validation number that moved is stated here, with what it was before

If it adds an environment:

- [ ] Registered in `target_gym.registry`, so the conformance suite covers it
- [ ] `PHYSICS.md` present; `docs/environments.md` regenerated
      (`python scripts/generate_env_reference.py`)
