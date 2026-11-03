"""The machinery every task's oracle is to share, as ``target_gym.experts``
holds what the experts share: one constrained solver (``ipm``) and one
multiple-shooting problem builder (``shooting``), per
docs/proposals/oracle-solver.md. No task uses it yet, so no record's
fingerprint covers it; each task's ``oracle.py`` moves onto it in turn.
"""
