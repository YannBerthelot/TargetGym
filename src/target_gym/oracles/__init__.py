"""The machinery every task's oracle is to share, as ``target_gym.experts``
holds what the experts share: one constrained solver (``ipm``), one
multiple-shooting problem builder (``shooting``), per
docs/proposals/oracle-solver.md, and the receding-horizon controller that
runs a task's problem (``controller``). The oracles on it declare their data
in ``EnvSpec.oracle_data``, and their oracle fingerprint covers these
modules; each task's ``oracle.py`` moves onto it in turn.
"""
