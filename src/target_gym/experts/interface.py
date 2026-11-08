"""Which params of each task's expert are gains.

A learner acting around a controller without changing it needs three facts about
it. Two are on the env: ``obs_target_index`` and ``obs_value_index`` say which
observation columns hold the targets and the readings. This table holds the third,
which only the laws say: the leaves of ``Controller.params`` that are gains,
dotted where nested, by the names the law uses. A gain whose tuned value is 0 is
left out, as no multiplier moves it.

This file is not in any fingerprint, and nothing a task's expert computes reads
it, so changing a row stales no record. A row changes when its law does, which the
tests here catch for a name that stops being a live gain.

Gaps a row shows on purpose:

- ``reactor`` has no gains: its rod program steps on deadbands.
- ``cement_kiln`` lists the inner burning-zone PI only: its fuzzy rule base has no
  gains.
- The aircraft list TECS and guidance gains only. Their attitude loops move into the
  plant as a fixed inner loop (docs/proposals/industrial-fixes.md, PR 7), so no
  learner should come to rely on those gains.
"""

_TECS = ("tecs.time_const", "tecs.thr_damp", "tecs.ptch_damp", "tecs.integ_gain")
_L1 = ("period", "damping")

#: Per registry name, read off each law at the sensor-noise plants.
GAIN_NAMES: dict[str, tuple[str, ...]] = {
    "battery": ("Kp", "Ki"),
    "boiler_drum": ("Kp_level", "Ki_level", "Kp_pressure", "Ki_pressure"),
    "cement_kiln": ("Kp_inner", "Ki_inner"),
    "compressor_surge": ("kp_speed", "ki_speed", "kp_surge", "ki_surge"),
    "cstr": ("Kp", "Ti"),
    "distillation": ("Kp_T", "Ti_T", "Kp_D", "Ti_D", "Kp_B", "Ti_B"),
    "four_tank": ("Kp_1", "Ti_1", "Kp_2", "Ti_2"),
    "glass_furnace": ("Kp", "Ki"),
    "hvac": ("kp", "Ti"),
    "ph_neutralization": ("Kp", "Ti"),
    "plane": _TECS,
    "plane3d_circle": _L1 + _TECS,
    "plane3d_figure8": _L1 + _TECS,
    "plane3d_heading": ("tau_hdg", *_TECS),
    "plane3d_racetrack": _L1 + _TECS,
    "plane_energy": _TECS,
    "plane_sine": _TECS,
    "reactor": (),
    "unstable_cstr": ("Kp", "Ti"),
    "wind_turbine": ("Kp", "Ki"),
}
