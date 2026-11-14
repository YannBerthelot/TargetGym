# Aircraft tasks moved to Extra

Decided by the owner on 2026-11-08, under charter rule 5 and Amendment 2, after the
reality audit ([REALITY.md](REALITY.md), [COLD_REVIEW.md](COLD_REVIEW.md)). Extra tasks
stay in the repository and the registry, are never scored, and are exempt from
Amendment 1. They get no second design.

| Task | Reason | Evidence |
|---|---|---|
| `plane_sine` | No real duty. | REALITY.md, missions: "No real duty found. Neither ATC, the RVSM rules nor the autopilot documents describe following a periodic altitude profile." Altitude is assigned and held, and changes are discrete level changes [RVSM, IFH ch. 7, ch. 10]. The task's own docstring calls it a bandwidth probe, a test of the controller rather than a mission (COLD_REVIEW C1). |
| `plane3d_circle` | No airliner duty. | REALITY.md, missions: ArduPlane LOITER circles at WP_LOITER_RAD [P409], but "for an airliner, a constant-radius orbit as a routine duty: not found". The 8 to 12 km radius and the left-hand direction have no source (COLD_REVIEW G1). The airliner's real holding duty is `plane3d_racetrack`. |
| `plane3d_figure8` | No transport-aircraft duty, and no sourced manoeuvre matches it. | REALITY.md, missions: real "only as a training manoeuvre in light aircraft", eights on pylons with bank "not to exceed 40 deg" [ACS]; "for a transport aircraft: no duty found". The task is a 34 to 36 km lemniscate with a +-200 m altitude twist, which is not eights on pylons either (COLD_REVIEW I1). |
| `plane_energy` | Rule 18: the named trade, speed against height, never happens, because thrust never runs out. Moved on 2026-11-08 under charter Amendment 3 by the owner's criterion: a heavy family member (78 t) at FL330 to 370, to Extra if residual climb stays well above 500 fpm. | Residual climb at full thrust, V (T_max - D) / W at the 290 kt / M0.78 schedule and 78 t: 1 163 fpm at FL330, 1 041 at FL350, 909 at FL370. A full-thrust climb at held airspeed in the plant: 1 123, 1 006 and 879 fpm over its first 100 s. Both against the AIM's 500 fpm minimum [AIM 4-4-10]. The reviewer's measurement at 73.4 t: 848 fpm even at FL390. `tests/plane/test_plane_energy_reality.py::test_thrust_does_not_run_out_at_heavy_cruise_levels` keeps the evidence. **Open finding (2026-11-14):** those figures predate the airframe rebuild of 2026-11-10. The plant today, on the standard atmosphere above the tropopause, gives 1 080, 922 and 739 fpm at FL330, 350 and 370: FL370 is under the criterion's 750 fpm (1.5 times 500, ours), and was at 757 fpm before the atmosphere fix. The test's FL370 case is a strict xfail until the owner decides whether `plane_energy` stays here. The Amendment 2 clearances and bands stay as built; the reference's AIM-rate limits and capture lead were not built, since the decision rests on the plant alone. |

`plane_sine` shares the 2D plant, so the plant fixes of Amendment 2 (protected limits,
readings, turbulence zones, established start) reach it; its mission and score are left as
they were. The 3D Extra tasks keep the earlier 3D plant: the changes there sit behind
`PlaneParams3D.amended`, which only `plane3d_heading` and `plane3d_racetrack` set.
