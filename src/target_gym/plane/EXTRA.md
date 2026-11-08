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

`plane_sine` shares the 2D plant, so the plant fixes of Amendment 2 (protected limits,
readings, turbulence zones, established start) reach it; its mission and score are left as
they were. The 3D Extra tasks keep the earlier 3D plant: the changes there sit behind
`PlaneParams3D.amended`, which only `plane3d_heading` and `plane3d_racetrack` set.
