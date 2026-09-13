# Wind Turbine

<p align="center"><img src="../../videos/wind_turbine/pid_output.gif" width="480px"/></p>

Wind turbine — NREL 5 MW reference turbine, collective-pitch power regulation.

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((5,))` |
| Tracked variable(s) | electrical power (MW) |
| Episode length | 400 steps (100 s at 0.25 s per step) |
| Import | `from target_gym import WindTurbine, WindTurbineParams` |
| Cite as | `wind_turbine-v1` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | pitch_raw | -1 | 1 |
| 1 | torque_raw | -1 | 1 |

## Observation space

5 values. Indices (3,) carry the tracked variable(s) that the
reward scores.

## Rewards

Power tracking minus a pitch-activity penalty (a fatigue proxy).

Every environment in this suite scores on one contract: the reward is
`(tracking terms, multiplied) x (1 - weighted costs)`, bounded in
`[0, 1]`, and reaches 1 only while the target is held exactly. See
[Reward shaping](../reward-shaping.md).

## Starting state

`reset` samples the initial condition and the target; state has 9 fields.

## Episode end

**Termination.** See `check_is_terminal`.

**Truncation.** After 400 steps.

## Baselines

A tuned PID ships with this environment.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 0.25 |
| `max_steps_in_episode` | 400 |
| `R` | 63 |
| `rho` | 1.225 |
| `J_rotor` | 3.87592e+07 |
| `J_gen` | 534.116 |
| `N_gear` | 97 |
| `eta_gen` | 0.944 |
| `P_rated` | 5e+06 |
| `v_rated` | 11.4 |
| `omega_rated_rpm` | 12.1 |
| `v_cut_out` | 25 |
| `cp_c1` | 0.5176 |
| `cp_c2` | 116 |
| … | 19 more, see the params dataclass |

