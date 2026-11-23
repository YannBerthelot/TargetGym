"""The steam generators' safety valves, the bound on a hot Tavg (owner, 2026-11-17).

Contract, from the plant record (REALITY.md section 4), written before the code.
What the valves do is the standard specifications' (NUREG-1431 Bases B 3.7.1, read
in full): they protect the secondary side from overpressure and are the heat sink
when the reactor makes more heat than the turbine takes, which raises the steam's
pressure until they open; they hold it under 110 % of the design pressure. The
lowest valve's set pressure, 1176 psig, is two plants' (search summaries, flagged
in the record).

- The steam relieves at the saturation temperature of that set pressure. 1176 psig
  is 1190.7 psia, 8.2096 MPa, and on the plant's steam curve (through 5.503 MPa at
  270 C, slope 4618.5 K) that is 1 / (1 / 543.15 - ln(8.2096 / 5.503) / 4618.5) =
  569.96 K: the plant's 570.0 K, to 0.1 % of the pressure.
- On its program the steam never reaches it: 563.81 K at 15 % of load, 561.5 K at
  half load, 558.2 K at full load.
- The valves open in proportion to the pressure over the set pressure and pass
  the full-power steam 3 % over it. The steam's pressure rises 4618.5 / 570^2 =
  1.42 % a kelvin, so the steam sits 0.03 / 0.0142 = 2.11 K over the set
  temperature per unit of power relieved. Steam hotter than that does not last a
  sub-step. A plant lined out at half load with its primary and its steam 12 K
  over the program has steam at 573.5 K, which the turbine alone would leave
  there (the heat in is the heat out). With the valves the steam settles where
  the heat in, (584.5 - T) / 22 of full power, less the turbine's half, goes out
  of them, (T - 570.0) / 2.11: 0.31 K over the set temperature, in well under a
  sub-step (1e9 J/K against 1.4e9 W/K, 0.7 s). The hot core's power falls
  meanwhile, so a sub-step later the steam is between 570.0 and 570.31 K.
- So the reactor cannot sit as hot as it likes at part load: Tavg is the steam's
  temperature plus 22 K per unit of the power the steam generator carries. By
  hand at the start of the cycle, held at half load with the rods still: twelve
  steps of full dilution take out the boron's share 1 - exp(-0.0144), 24.16 ppm
  of 1690 ppm (the test reads the plant's own boron at the hold), worth 8 pcm a
  ppm, 193.3 pcm. With no relief Tavg would rise 193.3 / 15 = 12.9 K, to 585.4 K,
  on steam at 574.4 K. With the valves the power rises instead, and the heat over
  the turbine's goes out of them, the steam 2.11 K over the set temperature per
  unit of power relieved: Tavg = 570.0 + 2.11 (n - 0.5) + 22 n,
  and the balance 8 ppm = 15 x (8.5 + 24.11 (n - 0.5)) + 3 x 471.2 x (n - 0.5)
  gives n = 0.5 + (8 ppm - 127.5) / 1775.3: 0.5371 for 24.16 ppm, the steam at
  570.08 K and Tavg at 581.9 K. The xenon the extra power burns adds some 12 pcm
  an hour, 0.007 of power.

Every plant here is quiet: no load wobble, no sensor noise, no burnup, no debt carried
in, at the start of its cycle, lined out at half load with its xenon, on a plan that
holds it there.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.env import (
    compute_next_state,
    steady_state_temperatures,
    steam_pressure,
)

HELD = ReactorParams(
    demand_sigma=0.0,
    sensor_noise_scale=0.0,
    burnup_range=(0.0, 0.0),
    carried_days_range=(0.0, 0.0),
    cycle_range=(0.0, 0.0),
    initial_tilt_share=0.0,
    initial_xenon_power_range=(0.5, 0.5),
    ramp_down_range=(-3000.0, -3000.0),
    hold_range=(1e6, 1e6),
)
RELIEF = 570.0  # K


def test_the_steam_relieves_at_the_safety_valves_set_pressure():
    assert float(HELD.relief_temperature) == pytest.approx(RELIEF, abs=0.05)
    at = lambda T: float(steam_pressure(T, HELD, np))  # noqa: E731
    assert at(HELD.relief_temperature) / at(543.15) == pytest.approx(
        8.2096 / 5.503, rel=1e-3
    )


@pytest.mark.parametrize("load, steam", [(0.15, 563.81), (0.5, 561.5), (1.0, 558.2)])
def test_on_its_program_the_steam_never_reaches_the_relief(load, steam):
    T_sg = float(steady_state_temperatures(load, HELD)[3])
    assert T_sg == pytest.approx(steam, abs=0.01)
    assert T_sg < float(HELD.relief_temperature)


def _held():
    """The plant lined out at half load, and the action that holds its rods and
    its turbine."""
    env = Reactor()
    _, s = env.reset_env(jax.random.PRNGKey(0), HELD)
    assert float(s.T_sg) == pytest.approx(561.5, abs=0.01)
    return env, s, [1.0 - 2.0 * float(s.rod_in), 0.0, 0.0]


def test_steam_over_the_relief_does_not_last_a_sub_step():
    _, s, hold = _held()
    hot = s.replace(
        T_fuel=s.T_fuel + 12.0,
        T_coolant=s.T_coolant + 12.0,
        T_cold=s.T_cold + 12.0,
        T_sg=s.T_sg + 12.0,
    )
    out, _ = compute_next_state(jnp.array(hold), hot, HELD)
    assert RELIEF < float(out.T_sg) < RELIEF + 0.31
    # Under the relief the same step leaves the steam alone: 6 K over the program.
    warm = s.replace(
        T_fuel=s.T_fuel + 6.0,
        T_coolant=s.T_coolant + 6.0,
        T_cold=s.T_cold + 6.0,
        T_sg=s.T_sg + 6.0,
    )
    out, _ = compute_next_state(jnp.array(hold), warm, HELD)
    assert float(out.T_sg) == pytest.approx(567.5, abs=0.5)


def test_a_reactor_held_at_half_load_cannot_sit_hotter_than_its_steam_allows():
    env, s, hold = _held()
    step = jax.jit(env.step_env)
    dilute = [hold[0], -1.0, 0.0]
    ppm = float(s.boron) * (1.0 - np.exp(-0.0144))  # twelve steps of full dilution
    power = 0.5 + (8.0 * ppm - 127.5) / 1775.3
    steam = RELIEF + 2.11 * (power - 0.5)
    assert 0.53 < power < 0.56
    tripped = False
    for k in range(100):  # 12 steps of full dilution, then 44 min to line out
        action = np.array(dilute if k < 12 else hold, np.float32)
        obs, s, _, _, info = step(jax.random.PRNGKey(k), s, action, HELD)
        tripped = tripped or bool(info["tripped"])
    assert not tripped
    assert float(obs[3]) == pytest.approx(0.5, abs=1e-6)  # the turbine's load
    assert float(s.T_sg) == pytest.approx(steam, abs=0.05)
    assert float(s.n) == pytest.approx(power, abs=0.012)
    assert float(0.5 * (s.T_coolant + s.T_cold)) == pytest.approx(
        steam + 22.0 * power, abs=0.3
    )
