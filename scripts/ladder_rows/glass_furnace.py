"""The reference ladder's row for ``glass_furnace`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score. The plant's
section "The reference ladder's row" in ``glass_furnace/PHYSICS.md`` names the loop: the
fuel valve (raw in [-1, 1] for 0.25 to 0.75 kg/s) on the fining share read at the cold
end (observation 4), kept in the zone 0.5 to 1 with its cheap side low, since less fuel
fines less. The canal is held by the plant's own cooling loop and the crown by its own
override, so neither asks a rung here.

Sizes come from the plant's own figures:

- the fining share's step test on the registered plant (``pid_gains.json`` under
  ``glass_furnace``, its ``model``): 41.75 h of lag after 3.83 h of dead time;
- the crown's step on the fuel, 808 K per kg/s over a 4.1 h lag (``env_jax``, where the
  override is designed on it);
- the walks' spreads (``GlassFurnaceParams.walk_sd``): the gas's heating value wanders by
  2 % on a 24 h memory, which a bump reads against;
- the design state (``design_state.npz``): the valve and the cooling that settled it, and
  its balance sheet from one plant step (``physical.plant_step``'s books).
"""

from __future__ import annotations

import math

from reference_ladder import (  # noqa: E402
    Cv,
    Inner,
    Row,
)

from target_gym.registry import REGISTRY


def row() -> Row:
    import jax.numpy as jnp

    from target_gym.glass_furnace import env_jax as E
    from target_gym.glass_furnace import physical as P
    from target_gym.glass_furnace.physical_scenarios import GAS_STEP, PULL_STEP

    p = REGISTRY["glass_furnace"].make_test_params()
    pp = p.plant
    hour = 3600.0 / float(p.delta_t)  # steps
    span = float(p.fuel_max - p.fuel_min)  # kg/s over the raw action's 2
    per_raw = span / 2.0  # kg/s per unit of raw action

    # The design point: the valve and cooling that settled the plant at its pull and
    # cullet, and the stack's temperature in its balance sheet.
    fuel0, cooling0, fields = E.design_state()
    state0 = P.Fields(**{k: jnp.asarray(v) for k, v in fields.items()})
    _, books = P.plant_step(
        state0, fuel0, p.pull, p.cullet, p.thickness, cooling0, p.grid, pp
    )
    fired = float(P.mean_firing(pp))
    # Available heat: what a kg of fuel leaves inside the furnace once its flue gas
    # leaves the regenerators at the stack. The walls, the crown and the refiner lose
    # what their temperatures set, so a held fining state holds them, and only the
    # stack's share grows with the fuel.
    flue_per_fuel = 1.0 + float(pp.AFR) * (1.0 + float(pp.excess_air))
    stack = flue_per_fuel * float(pp.c_p_gas) * (float(books["T_stack"]) - pp.T_ambient)
    available = float(pp.LHV) * fired * (1.0 - stack / float(pp.LHV))  # J per kg valve
    # A kg of glass takes the batch's melting heat, less what its cullet saves
    # (``physical.plant_step``'s ``H_melt``), then is heated from the melt's 1450 K to
    # the middle of the fining band, where the pass mark sits (``fining_pass`` 0.5 of
    # Beerkens' 1420 to 1500 C). The refiner takes that heat back before the canal.
    t_pass = float(pp.T_fine_low + p.fining_pass * (pp.T_fine_high - pp.T_fine_low))
    lift = float(pp.c_glass) * (t_pass - float(pp.T_melt_K))
    pull0, cullet0 = float(p.pull), float(p.cullet)

    def need(pull, cullet):
        return pull * (pp.H_melt * (1.0 - pp.cullet_gain * cullet) + lift)

    base_need = float(need(pull0, cullet0))

    # The steady feedforward reads the pull and cullet charged (observations 8, 10).
    # Not the pull planned ahead (9): moving on it is a lead, dynamics a steady balance
    # does not give. Not the gas, which is hidden, nor the batch line (11), a state of
    # the blanket rather than a load.
    def fuel_for(o):
        """kg/s on the valve for the pull and cullet charged."""
        return fuel0 + (need(o[8], o[10]) - base_need) / available

    def raw(fuel, xp):
        return xp.reshape(xp.clip((fuel - p.fuel_min) / per_raw - 1.0, -1.0, 1.0), (1,))

    def balance(o, xp):
        return raw(fuel_for(o), xp)

    def rest_at(q):
        # The same heat at a point's gas: the valve over its heating value's factor.
        gas = 1.0 + float(q.gas_swing)
        return lambda o, xp: raw(fuel_for(o) / gas, xp)

    # Each load range's ends, from the scenario suite, the event moved to the episode's
    # start so the plant holds it a whole episode before its test.
    start = {"pull_change_window_h": (0.0, 0.0)}
    points = (
        {"pull_change": PULL_STEP, **start},
        {"pull_change": -PULL_STEP, **start},
        {"gas_swing": GAS_STEP, **start},
        {"gas_swing": -GAS_STEP, **start},
    )

    # The step: the fuel worth two of the gas walk's spreads, so it stands out of the
    # wander it is read against, and never more than the crown's headroom under its
    # override at the design point, so the plant's protection does not cut it.
    gas_sd = float(p.walk_sd[E.WALKS.index("gas")])
    du = 2.0 * gas_sd * fuel0 / per_raw
    crown_k = 808.0  # K per kg/s, the hottest face's step (env_jax, the override)
    hot0 = float(P.sensors(state0, p.grid, pp)["hot"])
    headroom = (float(p.crown_high - p.override_margin) - hot0) / crown_k / per_raw
    # The fining share's response: its dead time and four lags.
    tau, theta = 41.75, 3.833  # h, the valve's step test (pid_gains.json, model)
    window = math.ceil((theta + 4.0 * tau) * hour)
    # The crown answers in its own 4.1 h lag: four of them.
    crown_window = math.ceil(4.0 * 4.1 * hour)

    fining = Cv(
        "fining on fuel",
        0,
        lambda o, xp: o[4],
        lambda o, xp: o[5],
        4,
        du,
        window,
        room=headroom,
        zone=(float(p.fining_pass), 1.0, "low"),
    )
    # Firing follows the crown below the operators' setpoints (REALITY.md section 2,
    # [S4]): the crown couple in bay 3 (observation 0) is the inner loop, its setpoint
    # moved by the fining loop. The outer test moves that setpoint by the crown change
    # the same fuel step makes.
    crown = Inner(
        0,
        "crown on fuel",
        lambda o, xp: o[0],
        None,
        0,
        du,
        crown_window,
        du * per_raw * crown_k,
        window,
    )
    return Row(
        int(p.max_steps_in_episode),
        balance,
        (fining,),
        points=points,
        rest_at=rest_at,
        balance=balance,
        cascades=(crown,),
        why={
            "decoupler": "one loop on one valve",
            "override": (
                "the plant holds its crown 5 K under 1620 C by its own override and its"
                " canal by its own cooling loop; the fining zone is priced, not a trip"
            ),
        },
    )
