from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym.base import (
    FieldReading,
    actuate,
    canonical_reset,
    finish_step,
    measured_obs,
)
from target_gym.energy.wind_turbine.env import (
    WindTurbineParams,
    WindTurbineState,
    balance_pitch,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    electrical_power,
    get_obs,
    gust_magnitude,
    induction,
    omega_rated,
    rotor_thrust,
    tower_base_moment,
    tower_stiffness,
    turbulence_sigma,
)
from target_gym.energy.wind_turbine.rendering import _render
from target_gym.expert_access import RegistryExperts
from target_gym.video import SavesVideo


class WindTurbine(
    RegistryExperts,
    SavesVideo,
    environment.Environment[WindTurbineState, WindTurbineParams],
):
    """NREL 5 MW reference turbine above rated wind, with its tower.

    Observation (7,): [omega_rpm, pitch_deg, torque_pct, P_MW, rating_P_MW,
    a_fa_m_s2, v_nacelle_m_s]
    Action      (2,): [pitch_raw, torque_raw] in [-1, 1]
    """

    #: What a trip's restart carries over (``base.failure_kernel``): the outside
    #: world and the records, never the plant (docs/proposals/long-strata.md, S2).
    OUTSIDE_WORLD = (
        "v_wind",
        "v_mean",
        "v_turb",
        "gust_t0",
        "derate_t0",
        "target_power",
        "v_mean_z",
        "gust_size",
        "derate_level",
        "derate_until",
        "gust_c0",
        "derate_c0",
    )
    #: Runs as a continuing plant (``target_gym.stream``): its gust and its
    #: curtailment start through ``start_event`` and carry their size, the
    #: curtailment returns, and the mean wind moves (``env.mean_wind``).
    STREAM_READY = True

    registry_name: str = "wind_turbine"

    render_turbine = classmethod(_render)
    screen_width = 700
    screen_height = 800

    obs_value_index: int = 3  # electrical power (MW)
    tracked_names: tuple = ("electrical power (MW)",)
    obs_target_index: int = 4  # the rating's power (MW)

    # Speed, pitch and torque sensors and a power meter (PHYSICS.md, Readings).
    # Power has a field of its own because the measured reward tracks it, and a
    # channel reading stays out of the measured copy. The nacelle anemometer
    # scatters by its transfer function's residual, 0.36 m/s RMSE [S6]; the
    # accelerometer's noise is not found, so it reads true.
    readings: tuple = (
        FieldReading("omega", sigma=5.2e-4),
        FieldReading("pitch", sigma=0.2),
        FieldReading("torque", sigma=90.0),
        FieldReading("power", sigma=1e3),
        FieldReading("v_wind", sigma=0.36),
    )

    def __init__(self, integration_method: str = "rk4_2"):
        self.obs_shape = (7,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> WindTurbineParams:
        return WindTurbineParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params, tripped=None):
        return compute_reward_terms(state, params, tripped=tripped)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: WindTurbineState,
        action: jnp.ndarray,
        params: WindTurbineParams = None,
    ):
        if params is None:
            params = self.default_params
        action = jnp.atleast_1d(jnp.asarray(action)).reshape((2,))
        action, state = actuate(self, state, action, params)

        proposal, _ = compute_next_state(
            action, state, params, key, integration_method=self.integration_method
        )
        # A trip is part of the kernel: the step is charged the trip cost and
        # the plant restarts at once, or is lost (``terminated``) with
        # ``params.continuing = False`` (``base.finish_step``).
        new_state, reward, terminated, info = finish_step(
            self, key, state, proposal, params
        )
        # The plant's own actuators set the rates, so the applied action is
        # where they went.
        info["applied_action"] = self.initial_action(proposal, params)
        return self.get_obs(new_state, params), new_state, reward, terminated, info

    def initial_action(self, state, params):
        """The raw action that holds the pitch and torque where they are."""
        return jnp.stack(
            [
                2.0 * state.pitch / params.pitch_max - 1.0,
                2.0 * state.torque / params.torque_max - 1.0,
            ]
        )

    @measured_obs
    def get_obs(self, state, params: WindTurbineParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminated(self, state, params) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    @canonical_reset
    def reset_env(
        self, key: chex.PRNGKey, params: WindTurbineParams = None
    ) -> Tuple[jnp.ndarray, WindTurbineState]:
        if params is None:
            params = self.default_params

        # Established operation above rated (REALITY.md sec. 5, S1 Sec. 7): a
        # mean wind from S2's above-rated load cases, the turbulence drawn from
        # its stationary law, the rotor at rated speed delivering the rating,
        # the pitch that balances it, the tower at rest under the thrust.
        key, wind_key, turb_key, gust_key, derate_key = jax.random.split(key, 5)
        v_mean = jax.random.uniform(
            wind_key,
            minval=params.v_mean_range[0],
            maxval=params.v_mean_range[1],
        )
        lo, hi = params.v_mean_range
        share = jnp.where(hi > lo, (v_mean - lo) / jnp.maximum(hi - lo, 1e-9), 0.5)
        v_mean_z = jax.scipy.special.ndtri(jnp.clip(share, 1e-6, 1.0 - 1e-6))
        v_turb = turbulence_sigma(v_mean, params) * jax.random.normal(turb_key)
        v_wind = jnp.maximum(v_mean + v_turb, 0.1)
        target_power = params.rating * params.P_rated
        w0 = jnp.asarray(omega_rated(params))
        torque0 = jnp.clip(
            target_power / (params.eta_gen * params.N_gear * w0),
            0.0,
            params.torque_max,
        )
        pitch0 = balance_pitch(w0, v_wind, target_power / params.eta_gen, params)
        x0 = rotor_thrust(w0, v_wind, pitch0, params) / tower_stiffness(params)
        # The scenario's events, timed by the seed inside their windows.
        gust_t0 = jnp.where(
            params.gust > 0.5,
            jax.random.uniform(
                gust_key, minval=params.gust_window[0], maxval=params.gust_window[1]
            ),
            -1.0,
        )
        derate_t0 = jnp.where(
            params.derate_to < 1.0,
            jax.random.uniform(
                derate_key,
                minval=params.derate_window[0],
                maxval=params.derate_window[1],
            ),
            -1.0,
        )

        state = WindTurbineState(
            time=0,
            omega=w0,
            pitch=pitch0,
            torque=torque0,
            v_wind=v_wind,
            v_mean=v_mean,
            torque_cmd=torque0,
            target_power=target_power,
            power=electrical_power(w0, torque0, params),
            x_tower=x0,
            v_tower=jnp.zeros_like(x0),
            # Established: the wake in equilibrium with the rotor.
            w_induced=induction(w0, v_wind, pitch0, params)[0],
            a_tower=jnp.zeros_like(x0),
            v_turb=v_turb,
            feathered=jnp.zeros_like(x0),
            gust_t0=gust_t0,
            derate_t0=derate_t0,
            moment=tower_base_moment(x0, params),
            moment_turn=tower_base_moment(x0, params),
            v_mean_z=v_mean_z,
            derate_level=jnp.asarray(params.derate_to, jnp.float32),
            derate_until=derate_t0 + params.derate_hold,
            gust_c0=jnp.zeros((), jnp.int32),
            derate_c0=jnp.zeros((), jnp.int32),
        )
        return self.get_obs(state), state

    def start_event(self, state, key, params: WindTurbineParams):
        """The stream's event hook (docs/proposals/long-strata.md, S1): the
        scenario's event starts now. An extreme operating gust (``gust``) starts
        with the size the mean wind gives it now; a curtailment (``derate_to`` <
        1) cuts the rating to ``derate_to`` for ``derate_hold``. Other scenarios
        have no event. Each event's clock counts from its onset, so it keeps
        exact time however long the plant has run. Sizes are the scenario's own,
        so ``key`` is unused."""
        del key
        p = params
        now = jnp.asarray(state.time, jnp.int32)
        gusted = state.replace(
            gust_t0=jnp.zeros(()),
            gust_c0=now,
            gust_size=gust_magnitude(state.v_mean, p) + jnp.zeros(()),
        )
        curtailed = state.replace(
            derate_t0=jnp.zeros(()),
            derate_c0=now,
            derate_level=jnp.asarray(p.derate_to, jnp.float32),
            derate_until=jnp.asarray(p.derate_hold, jnp.float32),
        )
        is_gust, is_cut = p.gust > 0.5, p.derate_to < 1.0
        return jax.tree_util.tree_map(
            lambda g, c, s: jnp.where(is_gust, g, jnp.where(is_cut, c, s)),
            gusted,
            curtailed,
            state,
        )

    def action_space(self, params: WindTurbineParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0, -1.0]),
            high=jnp.array([1.0, 1.0]),
            shape=(2,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: WindTurbineParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: WindTurbineParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(WindTurbineState.__dataclass_fields__), dtype=jnp.float32
        )

    def render(self, screen, state, params, frames, clock):
        frames, screen, clock = self.render_turbine(
            screen, state, params, frames, clock
        )
        return frames, screen, clock
