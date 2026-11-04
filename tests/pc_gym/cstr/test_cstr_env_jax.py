import jax

from target_gym.pc_gym.cstr.env import CSTRState
from target_gym.pc_gym.cstr.env_jax import CSTR


def test_reset_draws_inside_its_ranges():
    env = CSTR()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key)

    # Initial C_a is within allowed range
    assert (
        env.default_params.initial_CA_range[0]
        <= state.C_a
        <= env.default_params.initial_CA_range[1]
    )

    # Target_Ca is within allowed range
    assert (
        env.default_params.target_CA_range[0]
        <= state.target_CA
        <= env.default_params.target_CA_range[1]
    )


def test_a_state_on_the_envelope_limit_does_not_terminate():
    """The CSTR has no trip: its envelope limits are unreachable from any
    coolant setting (the reactor settles between 318 and 329 K), so even a
    state placed on a limit does not terminate."""
    env = CSTR()
    p = env.default_params

    hot = CSTRState(time=0, C_a=0.5, T=p.T_max, target_CA=0.6, T_c=298.0)
    assert not bool(env.is_terminated(hot, p))
