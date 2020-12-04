from gym.envs.registration import register

register(
    'chakra-v0',
    entry_point='rlpa2.chakra:chakra',
    timestep_limit=40,
)
register(
    'vishamC-v0',
    entry_point='rlpa2.vishamC:vishamC',
    timestep_limit=40,
)