from gym.envs.registration import register

print('Load gym_token: id Token-v0, Token-v1 available. Goodluck on your trading.')

register(
    id='Token-v0',
    entry_point='gym_token.envs:TokenEnv',
)


register(
    id='Token-v1',
    entry_point='gym_token.envs:TokenEnvFragment',
)
