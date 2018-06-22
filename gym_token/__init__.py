from gym.envs.registration import register

print('Load gym_token, id token-v0 available. Goodluck on your trading.')

register(
    id='token-v0',
    entry_point='gym_token.envs:TokenEnv',
)
