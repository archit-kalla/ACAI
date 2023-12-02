from gymnasium.envs.registration import register

register(
    id='ACAI-v0',
    entry_point='ACAI.envs:ACEnv',
    max_episode_steps=300,
)