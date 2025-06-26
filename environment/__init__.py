from gymnasium.envs.registration import register

register(
    id="gymnasium_env/QuadNav-v0",
    entry_point="environment.x2:QuadNavEnv",
) 