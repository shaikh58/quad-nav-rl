from gymnasium.envs.registration import register

register(
    id="envs/QuadNav-v0",
    entry_point="envs.x2:QuadNavEnv",
)