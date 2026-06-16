from gym.envs.registration import register

register(
    id="avoiding-v0",
    entry_point="d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding:WrappedObstacleAvoidanceEnv",
    max_episode_steps=150,
    kwargs={"render": False},
)
