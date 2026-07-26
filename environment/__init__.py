"""AdaptLearn-v1 environment package.

Importing this package registers "AdaptLearn-v1" with Gymnasium's global
registry, so the environment can be created either directly:

    from environment.custom_env import AdaptLearnEnv
    env = AdaptLearnEnv()

or through the standard Gymnasium factory, once this package has been
imported at least once:

    import gymnasium as gym
    import environment  # noqa: F401 -- registers "AdaptLearn-v1"
    env = gym.make("AdaptLearn-v1")
"""

from gymnasium.envs.registration import register

from environment.custom_env import AdaptLearnEnv

register(
    id="AdaptLearn-v1",
    entry_point="environment.custom_env:AdaptLearnEnv",
    max_episode_steps=None,  # the env enforces its own truncation internally
)

__all__ = ["AdaptLearnEnv"]
