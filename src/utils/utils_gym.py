import gymnasium as gym
import random
import numpy as np
import torch
import os


def init_seed(seed_env: int):
    """
    Function used the seed environment to init random, numpy and torch seed.

    :param env_id: The ID of the environment to create.
    :type env_id: str
    """

    random.seed(seed_env)
    np.random.seed(seed_env)
    os.environ["PYTHONHASHSEED"] = str(seed_env)
    torch.manual_seed(seed_env)
    torch.cuda.manual_seed(seed_env)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed_env)


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(
                env_id, render_mode="rgb_array", continuous=True
            )  # , continuous=True)
            env = gym.wrappers.RecordVideo(env, f"{run_name}")
        else:
            env = gym.make(env_id, continuous=True)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def wrapper_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str):
    """
    Factory function to create Gym environments with customizable settings.

    :param env_id: The ID of the environment to create.
    :type env_id: str
    :param seed: The random seed to use for the environment.
    :type seed: int
    :param idx: An index.
    :type idx: int
    :param capture_video: Whether to capture video during environment execution.
    :type capture_video: bool
    :param run_name: The name of the run.
    :type run_name: str
    """
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, idx, capture_video, run_name)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    return envs
