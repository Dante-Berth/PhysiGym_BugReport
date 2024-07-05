import unittest
from collections import deque, namedtuple
import random
import sys
import os

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("Deep_Reinforcement_Learning")
    + len("Deep_Reinforcement_Learning")
]
sys.path.append(absolute_path)
from src.utils.utils_gym import init_seed, wrapper_env
import pytest


def test_init_seed():
    seed_env = 123
    init_seed(seed_env)


def test_wrapper_with_video():
    env_id = "Pendulum-v1"
    seed = 123
    idx = 0
    capture_video = True
    run_name = "test_run"

    env = wrapper_env(env_id, seed, idx, capture_video, run_name)

    assert env is not None


def test_wrapper_without_video():
    env_id = "Hopper-v4"
    seed = 123
    idx = 1
    capture_video = False
    run_name = "test_run"

    env = wrapper_env(env_id, seed, idx, capture_video, run_name)

    assert env is not None


if __name__ == "__main__":
    pytest.main()
