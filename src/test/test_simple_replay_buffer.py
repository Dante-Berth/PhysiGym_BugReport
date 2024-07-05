import unittest
from collections import deque, namedtuple
import random
import sys
import os
import pytest

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("Deep_Reinforcement_Learning")
    + len("Deep_Reinforcement_Learning")
]
sys.path.append(absolute_path)
from src.buffers.simple_replay_buffer import replay_buffer
import torch


def test_push_sample_and_reset():
    buffer = replay_buffer(memory_size=1000)

    for i in range(10):
        trajectory = {
            "state": i,
            "action": i + 1,
            "reward": i * 2,
            "next_state": i + 2,
            "done": False,
        }
        buffer.push(trajectory)

    sampled_trajectories = buffer.sample(batch_size=3)

    assert len(buffer.buffer) == 10
    assert len(sampled_trajectories) == 3

    list_namedtuple = ["state", "action", "next_state", "reward", "done"]
    pytorch_buffer = replay_buffer(
        memory_size=1000,
        device="cpu",
        list_namedtuple=list_namedtuple,
    )

    for i in range(10):
        pytorch_buffer.push([i, i + 1, i * 2, i + 2, (True and False)])

    sampled_trajectories = pytorch_buffer.torch_sample(batch_size=3)
    predicted_action = torch.nn.LazyLinear(5)(sampled_trajectories.state)


if __name__ == "__main__":
    pytest.main()
