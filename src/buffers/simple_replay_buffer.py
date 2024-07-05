from collections import deque
import random
from typing import Union
import torch
import os
import numpy as np
from tensordict import TensorDict

python_program_name = os.path.splitext(os.path.basename(__file__))[0].upper()
APPLICATION_NAME = f"[{python_program_name}]"


class replay_buffer:
    def __init__(
        self,
        memory_size: int = 1000,
        device: str = "cpu",
        list_namedtuple: list = ["state", "action", "next_state", "reward", "done"],
    ) -> None:
        """
        Initialize a replay buffer.

        This class represents a replay buffer used in reinforcement learning.

        :param memory_size: The maximum capacity of the replay buffer, defaults to 1000
        :type memory_size: int, optional
        :param device: The device on which tensors will be stored (e.g., 'cpu' or 'cuda'), defaults to 'cpu'
        :type device: str, optional
        :param list_namedtuple: The list used to create a namedtuple class to use for storing trajectories, defaults to ['state', 'action', 'next_state', 'reward', 'done']
        :type named_tuple: list
        """
        self.buffer = deque([], maxlen=memory_size)
        self.memory_size = memory_size
        self.dict_tensors = {key: None for key in list_namedtuple}
        self.keys = list_namedtuple
        self.device = device

    def push(self, trajectory: Union[tuple, list[tuple]]) -> None:
        """
        Add a trajectory or a list of trajectories to the replay buffer.

        :param trajectory: A single trajectory or a list of trajectories
        :type trajectory: Union[tuple, list[tuple]]
        """
        self.buffer.append(trajectory)

    def sample(self, batch_size: int):
        """
        Sample trajectories from the replay buffer.

        :param batch_size: The number of trajectories to sample
        :type batch_size: int
        :return: A list of sampled trajectories
        :rtype: list
        """
        return random.sample(self.buffer, batch_size)

    def torch_sample(self, batch_size: int) -> TensorDict:
        """
        Torch Sample trajectories from the replay buffer.

        :param batch_size: The number of trajectories to sample
        :type batch_size: int
        :return: A list of sampled trajectories
        :rtype: l
        """
        source = {
            key: torch.as_tensor(values).float()
            for (key, values) in zip(
                self.keys, zip(*self.sample(batch_size=batch_size))
            )
        }
        trajectory = TensorDict(
            source,
            batch_size=batch_size,
            device=self.device,
        )

        return trajectory

    def __len__(self) -> int:
        """
        Get the current size of the replay buffer.

        :return: The number of trajectories in the buffer
        :rtype: int
        """
        return len(self.buffer)

    def get_memorysize(self) -> int:
        """
        Get the maximum capacity of the replay buffer.

        :return: The maximum capacity of the buffer
        :rtype: int
        """
        return self.memory_size


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, memory_size):
        self.device = device
        self.memory_size = int(memory_size)

        self.state = np.empty((self.memory_size, state_dim), dtype=np.float32)
        self.next_state = np.empty((self.memory_size, state_dim), dtype=np.float32)
        self.action = np.empty((self.memory_size, action_dim), dtype=np.float32)
        self.reward = np.empty((self.memory_size, 1), dtype=np.float32)
        self.done = np.empty((self.memory_size, 1), dtype=np.float32)

        self.buffer_index = 0
        self.full = False

    def __len__(self):
        return self.memory_size if self.full else self.buffer_index

    def push(self, state, action, reward, next_state, done):
        self.state[self.buffer_index] = state
        self.action[self.buffer_index] = action
        self.reward[self.buffer_index] = reward
        self.next_state[self.buffer_index] = next_state
        self.done[self.buffer_index] = done

        self.buffer_index = (self.buffer_index + 1) % self.memory_size
        self.full = self.full or self.buffer_index == 0

    def sample(self, batch_size, chunk_size):
        """
        (batch_size, chunk_size, input_size)
        """
        last_filled_index = self.buffer_index - chunk_size + 1
        assert self.full or (
            last_filled_index > batch_size
        ), "too short dataset or too long chunk_size"
        sample_index = np.random.randint(
            0, self.capacity if self.full else last_filled_index, batch_size
        ).reshape(-1, 1)
        chunk_length = np.arange(chunk_size).reshape(1, -1)

        sample_index = (sample_index + chunk_length) % self.memory_size

        state = torch.as_tensor(self.state[sample_index]).float()
        next_state = torch.as_tensor(self.next_state[sample_index]).float()

        action = torch.as_tensor(self.action[sample_index])
        reward = torch.as_tensor(self.reward[sample_index])
        done = torch.as_tensor(self.done[sample_index])

        sample = TensorDict(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
            },
            batch_size=batch_size,
            device=self.device,
        )
        return sample
