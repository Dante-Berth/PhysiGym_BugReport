import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import os, sys
import gymnasium as gym
from embedding import physicell

absolute_path: str = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("Deep_Reinforcement_Learning")
    + len("Deep_Reinforcement_Learning")
]
sys.path.append(absolute_path)
from src.agents.agents_td3 import td3


class QNetwork(nn.Module):
    def __init__(self, env, config: dict):
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=2,
            out_features=config["layer_1_out_features"],
        )
        self.fc2 = nn.Linear(
            in_features=config["layer_1_out_features"],
            out_features=config["layer_2_out_features"],
        )
        self.fc3 = nn.Linear(in_features=config["layer_2_out_features"], out_features=1)

    def forward(self, x, a):
        x = torch.cat((x, a), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env, config: dict):
        super().__init__()
        self.fc1 = nn.Linear(
            in_features=1,
            out_features=config["layer_1_out_features"],
        )
        """
        self.fc2 = SpectralConv0d(
            in_channels=config["layer_1_out_features"],
            modes=32,
        )
        """

        self.fc2 = nn.Linear(
            in_features=config["layer_1_out_features"],
            out_features=config["layer_2_out_features"],
        )

        self.fc_mu = nn.Linear(
            in_features=config["layer_2_out_features"],
            out_features=np.prod(env.action_space.shape),
        )
        # action rescalinglearning_start
        self.action_scale = torch.tensor(
            (env.action_space.high - env.action_space.low) / 2.0,
            dtype=torch.float32,
        )
        self.action_bias = torch.tensor(
            (env.action_space.high + env.action_space.low) / 2.0,
            dtype=torch.float32,
        )

    def forward(self, x):
        if len(x.size()) == 0:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


os.chdir("../PhysiCell")


class DummyModelWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, variable_name: str):
        super().__init__(env)
        self.cell_count_target = int(self.x_root.xpath("//cell_count_target")[0].text)
        if not isinstance(variable_name, str):
            raise ValueError(
                f"Expected variable_name to be of type str, but got {type(variable_name).__name__}"
            )

        self.variable_name = variable_name

    def reset(self, seed=None, options={}):
        o_observation, d_info = self.env.reset(seed=seed, options=options)
        o_observation = o_observation.astype(float) / self.cell_count_target
        return o_observation[0], d_info

    def step(self, r_dose: float):
        d_action = {self.variable_name: np.array([r_dose])}
        o_observation, r_reward, b_terminated, b_truncated, d_info = self.env.step(
            d_action
        )
        o_observation = o_observation.astype(float) / self.cell_count_target
        return o_observation, r_reward, b_terminated, b_truncated, d_info

    @property
    def action_space(self):
        return self.env.action_space[self.variable_name]


import physigym

env = gym.make(
    "physigym/ModelPhysiCellEnv-v0",
    # settingxml='config/PhysiCell_settings.xml',
    # render_mode='rgb_array',
    # render_fps=10
)
env = DummyModelWrapper(env, variable_name="drug_dose")

td3 = td3(
    config_path="../Deep_Reinforcement_Learning/src/_config/_physigym_td3/config_0.yaml",
    actor=Actor,
    critic=QNetwork,
    env=env,
)
td3._trainer()
