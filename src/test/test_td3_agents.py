import pytest
import sys
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("Deep_Reinforcement_Learning")
    + len("Deep_Reinforcement_Learning")
]
sys.path.append(absolute_path)
from src.agents.agents_td3 import td3

CONFIG = {
    "run": "td3",
    "run_name": "your_run_name",
    "seed": 42,  # Your seed
    "env_id": "Pendulum-v1",  # Your environment ID
    "exp_name": "exp1",  # Your experiment name
    "idx": 0,  # Your index
    "capture_video": False,  # Whether to capture video
    "replay_buffer": {
        "memory_size": 1000,  # Size of replay buffer
        "list_named_tuple": [
            "state",
            "action",
            "next_state",
            "reward",
            "done",
        ],  # Your list of named tuples (if any)
    },
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "tau": 5e-3,
    "exploration_noise": 0.1,
    "policy_frequency": 2,
    "noise_clip": 0.5,
    "device": "cpu",
    "batch_size": 164,  # Batch size
    "total_steps": 1e-6,  # Total steps
    "learning_start": 10,  # Learning start steps
    "wandb": True,
    "lr_actor": 1e-3,
    "lr_critic": 1e-3,
    "actor": {"layer_1_out_features": int(256), "layer_2_out_features": int(256)},
    "critic_1": {"layer_1_out_features": int(256), "layer_2_out_features": int(256)},
    "critic_2": {"layer_1_out_features": int(256), "layer_2_out_features": int(256)},
}


class QNetwork(nn.Module):
    def __init__(self, env, config: dict):
        super().__init__()
        self.fc1 = nn.LazyLinear(
            out_features=config["layer_1_out_features"],
        )
        self.fc2 = nn.Linear(
            in_features=config["layer_1_out_features"],
            out_features=config["layer_2_out_features"],
        )
        self.fc_mu = nn.Linear(
            in_features=config["layer_2_out_features"], out_features=1
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env, config: dict):
        super().__init__()
        self.fc1 = nn.LazyLinear(
            out_features=config["layer_1_out_features"],
        )
        self.fc2 = nn.Linear(
            in_features=config["layer_1_out_features"],
            out_features=config["layer_2_out_features"],
        )
        self.fc_mu = nn.Linear(
            in_features=config["layer_2_out_features"],
            out_features=int(
                np.prod(env.action_space.shape)
            ),  # needed for torch script
        )
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


@pytest.fixture
def td3_agents(config=CONFIG, actor=Actor, critic=QNetwork):
    return td3(CONFIG, actor=actor, critic=critic)


def test_init_CoreAgents(td3_agents):
    """Function to test if the init function is correct"""

    assert os.listdir(f"{CONFIG['run']}") is not None
    assert os.listdir(f"{CONFIG['run']}/{CONFIG['run_name']}") is not None
    assert td3_agents.env is not None
    assert td3_agents.seed_env is not None
    assert td3_agents.writer is not None
    assert td3_agents.time is not None
    assert (
        td3_agents.run_name
        == f"{CONFIG['env_id']}__{CONFIG['exp_name']}__{CONFIG['seed']}__{td3_agents.time}"
    )
    assert td3_agents.replay_buffer is not None
    shutil.rmtree(CONFIG["run"])
    """
    model_path = f"{CONFIG['run']}/{CONFIG['run_name']}/models/"
    assert td3_agents.dict_models["actor"] is not None
    td3_agents._save_models()
    assert model_path == td3_agents.model_path
    assert os.listdir(td3_agents.model_path) is not None
    td3_agents._load_models(model_path)
    shutil.rmtree(CONFIG["run"])
    """


if __name__ == "__main__":
    pytest.main()
