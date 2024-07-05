import pytest
import sys
import os
import shutil

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("Deep_Reinforcement_Learning")
    + len("Deep_Reinforcement_Learning")
]
sys.path.append(absolute_path)
from src.agents.core_agents import CoreAgents

CONFIG = {
    "run": "your_run_folder",
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
    "device": {
        "replay_buffer": "cpu",
        "model": "cpu",  # Device for replay buffer
    },
    "wandb": False,
    "batch_size": 32,  # Batch size
    "total_steps": 1000,  # Total steps
    "lr_model": 10,  # Learning start steps
}


@pytest.fixture
def core_agents(config=CONFIG):
    return CoreAgents(CONFIG)


def test_init_CoreAgents(core_agents):
    """Function to test if the init function is correct"""

    assert os.listdir(f"{CONFIG['run']}") is not None
    assert os.listdir(f"{CONFIG['run']}/{CONFIG['run_name']}") is not None
    assert core_agents.env is not None
    assert core_agents.seed_env is not None
    assert core_agents.writer is not None
    assert core_agents.time is not None
    assert (
        core_agents.run_name
        == f"{CONFIG['env_id']}__{CONFIG['exp_name']}__{CONFIG['seed']}__{core_agents.time}"
    )
    assert core_agents.replay_buffer is not None

    model_path = f"{CONFIG['run']}/{CONFIG['run_name']}/models/"
    assert core_agents.dict_models["model"] is not None
    core_agents._save_models()
    assert model_path == core_agents.model_path
    assert os.listdir(core_agents.model_path) is not None
    core_agents._load_models(model_path)
    shutil.rmtree(CONFIG["run"])


CONFIG["wandb"] = True
CONFIG["run"] = "wandb_run"


def test_wandb_CoreAgents(core_agents):
    assert core_agents.writer is not None
    assert os.listdir(f"{CONFIG['run']}") is not None
    assert os.listdir(f"{CONFIG['run']}/{CONFIG['run_name']}") is not None
    shutil.rmtree(CONFIG["run"])


if __name__ == "__main__":
    pytest.main()
