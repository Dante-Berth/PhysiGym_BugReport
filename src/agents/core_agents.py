import torch
import time
import yaml
import os, sys
import wandb

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("Deep_Reinforcement_Learning")
    + len("Deep_Reinforcement_Learning")
]
sys.path.append(absolute_path)
from src.utils.utils_gym import init_seed, wrapper_env
from src.buffers.simple_replay_buffer import replay_buffer


class CoreAgents:
    def __init__(
        self,
        config_path: str = None,
        env=None,
        *args,
        **kwargs,
    ) -> None:
        self.env = env
        self._init_agent(config_path)
        self._init_models(*args)  # init models
        self._init_replay_buffer()  # init the replay buffer

    def _init_agent(self, config_path):
        self._init_config(config_path=config_path)
        self._init_folders()  # not to modify
        self._init_writer()  # not to modify
        self._init_seed()  # not to modify

    def _init_folders(self):
        os.makedirs(name=f"{self.config['run']}", exist_ok=True)  # create run folder
        os.makedirs(
            name=f"{self.config['run']}/{self.config['run_name']}", exist_ok=True
        )
        import time

        self.time = int(time.time())
        self.run_name = f"{self.config['env_id']}__{self.config['exp_name']}__{self.config['seed']}__{self.time}"

    def _init_config(self, config_path: str):
        """
        Init config entity

        Args:
            config_path (str): The configuration path
        """
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def _init_models(self, *args):
        self.dict_models: dict = {}
        self.dict_optimizers: dict = {}
        device = self.config["device"]["model"]
        self.dict_models["model"] = args[0].to(device)
        self.dict_optimizers["model"] = torch.optim.Adam(
            list(self.dict_models["model"].parameters()), lr=self.config["lr_model"]
        )

    def _init_writer(self) -> None:
        # We initialize the writer
        self.writer = wandb.init(
            dir=f"{self.config['run']}/{self.config['run_name']}",
            project="Deep_Reinforcement_Learning",
            config=self.config,
            monitor_gym=True,
            save_code=True,
            sync_tensorboard=True,
        )

    def _init_seed(self):
        self.seed_env = self.config["seed"]
        init_seed(self.seed_env)

    def _init_env(self):
        self.time = int(time.time())
        self.run_name = f"{self.config['env_id']}__{self.config['exp_name']}__{self.config['seed']}__{self.time}"
        self.env = wrapper_env(
            self.config["env_id"],
            self.seed_env,
            self.config["idx"],
            self.config["capture_video"],
            run_name=f"{self.config['run']}/{self.config['run_name']}/videos/",
        )

    def _init_replay_buffer(self) -> None:
        """
        Initialize replay buffer

        Args:

        Raises:
            NotImplementedError: This method should be implemented by the subclass.
        """
        raise NotImplementedError("To be implemented")

    def _save_models(self):
        """
        Save the models to the specified directory.
        """
        self.model_path = f"{self.config['run']}/{self.config['run_name']}/models/"
        os.makedirs(self.model_path, exist_ok=True)
        for name_model in self.dict_models:
            model_scripted = torch.jit.script(
                self.dict_models[name_model]
            )  # Export to TorchScript
            model_path = os.path.join(self.model_path, f"model_{name_model}.pt")
            model_scripted.save(model_path)  # Save

    def _load_models(self, PATH: str):
        """
        Load models from the specified directory.

        Args:
            PATH (str): Path to the directory containing the models.
        """
        list_files = os.listdir(PATH)
        self.dict_models = {} if self.dict_models is None else self.dict_models
        for filename in list_files:
            if filename.endswith(".pt"):
                name_model = os.path.splitext(filename)[0]  # Extract model name
                self.dict_models[name_model] = torch.jit.load(
                    PATH + filename
                )  # Extract load model

    def _select_action(self, state: torch.tensor, *args, **kwargs):
        """
        Select an action based on the current state.

        Args:
            state (torch.tensor): Current state of the environment.

        Raises:
            NotImplementedError: This method should be implemented by the subclass.
        """
        raise NotImplementedError("To be implemented")

    def _agent_training(self, step: int) -> None:
        """
        Train the agent for the given step.

        Args:
            step (int): Current training step.

        Raises:
            NotImplementedError: This method should be implemented by the subclass.
        """
        raise NotImplementedError("To be implemented")

    def _validation(self, step: int, step_multiple_validation: int) -> None:
        """
        Perform validation at the given step.

        Args:
            step (int): Current training step.
            step_multiple_validation (int): Frequency of validation steps.

        Raises:
            NotImplementedError: This method should be implemented by the subclass.
        """
        raise NotImplementedError("To be implemented")
