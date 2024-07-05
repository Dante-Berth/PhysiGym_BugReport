import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
import os, sys
import numpy as np

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("Deep_Reinforcement_Learning")
    + len("Deep_Reinforcement_Learning")
]
sys.path.append(absolute_path)
from src.agents.agents_sac import sac
from src.utils.update_weights import soft_update
import tensordict
from src.nn.fno.fno1d import SpectralConv1d, SpectralConv0d
from src.nn.kan.kan_layer import KANLinear


class meta_sac(sac):
    def _agent_training(self, step, dict_tensors: tensordict) -> None:
        state = dict_tensors["state"].squeeze(1)
        action = dict_tensors["action"].squeeze(1)
        next_state = dict_tensors["next_state"].squeeze(1)
        reward = dict_tensors["reward"].squeeze(1)
        done = dict_tensors["done"].squeeze(1)
        tensor_alpha = self.alpha.repeat(action.size(0), 1).detach()
        with torch.no_grad():
            next_actions, next_log_probs, _ = self._select_action(next_state)

            next_critic_1_value = self.dict_target_models["target_critic_1"](
                next_state, next_actions, tensor_alpha
            )
            next_critic_2_value = self.dict_target_models["target_critic_2"](
                next_state, next_actions, tensor_alpha
            )

            min_critic_next_target = (
                torch.min(next_critic_1_value, next_critic_2_value)
                - self.alpha.item() * next_log_probs
            )
            next_critic_value = reward + (1 - done) * self.config["gamma"] * (
                min_critic_next_target
            )
        critic_1_values = self.dict_models["critic_1"](state, action, tensor_alpha)
        critic_2_values = self.dict_models["critic_2"](state, action, tensor_alpha)
        critics_loss: torch.Tensor = F.mse_loss(
            input=critic_1_values, target=next_critic_value
        ) + F.mse_loss(input=critic_2_values, target=next_critic_value)

        # optimize the model
        self.dict_optimizers["critic_1_2"].zero_grad()
        critics_loss.backward()
        self.dict_optimizers["critic_1_2"].step()

        if step % self.config["policy_frequency"] == 0:
            for _ in range(self.config["policy_frequency"]):
                pi, log_pi, _ = self._select_action(next_state)
                critic_1_pi = self.dict_target_models["target_critic_1"](
                    state, pi, tensor_alpha
                )
                critic_2_pi = self.dict_target_models["target_critic_2"](
                    state, pi, tensor_alpha
                )
                actor_loss = -(
                    torch.min(critic_1_pi, critic_2_pi) - self.config["alpha"] * log_pi
                ).mean()

                self.dict_optimizers["actor"].zero_grad()
                actor_loss.backward()
                self.dict_optimizers["actor"].step()

                _, log_pi, _ = self._select_action(next_state, with_no_grad=True)
                alpha_loss = (
                    -self.log_alpha.exp() * (log_pi + self.target_entropy)
                ).mean()

                self.dict_optimizers["alpha_optimizer"].zero_grad()
                alpha_loss.backward()
                self.dict_optimizers["alpha_optimizer"].step()
                self.alpha = self.log_alpha.exp()

            soft_update(
                self.dict_models["critic_1"],
                self.dict_target_models["target_critic_1"],
                tau=self.config["tau"],
            )
            soft_update(
                self.dict_models["critic_2"],
                self.dict_target_models["target_critic_2"],
                tau=self.config["tau"],
            )
            self._validation(
                step,
                step_multiple_validation=self.config["step_multiple_validation"],
                critic_1_values=critic_1_values,
                critic_2_values=critic_2_values,
                actor_loss=actor_loss,
                alpha_loss=alpha_loss,
                alpha=self.alpha,
            )


if __name__ == "__main__":
    import torch.nn as nn
    import numpy as np

    class QNetwork(nn.Module):
        def __init__(self, env, config: dict):
            super().__init__()
            self.fc1 = nn.Linear(
                in_features=np.array(env.single_observation_space.shape).prod()
                + np.prod(env.single_action_space.shape)
                + 1,
                out_features=config["layer_1_out_features"],
            )
            self.fc2 = nn.Linear(
                in_features=config["layer_1_out_features"],
                out_features=config["layer_2_out_features"],
            )
            self.fc3 = nn.Linear(
                in_features=config["layer_2_out_features"], out_features=1
            )

        def forward(self, x, a, alpha):
            x = torch.cat((x, a, alpha), dim=1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    class Actor(nn.Module):
        def __init__(self, env, config: dict):
            super().__init__()
            self.fc1 = nn.Linear(
                np.array(env.single_observation_space.shape).prod(),
                out_features=config["layer_1_out_features"],
            )
            self.fc2 = nn.Linear(
                in_features=config["layer_1_out_features"],
                out_features=config["layer_2_out_features"],
            )
            self.fc_mu = nn.Linear(
                in_features=config["layer_2_out_features"],
                out_features=np.prod(env.action_space.shape),
            )
            self.fc_logstd = nn.Linear(
                in_features=config["layer_2_out_features"],
                out_features=np.prod(env.action_space.shape),
            )
            # action rescaling
            self.action_scale = torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            )
            self.action_bias = torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            )

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            mean = self.fc_mu(x)
            log_std = self.fc_logstd(x)
            log_std = torch.tanh(log_std)
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
                log_std + 1
            )  # From SpinUp / Denis Yarats

            return mean, log_std

    meta_sac = meta_sac(
        config_path="src/_config/sac/config_0.yaml", actor=Actor, critic=QNetwork
    )
    meta_sac._trainer()
