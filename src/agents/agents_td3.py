import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
import os, sys

absolute_path: str = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("Deep_Reinforcement_Learning")
    + len("Deep_Reinforcement_Learning")
]
sys.path.append(absolute_path)
import tensordict
from src.agents.core_agents import CoreAgents
from src.utils.update_weights import soft_update
from src.nn.fno.fno1d import SpectralConv1d, SpectralConv0d
from src.nn.kan.kan_layer import KANLinear


class td3(CoreAgents):
    def __init__(
        self,
        config_path: str = None,
        actor: torch.nn.Module = None,
        critic: torch.nn.Module = None,
        env=None,
    ) -> None:
        self.env = env
        self._init_agent(config_path)  # init the config files
        self._init_replay_buffer()
        self._init_models(actor, critic)  # init models

    def _init_agent(self, config_path):
        return super()._init_agent(config_path)

    def _init_models(self, actor: torch.nn.Module, critic: torch.nn.Module):
        self.dict_models = {}
        self.dict_models["actor"] = actor(env=self.env, config=self.config["actor"]).to(
            self.config["device"]
        )
        self.dict_models["critic_1"] = critic(
            env=self.env, config=self.config["critic_1"]
        ).to(self.config["device"])
        self.dict_models["critic_2"] = critic(
            env=self.env, config=self.config["critic_2"]
        ).to(self.config["device"])

        self.dict_optimizers = {}
        self.dict_optimizers["actor"] = optim.Adam(
            list(self.dict_models["actor"].parameters()), lr=self.config["lr_actor"]
        )
        self.dict_optimizers["critic_1_2"] = optim.Adam(
            list(self.dict_models["critic_1"].parameters())
            + list(self.dict_models["critic_2"].parameters()),
            lr=self.config["lr_critic"],
        )

        self.dict_target_models = {}
        self.dict_target_models["target_actor"] = copy.deepcopy(
            self.dict_models["actor"]
        ).to(self.config["device"])
        self.dict_target_models["target_critic_1"] = copy.deepcopy(
            self.dict_models["critic_1"]
        ).to(self.config["device"])
        self.dict_target_models["target_critic_2"] = copy.deepcopy(
            self.dict_models["critic_2"]
        ).to(self.config["device"])

    def _init_replay_buffer(self) -> None:
        from src.buffers.simple_replay_buffer import ReplayBuffer
        import numpy as np

        self.replay_buffer = ReplayBuffer(
            state_dim=np.prod(self.env.observation_space.shape),
            action_dim=np.prod(self.env.action_space.shape),
            device=self.config["device"],
            memory_size=self.config["replay_buffer"]["memory_size"],
        )

    def _select_action(
        self, state: torch.tensor, stochastic: bool, with_no_grad: bool = True
    ):
        action = self.dict_models["actor"](state)
        action = action.detach() if with_no_grad else action
        if stochastic:
            action += torch.normal(
                0,
                self.dict_models["actor"].action_scale
                * self.config["exploration_noise"],
            )
        action = action.cpu()
        action = torch.clip(
            action,
            torch.Tensor(self.env.action_space.low),
            torch.Tensor(self.env.action_space.high),
        )

        return action

    def _agent_training(self, step, dict_tensors: tensordict):
        state = dict_tensors["state"].squeeze(1)
        action = dict_tensors["action"].squeeze(1)
        next_state = dict_tensors["next_state"].squeeze(1)
        reward = dict_tensors["reward"].squeeze(1)
        done = dict_tensors["done"].squeeze(1)
        with torch.no_grad():
            clipped_noise = (
                torch.randn_like(action, device=self.config["device"])
                * self.config["policy_noise"]
            ).clamp(
                -self.config["noise_clip"], self.config["noise_clip"]
            ) * self.dict_target_models["target_actor"].action_scale

            next_state_actions = (
                self.dict_target_models["target_actor"](next_state) + clipped_noise
            ).clamp(
                self.env.action_space.low[0],
                self.env.action_space.high[0],
            )
            # envs.step(actions) = array([[0.992511  , 0.12215547, 0.9636772 ]], dtype=float32), array([-0.08668639]), array([False]), array([False]), {})
            next_critic_1_value = self.dict_target_models["target_critic_1"](
                next_state, next_state_actions
            )
            next_critic_2_value = self.dict_target_models["target_critic_2"](
                next_state, next_state_actions
            )
            min_critic_next_target = torch.min(next_critic_1_value, next_critic_2_value)
            next_critic_value = reward + (1 - done) * self.config["gamma"] * (
                min_critic_next_target
            )
        critic_1_values = self.dict_models["critic_1"](state, action)
        critic_2_values = self.dict_models["critic_2"](state, action)
        critics_loss = F.mse_loss(critic_1_values, next_critic_value) + F.mse_loss(
            critic_2_values, next_critic_value
        )

        # optimize the model
        self.dict_optimizers["critic_1_2"].zero_grad()
        critics_loss.backward()
        self.dict_optimizers["critic_1_2"].step()

        if step % self.config["policy_frequency"] == 0:
            actor_loss = -self.dict_models["critic_1"](
                state, self.dict_models["actor"](state)
            ).mean()
            self.dict_optimizers["actor"].zero_grad()
            actor_loss.backward()
            self.dict_optimizers["actor"].step()
            soft_update(
                self.dict_models["actor"],
                self.dict_target_models["target_actor"],
                tau=self.config["tau"],
            )
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
                critics_loss=critics_loss,
                actor_loss=actor_loss,
            )

    def _validation(
        self,
        step: int,
        step_multiple_validation: int,
        critic_1_values: torch.tensor,
        critic_2_values: torch.tensor,
        critics_loss: torch.tensor,
        actor_loss: torch.tensor,
    ):
        if (
            step % step_multiple_validation == 0
            and step % self.config["policy_frequency"] == 0
        ):
            values_dict = {
                "loss/critic_1_values": critic_1_values.mean().item(),
                "loss/critic_2_values": critic_2_values.mean().item(),
                "loss/critics_loss": critics_loss.item() / 2.0,
                "loss/actor_loss": actor_loss.item(),
            }
            self.writer.log(values_dict)

    def _trainer(self):
        env = self.env
        state, _ = env.reset(seed=self.seed_env)
        one_epoch_reward = 0
        sum_reward = 0
        for step in range(self.config["total_steps"]):
            state = torch.tensor(
                state,
                dtype=torch.float,
                device=self.config["device"],
            )
            if step < self.config["learning_start"]:
                action = torch.tensor(
                    env.action_space.sample(),
                    dtype=torch.float,
                    device=self.config["device"],
                )
            else:
                action = self._select_action(
                    state=state, stochastic=True, with_no_grad=True
                )
            action = action.numpy()
            next_state, reward, terminated, truncations, _ = env.step(action)
            one_epoch_reward += reward
            state = state.numpy()
            self.replay_buffer.push(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=(truncations or terminated),
            )
            if truncations or terminated:
                state, _ = env.reset(seed=self.seed_env + step)
                self.writer.log({"reward/one_epoch_reward": one_epoch_reward})
                sum_reward = (
                    2 / (30 + 1) * one_epoch_reward + (1 - 2 / (30 + 1)) * sum_reward
                )
                one_epoch_reward = 0
                self.writer.log({"reward/sum_reward": sum_reward})
                continue

            state = next_state

            if step >= self.config["learning_start"]:
                dict_tensor = self.replay_buffer.sample(self.config["batch_size"], 1)
                self._agent_training(step, dict_tensor)
            if step % 10000 == 0:
                print("Step", step)
                print("Cumulative reward", sum_reward)


if __name__ == "__main__":
    import torch.nn as nn
    import numpy as np

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
            self.fc3 = nn.Linear(
                in_features=config["layer_2_out_features"], out_features=1
            )

        def forward(self, x, a):
            x = torch.cat((x, a), dim=1)
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
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.tanh(self.fc_mu(x))
            return x * self.action_scale + self.action_bias

    td3 = td3(config_path="src/_config/td3/config_0.yaml", actor=Actor, critic=QNetwork)
    td3._trainer()
