import torch
import torch.nn.functional as F
import torch.optim as optim
import os, sys
import numpy as np

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("Deep_Reinforcement_Learning")
    + len("Deep_Reinforcement_Learning")
]
sys.path.append(absolute_path)
from src.agents.agents_td3 import td3
from src.utils.update_weights import soft_update
from src.nn.fno.fno1d import SpectralConv1d, SpectralConv0d
from src.nn.kan.kan_layer import KANLinear
from src.agents.neural_nets.actor_critic import Actor_mean_logstd, QNetwork
import tensordict


class sac(td3):
    def __init__(
        self,
        config_path: str = None,
        actor: torch.nn.Module = None,
        critic: torch.nn.Module = None,
        env=None,
    ) -> None:
        super(td3).__init__(
            config_path=config_path,
            actor=actor,
            critic=critic,
            env=env,
        )
        self.target_entropy = -np.prod(
            self.env.action_space.shape
        )  # This is the target entropy

        self.alpha: torch.Tensor = torch.tensor(
            self.config["alpha"], dtype=torch.float32, requires_grad=False
        ).to(self.config["device"])
        self.log_alpha = self.alpha.log().to(self.config["device"])
        self.log_alpha.requires_grad = True
        self.dict_optimizers["alpha_optimizer"] = optim.Adam(
            [self.log_alpha], lr=self.config["lr_critic"]
        )
        self.config["action_scale"] = torch.tensor(
            (self.env.action_space.high - self.env.action_space.low) / 2.0,
            dtype=torch.float32,
        )

        self.config["action_bias"] = torch.tensor(
            (self.env.action_space.high + self.env.action_space.low) / 2.0,
            dtype=torch.float32,
        )
        self.dict_target_models.pop("target_actor")

    def _select_action(
        self, state: torch.Tensor, with_no_grad: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            state (torch.Tensor): _description_
            with_no_grad (bool, optional): _description_. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        action_mean, action_logstd = self.dict_models["actor"](state)
        dist = torch.distributions.Normal(action_mean, action_logstd.exp())
        x_t = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(input=x_t)
        action = y_t * self.config["action_scale"] + self.config["action_bias"]
        log_prob = dist.log_prob(value=x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.config["action_scale"] * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        action_mean = (
            torch.tanh(action_mean) * self.config["action_scale"]
            + self.config["action_bias"]
        )
        action = action.detach() if with_no_grad else action
        action_mean = action_mean.detach() if with_no_grad else action_mean
        return action, log_prob, action_mean

    def _agent_training(self, step, dict_tensors: tensordict) -> None:
        state = dict_tensors["state"].squeeze(1)
        action = dict_tensors["action"].squeeze(1)
        next_state = dict_tensors["next_state"].squeeze(1)
        reward = dict_tensors["reward"].squeeze(1)
        done = dict_tensors["done"].squeeze(1)
        with torch.no_grad():
            next_actions, next_log_probs, _ = self._select_action(next_state)

            next_critic_1_value = self.dict_target_models["target_critic_1"](
                next_state, next_actions
            )
            next_critic_2_value = self.dict_target_models["target_critic_2"](
                next_state, next_actions
            )

            min_critic_next_target = (
                torch.min(next_critic_1_value, next_critic_2_value)
                - self.alpha.item() * next_log_probs
            )
            next_critic_value = reward + (1 - done) * self.config["gamma"] * (
                min_critic_next_target
            )

        critic_1_values = self.dict_models["critic_1"](state, action)
        critic_2_values = self.dict_models["critic_2"](state, action)
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
                critic_1_pi = self.dict_target_models["target_critic_1"](state, pi)
                critic_2_pi = self.dict_target_models["target_critic_2"](state, pi)
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

    def _validation(
        self,
        step: int,
        step_multiple_validation: int,
        critic_1_values: torch.tensor,
        critic_2_values: torch.tensor,
        actor_loss: torch.tensor,
        alpha_loss: torch.tensor,
        alpha: torch.tensor,
    ):
        if (
            step % step_multiple_validation == 0
            and step % self.config["policy_frequency"] == 0
        ):
            values_dict = {
                "critic_1_values": critic_1_values.mean().item(),
                "critic_2_values": critic_2_values.mean().item(),
                "actor_loss": actor_loss.item(),
                "alpha_loss": alpha_loss.item(),
                "alpha": alpha.item(),
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
                action, _, _ = self._select_action(state=state, with_no_grad=True)
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
    sac = sac(
        config_path="src/_config/sac/config_0.yaml",
        actor=Actor_mean_logstd,
        critic=QNetwork,
    )
    sac._trainer()
