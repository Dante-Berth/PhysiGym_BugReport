import torch
import numpy as np
import torch.optim as optim
from torch.distributions.normal import Normal
from typing import Union
import os, sys

absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("Deep_Reinforcement_Learning")
    + len("Deep_Reinforcement_Learning")
]
sys.path.append(absolute_path)
from src.agents.core_agents import CoreAgents
from src.utils.update_weights import soft_update

python_program_name = os.path.splitext(os.path.basename(__file__))[0].upper()
APPLICATION_NAME = f"[{python_program_name}]"


class ppo(CoreAgents):
    def __init__(self, config, model: torch.nn.Module) -> None:
        super().__init__(self, config=config, model=model)
        print("Init: " + APPLICATION_NAME)

    def _init_models(self, model: torch.nn.Module):
        self.dict_models = {}
        self.dict_optimizers = {}
        self.dict_models["model"] = model(self.env).to(self.config["device"])
        self.dict_optimizers["model"] = optim.Adam(
            list(self.dict_models["model"].parameters()), lr=self.config["lr_model"]
        )

    def _select_action(
        self, state: torch.tensor, action: Union[torch.tensor, None] = None
    ):
        action_mean = self.dict_models["model"].actor_mean(state)
        action_logstd = self.dict_models["model"].actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(-1),
            probs.entropy().sum(-1),
            self.dict_models["model"].critic(state),
        )

    def _validation(
        self,
        global_step: int,
        step_multiple_validation: int,
        value_loss: torch.tensor,
        policy_loss: torch.tensor,
        entropy: torch.tensor,
        old_approx_kl: torch.tensor,
        approx_kl: torch.tensor,
        clipfracs: list,
        explained_var: torch.tensor,
    ):
        if global_step % step_multiple_validation == 0:
            values_dict = {
                "value_loss": value_loss.item(),
                "policy_loss": policy_loss.item(),
                "entropy": entropy.item(),
                "old_approx_kl": old_approx_kl.item(),
                "approx_kl": approx_kl.item(),
                "clipfracs": np.mean(clipfracs),
                "explained_var": explained_var.item(),
            }
            if self.config["wandb"]:
                self.writer = values_dict
            else:
                for key in values_dict:
                    self.writer.add_scalar(f"{key}", values_dict[key], global_step)

    def _trainer(self):
        # copied from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
        env = self.env
        self.config["batch_size"] = self.config["num_envs"] * self.config["num_steps"]
        self.config["minibatch_size"] = int(
            self.config["batch_size"] // self.config["num_minibatches"]
        )
        num_iterations = self.config["total_timesteps"] // self.config["batch_size"]
        tensor_shapes = {
            "obs": (self.config["num_steps"], self.config["num_envs"])
            + env.single_observation_space.shape,
            "actions": (self.config["num_steps"], self.config["num_envs"])
            + env.single_action_space.shape,
            "logprobs": (self.config["num_steps"], self.config["num_envs"]),
            "rewards": (self.config["num_steps"], self.config["num_envs"]),
            "dones": (self.config["num_steps"], self.config["num_envs"]),
            "values": (self.config["num_steps"], self.config["num_envs"]),
        }

        data = {
            key: torch.zeros(shape).to(self.config["device"])
            for key, shape in tensor_shapes.items()
        }

        global_step = 0
        next_obs, _ = env.reset(seed=self.seed_env)
        next_obs = torch.Tensor(next_obs[0]).to(self.config["device"])
        next_done = torch.zeros(self.config["num_envs"]).to(self.config["device"])
        for iteration in range(1, num_iterations + 1):
            if self.config["anneal_lr"]:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                self.dict_optimizers["model"].param_groups[0]["lr"] = (
                    frac * self.config["lr_model"]
                )
            for step in range(0, self.config["num_envs"]):
                global_step += self.config["num_envs"]
                data["obs"][step] = next_obs
                data["dones"][step] = next_done
                with torch.no_grad():
                    action, logprob, _, value = self.dict_models[
                        "model"
                    ].get_action_and_value(next_obs)
                    "values"[step] = value.flatten()
                data["actions"][step] = action
                data["logprobs"][step] = logprob

                next_obs, reward, terminations, truncations, infos = env.step(
                    action.cpu().numpy()
                )
                next_done = np.logical_or(terminations, truncations)
                data["rewards"][step] = (
                    torch.tensor(reward).to(self.config["device"]).view(-1)
                )
                next_obs, next_done = (
                    torch.Tensor(next_obs).to(self.config["device"]),
                    torch.Tensor(next_done).to(self.config["device"]),
                )

            # bootstrap value if not done
            with torch.no_grad():
                next_value = (
                    self.dict_models["model"].get_value(next_obs).reshape(1, -1)
                )
                advantages = torch.zeros_like("rewards").to(self.config["device"])
                lastgaelam = 0
                for t in reversed(range(self.config["num_steps"])):
                    if t == self.config["num_steps"] - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - data["dones"][t + 1]
                        nextvalues = data["values"][t + 1]
                    delta = (
                        data["rewards"][t]
                        + self.config["gamma"] * nextvalues * nextnonterminal
                        - data["values"][t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.config["gamma"]
                        * self.config["gae_lambda"]
                        * nextnonterminal
                        * lastgaelam
                    )
                returns = advantages + data["values"]

            # flatten the batch
            b_obs = data["obs"].reshape((-1,) + env.single_observation_space.shape)
            b_logprobs = data["logprobs"].reshape(-1)
            b_actions = data["actions"].reshape((-1,) + env.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = data["values"].reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.config["batch_size"])
            clipfracs = []
            for epoch in range(self.config["update_epochs"]):
                np.random.shuffle(b_inds)
                for start in range(
                    0, self.config["batch_size"], self.config["minibatch_size"]
                ):
                    end = start + self.config["minibatch_size"]
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.config[
                        "model"
                    ].get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.config["clip_coef"])
                            .float()
                            .mean()
                            .item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if self.config["norm_adv"]:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio,
                        1 - self.config["clip_coef"],
                        1 + self.config["clip_coef"],
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.config["clip_vloss"]:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.config["clip_coef"],
                            self.config["clip_coef"],
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss
                        - self.config["ent_coef"] * entropy_loss
                        + v_loss * self.config["vf_coef"]
                    )

                    self.dict_optimizers["model"].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.dict_models["model"].parameters(),
                        self.config["max_grad_norm"],
                    )
                    self.dict_optimizers["model"].step()

                if (
                    self.config["target_kl"] is not None
                    and approx_kl > self.config["target_kl"]
                ):
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )
            self._validation(
                global_step=global_step,
                step_multiple_validation=self.config["step_multiple_validation"],
                value_loss=v_loss,
                policy_loss=pg_loss,
                entropy=entropy_loss,
                old_approx_kl=old_approx_kl,
                approx_kl=approx_kl,
                clipfracs=clipfracs,
                explained_var=explained_var,
            )
        self.env.close()

    def _trainer(self):
        env = self.env
        state, _ = env.reset(seed=self.seed_env)
        state = state[0]
        total_reward = 0
        for step in range(self.config["total_steps"]):
            if step < self.config["learning_start"]:
                action = torch.tensor(
                    env.action_space.sample(),
                    dtype=torch.float,
                    device=self.config["device"],
                )
            else:
                action = self._select_action(state).detach()

            next_state, reward, terminated, truncated, _ = env.step(action.numpy())

            self.replay_buffer.push(
                [state, action, reward, next_state, (truncated and terminated)]
            )
            total_reward += reward
            state = next_state

            if step >= self.config["learning_start"]:
                self._training(step)
