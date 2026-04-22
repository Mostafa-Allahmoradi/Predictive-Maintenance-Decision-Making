"""pdm/dyna_agent.py
==================
Dyna-Q agent: model-based RL extending DoubleDQNAgent with a jointly-trained
world model (Sutton 1990 Dyna-Q + deep approximations).

Public API
----------
    DynaQAgent — DoubleDQNAgent + world model + Dyna planning updates
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn, optim

from pdm import config as CFG
from pdm.ddqn_agent import DoubleDQNAgent, MLP, Transition, device  # noqa: F401


class DynaQAgent(DoubleDQNAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str,
        planning_steps: int = CFG.DYNAQ_PLANNING_STEPS,
        world_model_learning_rate: float = CFG.DYNAQ_WORLD_MODEL_LR,
        **kwargs,
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, device=device, **kwargs)
        self.planning_steps = planning_steps
        # World model predicts (delta_s, reward, done_logit) jointly.
        # Output layout: [:, :state_dim] = Δs, [:, state_dim] = reward, [:, state_dim+1] = done logit.
        # Predicting reward and termination from data rather than with hardcoded heuristics ensures
        # the proximity penalty and the −100 cliff are correctly propagated into planning.
        self.world_model = MLP(state_dim + action_dim, state_dim + 2, hidden_dim=self.hidden_dim).to(self.device)
        self.world_model_optimizer = optim.Adam(self.world_model.parameters(), lr=world_model_learning_rate)

    def save_checkpoint(self, path: str | Path) -> None:
        """Extend the base checkpoint with world-model weights for training resumption."""
        torch.save(
            {
                "online_state_dict": self.online_network.state_dict(),
                "target_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "world_model_state_dict": self.world_model.state_dict(),
                "world_model_optimizer_state_dict": self.world_model_optimizer.state_dict(),
                "epsilon": self.epsilon,
                "learn_step": self.learn_step,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "planning_steps": self.planning_steps,
            },
            path,
        )

    def observe(self, transition: Transition) -> dict[str, float] | None:
        self.replay_buffer.append(transition)
        model_loss = self._learn_world_model()
        result = self.learn()
        if len(self.replay_buffer) >= self.batch_size:
            self._planning_updates()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if result is None:
            return None
        return {
            "loss": result["loss"] + (model_loss if model_loss is not None else 0.0),
            "mean_q": result["mean_q"],
        }

    def _learn_world_model(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions, _, _ = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(np.vstack([item.state for item in transitions]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([item.action for item in transitions], dtype=torch.long, device=self.device)
        next_states = torch.as_tensor(np.vstack([item.next_state for item in transitions]), dtype=torch.float32, device=self.device)
        action_one_hot = functional.one_hot(actions, num_classes=self.action_dim).to(dtype=torch.float32)
        model_input = torch.cat([states, action_one_hot], dim=1)
        world_output = self.world_model(model_input)          # (B, state_dim + 2)
        predicted_delta = world_output[:, :self.state_dim]    # Δs residual
        predicted_reward = world_output[:, self.state_dim]    # reward scalar
        predicted_done_logit = world_output[:, self.state_dim + 1]  # done logit

        actual_delta = next_states - states
        actual_rewards = torch.as_tensor(
            [item.reward for item in transitions], dtype=torch.float32, device=self.device
        )
        actual_dones = torch.as_tensor(
            [item.done for item in transitions], dtype=torch.float32, device=self.device
        )

        # Three-term loss: state residual (MSE) + reward (MSE) + done (BCE).
        # Predicting reward and termination from data means the −100 failure penalty and
        # the proximity-penalty ramp are automatically embedded in the world model.
        loss_delta = functional.mse_loss(predicted_delta, actual_delta)
        loss_reward = functional.mse_loss(predicted_reward, actual_rewards)
        loss_done = functional.binary_cross_entropy_with_logits(predicted_done_logit, actual_dones)
        loss = loss_delta + loss_reward + loss_done
        self.world_model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=self.max_grad_norm)
        self.world_model_optimizer.step()
        return float(loss.item())

    def _planning_updates(self) -> None:
        for _ in range(self.planning_steps):
            transitions, _, _ = self.replay_buffer.sample(self.batch_size)
            states = torch.as_tensor(np.vstack([item.state for item in transitions]), dtype=torch.float32, device=self.device)
            sampled_actions = torch.randint(0, self.action_dim, (self.batch_size,), device=self.device)
            action_one_hot = functional.one_hot(sampled_actions, num_classes=self.action_dim).to(dtype=torch.float32)

            with torch.no_grad():
                world_output = self.world_model(torch.cat([states, action_one_hot], dim=1))
                # Recover predicted next state, reward, and termination from the joint world model.
                # Using model predictions (rather than hardcoded rules) means the proximity penalty
                # and the −100 cliff are faithfully propagated through every planning Q-update.
                predicted_next_states = states + world_output[:, :self.state_dim]
                rewards = world_output[:, self.state_dim]                              # (B,)
                dones = (torch.sigmoid(world_output[:, self.state_dim + 1]) > 0.5).to(dtype=torch.float32)

            simulated_transitions = [
                Transition(
                    state=states[index].detach().cpu().numpy(),
                    action=int(sampled_actions[index].item()),
                    reward=float(rewards[index].item()),
                    next_state=predicted_next_states[index].detach().cpu().numpy(),
                    done=float(dones[index].item()),
                )
                for index in range(self.batch_size)
            ]
            self._update_from_transitions(simulated_transitions)
