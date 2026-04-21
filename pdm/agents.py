from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn, optim


@dataclass(slots=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[index] for index in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class DoubleDQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 20_000,
        target_sync_interval: int = 100,
        hidden_dim: int = 128,
        max_grad_norm: float = 10.0,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_sync_interval = target_sync_interval
        self.max_grad_norm = max_grad_norm
        self.learn_step = 0

        self.online_network = MLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = MLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def observe(self, transition: Transition) -> float | None:
        self.replay_buffer.append(transition)
        loss = self.learn()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return loss

    def learn(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample(self.batch_size)
        loss = self._update_from_transitions(transitions)
        self.learn_step += 1
        if self.learn_step % self.target_sync_interval == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        return float(loss)

    def _update_from_transitions(self, transitions: Iterable[Transition]) -> float:
        batch = list(transitions)
        states = torch.as_tensor(np.vstack([item.state for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.vstack([item.next_state for item in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.online_network(states).gather(1, actions)
        with torch.no_grad():
            next_actions = torch.argmax(self.online_network(next_states), dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = functional.mse_loss(q_values, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())


class DynaQAgent(DoubleDQNAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str,
        planning_steps: int = 50,
        world_model_learning_rate: float = 1e-3,
        **kwargs: float,
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, device=device, **kwargs)
        self.planning_steps = planning_steps
        self.world_model = MLP(state_dim + action_dim, state_dim).to(self.device)
        self.world_model_optimizer = optim.Adam(self.world_model.parameters(), lr=world_model_learning_rate)

    def observe(self, transition: Transition) -> float | None:
        self.replay_buffer.append(transition)
        model_loss = self._learn_world_model()
        value_loss = self.learn()
        if len(self.replay_buffer) >= self.batch_size:
            self._planning_updates()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if value_loss is None:
            return model_loss
        if model_loss is None:
            return value_loss
        return float(value_loss + model_loss)

    def _learn_world_model(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(np.vstack([item.state for item in transitions]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([item.action for item in transitions], dtype=torch.long, device=self.device)
        next_states = torch.as_tensor(np.vstack([item.next_state for item in transitions]), dtype=torch.float32, device=self.device)
        action_one_hot = functional.one_hot(actions, num_classes=self.action_dim).to(dtype=torch.float32)
        model_input = torch.cat([states, action_one_hot], dim=1)
        predicted_delta = self.world_model(model_input)

        # Predict the state residual (Δs = s' - s) instead of the absolute next state.
        # This equalises the gradient contribution of each feature dimension:
        # raw cycle count changes by +1 per step while normalised sensor residuals are O(0.01),
        # so absolute-state MSE would be dominated by the cycle term.
        actual_delta = next_states - states
        loss = functional.mse_loss(predicted_delta, actual_delta)
        self.world_model_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=self.max_grad_norm)
        self.world_model_optimizer.step()
        return float(loss.item())

    def _planning_updates(self) -> None:
        for _ in range(self.planning_steps):
            transitions = self.replay_buffer.sample(self.batch_size)
            states = torch.as_tensor(np.vstack([item.state for item in transitions]), dtype=torch.float32, device=self.device)
            sampled_actions = torch.randint(0, self.action_dim, (self.batch_size,), device=self.device)
            action_one_hot = functional.one_hot(sampled_actions, num_classes=self.action_dim).to(dtype=torch.float32)

            with torch.no_grad():
                # Recover absolute predicted next state via the learned residual.
                predicted_next_states = states + self.world_model(torch.cat([states, action_one_hot], dim=1))
                current_cycles = states[:, -1]
                predicted_cycles = predicted_next_states[:, -1]
                maintenance_actions = sampled_actions == 1
                catastrophic_failures = (~maintenance_actions) & (predicted_cycles <= current_cycles)
                rewards = torch.where(
                    maintenance_actions,
                    torch.full_like(current_cycles, -20.0),
                    torch.where(catastrophic_failures, torch.full_like(current_cycles, -100.0), torch.ones_like(current_cycles)),
                )
                dones = (maintenance_actions | catastrophic_failures).to(dtype=torch.float32)

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
