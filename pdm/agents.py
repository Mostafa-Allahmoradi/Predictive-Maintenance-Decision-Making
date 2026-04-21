from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
    rul: int = -1  # remaining useful life; -1 = unknown (e.g. simulated transitions)


class SumTree:
    """Binary sum-tree for O(log n) priority sampling (Schaul et al., 2015 PER).

    Uses 1-based indexing: root at index 1, leaves at [capacity, 2*capacity).
    Internal nodes store the sum of their subtrees; leaves store priorities.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data: list[Transition | None] = [None] * capacity
        self.write_ptr: int = 0
        self.size: int = 0

    def add(self, priority: float, transition: Transition) -> None:
        tree_idx = self.write_ptr + self.capacity
        self.data[self.write_ptr] = transition
        self._update(tree_idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _update(self, tree_idx: int, priority: float) -> None:
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 1:
            tree_idx >>= 1
            self.tree[tree_idx] += delta

    def _retrieve(self, value: float) -> tuple[int, float]:
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx, float(self.tree[idx])

    def sample_one(self, value: float) -> tuple[int, float, Transition]:
        # Clamp to avoid floating-point overshoot past the last leaf.
        value = min(value, self.total_priority - 1e-8)
        tree_idx, priority = self._retrieve(value)
        transition = self.data[tree_idx - self.capacity]
        assert transition is not None
        return tree_idx, priority, transition

    def update_priority(self, tree_idx: int, priority: float) -> None:
        self._update(tree_idx, priority)

    @property
    def total_priority(self) -> float:
        return float(self.tree[1])


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer (PER, Schaul et al., 2015).

    Failure transitions (RUL < threshold or catastrophic reward) receive a
    boosted initial priority so the agent samples them more frequently.
    Importance-sampling (IS) weights correct for the resulting bias:

        P(i) = p_i^alpha / sum_k p_k^alpha   (sampling probability)
        w_i  = (N * P(i))^{-beta}             (IS weight, normalised by max)

    beta is annealed from beta_start to 1.0 over beta_anneal_steps,
    progressively removing the IS bias as training stabilises.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_anneal_steps: int = 100_000,
        epsilon: float = 1e-6,
        failure_boost: float = 5.0,
        failure_rul_threshold: int = 20,
    ) -> None:
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.epsilon = epsilon
        self.failure_boost = failure_boost
        self.failure_rul_threshold = failure_rul_threshold
        self._anneal_step: int = 0
        self._max_priority: float = 1.0

    @property
    def beta(self) -> float:
        fraction = min(self._anneal_step / max(self.beta_anneal_steps, 1), 1.0)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def append(self, transition: Transition) -> None:
        priority = self._max_priority
        # Boost priority for near-failure (RUL < threshold) and catastrophic failures.
        if 0 <= transition.rul < self.failure_rul_threshold:
            priority *= self.failure_boost
        elif transition.reward <= -50.0:
            priority *= self.failure_boost
        self.tree.add(priority ** self.alpha, transition)

    def sample(
        self, batch_size: int
    ) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        self._anneal_step += 1
        tree_indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        transitions: list[Transition] = []
        total = self.tree.total_priority
        segment = total / batch_size
        current_beta = self.beta
        for index in range(batch_size):
            value = np.random.uniform(segment * index, segment * (index + 1))
            tree_idx, priority, transition = self.tree.sample_one(value)
            tree_indices[index] = tree_idx
            priorities[index] = priority
            transitions.append(transition)
        sampling_probs = priorities / total
        weights = (self.tree.size * sampling_probs) ** (-current_beta)
        weights = (weights / weights.max()).astype(np.float32)
        return transitions, tree_indices, weights

    def update_priorities(
        self, tree_indices: np.ndarray, td_errors: np.ndarray
    ) -> None:
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        new_max = float(priorities.max())
        if new_max > self._max_priority:
            self._max_priority = new_max
        for tree_idx, priority in zip(tree_indices, priorities):
            self.tree.update_priority(int(tree_idx), float(priority))

    def __len__(self) -> int:
        return self.tree.size


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256) -> None:
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
        # --- Optimised defaults for RTX 3060 Ti + CMAPSS scale ---
        learning_rate: float = 3e-4,
        lr_end_factor: float = 0.05,    # decay LR to 5 % of initial over lr_decay_steps
        lr_decay_steps: int = 50_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,          # lower floor → finer exploitation
        epsilon_decay: float = 0.99995,     # reaches floor at ~78k steps (~500 ep × 200 avg cycles)
        batch_size: int = 256,              # larger batch saturates GPU for small state dim
        buffer_capacity: int = 100_000,     # stores ~500 full engine runs; retains rare failures longer
        target_sync_interval: int = 200,    # more stable with larger network and −100 penalty scale
        hidden_dim: int = 256,             # more capacity for 22-dim state → 2-action mapping
        max_grad_norm: float = 5.0,        # tighter clipping for double-width hidden layers
        # --- PER hyper-parameters ---
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_anneal_steps: int = 200_000,   # anneal β over full training run
        per_failure_boost: float = 5.0,
        per_failure_rul_threshold: int = 30,    # wider near-failure zone for boosted sampling
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
        self.hidden_dim = hidden_dim  # stored so subclasses can match capacity

        self.online_network = MLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = MLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=learning_rate)
        # Linear LR decay: lr * 1.0 → lr * lr_end_factor over lr_decay_steps gradient steps.
        self.lr_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=lr_end_factor,
            total_iters=lr_decay_steps,
        )
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_end=per_beta_end,
            beta_anneal_steps=per_beta_anneal_steps,
            failure_boost=per_failure_boost,
            failure_rul_threshold=per_failure_rul_threshold,
        )

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def observe(self, transition: Transition) -> dict[str, float] | None:
        self.replay_buffer.append(transition)
        result = self.learn()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return result

    def learn(self) -> dict[str, float] | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions, tree_indices, weights = self.replay_buffer.sample(self.batch_size)
        loss, mean_q, td_errors = self._update_from_transitions(transitions, weights=weights)
        self.replay_buffer.update_priorities(tree_indices, td_errors)
        self.lr_scheduler.step()
        self.learn_step += 1
        if self.learn_step % self.target_sync_interval == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        return {"loss": loss, "mean_q": mean_q}

    def _update_from_transitions(
        self,
        transitions: Iterable[Transition],
        weights: np.ndarray | None = None,
    ) -> tuple[float, float, np.ndarray]:
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

        # Per-element squared TD errors (used for priority updates and IS weighting).
        element_loss = functional.mse_loss(q_values, target, reduction="none")  # (B, 1)
        td_errors = (q_values - target).detach().squeeze(1).cpu().numpy()       # (B,)
        mean_q = float(q_values.detach().mean().item())

        if weights is not None:
            w_tensor = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
            loss = (w_tensor * element_loss).mean()
        else:
            loss = element_loss.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        return float(loss.item()), mean_q, td_errors

    def save_checkpoint(self, path: str | Path) -> None:
        """Persist Q-networks, optimizer, scheduler, and training state to disk."""
        torch.save(
            {
                "online_state_dict": self.online_network.state_dict(),
                "target_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "epsilon": self.epsilon,
                "learn_step": self.learn_step,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            path,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str = "cpu",
        **init_kwargs,
    ) -> "DoubleDQNAgent":
        """Load a trained agent for inference. Only a DoubleDQNAgent is returned
        regardless of the concrete subclass that saved the checkpoint, because
        evaluation only needs the Q-network (no world model required).
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)  # trusted local file
        agent = DoubleDQNAgent(
            state_dim=int(ckpt["state_dim"]),
            action_dim=int(ckpt["action_dim"]),
            device=device,
            **init_kwargs,
        )
        agent.online_network.load_state_dict(ckpt["online_state_dict"])
        agent.target_network.load_state_dict(ckpt["target_state_dict"])
        agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "lr_scheduler_state_dict" in ckpt:
            agent.lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
        agent.epsilon = float(ckpt["epsilon"])
        agent.learn_step = int(ckpt["learn_step"])
        agent.online_network.eval()
        agent.target_network.eval()
        return agent


class DynaQAgent(DoubleDQNAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str,
        planning_steps: int = 100,           # 100 simulated steps per real step
        world_model_learning_rate: float = 3e-4,  # matched to Q-network LR for stable joint training
        **kwargs,
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, device=device, **kwargs)
        self.planning_steps = planning_steps
        # Match hidden width of Q-networks so world model has equal representational capacity.
        self.world_model = MLP(state_dim + action_dim, state_dim, hidden_dim=self.hidden_dim).to(self.device)
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
            transitions, _, _ = self.replay_buffer.sample(self.batch_size)
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
