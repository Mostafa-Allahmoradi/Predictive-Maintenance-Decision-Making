"""pdm/ddqn_agent.py
===================
Double DQN agent with Prioritised Experience Replay (PER).

Public API
----------
    Transition            — named dataclass for replay buffer entries
    SumTree               — binary sum-tree for O(log n) priority sampling
    PrioritizedReplayBuffer — PER buffer (Schaul et al., 2015)
    MLP                   — shared two-hidden-layer network backbone
    DoubleDQNAgent        — model-free Double DQN with Polyak target updates
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn, optim

from pdm import config as CFG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Transition
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: float
    rul: int = -1  # remaining useful life; -1 = unknown (e.g. simulated transitions)


# ──────────────────────────────────────────────────────────────────────────────
# SumTree
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# PrioritizedReplayBuffer
# ──────────────────────────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer (PER, Schaul et al., 2015).

    Failure transitions (RUL < threshold or catastrophic reward) receive a
    boosted initial priority so the agent samples them more frequently.
    Importance-sampling (IS) weights correct for the resulting bias:

        P(i) = p_i^alpha / sum_k p_k^alpha   (sampling probability)
        w_i  = (N * P(i))^{-beta}             (IS weight, normalized by max)

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


# ──────────────────────────────────────────────────────────────────────────────
# MLP backbone
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# DoubleDQNAgent
# ──────────────────────────────────────────────────────────────────────────────

class DoubleDQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str,
        gamma: float = CFG.DDQN_GAMMA,
        # --- Optimised defaults for RTX 3060 Ti + CMAPSS scale ---
        learning_rate: float = CFG.DDQN_LEARNING_RATE,
        lr_end_factor: float = CFG.DDQN_LR_END_FACTOR,
        lr_decay_steps: int = CFG.DDQN_LR_DECAY_STEPS,
        epsilon_start: float = CFG.DDQN_EPSILON_START,
        epsilon_end: float = CFG.DDQN_EPSILON_END,
        epsilon_decay: float = CFG.DDQN_EPSILON_DECAY,
        batch_size: int = CFG.DDQN_BATCH_SIZE,
        buffer_capacity: int = CFG.DDQN_BUFFER_CAPACITY,
        tau: float = CFG.DDQN_TAU,
        hidden_dim: int = CFG.DDQN_HIDDEN_DIM,
        max_grad_norm: float = CFG.DDQN_MAX_GRAD_NORM,
        # --- PER hyper-parameters ---
        per_alpha: float = CFG.PER_ALPHA,
        per_beta_start: float = CFG.PER_BETA_START,
        per_beta_end: float = CFG.PER_BETA_END,
        per_beta_anneal_steps: int = CFG.PER_BETA_ANNEAL_STEPS,
        per_failure_boost: float = CFG.PER_FAILURE_BOOST,
        per_failure_rul_threshold: int = CFG.PER_FAILURE_RUL_THRESHOLD,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
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
        self._soft_update_target()
        return {"loss": loss, "mean_q": mean_q}

    def _soft_update_target(self) -> None:
        """Polyak averaging: θ_target ← τ·θ_online + (1−τ)·θ_target.
        Runs every gradient step — produces smoother Q-value targets than hard
        copies and eliminates the periodic discontinuity in the loss curve.
        """
        for target_param, online_param in zip(
            self.target_network.parameters(), self.online_network.parameters()
        ):
            target_param.data.mul_(1.0 - self.tau).add_(online_param.data * self.tau)

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
