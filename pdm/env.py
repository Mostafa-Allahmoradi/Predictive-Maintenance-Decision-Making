from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pdm.data import EngineEpisode


class TurboFanEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        episodes: list[EngineEpisode],
        scheduled_maintenance_cost: float = -20.0,
        safe_cycle_reward: float = 1.0,
        catastrophic_failure_penalty: float = -100.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not episodes:
            raise ValueError("TurboFanEnv requires at least one engine episode.")

        self.episodes = episodes
        self.scheduled_maintenance_cost = scheduled_maintenance_cost
        self.safe_cycle_reward = safe_cycle_reward
        self.catastrophic_failure_penalty = catastrophic_failure_penalty
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(22,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        self.rng = np.random.default_rng(seed)
        self.current_episode: EngineEpisode | None = None
        self.current_index = 0
        self.cumulative_reward = 0.0
        self.last_action: int | None = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if options and "engine_index" in options:
            episode_index = int(options["engine_index"])
        else:
            episode_index = int(self.rng.integers(0, len(self.episodes)))

        self.current_episode = self.episodes[episode_index]
        self.current_index = 0
        self.cumulative_reward = 0.0
        self.last_action = None
        observation = self.current_episode.states[self.current_index].astype(np.float32)
        info = {
            "engine_id": self.current_episode.engine_id,
            "rul": int(self.current_episode.rul[self.current_index]),
            "cycle": float(observation[-1]),
        }
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.current_episode is None:
            raise RuntimeError("Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        episode = self.current_episode
        state = episode.states[self.current_index].astype(np.float32)
        current_rul = int(episode.rul[self.current_index])
        self.last_action = int(action)
        terminated = False
        truncated = False

        if action == 1:
            reward = self.scheduled_maintenance_cost
            terminated = True
            next_state = state.copy()
            event = "scheduled_maintenance"
        elif current_rul <= 0:
            reward = self.catastrophic_failure_penalty
            terminated = True
            next_state = state.copy()
            event = "catastrophic_failure"
        else:
            reward = self.safe_cycle_reward
            self.current_index += 1
            event = "safe_operation"
            if self.current_index >= len(episode.states):
                terminated = True
                next_state = state.copy()
            else:
                next_state = episode.states[self.current_index].astype(np.float32)

        self.cumulative_reward += float(reward)
        info = {
            "engine_id": episode.engine_id,
            "event": event,
            "rul": current_rul,
            "cycle": float(state[-1]),
            "cycles_survived": int(state[-1]),
            "cumulative_reward": self.cumulative_reward,
            "total_cost": -self.cumulative_reward,
        }
        return next_state, float(reward), terminated, truncated, info
