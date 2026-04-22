from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pdm import config as CFG
from pdm.data import EngineEpisode


class TurboFanEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        episodes: list[EngineEpisode],
        scheduled_maintenance_cost: float = CFG.ENV_SCHEDULED_MAINTENANCE_COST,
        safe_cycle_reward: float = CFG.ENV_SAFE_CYCLE_REWARD,
        catastrophic_failure_penalty: float = CFG.ENV_CATASTROPHIC_FAILURE_PENALTY,
        proximity_penalty_threshold: int = CFG.ENV_PROXIMITY_PENALTY_THRESHOLD,
        proximity_penalty_scale: float = CFG.ENV_PROXIMITY_PENALTY_SCALE,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not episodes:
            raise ValueError("TurboFanEnv requires at least one engine episode.")

        self.episodes = episodes
        self.scheduled_maintenance_cost = scheduled_maintenance_cost
        self.safe_cycle_reward = safe_cycle_reward
        self.catastrophic_failure_penalty = catastrophic_failure_penalty
        self.proximity_penalty_threshold = proximity_penalty_threshold
        self.proximity_penalty_scale = proximity_penalty_scale
        # Derive observation shape from the actual state vectors — automatically
        # reflects whatever feature set was built by CMAPSSPreprocessor.
        state_dim = episodes[0].states.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        self.rng = np.random.default_rng(seed)
        self.current_episode: EngineEpisode | None = None
        self.current_index = 0
        self.cumulative_reward = 0.0
        self.last_action: int | None = None
        self._terminated: bool = False

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
        self._terminated = False
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
        if self._terminated:
            raise RuntimeError(
                "Episode has already terminated. Call reset() before stepping again."
            )
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
            # Base safe-cycle reward plus a proximity-shaping penalty that ramps
            # linearly from 0 at RUL=threshold down to −scale at RUL=1.
            # This removes the reward "cliff" at RUL=0 and provides a smooth
            # gradient signal to guide the agent toward timely maintenance.
            reward = self.safe_cycle_reward
            if 0 < current_rul < self.proximity_penalty_threshold:
                reward -= (
                    self.proximity_penalty_scale
                    * (self.proximity_penalty_threshold - current_rul)
                    / self.proximity_penalty_threshold
                )
            self.current_index += 1
            event = "safe_operation"
            if self.current_index >= len(episode.states):
                terminated = True
                next_state = state.copy()
            else:
                next_state = episode.states[self.current_index].astype(np.float32)

        self.cumulative_reward += float(reward)
        if terminated:
            self._terminated = True
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
