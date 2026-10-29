"""env/netra_env.py: NETRA-RL Week 2 Task C - Gymnasium environment.

Each step the agent picks a threshold tau; the NEXT sensor row is sampled
(with self.np_random), fused, and scored with the Task B confusion reward.
Battery state makes the problem a true MDP (not a bandit).
"""
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from env.fusion_reward import (
    compute_alert,
    compute_fusion_score,
    compute_p_dist,
    compute_reward,
)

TAUS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
BATTERY_STEP_COST = 0.001
BATTERY_ALERT_COST = 0.01
BATTERY_DEAD_PENALTY = -50.0


class NetraEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, csv_path: str, episode_len: int = 200, render_mode=None):
        super().__init__()
        df = pd.read_csv(csv_path)
        missing = {"hour", "pir", "distance", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        self.hours = df["hour"].to_numpy(dtype=np.float64)
        self.pir = df["pir"].to_numpy(dtype=np.float64)
        self.p_dist = np.asarray(compute_p_dist(df["distance"].to_numpy(dtype=float)), dtype=np.float64)
        self.labels = df["label"].to_numpy(dtype=int)
        self.n_rows = len(df)
        self.episode_len = int(episode_len)
        self.render_mode = render_mode
        self.taus = np.array(TAUS, dtype=np.float64)

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(TAUS))

        self.battery = 1.0
        self.steps = 0
        self.row_idx = 0

    def _sample_row(self) -> int:
        return int(self.np_random.integers(0, self.n_rows))

    def _obs(self) -> np.ndarray:
        i = self.row_idx
        ang = 2.0 * np.pi * self.hours[i] / 24.0
        obs = np.array(
            [self.pir[i], self.p_dist[i], np.sin(ang), np.cos(ang), self.battery],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.battery = 1.0
        self.steps = 0
        self.row_idx = self._sample_row()
        return self._obs(), {"battery": self.battery, "row": self.row_idx}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        tau = float(self.taus[int(action)])

        # Reward uses the NEXT row, drawn after the action is chosen.
        self.row_idx = self._sample_row()
        i = self.row_idx
        score = compute_fusion_score(float(self.pir[i]), float(self.p_dist[i]))
        alert = int(compute_alert(score, tau))
        label = int(self.labels[i])
        reward, outcome = compute_reward(label, alert)
        reward = float(reward)

        self.battery -= BATTERY_STEP_COST + (BATTERY_ALERT_COST if alert else 0.0)
        self.steps += 1

        terminated = False
        if self.battery <= 0.0:
            self.battery = 0.0
            terminated = True
            reward += BATTERY_DEAD_PENALTY
        truncated = (not terminated) and self.steps >= self.episode_len

        info = {
            "tau": tau, "alert": alert, "label": label, "outcome": outcome,
            "score": float(score), "battery": self.battery, "step": self.steps,
        }
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        return None
