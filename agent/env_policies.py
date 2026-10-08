"""
env_policies.py -- baseline policies for env/netra_gym_env.py
=============================================================

* FixedRulePolicy : the paper's Algorithms 1 + 2 (agent/fixed_rule_baseline.py) expressed as
                    an observation -> action mapping inside NetraEnv. Equivalent to the offline
                    baseline because obs[1] (p_dist) is 0 whenever Algorithm 1 would reject the
                    reading, so P_intrusion = 0.4*PIR + 0.6*p_dist is reproduced exactly (for
                    tau_c > 0.4). Fixed SF (default 9), gateway 0, never sleeps.
* RandomPolicy    : uniform over the Discrete action space.
* run_policy()    : rolls out a policy for N episodes and returns per-episode stats.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("data", "agent", "env"):
    p = os.path.join(_BASE, sub)
    if p not in sys.path:
        sys.path.insert(0, p)
from fixed_rule_baseline import BaselineConfig, algorithm2  # noqa: E402
from netra_gym_env import CLASSES, SF_OPTIONS, NetraEnv  # noqa: E402


class FixedRulePolicy:
    def __init__(self, env: NetraEnv, cfg: Optional[BaselineConfig] = None, sleep_idx: int = 0):
        self.env = env
        self.cfg = cfg or BaselineConfig()
        self.sleep_idx = sleep_idx
        if self.cfg.method != "probabilistic":
            raise ValueError("FixedRulePolicy implements the probabilistic fusion rule")
        self.tx_idx = 1 + SF_OPTIONS.index(self.cfg.sf)

    def __call__(self, obs: np.ndarray) -> int:
        c = self.cfg
        pir, p_dist, cam_used = float(obs[0]), float(obs[1]), obs[5] > 0.5
        p_intr = c.w_pir * pir + c.w_dist * p_dist
        env = self.env
        if pir < 0.5:                                   # heartbeat epoch
            return env.encode_action(0, 0, self.sleep_idx, 0)
        if not cam_used:                                # Algorithm 1
            return env.encode_action(1 if p_intr >= c.tau_c else 0, 0, self.sleep_idx, 0)
        onehot = obs[6:10]
        label = CLASSES[int(np.argmax(onehot))] if onehot.max() > 0.5 else "background"
        decision, _, _ = algorithm2(True, p_intr, label, float(obs[4]), c)   # Algorithm 2
        return env.encode_action(0, self.tx_idx if decision == "transmit" else 0, self.sleep_idx, 0)


class RandomPolicy:
    def __init__(self, env: NetraEnv, seed: int = 0):
        self.space = env.action_space
        self.space.seed(seed)

    def __call__(self, obs: np.ndarray) -> int:
        return int(self.space.sample())


def run_policy(env: NetraEnv, policy: Callable[[np.ndarray], int], n_episodes: int,
               seed: int = 0) -> pd.DataFrame:
    rows: List[dict] = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            done = term or trunc
        s = info["episode_stats"]
        s["episode"] = ep
        rows.append(s)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, episode_days: float = 1.0) -> dict:
    intr = df.intrusions.sum()
    return dict(
        episodes=len(df),
        mean_return=df.reward.mean(),
        detection_rate=df.detected.sum() / intr if intr else float("nan"),
        missed_asleep=df.missed_asleep.sum() / intr if intr else float("nan"),
        false_alarms_per_ep=df.false_alarms.mean(),
        false_alarm_rate=df.false_alarms.sum() / max(df.non_intrusions.sum(), 1),
        alerts_sent_per_ep=df.alerts_sent.mean(),
        delivery_rate=df.alerts_delivered.sum() / df.alerts_sent.sum() if df.alerts_sent.sum() else float("nan"),
        energy_per_alert_j=df.energy_action_j.sum() / df.alerts_delivered.sum() if df.alerts_delivered.sum() else float("nan"),
        camera_uses_per_ep=df.camera_uses.mean(),
        sleep_frac=df.sleep_s.sum() / (len(df) * episode_days * 86400.0),
        battery_end_soc=df.battery_end_soc.mean(),
        battery_min_soc=df.battery_min_soc.mean(),
        brownouts=int(df.brownout.sum()),
    )
