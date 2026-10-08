"""
dqn_sanity_run.py -- very short Stable-Baselines3 DQN run on NetraEnv (NOT full training)
=========================================================================================
Confirms the env + SB3 + torch stack trains end to end: trains a DQN with the PhD's
3-layer MLP (128 -> 64 -> |A|) for a few thousand steps and compares its greedy policy with
the random and fixed-rule policies on the same evaluation seeds. Nothing is saved unless
--save is given.

Usage
    python agent/dqn_sanity_run.py                      # 5000 steps, Pi Zero, 1-day episodes
    python agent/dqn_sanity_run.py --steps 20000 --platform pi4 --save results/dqn_sanity.zip
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("data", "agent", "env"):
    sys.path.insert(0, os.path.join(_BASE, sub))
from stable_baselines3 import DQN  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from env_policies import FixedRulePolicy, RandomPolicy, run_policy, summarize  # noqa: E402
from netra_gym_env import EnvConfig, NetraEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--platform", default="pi_zero")
    ap.add_argument("--days", type=float, default=1.0)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="mixed_113")
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    cfg = EnvConfig(platform=a.platform, episode_days=a.days, event_profile=a.profile)
    env = Monitor(NetraEnv(cfg))
    model = DQN("MlpPolicy", env, policy_kwargs=dict(net_arch=[128, 64]), learning_rate=5e-4,
                buffer_size=50_000, learning_starts=500, batch_size=64, gamma=0.99,
                train_freq=1, target_update_interval=500, exploration_fraction=0.5,
                exploration_final_eps=0.05, seed=a.seed, verbose=0, device="cpu")
    t0 = time.time()
    model.learn(total_timesteps=a.steps)
    dt = time.time() - t0
    rets = [float(x) for x in env.get_episode_rewards()]
    k = max(1, len(rets) // 4)
    print(f"trained {a.steps} steps in {dt:.1f} s ({len(rets)} episodes); "
          f"train return first-quarter mean {np.mean(rets[:k]):.1f} -> last-quarter mean {np.mean(rets[-k:]):.1f}")

    eval_env = NetraEnv(cfg)
    dqn_pol = lambda o: int(model.predict(o, deterministic=True)[0])  # noqa: E731
    for name, pol in [("random", RandomPolicy(eval_env, seed=1)), ("fixed_rule_SF9", FixedRulePolicy(eval_env)),
                      (f"dqn_{a.steps}_steps", dqn_pol)]:
        s = summarize(run_policy(eval_env, pol, a.eval_episodes, seed=10_000), a.days)
        print(f"{name:<18} return={s['mean_return']:8.1f}  detection={s['detection_rate']:.3f}  "
              f"false_alarms/ep={s['false_alarms_per_ep']:.2f}  delivery={s['delivery_rate']:.3f}  "
              f"J/alert={s['energy_per_alert_j']:.1f}  sleep_frac={s['sleep_frac']:.3f}  "
              f"batt_end={s['battery_end_soc']:.3f}")
    if a.save:
        model.save(a.save)
        print(f"saved {a.save}")


if __name__ == "__main__":
    main()
