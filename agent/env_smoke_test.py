"""
env_smoke_test.py -- sanity checks for env/netra_gym_env.py
===========================================================
1. gymnasium.utils.env_checker.check_env on the default env (and a 2-gateway, lagrangian and
   lexicographic variant).
2. Rolls out RandomPolicy and FixedRulePolicy (paper Algorithms 1+2, SF9, no sleep) for N
   episodes on each platform and prints detection, false alarms, energy per alert, delivery,
   battery.

Usage
    python agent/env_smoke_test.py                  # 20 one-day episodes per policy/platform
    python agent/env_smoke_test.py --episodes 5 --days 7 --out results/env_smoke_test.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("data", "agent", "env"):
    sys.path.insert(0, os.path.join(_BASE, sub))
from gymnasium.utils.env_checker import check_env  # noqa: E402

from env_policies import FixedRulePolicy, RandomPolicy, run_policy, summarize  # noqa: E402
from fixed_rule_baseline import BaselineConfig  # noqa: E402
from netra_gym_env import EnvConfig, NetraEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--days", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--profile", default="mixed_113")
    ap.add_argument("--out", default=None, help="optional CSV path for the summary table")
    a = ap.parse_args()
    t0 = time.time()

    for name, kw in [("default", {}), ("2 gateways", dict(n_gateways=2)),
                     ("lagrangian", dict(reward_mode="lagrangian")),
                     ("lexicographic", dict(reward_mode="lexicographic")),
                     ("week, no time features", dict(episode_days=7, time_features=False))]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_env(NetraEnv(EnvConfig(**kw)), skip_render_check=True)
        msgs = sorted({str(x.message)[:120] for x in w})
        print(f"check_env [{name}]: PASSED" + (f" (warnings: {msgs})" if msgs else ""))
    env = NetraEnv()
    print(f"obs dim = {env.observation_space.shape[0]}, actions = {env.action_space.n}")

    rows = []
    for platform in ("pi_zero", "pi4"):
        env = NetraEnv(EnvConfig(platform=platform, episode_days=a.days, event_profile=a.profile))
        policies = {
            "random": RandomPolicy(env, seed=a.seed),
            "fixed_rule_SF9": FixedRulePolicy(env, BaselineConfig(sf=9)),
            "fixed_rule_SF9_send_cattle": FixedRulePolicy(env, BaselineConfig(sf=9, transmit_medium=True)),
            "fixed_rule_SF12_send_cattle": FixedRulePolicy(env, BaselineConfig(sf=12, transmit_medium=True)),
        }
        for pname, pol in policies.items():
            df = run_policy(env, pol, a.episodes, seed=a.seed)
            s = summarize(df, a.days)
            rows.append(dict(platform=platform, policy=pname, **s))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    cols = ["platform", "policy", "mean_return", "detection_rate", "missed_asleep", "false_alarms_per_ep",
            "false_alarm_rate", "alerts_sent_per_ep", "delivery_rate", "energy_per_alert_j",
            "camera_uses_per_ep", "sleep_frac", "battery_end_soc", "battery_min_soc", "brownouts"]
    print(f"\n=== {a.episodes} episodes x {a.days:g} day(s), link 1.5-3 km, {a.profile} event rates ===")
    print(out[cols].round(3).to_string(index=False))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        out.to_csv(a.out, index=False)
        print(f"Saved {a.out}")
    print(f"({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
