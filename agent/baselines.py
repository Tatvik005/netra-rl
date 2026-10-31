"""
agent/baselines.py: Baseline policies evaluated on NetraEnv (Week 2, Task D).

Policies:
  1. Random action every step
  2. Fixed tau_c  (NETRA's 0.65 is not in the action set; default 0.6, see --tau-c)
  3. Fixed tau*   (best fixed threshold from the Task B sweep, snapped to the action set)
  4. Always most sensitive (tau = 0.3) and always least sensitive (tau = 0.8)

Protocol: 100 episodes per seed, 5 seeds [42, 100, 287, 523, 1024].
Each seed uses its own dataset (data/sensor_v2_seed{N}.csv) and env.reset(seed=N)
on the first episode, so every number is reproducible.

Metrics (mean +/- std ACROSS SEEDS, as the task sheet asks):
  episode return, detection rate (recall), false alarm rate, alerts per episode,
  episode length, and the fraction of episodes ended by battery depletion.
  Per seed: return/alerts/length are averaged over its 100 episodes; detection
  rate and FAR are computed from confusion counts pooled over those episodes.
"""

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for p in (PROJECT_ROOT, Path.cwd()):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from env.netra_env import NetraEnv, TAUS  # noqa: E402

BENCHMARK_SEEDS: List[int] = [42, 100, 287, 523, 1024]
EPISODE_LEN = 200

Policy = Callable[[np.ndarray, np.random.Generator], int]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def tau_to_action(tau: float) -> int:
    """Index of the closest tau in TAUS (ties go to the lower tau)."""
    diffs = np.round(np.abs(np.array(TAUS) - tau), 9)
    return int(np.argmin(diffs))


def resolve_dirs():
    data_dir = PROJECT_ROOT / "data" if (PROJECT_ROOT / "data").exists() else Path.cwd() / "data"
    results_dir = PROJECT_ROOT / "results" if (PROJECT_ROOT / "results").exists() else Path.cwd() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, results_dir


def load_tau_star(results_dir: Path) -> Optional[float]:
    """Best fixed tau (highest mean reward) from the Task B sweep summary CSV."""
    candidates = [results_dir / "threshold_sweep.csv"]
    candidates += sorted(results_dir.glob("threshold_sweep_5seeds_*runs.csv"))
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if {"tau", "reward_mean"} <= set(df.columns):
                return float(df.loc[df["reward_mean"].idxmax(), "tau"])
    return None


def make_policies(tau_c: float, tau_star: float) -> Dict[str, Policy]:
    a_c, a_star = tau_to_action(tau_c), tau_to_action(tau_star)
    a_low, a_high = tau_to_action(0.3), tau_to_action(0.8)
    n_actions = len(TAUS)
    return {
        "Random": lambda obs, rng: int(rng.integers(0, n_actions)),
        f"Fixed tau_c ({TAUS[a_c]:.1f})": lambda obs, rng: a_c,
        f"Fixed tau* ({TAUS[a_star]:.1f})": lambda obs, rng: a_star,
        "Always 0.3": lambda obs, rng: a_low,
        "Always 0.8": lambda obs, rng: a_high,
    }


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def run_episode(env: NetraEnv, policy: Policy, rng: np.random.Generator,
                seed: Optional[int] = None) -> Dict[str, float]:
    obs, _ = env.reset(seed=seed)
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    total_reward, alerts, length = 0.0, 0, 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = policy(obs, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        alerts += info["alert"]
        counts[info["outcome"]] += 1
        length += 1

    return {
        "return": total_reward,
        "alerts": alerts,
        "length": length,
        "battery_dead": float(terminated),
        **counts,
    }


def evaluate(policies: Dict[str, Policy], data_dir: Path, episodes: int):
    episode_rows = []
    seed_rows = []

    for name, policy in policies.items():
        for seed in BENCHMARK_SEEDS:
            csv_path = data_dir / f"sensor_v2_seed{seed}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing dataset: {csv_path}")
            env = NetraEnv(str(csv_path), episode_len=EPISODE_LEN)
            rng = np.random.default_rng(seed)  # policy randomness (Random baseline)

            eps = []
            for ep in range(episodes):
                # Seed the env once per (policy, seed); later resets continue the stream.
                res = run_episode(env, policy, rng, seed=seed if ep == 0 else None)
                res.update(policy=name, seed=seed, episode=ep)
                eps.append(res)
            episode_rows.extend(eps)

            df = pd.DataFrame(eps)
            tp, fp, fn, tn = (df[k].sum() for k in ("TP", "FP", "FN", "TN"))
            seed_rows.append({
                "policy": name,
                "seed": seed,
                "return": df["return"].mean(),
                "detection_rate": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
                "far": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
                "alerts": df["alerts"].mean(),
                "length": df["length"].mean(),
                "battery_dead_frac": df["battery_dead"].mean(),
            })
            env.close()

    return pd.DataFrame(episode_rows), pd.DataFrame(seed_rows)


METRICS = [
    ("return", "Episode Return"),
    ("detection_rate", "Detection Rate"),
    ("far", "False Alarm Rate"),
    ("alerts", "Alerts / Episode"),
    ("length", "Episode Length"),
    ("battery_dead_frac", "Battery-Depleted Fraction"),
]


def summarize(df_seed: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    """Mean +/- std across the 5 seeds, one row per policy."""
    rows = []
    for name in order:
        sub = df_seed[df_seed["policy"] == name]
        row = {"policy": name}
        for key, _ in METRICS:
            row[f"{key}_mean"] = sub[key].mean()
            row[f"{key}_std"] = sub[key].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def print_table(summary: pd.DataFrame, episodes: int) -> None:
    print("\n" + "=" * 118)
    print(f"  NETRA-RL TASK D: BASELINES ({episodes} episodes x {len(BENCHMARK_SEEDS)} seeds, "
          f"episode_len={EPISODE_LEN}); mean +/- std across seeds")
    print("=" * 118)
    print(f"{'Policy':<20} | {'Return':>16} | {'Detection':>15} | {'FAR':>15} | "
          f"{'Alerts/ep':>15} | {'Ep. length':>14} | {'Dead':>6}")
    print("-" * 118)
    for _, r in summary.iterrows():
        print(
            f"{r['policy']:<20} | "
            f"{r['return_mean']:>8.1f} ± {r['return_std']:<5.1f} | "
            f"{r['detection_rate_mean']:>6.1%} ± {r['detection_rate_std']:<5.1%} | "
            f"{r['far_mean']:>6.2%} ± {r['far_std']:<5.2%} | "
            f"{r['alerts_mean']:>7.1f} ± {r['alerts_std']:<4.1f} | "
            f"{r['length_mean']:>6.1f} ± {r['length_std']:<4.1f} | "
            f"{r['battery_dead_frac_mean']:>5.0%}"
        )
    print("=" * 118)
    best = summary.loc[summary["return_mean"].idxmax()]
    print(f"Highest mean return: {best['policy']} ({best['return_mean']:.1f}). "
          f"This is the bar the Week 3 DQN has to beat.\n")


def plot_bars(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = ["#7f7f7f", "#ff7f0e", "#9467bd", "#2ca02c", "#d62728", "#1f77b4"]
    x = np.arange(len(summary))

    for ax, (key, title) in zip(axes.ravel(), METRICS):
        ax.bar(
            x, summary[f"{key}_mean"], yerr=summary[f"{key}_std"], capsize=4,
            color=colors[: len(summary)], edgecolor="black", linewidth=0.6,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(summary["policy"], rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.axhline(0, color="black", lw=0.8)

    fig.suptitle("NETRA-RL Baseline Policies (mean ± std across 5 seeds)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved 300 DPI bar chart to: {out_path}")


# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="NETRA-RL baseline policies (Task D)")
    parser.add_argument("--episodes", type=int, default=100, help="episodes per seed (default 100)")
    parser.add_argument("--tau-c", type=float, default=0.6,
                        help="action used for NETRA's tau_c=0.65. 0.65 is not in the action set; "
                             "0.6 and 0.7 are equally close. Default 0.6.")
    parser.add_argument("--tau-star", type=float, default=None,
                        help="best fixed tau; default: read from the Task B sweep CSV")
    args = parser.parse_args()

    data_dir, results_dir = resolve_dirs()

    tau_star = args.tau_star if args.tau_star is not None else load_tau_star(results_dir)
    if tau_star is None:
        print("No sweep CSV found in results/; falling back to tau* = 0.3. "
              "Run agent/threshold_sweep.py or pass --tau-star.")
        tau_star = 0.3
    print(f"tau_c action: {TAUS[tau_to_action(args.tau_c)]:.1f} (closest to 0.65; tie, chose per --tau-c)")
    print(f"tau* from sweep: {tau_star:.2f} -> env action {TAUS[tau_to_action(tau_star)]:.1f}")

    policies = make_policies(args.tau_c, tau_star)
    df_eps, df_seed = evaluate(policies, data_dir, args.episodes)
    summary = summarize(df_seed, list(policies.keys()))

    df_eps.to_csv(results_dir / "baselines_episodes.csv", index=False)
    df_seed.to_csv(results_dir / "baselines_per_seed.csv", index=False)
    summary.to_csv(results_dir / "baselines_summary.csv", index=False)
    print(f"Saved CSVs to: {results_dir}")

    print_table(summary, args.episodes)
    plot_return_bar(summary, results_dir / "baselines_return_bar.png")
    # optional supplementary figure:
    plot_bars(summary, results_dir / "baselines_bar_chart.png")


def plot_return_bar(summary: pd.DataFrame, out_path: Path) -> None:
    """Single bar chart: mean episode return per policy, error bars = std across seeds."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary))
    colors = ["#7f7f7f", "#ff7f0e", "#9467bd", "#2ca02c", "#d62728", "#1f77b4"]

    ax.bar(
        x, summary["return_mean"], yerr=summary["return_std"], capsize=5,
        color=colors[: len(summary)], edgecolor="black", linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["policy"], rotation=20, ha="right")
    ax.set_ylabel("Episode Return (mean ± std across 5 seeds)", fontweight="bold")
    ax.set_title("NETRA-RL Baseline Policies: Episode Return", fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved 300 DPI return chart to: {out_path}")

if __name__ == "__main__":
    main()
