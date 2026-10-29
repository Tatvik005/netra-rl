"""
simulate_sensors_v2.py
======================
NETRA-RL (Week 2, Task A) - Causal Sensor Data Generator v2.

This script implements a Structural Causal Model (SCM) to generate synthetic
sensor data for railway/perimeter intrusion monitoring:

    Causal Architecture:
        Hour (h) -> Event (E) -> { PIR, Distance (d), Label }

Key Steps:
    1. Hour (h): Discrete uniform {0, ..., 23}.
       Regime: Day (5 <= h <= 19) vs. Night (h > 19 or h < 5).
    2. Event (E): Ground truth class in {"none", "human", "animal"} conditioned
       on Day vs. Night prior probabilities.
    3. PIR Sensor: Binary motion sensor {0, 1} sampled as a Bernoulli trial
       conditioned on Event (including thermal/wind false triggers).
    4. Distance Sensor (d): Base ultrasonic distance conditioned on Event,
       with additive Gaussian measurement noise, clipped to [0.5, 5.0] m.
    5. Label: Deterministic ground-truth intrusion target (1 if E != 'none' else 0).

Deliverables:
    - 5 reproducible datasets for seeds [42, 100, 287, 523, 1024] saved to data/
    - Automated sanity check verification report for Seed 42 saved to results/
    - High-resolution (300 DPI) causal graph DAG saved to results/causal_graph.png
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Default fixed seeds and row count as per specification
SEEDS = [42, 100, 287, 523, 1024]
NUM_SAMPLES = 5000

# Operational Scenarios
SCENARIOS = {
    "benchmark": {
        "name": "Benchmark SCM v2",
        "p_day": (0.85, 0.10, 0.05),
        "p_night": (0.80, 0.05, 0.15),
        "p_pir_none": 0.10,
        "noise_std": 0.05,
        "prefix": "sensor_v2",
        "desc": "Task A Baseline Standard (16% Intrusions)"
    },
    "intrusion_surge": {
        "name": "Intrusion Surge (High Human Alert)",
        "p_day": (0.45, 0.40, 0.15),
        "p_night": (0.40, 0.45, 0.15),
        "p_pir_none": 0.12,
        "noise_std": 0.06,
        "prefix": "sensor_v2_intrusion_surge",
        "desc": "Heavy Human Infiltration (~55% Intrusions)"
    },
    "wildlife_corridor": {
        "name": "Wildlife Migration Corridor",
        "p_day": (0.60, 0.08, 0.32),
        "p_night": (0.30, 0.05, 0.65),
        "p_pir_none": 0.10,
        "noise_std": 0.07,
        "prefix": "sensor_v2_wildlife_corridor",
        "desc": "Heavy Animal Movement (~50% Animals, nocturnal peak)"
    },
    "harsh_weather": {
        "name": "Harsh Weather / Wind Glare",
        "p_day": (0.75, 0.12, 0.13),
        "p_night": (0.70, 0.10, 0.20),
        "p_pir_none": 0.45,
        "noise_std": 0.25,
        "prefix": "sensor_v2_harsh_weather",
        "desc": "Storm/High Wind with 45% False PIR Triggers & 0.25m σ Noise"
    }
}

SCENARIO_ALIASES = {
    "stormy_weather": "harsh_weather",
    "weather": "harsh_weather",
    "harsh": "harsh_weather",
    "surge": "intrusion_surge",
    "wildlife": "wildlife_corridor",
    "corridor": "wildlife_corridor"
}

# Scenario PRNG offsets to guarantee distinct, independent pseudorandom streams per scenario
SCENARIO_OFFSETS = {
    "benchmark": 0,
    "intrusion_surge": 13337,
    "wildlife_corridor": 37783,
    "harsh_weather": 71153
}


def generate_sensor_data(
    seed: int,
    n_samples: int = NUM_SAMPLES,
    p_day: tuple = (0.85, 0.10, 0.05),
    p_night: tuple = (0.80, 0.05, 0.15),
    p_pir_none: float = 0.10,
    noise_std: float = 0.05
) -> pd.DataFrame:
    """
    Generate synthetic sensor data for a single seed following the 5-step causal chain.

    Parameters
    ----------
    seed : int
        Seed for numpy.random.default_rng for strict reproducibility.
    n_samples : int
        Number of rows to generate (default 5000).
    p_day : tuple
        (P(none), P(human), P(animal)) probabilities during day (5 <= h <= 19).
    p_night : tuple
        (P(none), P(human), P(animal)) probabilities during night (h > 19 or h < 5).
    p_pir_none : float
        False trigger probability P(pir=1 | none). Default 0.10.
    noise_std : float
        Standard deviation of Gaussian sensor noise. Default 0.05.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['hour', 'event', 'pir', 'distance', 'label'].
    """
    rng = np.random.default_rng(seed)

    # -------------------------------------------------------------------------
    # Step 1: Hour (hour)
    # Discrete uniform integer over {0, 1, ..., 23}
    # -------------------------------------------------------------------------
    hour = rng.integers(0, 24, size=n_samples)

    # Time-of-day regime:
    # Night: hour > 19 or hour < 5 (hours 20, 21, 22, 23, 0, 1, 2, 3, 4)
    # Day:   5 <= hour <= 19
    is_night = (hour > 19) | (hour < 5)
    day_indices = np.where(~is_night)[0]
    night_indices = np.where(is_night)[0]

    # -------------------------------------------------------------------------
    # Step 2: Ground-Truth Event (event)
    # Conditioned on Day vs. Night regime:
    #   Day:   P(none), P(human), P(animal)
    #   Night: P(none), P(human), P(animal)
    # -------------------------------------------------------------------------
    event = np.empty(n_samples, dtype=object)
    event_classes = ["none", "human", "animal"]

    event[day_indices] = rng.choice(
        event_classes, size=len(day_indices), p=list(p_day)
    )
    event[night_indices] = rng.choice(
        event_classes, size=len(night_indices), p=list(p_night)
    )

    # -------------------------------------------------------------------------
    # Step 3: PIR Sensor (pir)
    # Bernoulli trial in {0, 1} conditioned on event:
    #   P(pir = 1 | none)   = p_pir_none (simulates false positives)
    #   P(pir = 1 | human)  = 0.90
    #   P(pir = 1 | animal) = 0.85
    # -------------------------------------------------------------------------
    p_pir = np.select(
        [event == "none", event == "human", event == "animal"],
        [p_pir_none, 0.90, 0.85]
    )
    pir = (rng.random(size=n_samples) < p_pir).astype(int)

    # -------------------------------------------------------------------------
    # Step 4: Distance Sensor (distance)
    # Conditioned on event:
    #   If event == "none":           d_base ~ Uniform(3.5, 5.0)
    #   If event in [human, animal]:  d_base ~ Normal(loc=1.5, scale=0.6)
    # Add independent Gaussian sensor noise: noise ~ Normal(loc=0.0, scale=noise_std)
    # Physical sensor clipping: distance in [0.5, 5.0] meters.
    # -------------------------------------------------------------------------
    none_mask = (event == "none")
    target_mask = ~none_mask

    d_base = np.empty(n_samples, dtype=float)
    d_base[none_mask] = rng.uniform(3.5, 5.0, size=np.sum(none_mask))
    d_base[target_mask] = rng.normal(1.5, 0.6, size=np.sum(target_mask))

    sensor_noise = rng.normal(0.0, noise_std, size=n_samples)
    distance = np.clip(d_base + sensor_noise, 0.5, 5.0)

    # -------------------------------------------------------------------------
    # Step 5: Intrusion Label (label)
    # Deterministic mapping:
    #   label = 1 if event != "none" (human or animal intrusion)
    #   label = 0 if event == "none" (safe / background)
    # -------------------------------------------------------------------------
    label = np.where(event != "none", 1, 0)

    df = pd.DataFrame({
        "hour": hour,
        "event": event,
        "pir": pir,
        "distance": distance,
        "label": label
    })
    return df


def compute_sanity_metrics(df_data: pd.DataFrame, seed: int = None) -> list:
    """
    Compute theoretical vs empirical sanity check metrics for a given sensor dataset.
    """
    total_rows = len(df_data)
    is_night = (df_data["hour"] > 19) | (df_data["hour"] < 5)
    is_day = ~is_night

    count_none = int((df_data["event"] == "none").sum())
    count_human = int((df_data["event"] == "human").sum())
    count_animal = int((df_data["event"] == "animal").sum())

    pct_none = count_none / total_rows * 100.0
    pct_human = count_human / total_rows * 100.0
    pct_animal = count_animal / total_rows * 100.0

    p_label_day = float(df_data[is_day]["label"].mean()) if is_day.sum() > 0 else 0.0
    p_label_night = float(df_data[is_night]["label"].mean()) if is_night.sum() > 0 else 0.0

    p_label_pir1 = float(df_data[df_data["pir"] == 1]["label"].mean()) if (df_data["pir"] == 1).sum() > 0 else 0.0
    p_label_pir0 = float(df_data[df_data["pir"] == 0]["label"].mean()) if (df_data["pir"] == 0).sum() > 0 else 0.0

    mean_d_none = float(df_data[df_data["event"] == "none"]["distance"].mean()) if count_none > 0 else 0.0
    mean_d_human = float(df_data[df_data["event"] == "human"]["distance"].mean()) if count_human > 0 else 0.0
    mean_d_animal = float(df_data[df_data["event"] == "animal"]["distance"].mean()) if count_animal > 0 else 0.0

    records = [
        {
            "Metric": "Class Count: none",
            "Computed_Value": f"{count_none} ({pct_none:.1f}%)",
            "Expected_Value": "4156 (~83.1%)",
            "Tolerance": "approx ~4156 rows",
            "Passed": 3900 <= count_none <= 4400,
            "Raw_Value": count_none
        },
        {
            "Metric": "Class Count: human",
            "Computed_Value": f"{count_human} ({pct_human:.1f}%)",
            "Expected_Value": "406 (~8.1%)",
            "Tolerance": "approx ~406 rows",
            "Passed": 350 <= count_human <= 460,
            "Raw_Value": count_human
        },
        {
            "Metric": "Class Count: animal",
            "Computed_Value": f"{count_animal} ({pct_animal:.1f}%)",
            "Expected_Value": "438 (~8.8%)",
            "Tolerance": "approx ~438 rows",
            "Passed": 380 <= count_animal <= 500,
            "Raw_Value": count_animal
        },
        {
            "Metric": "P(label = 1 | Day)",
            "Computed_Value": f"{p_label_day:.4f}",
            "Expected_Value": "0.1500",
            "Tolerance": "[0.12, 0.18]",
            "Passed": 0.12 <= p_label_day <= 0.18,
            "Raw_Value": p_label_day
        },
        {
            "Metric": "P(label = 1 | Night)",
            "Computed_Value": f"{p_label_night:.4f}",
            "Expected_Value": "0.2000",
            "Tolerance": "[0.17, 0.23]",
            "Passed": 0.17 <= p_label_night <= 0.23,
            "Raw_Value": p_label_night
        },
        {
            "Metric": "P(label = 1 | PIR = 1)",
            "Computed_Value": f"{p_label_pir1:.4f}",
            "Expected_Value": "0.6400",
            "Tolerance": "[0.58, 0.70]",
            "Passed": 0.58 <= p_label_pir1 <= 0.70,
            "Raw_Value": p_label_pir1
        },
        {
            "Metric": "P(label = 1 | PIR = 0)",
            "Computed_Value": f"{p_label_pir0:.4f}",
            "Expected_Value": "0.0280",
            "Tolerance": "[0.015, 0.045]",
            "Passed": 0.015 <= p_label_pir0 <= 0.045,
            "Raw_Value": p_label_pir0
        },
        {
            "Metric": "Mean Distance: none",
            "Computed_Value": f"{mean_d_none:.4f} m",
            "Expected_Value": "4.2500 m",
            "Tolerance": "~4.25 m (+-0.10)",
            "Passed": 4.15 <= mean_d_none <= 4.35,
            "Raw_Value": mean_d_none
        },
        {
            "Metric": "Mean Distance: human",
            "Computed_Value": f"{mean_d_human:.4f} m",
            "Expected_Value": "1.5000 m",
            "Tolerance": "~1.50 m (+-0.15)",
            "Passed": 1.35 <= mean_d_human <= 1.65,
            "Raw_Value": mean_d_human
        },
        {
            "Metric": "Mean Distance: animal",
            "Computed_Value": f"{mean_d_animal:.4f} m",
            "Expected_Value": "1.5000 m",
            "Tolerance": "~1.50 m (+-0.15)",
            "Passed": 1.35 <= mean_d_animal <= 1.65,
            "Raw_Value": mean_d_animal
        },
    ]
    return records


def run_sanity_checks(df_data: pd.DataFrame, output_csv: str, seed: int = 42, verbose: bool = True) -> pd.DataFrame:
    """
    Run automated sanity check verification pipeline on a specific dataset.
    Prints formatted verification table to terminal and saves to output_csv.
    """
    records = compute_sanity_metrics(df_data, seed=seed)
    summary_df = pd.DataFrame(records)
    summary_df["Status"] = summary_df["Passed"].map({True: "PASS", False: "FAIL"})

    if verbose:
        header = "=" * 88
        print("\n" + header)
        print(f"NETRA-RL Task A: Seed {seed} Sanity Checks & Theoretical Verification")
        print(header)
        print(f"{'Metric':<25} | {'Computed':<15} | {'Expected':<15} | {'Tolerance':<18} | {'Status'}")
        print("-" * 88)
        for _, row in summary_df.iterrows():
            print(f"{row['Metric']:<25} | {row['Computed_Value']:<15} | {row['Expected_Value']:<15} | {row['Tolerance']:<18} | {row['Status']}")
        print(header)

        all_passed = summary_df["Passed"].all()
        if all_passed:
            print(f">>> ALL SANITY CHECKS PASSED FOR SEED {seed} <<<\n")
        else:
            print(f">>> WARNING: SOME SANITY CHECKS OUTSIDE TOLERANCE FOR SEED {seed} <<<\n")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    summary_df[["Metric", "Computed_Value", "Expected_Value", "Tolerance", "Status"]].to_csv(output_csv, index=False)
    return summary_df


def run_all_seeds_sanity_checks(datasets_dict: dict, output_csv: str) -> pd.DataFrame:
    """
    Generate comprehensive multi-seed sanity verification report covering all generated seeds.
    Produces a master comparison table showing every seed's computed values alongside expectations.
    """
    seeds = sorted(datasets_dict.keys())
    if not seeds:
        return pd.DataFrame()

    per_seed_records = {s: compute_sanity_metrics(datasets_dict[s], seed=s) for s in seeds}
    sample_records = per_seed_records[seeds[0]]

    table_rows = []
    for i, rec in enumerate(sample_records):
        row = {
            "Metric": rec["Metric"],
            "Expected_Value": rec["Expected_Value"],
            "Tolerance": rec["Tolerance"]
        }
        all_passed = True
        raw_vals = []
        for s in seeds:
            s_rec = per_seed_records[s][i]
            row[f"Seed_{s}"] = s_rec["Computed_Value"]
            raw_vals.append(s_rec["Raw_Value"])
            if not s_rec["Passed"]:
                all_passed = False

        if isinstance(raw_vals[0], (int, float)):
            mean_val = sum(raw_vals) / len(raw_vals)
            row["Mean_Value"] = f"{mean_val:.2f}" if isinstance(raw_vals[0], int) else f"{mean_val:.4f}"
        else:
            row["Mean_Value"] = "N/A"

        row["Overall_Status"] = "PASS" if all_passed else "FAIL"
        table_rows.append(row)

    master_df = pd.DataFrame(table_rows)

    # Print multi-seed master verification table
    header = "=" * 108
    print("\n" + header)
    print("NETRA-RL Task A: Consolidated Sanity Verification Covering ALL Seeds")
    print(header)
    col_str = f"{'Metric':<24} | {'Expected':<14} | " + " | ".join([f"Seed {s:<5}" for s in seeds]) + " | Status"
    print(col_str)
    print("-" * 108)
    for _, r in master_df.iterrows():
        seed_cols = " | ".join([f"{str(r[f'Seed_{s}']).split()[0]:<10}" for s in seeds])
        print(f"{r['Metric']:<24} | {r['Expected_Value']:<14} | {seed_cols} | {r['Overall_Status']}")
    print(header)
    all_all_passed = (master_df["Overall_Status"] == "PASS").all()
    if all_all_passed:
        print(">>> ALL SANITY CHECKS PASSED ACROSS ALL SEEDS <<<\n")
    else:
        print(">>> SOME CHECKS OUTSIDE TOLERANCE (CHECK INDIVIDUAL SEEDS) <<<\n")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    master_df.to_csv(output_csv, index=False)
    return master_df


def plot_causal_graph(output_png: str) -> None:
    """
    Draw and save the Structural Causal Model DAG to output_png at 300 DPI.

    Causal Graph Nodes:
        - Hour (h)
        - Event (E)
        - PIR
        - Distance (d)
        - Label

    Edges:
        Hour (h) -> Event (E)
        Event (E) -> PIR
        Event (E) -> Distance (d)
        Event (E) -> Label
    """
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    # Define nodes: positions, labels, descriptions, and styling colors
    nodes = {
        "hour": {
            "pos": (1.8, 3.5),
            "title": "Hour (h)",
            "desc": "Exogenous Context\nh in {0, ..., 23}\nDay vs. Night Regime",
            "border": "#2980B9",
            "bg": "#EBF5FB"
        },
        "event": {
            "pos": (5.0, 3.5),
            "title": "Event (E)",
            "desc": "Latent Ground Truth\nE in {none, human, animal}\nP(E | Day/Night)",
            "border": "#D35400",
            "bg": "#FEF9E7"
        },
        "pir": {
            "pos": (8.6, 5.3),
            "title": "PIR Sensor",
            "desc": "Binary Motion {0, 1}\nBernoulli(P(PIR=1 | E))\nFalse Triggers (wind/heat)",
            "border": "#16A085",
            "bg": "#E8F8F5"
        },
        "distance": {
            "pos": (8.6, 3.5),
            "title": "Distance (d)",
            "desc": "Ultrasonic Sensor [0.5, 5.0]m\nd = clip(d_base(E) + noise)\nnoise ~ N(0, 0.05²)",
            "border": "#16A085",
            "bg": "#E8F8F5"
        },
        "label": {
            "pos": (8.6, 1.7),
            "title": "Label",
            "desc": "Deterministic Target {0, 1}\n1 if E != 'none' else 0\n(Intrusion Ground Truth)",
            "border": "#8E44AD",
            "bg": "#F4ECF7"
        }
    }

    box_w, box_h = 2.4, 1.25

    # Draw node boxes with styled rounded borders and clean typography
    for node_id, data in nodes.items():
        x, y = data["pos"]
        rect = patches.FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=data["bg"],
            edgecolor=data["border"],
            linewidth=2.2
        )
        ax.add_patch(rect)
        ax.text(
            x, y + 0.24, data["title"],
            fontsize=12, fontweight="bold", ha="center", va="center",
            color=data["border"]
        )
        ax.text(
            x, y - 0.22, data["desc"],
            fontsize=8.5, ha="center", va="center",
            color="#2C3E50", linespacing=1.2
        )

    # Directed causal edges
    edges = [
        ("hour", "event"),
        ("event", "pir"),
        ("event", "distance"),
        ("event", "label")
    ]

    for src, dst in edges:
        x1, y1 = nodes[src]["pos"]
        x2, y2 = nodes[dst]["pos"]

        start_x = x1 + box_w / 2
        start_y = y1
        end_x = x2 - box_w / 2
        end_y = y2

        ax.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=2.4,
                color="#2C3E50",
                shrinkA=3,
                shrinkB=3,
                mutation_scale=18
            )
        )

    # Main graph title and descriptive causal equation
    plt.title(
        "NETRA-RL: Structural Causal Model (SCM)\nHour (h) → Event (E) → { PIR, Distance (d), Label }",
        fontsize=14,
        fontweight="bold",
        pad=18,
        color="#1A252F"
    )

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Causal graph DAG saved to: {output_png} (300 DPI)")


def main():
    parser = argparse.ArgumentParser(description="NETRA-RL Synthetic Causal Sensor Data Generator v2")
    parser.add_argument("--batch", type=int, default=None, help="Batch iteration number (advances PRNG stream for fresh randomized data)")
    parser.add_argument("--randomize", "--fresh", "--random", action="store_true", help="Generate fresh randomized samples for the 5 assignment seeds")
    parser.add_argument("--scenario", type=str, default="benchmark", help="Scenario to generate: benchmark, intrusion_surge, wildlife_corridor, harsh_weather, or all")
    parser.add_argument("--count", type=int, default=5, help="Number of datasets to generate (default: 5)")
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES, help="Number of rows per dataset (default: 5000)")
    args = parser.parse_args()

    # Resolve directory paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, "data")
    results_dir = os.path.join(project_root, "results")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Determine batch iteration for sample stream advancement
    counter_file = os.path.join(data_dir, ".batch_counter")
    if args.batch is not None:
        batch_index = max(1, args.batch)
    elif args.randomize:
        try:
            if os.path.exists(counter_file):
                with open(counter_file, "r") as f:
                    val = int(f.read().strip())
            else:
                val = 1
            batch_index = val + 1
            with open(counter_file, "w") as f:
                f.write(str(batch_index))
        except Exception:
            batch_index = int(time.time()) % 100000 + 2
    else:
        batch_index = 1

    raw_scenario = (args.scenario or "benchmark").lower().strip()
    scenario_key = SCENARIO_ALIASES.get(raw_scenario, raw_scenario)

    if scenario_key == "all":
        scenarios_to_run = ["benchmark", "intrusion_surge", "wildlife_corridor", "harsh_weather"]
    elif scenario_key in SCENARIOS:
        scenarios_to_run = [scenario_key]
    else:
        print(f"Warning: Unknown scenario '{args.scenario}', falling back to benchmark.")
        scenarios_to_run = ["benchmark"]

    print("=" * 88)
    print("NETRA-RL Task A: Synthetic Causal Sensor Data Generator v2")
    print("=" * 88)
    print(f"Target data directory:    {data_dir}")
    print(f"Target results directory: {results_dir}")
    print(f"Scenario(s) to generate:  {', '.join(scenarios_to_run)}")
    print(f"Seeds to generate:        {SEEDS}")
    print(f"Batch iteration:          Batch #{batch_index} ({'Fresh Randomized Samples' if batch_index > 1 else 'Canonical Standard'})")
    print(f"Samples per dataset:      {NUM_SAMPLES}")
    print("-" * 88)

    benchmark_datasets = {}
    total_files_generated = 0

    for sc_key in scenarios_to_run:
        cfg = SCENARIOS[sc_key]
        prefix = cfg["prefix"]
        print(f"\n[Scenario: {cfg['name']}]")
        print(f"  Distribution: Day={cfg['p_day']}, Night={cfg['p_night']}, False PIR={cfg['p_pir_none']}, Noise σ={cfg['noise_std']}")

        for seed in SEEDS:
            sc_offset = SCENARIO_OFFSETS.get(sc_key, 0)
            if sc_key == "benchmark" and batch_index == 1:
                effective_seed = seed
            else:
                effective_seed = (seed + sc_offset + (batch_index - 1) * 7919) % 2147483647
            df = generate_sensor_data(
                effective_seed,
                NUM_SAMPLES,
                p_day=cfg["p_day"],
                p_night=cfg["p_night"],
                p_pir_none=cfg["p_pir_none"],
                noise_std=cfg["noise_std"]
            )

            csv_filename = f"{prefix}_seed{seed}.csv"
            csv_path = os.path.join(data_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            total_files_generated += 1

            if sc_key == "benchmark":
                benchmark_datasets[seed] = df
            elif len(scenarios_to_run) == 1:
                # If generating a specific single scenario, mirror to sensor_v2_seed{seed}.csv for active work
                active_csv_path = os.path.join(data_dir, f"sensor_v2_seed{seed}.csv")
                df.to_csv(active_csv_path, index=False)
                benchmark_datasets[seed] = df

            pct_intrusion = (df["label"] == 1).mean() * 100.0
            pct_human = (df["event"] == "human").mean() * 100.0
            pct_animal = (df["event"] == "animal").mean() * 100.0
            pir_rate = (df["pir"] == 1).mean() * 100.0
            print(f"  • {csv_filename:<36} | Intrusions: {pct_intrusion:5.1f}% | Human: {pct_human:4.1f}% | Animal: {pct_animal:4.1f}% | PIR: {pir_rate:4.1f}%")

    # Run automated sanity checks covering ALL seeds of data that were just saved
    sanity_dict = benchmark_datasets if len(benchmark_datasets) > 0 else {42: df}
    for s_id, s_df in sanity_dict.items():
        s_csv_path = os.path.join(data_dir, f"task_a_sanity_seed{s_id}.csv")
        run_sanity_checks(s_df, s_csv_path, seed=s_id, verbose=(s_id == 42))
        if os.path.exists(results_dir):
            try:
                run_sanity_checks(s_df, os.path.join(results_dir, f"task_a_sanity_seed{s_id}.csv"), seed=s_id, verbose=False)
            except Exception:
                pass
        print(f"Sanity verification for Seed {s_id} saved to: {s_csv_path}")

    # Generate and save master consolidated report covering ALL seeds
    all_seeds_data_path = os.path.join(data_dir, "task_a_sanity_all_seeds.csv")
    run_all_seeds_sanity_checks(sanity_dict, all_seeds_data_path)
    if os.path.exists(results_dir):
        try:
            run_all_seeds_sanity_checks(sanity_dict, os.path.join(results_dir, "task_a_sanity_all_seeds.csv"))
        except Exception:
            pass
    print(f"Consolidated sanity verification covering ALL seeds saved to: {all_seeds_data_path}")

    # Generate causal graph diagram
    causal_data_path = os.path.join(data_dir, "causal_graph.png")
    plot_causal_graph(causal_data_path)
    if os.path.exists(results_dir):
        try:
            causal_results_path = os.path.join(results_dir, "causal_graph.png")
            plot_causal_graph(causal_results_path)
        except Exception:
            pass
    print(f"Causal graph DAG saved to: {causal_data_path} (300 DPI)")

    print("=" * 88)
    print(f"Batch generation complete: {total_files_generated} dataset(s) generated successfully!")
    print("=" * 88)


if __name__ == "__main__":
    main()
