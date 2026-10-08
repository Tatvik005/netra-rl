"""
run_baseline_eval.py -- evaluate the NETRA fixed-rule baseline on simulated scenarios
====================================================================================

For every scenario (fusion_79, events_113, field_97) x platform (pi_zero, pi4) x method
(pir_only, binary, probabilistic tau=0.65, probabilistic tau=0.45) it simulates
N seeds, runs the paper's Algorithms 1+2, and reports mean / std of the metrics.
It then checks the fusion_79 scenario against the paper's Tables III-V and
Sec. IV-C, the vision sampler against Tables VI-VII, and latency against Table II.

Outputs
    results/baseline_metrics.csv      mean/std of every metric per configuration
    results/baseline_validation.csv   simulator vs paper numbers
    results/baseline_eval.png         (only if matplotlib is installed)

Usage
    python agent/run_baseline_eval.py                # 200 seeds (about 1-2 min)
    python agent/run_baseline_eval.py --seeds 50 --no-plot
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_BASE, "data"))
sys.path.insert(0, os.path.join(_BASE, "agent"))
from netra_simulator import (CLASSES, PAPER_CLASSIFICATION, PLATFORMS, SCENARIOS,  # noqa: E402
                             SX127X_SENSITIVITY_DBM, SimConfig, confusion_probs, lora_airtime_s,
                             lora_delivery_prob, lora_tx_energy_j, path_loss_db, simulate)
from fixed_rule_baseline import BaselineConfig, evaluate  # noqa: E402

METHODS = [BaselineConfig(method="pir_only"), BaselineConfig(method="binary"),
           BaselineConfig(method="probabilistic", tau_c=0.65),
           BaselineConfig(method="probabilistic", tau_c=0.45)]
KEY = ["scenario", "platform", "method"]


def run(n_seeds: int, seed0: int) -> pd.DataFrame:
    rows = []
    for sc in SCENARIOS:
        for pf in PLATFORMS:
            for s in range(seed0, seed0 + n_seeds):
                res = simulate(SimConfig(scenario=sc, platform=pf, seed=s))
                ms = []
                for bc in METHODS:
                    bc.seed = s
                    ms.append(evaluate(res, bc))
                pir_cam = ms[0]["camera_energy_wh"]; pir_act = ms[0]["action_energy_wh"]
                for m in ms:
                    m["camera_energy_saving_vs_pir_only"] = 1 - m["camera_energy_wh"] / pir_cam
                    m["action_energy_saving_vs_pir_only"] = 1 - m["action_energy_wh"] / pir_act
                rows += ms
    return pd.DataFrame(rows)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in raw.columns if c not in KEY + ["sim_seed", "tau_c", "sf", "lam"]]
    g = raw.groupby(KEY, sort=False)[num]
    mean = g.mean().add_suffix("_mean"); std = g.std().add_suffix("_std")
    out = pd.concat([mean, std], axis=1)
    cols = [c for pair in zip(mean.columns, std.columns) for c in pair]
    out = out[cols].reset_index()
    out.insert(3, "n_seeds", raw.groupby(KEY, sort=False).size().values)
    return out


def classification_check(n_per_class: int = 20000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pf in PLATFORMS:
        preds, trues = [], []
        for c in CLASSES:
            p = confusion_probs(pf, c)
            preds += list(rng.choice(CLASSES, size=n_per_class, p=p)); trues += [c] * n_per_class
        preds = np.array(preds); trues = np.array(trues)
        # weight classes like the paper's test set
        counts = {c: sum(PLATFORMS[pf]["confusion"][c].values()) for c in CLASSES}
        w = np.array([counts[t] / n_per_class for t in trues])
        for c in CLASSES:
            tp = w[(preds == c) & (trues == c)].sum(); fp = w[(preds == c) & (trues != c)].sum()
            fn = w[(preds != c) & (trues == c)].sum()
            prec = tp / (tp + fp); rec = tp / (tp + fn); f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            pp, pr, pf1 = PAPER_CLASSIFICATION[pf][c]
            rows.append(dict(check=f"vision {pf} {c} F1 %", paper=pf1, simulated=round(100 * f1, 1)))
        acc = w[preds == trues].sum() / w.sum()
        rows.append(dict(check=f"vision {pf} overall accuracy %", paper=PAPER_CLASSIFICATION[pf]["overall"],
                         simulated=round(100 * acc, 1)))
    return pd.DataFrame(rows)


def lora_sf_table(tx_dbm: float = 20.0, gain_dbi: float = 2.15, shadow_sd: float = 4.0,
                  distances=(1000, 1500, 2000, 3000), n: int = 20000, seed: int = 0) -> pd.DataFrame:
    """SYNTHETIC link model: mean per-attempt delivery vs SF and link distance (clear weather)."""
    rng = np.random.default_rng(seed)
    rows = []
    for sf in range(7, 13):
        r = dict(sf=sf, airtime_ms=round(1000 * lora_airtime_s(sf), 1),
                 tx_energy_mJ=round(1000 * lora_tx_energy_j(sf, tx_dbm), 1),
                 sensitivity_dbm=SX127X_SENSITIVITY_DBM[sf])
        for d in distances:
            rssi = tx_dbm + 2 * gain_dbi - path_loss_db(d) + rng.normal(0, shadow_sd, n)
            r[f"p_deliver_{d}m"] = round(float(np.mean([lora_delivery_prob(x, sf) for x in rssi[:4000]])), 4)
        rows.append(r)
    return pd.DataFrame(rows)


def validation(agg: pd.DataFrame) -> pd.DataFrame:
    def get(sc, pf, method, col):
        r = agg[(agg.scenario == sc) & (agg.platform == pf) & (agg.method == method)]
        return float(r[col + "_mean"].iloc[0]), float(r[col + "_std"].iloc[0])
    rows = []
    def add(name, paper, sc, pf, method, col, scale=1.0, tol=None):
        m, s = get(sc, pf, method, col)
        rows.append(dict(check=name, paper=paper, simulated=round(m * scale, 2), sim_std=round(s * scale, 2),
                         scenario=sc, platform=pf, method=method,
                         within_tol=(abs(m * scale - paper) <= tol) if tol is not None else None))
    P = "pi_zero"
    add("PIR-only camera activations", 79, "fusion_79", P, "pir_only", "camera_activations", tol=0.5)
    add("PIR-only false triggers", 39, "fusion_79", P, "pir_only", "false_activations", tol=0.5)
    add("Binary fusion true detections (of 40)", 34, "fusion_79", P, "binary", "true_activations", tol=2)
    add("Binary fusion detection rate %", 85.0, "fusion_79", P, "binary", "fusion_detection_rate", 100, tol=5)
    add("Binary fusion false triggers", 0, "fusion_79", P, "binary", "false_activations", tol=0.5)
    add("Prob. fusion tau=0.65 detection rate %", 95.0, "fusion_79", P, "probabilistic_tau0.65", "fusion_detection_rate", 100, tol=3)
    add("Prob. fusion tau=0.65 camera activations", 38, "fusion_79", P, "probabilistic_tau0.65", "camera_activations", tol=2)
    add("Prob. fusion tau=0.65 false activations", 0, "fusion_79", P, "probabilistic_tau0.65", "false_activations", tol=0.5)
    add("Prob. fusion tau=0.45 camera activations", 40, "fusion_79", P, "probabilistic_tau0.45", "camera_activations", tol=2)
    add("Prob. fusion tau=0.45 false activations", 2, "fusion_79", P, "probabilistic_tau0.45", "false_activations", tol=1)
    add("Camera energy PIR-only (Wh)", 0.220, "fusion_79", P, "pir_only", "camera_energy_wh", tol=0.01)
    add("Camera energy tau=0.65 (Wh)", 0.106, "fusion_79", P, "probabilistic_tau0.65", "camera_energy_wh", tol=0.01)
    add("Camera energy saving tau=0.65 vs PIR-only %", 51.9, "fusion_79", P, "probabilistic_tau0.65", "camera_energy_saving_vs_pir_only", 100, tol=3)
    add("Pi Zero end-to-end latency (s)", 6.5, "fusion_79", "pi_zero", "probabilistic_tau0.65", "latency_mean_s", tol=0.5)
    add("Pi 4 end-to-end latency (s)", 2.4, "fusion_79", "pi4", "probabilistic_tau0.65", "latency_mean_s", tol=0.5)
    add("LoRa packet delivery % (1 km)", 100.0, "fusion_79", P, "probabilistic_tau0.65", "lora_delivery_rate", 100, tol=1)
    add("113-events: triggers removed by fusion %", 62.8, "events_113", P, "probabilistic_tau0.65", "camera_activations", tol=None)
    rows[-1]["simulated"] = round(100 * (1 - get("events_113", P, "probabilistic_tau0.65", "camera_activations")[0] / 113), 2)
    rows[-1]["within_tol"] = abs(rows[-1]["simulated"] - 62.8) <= 5
    add("113-events: confirmed alerts", 10, "events_113", P, "probabilistic_tau0.65", "alerts_transmitted", tol=2)
    add("113-events: raw trigger reduction %", 91.2, "events_113", P, "probabilistic_tau0.65", "raw_trigger_reduction", 100, tol=3)
    add("Field-97 alerts transmitted (medium sent)", 77, "field_97", P, "probabilistic_tau0.65", "alerts_transmitted", tol=5)
    add("Field-97 false alarm rate %", 0.0, "field_97", P, "probabilistic_tau0.65", "false_alarm_rate", 100, tol=1)
    add("20-intrusion field detection Pi Zero % (Table II 14/20)", 70.0, "events_113", P, "probabilistic_tau0.65", "dangerous_alert_rate", 100, tol=10)
    add("20-intrusion field detection Pi 4 % (Table II 19/20)", 95.0, "events_113", "pi4", "probabilistic_tau0.65", "dangerous_alert_rate", 100, tol=10)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(_BASE, "results"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()
    raw = run(a.seeds, a.seed0)
    agg = aggregate(raw)
    # extra field_97 variant: transmit Medium (cattle) alerts as Table X implies
    extra = []
    for s in range(a.seed0, a.seed0 + a.seeds):
        for pf in PLATFORMS:
            res = simulate(SimConfig(scenario="field_97", platform=pf, seed=s))
            m = evaluate(res, BaselineConfig(tau_c=0.65, transmit_medium=True, seed=s))
            m["method"] = "probabilistic_tau0.65_send_medium"
            m["frac_events_conf_gt_0.8"] = float((res.events.pred_conf > 0.8).mean())
            extra.append(m)
    ex = aggregate(pd.DataFrame(extra).assign(camera_energy_saving_vs_pir_only=np.nan, action_energy_saving_vs_pir_only=np.nan))
    agg = pd.concat([agg, ex], ignore_index=True)
    agg.to_csv(os.path.join(a.outdir, "baseline_metrics.csv"), index=False)

    val = validation(agg)
    # use the send-medium variant for the Table X alert count
    r = ex[(ex.platform == "pi_zero")]
    idx = val.index[val.check.str.startswith("Field-97 alerts")][0]
    val.loc[idx, ["simulated", "sim_std"]] = [round(float(r["alerts_transmitted_mean"].iloc[0]), 2), round(float(r["alerts_transmitted_std"].iloc[0]), 2)]
    val.loc[idx, "method"] = "probabilistic_tau0.65_send_medium"
    val.loc[idx, "within_tol"] = abs(val.loc[idx, "simulated"] - 77) <= 5
    val = pd.concat([val, pd.DataFrame([dict(
        check="Field-97 events with confidence > 0.8 % (Table X 38/97)", paper=39.2,
        simulated=round(100 * float(r["frac_events_conf_gt_0.8_mean"].iloc[0]), 2),
        sim_std=round(100 * float(r["frac_events_conf_gt_0.8_std"].iloc[0]), 2),
        scenario="field_97", platform="pi_zero", method="(vision sampler)",
        within_tol=abs(100 * float(r["frac_events_conf_gt_0.8_mean"].iloc[0]) - 39.2) <= 5)])], ignore_index=True)
    val = pd.concat([val, classification_check().assign(within_tol=lambda x: (x.paper - x.simulated).abs() <= 2)], ignore_index=True)
    val.to_csv(os.path.join(a.outdir, "baseline_validation.csv"), index=False)

    sft = lora_sf_table(); sft.to_csv(os.path.join(a.outdir, "lora_sf_table.csv"), index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    cols = ["scenario", "platform", "method", "camera_activations_mean", "false_activations_mean",
            "fusion_detection_rate_mean", "camera_energy_saving_vs_pir_only_mean", "alerts_transmitted_mean",
            "dangerous_alert_rate_mean", "false_alarm_rate_mean", "lora_delivery_rate_mean",
            "energy_per_alert_j_mean", "latency_mean_s_mean", "battery_end_frac_mean"]
    short = agg[cols].copy()
    short.columns = ["scenario", "platform", "method", "cam_act", "false_act", "fus_det", "camE_save",
                     "alerts", "dang_alert", "FAR", "LoRa", "J/alert", "lat_s", "batt_end"]
    print(f"\n=== Fixed-rule baseline, mean over {a.seeds} seeds (SF9 fixed, lambda=0.5) ===")
    print(short.round(3).to_string(index=False))
    print("\n=== Validation vs arXiv:2605.08246v1 ===")
    print(val[["check", "paper", "simulated", "sim_std", "within_tol"]].to_string(index=False))
    print("\n=== SYNTHETIC LoRa link model: per-attempt delivery by SF / distance (+20 dBm) ===")
    print(sft.to_string(index=False))
    print(f"\nSaved {os.path.join(a.outdir, 'baseline_metrics.csv')}, baseline_validation.csv, lora_sf_table.csv "
          f"({time.time() - t0:.1f} s)")

    if not a.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            sub = agg[(agg.scenario == "fusion_79") & (agg.platform == "pi_zero")]
            fig, ax = plt.subplots(1, 3, figsize=(14, 4))
            names = [m.replace("probabilistic_", "prob ") for m in sub.method]
            ax[0].bar(names, sub.camera_activations_mean, yerr=sub.camera_activations_std, color="#3498db", label="all")
            ax[0].bar(names, sub.false_activations_mean, color="#e74c3c", label="false")
            ax[0].set_title("Camera activations (fusion_79, Pi Zero)"); ax[0].legend()
            ax[1].bar(names, 100 * sub.fusion_detection_rate_mean, yerr=100 * sub.fusion_detection_rate_std, color="#2ecc71")
            ax[1].axhline(95, ls="--", c="k", lw=1); ax[1].set_title("Fusion detection rate % (paper 95/85)")
            ax[2].bar(names, sub.camera_energy_wh_mean, color="#f39c12")
            ax[2].axhline(0.106, ls="--", c="k", lw=1); ax[2].axhline(0.220, ls=":", c="k", lw=1)
            ax[2].set_title("Camera energy Wh (paper 0.106 / 0.220)")
            for x in ax: x.tick_params(axis="x", rotation=20)
            plt.tight_layout(); p = os.path.join(a.outdir, "baseline_eval.png"); plt.savefig(p, dpi=150); plt.close()
            print(f"Saved {p}")
        except ImportError:
            print("matplotlib not installed -- skipping plot")


if __name__ == "__main__":
    main()
