"""
fixed_rule_baseline.py -- NETRA fixed-rule pipeline (paper Algorithms 1 + 2)
============================================================================

Implements the legacy NETRA decision logic of arXiv:2605.08246v1 so that any learned
policy (e.g. the planned DQN) can be compared against it on identical simulated events
from data/netra_simulator.py.

Algorithm 1 -- event-driven camera activation                              [PAPER]
    if PIR == 0                         -> camera off
    dd = D_bg - D
    if dd <= 0 or D < 4 m or D > 15 m   -> camera off
    P_dist      = min(dd / 1.5, 1)
    P_intrusion = 0.4 * PIR + 0.6 * P_dist
    camera on  iff P_intrusion >= tau_c   (tau_c = 0.65 default, 0.45 variant)
  With PIR in {0,1} this is exactly: PIR=1, 4<=D<=15, dd >= 0.625 m (tau 0.45: dd >= 0.125 m).

Reference methods (paper Tables III-V)
    pir_only : camera on every PIR trigger                                     [PAPER]
    binary   : PIR AND ultrasonic-validated (U_t = 1). The paper does not define U_t;
               here U_t = 1 iff 4<=D<=15 and dd >= binary_delta_m (1.0 m)       [ASSUMED]
  For pir_only/binary, P_intrusion for Algorithm 2 is still 0.4*PIR + 0.6*P_dist
  (P_dist = 0 when the reading is rejected)                                    [ASSUMED]

Algorithm 2 -- classification + LoRa alert                                  [PAPER]
    drop if label == background or conf < tau_AI (0.50)
    IPS = lambda * conf + (1 - lambda) * P_intrusion ; drop if IPS < tau_alert (0.60)
    lambda = 0.5                                                         [ASSUMED -- not given]
    priority (eq. 7): Critical  label in {elephant, human} and conf >= 0.7
                      High      label == obstacle and conf >= 0.6
                      Medium    label == animal(cow) and conf >= 0.5  -> logged, not sent
                      Low       otherwise                             -> suppressed
    Critical/High -> LoRa payload with ACK; up to max_attempts (3) tries   [PAPER "retry", 3 ASSUMED]
    transmit_medium=True also sends Medium alerts (the paper's Table X counts
    cattle/horse among transmitted alerts, contradicting eq. 7)

LoRa baseline: FIXED SF9 at +20 dBm (mid-range SF; ~100% delivery at the paper's 1 km
link in the synthetic link model, 185 ms airtime). The paper says SF is "adaptive"
but gives no rule, so a fixed SF is the simplest honest baseline.            [ASSUMED]

Energy accounting (increments on top of continuous idle power, which is charged on
the simulator's time grid):
    ultrasonic ping 0.005 J per PIR event                                   [SYNTHETIC]
    camera 2 W x 5 s = 10 J per activation                                   [PAPER IV-C]
    inference (P_inf - P_idle) * t_inf : Pi Zero 2.0 W x 5.2 s = 10.4 J,
                                         Pi 4    4.8 W x 0.8 s = 3.84 J      [DERIVED]
    LoRa TX I_tx*V*airtime per attempt + RX ACK window                     [DATASHEET]
Latency (alerts only): fusion 0.18 s + camera 1.0 s + inference + 0.2 s processing +
    per attempt (airtime + ACK airtime) + 1.0 s back-off per failed attempt.

Metrics returned by evaluate(): see METRIC_DOC.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_BASE, "data"))
from netra_simulator import (  # noqa: E402
    PAPER, PLATFORMS, SimResult, lora_airtime_s, lora_delivery_prob, lora_rx_energy_j,
    lora_tx_energy_j,
)

LATENCY = {"camera_capture_s": 1.0, "processing_s": 0.2, "retry_backoff_s": 1.0}   # [SYNTHETIC]
ULTRASONIC_PING_J = 0.005                                                         # [SYNTHETIC]


@dataclass
class BaselineConfig:
    method: str = "probabilistic"          # 'pir_only' | 'binary' | 'probabilistic'
    tau_c: float = PAPER["tau_c"]
    w_pir: float = PAPER["w_pir"]
    w_dist: float = PAPER["w_dist"]
    d_max_norm_m: float = PAPER["d_max_norm_m"]
    d_valid_min_m: float = PAPER["d_valid_min_m"]
    d_valid_max_m: float = PAPER["d_valid_max_m"]
    binary_delta_m: float = 1.0            # ASSUMED
    tau_ai: float = PAPER["tau_ai"]
    tau_alert: float = PAPER["tau_alert"]
    lam: float = 0.5                       # ASSUMED
    prio_critical: float = PAPER["prio_critical"]
    prio_high: float = PAPER["prio_high"]
    prio_medium: float = PAPER["prio_medium"]
    transmit_medium: bool = False
    sf: int = 9                            # ASSUMED fixed SF
    tx_power_dbm: float = 20.0
    max_attempts: int = 3                  # ASSUMED
    sigma_fast_db: float = 3.0             # SYNTHETIC
    seed: int = 0                          # RNG for LoRa delivery draws
    brownout: bool = False                 # if True, node takes no action while battery is empty

    @property
    def name(self) -> str:
        if self.method == "probabilistic":
            return f"probabilistic_tau{self.tau_c:.2f}"
        return self.method


def algorithm1(pir: int, distance_m: float, d_bg_est_m: float, cfg: BaselineConfig):
    """Returns (camera_on, p_intrusion, p_dist, reason)."""
    dd = d_bg_est_m - distance_m
    valid = (dd > 0) and (cfg.d_valid_min_m <= distance_m <= cfg.d_valid_max_m)
    p_dist = min(dd / cfg.d_max_norm_m, 1.0) if valid else 0.0
    p_intr = cfg.w_pir * pir + cfg.w_dist * p_dist
    if cfg.method == "pir_only":
        return bool(pir == 1), p_intr, p_dist, "pir" if pir else "pir0"
    if pir == 0:
        return False, 0.0, 0.0, "pir0"
    if not valid:
        return False, p_intr, 0.0, "rejected_range"
    if cfg.method == "binary":
        on = dd >= cfg.binary_delta_m
        return on, p_intr, p_dist, "binary_on" if on else "binary_off"
    on = p_intr >= cfg.tau_c
    return on, p_intr, p_dist, "fusion_on" if on else "below_tau"


def priority(label: str, conf: float, cfg: BaselineConfig) -> str:
    if label in ("elephant", "human") and conf >= cfg.prio_critical:
        return "critical"
    if label == "obstacle" and conf >= cfg.prio_high:
        return "high"
    if label == "cow" and conf >= cfg.prio_medium:
        return "medium"
    return "low"


def algorithm2(camera_on: bool, p_intr: float, label: str, conf: float, cfg: BaselineConfig):
    """Returns (decision, priority, ips). decision in none/dropped_ai/dropped_ips/suppressed/logged/transmit."""
    if not camera_on:
        return "none", "", float("nan")
    if label == "background" or conf < cfg.tau_ai:
        return "dropped_ai", "", float("nan")
    ips = cfg.lam * conf + (1 - cfg.lam) * p_intr
    if ips < cfg.tau_alert:
        return "dropped_ips", "", ips
    pr = priority(label, conf, cfg)
    if pr in ("critical", "high") or (pr == "medium" and cfg.transmit_medium):
        return "transmit", pr, ips
    return ("logged" if pr == "medium" else "suppressed"), pr, ips


def lora_send(rssi_dbm: float, cfg: BaselineConfig, rng: np.random.Generator):
    """Returns (delivered, attempts, energy_j, latency_s)."""
    p = lora_delivery_prob(rssi_dbm, cfg.sf, cfg.sigma_fast_db)
    e_tx = lora_tx_energy_j(cfg.sf, cfg.tx_power_dbm)
    e_rx = lora_rx_energy_j(cfg.sf)
    t_air = lora_airtime_s(cfg.sf) + lora_airtime_s(cfg.sf, 4)
    delivered, energy, lat = False, 0.0, 0.0
    for a in range(1, cfg.max_attempts + 1):
        energy += e_tx + e_rx
        lat += t_air
        data_ok = rng.random() < p
        ack_ok = data_ok and (rng.random() < p)
        delivered = delivered or data_ok
        if ack_ok:
            return delivered, a, energy, lat
        lat += LATENCY["retry_backoff_s"]
    return delivered, cfg.max_attempts, energy, lat


METRIC_DOC = {
    "n_events": "PIR/quiet events in the run",
    "n_intrusions / n_non_intrusions": "ground-truth counts",
    "camera_activations / true_activations / false_activations": "Algorithm 1 output",
    "fusion_detection_rate": "true_activations / n_intrusions (paper Tables III-IV metric)",
    "camera_energy_wh": "activations x 10 J (paper Sec. IV-C energy metric)",
    "action_energy_wh": "ultrasonic + camera + inference increment + LoRa TX/RX",
    "alerts_transmitted / true_alerts / false_alerts": "Algorithm 2 transmit decisions",
    "dangerous_alert_rate": "dangerous intrusions (human/elephant/obstacle) with a transmitted alert / all dangerous intrusions",
    "e2e_detection_rate": "intrusions transmitted OR logged / n_intrusions",
    "false_alarm_rate": "false_alerts / n_non_intrusions (paper definition)",
    "correct_label_rate": "transmitted alerts whose label == true class",
    "raw_trigger_reduction": "1 - alerts_transmitted / PIR events (paper's '91.2%' style metric)",
    "lora_delivery_rate": "delivered / transmitted",
    "energy_per_alert_j": "action energy / delivered alerts",
    "latency_mean_s": "mean detection-to-alert latency over transmitted alerts",
    "battery_min_frac / battery_end_frac": "SYNTHETIC battery incl. actions (at events)",
    "events_at_empty_battery": "events that occurred while the battery was empty (node would be down if brownout=True)",
}


def evaluate(sim: Union[SimResult, pd.DataFrame], cfg: BaselineConfig,
             return_events: bool = False):
    """Run the fixed-rule pipeline over one simulated run. Accepts a SimResult (exact
    fine-grid battery) or a DataFrame loaded from CSV (coarse battery between events)."""
    if isinstance(sim, SimResult):
        ev = sim.events
    else:
        ev = sim
    platform = ev["platform"].iloc[0]
    plat = PLATFORMS[platform]
    rng = np.random.default_rng(cfg.seed)
    infer_j = (plat["infer_w"] - plat["idle_w"]) * plat["infer_s"]
    cam_j = PAPER["camera_w"] * PAPER["camera_on_s"]

    cap = float(ev["battery_capacity_wh"].iloc[0])
    fine = isinstance(sim, SimResult)
    if fine:
        b = cap * sim.config.battery_init_frac
        net = sim.grid_net_wh
    else:
        b = float(ev["battery_wh_idle_only"].iloc[0])
    pos = 0
    recs = []
    for i, r in enumerate(ev.itertuples(index=False)):
        # advance battery to this event
        if fine:
            step = int(r.grid_step)
            while pos < step:
                b = min(cap, max(0.0, b + net[pos])); pos += 1
        elif i > 0:
            b = min(cap, max(0.0, b + r.solar_wh_since_prev - r.idle_wh_since_prev))
        empty = b <= 1e-9
        if empty and cfg.brownout:
            recs.append(dict(camera_on=0, p_intrusion=0.0, p_dist=0.0, a1_reason="brownout",
                             a2_decision="none", priority="", ips=float("nan"), delivered=0,
                             attempts=0, energy_camera_j=0.0, energy_infer_j=0.0, energy_lora_j=0.0,
                             energy_action_j=0.0, latency_s=float("nan"), battery_empty=1, battery_wh=b))
            continue
        cam, p_intr, p_dist, reason = algorithm1(int(r.pir), float(r.distance_m), float(r.d_bg_est_m), cfg)
        e = ULTRASONIC_PING_J if r.pir == 1 else 0.0
        e_cam = cam_j if cam else 0.0
        e_inf = infer_j if cam else 0.0
        decision, pr, ips = algorithm2(cam, p_intr, r.pred_label, float(r.pred_conf), cfg)
        delivered, attempts, e_lora, lat = False, 0, 0.0, float("nan")
        if decision == "transmit":
            delivered, attempts, e_lora, t_lora = lora_send(float(r.rssi_dbm) + (cfg.tx_power_dbm - float(r.tx_power_dbm)), cfg, rng)
            lat = PAPER["fusion_runtime_s"] + LATENCY["camera_capture_s"] + plat["infer_s"] + \
                LATENCY["processing_s"] + t_lora
        recs.append(dict(camera_on=int(cam), p_intrusion=p_intr, p_dist=p_dist, a1_reason=reason,
                         a2_decision=decision, priority=pr, ips=ips, delivered=int(delivered),
                         attempts=attempts, energy_camera_j=e_cam, energy_infer_j=e_inf,
                         energy_lora_j=e_lora, energy_action_j=e + e_cam + e_inf + e_lora,
                         latency_s=lat, battery_empty=int(empty)))
        b = max(0.0, b - (e + e_cam + e_inf + e_lora) / 3600.0)
        recs[-1]["battery_wh"] = b
    d = pd.concat([ev.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    batt = d["battery_wh"].to_numpy()

    intr = d["is_intrusion"] == 1
    dang = d["is_dangerous"] == 1
    tx = d["a2_decision"] == "transmit"
    logged = d["a2_decision"] == "logged"
    n_intr = int(intr.sum()); n_non = int((~intr).sum()); n_dang = int(dang.sum())
    n_tx = int(tx.sum()); n_deliv = int(d["delivered"].sum())
    act_e = float(d["energy_action_j"].sum())
    pir_events = int((d["pir"] == 1).sum())
    m = dict(
        scenario=d["scenario"].iloc[0], platform=platform, method=cfg.name, tau_c=cfg.tau_c if cfg.method == "probabilistic" else float("nan"),
        sf=cfg.sf, lam=cfg.lam, sim_seed=int(d["seed"].iloc[0]),
        n_events=len(d), pir_events=pir_events, n_intrusions=n_intr, n_non_intrusions=n_non,
        camera_activations=int(d["camera_on"].sum()),
        true_activations=int((d["camera_on"].astype(bool) & intr).sum()),
        false_activations=int((d["camera_on"].astype(bool) & ~intr).sum()),
        fusion_detection_rate=(d["camera_on"].astype(bool) & intr).sum() / max(n_intr, 1),
        camera_energy_wh=float(d["energy_camera_j"].sum()) / 3600.0,
        action_energy_wh=act_e / 3600.0,
        alerts_transmitted=n_tx, true_alerts=int((tx & intr).sum()), false_alerts=int((tx & ~intr).sum()),
        logged_medium=int(logged.sum()),
        dangerous_alert_rate=(tx & dang).sum() / max(n_dang, 1) if n_dang else float("nan"),
        e2e_detection_rate=((tx | logged) & intr).sum() / max(n_intr, 1),
        false_alarm_rate=(tx & ~intr).sum() / max(n_non, 1),
        correct_label_rate=(tx & (d["pred_label"] == d["true_class"])).sum() / n_tx if n_tx else float("nan"),
        raw_trigger_reduction=1 - n_tx / max(pir_events, 1),
        lora_delivery_rate=n_deliv / n_tx if n_tx else float("nan"),
        energy_per_alert_j=act_e / n_deliv if n_deliv else float("nan"),
        latency_mean_s=float(d.loc[tx, "latency_s"].mean()) if n_tx else float("nan"),
        battery_min_frac=float(batt.min()) / cap, battery_end_frac=float(batt[-1]) / cap,
        events_at_empty_battery=int(d["battery_empty"].sum()),
    )
    return (m, d) if return_events else m


if __name__ == "__main__":
    from netra_simulator import SimConfig, simulate
    res = simulate(SimConfig(scenario="fusion_79", platform="pi_zero", seed=42))
    for c in (BaselineConfig(method="pir_only"), BaselineConfig(method="binary"),
              BaselineConfig(tau_c=0.65), BaselineConfig(tau_c=0.45)):
        m = evaluate(res, c)
        print(f"{m['method']:<22} act={m['camera_activations']:3d} true={m['true_activations']:3d} "
              f"false={m['false_activations']:3d} det={m['fusion_detection_rate']:.1%} "
              f"camE={m['camera_energy_wh']:.3f} Wh alerts={m['alerts_transmitted']} "
              f"deliv={m['lora_delivery_rate']:.0%} lat={m['latency_mean_s']:.2f}s")
