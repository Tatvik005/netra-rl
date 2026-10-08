# NETRA simulator + fixed-rule baseline — notes

Sources of truth: arXiv:2605.08246v1 ("Smart Railway Obstruction Detection System using IoT and
Computer Vision", Kumar et al., 7 May 2026) and `PhD.pdf`. Full parameter provenance is in the
docstring of `data/netra_simulator.py`; this file is the short version.

## Files
| File | Purpose |
|---|---|
| `data/netra_simulator.py` | Seeded per-event generator (numpy + pandas only). `simulate(SimConfig(...))` → `SimResult` (events DataFrame + fine power grid). CLI writes `data/netra_sim_events.csv`. |
| `data/netra_sim_events.csv` | Seed 42, 3 scenarios × 2 platforms (578 rows). |
| `agent/fixed_rule_baseline.py` | Paper Algorithm 1 (fusion/camera) + Algorithm 2 (classification, priority, LoRa) + metrics via `evaluate()`. |
| `agent/run_baseline_eval.py` | Monte-Carlo evaluation (default 200 seeds), validation vs paper, plot. |
| `results/baseline_metrics.csv` | Mean/std of every metric per scenario × platform × method. |
| `results/baseline_validation.csv` | Simulator vs paper numbers. |
| `results/lora_sf_table.csv` | SYNTHETIC link model: airtime, TX energy, delivery vs SF/distance. |
| `results/baseline_eval.png`, `results/baseline_eval_stdout.txt` | Plot and console output of the reference run. |

## How to run (from `D:\netra me`)
```
netra_env\Scripts\python.exe data\netra_simulator.py            # regenerate CSV
netra_env\Scripts\python.exe agent\fixed_rule_baseline.py       # quick single-seed demo
netra_env\Scripts\python.exe agent\run_baseline_eval.py         # 200 seeds, ~1 min
netra_env\Scripts\python.exe agent\run_baseline_eval.py --seeds 50 --no-plot
```

## Provenance summary
| Quantity | Value | Source |
|---|---|---|
| Fusion rule | P = 0.4·PIR + 0.6·min(Δd/1.5, 1); camera if P ≥ 0.65 (0.45 variant) | PAPER eq. 3–5, Table V |
| Rejection | PIR=0, Δd ≤ 0, D < 4 m, D > 15 m | PAPER Alg. 1 |
| Background | D_bg ~ U(12, 15) m, mean of 5 readings | PAPER §III-A.2 |
| Ultrasonic noise | sd 0.03 m (paper spec ±3 mm) | SYNTHETIC |
| Intrusion Δd modes | clear 85% (Δd ≥ 1.2 m), partial 10% (0.65–0.95 m), miss 5% | CALIBRATED to Tables III–V |
| False-trigger mix | vegetation 50% (10% clutter Δd 0.15–0.55 m), bird 25% (30% too close), vehicle 25% (50% beyond bg) | CALIBRATED to Table V (2 false @0.45) |
| Binary fusion U_t | Δd ≥ 1.0 m | ASSUMED (not defined in paper) |
| Vision confusion | Pi 4 exact from Table VII; Pi Zero from Table VI + 28/26 elephant–cow errors (22 residual errors allocated consistently) | DERIVED |
| Confidence | correct Beta(0.82, sd 0.07), wrong Beta(0.62, 0.10), spurious Beta(0.35, 0.10) | CALIBRATED (Table X 39% > 0.8) |
| Alg. 2 | τ_AI 0.5, τ_alert 0.6, priorities 0.7/0.6/0.5, λ = 0.5 | PAPER (λ ASSUMED) |
| Power | idle 0.5/2.7 W, inference 2.5/7.5 W, 5.2/0.8 s, camera 2 W × 5 s, panels 10/25 W | PAPER Table II, §IV-C |
| Battery | 40/100 Wh, 80% start, derate 0.6 × canopy 0.6, weather factors | SYNTHETIC (canopy tuned so Pi 4 ≈ 14-day autonomy) |
| LoRa | Semtech airtime (20 B, BW125, CR4/5, 8 preamble, CRC, LDRO SF11–12); sensitivity −123…−136 dBm; 120 mA @ +20 dBm, 3.3 V; RX 11.5 mA | DATASHEET |
| Link / RSSI | log-distance n = 3.5, 1 km, shadowing 4 dB, fast fading 3 dB, weather loss | SYNTHETIC |
| Baseline SF | fixed SF9, +20 dBm, 3 attempts with ACK | ASSUMED |
| Latency | fusion 0.18 s (PAPER) + camera 1.0 s + inference + 0.2 s + LoRa | partly SYNTHETIC |

## Validation (200 seeds, Pi Zero unless noted)
| Check | Paper | Simulated (mean ± sd) | OK |
|---|---|---|---|
| PIR-only activations / false | 79 / 39 | 79 / 39 (by construction) | ✓ |
| Binary fusion detections (rate) | 34 (85%) | 34.2 ± 2.2 (85.6%) | ✓ |
| Prob. τ=0.65 detection rate | 95% | 95.1 ± 3.3% | ✓ |
| Prob. τ=0.65 activations / false | 38 / 0 | 38.0 ± 1.3 / 0 | ✓ |
| Prob. τ=0.45 activations / false | 40 / 2 | 40.1 ± 1.9 / 2.0 ± 1.4 | ✓ |
| Camera energy PIR-only / τ=0.65 | 0.220 / 0.106 Wh | 0.219 / 0.106 Wh | ✓ |
| Camera energy saving | 51.9% | 51.9 ± 1.7% | ✓ |
| Latency Pi Zero / Pi 4 | 6.5 / 2.4 s | 6.90 / 2.50 s | ≈ (+0.4 / +0.1 s) |
| LoRa delivery at 1 km (SF9) | 100% | 100% | ✓ |
| Vision F1 / accuracy (both platforms) | Tables VI–VII | within 0.3 pt | ✓ |
| Field-97 events with conf > 0.8 | 39.2% | 38.7 ± 4.2% | ✓ (tuned) |
| Field-97 false-alarm rate | 0% | 0% | ✓ |
| 113 events: triggers removed by fusion | 62.8% | 83.3% | ✗ |
| 113 events: confirmed alerts | 10 | 17.3 | ✗ |
| 113 events: raw-trigger reduction | 91.2% | 84.7% | ✗ |
| Field-97 alerts transmitted (incl. cattle) | 77 | 61.1 | ✗ |
| 20-intrusion field detection Pi Zero | 70% (14/20) | 86.5% | ✗ |
| 20-intrusion field detection Pi 4 | 95% (19/20) | 88.2% | ≈ (within 10 pt) |

### Why the ✗ rows do not match (not tuned away on purpose)
* **113-event numbers are inconsistent with Tables III–V.** Removing only 62.8% (71/113) of triggers
  means 42 events reached the camera; with 20 intrusions that implies ≥ 22 false activations,
  contradicting "0 false triggers". 10 confirmed alerts out of 20 intrusions contradicts 95% detection.
  Matching both is impossible with one model.
* **Field-97 (77/77 alerts)** implies a perfect pipeline, but the Pi Zero's own Table VI
  (elephant recall 12%, cow 38%) and the eq. 7 rule (cattle = Medium, logged only) make that
  unreachable. The paper does not say which platform the field test used (Fig. 4 shows a Pi Zero W).
* **Pi Zero 14/20** is lower than what fusion (95%) × human recall (94%) × confidence gating gives.
  It is not reproducible from the other Pi Zero tables.
* **Latency**: component sum is 0.1–0.4 s above the paper's figures; camera warm-up (1.0 s)
  is a guess.

## Observations useful for the DQN
* Per-alert action energy is ~22 J (Pi Zero) / ~15 J (Pi 4), dominated by camera (10 J) and
  inference (10.4 / 3.8 J). A LoRa packet costs 22–522 mJ (SF7–SF12). Idle power costs
  12 Wh/day (Pi Zero) and 65 Wh/day (Pi 4). Per-event choices barely move the battery unless
  the policy can also put the node to sleep or duty-cycle it.
* With the synthetic link model every SF delivers ~100% at 1 km. SF choice only matters beyond
  ~1.5 km (SF7 is about 55% per attempt at 2 km), so use `SimConfig(link_distance_m=(1500, 3000))`
  for RL scenarios where SF selection has a trade-off.
* The Pi 4 battery hits empty within 7 days in about 8% of seeds (rainy weeks). Use
  `BaselineConfig(brownout=True)` to make the node inactive while empty.
* The paper's baseline already has 0 false alarms, so the headroom is in energy, SF adaptation,
  robustness (animals, night, long links) and missed-intrusion recovery, not in the false-alarm rate.
* `pred_label`/`pred_conf` are counterfactual: an agent should only see them after choosing
  to turn the camera on.

## Known gaps / next steps
* No real sensor traces: the 113-event log is not available, so every distribution is fit to
  aggregate table numbers.
* `SimConfig(night_conf_penalty=...)` and `temperature_drift=True` exist but are off by default.
* Next: wrap `simulate()` in `env/netra_gym_env.py` (gymnasium) with the 5-D state
  [PIR, P_dist or Δd, vision conf (after camera), battery fraction, RSSI], and actions
  camera × transmit/wait × SF.
