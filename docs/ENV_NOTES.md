# NetraEnv (gymnasium) — design notes

`env/netra_gym_env.py` wraps `data/netra_simulator.py` in a `gymnasium.Env`. The agent replaces
the paper's fixed rules (Algorithms 1 + 2, fixed SF). It decides camera use, alert transmission,
LoRa spreading factor, gateway, and a new sleep/duty-cycle action, following the PhD proposal (Table 2).
All provenance tags (PAPER / DERIVED / CALIBRATED / SYNTHETIC) are the same as in
`docs/SIMULATOR_NOTES.md`.

## Files
| File | Purpose |
|---|---|
| `env/netra_gym_env.py` | `NetraEnv`, `EnvConfig`, `ENV_PROFILES`, `make_env()` |
| `agent/env_policies.py` | `FixedRulePolicy` (paper baseline inside the env), `RandomPolicy`, `run_policy()`, `summarize()` |
| `agent/env_smoke_test.py` | `check_env` on 5 configurations, then random vs fixed-rule rollouts |
| `agent/dqn_sanity_run.py` | Few-thousand-step SB3 DQN run to confirm training works (not full training) |
| `results/env_smoke_test_day.csv`, `results/env_smoke_test_week.csv` | Reference smoke-test output (50 × 1 day, 10 × 7 days) |

## Time model
Decision epochs are event-driven:
* **Event epoch**: a PIR trigger while the node is awake. The ultrasonic reading is automatic, as in Alg. 1.
* **Heartbeat epoch**: every 30 minutes without events, and on every wake-up.

Each event has two sub-steps:
1. **Sense**: `camera=on` runs the camera and the vision model and presents the same event again
   with confidence and label. `camera=off` resolves the event right away.
2. **Classified**: tx, gateway and sleep take effect.

On heartbeats only the sleep choice matters. Sleeping power-gates the Pi, and every event during
sleep + boot is missed. Power is integrated on the simulator's 1-minute grid, with solar from the simulator.

## Observation (Box[-1, 1], 15-dim by default)
| idx | feature | notes |
|---|---|---|
| 0 | PIR | 0/1 |
| 1 | P_dist | min(Δd/1.5, 1); 0 if Alg. 1 would reject the reading |
| 2 | dist_valid | 4 ≤ D ≤ 15 m and Δd > 0 |
| 3 | Δd | clip(Δd / 5 m) |
| 4 | vision confidence | 0 until the camera has been used on this event |
| 5 | camera_used | stage flag |
| 6–9 | label one-hot | human, elephant, cow, obstacle (0 if background / camera not used) |
| 10 | battery SoC | |
| 11.. | RSSI per gateway | (RSSI + 140)/60, clipped to 0..1 |
| +3 | sin/cos hour, solar fraction | `time_features=True` |

**Deviation from the PhD's 5-value state:** all five values are present (PIR, distance,
YOLO confidence, battery, RSSI). I added:
* a validity flag and raw Δd, because Alg. 1 depends on the valid range;
* a camera-used flag and label one-hot, because confidence only exists after the camera runs
  and means nothing without a label;
* time/solar features, for battery planning;
* one RSSI per gateway.

## Action (Discrete 42 = camera 2 × tx 7 × sleep 3 × gateway 1)
`a = ((gw * n_sleep + sleep) * 7 + tx) * 2 + camera`. Use `env.encode_action()` / `env.decode_action()`.
* tx ∈ {wait, SF7 … SF12}. "Wait" means do not send; there is no deferred-send queue.
* sleep ∈ {0, 15, 60} min (`sleep_options_min`).
* gateway: 1 by default, because the paper has a single receiver (ESP32 on the train or a Pi Pico).
  `n_gateways=2` gives 84 actions and one RSSI per gateway, each with its own link distance.

## Reward (defaults; all weights in `EnvConfig`)
| term | value |
|---|---|
| intrusion with delivered alert | +10 |
| delivered alert for a non-intrusion | −5 |
| missed intrusion (not sent, not delivered, asleep, PIR=0, or remaining after brownout) | −8 |
| correct non-alert | 0 (`r_correct_reject`) |
| action energy | −2 × E / 20 J (camera + inference + LoRa; `energy_tx_only` gives the literal PhD transmit-only term; `energy_include_boot` adds boot energy) |
| idle energy | −w_idle × E / 1 Wh, off by default |
| battery low | −1 × max(0, 0.2 − SoC)/0.2 per epoch |
| brownout | −50, and the episode terminates |

* Cattle count as intrusions by default (`cow_is_intrusion=True`), as Table X does.
  Paper eq. 7 only logs them, so the pure paper rule is penalised for cattle misses.
* **Lagrangian** mode: reward − λ·n_missed. λ is updated at each reset from the previous
  episode's miss rate: λ ← clip(λ + 5·(miss_rate − 0.05), 0, 100). This is a primal-dual CMDP
  heuristic, so the reward is non-stationary on purpose.
* **Lexicographic** mode: 10 × (miss + brownout terms) + 0.1 × (all other terms). This scale
  separation approximates "minimise misses first".
* `info["cost"]` holds the number of misses in each step, for external constrained-RL code.

## Config highlights
* `platform`: `pi_zero` (default) or `pi4`.
* `episode_days`: 1 (day, default) or 7 (week).
* `link_distance_m`: (1500, 3000), drawn per episode per gateway.
* `battery_init_range`: (0.5, 1.0).
* `event_profile`: `mixed_113` (default) or any simulator scenario.
* Sleep power 0.05 / 0.10 W and boot 25 s @ 1 W / 20 s @ 4 W assume an external power-gating
  controller. These are SYNTHETIC; the paper has no sleep mode.
* **Why `mixed_113` is the default:** it combines the paper's 113-event ratio (2.86 intrusions and
  13.3 false triggers per day) with the Table X class mix. With `field_97` rates (79% intrusions),
  "alert on every PIR trigger without the camera" is near-optimal: a 5 k-step DQN found exactly
  that (97% detection at 3 J/alert).

## Smoke-test results (box, 50 one-day episodes, `mixed_113`, link 1.5–3 km)
| platform | policy | return | detection | false alarms/ep | delivery | J/alert | battery end |
|---|---|---|---|---|---|---|---|
| Pi Zero | random | −28.2 | 0.230 | 2.70 | 0.869 | 229 | 0.958 |
| Pi Zero | fixed rule SF9 (paper) | −0.8 | 0.533 | 0 | 0.911 | 35.8 | 0.878 |
| Pi Zero | fixed rule SF9 + send cattle | 6.7 | 0.689 | 0 | 0.921 | 27.8 | 0.878 |
| Pi Zero | fixed rule SF12 + send cattle | 9.5 | 0.748 | 0 | 1.000 | 26.0 | 0.878 |
| Pi 4 | random | −24.0 | 0.274 | 2.58 | 0.892 | 698 | 0.921 |
| Pi 4 | fixed rule SF9 (paper) | 5.5 | 0.630 | 0 | 0.885 | 20.3 | 0.659 |
| Pi 4 | fixed rule SF12 + send cattle | 9.4 | 0.711 | 0 | 1.000 | 18.5 | 0.659 |

On 7-day episodes the fixed-rule Pi 4 browns out in 4 of 10 episodes (end SoC 0.34), because it
never sleeps. Random policies waste energy on boots because they sleep about 75% of the time.
J/alert counts boot energy.

## DQN sanity run (box, SB3 2.9.0, torch 2.14 CPU, net [128, 64], Pi Zero, 1-day episodes)
| steps | train return (1st → last quarter) | eval return | detection | FA/ep | sleep frac |
|---|---|---|---|---|---|
| 5 000 (8 s) | −34 → +3 | 10.3 | 0.695 | 0.30 | 0.25 |
| 30 000 (55 s) | −27 → −1 | 6.0 | 0.610 | 0.17 | 0.39 |

Reference on the same eval seeds: fixed rule −1.9 (detection 0.512), random −27.3.
The runs only show that the stack trains. They are not converged and not tuned. The DQN still
sleeps far too much for a Pi Zero, which has no energy deficit.

## Recommended next steps for full training
1. Write `agent/train_dqn.py`:
   * vectorised envs (`SubprocVecEnv`, 8–16 workers), 1–5 M steps;
   * `EvalCallback` on fixed seeds, and comparison against `FixedRulePolicy` with SF9 and SF12;
   * save models under `results/`.
2. Tune: lr 1e-4–5e-4, buffer 200 k–1 M, target update 1–5 k, exploration over 20–30% of steps,
   γ = 0.99 for 1-day and 0.995 for 7-day episodes. Try Double DQN / QR-DQN (sb3-contrib) and
   action masking (MaskablePPO), since many action parts are ignored depending on the stage.
3. Run curricula over platform (Pi Zero surplus vs Pi 4 deficit), episode length (week for
   battery learning), link distance and `battery_init_range`.
4. Research question: compare the standard, Lagrangian (sweep the miss-rate target) and
   lexicographic modes. Report miss rate against energy per alert.
5. Evaluate with ≥ 200 seeds and confidence intervals. Report detection, false-alarm rate,
   energy per alert, delivery, brownouts and SoC, against both fixed-rule variants.
6. For deployment, export the Q-network (PyTorch → ONNX → TFLite INT8) and check latency on a
   Pi Zero 2W.
