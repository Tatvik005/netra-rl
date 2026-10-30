# NETRA-RL: Networked Edge Threat Recognition & Alert

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-compatible-green.svg)](https://gymnasium.farama.org/)
[![Stable--Baselines3](https://img.shields.io/badge/SB3-compatible-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

**Week 2 Deliverable: From a Random Loop to a Proper Reinforcement Learning Environment**  
*Project NETRA: Edge-AI and Reinforcement Learning for Wildlife & Intrusion Detection under Energy Constraints*  
*Supervised by: Dr. Pravin Kumar (UPES Dehradun)*  
*Team: 4 BTech Students (4th Year)*

---

## 📖 Overview

NETRA-RL bridges edge computing, sensor fusion, and reinforcement learning for wildlife monitoring and boundary intrusion detection. Low-power field sensor nodes combine Passive Infrared (PIR) sensing and ultrasonic distance telemetry to detect intrusion events while optimizing battery lifetime and wireless LoRa communication costs.

Week 2 transitions the system from Week 1's stateless bandit formulation into a formal **Markov Decision Process (MDP)** with:
1. A **Structural Causal Model (SCM)** data generator removing arbitrary label assignments.
2. An exact **Sensor Fusion** metric ($P_{\text{dist}}$ and fused score $S$) with an asymmetric **Confusion-Matrix Reward** and LoRa transmission penalty.
3. A **Threshold Sweep ($\tau \in [0.30, 0.80]$)** benchmarking detection recall, false alarm rates, and expected reward to locate the optimal fixed threshold $\tau^*$.
4. An official **Gymnasium MDP Environment (`NetraEnv`)** integrating real-time battery discharge dynamics that passes strict `gymnasium` and `stable-baselines3` checkers.
5. Standard **Baseline Policies** to establish the benchmark that subsequent RL models (DQN) must surpass in Week 3.

---

## 📂 Repository Structure

```text
netra-rl/
├── data/
│   ├── simulate_sensors_v2.py       # Task A: Structural Causal Model data generator
│   ├── sensor_v2_seed42.csv         # Benchmark dataset (5,000 samples, seed 42)
│   ├── sensor_v2_seed100.csv        # Benchmark dataset (5,000 samples, seed 100)
│   ├── sensor_v2_seed287.csv        # Benchmark dataset (5,000 samples, seed 287)
│   ├── sensor_v2_seed523.csv        # Benchmark dataset (5,000 samples, seed 523)
│   ├── sensor_v2_seed1024.csv       # Benchmark dataset (5,000 samples, seed 1024)
│   └── causal_graph.png             # Directed Acyclic Graph (DAG) for the SCM
├── env/
│   ├── __init__.py                  # Exports NetraEnv
│   ├── fusion_reward.py             # Task B: Sensor fusion math, alert logic & reward matrix
│   └── netra_env.py                 # Task C: Gymnasium MDP environment with battery state
├── agent/
│   ├── __init__.py
│   ├── threshold_sweep.py           # Task B: Multi-seed threshold sweep & Pareto evaluation
│   └── baselines.py                 # Task D: Evaluates Random, Fixed tau, and Extreme baselines
├── tests/
│   ├── __init__.py
│   └── test_reward.py               # Unit tests verifying fusion, reward matrix & edge cases
├── results/
│   ├── causal_graph.png             # Causal DAG visualization (300 DPI)
│   ├── task_a_sanity_seed42.csv     # Task A sanity check verification table
│   ├── threshold_sweep.csv          # Task B summary table across 5 seeds
│   ├── threshold_sweep.png          # Task B publication plot: Recall, FAR, Reward vs tau
│   ├── baseline_metrics.csv         # Task D baseline evaluation table (100 episodes x 5 seeds)
│   └── baseline_comparison.png      # Task D comparative bar chart with error bars
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation and reproduction guide
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone git@github.com:Tatvik005/netra-rl.git
   cd netra-rl
   ```

2. **Create and activate a virtual environment (Python 3.10+):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Reproduce Every Result (One Command Each)

### 1. Task A — Generate Synthetic Causal Datasets & SCM DAG
Generates 5,000 rows across all 5 benchmark seeds (`42, 100, 287, 523, 1024`), executes theoretical sanity checks for Seed 42, and renders the SCM graph.
```bash
python data/simulate_sensors_v2.py
```
* **Outputs**: `data/sensor_v2_seed*.csv`, `results/task_a_sanity_seed42.csv`, `results/causal_graph.png`.

---

### 2. Task B — Run Threshold Sweep ($\tau \in [0.30, 0.80]$)
Evaluates detection rate (recall), false alarm rate (FAR), and mean reward per step across all 5 seeds, determining the optimal fixed threshold $\tau^*$:
```bash
python agent/threshold_sweep.py
```
* **Outputs**: `results/threshold_sweep.csv`, `results/threshold_sweep.png`.
* **Key Finding**: $\tau^* = 0.30$ maximizes mean reward ($+0.974 \pm 0.057$) with $99.8\%$ detection recall and $9.7\%$ FAR, outperforming NETRA's conservative threshold $\tau_c = 0.65$ ($+0.858 \pm 0.071$, $87.1\%$ recall) and $\tau = 0.60$ ($+0.891 \pm 0.080$).

---

### 3. Task C — Verify Gymnasium Environment & Run Unit Tests
Verifies that `NetraEnv` satisfies official Gymnasium and Stable-Baselines3 environment contracts and passes all unit tests:

* **Run Environment Checkers:**
  ```bash
  python -c "from gymnasium.utils.env_checker import check_env as gym_check; from stable_baselines3.common.env_checker import check_env as sb3_check; from env.netra_env import NetraEnv; env = NetraEnv('data/sensor_v2_seed42.csv'); gym_check(env); sb3_check(env); print('✅ All checkers passed with 0 warnings!')"
  ```

* **Run Pytest Test Suite:**
  ```bash
  pytest tests/test_reward.py -v
  ```
* **Outputs**: Validates TP ($+9.0$), FP ($-6.0$), FN ($-20.0$), TN ($0.0$), distance clipping, and vectorized evaluation.

---

### 4. Task D — Evaluate Baseline Policies
Runs 100 episodes $\times$ 5 seeds across the 4 required baseline policies (`Random`, `Fixed tau = 0.65`, `Fixed tau* = 0.30`, `Extreme tau = 0.30 / 0.80`):
```bash
python agent/baselines.py
```
* **Outputs**: `results/baseline_metrics.csv`, `results/baseline_comparison.png`.

---

## 📐 Mathematical Formulation

### 1. Structural Causal Model (SCM)
Data is synthesized along the causal chain:
$$\text{Hour } (h) \longrightarrow \text{Event } (E) \longrightarrow \{\text{PIR}, \text{Distance } (d), \text{Label}\}$$

- **Hour ($h$):** $h \sim \text{Uniform}\{0, \dots, 23\}$. Diurnal split: Night ($h \ge 19$ or $h \le 5$), Day ($5 < h < 19$).
- **Event ($E \in \{\text{none}, \text{human}, \text{animal}\}$):**
  - Day: $P(\text{none})=0.85,\; P(\text{human})=0.10,\; P(\text{animal})=0.05$
  - Night: $P(\text{none})=0.80,\; P(\text{human})=0.05,\; P(\text{animal})=0.15$
- **PIR ($PIR \in \{0, 1\}$):** $P(1|\text{none}) = 0.10$ (thermal false alarm), $P(1|\text{human}) = 0.90$, $P(1|\text{animal}) = 0.85$.
- **Distance ($d$):**
  - If $E = \text{none}$: $d \sim \text{Uniform}(3.5, 5.0)$
  - If $E \in \{\text{human}, \text{animal}\}$: $d \sim \mathcal{N}(1.5, 0.6^2)$
  - Sensor noise: $\epsilon \sim \mathcal{N}(0, 0.05^2)$, clipped to $[0.5, 5.0]\,\text{m}$.
- **Ground Truth Label:** $\text{Label} = \mathbb{I}(E \ne \text{none})$.

### 2. Sensor Fusion & Alert Logic
- **Proximity Score ($P_{\text{dist}}$):**
  $$P_{\text{dist}} = \text{clip}\left(\frac{5.0 - d}{5.0 - 0.5}, 0, 1\right)$$
- **Fused Threat Score ($S$):**
  $$S = 0.4 \cdot \text{PIR} + 0.6 \cdot P_{\text{dist}}$$
- **Alert Decision:**
  $$\text{alert} = \begin{cases} 1 & \text{if } S \ge \tau \\ 0 & \text{otherwise} \end{cases}$$

### 3. Confusion Matrix Reward with LoRa Energy Cost
Transmission over LoRa incurs an energy cost penalty of $-1.0$:

| Outcome | Label | Alert | Base Reward | LoRa Cost | Net Reward |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **True Positive (TP)** | 1 | 1 | $+10.0$ | $-1.0$ | **$+9.0$** |
| **False Positive (FP)** | 0 | 1 | $-5.0$ | $-1.0$ | **$-6.0$** |
| **False Negative (FN)** | 1 | 0 | $-20.0$ | $0.0$ | **$-20.0$** |
| **True Negative (TN)** | 0 | 0 | $0.0$ | $0.0$ | **$0.0$** |

### 4. Markov Decision Process (MDP) Transition
* **Why Week 1 was a Contextual Bandit:** In Week 1, rows were sampled independently and the agent's threshold decision did not alter future states ($S_{t+1}$ was independent of $A_t$).
* **Why Week 2 is a True MDP:** Battery state $B_t \in [0.0, 1.0]$ links actions across time:
  $$B_{t+1} = B_t - 0.001 - 0.01 \cdot \text{alert}_t$$
  Triggering alerts drains battery life faster. If $B_{t+1} \le 0$, the episode terminates with a heavy terminal penalty of $-50.0$. The agent must balance detection precision against battery longevity.

---

## 🛠️ Verification & Quality Assurance

- **Gymnasium Standard:** Passes `gymnasium.utils.env_checker.check_env` with 0 warnings.
- **Stable-Baselines3 Ready:** Passes `sb3.common.env_checker.check_env` for seamless DQN training in Week 3.
- **Reproducibility:** All seeds (`42, 100, 287, 523, 1024`) use NumPy's decoupled Generator (`np.random.default_rng(seed)` and `self.np_random`).
