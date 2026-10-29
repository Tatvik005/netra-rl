"""
env/fusion_reward.py: Core sensor fusion and confusion-matrix reward logic for NETRA-RL.

Week 2 - Task B:
Implements:
1. compute_p_dist: Distance proximity scoring with linear clipping.
2. compute_fusion_score: Weighted fusion of PIR binary detection and ultrasonic proximity.
3. compute_alert: Threshold-based alert trigger.
4. compute_reward: Confusion-matrix-driven reward including LoRa communication energy cost.
5. evaluate_sensor_step: End-to-end evaluation helper for single-step or environment interaction.
"""

from typing import Any, Dict, Tuple, Union
import numpy as np

# Proximity normalization bounds (meters)
DEFAULT_D_MIN: float = 0.5
DEFAULT_D_MAX: float = 5.0

# Sensor fusion weights (sum to 1.0)
PIR_WEIGHT: float = 0.4
DIST_WEIGHT: float = 0.6

# Base confusion matrix rewards
REWARD_TP_BASE: float = 10.0
REWARD_FP_BASE: float = -5.0
REWARD_FN_BASE: float = -20.0
REWARD_TN_BASE: float = 0.0

# LoRa alert transmission energy penalty (incurred whenever alert == 1)
LORA_ALERT_COST: float = 1.0

# Net rewards
REWARD_TP: float = REWARD_TP_BASE - LORA_ALERT_COST   # +9.0
REWARD_FP: float = REWARD_FP_BASE - LORA_ALERT_COST   # -6.0
REWARD_FN: float = REWARD_FN_BASE                      # -20.0
REWARD_TN: float = REWARD_TN_BASE                      # 0.0


def compute_p_dist(
    d: Union[float, int, np.ndarray],
    d_min: float = DEFAULT_D_MIN,
    d_max: float = DEFAULT_D_MAX,
) -> Union[float, np.ndarray]:
    """
    Computes distance proximity score P_dist in [0.0, 1.0].
    
    Higher score indicates closer proximity (greater threat):
        - d <= d_min (0.5m) -> 1.0
        - d >= d_max (5.0m) -> 0.0
        - d in (d_min, d_max) -> linear interpolation: (d_max - d) / (d_max - d_min)

    Supports both single scalar values and NumPy arrays.
    """
    d_arr = np.asarray(d, dtype=float)
    p_dist = np.clip((d_max - d_arr) / (d_max - d_min), 0.0, 1.0)
    if np.ndim(d) == 0:
        return float(p_dist)
    return p_dist


def compute_fusion_score(
    pir: Union[int, float, np.ndarray],
    p_dist: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Computes fused sensor score S = 0.4 * pir + 0.6 * p_dist.

    Parameters:
        pir: Binary PIR detection state (0 or 1).
        p_dist: Distance proximity score in [0.0, 1.0].

    Returns:
        S: Fused score in [0.0, 1.0]. Supports scalars and NumPy arrays.
    """
    pir_arr = np.asarray(pir, dtype=float)
    p_dist_arr = np.asarray(p_dist, dtype=float)
    score = PIR_WEIGHT * pir_arr + DIST_WEIGHT * p_dist_arr
    if np.ndim(pir) == 0 and np.ndim(p_dist) == 0:
        return float(score)
    return score


def compute_alert(
    score: Union[float, np.ndarray],
    tau: Union[float, np.ndarray],
) -> Union[int, np.ndarray]:
    """
    Determines alert decision based on whether score S meets or exceeds threshold tau.

    Formula:
        alert = 1 if S >= tau else 0

    Supports both single scalar values and NumPy arrays.
    """
    s_arr = np.asarray(score, dtype=float)
    tau_arr = np.asarray(tau, dtype=float)
    alert = (s_arr >= tau_arr).astype(int)
    if np.ndim(score) == 0 and np.ndim(tau) == 0:
        return int(alert)
    return alert


def compute_reward(
    label: Union[int, np.ndarray],
    alert: Union[int, np.ndarray],
) -> Tuple[Union[float, np.ndarray], Union[str, np.ndarray]]:
    """
    Computes confusion-matrix-based reward and outcome string.

    Outcomes:
        - True Positive  (label == 1, alert == 1):
            Reward = +10.0 (base) - 1.0 (LoRa) = +9.0, Outcome = "TP"
        - False Positive (label == 0, alert == 1):
            Reward = -5.0 (base) - 1.0 (LoRa)  = -6.0, Outcome = "FP"
        - False Negative (label == 1, alert == 0):
            Reward = -20.0 (base) - 0.0 (LoRa) = -20.0, Outcome = "FN"
        - True Negative  (label == 0, alert == 0):
            Reward = 0.0 (base) - 0.0 (LoRa)   = 0.0, Outcome = "TN"

    Returns:
        (reward, outcome_str):
            - If inputs are scalars: (float, str) e.g. (9.0, "TP")
            - If inputs are arrays: (np.ndarray of float, np.ndarray of str)
    """
    l_arr = np.asarray(label, dtype=int)
    a_arr = np.asarray(alert, dtype=int)

    # Scalar branch
    if l_arr.ndim == 0 and a_arr.ndim == 0:
        l_val = int(l_arr)
        a_val = int(a_arr)
        if l_val == 1 and a_val == 1:
            return REWARD_TP, "TP"
        elif l_val == 0 and a_val == 1:
            return REWARD_FP, "FP"
        elif l_val == 1 and a_val == 0:
            return REWARD_FN, "FN"
        elif l_val == 0 and a_val == 0:
            return REWARD_TN, "TN"
        else:
            raise ValueError(
                f"Invalid binary inputs: label={label}, alert={alert}. Must be 0 or 1."
            )

    # Array branch
    reward = np.zeros(l_arr.shape, dtype=float)
    outcome = np.empty(l_arr.shape, dtype=object)

    tp_mask = (l_arr == 1) & (a_arr == 1)
    fp_mask = (l_arr == 0) & (a_arr == 1)
    fn_mask = (l_arr == 1) & (a_arr == 0)
    tn_mask = (l_arr == 0) & (a_arr == 0)

    reward[tp_mask] = REWARD_TP
    outcome[tp_mask] = "TP"

    reward[fp_mask] = REWARD_FP
    outcome[fp_mask] = "FP"

    reward[fn_mask] = REWARD_FN
    outcome[fn_mask] = "FN"

    reward[tn_mask] = REWARD_TN
    outcome[tn_mask] = "TN"

    return reward, outcome


def evaluate_sensor_step(
    pir: Union[int, float],
    distance: Union[float, int],
    label: int,
    tau: float,
) -> Dict[str, Any]:
    """
    Convenience step evaluator for single-step sensor reading.

    Takes:
        pir: Binary PIR sensor reading (0 or 1)
        distance: Ultrasonic distance measurement in meters
        label: Ground-truth intrusion status (1 = human intruder, 0 = none/animal)
        tau: Decision threshold

    Returns dict:
        {
            "p_dist": float,
            "score": float,
            "alert": int,
            "reward": float,
            "outcome": str,
        }
    """
    p_dist = compute_p_dist(distance)
    score = compute_fusion_score(pir, p_dist)
    alert = compute_alert(score, tau)
    reward, outcome = compute_reward(label, alert)

    return {
        "p_dist": p_dist,
        "score": score,
        "alert": alert,
        "reward": reward,
        "outcome": outcome,
    }
