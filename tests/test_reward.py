"""
tests/test_reward.py: Unit tests for NETRA-RL fusion, confusion-matrix reward, and LoRa cost logic.

Verifies:
1. TP (label=1, alert=1) -> +9.0 (+10 base - 1 LoRa cost), outcome="TP"
2. FP (label=0, alert=1) -> -6.0 (-5 base - 1 LoRa cost), outcome="FP"
3. FN (label=1, alert=0) -> -20.0 (no LoRa cost), outcome="FN"
4. TN (label=0, alert=0) -> 0.0 (no LoRa cost), outcome="TN"
5. compute_p_dist boundary checks (d=0.5, 5.0, 2.75, and clipping outside [0.5, 5.0])
6. evaluate_sensor_step end-to-end check on intruder reading (pir=1, d=1.4, label=1, tau=0.65)
"""

import pytest
import numpy as np

from env.fusion_reward import (
    compute_p_dist,
    compute_fusion_score,
    compute_alert,
    compute_reward,
    evaluate_sensor_step,
    REWARD_TP,
    REWARD_FP,
    REWARD_FN,
    REWARD_TN,
    LORA_ALERT_COST,
)


def test_true_positive_reward():
    """1. TP (label=1, alert=1) returns +9.0 (+10 base - 1 LoRa alert cost) and outcome 'TP'."""
    reward, outcome = compute_reward(label=1, alert=1)
    assert reward == 9.0
    assert outcome == "TP"
    assert reward == REWARD_TP
    assert reward == (10.0 - LORA_ALERT_COST)


def test_false_positive_reward():
    """2. FP (label=0, alert=1) returns -6.0 (-5 base - 1 LoRa alert cost) and outcome 'FP'."""
    reward, outcome = compute_reward(label=0, alert=1)
    assert reward == -6.0
    assert outcome == "FP"
    assert reward == REWARD_FP
    assert reward == (-5.0 - LORA_ALERT_COST)


def test_false_negative_reward():
    """3. FN (label=1, alert=0) returns -20.0 and outcome 'FN' (no LoRa alert sent)."""
    reward, outcome = compute_reward(label=1, alert=0)
    assert reward == -20.0
    assert outcome == "FN"
    assert reward == REWARD_FN


def test_true_negative_reward():
    """4. TN (label=0, alert=0) returns 0.0 and outcome 'TN' (no LoRa alert sent)."""
    reward, outcome = compute_reward(label=0, alert=0)
    assert reward == 0.0
    assert outcome == "TN"
    assert reward == REWARD_TN


def test_compute_p_dist_boundaries_and_clipping():
    """5. compute_p_dist boundary checks: d=0.5 -> 1.0, d=5.0 -> 0.0, d=2.75 -> 0.5, and clipping."""
    # Exact boundary checks
    assert pytest.approx(compute_p_dist(0.5), rel=1e-6) == 1.0
    assert pytest.approx(compute_p_dist(5.0), rel=1e-6) == 0.0
    assert pytest.approx(compute_p_dist(2.75), rel=1e-6) == 0.5

    # Linear proportionality check at 1.4m: (5.0 - 1.4) / 4.5 = 3.6 / 4.5 = 0.8
    assert pytest.approx(compute_p_dist(1.4), rel=1e-6) == 0.8

    # Clipping below minimum distance (d < 0.5m)
    assert compute_p_dist(0.0) == 1.0
    assert compute_p_dist(-1.5) == 1.0

    # Clipping above maximum distance (d > 5.0m)
    assert compute_p_dist(5.5) == 0.0
    assert compute_p_dist(10.0) == 0.0


def test_evaluate_sensor_step_end_to_end():
    """6. evaluate_sensor_step end-to-end check on intruder reading:
    pir=1, distance=1.4, label=1, tau=0.65 -> p_dist=0.8, S=0.88, alert=1, reward=9.0, outcome='TP'.
    """
    res = evaluate_sensor_step(pir=1, distance=1.4, label=1, tau=0.65)

    assert pytest.approx(res["p_dist"], rel=1e-6) == 0.8
    assert pytest.approx(res["score"], rel=1e-6) == 0.88
    assert res["alert"] == 1
    assert res["reward"] == 9.0
    assert res["outcome"] == "TP"


def test_array_vectorized_processing():
    """Vectorized NumPy array inputs for batch and offline dataset evaluation."""
    labels = np.array([1, 0, 1, 0])
    alerts = np.array([1, 1, 0, 0])

    rewards, outcomes = compute_reward(labels, alerts)
    np.testing.assert_array_equal(rewards, np.array([9.0, -6.0, -20.0, 0.0]))
    np.testing.assert_array_equal(outcomes, np.array(["TP", "FP", "FN", "TN"]))

    distances = np.array([0.5, 1.4, 2.75, 5.0, 6.0, 0.1])
    expected_p = np.array([1.0, 0.8, 0.5, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(compute_p_dist(distances), expected_p, rtol=1e-5)

    pirs = np.array([1, 1, 0, 0])
    p_dists = np.array([0.8, 0.2, 0.8, 0.2])
    # 0.4*1 + 0.6*0.8 = 0.88
    # 0.4*1 + 0.6*0.2 = 0.52
    # 0.4*0 + 0.6*0.8 = 0.48
    # 0.4*0 + 0.6*0.2 = 0.12
    expected_s = np.array([0.88, 0.52, 0.48, 0.12])
    np.testing.assert_allclose(compute_fusion_score(pirs, p_dists), expected_s, rtol=1e-5)

    alerts_computed = compute_alert(expected_s, tau=0.50)
    np.testing.assert_array_equal(alerts_computed, np.array([1, 1, 0, 0]))


def test_invalid_input_error():
    """Ensure invalid non-binary label/alert raises ValueError."""
    with pytest.raises(ValueError):
        compute_reward(label=2, alert=1)
    with pytest.raises(ValueError):
        compute_reward(label=1, alert=3)
