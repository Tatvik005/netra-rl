import os
import pandas as pd

# Load sensor_data.csv
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'data', 'sensor_data.csv')

df = pd.read_csv(data_path)
print(f"Loaded sensor dataset with {len(df)} records.\n")

# Action Space: Available sensitivity thresholds
actions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

def state_aware_policy(state):
    """
    Intelligent policy that inspects the sensors to choose the optimal threshold:
    - High sensitivity (0.3) when:
        * Motion is detected (pir == 1) AND an obstacle is in the danger zone (< 2.5m)
        * OR motion is detected at night (hour < 6 or hour >= 22)
    - Low sensitivity / conservative (0.7) when:
        * Track is clear or motion is far away (> 2.5m in daytime)
    """
    pir, dist, hour = state
    
    # State-driven decision rule
    if pir == 1 and (dist < 2.5 or hour < 6 or hour >= 22):
        return 0.3  # Sensitive threshold (catch threat)
    else:
        return 0.7  # Relaxed threshold (conserve energy, prevent false alarms)

def compute_reward(label, threshold):
    """
    Environment Reward Function (uses hidden ground-truth label):
    - Intrusion (1) + low threshold (<= 0.5)  -> +10 (Caught)
    - Safe (0) + low threshold (<= 0.5)       -> -5  (False Alarm)
    - Intrusion (1) + high threshold (> 0.5)  -> -20 (Missed Threat)
    - Safe (0) + high threshold (> 0.5)       -> 0   (Quiet & Normal)
    """
    is_low_threshold = threshold <= 0.5
    
    if label == 1 and is_low_threshold:
        return 10
    elif label == 0 and is_low_threshold:
        return -5
    elif label == 1 and not is_low_threshold:
        return -20
    else:
        return 0

print("=" * 80)
print("STATE-AWARE AGENT (Using State without Label to make decisions)")
print("=" * 80)
print(f"{'Step':<6} | {'State (PIR, Distance, Hour)':<35} | {'Action':<10} | {'Hidden Label':<14} | {'Reward':<8}")
print("=" * 80)

total_reward = 0

for step in range(1, 21):
    # Sample a row from the environment
    sample = df.sample(n=1, random_state=step * 42).iloc[0]
    
    pir = int(sample['pir_reading'])
    dist = float(sample['ultrasonic_distance'])
    hour = int(sample['time_of_day'])
    hidden_label = int(sample['label'])  # Environment keeps this secret!
    
    # NOTICE: 'state' only contains sensor readings, NO label!
    state = (pir, dist, hour)
    
    # The agent uses 'state' to pick its action:
    action = state_aware_policy(state)
    
    # Environment evaluates the action against hidden ground truth
    reward = compute_reward(hidden_label, action)
    total_reward += reward
    
    state_str = f"PIR:{pir}, Dist:{dist:.2f}m, Hour:{hour:02d}"
    print(f"{step:<6} | {state_str:<35} | {action:<10} | {hidden_label:<14} | {reward:<+8}")

print("=" * 80)
print(f"Total Cumulative Reward: {total_reward:+d} points")
print("=" * 80)
