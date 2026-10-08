import os
import random
import pandas as pd

# Load sensor_data.csv
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'data', 'sensor_data.csv')

df = pd.read_csv(data_path)

actions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

def state_aware_policy(state):
    """Inspects sensor state to pick optimal action."""
    pir, dist, hour = state
    if pir == 1 and (dist < 2.5 or hour < 6 or hour >= 22):
        return 0.3  # Sensitive threshold
    else:
        return 0.7  # Relaxed threshold

def compute_reward(label, threshold):
    is_low_threshold = threshold <= 0.5
    if label == 1 and is_low_threshold:
        return 10
    elif label == 0 and is_low_threshold:
        return -5
    elif label == 1 and not is_low_threshold:
        return -20
    else:
        return 0


print(f"{'Step':<5}  {'State (PIR, Dist, Hour)':<30}  {'Label':<6}  {'Random (Act/Rew)':<18}  {'State-Aware (Act/Rew)':<22}")


random_total = 0
smart_total = 0

for step in range(1, 21):
    sample = df.sample(n=1, random_state=step * 100).iloc[0]
    
    pir = int(sample['pir_reading'])
    dist = float(sample['ultrasonic_distance'])
    hour = int(sample['time_of_day'])
    label = int(sample['label'])
    
    # State has NO label
    state = (pir, dist, hour)
    
    # 1. Random Agent Action & Reward
    random_act = random.choice(actions)
    random_rew = compute_reward(label, random_act)
    random_total += random_rew
    
    # 2. State-Aware Agent Action & Reward
    smart_act = state_aware_policy(state)
    smart_rew = compute_reward(label, smart_act)
    smart_total += smart_rew
    
    state_str = f"PIR:{pir}, Dist:{dist:.2f}m, Hr:{hour:02d}"
    rand_str = f"{random_act} -> {random_rew:+3d}"
    smart_str = f"{smart_act} -> {smart_rew:+3d}"
    
    print(f"{step:<5}  {state_str:<30}  {label:<6}  {rand_str:<18}  {smart_str:<22}")



print(f"  * Random Agent (Blind Guess):        {random_total:+d} points")
print(f"  * State-Aware Agent (Uses State):     {smart_total:+d} points")
print(f"  * Score Difference:                   {smart_total - random_total:+d} points improvement!")

