import os
import random
import pandas as pd


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'data', 'sensor_data.csv')

df = pd.read_csv(data_path)
print(f"Loaded sensor dataset with {len(df)} records.\n")

# sensitivity thresholds we going random here 
actions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

def compute_reward(label, threshold):
    """
    Calculate reward based on ground truth intrusion label and chosen threshold action:
    - Intrusion (1) + low threshold (<= 0.5)  -> +10
    - No intrusion (0) + low threshold (<= 0.5) -> -5
    - Intrusion (1) + high threshold (> 0.5) -> -20
    - No intrusion (0) + high threshold (> 0.5) -> 0
    """
    is_low_threshold = threshold <= 0.5
    
    if label == 1 and is_low_threshold:
        return 10
    elif label == 0 and is_low_threshold:
        return -5
    elif label == 1 and not is_low_threshold:
        return -20
    else:  # meaning ki label == 0 and high threshold
        return 0


print(f"{'Step':<6} | {'State (PIR, Distance, Hour, Label)':<40} | {'Action (Threshold)':<20} | {'Reward':<8}")


# Runnning basic RL loop for 20 steps here
for step in range(1, 21):
    # yaha par Sampling a row from sensor dataset 
    sample = df.sample(n=1).iloc[0]
    
    pir = int(sample['pir_reading'])
    dist = float(sample['ultrasonic_distance'])
    hour = int(sample['time_of_day'])
    label = int(sample['label'])
    
    state = (pir, dist, hour, label)
    
    # Randomly selecting action from action space
    action = random.choice(actions)
    
    # Compute reward
    reward = compute_reward(label, action)
    
    state_str = f"PIR:{pir}, Dist:{dist:.2f}m, Hour:{hour:02d}, Label:{label}"
    print(f"{step:<6} | {state_str:<40} | {action:<20} | {reward:<+8}")



