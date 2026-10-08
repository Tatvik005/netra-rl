import os
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

num_samples = 500

# 1. PIR reading — 0 (no motion) or 1 (motion detected), 30% probability of motion
pir_reading = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])

# 2. Ultrasonic distance — random float between 0.5 m and 5.0 m
ultrasonic_distance = np.round(np.random.uniform(0.5, 5.0, size=num_samples), 2)

# 3. Time of day — integer 0–23 representing hour
time_of_day = np.random.randint(0, 24, size=num_samples)

# 4. Label — 0 (safe) or 1 (intrusion), ground truth for reward calculation
# Intrusion logic: Motion detected (PIR == 1) and close proximity (< 2.5m) or nighttime motion (night hours: 22-5)
label = np.where(
    (pir_reading == 1) & ((ultrasonic_distance < 2.5) | (time_of_day < 6) | (time_of_day >= 22)),1,0)

# Create DataFrame
df = pd.DataFrame({
    'pir_reading': pir_reading,
    'ultrasonic_distance': ultrasonic_distance,
    'time_of_day': time_of_day,
    'label': label
})

# Save output CSV to data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path_underscore = os.path.join(script_dir, 'sensor_data.csv')
csv_path_space = os.path.join(script_dir, 'sensor data.csv')

df.to_csv(csv_path_underscore, index=False)
df.to_csv(csv_path_space, index=False)

print(f"Successfully generated {num_samples} rows of sensor data.")
print(f"Saved to: {csv_path_underscore}")
print(df.head())
