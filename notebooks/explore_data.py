import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load sensor_data.csv using pandas
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'data', 'sensor_data.csv')
results_dir = os.path.join(base_dir, 'results')

os.makedirs(results_dir, exist_ok=True)

df = pd.read_csv(data_path)
print(f"Successfully loaded sensor data from {data_path}. Shape: {df.shape}")

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 2. Plot PIR readings over time
plt.figure(figsize=(12, 4))
plt.plot(df.index, df['pir_reading'], color='teal', alpha=0.7, linewidth=1, label='PIR Motion (0/1)')
plt.title('PIR Motion Sensor Readings Over Time', fontsize=14)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('PIR Reading', fontsize=12)
plt.yticks([0, 1], ['No Motion (0)', 'Motion (1)'])
plt.legend(loc='upper right')
plt.tight_layout()
pir_plot_path = os.path.join(results_dir, 'pir_over_time.png')
plt.savefig(pir_plot_path, dpi=300)
plt.close()
print(f"Saved PIR plot to: {pir_plot_path}")

# 3. Plot ultrasonic distance over time
plt.figure(figsize=(12, 4))
plt.plot(df.index, df['ultrasonic_distance'], color='coral', alpha=0.8, linewidth=1, label='Distance (m)')
plt.axhline(y=2.5, color='red', linestyle='--', linewidth=1.5, label='Proximity Threshold (2.5m)')
plt.title('Ultrasonic Distance Sensor Readings Over Time', fontsize=14)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Distance (meters)', fontsize=12)
plt.legend(loc='upper right')
plt.tight_layout()
distance_plot_path = os.path.join(results_dir, 'ultrasonic_distance_over_time.png')
plt.savefig(distance_plot_path, dpi=300)
plt.close()
print(f"Saved Ultrasonic Distance plot to: {distance_plot_path}")

# 4. Show distribution of intrusion vs. safe events (bar chart)
plt.figure(figsize=(6, 5))
label_counts = df['label'].value_counts().sort_index()
bars = plt.bar(['Safe (0)', 'Intrusion (1)'], label_counts.values, color=['#2ecc71', '#e74c3c'], width=0.5)
plt.title('Distribution of Safe vs. Intrusion Events', fontsize=14)
plt.xlabel('Event Class', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Annotate count values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 5, int(yval), ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.ylim(0, max(label_counts.values) * 1.15)
plt.tight_layout()
dist_plot_path = os.path.join(results_dir, 'event_distribution.png')
plt.savefig(dist_plot_path, dpi=300)
plt.close()
print(f"Saved Event Distribution plot to: {dist_plot_path}")

# 5. Combined summary dashboard plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Subplot 1: PIR
axes[0].plot(df.index, df['pir_reading'], color='teal', alpha=0.7)
axes[0].set_title('PIR Motion Sensor Readings')
axes[0].set_yticks([0, 1])

# Subplot 2: Ultrasonic Distance
axes[1].plot(df.index, df['ultrasonic_distance'], color='coral', alpha=0.8)
axes[1].axhline(y=2.5, color='red', linestyle='--')
axes[1].set_title('Ultrasonic Distance (m)')

# Subplot 3: Bar chart of Events
bars_sub = axes[2].bar(['Safe (0)', 'Intrusion (1)'], label_counts.values, color=['#2ecc71', '#e74c3c'], width=0.4)
axes[2].set_title('Event Distribution (Safe vs Intrusion)')
for bar in bars_sub:
    yval = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2.0, yval + 5, int(yval), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
combined_plot_path = os.path.join(results_dir, 'eda_summary.png')
plt.savefig(combined_plot_path, dpi=300)
plt.close()
print(f"Saved Combined Dashboard plot to: {combined_plot_path}")

print("Exploratory Data Analysis completed successfully!")
