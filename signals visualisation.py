# Visualisation of the initial signals

import pandas as pd
import matplotlib.pyplot as plt
import glob

# Load all CSV files
file_names = [f for f in glob.glob('signal_*.csv') if '_imfs' not in f]

# Prepare the plot
fig, axes = plt.subplots(3, 2, figsize=(16, 10))  # 3 rows, 2 columns
axes = axes.flatten()  # Flatten the axes for easy iteration

for i, file_name in enumerate(file_names):
    # Load the CSV file
    df = pd.read_csv(file_name)

    # Extract time and amplitude
    time = df['Time (s)']
    amplitude = df['Amplitude']

    # Plot each signal
    axes[i].plot(time, amplitude, label=f"Signal {i+1}", color="black")
    axes[i].set_title(f"Signal {i+1}", fontsize=14)
    axes[i].set_xlabel("Time (s)", fontsize=12)
    axes[i].set_ylabel("Amplitude", fontsize=12)
    axes[i].grid(True)
    axes[i].legend()

# Remove unused subplots (if any)
for j in range(len(file_names), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig("all_signals.png", dpi=600, bbox_inches="tight")
plt.show()





