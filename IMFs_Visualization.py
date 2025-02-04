import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
file_path = "signal_1_imfs.csv"
data = pd.read_csv(file_path)

# Assuming the first column is time, and the rest are IMFs
time = data.iloc[:, 0]  # Extract time
imfs = data.iloc[:, 1:]  # Extract IMFs

# Define number of IMFs
num_imfs = imfs.shape[1]
num_cols = 4  # Keep 3 columns as per your request
num_rows = int(np.ceil(num_imfs / num_cols))

# Create figure
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 10), sharex=True)

# Flatten axes array for easy iteration
axes = axes.flatten()

# Plot each IMF
for i in range(num_imfs):
    axes[i].plot(time, imfs.iloc[:, i], label=f'IMF {i+1}', linewidth=1)
    axes[i].legend(fontsize=10, loc="upper right")  # Increased font size and positioned at the top right
    axes[i].grid()

# Hide empty subplots if any
for i in range(num_imfs, len(axes)):
    fig.delaxes(axes[i])

# Set common labels with larger font size
fig.text(0.5, 0.05, "Time", ha="center", fontsize=14)  # X-label closer to the plots
fig.suptitle("Intrinsic Mode Functions (IMFs) vs Time", fontsize=16, y=0.93)  # Title closer to plots

# Save figure as PNG with 600 DPI
output_path = "imfs_visualization_updated.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight')

# Show the plot
plt.show()

print(f"Updated figure saved at {output_path}")
