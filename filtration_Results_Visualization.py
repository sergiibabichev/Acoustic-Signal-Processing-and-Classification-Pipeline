import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend

def visualize_signals_with_detrended_noise(original_file, filtered_file, output_png, downsample_factor=1):
    """
    Visualize original signal, filtered signal, noise, and detrended noise in one window.
    Save the visualization as a PNG file.

    Parameters:
        original_file (str): Path to the CSV file containing the original signal.
        filtered_file (str): Path to the CSV file containing the filtered signal.
        output_png (str): Path to save the resulting PNG file.
        downsample_factor (int): Factor to downsample the data. Default is 1 (no downsampling).
    """
    # Load the data
    original_data = pd.read_csv(original_file)
    filtered_data = pd.read_csv(filtered_file)

    # Ensure both files have the same structure and length
    assert len(original_data) == len(filtered_data), "Mismatch in data lengths."

    # Extract time, original signal, and filtered signal
    time = original_data.iloc[:, 0]
    original_signal = original_data.iloc[:, 1]
    filtered_signal = filtered_data.iloc[:, 1]

    # Downsample the data if required
    if downsample_factor > 1:
        time = time[::downsample_factor]
        original_signal = original_signal[::downsample_factor]
        filtered_signal = filtered_signal[::downsample_factor]

    # Calculate noise
    noise = detrend(original_signal - filtered_signal)

    # Adjust Matplotlib settings to avoid rendering issues
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000  # Adjust this value if needed

    # Create the figure and axes for four rows
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot original signal
    axes[0].plot(time, original_signal, color='black', label='Original Signal')
    axes[0].set_title('Original Signal vs Time', fontsize=14)
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].grid(True)
    axes[0].legend(fontsize=10)

    # Plot filtered signal
    axes[1].plot(time, filtered_signal, color='black', label='Filtered Signal')
    axes[1].set_title('Filtered Signal vs Time', fontsize=14)
    axes[1].set_ylabel('Amplitude', fontsize=12)
    axes[1].grid(True)
    axes[1].legend(fontsize=10)

    # Plot noise
    axes[2].plot(time, noise, color='black', label='Noise (Original - Filtered)')
    axes[2].set_title('Noise vs Time', fontsize=14)
    axes[2].set_ylabel('Amplitude', fontsize=12)
    axes[2].grid(True)
    axes[2].legend(fontsize=10)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_png, dpi=600)

# Example usage
original_file = 'signal_1.csv'
filtered_file = 'filtered_signal_1.csv'
output_png = 'signals_visualization_1.png'

visualize_signals_with_detrended_noise(original_file, filtered_file, output_png, downsample_factor=10)
