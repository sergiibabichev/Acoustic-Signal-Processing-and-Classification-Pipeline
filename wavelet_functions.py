import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from criteria_functions import mad, snr_crit, sure_threshold

def waveletSmooth(x, wavelet, level):
    """
    Perform wavelet-based smoothing on the signal.
    """
    coeff = pywt.wavedec(x, wavelet, level)
    sigma = mad(coeff[-1])
    uthresh = 0.1 * sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = [pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:]]
    return pd.Series(pywt.waverec(coeff, wavelet)[:len(x)], index=np.arange(len(x)))

# Function for selecting the optimal wavelet for a single IMF
def typeWaveletSelection(imf, wavList, level):
    results = {'Wavelet': [], 'SNR_value': []}

    for wavelet in wavList:
        # Smoothed signal
        smoothed_signal = waveletSmooth(imf, wavelet, level)

        # Noise component
        noise = imf - smoothed_signal

        # Calculate SNR
        snr_value = snr_crit(smoothed_signal, noise)

        # Append results
        results['Wavelet'].append(wavelet)
        results['SNR_value'].append(snr_value)

    return pd.DataFrame(results)

# Function for visualizing results for all IMFs
def visualize_wavelet_results_all_imfs(results_dict, wavelet_types, output_name):
    fig, axes = plt.subplots(len(wavelet_types), 1, figsize=(14, 18), sharex=False)

    for ax, wavelet_type in zip(axes, wavelet_types):
        for imf_key, results in results_dict.items():
            # Visualization for each IMF
            ax.plot(results[wavelet_type]['Wavelet'], results[wavelet_type]['SNR_value'],
                    label=f'{imf_key}', marker='o')

        # Configure axes and titles
        ax.set_title(f'{wavelet_type.capitalize()} Wavelets', fontsize=22)
        ax.set_ylabel('SNR Values', fontsize=18)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
        ax.legend(title='IMFs', fontsize=10)

    # Global X-axis label
    plt.xlabel('Wavelet Type', fontsize=18)
    plt.tight_layout()

    # Save plot
    plt.savefig(output_name, dpi=600)

# Function to determine the optimal wavelet for a single IMF
def optimalWaveletForIMF(imf, wavelet_types, decomposition_level):
    optimal_wavelet = None
    max_snr = -np.inf

    all_results = {}

    for wavelet_type in wavelet_types:
        wavelets = pywt.wavelist(wavelet_type)
        results = typeWaveletSelection(imf, wavelets, decomposition_level)
        all_results[wavelet_type] = results

        # Find wavelet with maximum SNR
        max_index = results['SNR_value'].idxmax()
        if results['SNR_value'][max_index] > max_snr:
            max_snr = results['SNR_value'][max_index]
            optimal_wavelet = results['Wavelet'][max_index]

    return optimal_wavelet, max_snr, all_results

# Function for analyzing all IMFs and selecting an optimal wavelet for each IMF
def analyze_all_imfs(df, decomposition_level, imf_indices, path_png_file):
    results_dict = {}
    optimal_results = []

    # List of wavelet types
    wavelet_types = ['bior', 'coif', 'sym', 'db']

    # Calculate results for each selected IMF
    for i in imf_indices:
        imf = df.iloc[:, i].values
        optimal_wavelet, max_snr, all_results = optimalWaveletForIMF(imf, wavelet_types, decomposition_level)

        # Save results for plots
        results_dict[f'IMF_{i}'] = all_results

        # Append optimal results
        optimal_results.append({
            'IMF': f'IMF_{i}',
            'Optimal_Wavelet': optimal_wavelet,
            'Max_SNR': max_snr
        })

    # Visualize results
    visualize_wavelet_results_all_imfs(results_dict, wavelet_types, output_name=path_png_file)

    return pd.DataFrame(optimal_results)

# Function to visualize Max_SNR by IMF
def visualize_max_snr_by_imf(data, output_name):

    values = data['Max_SNR']
    labels = data['IMF']
    wavelets = data['Optimal_Wavelet']

    # Calculate median of Max_SNR
    median_snr = data['Max_SNR'].median()

    # Create bar plot
    plt.figure(figsize=(12, 4))
    colors = plt.cm.tab10(range(len(values)))  # Use colormap for colors
    bars = plt.bar(labels, values, color=colors)

    # Add legend
    for bar, wavelet in zip(bars, wavelets):
        bar.set_label(wavelet)

    plt.legend(
        title='Optimal Wavelets',
        fontsize=10,
        loc='upper left',
        bbox_to_anchor=(1, 1)  # Place legend outside the plot
    )

    # Add horizontal line for median
    plt.axhline(y=median_snr, color='red', linestyle='--', label=f'Median SNR = {median_snr:.2f}')

    # Configure plot
    plt.title('Max SNR vs IMF for Optimal Wavelets (Bar Plot)', fontsize=16)
    plt.xlabel('IMFs', fontsize=14)
    plt.ylabel('Max_SNR values', fontsize=14)
    plt.xticks(rotation=45, ha='right')  # Rotate X-axis labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display plot
    plt.tight_layout()

    # Save plot
    plt.savefig(output_name, dpi=600)
    plt.close()

def analyzeWaveletDecompositionLevel(signal, wavelet, max_level=4):

    results = {'Level': [], 'SNR_value': []}

    for level in range(1, max_level + 1):
        # Filter the signal at the given decomposition level
        filtered_signal = waveletSmooth(signal, wavelet, level)
        noise = signal - filtered_signal

        # Calculate SNR
        snr_value = snr_crit(filtered_signal, noise)

        # Add results
        results['Level'].append(level)
        results['SNR_value'].append(snr_value)

    return pd.DataFrame(results)

def analyzeOptimalDL(df, optimal_wavelets, max_level=4):
    # Filter IMFs based on the condition Max_SNR < median
    median_snr = optimal_wavelets['Max_SNR'].median()
    filtered_wavelets = optimal_wavelets[optimal_wavelets['Max_SNR'] < median_snr]

    optimal_levels = []

    for _, row in filtered_wavelets.iterrows():
        imf_key = row['IMF']
        optimal_wavelet = row['Optimal_Wavelet']
        imf_index = int(imf_key.split('_')[1])
        imf = df.iloc[:, imf_index].values

        # Analyze decomposition levels
        decomposition_results = analyzeWaveletDecompositionLevel(imf, optimal_wavelet, max_level)

        # Find the optimal decomposition level
        max_index = decomposition_results['SNR_value'].idxmax()
        optimal_level = decomposition_results['Level'][max_index]

        optimal_levels.append({
            'IMF': imf_key,
            'Optimal_Wavelet': optimal_wavelet,
            'Optimal_Level': optimal_level,
            'Max_SNR': decomposition_results['SNR_value'][max_index]
        })

    return pd.DataFrame(optimal_levels)

def visualizeOLByWavelet(data, output_name):

    values = data['Optimal_Level']
    labels = data['IMF']
    wavelets = data['Optimal_Wavelet']

    # Create a bar plot
    plt.figure(figsize=(8, 4))
    colors = plt.cm.tab10(range(len(values)))  # Use colormap for colors
    bars = plt.bar(labels, values, color=colors)

    # Add legend
    for bar, wavelet in zip(bars, wavelets):
        bar.set_label(wavelet)

    plt.legend(
        title='Optimal Wavelets',
        fontsize=10,
        loc='upper left',
        bbox_to_anchor=(1, 1)  # Place the legend outside the plot
    )

    # Configure the plot
    plt.title('Bar Plot for Optimal Levels', fontsize=16)
    plt.xlabel('IMFs', fontsize=14)
    plt.ylabel('Optimal Levels', fontsize=14)
    plt.xticks(rotation=45, ha='right')  # Rotate X-axis labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_name, dpi=600)
    plt.close()

# Function for processing selected IMFs
def process_selected_imfs(df, optimal_levels, output_csv, output_png):

    results = []

    # Prepare figure for plots
    rows = 2
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()  # Convert axes into an array for easy access

    for i, row in enumerate(optimal_levels.itertuples(index=False)):
        imf_key = row.IMF
        wavelet = row.Optimal_Wavelet
        level = row.Optimal_Level
        imf = df.loc[:, imf_key].values  # Select IMF from the df table

        # Compute SURE
        optimal_lambda, lambdas, sure_values = sure_threshold(imf, wavelet, level)

        # Add to results
        results.append({
            'IMF': imf_key,
            'Optimal_Wavelet': wavelet,
            'Optimal_Level': level,
            'Optimal_Threshold': optimal_lambda
        })

        # Visualization for each IMF
        ax = axes[i]
        ax.plot(lambdas, sure_values, label=f'{imf_key}')
        ax.axvline(optimal_lambda, linestyle="--", color="red", alpha=0.7)
        ax.set_title(f"SURE Criterion for {imf_key}", fontsize=12)
        ax.set_xlabel("Threshold λ", fontsize=12)
        ax.set_ylabel("SURE Value", fontsize=12)
        # ax.legend(fontsize=10)
        ax.grid(True)

        # Add text with the optimal λ
        ax.text(0.95, 0.95, f'Optimal λ={optimal_lambda:.4f}', transform=ax.transAxes,
                fontsize=10, color='black', verticalalignment='top', horizontalalignment='right')

    # Remove empty subplots if there are more subplots than IMFs
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Configure and save the plot
    plt.tight_layout()
    plt.savefig(output_png, dpi=600)

    # Save results table
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    return results_df

# Function for wavelet filtering
def wavelet_filter(imf, wavelet, level, threshold):
    """
    Applies wavelet-based filtering to a given IMF using the specified wavelet, decomposition level, and threshold.
    """
    coeffs = pywt.wavedec(imf, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[:len(imf)]

# Function to replace selected IMFs in the df table
def replace_filtered_imfs(df, results_df):
    """
    Replaces the specified IMFs in the DataFrame with their filtered versions.

    Parameters:
        df (pd.DataFrame): Original DataFrame containing IMFs.
        results_df (pd.DataFrame): DataFrame with optimal wavelet parameters for filtering.

    Returns:
        pd.DataFrame: Updated DataFrame with filtered IMFs.
    """
    updated_df = df.copy()

    for _, row in results_df.iterrows():
        imf_key = row['IMF']
        wavelet = row['Optimal_Wavelet']
        level = int(row['Optimal_Level'])
        threshold = row['Optimal_Threshold']

        if imf_key in updated_df.columns:
            imf = updated_df[imf_key].values
            filtered_imf = wavelet_filter(imf, wavelet, level, threshold)
            updated_df[imf_key] = filtered_imf  # Replace the IMF in the table

    return updated_df

# Function to reconstruct the signal
def reconstruct_signal(df):
    """
    Reconstructs the signal by summing all IMFs in the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing all IMFs.

    Returns:
        np.ndarray: Reconstructed signal.
    """
    imf_columns = [col for col in df.columns if col != 'Time']  # Exclude the 'Time' column
    imfs = df[imf_columns].values
    reconstructed_signal = np.sum(imfs, axis=1)  # Sum all IMFs to reconstruct the signal
    return reconstructed_signal

