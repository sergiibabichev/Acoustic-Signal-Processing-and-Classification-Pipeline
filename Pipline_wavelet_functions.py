import numpy as np
import pandas as pd
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
def analyze_all_imfs(df, decomposition_level, imf_indices):
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
    return pd.DataFrame(optimal_results)


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


# Function for processing selected IMFs
def process_selected_imfs(df, optimal_levels):

    results = []

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

    # Save results table
    results_df = pd.DataFrame(results)

    return results_df

# Function for wavelet filtering
def wavelet_filter(imf, wavelet, level, threshold):
    coeffs = pywt.wavedec(imf, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[:len(imf)]

# Function to replace selected IMFs in the df table
def replace_filtered_imfs(df, results_df):
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
    imf_columns = [col for col in df.columns if col != 'Time']  # Exclude the 'Time' column
    imfs = df[imf_columns].values
    reconstructed_signal = np.sum(imfs, axis=1)  # Sum all IMFs to reconstruct the signal
    return reconstructed_signal



