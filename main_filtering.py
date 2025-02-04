import pandas as pd
import math
from wavelet_functions import (analyze_all_imfs, visualize_max_snr_by_imf,
                               analyzeOptimalDL,visualizeOLByWavelet, process_selected_imfs,
                               replace_filtered_imfs, reconstruct_signal)

df = pd.read_csv('signal_1_imfs.csv')
df.columns = df.columns.str.replace(' ', '_')
path_filtered_signal = 'filtered_signal_1.csv'

# Determine the IMFs number
total_imfs = df.shape[1] - 1

# select hulf of imfs
half_imfs = math.ceil(total_imfs / 2)

# Select first IMFs
imf_indices = range(1, half_imfs+1)

#########################################################################################
# Optimal type of wavelet determination

decomp_level = 3
optimal_results = analyze_all_imfs(df, decomposition_level=decomp_level, imf_indices=imf_indices,
                                   path_png_file = 'snr_wavelet_all_imfs.png')

# Save results into csv file
optimal_results.to_csv('optimal_snr_wavelets_results.csv', index=False)

# Result visualization
visualize_max_snr_by_imf(optimal_results, 'max_snr_by_imf.png')

#######################################################################################
# Optimal wavelet decomposition level determination

max_level = 4
optimal_levels = analyzeOptimalDL( df, optimal_results, max_level=max_level)

# Visualization
visualizeOLByWavelet(optimal_levels, output_name='level_decomposition_analysis.png')

# Save results in csv file
optimal_levels.to_csv('optimal_levels_results.csv', index=False)

########################################################################################
# Optimal thresholding
results_df = process_selected_imfs(df, optimal_levels, 'optimal_thresholds.csv',
                                   'sure_criterion_all_imfs.png')

#################################################################################################
# Final signal filtering

# Replace filtered IMFs in df
updated_df = replace_filtered_imfs(df, results_df)

# Reconstruct the signal
reconstructed_signal = reconstruct_signal(updated_df)
updated_df['Filtered_Signal'] = reconstructed_signal  # Add the reconstructed signal to the DataFrame

filtered_signal = updated_df.loc[:, ['Time_(s)', 'Filtered_Signal']]

filtered_signal.to_csv(path_filtered_signal, index=False)


