import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import kurtosis, skew

# function for data loading
def load_signals(filepaths, labels):
    signals = []
    for filepath, label in zip(filepaths,labels):
        data = pd.read_csv(filepath, header=0)  # Load with headers
        signal = data.iloc[:, 1].values  # Use only the second column (signal)
        signals.append((signal, label))
    return signals

# Function for signal scaling
def standardize_signals(signals):
    scaler = StandardScaler()
    standardized_signals = []
    for signal, label in signals:
        scaled_signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
        standardized_signals.append((scaled_signal, label))
    return standardized_signals

# Function to segmentation of the signals into chunks of 1000 points and to add class labels
def segment_signals(signals, segment_size=1000):
    segmented_data = []
    for signal, label in signals:  # Unpack the signal and its label
        num_segments = len(signal) // segment_size  # Number of complete segments
        signal = signal[:num_segments * segment_size]  # Trim signal to fit into full segments
        segments = signal.reshape(num_segments, segment_size)  # Reshape into segments
        labels = np.full((num_segments, 1), label)  # Assign the same label to all segments
        segmented_data.append(np.hstack((segments, labels)))  # Append segments with labels
    return np.vstack(segmented_data)  # Combine all segments into one array

# Functions for features extraction from the signal's segments

# Statistical Features
def extract_statistical_features(segment):
    return {
        'mean': np.mean(segment),
        'std': np.std(segment),
        'min': np.min(segment),
        'max': np.max(segment),
        'energy': np.sum(segment**2),
        'kurtosis': kurtosis(segment),
        'skewness': skew(segment)
    }

# Frequency-Domain Features (FFT)
def extract_fft_features(segment, sampling_rate=1000):
    fft_values = np.fft.fft(segment)
    fft_magnitude = np.abs(fft_values)
    fft_frequencies = np.fft.fftfreq(len(segment), d=1 / sampling_rate)

    dominant_freq = fft_frequencies[np.argmax(fft_magnitude[:len(segment) // 2])]

    return {
        'fft_mean': np.mean(fft_magnitude),
        'fft_std': np.std(fft_magnitude),
        'fft_energy': np.sum(fft_magnitude ** 2),
        'fft_max': np.max(fft_magnitude),
        'fft_dominant_freq': dominant_freq
    }

# Time-Frequency Features (Wavelet Transform)
def extract_wavelet_features(segment, wavelet='bior3.9', level=3):
    coeffs = pywt.wavedec(segment, wavelet, level=level)
    features = {}
    for i, coeff in enumerate(coeffs):
        features[f'wavelet_mean_{i}'] = np.mean(coeff)
        features[f'wavelet_std_{i}'] = np.std(coeff)
    return features

# Combine all features
def extract_all_features(segment, sampling_rate=1000):
    features = {}
    features.update(extract_statistical_features(segment))
    features.update(extract_fft_features(segment, sampling_rate))
    features.update(extract_wavelet_features(segment))
    return features

#  Extract Features for All Segments
def extract_features_from_segments(segmented_data, sampling_rate=1000):
    features_list = []
    for segment, label in segmented_data:
        features = extract_all_features(segment, sampling_rate)
        features['label'] = label
        features_list.append(features)
    return pd.DataFrame(features_list)

# Define the function for classifier parameters optimization

def rf_cv(X_train, y_train, n_estimators, max_depth, min_samples_split, min_samples_leaf):
    # Create the model with the specified parameters
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),  # Needs to be integer
        max_depth=None if max_depth < 1 else int(max_depth),  # None for max_depth=0
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42  # To ensure reproducibility
    )

    # Perform cross-validation and return the mean accuracy
    return np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy'))