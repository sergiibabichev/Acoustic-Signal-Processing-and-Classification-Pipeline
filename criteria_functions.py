import numpy as np
import pywt

def snr_crit(S_filtered, S_noise):
    """
    Calculate Signal-to-Noise Ratio (SNR).
    """
    return 10 * np.log10(np.sum(S_filtered**2) / np.sum(S_noise**2))


def mad(data):
    """
    Median Absolute Deviation (MAD).
    """
    return np.median(np.abs(data - np.median(data)))

def sure_threshold(x, wavelet, level):
    """
    Обчислення оптимального трешолдингу за критерієм SURE.
    """
    # Вейвлет-декомпозиція
    coeff = pywt.wavedec(x, wavelet, level)
    detail_coeff = coeff[-1]  # Використовуємо найвищий рівень деталізації

    # Оцінка рівня шуму
    sigma = mad(detail_coeff) / 0.6745

    # Діапазон трешолдингу
    lambdas = np.linspace(0, 3 * sigma, 100)
    sure_values = []

    for lam in lambdas:
        # Soft thresholding
        thresholded_coeff = np.sign(detail_coeff) * np.maximum(np.abs(detail_coeff) - lam, 0)

        # SURE критерій
        residuals = np.minimum(detail_coeff**2, lam**2)
        sure = np.sum(residuals) + sigma**2 * np.sum(np.abs(detail_coeff) > lam)
        sure_values.append(sure)

    # Знаходження мінімального SURE
    min_sure_index = np.argmin(sure_values)
    optimal_lambda = lambdas[min_sure_index]

    return optimal_lambda, lambdas, sure_values