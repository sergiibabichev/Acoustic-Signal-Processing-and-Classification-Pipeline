# Module allows us to load, transform and save signals as csv files

import numpy as np
import pandas as pd
from scipy.io import wavfile

signals = {}
file_names = [
    'Pogran_1.wav', 'Pogran_2.wav', 'Hydrographic_boat.wav',
    'Rocket_Boat.wav', 'Slavutich_boat.wav'
]

for i, file_name in enumerate(file_names, 1):
    sample_rate, signal = wavfile.read(file_name)

    # Transforming stereo to mono
    if len(signal.shape) == 2:
        signal = signal.mean(axis=1)

    # Save data into dictionary
    signals[f'signal_{i}'] = {
        'sample_rate': sample_rate,
        'signal': signal
    }

# Saving as csv file
for key, value in signals.items():
    sample_rate = value['sample_rate']
    signal = value['signal']

    # Calculate times in second
    time = np.linspace(0, len(signal) / sample_rate, len(signal))

    # Create dataframe
    df = pd.DataFrame({
        'Time (s)': time,
        'Amplitude': signal
    })

    output_file = f'{key}.csv'
    df.to_csv(output_file, index=False)
    print(f"Saved {output_file}")



