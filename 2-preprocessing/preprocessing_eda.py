# Note: This file runs on the PMEmo2018 dataset and not on the 2019 dataset. 
# The 2018 dataset saves the EDA signals in a different format than the 2019 dataset.
# In 2018, the EDA signals from each subject are saved in a separate CSV file.
# In 2019, the EDA signals from all subjects are saved in a single CSV file.


import pandas as pd
import numpy as np
import cvxEDA
import os
import json

def convert(arousal_or_valence):
    path = 'EDA/' + arousal_or_valence + '/'
    all_eda_signals = []

    # Read every CSV file in the folder and collect all EDA signals
    for file in os.listdir(path):
        if file.endswith('.csv'):
            df = pd.read_csv(path + file)
            eda_signal = df['EDA(microsiemens)'].values
            all_eda_signals.append(eda_signal)

    # Concatenate all EDA signals into a single array
    all_eda_signals = np.concatenate(all_eda_signals)

    # Calculate the mean and standard deviation across all EDA signals
    eda_mean = np.mean(all_eda_signals)
    eda_std = np.std(all_eda_signals)

    # Process each file
    for file in os.listdir(path):
        if file.endswith('.csv'):
            # Read the CSV file
            df = pd.read_csv(path + file)

            # Assuming the first column is time(s) and the second column is EDA(microsiemens)
            eda_signal = df['EDA(microsiemens)'].values

            # Normalize the EDA signal using the global mean and std
            eda_signal_normalized = (eda_signal - eda_mean) / eda_std

            # Decompose the normalized EDA signal using cvxEDA
            [r, p, t, l, d, e, obj] = cvxEDA.cvxEDA(eda_signal_normalized, 0.02)

            # Reconstruct the original signal
            origin = r + t

            # Print file name
            print(file)

            file_name = file[:-4] + '.txt'

            # Save the components with the same file name but with .txt rather than .csv
            save_data_as_json(origin, f'txt/{arousal_or_valence}/origin/', file_name)
            save_data_as_json(t, f'txt/{arousal_or_valence}/tonic/', file_name)
            save_data_as_json(r, f'txt/{arousal_or_valence}/phasic/', file_name)

def save_data_as_json(data, folder_path, file_name):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        json_data = [float(val) for val in data]
        json.dump(json_data, file, ensure_ascii=False, indent=4)

convert('Valence')
convert('Arousal')