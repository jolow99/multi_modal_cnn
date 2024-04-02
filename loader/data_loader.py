import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import json
from scipy.interpolate import interp1d
import pandas as pd
from sklearn.feature_selection import SelectPercentile, mutual_info_regression


def multi_target_score(X, y):
    scores = []
    for i in range(y.shape[1]):  # Iterate over each target
        score = mutual_info_regression(X, y[:, i])
        scores.append(score)
    return np.mean(scores, axis=0)

class PMEmoDataset(data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.annotations_dir = os.path.join(root_dir, "annotations")
        self.eda_dir = os.path.join(root_dir, "eda")
        self.spectrograms_dir = os.path.join(root_dir, "spectrograms")
        self.music_ids = self._get_music_ids()
        self.music_df = pd.read_csv("/Users/joel-tay/Desktop/multi_modal_cnn/dataset/static_features.csv",
                               index_col="musicId")

        # feature selection for music_df
        feature_selector = SelectPercentile(multi_target_score, percentile=5)
        target_cols = ['target_arousal', 'target_valence']
        y = self.music_df[target_cols]
        X = self.music_df.drop(columns=target_cols)
        # resultant music_df does not have target columns, target cols are purely for feature selection
        feature_selector.fit(X, y)
        selected_features_mask = feature_selector.get_support()
        selected_cols = X.columns[selected_features_mask]
        self.music_df = X[selected_cols]

    def __len__(self):
        return len(self.music_ids)

    def __getitem__(self, index):
        music_id = self.music_ids[index]

        # Load annotations
        arousal_file = os.path.join(self.annotations_dir, "Arousal", f"{music_id}-A.csv")
        valence_file = os.path.join(self.annotations_dir, "Valence", f"{music_id}-V.csv")

        arousal_labels = []
        valence_labels = []
        subject_ids = []

        with open(arousal_file, 'r') as file:
            next(file)
            for line in file:
                values = line.split(',')
                subject_ids.append(values[0])
                arousal_labels.append(float(values[1]))

        with open(valence_file, 'r') as file:
            next(file)
            for line in file:
                values = line.split(',')
                valence_labels.append(float(values[1]))

        # If there are more than 10 subjects, only take the first 10
        if len(subject_ids) > 10:
            subject_ids = subject_ids[:10]
            arousal_labels = arousal_labels[:10]
            valence_labels = valence_labels[:10]

        # Load EDA data
        eda_data = []
        for subject_id in subject_ids:
            eda_file = os.path.join(self.eda_dir, "Arousal", "origin", f"{music_id}_{subject_id}_A.txt")
            with open(eda_file, "r") as f:
                eda_signal = np.array(json.loads(f.read()))
            eda_data.append(eda_signal)

        # Interpolate the EDA sequences to have a length of 896
        interpolated_eda_data = []
        for eda_signal in eda_data:
            if len(eda_signal) != 896:
                x = np.arange(len(eda_signal))
                f = interp1d(x, eda_signal, kind='linear')
                x_new = np.linspace(0, len(eda_signal) - 1, 896)
                interpolated_signal = f(x_new)
            else:
                interpolated_signal = eda_signal
            interpolated_eda_data.append(interpolated_signal)

        eda_data = np.array(interpolated_eda_data)

        # Load spectrogram
        spectrogram_file = os.path.join(self.spectrograms_dir, f"{music_id}.png")
        spectrogram = Image.open(spectrogram_file)
        spectrogram = spectrogram.convert("L")  # Convert to grayscale
        spectrogram = np.array(spectrogram)

        eda_data = torch.tensor(eda_data, dtype=torch.float32)
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)

        arousal_labels = torch.tensor(arousal_labels, dtype=torch.float32)
        valence_labels = torch.tensor(valence_labels, dtype=torch.float32)

        # opensmile music features
        music_features = self.music_df.loc[self.music_df.index == int(music_id)].iloc[0]  # here is pandas type series
        music_vector = torch.tensor(np.array(music_features))

        return spectrogram, eda_data, arousal_labels, valence_labels, music_vector

    def _get_music_ids(self):
        music_ids = set()
        for filename in os.listdir(os.path.join(self.annotations_dir, "Arousal")):
            music_id = filename.split("-")[0]
            music_ids.add(music_id)
        return sorted(list(music_ids))
