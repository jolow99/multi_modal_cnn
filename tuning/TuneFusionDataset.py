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
        self.music_df = pd.read_csv("/home/abram/Developer/multi_modal_cnn/dataset/static_features.csv",
                                    index_col="musicId")

        # feature selection for music_df
        feature_selector = SelectPercentile(multi_target_score, percentile=1)
        target_cols = ['target_arousal', 'target_valence']
        y = self.music_df[target_cols]
        X = self.music_df.drop(columns=target_cols)
        # resultant music_df does not have target columns, target cols are purely for feature selection
        feature_selector.fit(X, y)
        selected_features_mask = feature_selector.get_support()
        selected_cols = X.columns[selected_features_mask]
        # store these selected_cols in file so our app can choose the correct cols in preprocessing stage
        pd.DataFrame(selected_cols).to_csv("selected_music_features.csv", index=False, header=False)
        self.music_df = X[selected_cols]

    def __len__(self):
        # print(len(self.music_ids))
        dataset_length = len(self.music_ids) * 10
        # print(dataset_length)
        return dataset_length

    def __getitem__(self, i):
        valence_arousal_index = i % 10
        index = i // 10
        music_id = self.music_ids[index]

        # Load annotations
        arousal_file = os.path.join(self.annotations_dir, "Arousal", f"{music_id}-A.csv")
        valence_file = os.path.join(self.annotations_dir, "Valence", f"{music_id}-V.csv")

        arousal_labels = []
        valence_labels = []
        subject_ids = []

        with open(arousal_file, 'r') as file:
            line = file.readlines()[valence_arousal_index + 1].strip()
            values = line.split(',')
            subject_ids.append(values[0])
            arousal_value = float(values[1])
            arousal_labels.append(arousal_value)

        with open(valence_file, 'r') as file:
            line = file.readlines()[valence_arousal_index + 1].strip()
            values = line.split(',')
            valence_value = float(values[1])
            valence_labels.append(valence_value)

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