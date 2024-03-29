import os
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from PIL import Image

class PMEmoDataset(data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.annotations_dir = os.path.join(root_dir, "annotations")
        self.eda_dir = os.path.join(root_dir, "eda")
        self.spectrograms_dir = os.path.join(root_dir, "spectrograms")
        
        self.music_ids = self._get_music_ids()
        
    def __len__(self):
        return len(self.music_ids)
    
    def __getitem__(self, index):
        music_id = self.music_ids[index]
        
        # Load annotations
        arousal_file = os.path.join(self.annotations_dir, "Arousal", f"{music_id}-A.csv")
        valence_file = os.path.join(self.annotations_dir, "Valence", f"{music_id}-V.csv")
        
        arousal_data = pd.read_csv(arousal_file, usecols=["subjectId", "static"])
        valence_data = pd.read_csv(valence_file, usecols=["subjectId", "static"])
        
        subject_ids = arousal_data["subjectId"].tolist()
        arousal_labels = arousal_data["static"].values
        valence_labels = valence_data["static"].values
        
        # Load EDA data
        eda_data = []
        for subject_id in subject_ids:
            eda_file = os.path.join(self.eda_dir, "Arousal", "origin", f"{music_id}_{subject_id}_A.txt")
            with open(eda_file, "r") as f:
                eda_signal = np.array(json.loads(f.read()))
            eda_data.append(eda_signal)
        
        eda_data = np.array(eda_data)
        
        # Load spectrogram
        spectrogram_file = os.path.join(self.spectrograms_dir, f"{music_id}.png")
        spectrogram = Image.open(spectrogram_file)
        spectrogram = spectrogram.convert("L")  # Convert to grayscale
        spectrogram = np.array(spectrogram)
        
        # Convert to tensors
        eda_data = torch.tensor(eda_data, dtype=torch.float32)
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
        arousal_labels = torch.tensor(arousal_labels, dtype=torch.float32)
        valence_labels = torch.tensor(valence_labels, dtype=torch.float32)
        
        return spectrogram, eda_data, arousal_labels, valence_labels
    
    def _get_music_ids(self):
        music_ids = set()
        for filename in os.listdir(os.path.join(self.annotations_dir, "Arousal")):
            music_id = filename.split("-")[0]
            music_ids.add(music_id)
        return sorted(list(music_ids))

# Usage example
root_dir = "1-dataset"
dataset = PMEmoDataset(root_dir)
dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

for spectrogram, eda_data, arousal_labels, valence_labels in dataloader:
    # Pass the data to the multi-modal CNN
    # spectrogram: (batch_size, 1, height, width)
    # eda_data: (batch_size, num_subjects, length)
    # arousal_labels: (batch_size, num_subjects)
    # valence_labels: (batch_size, num_subjects)
    # ...