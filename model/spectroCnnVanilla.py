import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectroNet(nn.Module):
    def __init__(self,
                 usesSpectrogram=True,
                 usesEDA=True,
                 usesMusic=True,
                 predictsArousal=True,
                 predictsValence=True,
                 usesMusicLSTM=True):
        super(SpectroNet, self).__init__()
        self.usesSpectrogram = usesSpectrogram
        self.usesEDA = usesEDA
        self.usesMusic = usesMusic
        self.predictsArousal = predictsArousal
        self.predictsValence = predictsValence
        self.usesMusicLSTM = usesMusicLSTM

        # Spectrogram CNN
        self.spec_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # # Combined valence + arousal output layer
        # self.output = nn.Sequential(
        #     # nn.Linear(fusion_input_size, 128),
        #     nn.Linear(128, 256),
        #     nn.Linear(256, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 2),
        #     nn.ReLU(inplace=True)
        # )

        # Either valence or arousal output layer
        self.output = nn.Sequential(
            # nn.Linear(fusion_input_size, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, spectrogram, eda_data, music_vector):
        spectrogram = self.spec_cnn(spectrogram)
        spectrogram = self.output(spectrogram)
        return spectrogram