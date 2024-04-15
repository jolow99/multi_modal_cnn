import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectroEDANet(nn.Module):
    def __init__(self,
                 usesSpectrogram=True,
                 usesEDA=True,
                 usesMusic=True,
                 predictsArousal=True,
                 predictsValence=True,
                 usesMusicLSTM=True):
        super(SpectroEDANet, self).__init__()
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

        # EDA CNN
        self.eda_cnn = nn.Sequential(
            nn.Conv1d(10, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        music_features = 319
        self.music_cnn = nn.Sequential(
            nn.Linear(music_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        # LSTM model inspired by this paper: https://www.sciencedirect.com/science/article/pii/S2215098620342385
        lstm_hidden_size = 200
        self.music_lstm = nn.LSTM(music_features, lstm_hidden_size, batch_first=True)
        self.music_fc1 = nn.Linear(lstm_hidden_size, 128)
        self.music_fc2 = nn.Linear(128, 128)

        # Fusion layer
        fusion_input_size = 0
        if self.usesSpectrogram:
            fusion_input_size += 128
        if self.usesEDA:
            fusion_input_size += 128
        if self.usesMusic:
            fusion_input_size += 128

        # TODO: using fusion_input_size for fusion output dim and input to subsequent linear layers does not seem to
        #  matter much self.fusion = nn.Linear(fusion_input_size, fusion_input_size)
        self.fusion = nn.Linear(fusion_input_size, 256)

        # Multi-Task Output layers
        # self.arousal_output = nn.Linear(fusion_input_size, 10)
        # self.valence_output = nn.Linear(fusion_input_size, 10)
        self.arousal_output = nn.Linear(256, 10)
        self.valence_output = nn.Linear(256, 10)

        # Single-Task Output layer
        self.output = nn.Sequential(
            # nn.Linear(fusion_input_size, 128),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
            nn.ReLU(inplace=True)
        )

    def forward(self, spectrogram, eda_data, music_vector):
        # Initialize an empty tensor to store the fused features
        fused_features = []

        # Spectrogram feature extraction
        if self.usesSpectrogram:
            spec_features = self.spec_cnn(spectrogram)
            fused_features.append(spec_features)

        # EDA feature extraction
        if self.usesEDA:
            eda_features = self.eda_cnn(eda_data)
            fused_features.append(eda_features)

        if self.usesMusic:
            if self.usesMusicLSTM:
                music_features = music_vector.unsqueeze(1)
                lstm_out, _ = self.music_lstm(music_features)
                music_features = lstm_out[:, -1, :]
                music_features = F.relu(self.music_fc1(music_features))
                music_features = F.relu(self.music_fc2(music_features))
            else:
                music_features = self.music_cnn(music_vector)
            fused_features.append(music_features)

        # Fusion of spectrogram and EDA features
        fused_features = torch.cat(tuple(fused_features), dim=1)
        fused_features = self.fusion(fused_features)

        # Output layers    
        if self.predictsArousal and self.predictsValence:
            arousal_output = self.arousal_output(fused_features)
            valence_output = self.valence_output(fused_features)
            return arousal_output, valence_output
        else:
            return self.output(fused_features)