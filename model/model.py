import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectroEdaMusicNet(nn.Module):
    def __init__(self):
        super(SpectroEdaMusicNet, self).__init__()

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
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        music_features = 319
        lstm_hidden_size = 200
        # LSTM model inspired by this paper: https://www.sciencedirect.com/science/article/pii/S2215098620342385
        self.music_lstm = nn.LSTM(music_features, lstm_hidden_size, batch_first=True)
        self.music_fc1 = nn.Linear(lstm_hidden_size, 128)
        self.music_fc2 = nn.Linear(128, 128)

        # Fusion layer
        fusion_input_size = 384  # 3 * 128

        # TODO: using fusion_input_size for fusion output dim and input to subsequent linear layers does not seem to
        #  matter much self.fusion = nn.Linear(fusion_input_size, fusion_input_size)
        self.fusion = nn.Linear(fusion_input_size, 256)

        # Multi-Task Output layers
        self.arousal_output = nn.Linear(256, 1)
        self.valence_output = nn.Linear(256, 1)

    def forward(self, spectrogram, eda_data, music_vector):
        print('-------HELLO WORLD-------')
        print(spectrogram.size())
        print(eda_data.size())
        print(music_vector.size())
        # Initialize an empty tensor to store the fused features
        fused_features = []

        # Spectrogram feature extraction
        spec_features = self.spec_cnn(spectrogram)
        print(spec_features.size())
        fused_features.append(spec_features)

        # EDA feature extraction
        eda_features = self.eda_cnn(eda_data)
        print(eda_features.size())
        fused_features.append(eda_features)

        music_features = music_vector.unsqueeze(1)
        lstm_out, _ = self.music_lstm(music_features)
        music_features = lstm_out[:, -1, :]
        music_features = F.relu(self.music_fc1(music_features))
        music_features = F.relu(self.music_fc2(music_features))
        fused_features.append(music_features)
        print(music_features.size())

        # Fusion of spectrogram and EDA features
        fused_features = torch.cat(tuple(fused_features), dim=1)
        print(fused_features.size())
        fused_features = self.fusion(fused_features)

        # Output layers
        arousal_output = self.arousal_output(fused_features)
        valence_output = self.valence_output(fused_features)
        print(arousal_output.size())
        print(valence_output.size())
        return arousal_output, valence_output
