import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectroEDANet(nn.Module):
    def __init__(self,
                 usesSpectrogram,
                usesEDA,
                usesMusic,
                usesAttention,
                predictsArousal,
                predictsValence,
                fc_size=256, dropout_p=0.0):
        super(SpectroEDANet, self).__init__()
        self.usesSpectrogram = usesSpectrogram
        self.usesEDA = usesEDA
        self.usesMusic = usesMusic
        self.usesAttention = usesAttention
        self.predictsArousal = predictsArousal
        self.predictsValence = predictsValence

        # Spectrogram CNN
        self.spec_cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),
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

        # music_features = 319
        music_features = 64
        lstm_hidden_size = 200
        self.music_lstm = nn.LSTM(music_features, lstm_hidden_size, batch_first=True)
        self.music_fc1 = nn.Linear(lstm_hidden_size, 128)
        self.music_fc2 = nn.Linear(128, 128)

        # Fusion layer
        fusion_input_size = 384
        self.fusion_fc1 = nn.Linear(fusion_input_size, fc_size)
        self.bn1 = nn.BatchNorm1d(fc_size)
        self.drop1 = nn.Dropout(p=dropout_p)

        torch.nn.init.kaiming_uniform_(self.fusion_fc1.weight, 
                               a=0.1, mode="fan_in", 
                               nonlinearity="leaky_relu")

        self.fusion_fc2 = nn.Linear(fc_size, fc_size)
        self.bn2 = nn.BatchNorm1d(fc_size)
        self.drop2 = nn.Dropout(p=dropout_p)

        torch.nn.init.kaiming_uniform_(self.fusion_fc2.weight, 
                               a=0.1, mode="fan_in", 
                               nonlinearity="leaky_relu")

        self.arousal_output = nn.Linear(fc_size, 1)
        self.valence_output = nn.Linear(fc_size, 1)

    def forward(self, spectrogram, eda_data, music_vector):
        fused_features = []

        spec_features = self.spec_cnn(spectrogram)
        fused_features.append(spec_features)

        eda_features = self.eda_cnn(eda_data)
        fused_features.append(eda_features)

        music_features = music_vector.unsqueeze(1)
        lstm_out, _ = self.music_lstm(music_features)
        music_features = lstm_out[:, -1, :]
        music_features = F.relu(self.music_fc1(music_features))
        music_features = F.relu(self.music_fc2(music_features))
        fused_features.append(music_features)

        fused_features = torch.cat(tuple(fused_features), dim=1)
        fused_features = self.fusion_fc1(fused_features)
        fused_features = self.bn1(fused_features)
        fused_features = self.drop1(fused_features)

        fused_features = self.fusion_fc2(fused_features)
        fused_features = self.bn2(fused_features)
        fused_features = self.drop2(fused_features)

        arousal_output = self.arousal_output(fused_features)
        valence_output = self.valence_output(fused_features)
        return arousal_output, valence_output