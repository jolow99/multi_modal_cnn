import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Attention_module import NONLocal1D, CALayer1D, NONLocal2D, CALayer2D

device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class SpectroEDANet(nn.Module):
    def __init__(self,
                 usesSpectrogram=True,
                 usesEDA=True,
                 usesMusic=True,
                 usesAttention = True,
                 predictsArousal=True,
                 predictsValence=True):
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

        self.inter_channel = 128
        # Attention Mechanism
        self.Non_local1D = NONLocal1D(in_feat=self.inter_channel, inter_feat=self.inter_channel // 2)
        self.Attention1D = CALayer1D(channel=self.inter_channel)
        self.Non_local2D = NONLocal2D(in_feat=self.inter_channel, inter_feat=self.inter_channel // 2)
        self.Attention2D = CALayer2D(channel=self.inter_channel)

        if self.usesAttention:
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
                #            nn.MaxPool2d(2),
                #            nn.AdaptiveAvgPool2d((1, 1)),
                #            nn.Flatten(),
                nn.BatchNorm2d(128)
            )

            # EDA CNN
            self.eda_cnn = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                #            nn.MaxPool1d(2),
                #            nn.AdaptiveAvgPool1d(1),
                #            nn.Flatten()
                nn.BatchNorm1d(128)
            )

        # music_features = 319
        music_features = 64
        lstm_hidden_size = 200
        # LSTM model inspired by this paper: https://www.sciencedirect.com/science/article/pii/S2215098620342385
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
        # self.arousal_output = nn.Linear(fusion_input_size, 1)
        # self.valence_output = nn.Linear(fusion_input_size, 1)
        self.arousal_output = nn.Linear(256, 1)
        self.valence_output = nn.Linear(256, 1)

        # Single-Task Output layer
        self.output = nn.Sequential(
            # nn.Linear(fusion_input_size, 128),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 1),
            nn.ReLU(inplace=True)
        )

    def __str__(self):
        return f'Model with params: usesSpectrogram = {self.usesSpectrogram}, usesEDA = {self.usesEDA}, usesMusic = {self.usesMusic}, usesAttention = {self.usesAttention}'

    def forward(self, spectrogram, eda_data, music_vector):
        #print(spectrogram.shape)
        #print(eda_data.shape)
        #print(music_vector.shape)
        # Initialize an empty tensor to store the fused features
        fused_features = []

        # Spectrogram feature extraction
        if self.usesSpectrogram:
            spec_features = self.spec_cnn(spectrogram)
            fused_features.append(spec_features)
            # Using the attention mechanism
            if self.usesAttention:
                spec_features = self.Non_local2D(spec_features)
                spec_features = self.Attention2D(spec_features)

        # EDA feature extraction
        if self.usesEDA:
            eda_features = self.eda_cnn(eda_data)
            fused_features.append(eda_features)
            # Using the attention mechanism
            if self.usesAttention:
                eda_features = self.Non_local1D(eda_features)
                eda_features = self.Attention1D(eda_features)


        if self.usesMusic:
            music_features = music_vector.unsqueeze(1)
            lstm_out, _ = self.music_lstm(music_features)
            music_features = lstm_out[:, -1, :]
            music_features = F.relu(self.music_fc1(music_features))
            music_features = F.relu(self.music_fc2(music_features))
            fused_features.append(music_features)

        # Fusion of spectrogram and EDA features
        if not self.usesAttention:
            fused_features = torch.cat(tuple(fused_features), dim=1)
            fused_features = self.fusion(fused_features)
        else:
            # NOTE: attention NEEDS eda, music AND spec
            eda_features_reshaped = eda_features.unsqueeze(2).expand(-1, -1, 92, -1)
            fused_features = torch.cat((spec_features, eda_features_reshaped), dim=3)
            #print(fused_features.shape)
            #print(music_features.shape)
            music_features_reshaped = music_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 92, 572)
            fused_features = torch.cat((fused_features, music_features_reshaped), dim=1)
            fused_features = nn.AdaptiveAvgPool2d((1, 1))(fused_features)
            fused_features = fused_features.view(fused_features.size(0), -1) 
            #print(fused_features.shape)
            
            self.fusion = nn.Linear(256, 256).to(device)
            fused_features = self.fusion(fused_features)

        # Output layers
        if self.predictsArousal and self.predictsValence:
            arousal_output = self.arousal_output(fused_features)
            valence_output = self.valence_output(fused_features)
            return arousal_output, valence_output
        else:
            return self.output(fused_features)
