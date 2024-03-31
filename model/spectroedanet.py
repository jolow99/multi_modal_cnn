import torch
import torch.nn as nn

class SpectroEDANet(nn.Module):
    def __init__(self, isArousal=True):
        super(SpectroEDANet, self).__init__()
        self.isArousal = True
        
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
        
        # Fusion layer
        self.fusion = nn.Linear(128 + 128, 256)
        
        # # Output layers
        # self.arousal_output = nn.Linear(256, 10)
        # self.valence_output = nn.Linear(256, 10)

        # Output layer
        self.output = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, spectrogram, eda_data):
        # Spectrogram feature extraction
        spec_features = self.spec_cnn(spectrogram)
        
        # EDA feature extraction
        eda_features = self.eda_cnn(eda_data)
        
        # Fusion of spectrogram and EDA features
        fused_features = torch.cat((spec_features, eda_features), dim=1)
        fused_features = self.fusion(fused_features)

        return self.output(fused_features)
        
        # Regression outputs
        # arousal_output = self.arousal_output(fused_features)
        # valence_output = self.valence_output(fused_features)
        
        # return arousal_output, valence_output