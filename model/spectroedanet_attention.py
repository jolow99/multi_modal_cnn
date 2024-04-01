import torch
import torch.nn as nn
from model.Attention_module import NONLocal1D, CALayer1D, NONLocal2D, CALayer2D

class SpectroEDANet(nn.Module):
    def __init__(self, usesSpectrogram=True, usesEDA=True, usesMusic=True, predictsArousal=True, predictsValence=True):
        super(SpectroEDANet, self).__init__()
        self.usesSpectrogram = usesSpectrogram
        self.usesEDA = usesEDA
        self.usesMusic = usesMusic
        self.predictsArousal = predictsArousal
        self.predictsValence = predictsValence
        
        
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
            nn.Conv1d(10, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
#            nn.MaxPool1d(2),
#            nn.AdaptiveAvgPool1d(1),
#            nn.Flatten()
            nn.BatchNorm1d(128)
        )
          
        
        self.inter_channel = 128
        # Attention Mechanism
        self.Non_local1D = NONLocal1D(in_feat=self.inter_channel, inter_feat=self.inter_channel // 2)
        self.Attention1D = CALayer1D(channel=self.inter_channel)
        self.Non_local2D = NONLocal2D(in_feat=self.inter_channel, inter_feat=self.inter_channel // 2)
        self.Attention2D = CALayer2D(channel=self.inter_channel)
        
        
        # Fusion layer
        self.fusion = nn.Linear(128 + 128, 256)
        
        # Output layers
        self.arousal_output = nn.Linear(256, 10)
        self.valence_output = nn.Linear(256, 10)
    
    def forward(self, spectrogram, eda_data):
        # Spectrogram feature extraction
        # Using the attention mechanism
        spec_features = self.spec_cnn(spectrogram)
        spec_features = self.Non_local2D(spec_features)
        spec_features = self.Attention2D(spec_features)
        
        # EDA feature extraction
        # Using the attention mechanism
        eda_features = self.eda_cnn(eda_data)
        eda_features = self.Non_local1D(eda_features)
        eda_features = self.Attention1D(eda_features)
        
        #eda_features = eda_features.unsqueeze(2)  # Add a new dimension
        
        eda_features_reshaped = eda_features.unsqueeze(2).expand(-1, -1, 92, -1)
        print(spec_features.size())
        print(eda_features_reshaped.size())
        fused_features = torch.cat((spec_features, eda_features_reshaped), dim=3)


        # Fusion of spectrogram and EDA features       
        # fused_features = torch.cat((spec_features, eda_features), dim=1)
        fused_features = self.fusion(fused_features)
        
        # Regression outputs
        arousal_output = self.arousal_output(fused_features)
        valence_output = self.valence_output(fused_features)
        
        return arousal_output, valence_output