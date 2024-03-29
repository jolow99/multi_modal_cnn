import torch
import torch.nn as nn

class SpectroEDANet(nn.Module):
    def __init__(self, num_eda_annotations=10, num_classes=2):
        super(SpectroEDANet, self).__init__()
        
        # Spectrogram CNN
        self.spec_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.spec_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.spec_pool = nn.MaxPool2d(kernel_size=2)
        self.spec_dropout = nn.Dropout(0.25)
        
        # EDA CNN
        self.eda_conv1 = nn.Conv1d(num_eda_annotations, 64, kernel_size=3, padding=1)
        self.eda_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.eda_pool = nn.MaxPool1d(kernel_size=2)
        self.eda_dropout = nn.Dropout(0.25)
        
        # Fusion layer
        self.fusion = nn.Linear(128 * 64 + 128 * 840, 256)
        self.fusion_dropout = nn.Dropout(0.5)
        
        # Output layer
        self.output = nn.Linear(256, num_classes)
        
    def forward(self, spectrogram, eda):
        # Spectrogram CNN
        spec_x = self.spec_conv1(spectrogram)
        spec_x = nn.functional.relu(spec_x)
        spec_x = self.spec_pool(spec_x)
        spec_x = self.spec_dropout(spec_x)
        spec_x = self.spec_conv2(spec_x)
        spec_x = nn.functional.relu(spec_x)
        spec_x = self.spec_pool(spec_x)
        spec_x = self.spec_dropout(spec_x)
        spec_x = spec_x.view(spec_x.size(0), -1)
        
        # EDA CNN
        eda_x = self.eda_conv1(eda)
        eda_x = nn.functional.relu(eda_x)
        eda_x = self.eda_pool(eda_x)
        eda_x = self.eda_dropout(eda_x)
        eda_x = self.eda_conv2(eda_x)
        eda_x = nn.functional.relu(eda_x)
        eda_x = self.eda_pool(eda_x)
        eda_x = self.eda_dropout(eda_x)
        eda_x = eda_x.view(eda_x.size(0), -1)
        
        # Fusion
        x = torch.cat((spec_x, eda_x), dim=1)
        x = self.fusion(x)
        x = nn.functional.relu(x)
        x = self.fusion_dropout(x)
        
        # Output
        x = self.output(x)
        return x