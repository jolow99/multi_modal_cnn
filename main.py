from loader.data_loader import PMEmoDataset
import torch.utils.data as data
from model import spectroedanet

# Usage example
root_dir = "1-dataset"
dataset = PMEmoDataset(root_dir)
dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

# Print 1 batched sample  from the dataloader
spectrogram, eda_data, arousal_label, valence_label = (next(iter(dataloader)))

print(spectrogram.shape) # torch.Size([32, 1, 369, 496])
print(eda_data.shape) # torch.Size([32, 10, 896])
print(arousal_label.shape) # torch.Size([32, 10])
print(valence_label.shape) # torch.Size([32, 10])

# TODO: Implement the training loop
model = spectroedanet.SpectroEDANet()





