from loader.data_loader import PMEmoDataset
import torch.utils.data as data
# Note that data loader and model have not yet been tested. I've simply put them there as placeholders. 

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




