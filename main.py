import torch
import torch.nn as nn
import torch.optim as optim
from loader.data_loader import PMEmoDataset
import torch.utils.data as data
from model import spectroedanet

# TODO: Add k-fold cross-validation to the training loop, and print out relevant metrics on each loop. 

root_dir = "dataset"
dataset = PMEmoDataset(root_dir)
dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate the model
model = spectroedanet.SpectroEDANet()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device("mps")
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        spectrogram, eda_data, arousal_label, valence_label = data
        spectrogram = spectrogram.to(device)
        eda_data = eda_data.to(device)
        arousal_label = arousal_label.to(device)
        valence_label = valence_label.to(device)

        optimizer.zero_grad()

        arousal_output, valence_output = model(spectrogram, eda_data)

        arousal_loss = criterion(arousal_output, arousal_label)
        valence_loss = criterion(valence_output, valence_label)
        loss = arousal_loss + valence_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

print("Training finished.")