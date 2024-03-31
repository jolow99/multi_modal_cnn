import torch
import torch.nn as nn
import torch.optim as optim
from loader.data_loader import PMEmoDataset
import torch.utils.data as data_utils
from model import spectroedanet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

root_dir = "dataset"
dataset = PMEmoDataset(root_dir)

# Instantiate the model
model = spectroedanet.SpectroEDANet("Arousal")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
device = torch.device("mps")
model.to(device)

# Define the number of folds for cross-validation
num_folds = 5

# Create a KFold object
kfold = KFold(n_splits=num_folds, shuffle=True)

# Iterate over the folds
for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    print(f"Fold {fold + 1}")

    # Create data loaders for the current fold
    train_sampler = data_utils.SubsetRandomSampler(train_idx)
    val_sampler = data_utils.SubsetRandomSampler(val_idx)
    train_loader = data_utils.DataLoader(dataset, batch_size=32, sampler=train_sampler)
    val_loader = data_utils.DataLoader(dataset, batch_size=32, sampler=val_sampler)

    # Reset the model weights
    model.apply(lambda m: isinstance(m, nn.Linear) and m.reset_parameters())

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            spectrogram, eda_data, arousal_label, valence_label = data
            spectrogram = spectrogram.to(device)
            eda_data = eda_data.to(device)
            arousal_label = arousal_label.to(device)
            valence_label = valence_label.to(device)

            optimizer.zero_grad()

            output = model(spectrogram, eda_data)

            # arousal_output, valence_output = model(spectrogram, eda_data)

            # arousal_loss = criterion(arousal_output, arousal_label)
            # valence_loss = criterion(valence_output, valence_label)
            # loss = arousal_loss + valence_loss

            loss = criterion(output, arousal_label if model.isArousal else valence_label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Evaluate on the validation set
        model.eval()
        val_loss = 0.0
        preds = []
        labels = []

        # arousal_preds = []
        # valence_preds = []
        # arousal_labels = []
        # valence_labels = []
        with torch.no_grad():
            for data in val_loader:
                spectrogram, eda_data, arousal_label, valence_label = data
                spectrogram = spectrogram.to(device)
                eda_data = eda_data.to(device)
                arousal_label = arousal_label.to(device)
                valence_label = valence_label.to(device)

                output = model(spectrogram, eda_data)

                # arousal_output, valence_output = model(spectrogram, eda_data)

                # arousal_loss = criterion(arousal_output, arousal_label)
                # valence_loss = criterion(valence_output, valence_label)
                # val_loss += arousal_loss.item() + valence_loss.item()

                val_loss = criterion(output, arousal_label if model.isArousal else valence_label)
                preds.extend(output.cpu().numpy() if model.isArousal else output.cpu().numpy())
                labels.extend(arousal_label.cpu().numpy() if model.isArousal else valence_label.cpu().numpy())

                # arousal_preds.extend(arousal_output.cpu().numpy())
                # valence_preds.extend(valence_output.cpu().numpy())
                # arousal_labels.extend(arousal_label.cpu().numpy())
                # valence_labels.extend(valence_label.cpu().numpy())

        # Calculate evaluation metrics
                mse = mean_squared_error(labels, preds)
                r2 = r2_score(labels, preds)
        # arousal_mse = mean_squared_error(arousal_labels, arousal_preds)
        #valence_mse = mean_squared_error(valence_labels, valence_preds)
        # arousal_r2 = r2_score(arousal_labels, arousal_preds)
        # valence_r2 = r2_score(valence_labels, valence_preds)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, "
                f"MSE: {mse:.4f}, "
                f"R2: {r2:.4f}"
              # f"Arousal MSE: {arousal_mse:.4f}, "
              # f"Valence MSE: {valence_mse:.4f}, "
              # f"Arousal R2: {arousal_r2:.4f}, "
              # f"Valence R2: {valence_r2:.4f}")
        )

print("Training finished.")