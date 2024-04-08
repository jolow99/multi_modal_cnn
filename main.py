import torch
import torch.nn as nn
import torch.optim as optim
from loader.data_loader import PMEmoDataset
import torch.utils.data as torchData
from model import spectroedanet
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, r2_score

root_dir = "dataset"
dataset = PMEmoDataset(root_dir)

# Instantiate the model
usesSpectrogram = True
usesEDA = True
usesMusic = True
usesAttention = True
predictsArousal = True
predictsValence = True

# Ensure that predictArousal and predictValence are not both False
assert predictsArousal or predictsValence
# Ensure that at least one of usesSpectrogram, usesEDA, and usesMusic is True
assert usesSpectrogram or usesEDA or usesMusic

model = spectroedanet.SpectroEDANet(usesSpectrogram, usesEDA, usesMusic, usesAttention, predictsArousal, predictsValence)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the number of folds for cross-validation
num_folds = 5

batch_size = 16

# Create a KFold object
kfold = KFold(n_splits=num_folds, shuffle=True)


def unpack_data(data):
    spectrogram, eda_data, arousal_label, valence_label, music_vector = data
    spectrogram = spectrogram.to(device)
    eda_data = eda_data.to(device)
    arousal_label = arousal_label.to(device)
    valence_label = valence_label.to(device)
    music_vector = music_vector.to(device=device, dtype=torch.float32)
    return spectrogram, eda_data, arousal_label, valence_label, music_vector


# Iterate over the folds
for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    print(f"Fold {fold + 1}")

    # Create data loaders for the current fold
    train_sampler = torchData.SubsetRandomSampler(train_idx)
    val_sampler = torchData.SubsetRandomSampler(val_idx)
    train_loader = torchData.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = torchData.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    # Reset the model weights
    model.apply(lambda m: isinstance(m, nn.Linear) and m.reset_parameters())

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            spectrogram, eda_data, arousal_label, valence_label, music_vector = unpack_data(data)

            optimizer.zero_grad()

            # # Print shapes of all inputs
            # print("--------------")
            # print(f"Spectrogram shape: {spectrogram.shape}")
            # print(f"EDA data shape: {eda_data.shape}")
            # print(f"Arousal label shape: {arousal_label.shape}")
            # print(f"Valence label shape: {valence_label.shape}")
            # print(f"Music vector shape: {music_vector.shape}")

            output = model(spectrogram, eda_data, music_vector)

            if model.predictsArousal and model.predictsValence:
                arousal_loss = criterion(output[0], arousal_label)
                valence_loss = criterion(output[1], valence_label)
                loss = arousal_loss + valence_loss
            elif model.predictsArousal:
                loss = criterion(output, arousal_label)
            elif model.predictsValence:
                loss = criterion(output, valence_label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Evaluate on the validation set
        model.eval()
        val_loss = 0.0
        arousal_preds = []
        valence_preds = []
        arousal_labels = []
        valence_labels = []

        with torch.no_grad():
            for data in val_loader:
                spectrogram, eda_data, arousal_label, valence_label, music_vector = unpack_data(data)

                if model.predictsArousal and model.predictsValence:
                    arousal_output, valence_output = model(spectrogram, eda_data, music_vector)
                    arousal_loss = criterion(arousal_output, arousal_label)
                    valence_loss = criterion(valence_output, valence_label)
                    val_loss += arousal_loss.item() + valence_loss.item()
                    arousal_preds.extend(arousal_output.cpu().numpy())
                    valence_preds.extend(valence_output.cpu().numpy())
                    arousal_labels.extend(arousal_label.cpu().numpy())
                    valence_labels.extend(valence_label.cpu().numpy())

                elif model.predictsArousal:
                    output = model(spectrogram, eda_data, music_vector)
                    val_loss = criterion(output, arousal_label)
                    arousal_preds.extend(output.cpu().numpy())
                    arousal_labels.extend(arousal_label.cpu().numpy())

                elif model.predictsValence:
                    output = model(spectrogram, eda_data, music_vector)
                    val_loss = criterion(output, valence_label)
                    valence_preds.extend(output.cpu().numpy())
                    valence_labels.extend(valence_label.cpu().numpy())

            # Calculate and print losses and metrics
            if model.predictsArousal and model.predictsValence:
                arousal_mse = root_mean_squared_error(arousal_labels, arousal_preds)
                valence_mse = root_mean_squared_error(valence_labels, valence_preds)
                arousal_r2 = r2_score(arousal_labels, arousal_preds)
                valence_r2 = r2_score(valence_labels, valence_preds)

                print(f"Epoch [{epoch + 1}/{num_epochs}], "
                      f"Train Loss: {running_loss / len(train_loader):.4f}, "
                      f"Val Loss: {val_loss / len(val_loader):.4f}, "
                      f"Arousal RMSE: {arousal_mse:.4f}, "
                      f"Valence RMSE: {valence_mse:.4f}, "
                      f"Arousal R2: {arousal_r2:.4f}, "
                      f"Valence R2: {valence_r2:.4f}")

            elif model.predictsArousal:
                mse = root_mean_squared_error(arousal_labels, arousal_preds)
                r2 = r2_score(arousal_labels, arousal_preds)
                print(f"Epoch [{epoch + 1}/{num_epochs}], "
                      f"Train Loss: {running_loss / len(train_loader):.4f}, "
                      f"Val Loss: {val_loss / len(val_loader):.4f}, "
                      f"Arousal RMSE: {mse:.4f}, "
                      f"Arousal R2: {r2:.4f}")

            elif model.predictsValence:
                mse = root_mean_squared_error(valence_labels, valence_preds)
                r2 = r2_score(valence_labels, valence_preds)
                print(f"Epoch [{epoch + 1}/{num_epochs}], "
                      f"Train Loss: {running_loss / len(train_loader):.4f}, "
                      f"Val Loss: {val_loss / len(val_loader):.4f}, "
                      f"Valence RMSE: {mse:.4f}, "
                      f"Valence R2: {r2:.4f}")

print("Training finished.")
