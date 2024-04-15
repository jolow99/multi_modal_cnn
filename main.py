from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from loader.data_loader import PMEmoDataset
import torch.utils.data as torch_data
from model import spectroedanet
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from copy import deepcopy
from datetime import datetime

root_dir = "dataset"
dataset = PMEmoDataset(root_dir)


def unpack_data(data, device: torch.device):
    spectrogram, eda_data, arousal_label, valence_label, music_vector = data
    spectrogram = spectrogram.to(device)
    eda_data = eda_data.to(device)
    arousal_label = arousal_label.to(device)
    valence_label = valence_label.to(device)
    music_vector = music_vector.to(device=device, dtype=torch.float32)
    return spectrogram, eda_data, arousal_label, valence_label, music_vector


# Instantiate the model
def main(usesSpectrogram=True,
         usesEDA=False,
         usesMusic=False,
         usesAttention = True,
         predictsArousal=True,
         predictsValence=True) -> (
        tuple[
            SpectroEdaMusicNet,
            dict[str, Any],
            list[float],
            list[float],
            float | None,
            float | None,
            float | None,
            float | None,]
        | None
):
    # Ensure that predictArousal and predictValence are not both False
    is_predicting = predictsArousal or predictsValence
    # Ensure that at least one of usesSpectrogram, usesEDA, and usesMusic is True
    is_using_features = usesSpectrogram or usesEDA or usesMusic
    if (not is_predicting) or (not is_using_features):
        # if not predicting or not using any features, pass
        return

    model = spectroedanet.SpectroEDANet(usesSpectrogram,
                                        usesEDA,
                                        usesMusic,
                                        usesAttention,
                                        predictsArousal,
                                        predictsValence)
    
    # Split the dataset into training and testing sets
    train_val_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 2
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the number of folds for cross-validation
    num_folds = 2

    batch_size = 2

    # Create a KFold object
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # best model params
    best_val_loss = float('inf')
    best_model_weights = deepcopy(model.state_dict())
    best_arousal_r2: float | None = None
    best_valence_r2: float | None = None
    best_arousal_rmse: float | None = None
    best_valence_rmse: float | None = None
    trials = 0 # in this case trials is controlled model-level, not within each k-fold
    patience = 15
    best_found = False
    train_losses = {i: [] for i in range(num_folds)}
    val_losses = {i: [] for i in range(num_folds)}

    # Iterate over the folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_dataset)):
        if best_found:
            break

        print(f"Fold {fold + 1}")

        # Create data loaders for the current fold
        train_sampler = torch_data.SubsetRandomSampler(train_idx)
        val_sampler = torch_data.SubsetRandomSampler(val_idx)
        train_loader = torch_data.DataLoader(train_val_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch_data.DataLoader(train_val_dataset, batch_size=batch_size, sampler=val_sampler)

        # Reset the model weights
        model.apply(lambda m: isinstance(m, nn.Linear) and m.reset_parameters())

        for epoch in range(num_epochs):
            running_loss = 0.0
            model.train()
            for i, data in enumerate(train_loader, 0):
                spectrogram, eda_data, arousal_label, valence_label, music_vector = unpack_data(data, device)

                optimizer.zero_grad()

                output = model(spectrogram, eda_data, music_vector)

                # if model.predictsArousal and model.predictsValence:
                #     arousal_loss = criterion(output[0], arousal_label)
                #     valence_loss = criterion(output[1], valence_label)
                #     loss = arousal_loss + valence_loss
                # elif model.predictsArousal:
                #     loss = criterion(output, arousal_label)
                # elif model.predictsValence:
                #     loss = criterion(output, valence_label)

                arousal_loss = criterion(output[0], arousal_label)
                valence_loss = criterion(output[1], valence_label)
                loss = arousal_loss + valence_loss

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
                    spectrogram, eda_data, arousal_label, valence_label, music_vector = unpack_data(data, device)

                    arousal_output, valence_output = model(spectrogram, eda_data, music_vector)
                    arousal_loss = criterion(arousal_output, arousal_label)
                    valence_loss = criterion(valence_output, valence_label)
                    val_loss += arousal_loss.item() + valence_loss.item()
                    arousal_preds.extend(arousal_output.cpu().numpy())
                    valence_preds.extend(valence_output.cpu().numpy())
                    arousal_labels.extend(arousal_label.cpu().numpy())
                    valence_labels.extend(valence_label.cpu().numpy())

                    # if model.predictsArousal and model.predictsValence:
                    #     arousal_output, valence_output = model(spectrogram, eda_data, music_vector)
                    #     arousal_loss = criterion(arousal_output, arousal_label)
                    #     valence_loss = criterion(valence_output, valence_label)
                    #     val_loss += arousal_loss.item() + valence_loss.item()
                    #     arousal_preds.extend(arousal_output.cpu().numpy())
                    #     valence_preds.extend(valence_output.cpu().numpy())
                    #     arousal_labels.extend(arousal_label.cpu().numpy())
                    #     valence_labels.extend(valence_label.cpu().numpy())
                    #
                    # elif model.predictsArousal:
                    #     output = model(spectrogram, eda_data, music_vector)
                    #     val_loss = criterion(output, arousal_label)
                    #     arousal_preds.extend(output.cpu().numpy())
                    #     arousal_labels.extend(arousal_label.cpu().numpy())
                    #
                    # elif model.predictsValence:
                    #     output = model(spectrogram, eda_data, music_vector)
                    #     val_loss = criterion(output, valence_label)
                    #     valence_preds.extend(output.cpu().numpy())
                    #     valence_labels.extend(valence_label.cpu().numpy())

                arousal_r2 = None
                valence_r2 = None
                # Calculate and print losses and metrics
                arousal_rmse = root_mean_squared_error(arousal_labels, arousal_preds)
                valence_rmse = root_mean_squared_error(valence_labels, valence_preds)
                arousal_r2 = r2_score(arousal_labels, arousal_preds)
                valence_r2 = r2_score(valence_labels, valence_preds)

                print(f"Epoch [{epoch + 1}/{num_epochs}], "
                      f"Train Loss: {running_loss / len(train_loader):.4f}, "
                      f"Val Loss: {val_loss / len(val_loader):.4f}, "
                      f"Arousal RMSE: {arousal_rmse:.4f}, "
                      f"Valence RMSE: {valence_rmse:.4f}, "
                      f"Arousal R2: {arousal_r2:.4f}, "
                      f"Valence R2: {valence_r2:.4f}")
                # if model.predictsArousal and model.predictsValence:
                #     arousal_rmse = root_mean_squared_error(arousal_labels, arousal_preds)
                #     valence_rmse = root_mean_squared_error(valence_labels, valence_preds)
                #     arousal_r2 = r2_score(arousal_labels, arousal_preds)
                #     valence_r2 = r2_score(valence_labels, valence_preds)
                #
                #     print(f"Epoch [{epoch + 1}/{num_epochs}], "
                #           f"Train Loss: {running_loss / len(train_loader):.4f}, "
                #           f"Val Loss: {val_loss / len(val_loader):.4f}, "
                #           f"Arousal RMSE: {arousal_rmse:.4f}, "
                #           f"Valence RMSE: {valence_rmse:.4f}, "
                #           f"Arousal R2: {arousal_r2:.4f}, "
                #           f"Valence R2: {valence_r2:.4f}")
                #
                # elif model.predictsArousal:
                #     arousal_rmse = root_mean_squared_error(arousal_labels, arousal_preds)
                #     arousal_r2 = r2_score(arousal_labels, arousal_preds)
                #     print(f"Epoch [{epoch + 1}/{num_epochs}], "
                #           f"Train Loss: {running_loss / len(train_loader):.4f}, "
                #           f"Val Loss: {val_loss / len(val_loader):.4f}, "
                #           f"Arousal RMSE: {arousal_rmse:.4f}, "
                #           f"Arousal R2: {arousal_r2:.4f}")
                #
                # elif model.predictsValence:
                #     valence_rmse = root_mean_squared_error(valence_labels, valence_preds)
                #     valence_r2 = r2_score(valence_labels, valence_preds)
                #     print(f"Epoch [{epoch + 1}/{num_epochs}], "
                #           f"Train Loss: {running_loss / len(train_loader):.4f}, "
                #           f"Val Loss: {val_loss / len(val_loader):.4f}, "
                #           f"Valence RMSE: {valence_rmse:.4f}, "
                #           f"Valence R2: {valence_r2:.4f}")

                if val_loss < best_val_loss:
                    # NOTE: here best_val_loss is stored as val_loss without dividing by its length
                    # it should however not have any impact as it is just a means to detect better model weights
                    best_val_loss = val_loss
                    best_model_weights = deepcopy(model.state_dict())
                    best_arousal_r2 = arousal_r2
                    best_arousal_rmse = arousal_rmse
                    est_valence_r2 = valence_r2
                    best_valence_rmse = valence_rmse
                    # if model.predictsArousal:
                    #     best_arousal_r2 = arousal_r2
                    #     best_arousal_rmse = arousal_rmse
                    # if model.predictsValence:
                    #     best_valence_r2 = valence_r2
                    #     best_valence_rmse = valence_rmse
                    trials = 0
                else:
                    trials += 1
                    if trials >= patience:
                        print("Early stopping triggered")
                        best_found = True
                        break

            # store losses for plotting
            train_losses[fold].append(running_loss / len(train_loader))
            val_losses[fold].append(val_loss / len(val_loader))

    # plot average losses (epoch-average, across folds)
    avg_train_losses = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in zip(*train_losses.values())]
    avg_val_losses = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in zip(*val_losses.values())]
    
    # Save model weights
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # store the file in the checkpoints folder. the file name should use the boolean flags that are inputs to the main functions and end with the timestamp
    filename = 'checkpoints/model_weights_'
    if usesSpectrogram:
        filename += 'usesSpectrogram_'
    if usesEDA:
        filename += 'usesEDA_'
    if usesMusic:
        filename += 'usesMusic_'
    if predictsArousal:
        filename += 'predictsArousal_'
    if predictsValence:
        filename += 'predictsValence_'
    filename += f'{timestamp}.pt'

    print('Saving best model...')
    torch.save(model.state_dict(), filename)

    # return best params and losses
    return (model,
            best_model_weights,
            avg_train_losses,
            avg_val_losses,
            best_arousal_r2,
            best_valence_r2,
            best_arousal_rmse,
            best_valence_rmse)


if __name__ == '__main__':
    res = main()
