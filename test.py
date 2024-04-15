import json
from model import spectroedanet
import torch
import torch.utils.data as torch_data
import torch.nn as nn
from loader.data_loader import PMEmoDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

def unpack_data(data, device: torch.device):
    spectrogram, eda_data, arousal_label, valence_label, music_vector = data
    spectrogram = spectrogram.to(device)
    eda_data = eda_data.to(device)
    arousal_label = arousal_label.to(device)
    valence_label = valence_label.to(device)
    music_vector = music_vector.to(device=device, dtype=torch.float32)
    return spectrogram, eda_data, arousal_label, valence_label, music_vector

def evaluate_model(model_run):
    print("Evaluating Model: ", model_run)
    flags = model_run.split('_')

    usesSpectrogram = False
    usesEDA = False
    usesMusic = False
    predictsArousal = False
    predictsValence = False

    for flag in flags:
        if flag == 'usesSpectrogram':
            usesSpectrogram = True
        elif flag == 'usesEDA':
            usesEDA = True
        elif flag == 'usesMusic':
            usesMusic = True
        elif flag == 'predictsArousal':
            predictsArousal = True
        elif flag == 'predictsValence':
            predictsValence = True

    model = spectroedanet.SpectroEDANet(usesSpectrogram,
                                        usesEDA,
                                        usesMusic,
                                        predictsArousal,
                                        predictsValence)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load the weights from the checkpoints folder. The file is named model_run.pth
    model.load_state_dict(torch.load(f'checkpoints/{model_run}'))

    # Evaluate the model on the test set 
    dataset = PMEmoDataset("dataset")
    train_val_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    test_loader = torch_data.DataLoader(test_dataset, batch_size=16)
    model.eval()

    test_loss = 0.0
    arousal_preds = []
    valence_preds = []
    arousal_labels = []
    valence_labels = []

    criterion = nn.MSELoss()

    with torch.no_grad():
        for data in test_loader:
            spectrogram, eda_data, arousal_label, valence_label, music_vector = unpack_data(data, device)

            if model.predictsArousal and model.predictsValence:
                arousal_output, valence_output = model(spectrogram, eda_data, music_vector)
                arousal_loss = criterion(arousal_output, arousal_label)
                valence_loss = criterion(valence_output, valence_label)
                test_loss += arousal_loss.item() + valence_loss.item()
                arousal_preds.extend(arousal_output.cpu().numpy())
                valence_preds.extend(valence_output.cpu().numpy())
                arousal_labels.extend(arousal_label.cpu().numpy())
                valence_labels.extend(valence_label.cpu().numpy())

            elif model.predictsArousal:
                output = model(spectrogram, eda_data, music_vector)
                test_loss = criterion(output, arousal_label)
                arousal_preds.extend(output.cpu().numpy())
                arousal_labels.extend(arousal_label.cpu().numpy())

            elif model.predictsValence:
                output = model(spectrogram, eda_data, music_vector)
                test_loss = criterion(output, valence_label)
                valence_preds.extend(output.cpu().numpy())
                valence_labels.extend(valence_label.cpu().numpy())

        # Calculate and print test metrics
        if model.predictsArousal and model.predictsValence:
            test_arousal_rmse = root_mean_squared_error(arousal_labels, arousal_preds)
            test_valence_rmse = root_mean_squared_error(valence_labels, valence_preds)
            test_arousal_r2 = r2_score(arousal_labels, arousal_preds)
            test_valence_r2 = r2_score(valence_labels, valence_preds)
            print(f"Test Loss: {test_loss / len(test_loader):.4f}, "
                f"Arousal RMSE: {test_arousal_rmse:.4f}, "
                f"Valence RMSE: {test_valence_rmse:.4f}, "
                f"Arousal R2: {test_arousal_r2:.4f}, "
                f"Valence R2: {test_valence_r2:.4f}")

        elif model.predictsArousal:
            test_arousal_rmse = root_mean_squared_error(arousal_labels, arousal_preds)
            test_arousal_r2 = r2_score(arousal_labels, arousal_preds)
            print(f"Test Loss: {test_loss / len(test_loader):.4f}, "
                f"Arousal RMSE: {test_arousal_rmse:.4f}, "
                f"Arousal R2: {test_arousal_r2:.4f}")

        elif model.predictsValence:
            test_valence_rmse = root_mean_squared_error(valence_labels, valence_preds)
            test_valence_r2 = r2_score(valence_labels, valence_preds)
            print(f"Test Loss: {test_loss / len(test_loader):.4f}, "
                f"Valence RMSE: {test_valence_rmse:.4f}, "
                f"Valence R2: {test_valence_r2:.4f}")


def test():
    model_runs = json.load(open('final_models.json'))
    for model_run in model_runs: 
        evaluate_model(model_run)

if __name__ == "__main__":
    test()