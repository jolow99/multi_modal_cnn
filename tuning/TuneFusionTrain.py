from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from TuneFusionDataset import PMEmoDataset
import torch.utils.data as torch_data
from TuneFusionModel import SpectroEDANet
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from copy import deepcopy
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
import os
import tempfile
import numpy as np

# root_dir = "dataset"
# dataset = PMEmoDataset(root_dir)

lr = 0.00001
batch_size = 16
# writer = SummaryWriter()  # hmm maybe this and the writer fx should be within the loop

def unpack_data(data, device: torch.device):
    spectrogram, eda_data, arousal_label, valence_label, music_vector = data
    spectrogram = spectrogram.to(device)
    eda_data = eda_data.to(device)
    arousal_label = arousal_label.to(device)
    valence_label = valence_label.to(device)
    music_vector = music_vector.to(device=device, dtype=torch.float32)
    return spectrogram, eda_data, arousal_label, valence_label, music_vector

def train_model(config=None):        
    usesSpectrogram=True,
    usesEDA=True,
    usesMusic=True,
    usesAttention=False,
    predictsArousal=True,
    predictsValence=True,
    dataset = PMEmoDataset("/home/abram/Developer/multi_modal_cnn/dataset")

    model = SpectroEDANet(usesSpectrogram,
                                        usesEDA,
                                        usesMusic,
                                        usesAttention,
                                        predictsArousal,
                                        predictsValence,
                                        dropout_p=config['dropout_p'],
                                        fc_size=256)

    # Split the dataset into training and testing sets
    train_val_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.2, random_state=42)

    # Create data loaders
    train_loader = torch_data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = torch_data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # LR scheduler
    # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=30)
    # scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.000001, max_lr=0.001,step_size_up=5,mode="triangular2")
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=10, epochs=50)
    lrs = []

    # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    # Training loop
    num_epochs = 50
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train
    # writer = SummaryWriter()
    # writer.add_hparams({'lr': config['lr'],
    #         'bsize': batch_size,
    #         'fcsize': config['fc_size'],
    #         'dropout': config['dropout_p']},
    #         {'lr': config['lr'],
    #         'bsize': batch_size,
    #         'fcsize': config['fc_size'],
    #         'dropout': config['dropout_p']})
    
    final_loss = None
    arousal_rmse = None
    valence_rmse = None
    arousal_r2 = None
    valence_r2 = None

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            spectrogram, eda_data, arousal_label, valence_label, music_vector = unpack_data(data, device)

            optimizer.zero_grad()

            output = model(spectrogram, eda_data, music_vector)

            if model.predictsArousal and model.predictsValence:
                arousal_loss = criterion(output[0], arousal_label)
                valence_loss = criterion(output[1], valence_label)
                loss = arousal_loss + valence_loss
            elif model.predictsArousal:
                loss = criterion(output, arousal_label)
            elif model.predictsValence:
                loss = criterion(output, valence_label)

            lrs.append(optimizer.param_groups[0]['lr'])

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Update learning rate
        scheduler.step()

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

            elif model.predictsArousal:
                arousal_rmse = root_mean_squared_error(arousal_labels, arousal_preds)
                arousal_r2 = r2_score(arousal_labels, arousal_preds)
                print(f"Epoch [{epoch + 1}/{num_epochs}], "
                        f"Train Loss: {running_loss / len(train_loader):.4f}, "
                        f"Val Loss: {val_loss / len(val_loader):.4f}, "
                        f"Arousal RMSE: {arousal_rmse:.4f}, "
                        f"Arousal R2: {arousal_r2:.4f}")

            elif model.predictsValence:
                valence_rmse = root_mean_squared_error(valence_labels, valence_preds)
                valence_r2 = r2_score(valence_labels, valence_preds)
                print(f"Epoch [{epoch + 1}/{num_epochs}], "
                        f"Train Loss: {running_loss / len(train_loader):.4f}, "
                        f"Val Loss: {val_loss / len(val_loader):.4f}, "
                        f"Valence RMSE: {valence_rmse:.4f}, "
                        f"Valence R2: {valence_r2:.4f}")

        loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        final_loss = val_loss

        # writer.add_scalar("loss x epoch", loss, epoch+1)
        # writer.add_scalar("arousal_rmse x epoch", arousal_rmse, epoch+1)
        # writer.add_scalar("valence_rmse x epoch", valence_rmse, epoch+1)
        # writer.add_scalar("arousal_r2 x epoch", arousal_r2, epoch+1)
        # writer.add_scalar("valence_r2 x epoch", valence_r2, epoch+1)

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"lr": lrs[-1], "arousal_r2": arousal_r2, 
                 "valence_r2": valence_r2, "loss": (final_loss), 
                 "arousal_rmse": arousal_rmse, "valence_rmse": valence_rmse,},
                checkpoint=checkpoint,
            )
    
    # writer.flush()
    # writer.close()

    # Save model weights
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # store the file in the checkpoints folder. the file name should use the boolean flags that are inputs to the main functions and end with the timestamp
    filename = '/home/abram/Developer/multi_modal_cnn/checkpoints/model_weights_'
    if usesSpectrogram:
        filename += 'usesSpectrogram_'
    if usesEDA:
        filename += 'usesEDA_'
    if usesMusic:
        filename += 'usesMusic_'
    if usesAttention:
        filename += 'usesAttention_'
    if predictsArousal:
        filename += 'predictsArousal_'
    if predictsValence:
        filename += 'predictsValence_'
    # filename += f'{timestamp}.pt'
    filename += '.pt'

    print('Saving best model...')
    torch.save(model.state_dict(), filename)





if __name__ == '__main__':    
    num_samples = 10
    max_num_epochs = 100
    gpus_per_trial = 1

    config = {
    "dropout_p": tune.uniform(0.0, 0.5),
    # "lr": tune.loguniform(1e-6, 1e-2),
    # "fc_size": tune.sample_from(lambda _: 2**np.random.randint(5, 10)),
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model),
            resources={"cpu": 20, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="arousal_r2",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result("arousal_r2", "max")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial arousal_r2: {}".format(
        best_result.metrics["arousal_r2"]))