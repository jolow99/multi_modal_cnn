from main import main
import itertools
import matplotlib.pyplot as plt
from model import spectroedanet
from copy import deepcopy
import pandas as pd
import plotly.express as px
from loader.data_loader import PMEmoDataset

options = ['usesSpectrogram', 'usesEDA', 'usesMusic', 'usesAttention']
combinations = list(itertools.product([False, True], repeat=len(options)))
flag_combinations = [{flag: value for flag, value in zip(options, combination)} for combination in combinations]
dataset = PMEmoDataset("dataset")

def plot_losses(avg_train_losses, avg_val_losses, model: spectroedanet.SpectroEDANet, flags):
    print('avg_train_losses', len(avg_train_losses))
    print('avg_val_losses', len(avg_val_losses))
    plt.plot(avg_train_losses, label='Average Training Loss')
    plt.plot(avg_val_losses, '--', label='Average Validation Loss')
    plt.suptitle('Average Training and Validation Loss')
    plt.title(f'{str(model)}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    filename = 'plots/losses_'
    if 'usesSpectrogram' in flags and flags['usesSpectrogram']:
        filename += 'usesSpectrogram_'
    if 'usesEDA' in flags and flags['usesEDA']:
        filename += 'usesEDA_'
    if 'usesMusic' in flags and flags['usesMusic']:
        filename += 'usesMusic_'
    if 'usesAttention' in flags and flags['usesAttention']:
        filename += 'usesAttention_'
    filename += '.png'
    plt.savefig(filename)


# we have 2^3 = 8 combinations, but only 7 will run because the 1 case where all is false is invalid
# and at least 1 kind of feature must be used
def compare():
    data_flags = {option: [] for option in options}
    arousal_target = {'arousal_r2': []}
    valence_target = {'valence_r2': []}
    arousal_data = deepcopy({k: v for d in (data_flags, arousal_target) for k, v in d.items()})
    valence_data = deepcopy({k: v for d in (data_flags, valence_target) for k, v in d.items()})
    for flags in flag_combinations:
        print(flags)
        res = main(**flags, dataset=dataset)
        if res is None:
            continue

        (model,
         best_model_weights,
         avg_train_losses,
         avg_val_losses,
         best_arousal_r2,
         best_valence_r2,
         best_arousal_rmse,
         best_valence_rmse) = res

        # valid flags
        for flag, value in flags.items():
            if best_arousal_r2 is not None:
                arousal_data[flag].append(value)
            if best_valence_r2 is not None:
                valence_data[flag].append(value)

        if best_arousal_r2 is not None:
            arousal_data['arousal_r2'].append(best_arousal_r2)
        if best_valence_r2 is not None:
            valence_data['valence_r2'].append(best_valence_r2)

        plot_losses(avg_train_losses, avg_val_losses, model, flags)

    # draw parallel_coordinates
    arousal_df = pd.DataFrame(arousal_data)
    valence_df = pd.DataFrame(valence_data)

    for col in arousal_df.columns[:-1]:
        arousal_df[col] = arousal_df[col].astype(int)
    for col in valence_df.columns[:-1]:
        arousal_df[col] = arousal_df[col].astype(int)

    arousal_plot = px.parallel_coordinates(arousal_df, color='arousal_r2', color_continuous_scale=px.colors.diverging.Spectral)
    valence_plot = px.parallel_coordinates(valence_df, color='valence_r2', color_continuous_scale=px.colors.diverging.Spectral)
    arousal_plot.write_image('plots/arousal_r2_plot.png')
    valence_plot.write_image('plots/valence_r2_plot.png')


if __name__ == "__main__":
    compare()
