import subprocess
import pandas as pd
import os
import arff


def get_music_features(wav_dir, dist_file, opensmile_dir):
    # extract static features of all wavs and load into 1 file
    SMILExtract = os.path.join(opensmile_dir, "build", "progsrc", "smilextract", "SMILExtract")
    config_file = os.path.join(opensmile_dir, "config", "is09-13", "IS13_ComParE.conf")

    if os.path.exists(dist_file):
        os.remove(dist_file)

    wav = [f for f in os.listdir(wav_dir) if f[-4:] == ".wav"]
    for i, w in enumerate(wav):
        wavpath = os.path.join(wav_dir, w)
        subprocess.check_call([SMILExtract, "-C", config_file, "-I", wavpath, "-O", dist_file, "-instname", w])
        print(f"completed {i + 1}/{len(wav)} wav files")


def wav_to_features(normalize=True):
    static_features_file = "static_features.arff"
    # get_music_features("/Users/joel-tay/Desktop/multi_modal_cnn/dataset/wav",
    #                    static_features_file,
    #                    "/Users/joel-tay/Documents/opensmile")
    res = arff.load(open(static_features_file, "r"))
    data = res['data']
    cols = list(map(lambda t: t[0], res['attributes']))
    df = pd.DataFrame(data, columns=cols)
    # exclude last col "class", not relevant from opensmile
    df = df.drop(columns=['class'])
    df['name'] = df['name'].map(lambda s: int(s[:-4]))  # remove .wav and cast to int
    df.rename(columns={'name': 'musicId'}, inplace=True)
    df.set_index("musicId", inplace=True)

    if normalize:
        # do z-score normalization
        mean = df.mean()
        std = df.std()
        mean.to_csv('features_mean.csv', header=None)
        std.to_csv('features_std.csv', header=None)
        df = (df - mean) / std

    annotations_dir = '/Users/joel-tay/Desktop/multi_modal_cnn/dataset/annotations'
    music_ids = []
    arousal_targets = []
    valence_targets = []
    for music_id, _ in df.iterrows():
        arousal_file = os.path.join(annotations_dir, "Arousal", f"{music_id}-A.csv")
        valence_file = os.path.join(annotations_dir, "Valence", f"{music_id}-V.csv")
        arousal_labels = []
        valence_labels = []
        subject_ids = []

        try:
            with open(arousal_file, 'r') as file:
                next(file)
                for line in file:
                    values = line.split(',')
                    subject_ids.append(values[0])
                    arousal_labels.append(float(values[1]))
        except FileNotFoundError:
            print(f'skipping arousal file for music id {music_id}')

        try:
            with open(valence_file, 'r') as file:
                next(file)
                for line in file:
                    values = line.split(',')
                    valence_labels.append(float(values[1]))
        except FileNotFoundError:
            print(f'skipping valence file for music id {music_id}')

        # If there are more than 10 subjects, only take the first 10
        if len(subject_ids) > 10:
            arousal_labels = arousal_labels[:10]
            valence_labels = valence_labels[:10]

        # calculate mean as label for one music id
        music_ids.append(music_id)
        arousal = sum(arousal_labels) / len(arousal_labels) if len(arousal_labels) > 0 else 0
        valence = sum(valence_labels) / len(valence_labels) if len(valence_labels) > 0 else 0
        arousal_targets.append(arousal)
        valence_targets.append(valence)

    targets_df = pd.DataFrame(data={'target_arousal': arousal_targets,
                                    'target_valence': valence_targets,
                                    'musicId': music_ids})
    targets_df.set_index('musicId', inplace=True)
    joint_df = df.join(targets_df)

    joint_df.to_csv("/Users/joel-tay/Desktop/multi_modal_cnn/dataset/static_features.csv", index=True)


if __name__ == "__main__":
    wav_to_features()
