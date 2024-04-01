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
        print(f"completed {i+1}/{len(wav)} wav files")


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
        df = (df-df.mean())/df.std()

    df.to_csv("/Users/joel-tay/Desktop/multi_modal_cnn/dataset/static_features.csv", index=True)


if __name__ == "__main__":
    wav_to_features()
