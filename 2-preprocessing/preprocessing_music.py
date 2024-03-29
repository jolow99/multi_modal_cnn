import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Set the directories
input_dir = '../PMEmo2019/chorus'
output_dir = '../PMEmo2019/spectrograms'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.mp3'):
        # Load the audio file
        audio_path = os.path.join(input_dir, filename)
        y, sr = librosa.load(audio_path)

        # Convert to Mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

        # Convert to log-scaled Mel-spectrogram
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Save the Mel-spectrogram as an image
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
        librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

print('Mel-spectrograms saved to', output_dir)