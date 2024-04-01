import os
from pydub import AudioSegment

input_dir = '../PMEmo2019/chorus'
output_dir = '../dataset/wav'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Note: ffmpeg needs to be on your machine to run this
def mp3_to_wav():
    for filename in os.listdir(input_dir):
        if filename.endswith('.mp3'):
            file_path = os.path.join(input_dir, filename)
            audio = AudioSegment.from_mp3(file_path)
            dest_path = os.path.join(output_dir, filename[:-4] + ".wav")
            audio.export(dest_path, format="wav")


if __name__ == '__main__':
    mp3_to_wav()
