import sys
import os

from spleeter.separator import Separator

from spleeter.audio.adapter import AudioAdapter

#
separator = Separator('spleeter:4stems')
audio_adapter = AudioAdapter.default()
sample_rate = 22050


audio_path = sys.argv[1]
output_path = sys.argv[2]
for filename in os.listdir(audio_path):
    f = os.path.join(audio_path, filename)
    # checking if it is a file
    if os.path.isfile(f) and "08" in f and f.endswith(".wav"):

        save_path = os.path.join(output_path, filename.replace(".wav", "_drums.wav"))
        if not os.path.isfile(save_path):
            waveform, _ = audio_adapter.load(f, sample_rate=sample_rate)
        # Perform the separation :
            prediction = separator.separate(waveform)

            audio_adapter.save(save_path, prediction['drums'], sample_rate=sample_rate, codec='wav', bitrate='128k')
