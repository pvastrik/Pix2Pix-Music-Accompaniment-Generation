import sys
import os

import librosa
import numpy as np

def spectrogram_image2(y, sr, out, hop_length, n_mels):
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    np.savez(out, S_DB)

def main():
    hop_length = 512  # number of samples per time-step in spectrogram
    n_mels = 256  # number of bins in spectrogram. Height of image
    time_steps = 255  # number of time-steps. Width of image

    # load audio
    path = sys.argv[1]
    out_path = sys.argv[2]
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        # checking if it is a file
        if os.path.isfile(f):
            y, sr = librosa.load(f, sr=22050)

            # extract a fixed length window
            length_samples = time_steps * hop_length
            i = 0
            for start_sample in range(0, len(y), length_samples):

                out = os.path.join(out_path, f'{filename.split(".")[0]}_{i}.npz')
                i += 1
                if start_sample + length_samples > len(y):
                    break
                window = y[start_sample:start_sample + length_samples]
                # convert to PNG
                if not os.path.exists(out):

                    spectrogram_image2(window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels)
                    print('wrote file', out)

if __name__ == '__main__':
    main()
    #y, sr = librosa.load("stempeg/stempeg/output/The Easton Ellises - Falcon 69/Stem_0.wav", sr=22050)
    #spectrogram_image2(y, sr, "test4.npz", 512, 256)
    # hop_length = 512  # number of samples per time-step in spectrogram
    # n_mels = 256  # number of bins in spectrogram. Height of image
    # time_steps = 255  # number of time-steps. Width of image
    # filename = "Stem_0.wav"
    # out_path = "pls"
    # y, sr = librosa.load(filename, sr=22050)
    #
    # # extract a fixed length window
    # length_samples = time_steps * hop_length
    # i = 0
    # for start_sample in range(0, len(y), length_samples):
    #
    #     out = os.path.join(out_path, f'{filename.split(".")[0]}_{i}.npz')
    #     i += 1
    #     if start_sample + length_samples > len(y):
    #         break
    #     window = y[start_sample:start_sample + length_samples]
    #     # convert to PNG
    #     #if not os.path.exists(out):
    #     spectrogram_image2(window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels)
    #     print('wrote file', out)
    # print("done")

