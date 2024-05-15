import os

import numpy as np
import soundfile
from PIL import Image
import librosa
import matplotlib.pyplot as plt

#im=Image.open("real/0009fFIM1eYThaPg_0.tiff")
#im=Image.open("../train_results/spectr_at_epoch_02_step_020001.tiff")
#im= Image.open("input/bass_0.tiff")
#img=np.array(im)
#with np.load("real/0009fFIM1eYThaPg_0.npz") as data-processing:
i = 0
path = "../../2_square_drums_train_results"
path = "gen_accomp_musdb_test_results"
path = "../../2_square_drums_test_results"


out = "square_train_audio"
#out = "gen_accomp_test_audio"
#path = "../2_square_drums_train_results"
#path = "."
for file in os.listdir(path):
    f = os.path.join(path, file)
    if os.path.isfile(f) and f.endswith(".npz"):
        if "test" in path:
            nr = file.split(".")[0].split("_")[2]
            if int(nr) > 10:
                continue
            spectr = f'{out}/{nr}_{file.split(".")[0]}.png'
            sound = f"{out}/{nr}_{file.replace('.npz', '.wav')}"
        else:
            epoch = int(file.split("_")[3])

            spectr = f'{out}/{file.split(".")[0]}.png'
            sound = f"{out}/{file.replace('.npz', '.wav')}"

        if "trains" in path and os.path.isfile(spectr): #or "output" not in file:
            continue
        with np.load(f) as data:
            img = data['arr_0'].squeeze()
#print(file)
        print(img.max())
        print(img.min())

        #img = np.flip(img, axis=0).astype(np.float64)
        h = img.shape[0]
        h = h // 2

       # if "test" in path:
            #if "output" in file:
               #img = img[:-40, :]

               #for i in range(50):
               #    img[i][80:130] = np.full(50, -80)

        input_image = img
        #real_image = img[:h, :]
        #librosa.display.specshow(img)
        input_image = librosa.db_to_power(input_image, ref=50)
        #print(spectr.max())
        #print(spectr.min())
        plt.title("Input image")
        fig, ax = plt.subplots()
        # Getting the pixel values in the [0, 1] range to plot.
        imgs = librosa.display.specshow(img, ax=ax)
        fig.colorbar(imgs, ax=ax)
        plt.axis('off')
        plt.savefig(spectr)
        plt.close()
        wav=librosa.feature.inverse.mel_to_audio(input_image)
        #wav2=librosa.feature.inverse.mel_to_audio(real_image)
       # soundfile.write(f"drums_test_audio/{file.replace('.npz', '2.wav')}",wav2,samplerate=22050)
        soundfile.write(sound,wav,samplerate=22050)

# real_image = librosa.db_to_power(real_image, ref=300)
# #print(spectr.max())
# #print(spectr.min())
# wav=librosa.feature.inverse.mel_to_audio(real_image, fmax=9000, fmin=20)
    #soundfile.write("ckpt49.wav",wav,samplerate=22050)

