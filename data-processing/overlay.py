import os

from pydub import AudioSegment

stem2_path = "stems/stem_1"
stem3_path = "stems/stem_2"
stem4_path = "stems/stem_3"
out_path = "stems_wo_vocals"

for file in os.listdir(stem2_path):
    if file.endswith(".wav"):
        sound1 = AudioSegment.from_file(os.path.join(stem2_path, file), format="wav")
        sound2 = AudioSegment.from_file(os.path.join(stem3_path, file.replace("_stem_1", "_stem_2")), format="wav")
        sound3 = AudioSegment.from_file(os.path.join(stem4_path, file.replace("_stem_1", "_stem_3")), format="wav")
        overlay = sound1.overlay(sound2, position=0)
        overlay = overlay.overlay(sound3, position=0)
        overlay.export(os.path.join(out_path, file), format="wav")
        print(f"Overlayed: {file}")
