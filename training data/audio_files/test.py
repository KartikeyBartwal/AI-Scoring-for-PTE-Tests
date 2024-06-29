import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

# Or if you are using > Python 3.11:
with warnings.catch_warnings(action="ignore"):
    fxn()

import librosa

path = "audio_ff0c4829-52a7-429b-8a2d-3d76e68d2d50.m4a"
input_audio, sample_rate = librosa.load(path,  sr=16000)
print(type(input_audio))
print("WORKING")