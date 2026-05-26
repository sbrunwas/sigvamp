import sounddevice as sd
import numpy as np

samplerate = 48000
duration = 3

# Explicitly use SP404 device
sd.default.device = 1

print("Recording...")
audio = sd.rec(
    int(duration * samplerate),
    samplerate=samplerate,
    channels=2,
    dtype='float32'
)

sd.wait()

print("Playing back...")
sd.play(audio, samplerate)
sd.wait()

print("Done.")
