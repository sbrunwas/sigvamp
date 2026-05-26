import numpy as np
import sounddevice as sd
import mido

# ---------------------------------
# CONFIG
# ---------------------------------

SAMPLERATE = 48000
DURATION = 3

AUDIO_DEVICE = 1
MIDI_PORT_NAME = "Arturia MiniLab mkII MIDI 1"

START_CC = 112
END_CC = 114
VOLUME_CC = 74

ROOT_NOTE = 60

# ---------------------------------
# RECORD AUDIO
# ---------------------------------

print("Recording sample...")

audio = sd.rec(
    int(DURATION * SAMPLERATE),
    samplerate=SAMPLERATE,
    channels=2,
    dtype='float32',
    device=AUDIO_DEVICE
)

sd.wait()

print("Recording complete.")
print("Skipping normalization.")

# ---------------------------------
# PLAYBACK STATE
# ---------------------------------

loop_start = 0.0
loop_end = 1.0

# Slightly hotter default level
master_gain = 1.5

# ---------------------------------
# MIDI HELPERS
# ---------------------------------

def cc_relative_delta(value):

    if value == 64:
        return 0

    return value - 64

# ---------------------------------
# MIDI LOOP
# ---------------------------------

with mido.open_input(MIDI_PORT_NAME) as port:

    print("Ready.")
    print("Keys = playback")
    print("Knob 1 = loop start")
    print("Knob 2 = loop end")
    print("Knob 3 = gain")

    for msg in port:

        # -------------------------
        # NOTE ON
        # -------------------------

        if msg.type == "note_on" and msg.velocity > 0:

            start_idx = int(loop_start * len(audio))
            end_idx = int(loop_end * len(audio))

            # Prevent microscopic windows
            min_window = 1000

            if end_idx <= start_idx + min_window:
                end_idx = start_idx + min_window

            sample = audio[start_idx:end_idx]

            semitones = msg.note - ROOT_NOTE

            speed = 2 ** (semitones / 12)

            print(
                f"NOTE {msg.note} "
                f"SPEED {speed:.2f}"
            )

            # ---------------------------------
            # RESAMPLE FOR PITCH SHIFT
            # ---------------------------------

            indices = np.arange(
                0,
                len(sample),
                speed
            )

            indices = indices[
                indices < len(sample)
            ]

            pitched = sample[
                indices.astype(int)
            ]

            # ---------------------------------
            # APPLY GAIN
            # ---------------------------------

            pitched = pitched * master_gain

            # Prevent clipping
            pitched = np.clip(
                pitched,
                -1.0,
                1.0
            )

            # ---------------------------------
            # PLAY SAMPLE
            # ---------------------------------

            sd.play(
                pitched,
                samplerate=SAMPLERATE,
                device=AUDIO_DEVICE
            )

        # -------------------------
        # CONTROL CHANGES
        # -------------------------

        elif msg.type == "control_change":

            delta = cc_relative_delta(msg.value)

            # LOOP START
            if msg.control == START_CC:

                loop_start += delta * 0.01

                loop_start = float(
                    np.clip(
                        loop_start,
                        0.0,
                        0.95
                    )
                )

            # LOOP END
            elif msg.control == END_CC:

                loop_end += delta * 0.01

                loop_end = float(
                    np.clip(
                        loop_end,
                        0.05,
                        1.0
                    )
                )

            # GAIN
            elif msg.control == VOLUME_CC:

                master_gain += delta * 0.05

                master_gain = float(
                    np.clip(
                        master_gain,
                        0.0,
                        4.0
                    )
                )

            # Prevent invalid windows
            if loop_end <= loop_start + 0.02:
                loop_end = loop_start + 0.02

            print(
                f"START={loop_start:.3f} "
                f"END={loop_end:.3f} "
                f"GAIN={master_gain:.2f}"
            )
