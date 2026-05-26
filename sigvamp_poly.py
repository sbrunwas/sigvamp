import threading

import mido
import numpy as np
import sounddevice as sd

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
ATTACK_CC = 18
DECAY_CC = 19

ROOT_NOTE = 60

MAX_VOICES = 8
BLOCKSIZE = 512

ATTACK_SENSITIVITY = 0.5
DECAY_SENSITIVITY = 2.0

DEFAULT_ATTACK_MS = 5.0
DEFAULT_DECAY_MS = 20.0

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

attack_ms = DEFAULT_ATTACK_MS
decay_ms = DEFAULT_DECAY_MS

voices = []
voices_lock = threading.Lock()

# ---------------------------------
# MIDI HELPERS
# ---------------------------------


def cc_relative_delta(value):

    if value == 64:
        return 0

    return value - 64


def apply_envelope(buffer, attack_ms, decay_ms):

    shaped = buffer.copy()
    total_frames = len(shaped)

    if total_frames == 0:
        return shaped

    attack_frames = int((attack_ms / 1000) * SAMPLERATE)
    decay_frames = int((decay_ms / 1000) * SAMPLERATE)

    attack_frames = min(attack_frames, total_frames)
    decay_frames = min(decay_frames, total_frames)

    if attack_frames > 1:
        attack = np.linspace(
            0.0,
            1.0,
            attack_frames,
            dtype=np.float32
        )
        shaped[:attack_frames] *= attack[:, None]

    if decay_frames > 1:
        decay = np.linspace(
            1.0,
            0.0,
            decay_frames,
            dtype=np.float32
        )
        shaped[-decay_frames:] *= decay[:, None]

    return shaped


def build_voice(note):

    start_idx = int(loop_start * len(audio))
    end_idx = int(loop_end * len(audio))

    # Prevent microscopic windows
    min_window = 1000

    if end_idx <= start_idx + min_window:
        end_idx = start_idx + min_window

    sample = audio[start_idx:end_idx]

    semitones = note - ROOT_NOTE
    speed = 2 ** (semitones / 12)

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

    pitched = pitched * master_gain

    pitched = apply_envelope(
        pitched,
        attack_ms,
        decay_ms
    )

    pitched = np.clip(
        pitched,
        -1.0,
        1.0
    )

    return {
        "note": note,
        "buffer": pitched,
        "position": 0
    }, speed


def add_voice(voice):

    with voices_lock:
        voices.append(voice)

        while len(voices) > MAX_VOICES:
            voices.pop(0)


# ---------------------------------
# AUDIO CALLBACK
# ---------------------------------


def audio_callback(outdata, frames, time, status):

    if status:
        print(status)

    outdata.fill(0.0)

    with voices_lock:
        active_voices = []

        for voice in voices:
            buffer = voice["buffer"]
            position = voice["position"]
            remaining = len(buffer) - position

            if remaining <= 0:
                continue

            frames_to_copy = min(frames, remaining)

            outdata[:frames_to_copy] += buffer[
                position:position + frames_to_copy
            ]

            voice["position"] = position + frames_to_copy

            if voice["position"] < len(buffer):
                active_voices.append(voice)

        voices[:] = active_voices

    np.clip(
        outdata,
        -1.0,
        1.0,
        out=outdata
    )


# ---------------------------------
# MIDI LOOP
# ---------------------------------

with sd.OutputStream(
    samplerate=SAMPLERATE,
    channels=2,
    dtype='float32',
    device=AUDIO_DEVICE,
    blocksize=BLOCKSIZE,
    callback=audio_callback
):

    with mido.open_input(MIDI_PORT_NAME) as port:

        print("Ready.")
        print("Polyphonic test mode")
        print("Keys = playback")
        print("Knob 1 = loop start")
        print("Knob 2 = loop end")
        print("Knob 3 = gain")
        print("CC18 = attack")
        print("CC19 = decay")

        for msg in port:

            # -------------------------
            # NOTE ON
            # -------------------------

            if msg.type == "note_on" and msg.velocity > 0:

                voice, speed = build_voice(msg.note)
                add_voice(voice)

                print(
                    f"NOTE {msg.note} "
                    f"SPEED {speed:.2f} "
                    f"VOICES={len(voices)}"
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

                    loop_end += delta * 0.002

                    loop_end = float(
                        np.clip(
                            loop_end,
                            0.05,
                            1.0
                        )
                    )

                # GAIN
                elif msg.control == VOLUME_CC:

                    master_gain += delta * 0.01

                    master_gain = float(
                        np.clip(
                            master_gain,
                            0.0,
                            4.0
                        )
                    )

                # ATTACK
                elif msg.control == ATTACK_CC:

                    attack_ms += delta * ATTACK_SENSITIVITY

                    attack_ms = float(
                        np.clip(
                            attack_ms,
                            0.0,
                            500.0
                        )
                    )

                # DECAY
                elif msg.control == DECAY_CC:

                    decay_ms += delta * DECAY_SENSITIVITY

                    decay_ms = float(
                        np.clip(
                            decay_ms,
                            0.0,
                            2000.0
                        )
                    )

                # Prevent invalid windows
                if loop_end <= loop_start + 0.02:
                    loop_end = loop_start + 0.02

                print(
                    f"START={loop_start:.3f} "
                    f"END={loop_end:.3f} "
                    f"GAIN={master_gain:.2f} "
                    f"ATTACK={attack_ms:.1f}ms "
                    f"DECAY={decay_ms:.1f}ms"
                )
