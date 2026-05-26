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
RELEASE_CC = 19
ALT_RELEASE_CC = 16

ROOT_NOTE = 60

MAX_VOICES = 8
BLOCKSIZE = 512

ATTACK_SENSITIVITY = 0.5
RELEASE_SENSITIVITY = 2.0

DEFAULT_ATTACK_MS = 5.0
DEFAULT_RELEASE_MS = 80.0

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
release_ms = DEFAULT_RELEASE_MS

voices = []
voices_lock = threading.Lock()

# ---------------------------------
# MIDI HELPERS
# ---------------------------------


def cc_relative_delta(value):

    if value == 64:
        return 0

    return value - 64


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

    pitched = np.clip(
        pitched,
        -1.0,
        1.0
    )

    attack_frames = int((attack_ms / 1000) * SAMPLERATE)
    release_frames = int((release_ms / 1000) * SAMPLERATE)

    return {
        "note": note,
        "buffer": pitched,
        "position": 0,
        "amp": 0.0 if attack_frames > 0 else 1.0,
        "state": "attack" if attack_frames > 0 else "sustain",
        "attack_position": 0,
        "attack_frames": attack_frames,
        "release_position": 0,
        "release_frames": release_frames,
        "release_start_amp": 1.0
    }, speed


def add_voice(voice):

    with voices_lock:
        voices.append(voice)

        while len(voices) > MAX_VOICES:
            voices.pop(0)


def release_note(note):

    with voices_lock:
        for voice in voices:
            if voice["note"] == note and voice["state"] != "release":
                voice["state"] = "release"
                voice["release_position"] = 0
                voice["release_start_amp"] = voice["amp"]


def render_looping_voice(voice, frames):

    buffer = voice["buffer"]
    block = np.zeros(
        (frames, buffer.shape[1]),
        dtype=np.float32
    )

    if len(buffer) == 0:
        return block, False

    offset = 0

    while offset < frames:
        position = voice["position"]
        frames_to_copy = min(
            frames - offset,
            len(buffer) - position
        )

        block[offset:offset + frames_to_copy] = buffer[
            position:position + frames_to_copy
        ]

        position += frames_to_copy

        if position >= len(buffer):
            position = 0

        voice["position"] = position
        offset += frames_to_copy

    envelope = np.ones(frames, dtype=np.float32)

    if voice["state"] == "attack":
        attack_frames = max(1, voice["attack_frames"])
        positions = np.arange(
            voice["attack_position"],
            voice["attack_position"] + frames,
            dtype=np.float32
        )
        envelope = np.minimum(
            positions / attack_frames,
            1.0
        )
        voice["attack_position"] += frames

        if voice["attack_position"] >= attack_frames:
            voice["state"] = "sustain"
            voice["amp"] = 1.0
        else:
            voice["amp"] = float(envelope[-1])

    elif voice["state"] == "release":
        release_frames = voice["release_frames"]

        if release_frames <= 0:
            return block * 0.0, False

        positions = np.arange(
            voice["release_position"],
            voice["release_position"] + frames,
            dtype=np.float32
        )
        envelope = voice["release_start_amp"] * np.maximum(
            1.0 - (positions / release_frames),
            0.0
        )
        voice["release_position"] += frames
        voice["amp"] = float(envelope[-1])

        if voice["release_position"] >= release_frames:
            return block * envelope[:, None], False

    else:
        voice["amp"] = 1.0

    return block * envelope[:, None], True


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
            block, is_active = render_looping_voice(
                voice,
                frames
            )
            outdata += block

            if is_active:
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
        print("Polyphonic mode")
        print("Keys = playback")
        print("Knob 1 = loop start")
        print("Knob 2 = loop end")
        print("Knob 3 = gain")
        print("CC18 = attack")
        print("CC16/CC19 = release")

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
            # NOTE OFF
            # -------------------------

            elif (
                msg.type == "note_off"
                or (msg.type == "note_on" and msg.velocity == 0)
            ):

                release_note(msg.note)

                print(
                    f"NOTE OFF {msg.note}"
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

                # RELEASE
                elif msg.control in (RELEASE_CC, ALT_RELEASE_CC):

                    release_ms += delta * RELEASE_SENSITIVITY

                    release_ms = float(
                        np.clip(
                            release_ms,
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
                    f"RELEASE={release_ms:.1f}ms"
                )
