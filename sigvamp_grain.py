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

MAX_VOICES = 6
BLOCKSIZE = 512

GRAIN_SIZE = 2048
GRAIN_HOP = 512
GRAIN_LEVEL = 0.5

ATTACK_SENSITIVITY = 0.5
RELEASE_SENSITIVITY = 2.0

DEFAULT_ATTACK_MS = 20.0
DEFAULT_RELEASE_MS = 250.0

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

master_gain = 1.5

attack_ms = DEFAULT_ATTACK_MS
release_ms = DEFAULT_RELEASE_MS

grain_window = np.hanning(GRAIN_SIZE).astype(np.float32)

voices = []
voices_lock = threading.Lock()

# ---------------------------------
# MIDI HELPERS
# ---------------------------------


def cc_relative_delta(value):

    if value == 64:
        return 0

    return value - 64


def wrap_positions(positions, length):

    return np.mod(positions, length)


def read_interpolated(segment, positions):

    length = len(segment)
    wrapped = wrap_positions(positions, length)

    lower = np.floor(wrapped).astype(np.int32)
    upper = (lower + 1) % length
    fraction = (wrapped - lower).astype(np.float32)

    return (
        segment[lower] * (1.0 - fraction[:, None])
        + segment[upper] * fraction[:, None]
    )


def make_grain(segment, source_start, pitch_ratio):

    source_positions = source_start + (
        np.arange(GRAIN_SIZE, dtype=np.float32) * pitch_ratio
    )

    grain = read_interpolated(
        segment,
        source_positions
    )

    return grain * grain_window[:, None] * GRAIN_LEVEL


def build_voice(note):

    start_idx = int(loop_start * len(audio))
    end_idx = int(loop_end * len(audio))

    min_window = max(GRAIN_SIZE, 1000)

    if end_idx <= start_idx + min_window:
        end_idx = start_idx + min_window

    segment = audio[start_idx:end_idx].copy()

    semitones = note - ROOT_NOTE
    pitch_ratio = 2 ** (semitones / 12)

    attack_frames = int((attack_ms / 1000) * SAMPLERATE)
    release_frames = int((release_ms / 1000) * SAMPLERATE)

    return {
        "note": note,
        "segment": segment,
        "pitch_ratio": pitch_ratio,
        "read_position": 0.0,
        "samples_until_next_grain": 0,
        "grains": [],
        "amp": 0.0 if attack_frames > 0 else 1.0,
        "state": "attack" if attack_frames > 0 else "sustain",
        "attack_position": 0,
        "attack_frames": attack_frames,
        "release_position": 0,
        "release_frames": release_frames,
        "release_start_amp": 1.0
    }, pitch_ratio


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


def schedule_grains(voice, frames):

    countdown = voice["samples_until_next_grain"]

    while countdown < frames:
        source_start = voice["read_position"] + countdown
        grain = make_grain(
            voice["segment"],
            source_start,
            voice["pitch_ratio"]
        )

        voice["grains"].append({
            "buffer": grain,
            "position": 0,
            "offset": int(countdown)
        })

        countdown += GRAIN_HOP

    voice["samples_until_next_grain"] = countdown - frames
    voice["read_position"] = float(
        (voice["read_position"] + frames) % len(voice["segment"])
    )


def render_grains(voice, frames):

    block = np.zeros(
        (frames, voice["segment"].shape[1]),
        dtype=np.float32
    )

    active_grains = []

    for grain in voice["grains"]:
        grain_buffer = grain["buffer"]
        grain_position = grain["position"]
        output_offset = grain["offset"]

        if output_offset >= frames:
            grain["offset"] = output_offset - frames
            active_grains.append(grain)
            continue

        if output_offset < 0:
            grain_position += -output_offset
            output_offset = 0

        remaining_grain = len(grain_buffer) - grain_position
        remaining_block = frames - output_offset
        frames_to_copy = min(
            remaining_grain,
            remaining_block
        )

        if frames_to_copy > 0:
            block[output_offset:output_offset + frames_to_copy] += (
                grain_buffer[
                    grain_position:grain_position + frames_to_copy
                ]
            )

            grain["position"] = grain_position + frames_to_copy
            grain["offset"] = output_offset + frames_to_copy - frames

        if grain["position"] < len(grain_buffer):
            active_grains.append(grain)

    voice["grains"] = active_grains

    return block


def apply_voice_envelope(voice, block):

    frames = len(block)
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


def render_granular_voice(voice, frames):

    if len(voice["segment"]) == 0:
        empty = np.zeros(
            (frames, audio.shape[1]),
            dtype=np.float32
        )
        return empty, False

    schedule_grains(
        voice,
        frames
    )

    block = render_grains(
        voice,
        frames
    )

    block, is_active = apply_voice_envelope(
        voice,
        block
    )

    return block * master_gain, is_active


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
            block, is_active = render_granular_voice(
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
        print("Granular time-stable pitch mode")
        print("Keys = held-note granular looping")
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

                voice, pitch_ratio = build_voice(msg.note)
                add_voice(voice)

                print(
                    f"NOTE {msg.note} "
                    f"PITCH={pitch_ratio:.2f} "
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
