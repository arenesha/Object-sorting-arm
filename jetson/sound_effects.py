"""Autonomous Object-Sorting Robotic Arm - Sound Effects Engine

Cross-platform audio feedback for pick and place operations:
- PICK Sound: Crisp mechanical grip / suction chirp.
- PLACE Sound: Satisfying harmonic release / confirmation drop chime.

Platform Support:
- Windows: Native winsound (zero latency, in-memory, async).
- Linux / NVIDIA Jetson: ALSA (aplay), PulseAudio (paplay), PipeWire (pw-play).
- Headless / No Audio: Fails gracefully without throwing exceptions or blocking FPS.
"""

import os
import sys
import io
import math
import struct
import wave
import shutil
import tempfile
import threading
import subprocess
from typing import Optional

# Global sound enabled state
SOUND_ENABLED = True

# Cache generated WAV bytes in memory
_PICK_WAV_BYTES: Optional[bytes] = None
_PLACE_WAV_BYTES: Optional[bytes] = None
_TEMP_DIR: Optional[str] = None
_PICK_TEMP_FILE: Optional[str] = None
_PLACE_TEMP_FILE: Optional[str] = None


def _generate_pick_wav() -> bytes:
    """
    Generate an in-memory 44.1kHz 16-bit mono WAV for the 'PICK' action:
    A crisp mechanical servo grip / suction chirp (500Hz -> 1150Hz over 85ms).
    """
    sample_rate = 44100
    duration = 0.085
    num_samples = int(sample_rate * duration)
    buf = io.BytesIO()

    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        raw_frames = bytearray()
        phase = 0.0
        for i in range(num_samples):
            t = i / float(sample_rate)
            progress = t / duration

            # Pitch ramp from 520Hz up to 1180Hz
            freq = 520.0 + (660.0 * (progress ** 1.3))
            phase += 2.0 * math.pi * freq / sample_rate

            # Amplitude envelope: 6ms attack, slight sustain, smooth release
            if t < 0.006:
                env = t / 0.006
            elif progress < 0.65:
                env = 1.0 - (progress - 0.006) * 0.15
            else:
                rel_p = (progress - 0.65) / 0.35
                env = (1.0 - 0.15) * math.cos(rel_p * (math.pi / 2.0))

            # Primary tone + slight metallic 2nd harmonic
            sample_val = 0.78 * math.sin(phase) + 0.22 * math.sin(phase * 2.1)
            # Add subtle initial mechanical transient click in first 8ms
            if t < 0.008:
                click = math.sin(2.0 * math.pi * 1800.0 * t) * (1.0 - t / 0.008)
                sample_val = 0.65 * sample_val + 0.35 * click

            val_int = int(max(-32767, min(32767, sample_val * env * 24000.0)))
            raw_frames.extend(struct.pack("<h", val_int))

        wf.writeframes(raw_frames)

    return buf.getvalue()


def _generate_place_wav() -> bytes:
    """
    Generate an in-memory 44.1kHz 16-bit mono WAV for the 'PLACE' action:
    A pleasant dual-tone harmonic placement chime (G5 784Hz + C6 1046Hz, fading out smoothly over 150ms).
    """
    sample_rate = 44100
    duration = 0.150
    num_samples = int(sample_rate * duration)
    buf = io.BytesIO()

    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        raw_frames = bytearray()
        phase1 = 0.0
        phase2 = 0.0
        for i in range(num_samples):
            t = i / float(sample_rate)
            progress = t / duration

            # Dual frequency chime: 784.0 Hz (G5) and 1046.5 Hz (C6)
            # Gentle downward frequency glide (-40Hz) simulating object resting
            freq1 = 784.0 - (40.0 * progress)
            freq2 = 1046.5 - (60.0 * progress)

            phase1 += 2.0 * math.pi * freq1 / sample_rate
            phase2 += 2.0 * math.pi * freq2 / sample_rate

            # Fast attack (3ms) and exponential acoustic decay
            if t < 0.003:
                env = t / 0.003
            else:
                env = math.exp(-t * 22.0)

            sample_val = 0.58 * math.sin(phase1) + 0.42 * math.sin(phase2)
            val_int = int(max(-32767, min(32767, sample_val * env * 23000.0)))
            raw_frames.extend(struct.pack("<h", val_int))

        wf.writeframes(raw_frames)

    return buf.getvalue()


def _init_audio_assets():
    """Lazily generate and cache sound assets."""
    global _PICK_WAV_BYTES, _PLACE_WAV_BYTES, _TEMP_DIR, _PICK_TEMP_FILE, _PLACE_TEMP_FILE
    if _PICK_WAV_BYTES is None:
        _PICK_WAV_BYTES = _generate_pick_wav()
    if _PLACE_WAV_BYTES is None:
        _PLACE_WAV_BYTES = _generate_place_wav()

    # On Linux, tools like aplay/paplay require a file path on disk
    if sys.platform.startswith("linux") and _PICK_TEMP_FILE is None:
        try:
            _TEMP_DIR = os.path.join(tempfile.gettempdir(), "jetarm_audio")
            os.makedirs(_TEMP_DIR, exist_ok=True)
            _PICK_TEMP_FILE = os.path.join(_TEMP_DIR, "pick.wav")
            _PLACE_TEMP_FILE = os.path.join(_TEMP_DIR, "place.wav")
            with open(_PICK_TEMP_FILE, "wb") as f:
                f.write(_PICK_WAV_BYTES)
            with open(_PLACE_TEMP_FILE, "wb") as f:
                f.write(_PLACE_WAV_BYTES)
        except Exception:
            _PICK_TEMP_FILE = None
            _PLACE_TEMP_FILE = None


def _play_bytes_async(wav_bytes: bytes, file_path: Optional[str] = None):
    """Plays audio bytes asynchronously without blocking the calling thread."""
    if not SOUND_ENABLED:
        return

    def _worker():
        try:
            if sys.platform.startswith("win"):
                import winsound
                # Play directly from memory buffer, asynchronous
                winsound.PlaySound(wav_bytes, winsound.SND_MEMORY | winsound.SND_ASYNC)
            elif sys.platform.startswith("linux"):
                # Check for standard Linux audio players
                if file_path and os.path.exists(file_path):
                    player = (
                        shutil.which("pw-play")
                        or shutil.which("paplay")
                        or shutil.which("aplay")
                    )
                    if player:
                        subprocess.Popen(
                            [player, file_path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
        except Exception:
            # Headless or missing audio output device: ignore gracefully
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def play_pick_sound():
    """Play the mechanical gripper pickup sound effect."""
    try:
        _init_audio_assets()
        _play_bytes_async(_PICK_WAV_BYTES, _PICK_TEMP_FILE)
    except Exception:
        pass


def play_place_sound():
    """Play the satisfying object placement / bin drop sound effect."""
    try:
        _init_audio_assets()
        _play_bytes_async(_PLACE_WAV_BYTES, _PLACE_TEMP_FILE)
    except Exception:
        pass


def set_sound_enabled(enabled: bool):
    """Enable or disable audio effects globally."""
    global SOUND_ENABLED
    SOUND_ENABLED = bool(enabled)


def is_sound_enabled() -> bool:
    """Return whether sound effects are enabled."""
    return SOUND_ENABLED
