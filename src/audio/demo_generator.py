"""
Synthetic alien-language audio generator for demo mode.

Produces polyphonic sine-wave chords that mimic Eridian speech and pushes
float32 numpy chunks into a queue.Queue — identical to what AudioCapture
delivers from a real microphone.

Planned "sentences":
  Act 1 — Greeting   : Friend, Greet
  Act 2 — Navigation : Danger, Star, Many, Travel
  Act 3 — Needs      : Question, Food, Need  (→ "Need" falls back to "Help")
  Act 4 — Affirmation: Friend, Question, Understand, Yes
  Act 5 — Home       : Home, Star, Together, Alive
"""

import queue
import time
import threading
import numpy as np
import logging

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100
CHUNK_SIZE  = 2048

# Duration per chord and silence gap (seconds)
CHORD_DURATION   = 0.8
SILENCE_DURATION = 0.3
INTER_ACT_PAUSE  = 1.5

# Map chord_hint → MIDI semitone offsets relative to C4 (261.63 Hz)
_CHORD_NOTES: dict[str, list[int]] = {
    'C_maj':  [0, 4, 7],
    'C_min':  [0, 3, 7],
    'C_dom7': [0, 4, 7, 10],
    'C#_maj': [1, 5, 8],
    'D_maj':  [2, 6, 9],
    'D_min':  [2, 5, 9],
    'D#_maj': [3, 7, 10],
    'E_maj':  [4, 8, 11],
    'E_min':  [4, 7, 11],
    'F_maj':  [5, 9, 0],
    'F_min':  [5, 8, 0],
    'F#_maj': [6, 10, 1],
    'G_maj':  [7, 11, 2],
    'G_min':  [7, 10, 2],
    'G_dom7': [7, 11, 2, 5],
    'G#_maj': [8, 0, 3],
    'A_maj':  [9, 1, 4],
    'A_min':  [9, 0, 4],
    'A#_maj': [10, 2, 5],
    'B_maj':  [11, 3, 6],
    'B_min':  [11, 2, 6],
}

# Scripted acts: list of chord_hints to play in sequence
_ACTS: list[list[str]] = [
    ['C_maj', 'G_dom7'],                    # Friend, Greet
    ['A_min', 'D_min', 'G#_maj', 'D_maj'], # Danger, Star, Many, Travel
    ['G_maj', 'E_min', 'F_maj'],            # Question, Food, Help
    ['C_maj', 'G_maj', 'G_min', 'A_maj'],  # Friend, Question, Understand, Yes
    ['F#_maj', 'D_min', 'F_min', 'B_maj'], # Home, Star, Together, Alive
]


def _midi_to_freq(semitone: int, base: float = 261.63) -> float:
    """Convert semitone offset from C4 to Hz."""
    return base * (2 ** (semitone / 12.0))


def _render_chord(notes: list[int], duration: float, sample_rate: int = SAMPLE_RATE,
                  amplitude: float = 0.25) -> np.ndarray:
    """Synthesise a polyphonic chord as a mono float32 waveform with fade-in/out."""
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False, dtype=np.float64)

    wave = np.zeros(n_samples, dtype=np.float64)
    for semitone in notes:
        freq = _midi_to_freq(semitone)
        wave += np.sin(2 * np.pi * freq * t)

    # Normalise then scale
    peak = np.max(np.abs(wave))
    if peak > 1e-8:
        wave /= peak
    wave *= amplitude

    # 20 ms fade-in / fade-out to avoid clicks
    fade = int(0.02 * sample_rate)
    if fade > 0:
        ramp = np.linspace(0, 1, fade)
        wave[:fade]  *= ramp
        wave[-fade:] *= ramp[::-1]

    return wave.astype(np.float32)


def _render_silence(duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(duration * sample_rate), dtype=np.float32)


class DemoAudioGenerator:
    """
    Generates synthetic Eridian audio and pushes chunks into `out_queue`.
    Call start() in a background thread; call stop() to halt.
    """

    def __init__(self, out_queue: queue.Queue, loop: bool = True):
        self._queue  = out_queue
        self._loop   = loop
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name='DemoGenerator')
        self._thread.start()
        log.info('DemoAudioGenerator started (loop=%s)', self._loop)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop_event.is_set():
            for act in _ACTS:
                if self._stop_event.is_set():
                    return
                for chord_hint in act:
                    if self._stop_event.is_set():
                        return
                    notes = _CHORD_NOTES.get(chord_hint, [0, 4, 7])
                    log.debug('Demo: playing chord %s → %s', chord_hint, notes)
                    self._push_audio(_render_chord(notes, CHORD_DURATION))
                    self._push_audio(_render_silence(SILENCE_DURATION))
                # Inter-act pause
                self._push_audio(_render_silence(INTER_ACT_PAUSE))

            if not self._loop:
                log.info('DemoAudioGenerator: finished all acts, stopping.')
                break

    def _push_audio(self, audio: np.ndarray) -> None:
        """Slice audio into CHUNK_SIZE pieces and enqueue each."""
        offset = 0
        while offset < len(audio):
            if self._stop_event.is_set():
                return
            chunk = audio[offset: offset + CHUNK_SIZE]
            if len(chunk) < CHUNK_SIZE:
                # Pad last chunk to full size
                chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
            try:
                self._queue.put(chunk, timeout=0.5)
            except queue.Full:
                pass  # drop if consumer is too slow
            offset += CHUNK_SIZE
            # Pace the producer to real-time
            time.sleep(CHUNK_SIZE / SAMPLE_RATE * 0.9)
