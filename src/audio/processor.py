"""
Real-time audio processor.

Rolling buffer → STFT → Chroma (HPCP proxy) + spectrogram column.

Design choices from research docs:
  - center=False  (no future-sample padding in real-time streams)
  - np.roll()     for O(1) rolling buffer update
  - Spectral subtraction noise gate
  - Output chroma: 12-dim float32, L2-normalised
"""

import warnings
import numpy as np
import librosa
from dataclasses import dataclass
from typing import Optional

from src.audio.chord_templates import match_chord_template


N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 44100

# Rolling buffer holds 4 chunks so STFT has enough context
BUFFER_SIZE = N_FFT * 2      # 4096 samples ≈ 93 ms

# Spectrogram display uses a smaller FFT for speed
DISP_N_FFT = 1024
N_FREQ_BINS = DISP_N_FFT // 2 + 1   # 513 bins


@dataclass
class ProcessorResult:
    chroma: np.ndarray          # shape (12,), L2-normalised
    chord_name: str
    chord_similarity: float
    spec_column: np.ndarray     # shape (N_FREQ_BINS,), log-magnitude for display
    rms: float                  # root-mean-square energy


class AudioProcessor:
    def __init__(self, noise_gate_rms: float = 0.01):
        self._buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self._noise_floor: Optional[np.ndarray] = None
        self._gate_rms = noise_gate_rms
        self._frame_count = 0

    # ------------------------------------------------------------------
    def process(self, chunk: np.ndarray) -> Optional[ProcessorResult]:
        """
        Ingest one audio chunk and return a ProcessorResult (or None if silent).
        """
        chunk = chunk.astype(np.float32)
        chunk_len = len(chunk)

        # --- rolling buffer (np.roll is O(n) but clean; fine for 4096 floats) ---
        self._buffer = np.roll(self._buffer, -chunk_len)
        self._buffer[-chunk_len:] = chunk

        rms = float(np.sqrt(np.mean(self._buffer ** 2)))

        # --- spectrogram column (always emit so the display keeps scrolling) ---
        spec_col = self._compute_spec_column(self._buffer)

        # --- below noise gate: return silent result so spectrogram still updates ---
        if rms < self._gate_rms:
            return ProcessorResult(
                chroma=np.zeros(12, dtype=np.float32),
                chord_name='Silence',
                chord_similarity=0.0,
                spec_column=spec_col,
                rms=rms,
            )

        # --- chroma / HPCP ---
        chroma = self._compute_chroma(self._buffer)

        # --- spectral subtraction noise floor estimate (update every 20 frames) ---
        if self._noise_floor is None:
            self._noise_floor = chroma.copy()
        elif self._frame_count % 20 == 0 and rms < self._gate_rms * 3:
            self._noise_floor = 0.9 * self._noise_floor + 0.1 * chroma

        self._frame_count += 1

        # soft noise gate on chroma
        if self._noise_floor is not None:
            chroma = np.maximum(chroma - 0.5 * self._noise_floor, 0.0)

        norm = np.linalg.norm(chroma)
        if norm < 1e-8:
            chroma = np.zeros(12, dtype=np.float32)
            chord_name, similarity = 'Silence', 0.0
        else:
            chroma = chroma / norm
            chord_name, similarity = match_chord_template(chroma)

        return ProcessorResult(
            chroma=chroma,
            chord_name=chord_name,
            chord_similarity=similarity,
            spec_column=spec_col,
            rms=rms,
        )

    # ------------------------------------------------------------------
    def _compute_chroma(self, audio: np.ndarray) -> np.ndarray:
        """
        CQT-based chroma (librosa chroma_cqt) — best for polyphonic harmony.
        Falls back to STFT chroma on error.

        fmin=C3 (not C2) avoids the sub-1024-sample FFT windows that librosa's
        CQT uses for very low octaves, eliminating "n_fft too large" UserWarnings.
        """
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message='n_fft=.*is too large',
                    category=UserWarning,
                )
                chroma = librosa.feature.chroma_cqt(
                    y=audio,
                    sr=SAMPLE_RATE,
                    hop_length=HOP_LENGTH,
                    fmin=librosa.note_to_hz('C3'),
                    n_chroma=12,
                    bins_per_octave=36,
                )
            # Mean across time frames → single 12-dim vector
            return chroma.mean(axis=1).astype(np.float32)
        except Exception:
            # Fallback: STFT chroma with center=False
            chroma = librosa.feature.chroma_stft(
                y=audio,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                center=False,
            )
            return chroma.mean(axis=1).astype(np.float32)

    def _compute_spec_column(self, audio: np.ndarray) -> np.ndarray:
        """One column of log-magnitude spectrogram for display."""
        stft = np.abs(
            librosa.stft(audio, n_fft=DISP_N_FFT, hop_length=HOP_LENGTH, center=False)
        )
        col = stft.mean(axis=1)          # average across time frames
        col_db = librosa.amplitude_to_db(col + 1e-9, ref=np.max)
        return col_db.astype(np.float32)
