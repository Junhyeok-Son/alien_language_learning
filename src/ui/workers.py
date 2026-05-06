"""
QThread workers that run the audio and translation pipelines off the GUI thread.

AudioWorker
  • Feeds a rolling audio buffer through AudioProcessor every chunk
  • Emits spectrogram columns, chord detections, and unknown-chord events

TranslationWorker
  • Receives batches of words via a queue
  • Calls EridianTranslator.translate_keywords()
  • Emits finished translation strings

ChordHoldFilter (helper)
  • A chord must appear in N_HOLD consecutive frames before being "emitted"
    to suppress transient noise
"""

import queue
import logging
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal

from src.audio.capture import AudioCapture
from src.audio.processor import AudioProcessor
from src.llm.translator import EridianTranslator
from src.vector_db.lexicon import DynamicLexicon, SearchResult

log = logging.getLogger(__name__)

N_HOLD = 4          # consecutive frames a chord must appear before being reported
WORD_BATCH_SIZE = 4  # send to LLM after this many words accumulate


class ChordHoldFilter:
    def __init__(self, n_hold: int = N_HOLD):
        self._n = n_hold
        self._last: str = ''
        self._count: int = 0

    def update(self, chord: str) -> bool:
        """Return True on the frame the chord is 'confirmed' (held for N frames)."""
        if chord == self._last:
            self._count += 1
            if self._count == self._n:
                return True
        else:
            self._last = chord
            self._count = 1
        return False


# ---------------------------------------------------------------------------
# Audio + Chord Detection Worker
# ---------------------------------------------------------------------------

class AudioWorker(QThread):
    # (spec_column: np.ndarray)  — one column of log-mag spectrogram
    spec_update = pyqtSignal(object)

    # (word: str, similarity: float, chord_hint: str)
    word_detected = pyqtSignal(str, float, str)

    # (chroma: np.ndarray)  — unknown chord; request user labelling
    unknown_chord = pyqtSignal(object)

    # (rms: float)
    rms_update = pyqtSignal(float)

    def __init__(self, lexicon: DynamicLexicon, demo_queue: queue.Queue | None = None):
        super().__init__()
        self._lexicon = lexicon
        self._demo_queue = demo_queue   # if set, reads from demo instead of mic
        self._audio_queue: queue.Queue = queue.Queue(maxsize=32)
        self._capture: AudioCapture | None = None
        self._processor = AudioProcessor(noise_gate_rms=0.005)
        self._hold_filter = ChordHoldFilter()
        self._running = False

    # ------------------------------------------------------------------
    def run(self) -> None:
        self._running = True

        if self._demo_queue is None:
            # Real microphone
            self._capture = AudioCapture(self._audio_queue)
            try:
                self._capture.start()
            except Exception as exc:
                log.error('Failed to open audio device: %s', exc)
                self._running = False
                return

        while self._running:
            try:
                src = self._demo_queue if self._demo_queue else self._audio_queue
                chunk: np.ndarray = src.get(timeout=0.1)
            except queue.Empty:
                continue

            result = self._processor.process(chunk)
            if result is None:
                continue

            self.spec_update.emit(result.spec_column)
            self.rms_update.emit(result.rms)

            if result.chord_name == 'Silence':
                self._hold_filter.update('Silence')
                continue

            confirmed = self._hold_filter.update(result.chord_name)
            if not confirmed:
                continue

            # FAISS lookup
            search: SearchResult = self._lexicon.search(result.chroma)

            if search.is_known:
                self.word_detected.emit(
                    search.word,
                    search.similarity,
                    result.chord_name,
                )
            else:
                # similarity too low → ask user to label
                self.unknown_chord.emit(result.chroma.copy())

    def stop(self) -> None:
        self._running = False
        if self._capture:
            self._capture.stop()
        self.wait(2000)


# ---------------------------------------------------------------------------
# LLM Translation Worker
# ---------------------------------------------------------------------------

class TranslationWorker(QThread):
    # (sentence: str)
    translation_ready = pyqtSignal(str)

    def __init__(self, translator: EridianTranslator):
        super().__init__()
        self._translator = translator
        self._queue: queue.Queue[list[str]] = queue.Queue()
        self._running = False

    def submit_words(self, words: list[str]) -> None:
        self._queue.put_nowait(words)

    def run(self) -> None:
        self._running = True
        while self._running:
            try:
                words = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            sentence = self._translator.translate_keywords(words)
            if sentence:
                self.translation_ready.emit(sentence)

    def stop(self) -> None:
        self._running = False
        self.wait(3000)
