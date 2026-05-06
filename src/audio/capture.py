"""
Non-blocking microphone capture using sounddevice.

Puts mono float32 chunks into a queue for downstream processing.
"""

import queue
import numpy as np
import sounddevice as sd
from typing import Optional


class AudioCapture:
    SAMPLE_RATE = 44100
    CHUNK_SIZE = 2048       # ~46 ms per chunk
    CHANNELS = 1

    def __init__(self, audio_queue: queue.Queue, device: Optional[int] = None):
        self._queue = audio_queue
        self._device = device
        self._stream: Optional[sd.InputStream] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            blocksize=self.CHUNK_SIZE,
            device=self._device,
            channels=self.CHANNELS,
            dtype='float32',
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # ------------------------------------------------------------------
    def _callback(self, indata: np.ndarray, frames: int,
                  time_info, status) -> None:
        if status:
            pass  # non-blocking; drop xrun info silently
        mono = indata[:, 0].copy()
        try:
            self._queue.put_nowait(mono)
        except queue.Full:
            pass  # drop oldest implicitly; queue consumer is too slow

    # ------------------------------------------------------------------
    @staticmethod
    def list_devices() -> list[dict]:
        return sd.query_devices()
