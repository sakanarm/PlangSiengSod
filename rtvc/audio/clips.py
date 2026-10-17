"""Short clips for "voice from sample": record from the microphone, play a preview.

Both use their own callback stream (like ``engine.play_test_tone``): sd.rec / sd.play share
one global stream, and blocking read/write through WASAPI auto_convert loses samples.
Recorded audio stays in memory; nothing is written to disk.
"""

from __future__ import annotations

import threading

import numpy as np


def _extra(hostapi: str):
    import sounddevice as sd

    return sd.WasapiSettings(auto_convert=True) if "WASAPI" in hostapi else None


class ClipRecorder:
    """Records up to ``max_seconds`` of mono audio; stops by itself when full."""

    def __init__(self, device: int, hostapi: str, samplerate: int = 48000, max_seconds: float = 20.0):
        import sounddevice as sd

        self.sr = int(samplerate)
        self._buf = np.zeros(int(max_seconds * self.sr), dtype=np.float32)
        self._pos = 0
        self._full = threading.Event()
        self._lock = threading.Lock()

        def cb(indata, frames, t_info, status):
            with self._lock:
                n = min(frames, self._buf.size - self._pos)
                self._buf[self._pos:self._pos + n] = indata[:n, 0]
                self._pos += n
                if self._pos >= self._buf.size:
                    self._full.set()
                    raise sd.CallbackStop

        self._stream = sd.InputStream(device=device, channels=1, samplerate=self.sr, dtype="float32",
                                      callback=cb, extra_settings=_extra(hostapi))
        self._stream.start()

    @property
    def seconds(self) -> float:
        return self._pos / self.sr

    @property
    def full(self) -> bool:
        return self._full.is_set()

    def stop(self) -> np.ndarray:
        try:
            self._stream.stop()
        finally:
            self._stream.close()
        with self._lock:
            return self._buf[:self._pos].copy()


class ClipPlayer:
    """Plays one clip in the background; ``stop()`` ends it early."""

    def __init__(self, device: int | None, hostapi: str, x: np.ndarray, samplerate: int = 48000):
        import sounddevice as sd

        self._x = np.asarray(x, dtype=np.float32)
        self._pos = 0
        self._done = threading.Event()

        def cb(out, frames, t_info, status):
            n = min(frames, self._x.size - self._pos)
            out[:n, 0] = self._x[self._pos:self._pos + n]
            out[n:, 0] = 0.0
            self._pos += n
            if self._pos >= self._x.size:
                raise sd.CallbackStop

        self._stream = sd.OutputStream(device=device, channels=1, samplerate=int(samplerate), dtype="float32",
                                       callback=cb, extra_settings=_extra(hostapi),
                                       finished_callback=self._done.set)
        self._stream.start()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def stop(self) -> None:
        try:
            self._stream.stop()
        finally:
            self._stream.close()
