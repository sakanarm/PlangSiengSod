"""Save the processed (output) audio to a WAV file while the engine runs.

Disk writes happen on their own thread so a slow disk can never stall real-time audio.
Files are written only where the user chose; nothing is uploaded.
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class WavRecorder:
    def __init__(self, path: str | Path, samplerate: int):
        import soundfile as sf

        self.path = Path(path)
        self.sr = int(samplerate)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = sf.SoundFile(str(self.path), "w", samplerate=self.sr, channels=1, subtype="PCM_16")
        self._q: queue.Queue = queue.Queue(maxsize=200)   # ~10 s of 50 ms blocks
        self.samples = 0
        self.dropped = 0
        self.error = ""
        self._thread = threading.Thread(target=self._run, name="rtvc-recorder", daemon=True)
        self._thread.start()

    def write(self, block: np.ndarray) -> None:
        """Called from the audio worker; never blocks."""
        try:
            self._q.put_nowait(np.asarray(block, dtype=np.float32).copy())
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while True:
            block = self._q.get()
            if block is None:
                break
            try:
                self._file.write(block)
                self.samples += block.shape[0]
            except Exception as e:
                self.error = str(e)
                logger.exception("Recording write failed")
                break

    @property
    def seconds(self) -> float:
        return self.samples / self.sr

    def close(self) -> float:
        """Flush and close; returns the recorded duration in seconds."""
        self._q.put(None)
        self._thread.join(timeout=10)
        try:
            self._file.close()
        except Exception:
            logger.exception("Closing recording failed")
        logger.info("Recording saved: %s (%.1f s, %d blocks dropped)", self.path, self.seconds, self.dropped)
        return self.seconds
