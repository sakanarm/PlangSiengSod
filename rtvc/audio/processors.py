"""Block processors: take one block of mono float32 audio, return one block.

The real-time engine only knows this interface, so the RVC converter, the DSP robot
voice and the pass-through test are interchangeable.
"""

from __future__ import annotations

import numpy as np


class BlockProcessor:
    name = "base"
    is_ai_conversion = False

    def process(self, block: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:
        pass

    @property
    def algorithmic_latency_s(self) -> float:
        """Extra delay the processor itself adds (beyond block buffering)."""
        return 0.0

    @property
    def last_timings_ms(self) -> dict:
        return {}


class PassthroughProcessor(BlockProcessor):
    """Mic -> output unchanged. Used by the routing/audio test, never as 'conversion'."""

    name = "Pass-through (no conversion)"

    def __init__(self, gain_db: float = 0.0):
        self.gain = float(10 ** (gain_db / 20))

    def process(self, block):
        return block * self.gain


class RobotVoiceProcessor(BlockProcessor):
    """Classic DSP 'robotization' - NOT AI voice conversion.

    Each STFT frame keeps its magnitude and gets zero phase. Overlap-adding these
    frames at a fixed hop produces a monotone buzz at ``samplerate / hop`` Hz that
    still carries the spectral envelope (the words). Optional ring modulation adds
    a metallic character. Works on any CPU, needs no model.
    """

    name = "Robot voice v1 (DSP effect, no AI model)"

    def __init__(self, samplerate: int, robot_pitch_hz: float = 110.0, ring_mod_hz: float = 0.0,
                 ring_mix: float = 0.5, gain_db: float = 0.0):
        if not 40.0 <= robot_pitch_hz <= 400.0:
            raise ValueError("robot_pitch_hz must be within 40..400 Hz")
        self.sr = int(samplerate)
        self.hop = int(round(self.sr / robot_pitch_hz))
        self.n = 4 * self.hop                               # frame length, 75 % overlap
        self.window = np.hanning(self.n + 1)[:-1].astype(np.float32)   # periodic Hann
        # Synthesis normalization: analysis*synthesis window overlap-added at this hop.
        wsum = np.zeros(self.n + 3 * self.hop, dtype=np.float64)
        for k in range(4):
            wsum[k * self.hop:k * self.hop + self.n] += self.window.astype(np.float64) ** 2
        self.norm = float(wsum[self.n - self.hop:self.n].mean())
        self.ring_mod_hz = float(ring_mod_hz)
        self.ring_mix = float(np.clip(ring_mix, 0.0, 1.0))
        self.gain = float(10 ** (gain_db / 20))
        self.reset()

    def reset(self):
        self._in = np.zeros(0, dtype=np.float32)            # input not yet consumed by a frame
        self._hist = np.zeros(self.n - self.hop, dtype=np.float32)
        self._ola = np.zeros(self.n, dtype=np.float32)
        self._out = np.zeros(self.n, dtype=np.float32)      # primes the fixed output delay
        self._phase = 0.0

    @property
    def algorithmic_latency_s(self) -> float:
        return self.n / self.sr

    def _frame(self, frame: np.ndarray) -> np.ndarray:
        spec = np.abs(np.fft.rfft(frame * self.window))
        spec[0] = 0.0                                       # magnitude-only DC would leak an offset
        # Zero phase puts the impulse at sample 0; roll to the window center.
        y = np.roll(np.fft.irfft(spec, n=self.n), self.n // 2).astype(np.float32)
        return y * self.window / self.norm

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        self._in = np.concatenate([self._in, block])
        produced = []
        while self._in.shape[0] >= self.hop:
            hop_in = self._in[:self.hop]
            self._in = self._in[self.hop:]
            frame = np.concatenate([self._hist, hop_in])
            self._hist = frame[self.hop:]
            self._ola += self._frame(frame)
            produced.append(self._ola[:self.hop].copy())
            self._ola = np.concatenate([self._ola[self.hop:], np.zeros(self.hop, np.float32)])
        if produced:
            self._out = np.concatenate([self._out] + produced)
        out = self._out[:block.shape[0]]
        self._out = self._out[block.shape[0]:]
        if out.shape[0] < block.shape[0]:   # cannot happen after priming; keep length exact
            out = np.pad(out, (0, block.shape[0] - out.shape[0]))
        if self.ring_mod_hz > 0:
            t = self._phase + 2 * np.pi * self.ring_mod_hz * np.arange(out.shape[0]) / self.sr
            self._phase = float((t[-1] + 2 * np.pi * self.ring_mod_hz / self.sr) % (2 * np.pi))
            out = (1 - self.ring_mix) * out + self.ring_mix * out * np.sin(t).astype(np.float32)
        return np.clip(out * self.gain, -1.0, 1.0)
