"""Block processors: take one block of mono float32 audio, return one block.

The real-time engine only knows this interface, so the RVC converter, the DSP voices
and the pass-through test are interchangeable. Everything in this file is classic DSP
(no AI model, no downloads) and runs on any CPU.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


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
        # One hop of priming always covers the samples still waiting for a full hop; with the
        # (n - hop) history this gives a total delay of exactly n samples, for any block size.
        self._out = np.zeros(self.hop, dtype=np.float32)
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


# ============================================================================ STFT base
class _StftProcessor(BlockProcessor):
    """Streaming STFT -> per-frame spectral processing -> overlap-add.

    Frame length ``n``, hop ``n/4`` with a periodic Hann window for analysis and
    synthesis. Output is delayed by exactly ``n`` samples, so the result does not depend
    on how the input is cut into blocks.
    """

    def __init__(self, samplerate: int, n: int):
        self.sr = int(samplerate)
        self.n = int(n)
        self.hop = self.n // 4
        self.window = np.hanning(self.n + 1)[:-1]           # periodic Hann, float64
        self.norm = float(np.sum(self.window ** 2) / self.hop)
        self.bins = np.arange(self.n // 2 + 1, dtype=np.float64)
        self._reset_stream()

    def _reset_stream(self):
        self._in = np.zeros(0, dtype=np.float32)
        self._hist = np.zeros(self.n - self.hop, dtype=np.float32)
        self._ola = np.zeros(self.n, dtype=np.float64)
        # One hop of priming always covers the samples still waiting for a full hop; with the
        # (n - hop) history this gives a total delay of exactly n samples, for any block size.
        self._out = np.zeros(self.hop, dtype=np.float32)

    def reset(self):
        self._reset_stream()

    @property
    def algorithmic_latency_s(self) -> float:
        return self.n / self.sr

    def _frame(self, frame: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        """Windowed analysis/synthesis of one frame; return n samples (already windowed)."""
        raise NotImplementedError

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        self._in = np.concatenate([self._in, block])
        produced = []
        while self._in.shape[0] >= self.hop:
            hop_in = self._in[:self.hop]
            self._in = self._in[self.hop:]
            frame = np.concatenate([self._hist, hop_in])
            self._hist = frame[self.hop:]
            self._ola += self._frame(frame.astype(np.float64))
            produced.append(self._ola[:self.hop].astype(np.float32))
            self._ola = np.concatenate([self._ola[self.hop:], np.zeros(self.hop)])
        if produced:
            self._out = np.concatenate([self._out] + produced)
        out = self._out[:block.shape[0]]
        self._out = self._out[block.shape[0]:]
        if out.shape[0] < block.shape[0]:
            out = np.pad(out, (0, block.shape[0] - out.shape[0]))
        return out


def _envelope_log(logmag: np.ndarray, n: int, lifter: int) -> np.ndarray:
    """Smooth log-spectral envelope (formants) by cepstral liftering."""
    c = np.fft.irfft(logmag, n=n)
    c[lifter:n - lifter + 1] = 0.0
    return np.fft.rfft(c).real


def _peak_envelope_log(logmag: np.ndarray, half_width: int, smooth_bins: int) -> np.ndarray:
    """Log-spectral envelope through the dominant spectral peaks (harmonics), lightly smoothed.

    A peak counts if it is the maximum within +-half_width bins, so window sidelobes in
    the valleys between harmonics are skipped. Follows formants much more closely than a
    cepstral envelope, whose estimate is dominated by those valleys. Beyond the first and
    last peak the envelope stays flat.
    """
    h = half_width
    local_max = np.lib.stride_tricks.sliding_window_view(np.pad(logmag, h, mode="edge"), 2 * h + 1).max(axis=1)
    pk = np.flatnonzero(logmag >= local_max)
    e = np.interp(np.arange(logmag.size), pk, logmag[pk])
    k = smooth_bins
    return np.convolve(np.pad(e, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


# ============================================================================ voices
class VoiceShiftProcessor(_StftProcessor):
    """Independent pitch and formant shift ("different person" voices) - DSP, NOT AI.

    Phase-vocoder pitch shift with identity phase locking. Each frame is split into a
    spectral envelope (a line through the harmonic peaks: the vocal tract / formants) and the
    excitation (the harmonics: the pitch). The excitation is resampled by the pitch
    ratio and the envelope by the formant ratio, so pitch and "size of the speaker"
    change separately. Optional vibrato modulates the pitch (used by the Alien voice).
    Rhythm, accent and the shape of the intonation stay the speaker's own.
    """

    name = "Voice shift (pitch + formant, DSP)"

    def __init__(self, samplerate: int, pitch_st: float = 0.0, formant_st: float = 0.0,
                 vibrato_st: float = 0.0, vibrato_hz: float = 0.0, jitter_st: float = 0.0, seed: int = 11):
        super().__init__(samplerate, 2048)                  # ~43 ms at 48 kHz: resolves male harmonics
        self.peak_half_width = max(1, int(round(50.0 * self.n / self.sr)))   # peaks dominate +-50 Hz
        self.smooth_bins = int(round(150.0 * self.n / self.sr)) | 1          # ~150 Hz envelope smoothing
        self.expected = 2 * np.pi * self.hop * self.bins / self.n
        self.pitch_st = float(pitch_st)                     # may be changed live
        self.formant_st = float(formant_st)                 # may be changed live
        self.vibrato_st = float(vibrato_st)
        self.vibrato_hz = float(vibrato_hz)
        self.jitter_st = float(jitter_st)                   # random frame-to-frame pitch wobble (rough voice)
        self.seed = int(seed)
        self.reset()

    def reset(self):
        self._reset_stream()
        self._prev_ph = np.zeros(self.bins.size)
        self._syn_ph = np.zeros(self.bins.size)
        self._lfo = 0.0
        self._rng = np.random.default_rng(self.seed)
        self._jit = 0.0

    def _frame(self, frame):
        X = np.fft.rfft(frame * self.window)
        mag = np.abs(X)
        ph = np.angle(X)
        dphi = ph - self._prev_ph - self.expected
        dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
        advance = self.expected + dphi                      # true phase advance per hop
        self._prev_ph = ph

        st = self.pitch_st
        if self.vibrato_st and self.vibrato_hz:
            st += self.vibrato_st * np.sin(self._lfo)
            self._lfo = (self._lfo + 2 * np.pi * self.vibrato_hz * self.hop / self.sr) % (2 * np.pi)
        if self.jitter_st:
            # lightly smoothed random walk around 0 (unit variance), one step per hop (~11 ms)
            self._jit = 0.5 * self._jit + np.sqrt(0.75) * self._rng.standard_normal()
            st += self.jitter_st * self._jit
        if mag.max() < 1e-7:                                # silence: output zero, keep phases sane
            self._syn_ph = ph
            return np.zeros(self.n)
        r = 2.0 ** (st / 12)
        f = 2.0 ** (self.formant_st / 12)
        logmag = np.log(mag + 1e-9)
        env_log = _peak_envelope_log(logmag, self.peak_half_width, self.smooth_bins)
        exc = np.exp(np.minimum(logmag - env_log, np.log(2.0)))      # flat (~1) at the harmonics

        peaks = np.flatnonzero((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:])) + 1
        if abs(r - 1.0) < 1e-6 or peaks.size == 0:          # no pitch change: keep the original phases
            new_exc, phase = exc, ph
        else:
            # Peak shifting (Laroche & Dolson): every bin belongs to its nearest spectral peak;
            # each peak region moves as a whole so the window lobe keeps its shape, and the
            # peak's phase is propagated at the new frequency (identity phase locking).
            region = np.searchsorted((peaks[:-1] + peaks[1:]) / 2.0, self.bins)
            target_peak = np.rint(peaks * r).astype(int)
            target = self.bins.astype(int) + (target_peak - peaks)[region]
            tp = np.minimum(target_peak, self.bins.size - 1)
            rot = np.exp(1j * (self._syn_ph[tp] + advance[peaks] * r - ph[peaks]))
            ok = (target >= 0) & (target < self.bins.size)
            Z = np.zeros(self.bins.size, dtype=np.complex128)
            np.add.at(Z, target[ok], (exc * np.exp(1j * ph) * rot[region])[ok])
            new_exc, phase = np.abs(Z), np.angle(Z)
        new_env_log = np.interp(self.bins / f, self.bins, env_log) if abs(f - 1.0) > 1e-6 else env_log
        self._syn_ph = np.mod(phase, 2 * np.pi)
        new_mag = new_exc * np.exp(new_env_log)
        # Keep the frame energy: resampling/warping otherwise makes shifted voices quieter.
        new_mag *= np.clip(np.sqrt(np.sum(mag ** 2) / max(np.sum(new_mag ** 2), 1e-18)), 0.25, 4.0)
        Y = new_mag * np.exp(1j * phase)
        return np.fft.irfft(Y, n=self.n) * self.window / self.norm


class WhisperProcessor(_StftProcessor):
    """Whisper / ghost: keeps only the formant envelope and replaces the voice with noise.

    No pitch or harmonics of the speaker survive; only the words (spectral envelope).
    """

    name = "Whisper (DSP)"

    def __init__(self, samplerate: int, breath_db: float = 6.0, seed: int = 1234):
        super().__init__(samplerate, 1024)
        self.lifter = max(8, int(self.sr * 0.0012))
        self.seed = int(seed)
        freqs = self.bins * self.sr / self.n
        # breathy tilt: up to +breath_db above ~2 kHz, and no rumble below 150 Hz
        self.tilt = (10 ** (breath_db / 20 * np.clip((freqs - 1000) / 2000, 0, 1))
                     * np.clip((freqs - 80) / 120, 0, 1))
        self.reset()

    def reset(self):
        self._reset_stream()
        self._rng = np.random.default_rng(self.seed)

    def _frame(self, frame):
        mag = np.abs(np.fft.rfft(frame * self.window))
        phase = self._rng.uniform(-np.pi, np.pi, self.bins.size)
        if mag.max() < 1e-7:
            return np.zeros(self.n)
        env = np.exp(_envelope_log(np.log(mag + 1e-9), self.n, self.lifter))
        # The envelope of a harmonic voice is lower than its peaks; rescale to keep the loudness.
        env *= np.sqrt(np.sum(mag ** 2) / max(np.sum(env ** 2), 1e-18))
        Y = env * self.tilt * np.exp(1j * phase)
        return np.fft.irfft(Y, n=self.n) * self.window / self.norm


class RadioProcessor(_StftProcessor):
    """Walkie-talkie / radio: narrow band-pass plus mild saturation."""

    name = "Radio (DSP)"

    def __init__(self, samplerate: int, low_hz: float = 400.0, high_hz: float = 3000.0, drive: float = 4.0):
        super().__init__(samplerate, 1024)
        freqs = self.bins * self.sr / self.n
        lo = np.clip((freqs - low_hz * 0.6) / (low_hz * 0.4), 0, 1)
        hi = np.clip((high_hz * 1.4 - freqs) / (high_hz * 0.4), 0, 1)
        self.mask = (lo * hi) ** 2
        self.drive = float(drive)

    def _frame(self, frame):
        Y = np.fft.rfft(frame * self.window) * self.mask
        return np.fft.irfft(Y, n=self.n) * self.window / self.norm

    def process(self, block):
        y = super().process(block)
        return (np.tanh(self.drive * y) / np.tanh(self.drive) * 0.7).astype(np.float32)


class EqProcessor(_StftProcessor):
    """Smooth graphic EQ: gains at a few band centres, interpolated on a log-frequency axis.

    Used by the "voice from sample" voices to move the tone (spectral balance) towards a
    reference. Below the first and above the last band the gain stays at the end value.
    ``amount`` scales every gain (0 = flat, 1 = full curve) and may be changed live.
    """

    name = "EQ"

    def __init__(self, samplerate: int, bands: list[tuple[float, float]], amount: float = 1.0, n: int = 1024):
        super().__init__(samplerate, n)                     # ~21 ms at 48 kHz
        bands = sorted((float(hz), float(db)) for hz, db in bands)
        if not bands or bands[0][0] <= 0:
            raise ValueError("bands must be (frequency Hz > 0, gain dB) pairs")
        hz, db = np.array(bands).T
        freqs = np.maximum(self.bins * self.sr / self.n, hz[0])           # DC takes the first band's gain
        self.curve_db = np.interp(np.log2(freqs), np.log2(hz), db)
        self.amount = float(amount)                         # may be changed live
        self._gain_for = None
        self._gain = None

    def gain_at(self, hz: float) -> float:
        """Gain in dB applied at ``hz`` with the current amount (nearest FFT bin)."""
        return float(self.amount * self.curve_db[int(round(hz * self.n / self.sr))])

    def _frame(self, frame):
        if self._gain_for != self.amount:                   # recompute only when the amount changes
            self._gain = 10 ** (self.amount * self.curve_db / 20)
            self._gain_for = self.amount
        Y = np.fft.rfft(frame * self.window) * self._gain
        return np.fft.irfft(Y, n=self.n) * self.window / self.norm


class DistortionProcessor(BlockProcessor):
    """Soft-clipping saturation (stateless)."""

    name = "Distortion"

    def __init__(self, drive: float = 3.0, mix: float = 0.4):
        self.drive = float(drive)
        self.mix = float(np.clip(mix, 0.0, 1.0))

    def process(self, block):
        wet = np.tanh(self.drive * block) / np.tanh(self.drive)
        return ((1 - self.mix) * block + self.mix * wet).astype(np.float32)


class RingModProcessor(BlockProcessor):
    """Ring modulation with a continuous oscillator phase (metallic / sci-fi tone)."""

    name = "Ring modulation"

    def __init__(self, samplerate: int, freq_hz: float = 60.0, mix: float = 0.3):
        self.sr = int(samplerate)
        self.freq = float(freq_hz)
        self.mix = float(np.clip(mix, 0.0, 1.0))
        self.reset()

    def reset(self):
        self._phase = 0.0

    def process(self, block):
        t = self._phase + 2 * np.pi * self.freq * np.arange(block.shape[0]) / self.sr
        self._phase = float((self._phase + 2 * np.pi * self.freq * block.shape[0] / self.sr) % (2 * np.pi))
        return ((1 - self.mix) * block + self.mix * block * np.sin(t)).astype(np.float32)


class ReverbProcessor(BlockProcessor):
    """Cave / hall reverb: uniformly partitioned FFT convolution with a synthetic room.

    The impulse response is generated here (decaying filtered noise plus a few early
    reflections), so nothing is downloaded. Latency is one partition (~21 ms at 48 kHz).
    """

    name = "Reverb"

    def __init__(self, samplerate: int, rt60_s: float = 2.2, wet: float = 0.5, dry: float = 0.8,
                 seed: int = 7, partition: int = 1024):
        self.sr = int(samplerate)
        self.P = int(partition)
        self.wet, self.dry = float(wet), float(dry)
        self.ir = self._make_ir(rt60_s, seed)
        k = -(-self.ir.size // self.P)
        padded = np.zeros(k * self.P)
        padded[:self.ir.size] = self.ir
        parts = padded.reshape(k, self.P)
        self.H = np.fft.rfft(np.concatenate([parts, np.zeros_like(parts)], axis=1), axis=1)
        self.reset()

    def _make_ir(self, rt60, seed):
        rng = np.random.default_rng(seed)
        n = int(self.sr * rt60)
        t = np.arange(n) / self.sr
        noise = rng.standard_normal(n)
        spec = np.fft.rfft(noise)
        f = np.fft.rfftfreq(n, 1 / self.sr)
        spec *= 1.0 / (1.0 + (f / 3500.0) ** 2) * np.clip(f / 150.0, 0, 1)     # darker, no rumble
        ir = np.fft.irfft(spec, n=n) * 10 ** (-3 * t / rt60)                      # 60 dB decay at rt60
        ir *= np.clip(t / 0.02, 0, 1)                                             # pre-delay fade-in
        for delay_s, gain in ((0.011, 0.6), (0.023, 0.45), (0.037, 0.35), (0.051, 0.3)):
            ir[int(delay_s * self.sr)] += gain * np.max(np.abs(ir))               # early reflections
        return ir / np.sqrt(np.sum(ir ** 2))                                      # unit energy

    def reset(self):
        self._fdl = np.zeros_like(self.H)                   # frequency-domain delay line
        self._prev = np.zeros(self.P)
        self._in = np.zeros(0, dtype=np.float32)
        self._out = np.zeros(self.P, dtype=np.float32)      # primes the one-partition delay

    @property
    def algorithmic_latency_s(self) -> float:
        return self.P / self.sr

    def process(self, block):
        block = np.asarray(block, dtype=np.float32)
        self._in = np.concatenate([self._in, block])
        produced = []
        while self._in.shape[0] >= self.P:
            x = self._in[:self.P].astype(np.float64)
            self._in = self._in[self.P:]
            self._fdl = np.roll(self._fdl, 1, axis=0)
            self._fdl[0] = np.fft.rfft(np.concatenate([self._prev, x]))
            self._prev = x
            wet = np.fft.irfft(np.sum(self._fdl * self.H, axis=0), n=2 * self.P)[self.P:]
            produced.append((self.dry * x + self.wet * wet).astype(np.float32))
        if produced:
            self._out = np.concatenate([self._out] + produced)
        out = self._out[:block.shape[0]]
        self._out = self._out[block.shape[0]:]
        if out.shape[0] < block.shape[0]:
            out = np.pad(out, (0, block.shape[0] - out.shape[0]))
        return out


class ModDelayProcessor(BlockProcessor):
    """Delay line whose delay is swept by a sine LFO: flanger, pitch wobble or a plain echo.

    ``d[n] = v[n - delay(n)]``, ``v[n] = x[n] + feedback * d[n]``, output ``dry * x + wet * d``,
    with ``delay(n) = base + depth * sin(2 pi rate n / sr)`` read by linear interpolation.
    Only past samples are read, so the dry path is not delayed at all. The input is processed
    in chunks shorter than the smallest delay, which keeps the feedback exact and the output
    independent of the block size.
    """

    name = "Modulated delay"

    def __init__(self, samplerate: int, base_ms: float, depth_ms: float = 0.0, rate_hz: float = 0.0,
                 feedback: float = 0.0, wet: float = 0.5, dry: float = 1.0):
        self.sr = int(samplerate)
        self.base = base_ms * self.sr / 1000
        self.depth = depth_ms * self.sr / 1000
        if self.depth < 0 or self.base - self.depth < 2:
            raise ValueError("need 0 <= depth_ms and base_ms - depth_ms of at least 2 samples")
        if not -1 < feedback < 1:
            raise ValueError("feedback must be within (-1, 1)")
        self.rate = float(rate_hz)
        self.feedback = float(feedback)
        self.wet, self.dry = float(wet), float(dry)
        self.size = int(np.ceil(self.base + self.depth)) + 2
        self.chunk = int(np.floor(self.base - self.depth)) - 1      # every read is older than the chunk
        self.reset()

    def reset(self):
        self._buf = np.zeros(self.size)                     # the last `size` samples of v
        self._n = 0                                          # samples processed (LFO time)

    @property
    def algorithmic_latency_s(self) -> float:
        return 0.0 if self.dry else self.base / self.sr      # wet only: the voice arrives `base` late

    def process(self, block):
        x = np.asarray(block, dtype=np.float64)
        out = np.empty(x.shape[0])
        for s in range(0, x.shape[0], self.chunk):
            xs = x[s:s + self.chunk]
            k = np.arange(xs.shape[0])
            delay = self.base + self.depth * np.sin(2 * np.pi * self.rate * (self._n + k) / self.sr)
            pos = self.size + k - delay
            i = np.floor(pos).astype(int)
            fr = pos - i
            d = self._buf[i] * (1 - fr) + self._buf[i + 1] * fr
            self._buf = np.concatenate([self._buf[xs.shape[0]:], xs + self.feedback * d])
            out[s:s + xs.shape[0]] = self.dry * xs + self.wet * d
            self._n += xs.shape[0]
        return out.astype(np.float32)


class FilterProcessor(BlockProcessor):
    """Butterworth high-pass / low-pass / band-pass (IIR, state kept between blocks): no added latency.

    ``low_hz`` is the lower band edge (high-pass), ``high_hz`` the upper one (low-pass).
    """

    name = "Filter"

    def __init__(self, samplerate: int, low_hz: float | None = None, high_hz: float | None = None, order: int = 4):
        if low_hz and high_hz:
            wn, btype = [low_hz, high_hz], "bandpass"
        elif low_hz or high_hz:
            wn, btype = (low_hz, "highpass") if low_hz else (high_hz, "lowpass")
        else:
            raise ValueError("give low_hz, high_hz or both")
        self.sos = butter(int(order), wn, btype, fs=int(samplerate), output="sos")
        self.reset()

    def reset(self):
        self._zi = np.zeros((self.sos.shape[0], 2))

    def process(self, block):
        y, self._zi = sosfilt(self.sos, np.asarray(block, dtype=np.float64), zi=self._zi)
        return y.astype(np.float32)


class BitcrushProcessor(BlockProcessor):
    """Lo-fi "retro game" sound: sample-and-hold to a lower rate, then coarse quantisation."""

    name = "Bitcrush"

    def __init__(self, samplerate: int, rate_hz: float = 6000.0, bits: int = 8):
        self.hold = max(1, int(round(samplerate / rate_hz)))
        self.q = float(2 ** (int(bits) - 1))
        self.reset()

    def reset(self):
        self._pos = 0                                        # samples since the current hold started
        self._last = 0.0                                     # value being held

    def process(self, block):
        x = np.asarray(block, dtype=np.float32)
        if x.shape[0] == 0:
            return x.copy()
        j = np.arange(x.shape[0])
        start = j - (self._pos + j) % self.hold              # where each sample's hold began
        held = np.where(start >= 0, x[np.maximum(start, 0)], self._last)
        self._last = float(held[-1])
        self._pos = (self._pos + x.shape[0]) % self.hold
        return (np.round(held * self.q) / self.q).astype(np.float32)


class ToneProcessor(BlockProcessor):
    """Peaking EQ bands (RBJ biquads) plus optional Butterworth band edges, all IIR: no added latency.

    ``peaks`` is a list of (centre Hz, gain dB, Q). Used to colour a voice, e.g. a blocked nose.
    """

    name = "Tone (IIR EQ)"

    def __init__(self, samplerate: int, peaks: list[tuple[float, float, float]] = (),
                 low_hz: float | None = None, high_hz: float | None = None):
        sr = int(samplerate)
        sos = []
        for hz, db, q in peaks:
            if not 0 < hz < sr / 2 or q <= 0:
                raise ValueError("peaks must be (0 < Hz < Nyquist, gain dB, Q > 0)")
            a = 10 ** (db / 40)
            w = 2 * np.pi * hz / sr
            alpha = np.sin(w) / (2 * q)
            b0, b1, b2 = 1 + alpha * a, -2 * np.cos(w), 1 - alpha * a
            a0, a1, a2 = 1 + alpha / a, -2 * np.cos(w), 1 - alpha / a
            sos.append([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])
        if low_hz:
            sos.extend(butter(2, low_hz, "highpass", fs=sr, output="sos"))
        if high_hz:
            sos.extend(butter(2, high_hz, "lowpass", fs=sr, output="sos"))
        if not sos:
            raise ValueError("give at least one peak or band edge")
        self.sos = np.array(sos, dtype=np.float64)
        self.reset()

    def reset(self):
        self._zi = np.zeros((self.sos.shape[0], 2))

    def process(self, block):
        y, self._zi = sosfilt(self.sos, np.asarray(block, dtype=np.float64), zi=self._zi)
        return y.astype(np.float32)


class RoughnessProcessor(BlockProcessor):
    """Rough, strained voice: the level flickers irregularly 20-90 times a second (shimmer).

    Multiplies the signal by ``1 + depth * m`` where ``m`` is band-limited random noise with unit
    RMS. Works sample by sample, so it adds no latency; silence stays silence.
    """

    name = "Roughness"

    def __init__(self, samplerate: int, depth: float = 0.3, low_hz: float = 20.0, high_hz: float = 90.0,
                 seed: int = 5):
        self.depth = float(depth)
        self.seed = int(seed)
        self.sos = butter(2, [low_hz, high_hz], "bandpass", fs=int(samplerate), output="sos")
        # RMS of unit white noise through the band-pass, to normalise m to unit RMS
        w = sosfilt(self.sos, np.random.default_rng(0).standard_normal(int(samplerate) * 20))
        self.scale = 1.0 / float(np.sqrt(np.mean(w[int(samplerate):] ** 2)))
        self.reset()

    def reset(self):
        self._rng = np.random.default_rng(self.seed)
        self._zi = np.zeros((self.sos.shape[0], 2))

    def process(self, block):
        x = np.asarray(block, dtype=np.float64)
        m, self._zi = sosfilt(self.sos, self._rng.standard_normal(x.shape[0]), zi=self._zi)
        return (x * np.maximum(1.0 + self.depth * self.scale * m, 0.0)).astype(np.float32)


class DelayProcessor(BlockProcessor):
    """Pure delay by a whole number of samples (aligns parallel branches)."""

    name = "Delay"

    def __init__(self, samplerate: int, samples: int):
        self.sr = int(samplerate)
        self.samples = int(samples)
        self.reset()

    def reset(self):
        self._buf = np.zeros(self.samples, dtype=np.float32)

    @property
    def algorithmic_latency_s(self) -> float:
        return self.samples / self.sr

    def process(self, block):
        x = np.concatenate([self._buf, np.asarray(block, dtype=np.float32)])
        self._buf = x[x.shape[0] - self.samples:]
        return x[:x.shape[0] - self.samples]


class ParallelProcessor(BlockProcessor):
    """Feeds the same input to several branches and sums them with gains.

    Shorter branches are delayed so every branch lines up with the slowest one; the result is
    no later than that slowest branch alone (e.g. a voice plus a breath layer).
    """

    name = "Parallel"

    def __init__(self, samplerate: int, branches: list[tuple[BlockProcessor, float]]):
        sr = int(samplerate)
        lat = [int(round(p.algorithmic_latency_s * sr)) for p, _ in branches]
        self.branches = [(p if d == max(lat) else ChainProcessor("aligned", [p, DelayProcessor(sr, max(lat) - d)]), g)
                         for (p, g), d in zip(branches, lat)]
        self.sr = sr

    @property
    def children(self):
        return [p for p, _ in self.branches]

    def reset(self):
        for p in self.children:
            p.reset()

    @property
    def algorithmic_latency_s(self) -> float:
        return max(p.algorithmic_latency_s for p in self.children)

    def process(self, block):
        block = np.asarray(block, dtype=np.float32)
        return sum(g * p.process(block) for p, g in self.branches).astype(np.float32)


def _find(stages, cls):
    """First processor of type ``cls``, also inside chains and parallel branches."""
    for s in stages:
        if isinstance(s, cls):
            return s
        found = _find(getattr(s, "children", ()), cls)
        if found is not None:
            return found
    return None


class ChainProcessor(BlockProcessor):
    """Runs processors in series; exposes the first VoiceShiftProcessor as ``shift`` and the
    first EqProcessor as ``eq`` (for live slider changes), also when they sit in a parallel branch."""

    def __init__(self, name: str, stages: list[BlockProcessor], gain_db: float = 0.0):
        self.name = name
        self.stages = list(stages)
        self.gain = float(10 ** (gain_db / 20))
        self.shift = _find(self.stages, VoiceShiftProcessor)
        self.eq = _find(self.stages, EqProcessor)

    @property
    def children(self):
        return self.stages

    def reset(self):
        for s in self.stages:
            s.reset()

    @property
    def algorithmic_latency_s(self) -> float:
        return sum(s.algorithmic_latency_s for s in self.stages)

    def process(self, block):
        y = np.asarray(block, dtype=np.float32)
        for s in self.stages:
            y = s.process(y)
        return np.clip(y * self.gain, -1.0, 1.0).astype(np.float32)
