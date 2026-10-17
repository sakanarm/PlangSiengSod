"""Voice from sample: measure pitch, formants and tone of a recording and derive DSP settings.

This is NOT voice cloning and needs no AI model. It measures three average properties of
a reference recording and of the user's own voice:

* pitch     - fundamental frequency (YIN), median and spread over voiced speech
* formants  - F1..F3 from linear prediction (LPC) on voiced frames: the vocal-tract "size"
* tone      - long-term average spectrum in 1/3-octave bands and its slope (dB/octave)

From the differences it derives a pitch shift, a formant shift and an EQ curve for the
existing real-time DSP chain (VoiceShiftProcessor + EqProcessor). The result sounds closer
in pitch, size and brightness; rhythm, accent and timbre details stay the user's own, so it
does not sound like the same person.

Every number shown to the user comes from these measurements; nothing is estimated or
simulated. Audio stays in memory; nothing is uploaded or written to disk here.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import resample_poly

SR = 48000                         # internal analysis rate (the engine's rate)
MAX_SECONDS = 60.0                 # longer files: only the first minute is analysed
F0_MIN, F0_MAX = 60.0, 600.0
PITCH_LIMIT, FORMANT_LIMIT = 12.0, 8.0     # same ranges as the sliders
EQ_BOOST_DB, EQ_CUT_DB = 6.0, 10.0         # boosts are limited more: they also raise noise and hiss
EQ_LO_HZ, EQ_HI_HZ = 100.0, 8000.0  # EQ only where both mics and Teams carry the voice
SOUNDFILE_EXT = {".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff"}
AV_EXT = {".webm", ".m4a", ".mp4", ".opus", ".mka", ".aac"}
SUPPORTED_EXT = sorted(SOUNDFILE_EXT | AV_EXT)
# 1/3-octave band centres (ISO 266 nominal frequencies) used for the tone measurement
BANDS_HZ = (100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
            3150, 4000, 5000, 6300, 8000)


class AudioLoadError(Exception):
    pass


# ============================================================================ loading
def resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = math.gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g)


def _load_soundfile(path: Path) -> tuple[np.ndarray, int, float]:
    import soundfile as sf

    with sf.SoundFile(str(path)) as f:
        frames = min(f.frames, int(MAX_SECONDS * f.samplerate)) if f.frames > 0 else int(MAX_SECONDS * f.samplerate)
        x = f.read(frames, dtype="float32", always_2d=True)
        return x, f.samplerate, (f.frames if f.frames > 0 else x.shape[0]) / f.samplerate


def _load_av(path: Path) -> tuple[np.ndarray, int, float]:
    import av

    chunks, sr, total = [], None, 0
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise AudioLoadError("the file has no audio track")
        sr = stream.codec_context.sample_rate or stream.rate
        length = float(container.duration) / av.time_base if container.duration else 0.0
        # decode to float mono at the native rate; downmixing here avoids layout surprises
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                a = out.to_ndarray().reshape(-1)
                chunks.append(a)
                total += a.size
            if total >= MAX_SECONDS * sr:
                break
        for out in resampler.resample(None):                # flush
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks or sr is None:
        raise AudioLoadError("no audio could be decoded")
    x = np.concatenate(chunks).astype(np.float32)
    length = max(length, x.size / sr)                      # the container may not know its length
    return x[:int(MAX_SECONDS * sr), None], int(sr), length


def load_audio(path: str | Path) -> tuple[np.ndarray, float]:
    """Read a local audio file as mono float32 at ``SR``. Returns (samples, original duration s).

    WAV / FLAC / OGG / MP3 via soundfile (libsndfile), WebM / M4A / MP4 / Opus via PyAV (FFmpeg).
    Only the first ``MAX_SECONDS`` are read.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SOUNDFILE_EXT and ext not in AV_EXT:
        raise AudioLoadError(f"unsupported file type {ext or '(none)'}; use one of {', '.join(SUPPORTED_EXT)}")
    if not path.is_file():
        raise AudioLoadError(f"file not found: {path}")
    try:
        if ext in SOUNDFILE_EXT:
            try:
                x, sr, length = _load_soundfile(path)
            except Exception:
                x, sr, length = _load_av(path)              # e.g. an MP3 that libsndfile rejects
        else:
            x, sr, length = _load_av(path)
    except AudioLoadError:
        raise
    except ImportError as e:
        raise AudioLoadError(f"decoder not installed ({e.name}); run setup.ps1 again") from e
    except Exception as e:
        raise AudioLoadError(f"could not decode the file: {e}") from e
    mono = x.mean(axis=1).astype(np.float64)
    if mono.size < sr * 0.5:
        raise AudioLoadError("the recording is shorter than 0.5 s")
    return resample(mono, sr, SR).astype(np.float32), length


# ============================================================================ measurements
@dataclass
class VoiceProfile:
    duration_s: float
    voiced_s: float                     # seconds with a detected pitch
    f0_median: float | None             # Hz
    f0_p10: float | None
    f0_p90: float | None
    formants: list[float] | None        # median F1, F2, F3 in Hz
    formant_frames: int                 # voiced frames that gave all three formants
    band_db: list[float]                # level per BANDS_HZ band, dB re. the 300-3000 Hz mean
    tilt_db_oct: float | None           # spectral slope 200-5000 Hz
    centroid_hz: float | None           # "brightness"
    active_db: float                    # RMS of the non-silent part, dBFS
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        """Compact dict for storing next to a custom voice (no audio)."""
        d = asdict(self)
        d.pop("band_db")
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}


def _frames(x: np.ndarray, length: int, hop: int) -> np.ndarray:
    if x.size < length:
        x = np.pad(x, (0, length - x.size))
    return np.lib.stride_tricks.sliding_window_view(x, length)[::hop]


def _activity(x: np.ndarray, hop: int, length: int) -> tuple[np.ndarray, float]:
    """Per-frame activity mask (not silence) and the active RMS in dBFS."""
    fr = _frames(x, length, hop)
    rms_db = 10 * np.log10(np.mean(fr.astype(np.float64) ** 2, axis=1) + 1e-12)
    loud = np.percentile(rms_db, 95)
    active = (rms_db > loud - 35.0) & (rms_db > -60.0)
    level = 10 * np.log10(np.mean(10 ** (rms_db[active] / 10))) if active.any() else -120.0
    return active, float(level)


def estimate_f0(x: np.ndarray, sr: int = SR, threshold: float = 0.15) -> np.ndarray:
    """YIN pitch track, 10 ms hop. Returns Hz per frame, 0 where unvoiced or silent."""
    sr_a = 16000
    y = resample(np.asarray(x, dtype=np.float64), sr, sr_a)
    tau_min, tau_max = int(sr_a / F0_MAX), int(math.ceil(sr_a / F0_MIN))
    w, hop = int(0.025 * sr_a), int(0.010 * sr_a)
    fr = _frames(y, w + tau_max, hop)
    active, _ = _activity(y, hop, w + tau_max)
    nfft = 1 << int(math.ceil(math.log2(2 * (w + tau_max))))
    # difference function d(tau) = e0 + e_tau - 2 r(tau), all frames at once
    F = np.fft.rfft(fr, nfft, axis=1)
    H = np.fft.rfft(fr[:, :w], nfft, axis=1)
    r = np.fft.irfft(F * np.conj(H), nfft, axis=1)[:, :tau_max + 1]
    c = np.concatenate([np.zeros((fr.shape[0], 1)), np.cumsum(fr ** 2, axis=1)], axis=1)
    taus = np.arange(tau_max + 1)
    e_tau = c[:, taus + w] - c[:, taus]
    d = np.maximum(c[:, [w]] + e_tau - 2 * r, 0.0)
    cum = np.cumsum(d[:, 1:], axis=1)
    dn = np.ones_like(d)
    dn[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(cum, 1e-12)
    f0 = np.zeros(fr.shape[0])
    for i in np.flatnonzero(active):
        row = dn[i]
        below = np.flatnonzero(row[tau_min:tau_max] < threshold)
        if below.size == 0:
            continue
        t = tau_min + below[0]
        while t + 1 < tau_max and row[t + 1] < row[t]:     # walk down to the local minimum
            t += 1
        a, b, cc = row[t - 1], row[t], row[t + 1]
        den = a - 2 * b + cc
        t_ref = t + (0.5 * (a - cc) / den if den > 1e-12 else 0.0)
        f = sr_a / t_ref
        if F0_MIN <= f <= F0_MAX:
            f0[i] = f
    return f0


def estimate_formants(x: np.ndarray, sr: int = SR, f0_track: np.ndarray | None = None) -> np.ndarray:
    """F1..F3 per voiced frame by LPC (10 kHz, order 12). Returns an (N, 3) array of Hz."""
    sr_a = 10000
    y = resample(np.asarray(x, dtype=np.float64), sr, sr_a)
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])             # pre-emphasis
    w, hop, order = int(0.025 * sr_a), int(0.010 * sr_a), 12
    fr = _frames(y, w, hop)
    if f0_track is None:
        f0_track = estimate_f0(x, sr)
    n = min(fr.shape[0], f0_track.size)
    win = np.hamming(w)
    out = []
    for i in np.flatnonzero(f0_track[:n] > 0):
        s = fr[i] * win
        rr = np.correlate(s, s, "full")[w - 1:w + order]
        if rr[0] <= 1e-10:
            continue
        rr = rr * np.exp(-0.5 * (2 * np.pi * 40.0 * np.arange(order + 1) / sr_a) ** 2)   # 40 Hz lag window
        rr[0] *= 1.0001                                     # white-noise correction for stability
        try:
            a = solve_toeplitz(rr[:order], rr[1:order + 1])
        except np.linalg.LinAlgError:
            continue
        roots = np.roots(np.concatenate([[1.0], -a]))
        roots = roots[np.imag(roots) > 0]
        freq = np.angle(roots) * sr_a / (2 * np.pi)
        bw = -np.log(np.abs(roots)) * sr_a / np.pi
        cand = np.sort(freq[(bw < 400) & (freq > 150) & (freq < sr_a / 2 - 100)])
        f1 = cand[(cand >= 200) & (cand <= 1200)]
        if f1.size == 0:
            continue
        f2 = cand[(cand >= f1[0] + 200) & (cand <= 3200)]
        if f2.size == 0:
            continue
        f3 = cand[(cand >= f2[0] + 200) & (cand <= 4500)]
        if f3.size == 0:
            continue
        out.append((f1[0], f2[0], f3[0]))
    return np.array(out).reshape(-1, 3)


def band_levels(x: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    """Long-term average spectrum of the non-silent part in 1/3-octave bands.

    Returns (band dB re. the 300-3000 Hz mean, valid-band mask, tilt dB/oct, centroid Hz).
    """
    n, hop = 4096, 2048
    x = np.asarray(x, dtype=np.float64)
    active, _ = _activity(x, hop, n)
    fr = _frames(x, n, hop)[active]
    if fr.shape[0] == 0:
        return np.zeros(len(BANDS_HZ)), np.zeros(len(BANDS_HZ), bool), None, None
    psd = np.mean(np.abs(np.fft.rfft(fr * np.hanning(n), axis=1)) ** 2, axis=0)
    f = np.fft.rfftfreq(n, 1 / sr)
    levels, valid = [], []
    for fc in BANDS_HZ:
        sel = (f >= fc * 2 ** (-1 / 6)) & (f < fc * 2 ** (1 / 6))
        ok = sel.any() and fc * 2 ** (1 / 6) < sr / 2
        levels.append(10 * np.log10(np.mean(psd[sel]) + 1e-20) if ok else -200.0)
        valid.append(ok)
    levels, valid = np.array(levels), np.array(valid)
    mid = valid & (np.array(BANDS_HZ) >= 300) & (np.array(BANDS_HZ) <= 3000)
    levels = levels - levels[mid].mean()
    valid &= levels > -60.0                                 # bands with (almost) no energy say nothing
    sel = valid & (np.array(BANDS_HZ) >= 200) & (np.array(BANDS_HZ) <= 5000)
    tilt = float(np.polyfit(np.log2(np.array(BANDS_HZ)[sel]), levels[sel], 1)[0]) if sel.sum() >= 3 else None
    band = (f >= 80) & (f <= 8000)
    centroid = float(np.sum(f[band] * psd[band]) / np.sum(psd[band])) if psd[band].sum() > 0 else None
    return levels, valid, tilt, centroid


def analyze(x: np.ndarray, sr: int = SR, original_duration_s: float | None = None) -> VoiceProfile:
    """Measure one recording. ``x`` is mono float at ``sr``."""
    x = np.asarray(x, dtype=np.float64)
    dur = x.size / sr
    warnings = []
    if original_duration_s and original_duration_s > MAX_SECONDS + 0.5:
        warnings.append(f"Only the first {MAX_SECONDS:.0f} s of {original_duration_s:.0f} s were analysed.")
    _, level = _activity(x, int(0.01 * sr), int(0.04 * sr))
    f0 = estimate_f0(x, sr)
    voiced = f0[f0 > 0]
    voiced_s = voiced.size * 0.010
    fm = estimate_formants(x, sr, f0) if voiced.size else np.zeros((0, 3))
    levels, valid, tilt, centroid = band_levels(x, sr)
    clip = float(np.mean(np.abs(x) >= 0.99))
    if dur < 10:
        warnings.append(f"The recording is {dur:.1f} s long; 10-20 s of speech give steadier values.")
    if voiced_s < 3:
        warnings.append(f"Only {voiced_s:.1f} s of voiced speech found; the values are unreliable.")
    if clip > 0.001:
        warnings.append(f"{clip * 100:.1f} % of the samples are clipped (too loud); the tone may be off.")
    if level < -45:
        warnings.append(f"The recording is very quiet ({level:.0f} dBFS); background noise may dominate.")
    if voiced.size and fm.shape[0] < 50:
        warnings.append(f"Formants found in only {fm.shape[0]} frames; the voice size may be inaccurate.")
    return VoiceProfile(
        duration_s=dur, voiced_s=voiced_s,
        f0_median=float(np.median(voiced)) if voiced.size >= 20 else None,
        f0_p10=float(np.percentile(voiced, 10)) if voiced.size >= 20 else None,
        f0_p90=float(np.percentile(voiced, 90)) if voiced.size >= 20 else None,
        formants=[float(v) for v in np.median(fm, axis=0)] if fm.shape[0] >= 10 else None,
        formant_frames=int(fm.shape[0]),
        band_db=[float(v) if ok else float("nan") for v, ok in zip(levels, valid)],
        tilt_db_oct=tilt, centroid_hz=centroid, active_db=level, warnings=warnings)


# ============================================================================ matching
@dataclass
class MatchSettings:
    pitch_st: float
    formant_st: float
    eq: list[tuple[float, float]]       # (band Hz, gain dB)
    notes: list[str] = field(default_factory=list)


def _clip(v: float, lim: float, what: str, notes: list[str]) -> float:
    if abs(v) > lim:
        notes.append(f"{what} needed {v:+.1f} st but is limited to {math.copysign(lim, v):+.0f} st.")
    return float(np.clip(v, -lim, lim))


def match_pitch_formant(target: VoiceProfile, own: VoiceProfile) -> tuple[float, float, list[str]]:
    notes: list[str] = []
    if target.f0_median and own.f0_median:
        pitch = _clip(12 * math.log2(target.f0_median / own.f0_median), PITCH_LIMIT, "Pitch", notes)
    else:
        pitch = 0.0
        notes.append("Pitch could not be measured in both recordings; pitch is left at 0.")
    if target.formants and own.formants:
        ratios = [math.log2(t / o) for t, o in zip(target.formants, own.formants)]
        formant = _clip(12 * float(np.mean(ratios)), FORMANT_LIMIT, "Formant", notes)
    else:
        formant = 0.0
        notes.append("Formants could not be measured in both recordings; formant is left at 0.")
    return round(pitch, 1), round(formant, 1), notes


def eq_towards(target: VoiceProfile, shifted_own: VoiceProfile) -> list[tuple[float, float]]:
    """EQ that moves the long-term spectrum of the (already pitch/formant-shifted) own voice
    towards the target. Smoothed, limited to -EQ_CUT_DB..+EQ_BOOST_DB, 0 dB outside EQ_LO_HZ..EQ_HI_HZ."""
    t, o = np.array(target.band_db), np.array(shifted_own.band_db)
    hz = np.array(BANDS_HZ, dtype=float)
    ok = np.isfinite(t) & np.isfinite(o) & (hz >= EQ_LO_HZ) & (hz <= EQ_HI_HZ)
    diff = np.where(ok, t - o, 0.0)
    if ok.any():                                            # bridge invalid bands from their neighbours
        diff = np.interp(np.arange(hz.size), np.flatnonzero(ok), diff[ok])
    mid = ok & (hz >= 300) & (hz <= 3000)
    if mid.any():
        diff -= diff[mid].mean()                            # keep the overall loudness
    diff = np.convolve(np.pad(diff, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    diff = np.clip(diff, -EQ_CUT_DB, EQ_BOOST_DB)
    diff[(hz < EQ_LO_HZ) | (hz > EQ_HI_HZ)] = 0.0
    return [(float(h), round(float(g), 1)) for h, g in zip(hz, diff)]


def render(x: np.ndarray, pitch_st: float, formant_st: float, eq: list[tuple[float, float]] | None = None,
           eq_amount: float = 1.0, sr: int = SR) -> np.ndarray:
    """Run the same real-time DSP chain offline (for verification and the preview)."""
    from .processors import ChainProcessor, EqProcessor, VoiceShiftProcessor

    stages = [VoiceShiftProcessor(sr, pitch_st, formant_st)]
    if eq:
        stages.append(EqProcessor(sr, eq, eq_amount))
    chain = ChainProcessor("match", stages)
    delay = int(round(chain.algorithmic_latency_s * sr))
    x = np.concatenate([np.asarray(x, dtype=np.float32), np.zeros(delay, np.float32)])
    block = 4800
    y = np.concatenate([chain.process(x[i:i + block]) for i in range(0, x.size, block)])
    return y[delay:]


@dataclass
class MatchResult:
    target: VoiceProfile
    own: VoiceProfile
    after: VoiceProfile                 # own voice measured again after the full chain
    settings: MatchSettings
    after_audio: np.ndarray | None = None   # that processed own voice (for the preview, memory only)


def match(target_audio: np.ndarray, own_audio: np.ndarray, target_duration_s: float | None = None,
          own_duration_s: float | None = None, progress=lambda text: None) -> MatchResult:
    """Measure both recordings, derive pitch/formant/EQ and verify by measuring the result."""
    progress("Measuring the reference voice...")
    target = analyze(target_audio, SR, target_duration_s)
    progress("Measuring your voice...")
    own = analyze(own_audio, SR, own_duration_s)
    pitch, formant, notes = match_pitch_formant(target, own)
    progress("Shaping the tone (EQ)...")
    # The EQ is measured against the already shifted voice, so it only corrects what the
    # pitch / formant shift left over (it also compensates the microphone colouring).
    shifted = analyze(render(own_audio, pitch, formant), SR)
    eq = eq_towards(target, shifted)
    progress("Verifying: measuring your voice after matching...")
    after_audio = render(own_audio, pitch, formant, eq)
    after = analyze(after_audio, SR)
    return MatchResult(target, own, after, MatchSettings(pitch, formant, eq, notes), after_audio)
