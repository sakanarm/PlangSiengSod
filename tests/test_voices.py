import time

import numpy as np
import pytest

from rtvc.audio.processors import (ChainProcessor, RadioProcessor, ReverbProcessor, VoiceShiftProcessor,
                                   WhisperProcessor)
from rtvc.audio.voices import DEFAULT_VOICE, VOICES, key_for_label, make_voice
from rtvc.config import AppConfig

SR = 48000
ALL = list(VOICES)


def _harmonic(f0=150.0, seconds=2.0, formant=800.0):
    """Vowel-like test signal: harmonics of f0 shaped by one formant."""
    t = np.arange(int(SR * seconds)) / SR
    x = np.zeros_like(t)
    for h in range(1, 60):
        if f0 * h > 10000:
            break
        x += (1.0 / (1 + ((f0 * h - formant) / 300) ** 2) + 0.05) * np.sin(2 * np.pi * f0 * h * t)
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def _run(p, x, block):
    return np.concatenate([p.process(x[i:i + block]) for i in range(0, len(x), block)])


def _f0(x):
    seg = x[len(x) // 2: len(x) // 2 + 8192].astype(np.float64)
    seg = seg - seg.mean()
    ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
    lo, hi = int(SR / 800), int(SR / 50)
    lag = lo + int(np.argmax(ac[lo:hi]))
    # parabolic interpolation for sub-sample accuracy
    a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
    return SR / (lag + 0.5 * (a - c) / (a - 2 * b + c))


def _centroid(x):
    seg = x[len(x) // 2: len(x) // 2 + 16384]
    mag = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
    f = np.fft.rfftfreq(seg.size, 1 / SR)
    return float(np.sum(f * mag) / np.sum(mag))


def _rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


# ------------------------------------------------------------------ every voice
@pytest.mark.parametrize("key", ALL)
@pytest.mark.parametrize("block", [64, 441, 2400])
def test_voice_block_length_dtype_and_range(key, block):
    p = make_voice(key, SR)
    x = _harmonic(seconds=0.6) * 3                              # loud input: output must still be clipped
    for i in range(0, len(x), block):
        y = p.process(x[i:i + block])
        assert y.shape == x[i:i + block].shape and y.dtype == np.float32
        assert np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0


@pytest.mark.parametrize("key", ALL)
def test_voice_independent_of_block_size(key):
    x = _harmonic(seconds=1.0)
    a = _run(make_voice(key, SR), x, 480)
    b = _run(make_voice(key, SR), x, 4800)
    np.testing.assert_allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("key", ALL)
def test_voice_silence_in_silence_out(key):
    assert np.max(np.abs(_run(make_voice(key, SR), np.zeros(SR, np.float32), 480))) == 0


@pytest.mark.parametrize("key", ALL)
def test_voice_reset_reproducible(key):
    p = make_voice(key, SR)
    x = _harmonic(seconds=0.5)
    y1 = _run(p, x, 480)
    p.reset()
    np.testing.assert_allclose(_run(p, x, 480), y1, atol=1e-6)


@pytest.mark.parametrize("key", ALL)
def test_voice_level_close_to_input(key):
    x = _harmonic(seconds=2.0)
    y = _run(make_voice(key, SR), x, 2400)[SR // 2:]
    assert abs(_rms_db(y) - _rms_db(x[SR // 2:])) < 6.0        # audible, not blown up or vanished


@pytest.mark.parametrize("key", ALL)
def test_voice_runs_faster_than_real_time(key):
    p = make_voice(key, SR)
    x = _harmonic(seconds=2.0)
    t0 = time.perf_counter()
    _run(p, x, 2400)
    rtf = (time.perf_counter() - t0) / 2.0
    assert rtf < 0.5, f"{key}: real-time factor {rtf:.2f}"


def test_voice_catalog_consistency():
    labels = [v.label for v in VOICES.values()]
    assert len(set(labels)) == len(labels)
    for k, v in VOICES.items():
        assert v.key == k and key_for_label(v.label) == k and v.group in ("person", "effect")
        assert -12 <= v.pitch_st <= 12 and -8 <= v.formant_st <= 8
        assert not make_voice(k, SR).is_ai_conversion
    assert sum(v.group == "person" for v in VOICES.values()) >= 5
    d = VOICES[DEFAULT_VOICE]
    c = AppConfig()
    assert (c.voice, c.voice_pitch_st, c.voice_formant_st) == (DEFAULT_VOICE, d.pitch_st, d.formant_st)
    with pytest.raises(ValueError):
        make_voice("nobody", SR)


def test_config_validates_voice():
    assert any("voice must be one of" in e for e in AppConfig(voice="nobody").validate())
    errs = " ".join(AppConfig(voice_pitch_st=30, voice_formant_st=-20).validate())
    assert "voice_pitch_st" in errs and "voice_formant_st" in errs
    assert AppConfig(mode="voice", voice="female", voice_pitch_st=8, voice_formant_st=3).validate() == []


# ------------------------------------------------------------------ pitch / formant shift
def test_shift_identity_is_a_pure_delay():
    p = VoiceShiftProcessor(SR, 0.0, 0.0)
    x = _harmonic(seconds=1.0)
    y = _run(p, x, 480)
    d = p.n
    assert p.algorithmic_latency_s == pytest.approx(d / SR)
    np.testing.assert_allclose(y[d:], x[:-d], atol=1e-4)


@pytest.mark.parametrize("st", [-7, -4, 3, 6, 9, 12])
def test_shift_pitch_ratio(st):
    x = _harmonic(150.0)
    y = _run(VoiceShiftProcessor(SR, st, 0.0), x, 2400)
    assert _f0(y) == pytest.approx(150.0 * 2 ** (st / 12), rel=0.01)


@pytest.mark.parametrize("key", [k for k, v in VOICES.items() if v.adjustable and v.pitch_st])
def test_preset_pitch(key):
    v = VOICES[key]
    x = _harmonic(120.0)
    y = _run(make_voice(key, SR), x, 2400)
    if key == "alien":                                          # vibrato: pitch moves around the target
        return
    assert _f0(y) == pytest.approx(120.0 * 2 ** (v.pitch_st / 12), rel=0.02)


def _formant_hz(x, f0):
    """Formant peak: parabola through the log amplitudes of the strongest harmonic and its neighbours."""
    seg = x[len(x) // 2: len(x) // 2 + 32768]
    mag = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
    df = SR / seg.size
    hs = np.arange(1, int(3000 / f0))
    amp = np.log([mag[int(round(h * f0 / df)) - 3:int(round(h * f0 / df)) + 4].max() for h in hs])
    i = int(np.argmax(amp[1:-1])) + 1
    a, b, c = amp[i - 1], amp[i], amp[i + 1]
    return (hs[i] + 0.5 * (a - c) / (a - 2 * b + c)) * f0


def test_formant_shift_moves_formant_not_pitch():
    # The envelope runs through the harmonics, so a formant is only known to within about half a
    # harmonic spacing (~1.5 semitones at 600 Hz for a 110 Hz voice); the average must be close.
    errors = []
    for f0, formant in [(150, 800), (110, 600), (95, 500), (130, 1500), (220, 1000)]:
        x = _harmonic(f0, formant=formant)
        for formant_st in (4.0, -4.0):
            y = _run(VoiceShiftProcessor(SR, 0.0, formant_st), x, 2400)
            errors.append(12 * np.log2(_formant_hz(y, f0) / _formant_hz(x, f0)) - formant_st)
            assert _f0(y) == pytest.approx(f0, rel=0.01)
    assert np.max(np.abs(errors)) < 1.6 and np.sqrt(np.mean(np.square(errors))) < 0.9, errors


@pytest.mark.parametrize("st,formant_st", [(9, 0), (12, 0), (-7, 0), (8, 3), (11, 4.5), (12, 8), (-6, -3.5)])
@pytest.mark.parametrize("f0", [95, 150])
def test_shift_output_stays_harmonic(st, formant_st, f0):
    """No inharmonic junk: almost all energy (100 Hz..4 kHz) sits on the new harmonics."""
    want = f0 * 2 ** (st / 12)
    y = _run(VoiceShiftProcessor(SR, st, formant_st), _harmonic(f0), 2400)
    assert _f0(y) == pytest.approx(want, rel=0.01)
    seg = y[len(y) // 2: len(y) // 2 + 32768].astype(np.float64)
    S = np.abs(np.fft.rfft(seg * np.hanning(seg.size))) ** 2
    f = np.fft.rfftfreq(seg.size, 1 / SR)
    band = (f > 80) & (f < 4000)
    near = band & (np.abs(f / want - np.round(f / want)) * want < 15)
    assert S[near].sum() / S[band].sum() > 0.98


def test_shift_live_parameter_change():
    p = VoiceShiftProcessor(SR, 0.0, 0.0)
    x = _harmonic(150.0, seconds=3.0)
    first = _run(p, x[:SR], 2400)
    p.pitch_st, p.formant_st = 5.0, 2.0
    second = _run(p, x[SR:], 2400)
    assert np.all(np.isfinite(second)) and np.max(np.abs(second)) <= 1.0
    assert _f0(first) == pytest.approx(150.0, rel=0.01)
    assert _f0(second) == pytest.approx(150.0 * 2 ** (5 / 12), rel=0.01)


def test_chain_exposes_shift_stage():
    p = make_voice("female", SR)
    assert isinstance(p, ChainProcessor) and isinstance(p.shift, VoiceShiftProcessor)
    assert make_voice("whisper", SR).shift is None
    assert make_voice("cave", SR).algorithmic_latency_s == pytest.approx(2048 / SR + 1024 / SR)


# ------------------------------------------------------------------ effects
def test_whisper_removes_pitch():
    y = _run(WhisperProcessor(SR), _harmonic(150.0), 2400)[SR // 2:]
    seg = y[:8192].astype(np.float64)
    ac = np.correlate(seg, seg, "full")[seg.size - 1:]
    lo, hi = int(SR / 500), int(SR / 60)
    assert np.max(ac[lo:hi]) / ac[0] < 0.4                      # no periodicity left (input: ~1.0)
    assert _rms_db(y) > _rms_db(_harmonic(150.0)) - 6


def test_radio_band_limits():
    rng = np.random.default_rng(0)
    x = (0.1 * rng.standard_normal(2 * SR)).astype(np.float32)
    y = _run(RadioProcessor(SR, drive=0.01), x, 2400)[SR // 2:]   # ~linear drive to measure the filter
    f = np.fft.rfftfreq(y.size, 1 / SR)
    Y = np.abs(np.fft.rfft(y)) ** 2
    band = Y[(f > 800) & (f < 2500)].mean()
    assert Y[f < 150].mean() < band * 1e-3 and Y[f > 6000].mean() < band * 1e-3


def test_reverb_equals_direct_convolution_and_has_tail():
    p = ReverbProcessor(SR, rt60_s=0.5, wet=1.0, dry=0.0)
    x = np.zeros(SR, np.float32)
    x[:4800] = (0.1 * np.random.default_rng(1).standard_normal(4800)).astype(np.float32)
    y = _run(p, x, 441)
    ref = np.concatenate([np.zeros(p.P), np.convolve(x.astype(np.float64), p.ir)])[:x.size]
    np.testing.assert_allclose(y, ref, atol=1e-6)
    assert np.max(np.abs(y[int(0.2 * SR):int(0.3 * SR)])) > 1e-3   # still ringing after the input stopped
