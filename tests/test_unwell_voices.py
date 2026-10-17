"""Tone / roughness / jitter / parallel processors and the 'unwell' voices built from them
(cold, sore throat, hoarse, hoarse with your own voice)."""

import numpy as np
import pytest
from scipy.signal import sosfilt, sosfreqz

from rtvc.audio.processors import (ChainProcessor, DelayProcessor, ParallelProcessor, PassthroughProcessor,
                                   RoughnessProcessor, ToneProcessor, VoiceShiftProcessor)
from rtvc.audio.voices import VOICES, make_voice

SR = 48000
UNWELL = ["cold", "sore_throat", "hoarse", "hoarse_own"]


def _run(p, x, block):
    return np.concatenate([p.process(x[i:i + block]) for i in range(0, len(x), block)])


def _noise(seconds=1.0, seed=0):
    return (0.1 * np.random.default_rng(seed).standard_normal(int(SR * seconds))).astype(np.float32)


def _harmonic(f0=150.0, seconds=3.0):
    """Steady voiced sound: harmonics with a falling spectrum (like a sustained vowel)."""
    t = np.arange(int(SR * seconds)) / SR
    x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, int(8000 / f0)))
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def _harmonicity(y, f0):
    """Autocorrelation at the pitch period: ~1 for a clean periodic voice, lower when rough/breathy."""
    seg = y[SR:SR + 16384].astype(np.float64)
    seg -= seg.mean()
    ac = np.correlate(seg, seg, "full")[seg.size - 1:]
    lag = int(round(SR / f0))
    return float(ac[lag - 20:lag + 20].max() / ac[0])


def _env_flux(y):
    """How much the 5 ms level flickers (shimmer)."""
    r = np.sqrt((y[SR:2 * SR].astype(np.float64).reshape(-1, 240) ** 2).mean(1))
    return float(r.std() / r.mean())


def _band_db(y, lo, hi):
    S = np.abs(np.fft.rfft(y[SR // 2:].astype(np.float64))) ** 2
    f = np.fft.rfftfreq(y[SR // 2:].size, 1 / SR)
    return 10 * np.log10(S[(f >= lo) & (f < hi)].sum())


def _f0_track(y, f0_guess, win=1024, hop=512):
    """Frame-by-frame pitch (autocorrelation peak near the expected period, parabolic refinement)."""
    out = []
    lo, hi = int(SR / (f0_guess * 1.4)), int(SR / (f0_guess / 1.4))
    for i in range(SR // 2, y.size - 2 * win, hop):
        seg = y[i:i + 2 * win].astype(np.float64)
        ac = np.array([np.dot(seg[:win], seg[k:k + win]) for k in range(lo - 1, hi + 2)])
        j = int(np.argmax(ac[1:-1])) + 1
        a, b, c = ac[j - 1], ac[j], ac[j + 1]
        out.append(SR / (lo - 1 + j + 0.5 * (a - c) / (a - 2 * b + c)))
    return np.array(out)


# ------------------------------------------------------------------ tone (IIR EQ)
def test_tone_peak_gain_matches_the_setting():
    p = ToneProcessor(SR, peaks=[(450.0, 5.0, 1.0), (2500.0, -6.0, 1.0)])
    _, h = sosfreqz(p.sos, worN=[450.0, 2500.0], fs=SR)
    np.testing.assert_allclose(20 * np.log10(np.abs(h)), [5.0, -6.0], atol=0.3)
    _, h = sosfreqz(p.sos, worN=[20.0, 20000.0], fs=SR)
    np.testing.assert_allclose(20 * np.log10(np.abs(h)), [0.0, 0.0], atol=0.3)    # flat far away


def test_tone_equals_one_shot_filtering():
    p = ToneProcessor(SR, peaks=[(1500.0, 3.0, 1.2)], low_hz=100.0, high_hz=4500.0)
    x = _noise(1.0)
    np.testing.assert_allclose(_run(p, x, 437), sosfilt(p.sos, x.astype(np.float64)), atol=1e-6)
    assert p.algorithmic_latency_s == 0.0


@pytest.mark.parametrize("kw", [dict(), dict(peaks=[(0.0, 3.0, 1.0)]), dict(peaks=[(30000.0, 3.0, 1.0)]),
                                dict(peaks=[(1000.0, 3.0, 0.0)])])
def test_tone_rejects_bad_settings(kw):
    with pytest.raises(ValueError):
        ToneProcessor(SR, **kw)


# ------------------------------------------------------------------ roughness
def test_roughness_flickers_the_level_without_delay_or_level_change():
    x = _harmonic(150.0, 3.0)
    p = RoughnessProcessor(SR, depth=0.4)
    y = _run(p, x, 2400)
    assert p.algorithmic_latency_s == 0.0
    assert _env_flux(y) > _env_flux(x) + 0.15
    assert abs(_band_db(y, 0, SR / 2) - _band_db(x, 0, SR / 2)) < 1.0
    assert np.corrcoef(y, x)[0, 1] > 0.9                                   # same voice, same timing


def test_roughness_keeps_silence_and_is_reproducible():
    p = RoughnessProcessor(SR)
    assert not np.any(_run(p, np.zeros(SR, np.float32), 480))
    x = _harmonic(150.0, 1.0)
    p.reset()
    a = _run(p, x, 2400)
    p.reset()
    np.testing.assert_array_equal(a, _run(p, x, 1000))                     # also block-size independent


# ------------------------------------------------------------------ jitter
def test_jitter_wobbles_the_pitch_around_the_target():
    x = _harmonic(150.0, 4.0)
    target = 150.0 * 2 ** (-2 / 12)
    steady = _f0_track(_run(VoiceShiftProcessor(SR, -2.0, 0.0), x, 2400), target)
    rough = _f0_track(_run(VoiceShiftProcessor(SR, -2.0, 0.0, jitter_st=0.35), x, 2400), target)
    st = lambda f: 12 * np.log2(f / target)                               # noqa: E731
    assert np.std(st(steady)) < 0.05
    assert np.std(st(rough)) > 0.1
    assert abs(np.median(st(rough))) < 0.3                                 # still the chosen pitch on average


def test_jitter_is_reproducible_after_reset():
    x = _harmonic(150.0, 1.0)
    p = VoiceShiftProcessor(SR, -1.0, 0.0, jitter_st=0.3)
    a = _run(p, x, 2400)
    p.reset()
    np.testing.assert_array_equal(a, _run(p, x, 2400))


# ------------------------------------------------------------------ parallel / delay
def test_delay_processor_is_an_exact_delay():
    x = _noise(0.5)
    p = DelayProcessor(SR, 1000)
    y = _run(p, x, 333)
    assert p.algorithmic_latency_s == pytest.approx(1000 / SR)
    np.testing.assert_array_equal(y[1000:], x[:-1000])
    assert not np.any(y[:1000])


def test_parallel_aligns_branches_to_the_slowest():
    slow = DelayProcessor(SR, 700)
    p = ParallelProcessor(SR, [(slow, 0.5), (PassthroughProcessor(), 0.5)])
    assert p.algorithmic_latency_s == pytest.approx(700 / SR)
    x = _noise(0.5)
    y = _run(p, x, 480)
    np.testing.assert_allclose(y[700:], x[:-700], atol=1e-6)              # both halves arrive together


def test_chain_finds_the_pitch_shifter_inside_parallel_branches():
    for key in ("sore_throat", "hoarse"):
        p = make_voice(key, SR)
        assert isinstance(p.shift, VoiceShiftProcessor) and VOICES[key].adjustable
    assert make_voice("hoarse_own", SR).shift is None and not VOICES["hoarse_own"].adjustable
    p = make_voice("hoarse", SR, pitch_st=3.0)
    assert p.shift.pitch_st == pytest.approx(3.0)


# ------------------------------------------------------------------ the unwell voices
@pytest.fixture(scope="module")
def outputs():
    x = _harmonic(150.0, 3.0)
    return x, {k: _run(make_voice(k, SR), x, 2400) for k in ["deep_male", *UNWELL]}


@pytest.mark.parametrize("key,latency_s", [("cold", 2048 / SR), ("sore_throat", 2048 / SR), ("hoarse", 2048 / SR),
                                           ("hoarse_own", 1024 / SR)])
def test_unwell_latency(key, latency_s):
    assert make_voice(key, SR).algorithmic_latency_s == pytest.approx(latency_s)
    assert VOICES[key].group == "person"


def test_cold_sounds_blocked(outputs):
    x, y = outputs
    v = VOICES["cold"]
    plain = _run(VoiceShiftProcessor(SR, v.pitch_st, v.formant_st), x, 2400)   # same shift, no 'cold' tone
    tilt = lambda s: _band_db(s, 300, 600) - _band_db(s, 2000, 4000)          # noqa: E731
    assert tilt(y["cold"]) > tilt(plain) + 6                               # boomy lows, dull highs
    assert _band_db(y["cold"], 6000, 20000) < _band_db(plain, 6000, 20000) - 4
    assert _harmonicity(y["cold"], 150 * 2 ** (v.pitch_st / 12)) > 0.9     # still a clean voice


@pytest.mark.parametrize("key", ["sore_throat", "hoarse"])
def test_sore_and_hoarse_are_rough_but_still_voiced(outputs, key):
    x, y = outputs
    f0 = 150 * 2 ** (VOICES[key].pitch_st / 12)
    assert 0.4 < _harmonicity(y[key], f0) < 0.8                            # breathy / rough, not noise
    assert _harmonicity(y["deep_male"], 150 * 2 ** (VOICES["deep_male"].pitch_st / 12)) > 0.9
    assert _env_flux(y[key]) > _env_flux(x) + 0.08


def test_hoarse_is_rougher_than_sore_throat(outputs):
    _, y = outputs
    assert _env_flux(y["hoarse"]) > _env_flux(y["sore_throat"])


def test_hoarse_own_keeps_your_pitch_and_timing(outputs):
    x, y = outputs
    out = y["hoarse_own"]
    lag = 1024
    assert np.corrcoef(out[lag + SR // 2:], x[SR // 2:x.size - lag])[0, 1] > 0.8
    assert _harmonicity(out, 150.0) > 0.6                                  # same 150 Hz pitch is there
    assert _env_flux(out) > _env_flux(x) + 0.15                            # but rough
