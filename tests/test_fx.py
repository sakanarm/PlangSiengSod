"""Low-latency effect processors (delay / filter / bitcrush) and the voices built from them."""

import numpy as np
import pytest
from scipy.signal import sosfilt

from rtvc.audio.processors import BitcrushProcessor, FilterProcessor, ModDelayProcessor
from rtvc.audio.voices import VOICES, make_voice

SR = 48000
NEW = ["retro", "jet", "underwater", "ghost", "stadium"]


def _run(p, x, block):
    return np.concatenate([p.process(x[i:i + block]) for i in range(0, len(x), block)])


def _noise(seconds=1.0, seed=0):
    return (0.1 * np.random.default_rng(seed).standard_normal(int(SR * seconds))).astype(np.float32)


def _tone(hz, seconds=1.0, amp=0.3):
    return (amp * np.sin(2 * np.pi * hz * np.arange(int(SR * seconds)) / SR)).astype(np.float32)


def _band_energy(x, lo, hi):
    S = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    f = np.fft.rfftfreq(x.size, 1 / SR)
    return S[(f >= lo) & (f < hi)].sum() / S.sum()


# ------------------------------------------------------------------ modulated delay
def test_echo_positions_and_feedback():
    p = ModDelayProcessor(SR, base_ms=100.0, feedback=0.5, wet=0.4, dry=1.0)
    x = np.zeros(SR, np.float32)
    x[0] = 1.0
    y = _run(p, x, 441)
    d = int(0.1 * SR)
    assert y[0] == pytest.approx(1.0)                                   # dry: no delay at all
    assert y[d] == pytest.approx(0.4) and y[2 * d] == pytest.approx(0.2) and y[3 * d] == pytest.approx(0.1)
    rest = np.delete(y, [0, d, 2 * d, 3 * d, 4 * d, 5 * d, 6 * d, 7 * d, 8 * d, 9 * d])
    assert np.max(np.abs(rest)) < 1e-6


def test_swept_delay_matches_direct_interpolation():
    """Without feedback the wet path is exactly x read at n - delay(n)."""
    base, depth, rate = 8.0, 2.5, 1.2
    x = _noise(1.0)
    y = _run(ModDelayProcessor(SR, base, depth, rate, wet=1.0, dry=0.0), x, 480)
    n = np.arange(x.size)
    delay = (base + depth * np.sin(2 * np.pi * rate * n / SR)) * SR / 1000
    ref = np.interp(n - delay, np.arange(-1, x.size), np.concatenate([[0.0], x]), left=0.0)   # silence before
    np.testing.assert_allclose(y, ref, atol=1e-6)


def test_feedback_matches_sample_by_sample_reference():
    base, depth, rate, fb, wet, dry = 3.0, 2.0, 0.3, 0.6, 0.7, 0.7
    x = _noise(0.25, seed=3).astype(np.float64)
    y = _run(ModDelayProcessor(SR, base, depth, rate, fb, wet, dry), x.astype(np.float32), 333)
    v = np.zeros(x.size)
    ref = np.zeros(x.size)
    for n in range(x.size):
        pos = n - (base + depth * np.sin(2 * np.pi * rate * n / SR)) * SR / 1000
        i = int(np.floor(pos))
        a = v[i] if i >= 0 else 0.0
        b = v[i + 1] if i + 1 >= 0 else 0.0
        d = a + (b - a) * (pos - i)
        v[n] = x[n] + fb * d
        ref[n] = dry * x[n] + wet * d
    np.testing.assert_allclose(y, ref, atol=1e-5)


@pytest.mark.parametrize("kw", [dict(base_ms=1.0, depth_ms=1.0), dict(base_ms=5.0, depth_ms=-1.0),
                                dict(base_ms=5.0, feedback=1.0)])
def test_mod_delay_rejects_bad_settings(kw):
    with pytest.raises(ValueError):
        ModDelayProcessor(SR, **kw)


# ------------------------------------------------------------------ filter
def test_filter_equals_one_shot_filtering():
    p = FilterProcessor(SR, low_hz=300.0, high_hz=4500.0, order=2)
    x = _noise(1.0)
    np.testing.assert_allclose(_run(p, x, 441), sosfilt(p.sos, x.astype(np.float64)), atol=1e-6)


def test_lowpass_cuts_highs():
    x = _noise(1.0)
    y = _run(FilterProcessor(SR, high_hz=900.0), x, 480)
    assert _band_energy(y, 3000, SR / 2) < 1e-3 < _band_energy(x, 3000, SR / 2)


def test_filter_needs_a_cutoff():
    with pytest.raises(ValueError):
        FilterProcessor(SR)


# ------------------------------------------------------------------ bitcrush
def test_bitcrush_holds_and_quantises():
    p = BitcrushProcessor(SR, rate_hz=6000.0, bits=8)
    y = _run(p, _tone(440.0), 437)                                       # odd block: holds cross blocks
    assert p.hold == 8
    np.testing.assert_allclose(y * 128, np.round(y * 128), atol=1e-4)   # on the 8-bit grid
    steps = y.reshape(-1, 8)
    assert np.all(steps == steps[:, :1])                                 # constant within every hold
    assert np.max(np.abs(y - _tone(440.0))) < 0.25                       # still the same tone, no delay


def test_bitcrush_adds_aliasing():
    y = _run(make_voice("retro", SR), _tone(2500.0), 2400)
    assert _band_energy(y, 3000, SR / 2) > 0.05                          # images of the 2.5 kHz tone


# ------------------------------------------------------------------ the new voices
@pytest.mark.parametrize("key,latency_s", [("retro", 0.0), ("jet", 0.0), ("stadium", 0.0),
                                           ("underwater", 0.008), ("ghost", 2048 / SR)])
def test_new_voices_add_little_latency(key, latency_s):
    p = make_voice(key, SR)
    assert p.algorithmic_latency_s == pytest.approx(latency_s, abs=1e-4)
    assert p.shift is None and not VOICES[key].adjustable and VOICES[key].group == "effect"


def test_underwater_is_muffled_and_wobbles():
    x = _tone(600.0, 3.0)
    y = _run(make_voice("underwater", SR), x, 2400)
    assert _band_energy(make_voice("underwater", SR).process(_noise(1.0)), 3000, SR / 2) < 1e-3
    seg = y[SR // 2:]
    zc = np.flatnonzero((seg[:-1] < 0) & (seg[1:] >= 0))                # instantaneous frequency
    f = SR / np.diff(zc)
    st = 12 * np.log2(f / 600.0)
    assert np.ptp(st) > 0.3 and np.max(np.abs(st)) < 1.0                # audible but gentle wobble


def test_stadium_has_echoes():
    x = np.zeros(SR, np.float32)
    x[:480] = _tone(1000.0, 0.01)
    y = _run(make_voice("stadium", SR), x, 2400)
    d = int(0.22 * SR)
    env = [np.abs(y[k * d:k * d + 600]).max() for k in range(4)]
    assert env[0] > env[1] > env[2] > env[3] > 0.01                      # decaying repeats every 220 ms
    assert np.abs(y[700:d - 100]).max() < 0.01                           # silence between them


def test_jet_sweeps_its_delay():
    """The flanger's echo of the input moves between ~1 and ~5 ms (the 'whoosh')."""
    x = _noise(4.0, seed=5)
    y = _run(make_voice("jet", SR), x, 2400)
    lags = []
    for i in range(0, x.size - 4800, 4800):
        xs, ys = x[i:i + 4800].astype(np.float64), y[i:i + 4800].astype(np.float64)
        cc = [np.dot(ys[k:], xs[:xs.size - k]) for k in range(40, 260)]
        lags.append(40 + int(np.argmax(cc)))
    assert min(lags) < 0.0015 * SR and max(lags) > 0.0045 * SR, lags


def _vowel(f0=150.0, seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    x = sum((1 / (1 + ((f0 * h - 800) / 300) ** 2) + 0.05) * np.sin(2 * np.pi * f0 * h * t) for h in range(1, 60))
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def test_ghost_has_no_pitch_and_a_long_tail():
    x = np.concatenate([_vowel(), np.zeros(SR, np.float32)])
    y = _run(make_voice("ghost", SR), x, 2400)
    seg = y[SR // 2:SR // 2 + 8192].astype(np.float64)
    ac = np.correlate(seg, seg, "full")[seg.size - 1:]
    assert np.max(ac[int(SR / 500):int(SR / 60)]) / ac[0] < 0.5          # the voice's pitch is gone
    assert np.sqrt(np.mean(y[int(1.5 * SR):int(1.7 * SR)] ** 2)) > 0.003  # still haunting 0.5 s later


def test_whisper_and_ghost_are_separate_entries():
    assert "ผี" not in VOICES["whisper"].label and "ผี" in VOICES["ghost"].label
