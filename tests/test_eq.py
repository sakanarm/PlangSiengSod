import numpy as np
import pytest

from rtvc.audio.processors import ChainProcessor, EqProcessor, VoiceShiftProcessor

SR = 48000
N = 1024                                                    # EqProcessor default frame = its delay


def _run(p, x, block):
    return np.concatenate([p.process(x[i:i + block]) for i in range(0, len(x), block)])


def _noise(seconds=1.0, seed=0):
    return (0.1 * np.random.default_rng(seed).standard_normal(int(SR * seconds))).astype(np.float32)


def _tone(hz, seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return (0.2 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def test_flat_eq_is_a_pure_delay():
    x = _noise()
    y = _run(EqProcessor(SR, [(100, 0), (8000, 0)]), x, 480)
    np.testing.assert_allclose(y[N:], x[:-N], atol=1e-5)


@pytest.mark.parametrize("hz,db", [(250, 6.0), (1000, -10.0), (4000, 4.0)])
def test_band_gain_is_applied(hz, db):
    bands = [(125, 0), (250, 0), (500, 0), (1000, 0), (2000, 0), (4000, 0), (8000, 0)]
    bands = [(h, db if h == hz else 0.0) for h, _ in bands]
    p = EqProcessor(SR, bands)
    # 1024-point frames have 47 Hz bins, so a one-octave peak at 250 Hz is slightly rounded off
    assert p.gain_at(hz) == pytest.approx(db, abs=0.75)
    x = _tone(hz)
    y = _run(p, x, 480)
    measured = 20 * np.log10(_rms(y[2 * N:]) / _rms(x[2 * N:]))
    assert measured == pytest.approx(p.gain_at(hz), abs=0.5)    # what it reports is what it does


def test_gain_is_interpolated_on_log_frequency_and_held_at_the_ends():
    p = EqProcessor(SR, [(500, 0), (2000, 6)])
    assert p.gain_at(1000) == pytest.approx(3.0, abs=0.2)  # 1 kHz is halfway between 500 and 2000 in octaves
    assert p.gain_at(100) == pytest.approx(0.0, abs=1e-6)
    assert p.gain_at(10000) == pytest.approx(6.0, abs=1e-6)


def test_output_independent_of_block_size():
    x = _noise(seed=1)
    bands = [(200, -6), (1000, 2), (5000, 5)]
    a = _run(EqProcessor(SR, bands), x, 64)
    b = _run(EqProcessor(SR, bands), x, 4800)
    np.testing.assert_allclose(a, b, atol=1e-5)
    assert a.shape == x.shape and a.dtype == np.float32


def test_amount_zero_is_identity_and_can_change_live():
    x = _tone(1000, 2.0)
    p = EqProcessor(SR, [(500, -12), (1000, -12), (2000, -12)], amount=0.0)
    y0 = _run(p, x[:SR], 480)
    np.testing.assert_allclose(y0[N:], x[:SR - N], atol=1e-5)
    p.amount = 0.5                                          # the UI slider does this while running
    y1 = _run(p, x[SR:], 480)
    assert 20 * np.log10(_rms(y1[2 * N:]) / _rms(x[SR + 2 * N:])) == pytest.approx(-6.0, abs=0.5)


@pytest.mark.parametrize("bands", [[], [(0, 3)], [(-100, 3), (1000, 0)]])
def test_invalid_bands_raise(bands):
    with pytest.raises(ValueError):
        EqProcessor(SR, bands)


def test_chain_exposes_eq_for_live_updates():
    eq = EqProcessor(SR, [(1000, 3)])
    chain = ChainProcessor("test", [VoiceShiftProcessor(SR, 2, 1), eq])
    assert chain.eq is eq
    assert ChainProcessor("test", [VoiceShiftProcessor(SR, 2, 1)]).eq is None
    y = _run(chain, _noise(0.5), 480)
    assert np.all(np.isfinite(y))
