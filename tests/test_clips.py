import numpy as np
import pytest

from rtvc.audio.clips import render
from rtvc.audio.processors import BlockProcessor, PassthroughProcessor
from rtvc.audio.voices import VOICES, make_voice

SR = 48000


class Delay(BlockProcessor):
    """Delays the signal by ``n`` samples, like a processor with look-ahead."""

    name = "delay"

    def __init__(self, n):
        self.n, self.buf = n, np.zeros(n, np.float32)

    def process(self, block):
        y = np.concatenate([self.buf, block])
        self.buf = y[block.size:]
        return y[:block.size]

    @property
    def algorithmic_latency_s(self):
        return self.n / SR


def _speechish(seconds=1.3):
    """Voiced-speech stand-in: 150 Hz with harmonics up to ~4.5 kHz, amplitude-modulated like syllables."""
    t = np.arange(int(seconds * SR)) / SR
    tone = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 31))
    return (0.15 * tone * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def test_render_passthrough_is_identity_for_odd_lengths():
    x = _speechish(1.2345)
    y = render(PassthroughProcessor(), x, SR)
    assert y.shape == x.shape and y.dtype == np.float32
    np.testing.assert_array_equal(y, x)


def test_render_removes_the_processor_delay():
    x = _speechish()
    y = render(Delay(1000), x, SR)
    assert y.shape == x.shape
    np.testing.assert_allclose(y, x, atol=1e-7)


def test_render_applies_gain_clip_and_gate_like_the_engine():
    x = _speechish()
    np.testing.assert_allclose(render(PassthroughProcessor(), x, SR, gain_db=-6.0), x * 10 ** (-6 / 20), atol=1e-6)
    assert np.max(np.abs(render(PassthroughProcessor(), x, SR, gain_db=12.0))) <= 1.0
    quiet = np.full(SR, 1e-3, np.float32)          # -60 dBFS hum
    assert not np.any(render(PassthroughProcessor(), quiet, SR, gate_db=-45.0))
    assert np.any(render(PassthroughProcessor(), quiet, SR, gate_db=-60.0))   # -60 = gate off


def test_render_rejects_broken_output():
    class Bad(BlockProcessor):
        name = "bad"

        def process(self, block):
            return np.full_like(block, np.nan)

    with pytest.raises(RuntimeError, match="bad"):
        render(Bad(), _speechish(), SR)


@pytest.mark.parametrize("key", list(VOICES))
def test_render_every_voice_gives_finite_audible_audio(key):
    x = _speechish()
    p = make_voice(key, SR)
    p.reset()
    y = render(p, x, SR)
    assert y.shape == x.shape and np.all(np.isfinite(y))
    assert np.sqrt(np.mean(y[SR // 4:] ** 2)) > 1e-3, key
