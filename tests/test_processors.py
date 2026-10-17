import numpy as np
import pytest

from rtvc.audio.processors import PassthroughProcessor, RobotVoiceProcessor

SR = 48000


def _voice_like(seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    f = 180 * (1 + 0.2 * np.sin(2 * np.pi * 3 * t))
    return (0.3 * np.sin(2 * np.pi * np.cumsum(f) / SR)).astype(np.float32)


def _run(p, x, block):
    return np.concatenate([p.process(x[i:i + block]) for i in range(0, len(x), block)])


@pytest.mark.parametrize("block", [64, 441, 2400, 4800])
def test_robot_preserves_block_length(block):
    p = RobotVoiceProcessor(SR)
    x = _voice_like(0.5)
    for i in range(0, len(x), block):
        assert p.process(x[i:i + block]).shape[0] == len(x[i:i + block])


def test_robot_output_independent_of_block_size():
    x = _voice_like()
    a = _run(RobotVoiceProcessor(SR, ring_mod_hz=0), x, 480)
    b = _run(RobotVoiceProcessor(SR, ring_mod_hz=0), x, 4800)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_robot_monotone_pitch_and_level():
    p = RobotVoiceProcessor(SR, robot_pitch_hz=100, ring_mod_hz=0)
    t = np.arange(2 * SR) / SR                                  # steady vowel-like input (no FM sidebands)
    x = (0.2 * np.sin(2 * np.pi * 180 * t) + 0.1 * np.sin(2 * np.pi * 360 * t)).astype(np.float32)
    y = _run(p, x, 2400)[SR // 2:]
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0
    assert 0.05 < float(np.std(y)) < 0.5                       # audible, not blown up
    spec = np.abs(np.fft.rfft(y[:SR]))
    freqs = np.fft.rfftfreq(SR, 1 / SR)
    f0 = SR / p.hop
    for h in freqs[np.argsort(spec)[-3:]]:                      # energy on multiples of the robot pitch
        assert abs(h / f0 - round(h / f0)) < 0.02
    assert abs(float(np.mean(y))) < 1e-3                        # no DC offset


def test_robot_silence_in_silence_out():
    p = RobotVoiceProcessor(SR)
    assert np.max(np.abs(_run(p, np.zeros(SR, np.float32), 480))) == 0


def test_robot_reset_and_ring_mod_continuity():
    p = RobotVoiceProcessor(SR, ring_mod_hz=50, ring_mix=1.0)
    y1 = _run(p, _voice_like(), 480)
    p.reset()
    y2 = _run(p, _voice_like(), 480)
    np.testing.assert_allclose(y1, y2, atol=1e-6)


def test_robot_rejects_bad_pitch():
    with pytest.raises(ValueError):
        RobotVoiceProcessor(SR, robot_pitch_hz=10)


def test_robot_latency_reported():
    p = RobotVoiceProcessor(SR, robot_pitch_hz=100)
    assert p.algorithmic_latency_s == pytest.approx(4 * 480 / SR)


def test_passthrough():
    x = _voice_like(0.1)
    np.testing.assert_array_equal(PassthroughProcessor().process(x), x)
    assert not PassthroughProcessor.is_ai_conversion and not RobotVoiceProcessor.is_ai_conversion
