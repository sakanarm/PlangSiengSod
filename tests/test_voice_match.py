"""Voice-from-sample analysis on synthetic voices with known pitch and formants."""

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import lfilter

from rtvc.audio import voice_match as vm

SR = vm.SR


def synth(f0=120.0, formants=(700, 1200, 2600), seconds=12.0, seed=0, sr=SR):
    """Glottal pulse train with slow intonation, three formant resonators and syllable gaps."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    f = f0 * (1 + 0.06 * np.sin(2 * np.pi * 0.4 * t))
    pulses = np.diff(np.floor(np.cumsum(f / sr)), prepend=0.0)
    y = lfilter([1.0], [1, -0.97], pulses)
    for fc, bw in zip(formants, (80, 110, 160)):
        r = np.exp(-np.pi * bw / sr)
        y = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * fc / sr), r * r], y)
    env = (np.sin(2 * np.pi * 1.5 * t) > -0.3).astype(float)
    env = np.convolve(env, np.hanning(int(sr / 50)) / (sr / 100), "same")
    y = y * env + 1e-4 * rng.standard_normal(t.size)
    return (0.3 * y / np.abs(y).max()).astype(np.float32)


@pytest.fixture(scope="module")
def low():
    return synth(120, (620, 1150, 2450), seconds=8, seed=1)


@pytest.fixture(scope="module")
def high():
    return synth(210, (800, 1430, 2850), seconds=8, seed=2)


@pytest.fixture(scope="module")
def result(low, high):
    return vm.match(high, low)


def test_pitch_is_measured(low, high):
    for x, f0 in ((low, 120), (high, 210)):
        p = vm.analyze(x)
        assert p.f0_median == pytest.approx(f0, rel=0.02)
        assert p.f0_p10 < p.f0_median < p.f0_p90


def test_formants_are_measured(low, high):
    for x, fm in ((low, (620, 1150, 2450)), (high, (800, 1430, 2850))):
        p = vm.analyze(x)
        assert p.formants is not None
        for got, want in zip(p.formants, fm):
            assert got == pytest.approx(want, rel=0.12)


def test_tone_bands_and_summary(low):
    p = vm.analyze(low)
    assert len(p.band_db) == len(vm.BANDS_HZ)
    assert p.tilt_db_oct is not None and p.tilt_db_oct < 0         # voices lose level with frequency
    assert 300 < p.centroid_hz < 3000
    s = p.summary()
    assert "band_db" not in s and s["f0_median"] == pytest.approx(p.f0_median, abs=0.1)


def test_silence_gives_no_values_and_warnings():
    p = vm.analyze(np.zeros(SR * 3, dtype=np.float32))
    assert p.f0_median is None and p.formants is None
    text = " ".join(p.warnings)
    assert "3.0 s long" in text and "voiced speech" in text


def test_quiet_and_clipped_are_warned(low):
    assert any("quiet" in w for w in vm.analyze(low * 0.003).warnings)
    assert any("clipped" in w for w in vm.analyze(np.clip(low * 20, -1, 1)).warnings)


def test_match_settings(result):
    s = result.settings
    assert s.pitch_st == pytest.approx(12 * np.log2(210 / 120), abs=0.3)
    assert 1.5 < s.formant_st < 6                                  # bigger formants -> positive shift
    assert all(-vm.EQ_CUT_DB <= g <= vm.EQ_BOOST_DB for _, g in s.eq)
    assert [h for h, _ in s.eq] == list(vm.BANDS_HZ)


def test_after_matching_is_measured_and_closer(result):
    t, o, a = result.target, result.own, result.after
    assert a.f0_median == pytest.approx(t.f0_median, rel=0.03)
    assert abs(a.centroid_hz - t.centroid_hz) < abs(o.centroid_hz - t.centroid_hz)
    assert abs(a.tilt_db_oct - t.tilt_db_oct) < abs(o.tilt_db_oct - t.tilt_db_oct)
    assert result.after_audio is not None and result.after_audio.size == round(o.duration_s * SR)


def test_limits_are_applied():
    t = vm.analyze(synth(400, (900, 1600, 3000), seconds=4))
    o = vm.analyze(synth(90, (550, 1000, 2300), seconds=4, seed=3))
    pitch, formant, notes = vm.match_pitch_formant(t, o)
    assert pitch == vm.PITCH_LIMIT and notes


def test_render_keeps_length_and_flat_is_identity(low):
    y = vm.render(low[:SR], 0, 0)
    assert y.shape == (SR,)
    assert np.max(np.abs(y[4096:-4096] - low[4096:SR - 4096])) < 0.02


def test_load_wav_resamples_to_48k(tmp_path, low):
    p = tmp_path / "ref.wav"
    x = synth(150, seconds=5, sr=44100)                           # 2 whole cycles of the synthetic intonation
    sf.write(p, np.stack([x, x], 1), 44100)                       # stereo -> mono
    audio, dur = vm.load_audio(p)
    assert dur == pytest.approx(5.0, abs=0.01)
    assert audio.dtype == np.float32 and audio.size == pytest.approx(5 * SR, abs=2)
    assert vm.analyze(audio).f0_median == pytest.approx(150, rel=0.02)


def test_load_webm(tmp_path):
    av = pytest.importorskip("av")
    p = tmp_path / "ref.webm"
    x = synth(180, seconds=5)
    with av.open(str(p), "w", format="webm") as box:
        st = box.add_stream("libopus", rate=SR)
        st.layout = "mono"
        frame_len = 960
        pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        for i in range(0, pcm.size - frame_len + 1, frame_len):
            fr = av.AudioFrame.from_ndarray(pcm[None, i:i + frame_len], format="s16", layout="mono")
            fr.sample_rate = SR
            for pkt in st.encode(fr):
                box.mux(pkt)
        for pkt in st.encode(None):
            box.mux(pkt)
    audio, dur = vm.load_audio(p)
    assert dur == pytest.approx(5.0, abs=0.1)
    assert vm.analyze(audio).f0_median == pytest.approx(180, rel=0.03)


def test_long_files_are_truncated(tmp_path):
    p = tmp_path / "long.wav"
    sf.write(p, np.zeros(int(16000 * (vm.MAX_SECONDS + 5)), dtype=np.float32), 16000)
    audio, dur = vm.load_audio(p)
    assert audio.size == vm.MAX_SECONDS * SR and dur == pytest.approx(vm.MAX_SECONDS + 5)
    assert any("first" in w for w in vm.analyze(audio, SR, dur).warnings)


@pytest.mark.parametrize("name,data", [("notes.txt", b"hello"), ("broken.wav", b"RIFF0000WAVEjunk")])
def test_bad_files_raise_a_clear_error(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    with pytest.raises(vm.AudioLoadError):
        vm.load_audio(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(vm.AudioLoadError):
        vm.load_audio(tmp_path / "nope.wav")
