import numpy as np
import soundfile as sf

from rtvc.audio.recorder import WavRecorder


def test_recorder_writes_all_blocks_in_order(tmp_path):
    sr = 48000
    blocks = [np.full(2400, (i + 1) / 100, dtype=np.float32) for i in range(20)]
    rec = WavRecorder(tmp_path / "sub" / "out.wav", sr)
    for b in blocks:
        rec.write(b)
    secs = rec.close()

    assert rec.error == "" and rec.dropped == 0
    assert abs(secs - 1.0) < 1e-9
    data, file_sr = sf.read(tmp_path / "sub" / "out.wav", dtype="float32")
    assert file_sr == sr and data.shape == (48000,)
    np.testing.assert_allclose(data, np.concatenate(blocks), atol=1 / 32768 * 1.5)  # PCM_16 rounding


def test_recorder_counts_drops_instead_of_blocking(tmp_path):
    import queue

    rec = WavRecorder(tmp_path / "out.wav", 16000)
    full = queue.Queue(maxsize=1)  # simulate a disk writer that has fallen behind
    full.put(np.zeros(10, np.float32))
    real_q, rec._q = rec._q, full
    rec.write(np.zeros(10, np.float32))  # must return immediately
    assert rec.dropped == 1
    rec._q = real_q
    rec.close()
