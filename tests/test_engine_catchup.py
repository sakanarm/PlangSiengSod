"""Long calls: after a hiccup (the worker stalls, Windows pauses the process) the output FIFO is left
too full and, since mic and output run at the same rate, would keep that extra delay forever.
These tests drive the real worker step and output callback on a simulated clock (no audio device)."""

import numpy as np
import pytest

from rtvc.audio.engine import AudioEngine, _Catchup, _Fifo
from rtvc.audio.processors import PassthroughProcessor

SR, BLOCK = 48000, 2400
F0 = 300.0                                   # 300 Hz: exactly 15 periods per block


def make_engine(prebuffer_blocks=1):
    return AudioEngine(PassthroughProcessor(), hostapi="test", input_device=0, output_device=1,
                       samplerate=SR, block_time=0.05, prebuffer_blocks=prebuffer_blocks)


def talk(i, speech=20, pause=10):
    """Input block i: a 300 Hz 'voice' for `speech` blocks, then silence for `pause` blocks."""
    if pause and i % (speech + pause) >= speech:
        return np.zeros(BLOCK, np.float32)
    n = i * BLOCK + np.arange(BLOCK)
    return (0.3 * np.sin(2 * np.pi * F0 * n / SR)).astype(np.float32)


class Sim:
    """Output callback every `frames` samples of device time; mic blocks arrive every BLOCK samples
    (scaled by `ppm` clock difference); the worker handles queued blocks unless stalled."""

    def __init__(self, eng, frames=480, phase=0, ppm=0.0):
        self.eng, self.cb, self.frames = eng, eng._make_out_cb(0), frames
        self.t, self.i = 0, 0
        self.next_in = float(phase)
        self.period = BLOCK * (1 - ppm * 1e-6)
        self.played, self.levels = [], []

    def run(self, seconds, block_fn=talk, stall=None):
        end = self.t + int(seconds * SR)
        while self.t < end:
            while self.next_in <= self.t:
                self.eng._in_cb(block_fn(self.i)[:, None], BLOCK, None, None)
                self.i += 1
                self.next_in += self.period
            if not (stall and stall[0] * SR <= self.t < stall[1] * SR):
                while not self.eng._in_q.empty():
                    assert self.eng._process_block(self.eng._in_q.get_nowait())
            out = np.zeros((self.frames, 1), np.float32)
            self.cb(out, self.frames, None, None)
            self.played.append(out[:, 0].copy())
            self.levels.append(len(self.eng._fifos[0]))
            self.t += self.frames

    def lows(self, t0, t1):
        """Lowest buffer level (samples) after the output pulls between t0 and t1 seconds."""
        a, b = int(t0 * SR / self.frames), int(t1 * SR / self.frames)
        return min(self.levels[a:b]), max(self.levels[a:b])

    def audio(self, t0=0.0):
        return np.concatenate(self.played)[int(t0 * SR):]


def sine_breaks(y):
    """Samples where a 300 Hz sine does not continue smoothly (cuts / jumps inside the voice)."""
    c = 2 * np.cos(2 * np.pi * F0 / SR)
    y = y.astype(np.float64)
    voiced = (y[:-2] != 0) & (y[1:-1] != 0) & (y[2:] != 0)
    return int(np.sum(voiced & (np.abs(y[2:] - (c * y[1:-1] - y[:-2])) > 1e-3)))


# ------------------------------------------------------------------ building blocks
def test_fifo_reports_its_lowest_level_after_pulls():
    f = _Fifo(10000)
    assert f.take_low() is None
    f.push(np.ones(1000, np.float32))
    f.pull(300)
    f.push(np.ones(100, np.float32))
    f.pull(600)
    assert f.take_low() == 200 and f.take_low() is None
    f.push(np.ones(50, np.float32))
    f.clear()
    assert f.take_low() is None


def test_cut_crossfades_from_where_the_last_block_ended():
    cu = _Catchup(SR, BLOCK, BLOCK)
    cu.excess = 1000
    y = np.linspace(-1, 1, BLOCK).astype(np.float32)
    out, cut, _ = cu.step(y, _Fifo(1), quiet=True)
    assert cut == 1000 and out.size == BLOCK - 1000 and cu.excess == 0
    assert out[0] == y[0] and out[-1] == y[-1]                       # continuous at both ends
    assert np.all(np.abs(np.diff(out)) < 0.02)                       # no jump inside the fade


class _FullFifo:
    """Always reports a buffer 5000 samples above the target."""

    def take_low(self):
        return BLOCK + 5000


def test_no_cut_during_speech_until_the_wait_runs_out():
    cu = _Catchup(SR, BLOCK, BLOCK)
    y = np.ones(BLOCK, np.float32)
    started = [cu.step(y, _FullFifo(), quiet=False) for _ in range(cu.window)]
    assert [s[2] for s in started].count(True) == 1 and cu.excess == 5000
    for _ in range(cu.force_after - 1):
        assert cu.step(y, _FullFifo(), quiet=False)[1] == 0
    out, cut, _ = cu.step(y, _FullFifo(), quiet=False)
    assert cut == SR // 100 and out.size == BLOCK - SR // 100          # then only 10 ms per block


# ------------------------------------------------------------------ whole engine, simulated clock
@pytest.mark.parametrize("frames,phase", [(480, 0), (480, 1300), (441, 777), (1024, 2399), (128, 50)])
def test_normal_running_is_never_trimmed(frames, phase):
    eng = make_engine()
    sim = Sim(eng, frames, phase)
    sim.run(60)
    s = eng.stats
    assert s.catchups == 0 and s.catchup_ms == 0 and s.underruns == 0 and s.dropped_blocks == 0
    assert sine_breaks(sim.audio(0.5)) == 0
    assert sim.lows(1, 60)[0] <= eng.prebuffer + eng.block // 2


def test_a_hiccup_does_not_leave_extra_delay_for_the_rest_of_the_call():
    eng = make_engine()
    sim = Sim(eng)
    sim.run(10)
    normal_low, normal_high = sim.lows(2, 10)
    normal_latency = eng.stats.est_latency_ms
    sim.run(0.5, stall=(10.0, 10.5))                                 # worker frozen for 0.5 s
    sim.run(2.0)
    stuck_low, _ = sim.lows(11, 12)
    assert stuck_low > eng.prebuffer + eng.block                     # what used to stay for good
    assert eng.stats.est_latency_ms > normal_latency + 80            # the UI shows the real delay
    sim.run(8.0)
    s = eng.stats
    low, high = sim.lows(15, 20.5)
    assert low <= normal_low + eng.block // 2 and high <= normal_high + eng.block // 2
    assert s.catchups == 1 and 50 <= s.catchup_ms < 260
    assert eng.stats.est_latency_ms < normal_latency + 30
    under = s.underruns
    sim.run(30)
    assert eng.stats.underruns == under and eng.stats.catchups == 1  # stays settled, no new trims
    assert sine_breaks(sim.audio(11.0)) == 0                         # every cut fell in a pause


def test_catches_up_even_when_nobody_pauses():
    eng = make_engine()
    sim = Sim(eng)
    nonstop = lambda i: talk(i, pause=0)                             # noqa: E731
    sim.run(10, nonstop)
    normal_low, _ = sim.lows(2, 10)
    sim.run(0.5, nonstop, stall=(10.0, 10.5))
    sim.run(30, nonstop)
    low, _ = sim.lows(30, 40.5)
    assert low <= normal_low + eng.block // 2
    assert eng.stats.catchups == 1


@pytest.mark.parametrize("ppm", [2000.0, -2000.0])
def test_clock_drift_does_not_build_up_delay(ppm):
    """Mic and VB-CABLE clocks differ slightly; over a long call that must not add up."""
    eng = make_engine()
    sim = Sim(eng, ppm=ppm)
    sim.run(180)                                                     # 2000 ppm: 360 ms after 3 min
    s = eng.stats
    _, high = sim.lows(170, 180)
    assert high <= eng.prebuffer + 2 * eng.block + eng.block // 2
    assert s.dropped_blocks == 0                                     # never hits the hard limit
    if ppm > 0:
        assert s.catchup_ms > 250 and sine_breaks(sim.audio(1.0)) == 0
    else:                                                            # mic slower: rare refills instead
        assert s.catchups == 0 and s.underruns <= 10
