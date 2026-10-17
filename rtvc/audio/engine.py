"""Real-time audio engine: mic -> BlockProcessor -> output (VB-CABLE) [+ optional monitor].

Threads:
  * PortAudio input callback  - copies fixed-size blocks into ``_in_q`` (never blocks).
  * worker thread             - runs the processor, applies gain, writes the output FIFOs.
  * PortAudio output callback - drains the FIFO; plays silence and counts an underrun if empty.

A slow processor therefore causes counted drops/underruns, never a stalled callback.
After a hiccup (e.g. Windows pausing the process for a moment) a FIFO can be left holding more
audio than it needs; input and output then run at the same rate, so it would never drain and the
extra delay would stay for the rest of the call. ``_Catchup`` removes that surplus, in pauses
between words where possible (see its docstring).
All numbers in ``EngineStats`` are measured, none are estimated.
"""

from __future__ import annotations

import collections
import functools
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .processors import BlockProcessor

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_ERRORS = 20
CATCHUP_WINDOW_S = 1.0        # how long the buffer must stay too full before it is trimmed
CATCHUP_FORCE_AFTER_S = 8.0   # trim inside speech too if no pause came for this long
QUIET_DB = -50.0              # input below this (or near the room's noise floor) counts as a pause


@dataclass
class EngineStats:
    running: bool = False
    blocks_in: int = 0
    blocks_out: int = 0
    dropped_blocks: int = 0          # input blocks discarded because the worker was behind
    underruns: int = 0               # output callbacks that found the FIFO short
    catchups: int = 0                # times extra buffered delay was found (main output) and removed
    catchup_ms: float = 0.0          # total audio removed for that (ms)
    buffer_ms: float = 0.0           # main output buffer level after each push, smoothed (ms)
    processing_errors: int = 0
    stream_warnings: int = 0         # PortAudio status flags (xruns reported by the driver)
    proc_ms_last: float = 0.0
    proc_ms_avg: float = 0.0
    proc_ms_max: float = 0.0
    rtf: float = 0.0                 # processing time / block duration (must stay < 1)
    input_db: float = -120.0
    output_db: float = -120.0
    est_latency_ms: float = 0.0      # block + measured buffer wait + processor algorithmic + processing
    last_error: str = ""
    extra: dict = field(default_factory=dict)


def _db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    peak = float(np.max(np.abs(x)))
    return 20 * np.log10(peak) if peak > 1e-6 else -120.0


class _Fifo:
    """Thread-safe sample FIFO for one output stream."""

    def __init__(self, max_samples: int):
        self._buf = collections.deque()
        self._n = 0
        self._max = max_samples
        self._low = None               # lowest level seen right after a pull, since take_low()
        self._lock = threading.Lock()

    def push(self, x: np.ndarray) -> int:
        with self._lock:
            self._buf.append(x)
            self._n += x.shape[0]
            dropped = 0
            while self._n > self._max and len(self._buf) > 1:   # bound latency growth
                old = self._buf.popleft()
                self._n -= old.shape[0]
                dropped += 1
            return dropped

    def pull(self, n: int) -> tuple[np.ndarray, bool]:
        out = np.zeros(n, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < n and self._buf:
                head = self._buf[0]
                take = min(n - filled, head.shape[0])
                out[filled:filled + take] = head[:take]
                filled += take
                if take == head.shape[0]:
                    self._buf.popleft()
                else:
                    self._buf[0] = head[take:]
                self._n -= take
            self._low = self._n if self._low is None else min(self._low, self._n)
        return out, filled < n

    def take_low(self) -> int | None:
        """Lowest level after a pull since the previous call (None: nothing was pulled)."""
        with self._lock:
            low, self._low = self._low, None
            return low

    def __len__(self):
        return self._n

    def clear(self):
        with self._lock:
            self._buf.clear()
            self._n = 0
            self._low = None


class _Catchup:
    """Removes extra delay that a hiccup left in one output FIFO.

    Every CATCHUP_WINDOW_S the lowest FIFO level seen by the output callback is checked. Normally
    it is at most ``prebuffer`` (the safety margin the stream was primed with). If it stayed more
    than half a block above that for the whole window, the surplus is pure delay. It is cut out of
    the next blocks, preferably blocks where the microphone is quiet (pauses), with a 2 ms
    crossfade so nothing clicks. If no pause comes within CATCHUP_FORCE_AFTER_S, at most 10 ms per
    block is cut during speech (10 ms crossfade). After a window with cuts the level is measured
    again before cutting more, so the safety margin is never cut into.
    """

    def __init__(self, sr: int, block: int, prebuffer: int):
        self.target = prebuffer
        self.margin = block // 2
        self.window = max(1, int(round(CATCHUP_WINDOW_S * sr / block)))
        self.force_after = int(round(CATCHUP_FORCE_AFTER_S * sr / block))
        self.fade_quiet = max(8, sr // 500)
        self.fade_loud = max(8, sr // 100)
        self.loud_cap = max(8, sr // 100)
        self.reset()

    def reset(self):
        self.excess = 0        # samples still to remove
        self.found = 0         # surplus measured when the current catch-up started
        self.waiting = 0       # blocks that had surplus but no pause
        self.active = False    # a catch-up is in progress (until a clean window shows no surplus)
        self._n = 0
        self._dirty = False

    def step(self, y: np.ndarray, fifo: _Fifo, quiet: bool) -> tuple[np.ndarray, int, bool]:
        """Returns (block to push, samples removed, True if a new catch-up started now)."""
        started = False
        self._n += 1
        if self._n >= self.window:
            self._n = 0
            low = fifo.take_low()
            if self._dirty or low is None:
                self.excess = 0                      # measure again after cutting / while priming
            else:
                over = low - self.target
                self.excess = over if over > self.margin else 0
                if self.excess and not self.active:
                    self.active, self.found, started = True, self.excess, True
                elif not self.excess:
                    self.active, self.waiting = False, 0
            self._dirty = False
        if self.excess <= 0:
            return y, 0, started
        if quiet:
            fade, cap = self.fade_quiet, y.shape[0] - self.fade_quiet
        elif self.waiting >= self.force_after:
            fade, cap = self.fade_loud, min(self.loud_cap, y.shape[0] - self.fade_loud)
        else:
            self.waiting += 1
            return y, 0, started
        d = min(self.excess, cap)
        if d <= 0:
            return y, 0, started
        w = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        out = y[d:].copy()
        out[:fade] = y[:fade] * (1 - w) + y[d:d + fade] * w     # starts where the previous block ended
        self.excess -= d
        self._dirty = True
        return out, d, started


class AudioEngine:
    def __init__(self, processor: BlockProcessor, *, hostapi: str, input_device: int, output_device: int,
                 monitor_device: int | None = None, samplerate: int = 48000, block_time: float = 0.05,
                 prebuffer_blocks: int = 1, input_gate_db: float = -60.0, output_gain_db: float = 0.0,
                 on_fatal=None):
        self.processor = processor
        self.hostapi = hostapi
        self.input_device = input_device
        self.output_device = output_device
        self.monitor_device = monitor_device
        self.sr = int(samplerate)
        self.block = max(64, int(round(block_time * self.sr)))
        self.prebuffer = int(prebuffer_blocks) * self.block
        self.gate = 10 ** (input_gate_db / 20) if input_gate_db > -60 else 0.0
        self.gain = 10 ** (output_gain_db / 20)
        self.on_fatal = on_fatal
        self.stats = EngineStats()
        self._in_q: queue.Queue = queue.Queue(maxsize=4)
        self._fifos = [_Fifo(self.prebuffer + 4 * self.block)]
        if monitor_device is not None:
            self._fifos.append(_Fifo(self.prebuffer + 4 * self.block))
        self._primed = [False] * len(self._fifos)
        self._catchup = [_Catchup(self.sr, self.block, self.prebuffer) for _ in self._fifos]
        self._floor = 1.0                      # input noise floor (rms), for pause detection
        self._n_blocks = 0
        self._streams = []
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._watchdog: threading.Thread | None = None
        self._consecutive_errors = 0
        self._lock = threading.Lock()
        self.recorder = None   # optional WavRecorder receiving the processed output

    # ---------------------------------------------------------------- callbacks
    def _in_cb(self, indata, frames, t, status):
        if status:
            self.stats.stream_warnings += 1
        x = indata[:, 0].astype(np.float32, copy=True)
        self.stats.blocks_in += 1
        try:
            self._in_q.put_nowait(x)
        except queue.Full:
            self.stats.dropped_blocks += 1

    def _make_out_cb(self, k: int):
        def cb(outdata, frames, t, status):
            if status:
                self.stats.stream_warnings += 1
            fifo = self._fifos[k]
            if not self._primed[k]:
                if len(fifo) < self.prebuffer + frames:
                    outdata.fill(0)
                    return
                self._primed[k] = True
            y, short = fifo.pull(frames)
            if short:
                self.stats.underruns += 1
                self._primed[k] = False        # re-prime instead of crackling block by block
            outdata[:, :] = y[:, None]
        return cb

    # ---------------------------------------------------------------- worker
    def _work(self):
        while not self._stop.is_set():
            try:
                x = self._in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not self._process_block(x):
                return

    def _process_block(self, x: np.ndarray) -> bool:
        """One input block -> processor -> output FIFOs. False after a fatal error."""
        blk_s = self.block / self.sr
        self.stats.input_db = _db(x)
        rms = float(np.sqrt(np.mean(x * x)))
        if self.gate and rms < self.gate:
            x = np.zeros_like(x)
            rms = 0.0
        self._floor = max(1e-5, min(rms, self._floor * 1.002))   # follows the room noise, rises slowly
        quiet = rms <= max(10 ** (QUIET_DB / 20), 2.0 * self._floor)
        t0 = time.perf_counter()
        try:
            y = self.processor.process(x)
            if y.shape[0] != x.shape[0]:
                raise RuntimeError(f"processor returned {y.shape[0]} samples for {x.shape[0]}")
            if not np.all(np.isfinite(y)):
                raise RuntimeError("processor produced NaN/Inf")
            self._consecutive_errors = 0
        except Exception as e:  # keep the stream alive; output silence for this block
            y = np.zeros_like(x)
            self.stats.processing_errors += 1
            self._consecutive_errors += 1
            self.stats.last_error = f"{type(e).__name__}: {e}"
            logger.exception("Processing error (block %d)", self.stats.blocks_out)
            if self._consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                self._fatal(f"Processor failed {MAX_CONSECUTIVE_ERRORS} blocks in a row: {self.stats.last_error}")
                return False
        dt = (time.perf_counter() - t0) * 1000
        self._n_blocks += 1
        n = self._n_blocks
        s = self.stats
        s.proc_ms_last = dt
        s.proc_ms_avg = dt if n == 1 else 0.95 * s.proc_ms_avg + 0.05 * dt
        s.proc_ms_max = max(s.proc_ms_max, dt)
        s.rtf = s.proc_ms_avg / (blk_s * 1000)
        y = np.clip(y * self.gain, -1.0, 1.0).astype(np.float32)
        s.output_db = _db(y)
        for k, (f, cu) in enumerate(zip(self._fifos, self._catchup)):
            out, cut, started = cu.step(y, f, quiet)
            if started:
                logger.info("%s output held %.0f ms of extra delay (after a hiccup); removing it",
                            "Main" if k == 0 else "Monitor", 1000 * cu.found / self.sr)
                if k == 0:
                    s.catchups += 1
            if cut and k == 0:
                s.catchup_ms += 1000 * cut / self.sr
            dropped = f.push(out)
            if dropped:
                s.dropped_blocks += dropped
        lvl_ms = 1000 * len(self._fifos[0]) / self.sr
        s.buffer_ms = lvl_ms if n == 1 else 0.95 * s.buffer_ms + 0.05 * lvl_ms
        rec = self.recorder
        if rec is not None:
            rec.write(y)
        s.blocks_out += 1
        # a sample waits on average (level after push - half a block) in the FIFO
        s.est_latency_ms = 1000 * self.block / self.sr + max(0.0, s.buffer_ms - 500 * self.block / self.sr) \
            + 1000 * self.processor.algorithmic_latency_s + s.proc_ms_avg
        if n % 200 == 0 and (s.dropped_blocks or s.underruns):
            logger.info("stats: dropped=%d underruns=%d proc_avg=%.1fms rtf=%.2f",
                        s.dropped_blocks, s.underruns, s.proc_ms_avg, s.rtf)
        return True

    def _watch(self):
        """Detect device removal: PortAudio stops a stream whose device disappears."""
        while not self._stop.wait(0.5):
            for st in self._streams:
                if not st.active and not self._stop.is_set():
                    self._fatal("Audio device stopped or was disconnected. Check the device and press Start again.")
                    return

    def _fatal(self, msg: str):
        logger.error(msg)
        self.stats.last_error = msg
        threading.Thread(target=self.stop, daemon=True).start()
        if self.on_fatal:
            try:
                self.on_fatal(msg)
            except Exception:
                logger.exception("on_fatal handler failed")

    # ---------------------------------------------------------------- control
    def start(self):
        import sounddevice as sd

        extra = sd.WasapiSettings(auto_convert=True) if "WASAPI" in self.hostapi else None
        self.processor.reset()
        for cu in self._catchup:
            cu.reset()
        self._n_blocks = 0
        self._stop.clear()
        try:
            ins = sd.InputStream(device=self.input_device, channels=1, samplerate=self.sr, dtype="float32",
                                 blocksize=self.block, callback=self._in_cb, extra_settings=extra,
                                 latency="low")
            outs = [sd.OutputStream(device=self.output_device, channels=1, samplerate=self.sr,
                                    dtype="float32", callback=self._make_out_cb(0), extra_settings=extra,
                                    latency="low")]
            if self.monitor_device is not None:
                outs.append(sd.OutputStream(device=self.monitor_device, channels=1, samplerate=self.sr,
                                            dtype="float32", callback=self._make_out_cb(1),
                                            extra_settings=extra, latency="low"))
        except Exception as e:
            raise RuntimeError(f"Could not open audio device: {e}") from e
        self._streams = [ins] + outs
        self._worker = threading.Thread(target=self._work, name="rtvc-worker", daemon=True)
        self._worker.start()
        try:
            for st in self._streams:
                st.start()
        except Exception as e:
            self.stop()
            raise RuntimeError(f"Could not start audio stream: {e}") from e
        self.stats.running = True
        self.stats.extra["device_latency_ms"] = {
            "input": round(1000 * ins.latency, 1), "output": round(1000 * outs[0].latency, 1)}
        self._watchdog = threading.Thread(target=self._watch, name="rtvc-watchdog", daemon=True)
        self._watchdog.start()
        logger.info("Engine started: sr=%d block=%d (%.0f ms) prebuffer=%d processor=%s",
                    self.sr, self.block, 1000 * self.block / self.sr, self.prebuffer, self.processor.name)

    def stop(self):
        with self._lock:
            self._stop.set()
            for st in self._streams:
                try:
                    st.abort()
                    st.close()
                except Exception:
                    pass
            self._streams = []
            if self._worker and self._worker is not threading.current_thread():
                self._worker.join(timeout=2)
            if self.recorder is not None:
                rec, self.recorder = self.recorder, None
                rec.close()
            for f in self._fifos:
                f.clear()
            if self.stats.running:
                logger.info("Engine stopped: in=%d out=%d dropped=%d underruns=%d errors=%d catch-ups=%d (%.0f ms)",
                            self.stats.blocks_in, self.stats.blocks_out, self.stats.dropped_blocks,
                            self.stats.underruns, self.stats.processing_errors, self.stats.catchups,
                            self.stats.catchup_ms)
            self.stats.running = False


def _com_thread(fn):
    """Initialise COM for the calling thread while ``fn`` opens/uses/closes its streams.

    A WASAPI stream opened from a worker thread without COM failed to start ('Unanticipated host
    error', measured on Windows 11 with PortAudio 19.7); the UI runs the beep in a worker thread.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if sys.platform != "win32":
            return fn(*args, **kwargs)
        import ctypes

        ole32 = ctypes.windll.ole32
        hr = ole32.CoInitializeEx(None, 0)     # COINIT_MULTITHREADED; S_OK/S_FALSE must be balanced
        try:
            return fn(*args, **kwargs)
        finally:
            if hr in (0, 1):
                ole32.CoUninitialize()
    return wrapper


@_com_thread
def play_test_tone(device: int, hostapi: str, samplerate: int = 48000, seconds: float = 1.0,
                   freq: float = 440.0) -> None:
    """Blocking 440 Hz beep on one output device (routing test)."""
    import sounddevice as sd

    t = np.arange(int(seconds * samplerate)) / samplerate
    tone = (0.2 * np.sin(2 * np.pi * freq * t) * np.minimum(1, np.minimum(t, seconds - t) * 20)).astype(np.float32)
    extra = sd.WasapiSettings(auto_convert=True) if "WASAPI" in hostapi else None
    # Own callback stream: sd.play/sd.rec share one global stream and stop each other, and a
    # blocking write() through WASAPI auto_convert was measured to lose samples.
    pos = [0]
    done = threading.Event()

    def cb(out, frames, t_info, status):
        n = min(frames, tone.shape[0] - pos[0])
        out[:n, 0] = tone[pos[0]:pos[0] + n]
        out[n:] = 0
        pos[0] += n
        if pos[0] >= tone.shape[0]:
            done.set()

    with sd.OutputStream(device=device, channels=1, samplerate=samplerate, dtype="float32",
                         callback=cb, extra_settings=extra) as st:
        done.wait(seconds + 2)
        time.sleep(st.latency + 0.05)


@_com_thread
def play_and_record(play_device: int, record_device: int, hostapi: str, signal: np.ndarray,
                    samplerate: int = 48000, pre_s: float = 0.3, tail_s: float = 0.7) -> np.ndarray:
    """Play ``signal`` on one device while recording another (e.g. CABLE Input -> CABLE Output).

    Both streams are opened from the calling thread: opening the VB-CABLE capture and render
    streams concurrently from two threads was measured to fail with a PortAudio host error.
    """
    import sounddevice as sd

    extra = sd.WasapiSettings(auto_convert=True) if "WASAPI" in hostapi else None
    sig = np.asarray(signal, dtype=np.float32)
    n_rec = sig.shape[0] + int((pre_s + tail_s + 0.5) * samplerate)
    rec = np.zeros(n_rec, dtype=np.float32)
    rpos, ppos = [0], [0]
    done = threading.Event()

    def icb(indata, frames, t_info, status):
        k = min(frames, n_rec - rpos[0])
        rec[rpos[0]:rpos[0] + k] = indata[:k, 0]
        rpos[0] += k

    def ocb(out, frames, t_info, status):
        k = min(frames, sig.shape[0] - ppos[0])
        out[:k, 0] = sig[ppos[0]:ppos[0] + k]
        out[k:] = 0
        ppos[0] += k
        if ppos[0] >= sig.shape[0]:
            done.set()

    with sd.InputStream(device=record_device, channels=1, samplerate=samplerate, dtype="float32",
                        callback=icb, extra_settings=extra):
        time.sleep(pre_s)
        with sd.OutputStream(device=play_device, channels=1, samplerate=samplerate, dtype="float32",
                             callback=ocb, extra_settings=extra):
            done.wait(sig.shape[0] / samplerate + 3)
            time.sleep(tail_s)
    return rec[:rpos[0]].copy()


@_com_thread
def record_test(device: int, hostapi: str, samplerate: int = 48000, seconds: float = 2.0) -> np.ndarray:
    import sounddevice as sd

    extra = sd.WasapiSettings(auto_convert=True) if "WASAPI" in hostapi else None
    # Callback stream: a blocking read() through WASAPI auto_convert was measured to lose samples.
    n = int(seconds * samplerate)
    buf = np.zeros(n, dtype=np.float32)
    pos = [0]
    done = threading.Event()

    def cb(indata, frames, t_info, status):
        k = min(frames, n - pos[0])
        buf[pos[0]:pos[0] + k] = indata[:k, 0]
        pos[0] += k
        if pos[0] >= n:
            done.set()

    with sd.InputStream(device=device, channels=1, samplerate=samplerate, dtype="float32",
                        callback=cb, extra_settings=extra):
        done.wait(seconds + 3)
    return buf
