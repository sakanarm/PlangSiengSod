"""Real-time audio engine: mic -> BlockProcessor -> output (VB-CABLE) [+ optional monitor].

Threads:
  * PortAudio input callback  - copies fixed-size blocks into ``_in_q`` (never blocks).
  * worker thread             - runs the processor, applies gain, writes the output FIFOs.
  * PortAudio output callback - drains the FIFO; plays silence and counts an underrun if empty.

A slow processor therefore causes counted drops/underruns, never a stalled callback.
All numbers in ``EngineStats`` are measured, none are estimated.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .processors import BlockProcessor

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_ERRORS = 20


@dataclass
class EngineStats:
    running: bool = False
    blocks_in: int = 0
    blocks_out: int = 0
    dropped_blocks: int = 0          # input blocks discarded because the worker was behind
    underruns: int = 0               # output callbacks that found the FIFO short
    processing_errors: int = 0
    stream_warnings: int = 0         # PortAudio status flags (xruns reported by the driver)
    proc_ms_last: float = 0.0
    proc_ms_avg: float = 0.0
    proc_ms_max: float = 0.0
    rtf: float = 0.0                 # processing time / block duration (must stay < 1)
    input_db: float = -120.0
    output_db: float = -120.0
    est_latency_ms: float = 0.0      # block + prebuffer + processor algorithmic + measured processing
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
        return out, filled < n

    def __len__(self):
        return self._n

    def clear(self):
        with self._lock:
            self._buf.clear()
            self._n = 0


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
        blk_s = self.block / self.sr
        n = 0
        while not self._stop.is_set():
            try:
                x = self._in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            self.stats.input_db = _db(x)
            if self.gate and float(np.sqrt(np.mean(x * x))) < self.gate:
                x = np.zeros_like(x)
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
                    return
            dt = (time.perf_counter() - t0) * 1000
            n += 1
            s = self.stats
            s.proc_ms_last = dt
            s.proc_ms_avg = dt if n == 1 else 0.95 * s.proc_ms_avg + 0.05 * dt
            s.proc_ms_max = max(s.proc_ms_max, dt)
            s.rtf = s.proc_ms_avg / (blk_s * 1000)
            y = np.clip(y * self.gain, -1.0, 1.0).astype(np.float32)
            s.output_db = _db(y)
            for f in self._fifos:
                dropped = f.push(y)
                if dropped:
                    s.dropped_blocks += dropped
            rec = self.recorder
            if rec is not None:
                rec.write(y)
            s.blocks_out += 1
            s.est_latency_ms = 1000 * (self.block + self.prebuffer) / self.sr \
                + 1000 * self.processor.algorithmic_latency_s + s.proc_ms_avg
            if n % 200 == 0 and (s.dropped_blocks or s.underruns):
                logger.info("stats: dropped=%d underruns=%d proc_avg=%.1fms rtf=%.2f",
                            s.dropped_blocks, s.underruns, s.proc_ms_avg, s.rtf)

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
                logger.info("Engine stopped: in=%d out=%d dropped=%d underruns=%d errors=%d",
                            self.stats.blocks_in, self.stats.blocks_out, self.stats.dropped_blocks,
                            self.stats.underruns, self.stats.processing_errors)
            self.stats.running = False


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
