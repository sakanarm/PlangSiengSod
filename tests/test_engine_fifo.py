import numpy as np

from rtvc.audio.engine import _db, _Fifo


def test_fifo_order_and_partial_pulls():
    f = _Fifo(1000)
    f.push(np.arange(0, 10, dtype=np.float32))
    f.push(np.arange(10, 20, dtype=np.float32))
    a, short = f.pull(7)
    b, _ = f.pull(7)
    assert not short and list(a) == list(range(7)) and list(b) == list(range(7, 14))
    c, short = f.pull(10)
    assert short and list(c[:6]) == list(range(14, 20)) and np.all(c[6:] == 0)
    assert len(f) == 0


def test_fifo_bounds_latency_by_dropping_oldest():
    f = _Fifo(25)
    dropped = sum(f.push(np.full(10, i, np.float32)) for i in range(5))
    assert dropped == 3 and len(f) == 20           # never above the 25-sample bound
    y, _ = f.pull(10)
    assert np.all(y == 3)                          # oldest chunks were the ones dropped


def test_db():
    assert _db(np.zeros(10, np.float32)) == -120.0
    assert abs(_db(np.full(10, 0.5, np.float32)) - (-6.02)) < 0.01
