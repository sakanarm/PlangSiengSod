"""Checks on the real audio system. Skipped when VB-CABLE is not installed (or not on Windows)."""

import sys
import threading

import pytest

from rtvc.audio import devices as dv
from rtvc.audio.engine import play_test_tone

API = "Windows WASAPI"


def _cable_input():
    try:
        ds = dv.list_devices()
    except Exception:
        return None
    return next((d for d in dv.outputs(ds, API) if d.is_cable_playback and "16ch" not in d.name), None)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows audio only")
def test_beep_works_from_a_worker_thread():
    """The Beep button plays from a worker thread; WASAPI needs COM initialised there."""
    d = _cable_input()
    if d is None:
        pytest.skip("VB-CABLE not installed")
    err = []

    def run():
        try:
            play_test_tone(d.index, API, 48000, seconds=0.2)
        except Exception as e:  # noqa: BLE001 - reported below
            err.append(e)

    t = threading.Thread(target=run)
    t.start()
    t.join(10)
    assert not t.is_alive() and not err, err
