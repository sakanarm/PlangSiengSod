from rtvc.audio.devices import Device, cable_installed, find_device, suggest_cable_output, validate_routing

API = "Windows WASAPI"
MIC = "Microphone Array (Intel Smart Sound)"
CABLE_IN = "CABLE Input (VB-Audio Virtual Cable)"
CABLE_OUT = "CABLE Output (VB-Audio Virtual Cable)"
SPK = "Speakers (Realtek(R) Audio)"

DEVS = [
    Device(0, MIC, API, 2, 0, 48000),
    Device(1, CABLE_OUT, API, 2, 0, 48000),
    Device(2, SPK, API, 0, 2, 48000),
    Device(3, CABLE_IN, API, 0, 2, 48000),
    Device(4, MIC, "MME", 2, 0, 44100),
    Device(5, "Stereo Mix (Realtek(R) Audio)", API, 2, 0, 48000),
]


def test_normal_routing_is_valid():
    errors, warnings = validate_routing(MIC, CABLE_IN, "", DEVS, API)
    assert errors == [] and warnings == []


def test_feedback_loop_cable_to_cable_blocked():
    errors, _ = validate_routing(CABLE_OUT, CABLE_IN, "", DEVS, API)
    assert any("Feedback" in e for e in errors)


def test_stereo_mix_input_blocked():
    errors, _ = validate_routing("Stereo Mix (Realtek(R) Audio)", CABLE_IN, "", DEVS, API)
    assert any("loopback" in e for e in errors)


def test_missing_selection():
    errors, _ = validate_routing("", "", "", DEVS, API)
    assert len(errors) == 2


def test_output_not_cable_warns():
    errors, warnings = validate_routing(MIC, SPK, "", DEVS, API)
    assert errors == [] and any("CABLE Input" in w for w in warnings)


def test_monitor_rules():
    assert validate_routing(MIC, CABLE_IN, CABLE_IN, DEVS, API)[0]
    errors, warnings = validate_routing(MIC, CABLE_IN, SPK, DEVS, API)
    assert errors == [] and any("headphones" in w for w in warnings)


def test_disconnected_device_reported():
    errors, _ = validate_routing("USB Headset Mic", CABLE_IN, "", DEVS, API)
    assert any("not connected" in e for e in errors)


def test_find_device_respects_hostapi_and_kind():
    assert find_device(DEVS, MIC, API, "input").index == 0
    assert find_device(DEVS, MIC, "MME", "input").index == 4
    assert find_device(DEVS, MIC, API, "output") is None
    assert find_device(DEVS, "", API, "input") is None


def test_find_device_truncated_name():
    # MME truncates names to 31 chars; a full saved name must still match.
    devs = [Device(9, CABLE_IN[:31], "MME", 0, 2, 44100)]
    assert find_device(devs, CABLE_IN, "MME", "output").index == 9


def test_cable_detection():
    assert cable_installed(DEVS)
    assert suggest_cable_output(DEVS, API).name == CABLE_IN
    assert not cable_installed([d for d in DEVS if "CABLE" not in d.name])
