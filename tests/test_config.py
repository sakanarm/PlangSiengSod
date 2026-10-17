import json

from rtvc.config import AppConfig


def test_defaults_are_valid():
    assert AppConfig().validate() == []


def test_out_of_range_and_enums():
    c = AppConfig(block_time=5.0, pitch_shift=99, backend="cuda", mode="clone", robot_pitch_hz=5)
    errs = " ".join(c.validate())
    for key in ("block_time", "pitch_shift", "backend", "mode", "robot_pitch_hz"):
        assert key in errs


def test_type_errors_are_reported_not_raised():
    c = AppConfig(block_time="fast")
    assert any("block_time must be a number" in e for e in c.validate())


def test_bool_is_not_a_number():
    assert any("ring_mix" in e for e in AppConfig(ring_mix=True).validate())


def test_crossfade_shorter_than_block():
    assert any("crossfade" in e for e in AppConfig(block_time=0.05, crossfade_time=0.08).validate())


def test_same_input_output_rejected():
    c = AppConfig(input_device="X", output_device="X")
    assert any("different" in e for e in c.validate())


def test_monitor_equal_output_rejected():
    c = AppConfig(output_device="CABLE Input", monitor_device="CABLE Input")
    assert any("Monitor" in e for e in c.validate())


def test_roundtrip(tmp_path):
    p = tmp_path / "cfg.json"
    c = AppConfig(input_device="Mic", output_device="CABLE Input (VB-Audio Virtual Cable)", robot_pitch_hz=95.0)
    c.save(p)
    assert AppConfig.load(p) == c


def test_unknown_keys_ignored_and_ints_coerced(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"pitch_shift": 3.0, "samplerate": 48000.0, "bogus": 1}), encoding="utf-8")
    c = AppConfig.load(p)
    assert c.pitch_shift == 3 and isinstance(c.pitch_shift, int)
    assert c.samplerate == 48000 and isinstance(c.samplerate, int)


def test_corrupt_file_gives_defaults(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text("{not json", encoding="utf-8")
    assert AppConfig.load(p) == AppConfig()
    p.write_text("[1, 2]", encoding="utf-8")
    assert AppConfig.load(p) == AppConfig()


def test_missing_file_gives_defaults(tmp_path):
    assert AppConfig.load(tmp_path / "nope.json") == AppConfig()
