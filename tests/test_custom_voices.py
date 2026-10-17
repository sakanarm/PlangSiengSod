import json

import numpy as np
import pytest

from rtvc import paths
from rtvc.audio import custom_voices as cvs
from rtvc.audio.processors import EqProcessor, VoiceShiftProcessor
from rtvc.audio.voices import get_voice, make_voice, voice_exists
from rtvc.config import AppConfig

SR = 48000
EQ = [(250.0, -3.0), (1000.0, 0.0), (4000.0, 4.5)]


@pytest.fixture
def folder(tmp_path, monkeypatch):
    d = tmp_path / "custom_voices"
    monkeypatch.setattr(paths, "CUSTOM_VOICES_DIR", d)      # the default folder is used everywhere
    return d


def test_create_list_load_delete(folder):
    assert cvs.list_voices() == []
    cv = cvs.create("  Narrator  ", 4.04, -1.26, EQ, {"reference": {"f0_median": 180.0}})
    assert cv.name == "Narrator" and cv.pitch_st == 4.0 and cv.formant_st == -1.3
    assert cv.key.startswith(cvs.PREFIX) and (folder / f"{cv.id}.json").is_file()
    assert cvs.load(cv.id) == cv
    assert [v.id for v in cvs.list_voices()] == [cv.id]
    assert cvs.exists(cv.key) and not cvs.exists("custom:nope") and not cvs.exists("male")
    cvs.delete(cv.id)
    assert cvs.list_voices() == [] and not cvs.exists(cv.key)


def test_only_numbers_are_stored(folder):
    cv = cvs.create("Test", 1, 1, EQ, {"own": {"f0_median": 120.0}})
    d = json.loads((folder / f"{cv.id}.json").read_text(encoding="utf-8"))
    assert set(d) == {"id", "name", "pitch_st", "formant_st", "eq", "eq_amount", "created", "measured", "version"}


def test_never_overwrites(folder, monkeypatch):
    a = cvs.create("Same", 1, 1, EQ)
    b = cvs.create("Same", 2, 2, EQ)
    assert a.id != b.id and len(cvs.list_voices()) == 2
    monkeypatch.setattr(cvs.secrets, "token_hex", lambda n: a.id.rsplit("-", 1)[1])
    monkeypatch.setattr(cvs, "datetime", type("D", (), {"now": staticmethod(
        lambda: __import__("datetime").datetime.strptime(a.id[:15], "%Y%m%d-%H%M%S"))}))
    with pytest.raises(FileExistsError):                  # same id -> refuse, keep the old file
        cvs.create("Other", 3, 3, EQ)
    assert cvs.load(a.id).name == "Same"


@pytest.mark.parametrize("name", ["", "   ", "()[]", "\x00\x07"])
def test_empty_name_rejected(folder, name):
    with pytest.raises(ValueError):
        cvs.create(name, 0, 0, EQ)


def test_names_are_cleaned():
    assert cvs.clean_name("My (best)\nvoice [v2]") == "My bestvoice v2"
    assert len(cvs.clean_name("x" * 100)) == cvs.NAME_MAX


@pytest.mark.parametrize("pitch,formant,eq", [(20, 0, EQ), (0, -9, EQ), (0, 0, [(0, 3)]), (0, 0, [(100, 40)])])
def test_out_of_range_rejected(folder, pitch, formant, eq):
    with pytest.raises(ValueError):
        cvs.create("Bad", pitch, formant, eq)
    assert not any(folder.glob("*.json")) if folder.exists() else True


@pytest.mark.parametrize("vid", ["../evil", "a/b", "UPPER", "", "x" * 49])
def test_invalid_ids_rejected(folder, vid):
    with pytest.raises(ValueError):
        cvs.load(vid)
    assert not cvs.exists(cvs.PREFIX + vid)


def test_broken_files_are_skipped_not_deleted(folder):
    good = cvs.create("Good", 1, 1, EQ)
    (folder / "broken.json").write_text("{not json", encoding="utf-8")
    (folder / "wrongid.json").write_text(json.dumps({**json.loads(
        (folder / f"{good.id}.json").read_text(encoding="utf-8")), "id": "other"}), encoding="utf-8")
    assert [v.id for v in cvs.list_voices()] == [good.id]
    assert (folder / "broken.json").exists() and (folder / "wrongid.json").exists()


def test_custom_voice_in_engine_and_config(folder):
    cv = cvs.create("Narrator", 3.0, 1.5, EQ)
    v = get_voice(cv.key)
    assert v.group == "custom" and v.label == "Narrator (My voice)" and v.eq == tuple(EQ)
    assert voice_exists(cv.key)
    chain = make_voice(cv.key, SR)
    assert isinstance(chain.shift, VoiceShiftProcessor) and isinstance(chain.eq, EqProcessor)
    assert chain.shift.pitch_st == 3.0 and chain.shift.formant_st == 1.5 and chain.eq.amount == 1.0
    chain = make_voice(cv.key, SR, pitch_st=-2, formant_st=0, eq_amount=0.4)   # the sliders still work
    assert chain.shift.pitch_st == -2 and chain.eq.amount == 0.4
    y = chain.process(np.zeros(4800, np.float32))
    assert y.shape == (4800,) and np.all(np.isfinite(y))
    assert AppConfig(mode="voice", voice=cv.key).validate() == []


def test_flat_custom_voice_has_no_eq_stage(folder):
    cv = cvs.create("Flat", 2, 0, [])
    assert make_voice(cv.key, SR).eq is None


def test_deleted_or_unknown_custom_voice(folder):
    cv = cvs.create("Gone", 1, 1, EQ)
    cvs.delete(cv.id)
    assert not voice_exists(cv.key)
    with pytest.raises(ValueError):
        get_voice(cv.key)
    assert any("voice" in e for e in AppConfig(mode="voice", voice=cv.key).validate())
