"""User-made voices ("voice from sample"): small JSON files in ``paths.CUSTOM_VOICES_DIR``.

A custom voice is only numbers: pitch shift, formant shift, an EQ curve and the measured
values they came from. No audio is stored, so the reference recording never leaves the
moment of analysis. Files are never overwritten: every voice gets a new unique id.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .. import paths

logger = logging.getLogger(__name__)

PREFIX = "custom:"
NAME_MAX = 40
_ID_RE = re.compile(r"^[a-z0-9_-]{1,48}$")


@dataclass
class CustomVoice:
    id: str
    name: str
    pitch_st: float
    formant_st: float
    eq: list[list[float]]                       # [[band Hz, gain dB], ...]
    eq_amount: float = 1.0
    created: str = ""
    measured: dict = field(default_factory=dict)   # target / own / after summaries (numbers only)
    version: int = 1

    @property
    def key(self) -> str:
        return PREFIX + self.id


def is_custom_key(key: str) -> bool:
    return isinstance(key, str) and key.startswith(PREFIX)


def clean_name(name: str) -> str:
    """Printable, single line, no brackets (the UI splits labels on ' (')."""
    name = re.sub(r"[\x00-\x1f\x7f()\[\]]", "", str(name)).strip()
    return re.sub(r"\s+", " ", name)[:NAME_MAX]


def _dir(folder: Path | None) -> Path:
    return Path(folder) if folder is not None else paths.CUSTOM_VOICES_DIR


def _path(voice_id: str, folder: Path | None = None) -> Path:
    if not _ID_RE.match(voice_id or ""):
        raise ValueError(f"invalid custom voice id {voice_id!r}")
    return _dir(folder) / f"{voice_id}.json"


def _from_dict(d: dict) -> CustomVoice:
    cv = CustomVoice(id=str(d["id"]), name=clean_name(d["name"]) or "Custom voice",
                     pitch_st=float(d["pitch_st"]), formant_st=float(d["formant_st"]),
                     eq=[[float(h), float(g)] for h, g in d.get("eq", [])],
                     eq_amount=float(d.get("eq_amount", 1.0)), created=str(d.get("created", "")),
                     measured=dict(d.get("measured", {})), version=int(d.get("version", 1)))
    if not _ID_RE.match(cv.id):
        raise ValueError(f"invalid id {cv.id!r}")
    if not (-12 <= cv.pitch_st <= 12 and -8 <= cv.formant_st <= 8 and 0 <= cv.eq_amount <= 1):
        raise ValueError("pitch / formant / eq_amount out of range")
    if any(h <= 0 or abs(g) > 24 for h, g in cv.eq):
        raise ValueError("invalid EQ band")
    return cv


def load(voice_id: str, folder: Path | None = None) -> CustomVoice:
    p = _path(voice_id, folder)
    return _from_dict(json.loads(p.read_text(encoding="utf-8")))


def list_voices(folder: Path | None = None) -> list[CustomVoice]:
    """All valid custom voices, oldest first. Broken files are skipped (and logged), never deleted."""
    d = _dir(folder)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            cv = _from_dict(json.loads(p.read_text(encoding="utf-8")))
            if cv.id != p.stem:
                raise ValueError("file name and id differ")
            out.append(cv)
        except Exception as e:
            logger.warning("Skipping custom voice file %s: %s", p, e)
    return sorted(out, key=lambda v: (v.created, v.id))


def exists(key: str, folder: Path | None = None) -> bool:
    if not is_custom_key(key):
        return False
    try:
        return _path(key[len(PREFIX):], folder).is_file()
    except ValueError:
        return False


def create(name: str, pitch_st: float, formant_st: float, eq: list[tuple[float, float]],
           measured: dict | None = None, folder: Path | None = None) -> CustomVoice:
    """Save a new voice under a fresh id. Raises if the name is empty; never overwrites a file."""
    name = clean_name(name)
    if not name:
        raise ValueError("Please enter a name for the voice.")
    d = _dir(folder)
    d.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    cv = CustomVoice(id=f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}", name=name,
                     pitch_st=round(float(pitch_st), 1), formant_st=round(float(formant_st), 1),
                     eq=[[float(h), round(float(g), 1)] for h, g in eq], created=now.isoformat(timespec="seconds"),
                     measured=measured or {})
    _from_dict(asdict(cv))                                  # same validation as on load
    p = _path(cv.id, folder)
    with open(p, "x", encoding="utf-8") as f:               # "x": fail instead of overwriting
        json.dump(asdict(cv), f, indent=2, ensure_ascii=False)
    return cv


def delete(voice_id: str, folder: Path | None = None) -> None:
    _path(voice_id, folder).unlink()
