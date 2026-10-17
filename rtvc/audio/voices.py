"""Catalog of the DSP voices (no AI model, no download, nothing trained).

Every voice is a chain of the processors in ``processors.py``. Voices with a pitch /
formant stage can be fine-tuned live from the UI. These are effects: rhythm, accent and
the shape of the intonation remain the speaker's own.
"""

from __future__ import annotations

from dataclasses import dataclass

from .processors import (ChainProcessor, DistortionProcessor, RadioProcessor, ReverbProcessor,
                         RingModProcessor, VoiceShiftProcessor, WhisperProcessor)


@dataclass(frozen=True)
class Voice:
    key: str
    label: str
    group: str                  # "person" (a different-sounding human) or "effect"
    pitch_st: float = 0.0       # default pitch shift, semitones
    formant_st: float = 0.0     # default formant shift, semitones (+ = smaller/younger speaker)
    adjustable: bool = True     # has a pitch/formant stage the sliders can change
    gain_db: float = 0.0
    description: str = ""


VOICES: dict[str, Voice] = {v.key: v for v in (
    # --- other people (pitch + formant) ---
    Voice("deep_male", "ชายเสียงทุ้ม (Deep male)", "person", -4.0, -2.0,
          description="Lower pitch and a larger vocal tract."),
    Voice("big_man", "ชายร่างใหญ่ (Big man)", "person", -6.0, -3.5, gain_db=1.0,
          description="Very low, large, heavy-set voice."),
    Voice("young_male", "ชายหนุ่ม (Young male)", "person", 3.0, 1.0,
          description="Slightly higher, lighter male voice."),
    Voice("female", "ผู้หญิง (Female)", "person", 8.0, 3.0,
          description="Adult female range when the speaker has a typical male voice."),
    Voice("soft_female", "ผู้หญิงเสียงนุ่ม (Soft female)", "person", 6.0, 2.5,
          description="Lower, warmer female voice."),
    Voice("child", "เด็ก (Child)", "person", 11.0, 4.5,
          description="High pitch and a small vocal tract."),
    # --- effects ---
    Voice("chipmunk", "ชิปมังก์ / การ์ตูน (Chipmunk)", "effect", 12.0, 8.0,
          description="Cartoon voice: pitch and formants up."),
    Voice("monster", "ยักษ์ / มอนสเตอร์ (Monster)", "effect", -7.0, -5.0, gain_db=-6.0,
          description="Very low voice with saturation."),
    Voice("radio", "วิทยุสื่อสาร (Radio)", "effect", gain_db=-4.0,
          description="Narrow band-pass and saturation, like a walkie-talkie."),
    Voice("whisper", "กระซิบ / ผี (Whisper)", "effect", adjustable=False, gain_db=3.0,
          description="Pitch removed, breathy noise shaped like the words, a little echo."),
    Voice("cave", "ถ้ำ / ห้องโถง (Cave)", "effect", gain_db=-3.0,
          description="Large reverberant room."),
    Voice("alien", "เอเลี่ยน (Alien)", "effect", 3.0, 3.0, gain_db=3.5,
          description="Wobbling pitch (vibrato) and ring modulation."),
)}

DEFAULT_VOICE = "deep_male"


def voice_labels() -> list[str]:
    return [v.label for v in VOICES.values()]


def key_for_label(label: str) -> str:
    for v in VOICES.values():
        if v.label == label:
            return v.key
    raise KeyError(label)


def make_voice(key: str, samplerate: int, pitch_st: float | None = None, formant_st: float | None = None,
               gain_db: float = 0.0) -> ChainProcessor:
    """Build the processor chain for one voice. ``pitch_st``/``formant_st`` override the defaults."""
    if key not in VOICES:
        raise ValueError(f"unknown voice {key!r}")
    v = VOICES[key]
    sr = int(samplerate)
    p = v.pitch_st if pitch_st is None else float(pitch_st)
    f = v.formant_st if formant_st is None else float(formant_st)
    if key == "monster":
        stages = [VoiceShiftProcessor(sr, p, f), DistortionProcessor(drive=4.0, mix=0.45)]
    elif key == "radio":
        stages = [VoiceShiftProcessor(sr, p, f), RadioProcessor(sr)]
    elif key == "whisper":
        stages = [WhisperProcessor(sr), ReverbProcessor(sr, rt60_s=0.8, wet=0.25, dry=0.95)]
    elif key == "cave":
        stages = [VoiceShiftProcessor(sr, p, f), ReverbProcessor(sr, rt60_s=1.8, wet=0.4, dry=0.85)]
    elif key == "alien":
        stages = [VoiceShiftProcessor(sr, p, f, vibrato_st=1.0, vibrato_hz=6.0),
                  RingModProcessor(sr, freq_hz=70.0, mix=0.35)]
    else:
        stages = [VoiceShiftProcessor(sr, p, f)]
    return ChainProcessor(f"{v.label} (DSP effect, no AI model)", stages, gain_db=v.gain_db + gain_db)
