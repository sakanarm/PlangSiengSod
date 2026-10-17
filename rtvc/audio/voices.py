"""Catalog of the DSP voices (no AI model, no download, nothing trained).

Every voice is a chain of the processors in ``processors.py``. Voices with a pitch /
formant stage can be fine-tuned live from the UI. These are effects: rhythm, accent and
the shape of the intonation remain the speaker's own.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import custom_voices
from .processors import (BitcrushProcessor, ChainProcessor, DistortionProcessor, EqProcessor, FilterProcessor,
                         ModDelayProcessor, ParallelProcessor, PassthroughProcessor, RadioProcessor, ReverbProcessor, RingModProcessor,
                         RoughnessProcessor, ToneProcessor, VoiceShiftProcessor, WhisperProcessor)


@dataclass(frozen=True)
class Voice:
    key: str
    label: str
    group: str                  # "person" (a different-sounding human), "effect" or "custom" (from a sample)
    pitch_st: float = 0.0       # default pitch shift, semitones
    formant_st: float = 0.0     # default formant shift, semitones (+ = smaller/younger speaker)
    adjustable: bool = True     # has a pitch/formant stage the sliders can change
    gain_db: float = 0.0
    description: str = ""
    vibrato_st: float = 0.0     # pitch wobble depth, semitones (0 = none)
    vibrato_hz: float = 0.0
    eq: tuple = ()              # ((band Hz, gain dB), ...) - custom voices from a sample
    eq_amount: float = 1.0
    jitter_st: float = 0.0      # random frame-to-frame pitch wobble, semitones (rough / sick voices)


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
    Voice("elderly", "คนแก่เสียงสั่น (Elderly)", "person", 2.0, 0.5, vibrato_st=0.5, vibrato_hz=5.5,
          description="Shaky voice: a slow tremor on the pitch, slightly thinner."),
    Voice("tiny", "ตัวจิ๋ว (Tiny)", "person", 0.0, 5.0,
          description="Small vocal tract at the same pitch: sounds tiny without sounding higher."),
    Voice("gentle_giant", "ยักษ์ใจดี (Gentle giant)", "person", 0.0, -5.0,
          description="Large vocal tract at the same pitch: big and warm, not menacing."),
    # --- your own voice, but unwell: hides the usual voice quality, still sounds human ---
    Voice("cold", "เป็นหวัด (Blocked nose)", "person", -1.0, -0.5,
          description="Stuffy, boxy voice as if the nose is blocked by a cold."),
    Voice("sore_throat", "เจ็บคอ (Sore throat)", "person", -1.5, 0.0, gain_db=-1.0, jitter_st=0.35,
          description="Strained, slightly shaky and breathy, quieter highs."),
    Voice("hoarse", "เสียงแหบ (Hoarse)", "person", -2.0, -0.5, jitter_st=0.25,
          description="Raspy and airy, like after shouting all night."),
    Voice("hoarse_own", "แหบ (เสียงตัวเอง) (Hoarse, own voice)", "person", adjustable=False,
          description="Your own voice, unchanged in pitch and size, just raspy and airy."),
    # --- effects ---
    Voice("chipmunk", "ชิปมังก์ / การ์ตูน (Chipmunk)", "effect", 12.0, 8.0,
          description="Cartoon voice: pitch and formants up."),
    Voice("monster", "ยักษ์ / มอนสเตอร์ (Monster)", "effect", -7.0, -5.0, gain_db=-6.0,
          description="Very low voice with saturation."),
    Voice("radio", "วิทยุสื่อสาร (Radio)", "effect", gain_db=-4.0,
          description="Narrow band-pass and saturation, like a walkie-talkie."),
    Voice("whisper", "กระซิบ (Whisper)", "effect", adjustable=False, gain_db=3.0,
          description="Pitch removed, breathy noise shaped like the words, a little echo."),
    Voice("cave", "ถ้ำ / ห้องโถง (Cave)", "effect", gain_db=-3.0,
          description="Large reverberant room."),
    Voice("alien", "เอเลี่ยน (Alien)", "effect", 3.0, 3.0, gain_db=3.5, vibrato_st=1.0, vibrato_hz=6.0,
          description="Wobbling pitch (vibrato) and ring modulation."),
    Voice("masked", "วายร้ายใส่หน้ากาก (Masked villain)", "effect", -5.0, -3.0, gain_db=2.0,
          description="Low voice, muffled as if behind a mask, with a metallic edge."),
    # --- low-latency effects: no pitch stage, so (almost) no added delay ---
    Voice("retro", "เกม 8-bit (Retro game)", "effect", adjustable=False,
          description="Lo-fi game console sound: lower sample rate and coarse 8-bit steps."),
    Voice("jet", "เครื่องบินเจ็ท / ไซบอร์ก (Jet flanger)", "effect", adjustable=False, gain_db=-3.0,
          description="Sweeping jet-plane whoosh (flanger), like a sci-fi cyborg."),
    Voice("underwater", "ใต้น้ำ (Underwater)", "effect", adjustable=False, gain_db=2.0,
          description="Muffled and wobbling, as if talking under water."),
    Voice("ghost", "ผี / วิญญาณ (Ghost)", "effect", adjustable=False, gain_db=1.0,
          description="Breathy whisper with a slow swirl and a long, haunting tail."),
    Voice("stadium", "โฆษกสนาม (Stadium announcer)", "effect", adjustable=False, gain_db=-1.0,
          description="Loudspeaker tone with echoes bouncing around a stadium."),
)}

DEFAULT_VOICE = "deep_male"


def voice_labels() -> list[str]:
    return [v.label for v in VOICES.values()]


def key_for_label(label: str) -> str:
    for v in VOICES.values():
        if v.label == label:
            return v.key
    raise KeyError(label)


def custom_voice(cv: custom_voices.CustomVoice) -> Voice:
    return Voice(cv.key, f"{cv.name} (My voice)", "custom", cv.pitch_st, cv.formant_st,
                 description="Made from a voice sample: matched pitch, voice size and tone (EQ). "
                             "Similar, not the same person.",
                 eq=tuple((h, g) for h, g in cv.eq), eq_amount=cv.eq_amount)


def get_voice(key: str) -> Voice:
    """A built-in voice or a saved custom voice ("custom:<id>"). Raises ValueError if unknown."""
    if key in VOICES:
        return VOICES[key]
    if custom_voices.is_custom_key(key):
        try:
            return custom_voice(custom_voices.load(key[len(custom_voices.PREFIX):]))
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ValueError(f"custom voice {key!r} cannot be loaded: {e}") from e
    raise ValueError(f"unknown voice {key!r}")


def voice_exists(key: str) -> bool:
    return key in VOICES or custom_voices.exists(key)


def make_voice(key: str, samplerate: int, pitch_st: float | None = None, formant_st: float | None = None,
               gain_db: float = 0.0, eq_amount: float | None = None) -> ChainProcessor:
    """Build the processor chain for one voice. ``pitch_st``/``formant_st``/``eq_amount`` override the defaults."""
    v = get_voice(key)
    sr = int(samplerate)
    p = v.pitch_st if pitch_st is None else float(pitch_st)
    f = v.formant_st if formant_st is None else float(formant_st)
    if v.group == "custom":
        stages = [VoiceShiftProcessor(sr, p, f)]
        if v.eq:
            stages.append(EqProcessor(sr, list(v.eq), v.eq_amount if eq_amount is None else float(eq_amount)))
    elif key == "monster":
        stages = [VoiceShiftProcessor(sr, p, f), DistortionProcessor(drive=4.0, mix=0.45)]
    elif key == "radio":
        stages = [VoiceShiftProcessor(sr, p, f), RadioProcessor(sr)]
    elif key == "whisper":
        stages = [WhisperProcessor(sr), ReverbProcessor(sr, rt60_s=0.8, wet=0.25, dry=0.95)]
    elif key == "cave":
        stages = [VoiceShiftProcessor(sr, p, f), ReverbProcessor(sr, rt60_s=1.8, wet=0.4, dry=0.85)]
    elif key == "alien":
        stages = [VoiceShiftProcessor(sr, p, f, v.vibrato_st, v.vibrato_hz),
                  RingModProcessor(sr, freq_hz=70.0, mix=0.35)]
    elif key == "masked":
        stages = [VoiceShiftProcessor(sr, p, f), RadioProcessor(sr, low_hz=120.0, high_hz=3500.0, drive=2.0),
                  RingModProcessor(sr, freq_hz=35.0, mix=0.25)]
    elif key == "cold":
        stages = [VoiceShiftProcessor(sr, p, f),
                  ToneProcessor(sr, peaks=[(450.0, 5.0, 1.0), (2500.0, -6.0, 1.0)], high_hz=5000.0)]
    elif key == "sore_throat":
        stages = [ParallelProcessor(sr, [(VoiceShiftProcessor(sr, p, f, jitter_st=v.jitter_st), 0.85),
                                         (WhisperProcessor(sr, breath_db=0.0), 0.3)]),
                  RoughnessProcessor(sr, depth=0.3),
                  ToneProcessor(sr, peaks=[(1500.0, 3.0, 1.2)], high_hz=4500.0)]
    elif key == "hoarse":
        stages = [ParallelProcessor(sr, [(VoiceShiftProcessor(sr, p, f, jitter_st=v.jitter_st), 0.75),
                                         (WhisperProcessor(sr, breath_db=4.0), 0.5)]),
                  RoughnessProcessor(sr, depth=0.5, low_hz=30.0, high_hz=120.0),
                  DistortionProcessor(drive=2.5, mix=0.25)]
    elif key == "hoarse_own":
        # no pitch stage: the voice stays recognisably yours and only the breath layer adds delay
        stages = [ParallelProcessor(sr, [(PassthroughProcessor(), 0.8), (WhisperProcessor(sr, breath_db=4.0), 0.4)]),
                  RoughnessProcessor(sr, depth=0.4, low_hz=30.0, high_hz=120.0),
                  DistortionProcessor(drive=2.0, mix=0.2)]
    elif key == "retro":
        stages = [BitcrushProcessor(sr, rate_hz=6000.0, bits=8)]
    elif key == "jet":
        stages = [ModDelayProcessor(sr, base_ms=3.0, depth_ms=2.0, rate_hz=0.3, feedback=0.6, wet=0.7, dry=0.7)]
    elif key == "underwater":
        stages = [ModDelayProcessor(sr, base_ms=8.0, depth_ms=2.5, rate_hz=1.2, wet=1.0, dry=0.0),
                  FilterProcessor(sr, high_hz=900.0)]
    elif key == "ghost":
        stages = [WhisperProcessor(sr, breath_db=3.0),
                  ModDelayProcessor(sr, base_ms=4.0, depth_ms=3.0, rate_hz=0.15, feedback=0.5, wet=0.6, dry=0.8),
                  ReverbProcessor(sr, rt60_s=3.5, wet=0.55, dry=0.7)]
    elif key == "stadium":
        stages = [FilterProcessor(sr, low_hz=300.0, high_hz=4500.0, order=2), DistortionProcessor(drive=2.0, mix=0.3),
                  ModDelayProcessor(sr, base_ms=220.0, feedback=0.35, wet=0.4, dry=1.0)]
    else:
        stages = [VoiceShiftProcessor(sr, p, f, v.vibrato_st, v.vibrato_hz)]
    return ChainProcessor(f"{v.label} (DSP effect, no AI model)", stages, gain_db=v.gain_db + gain_db)
