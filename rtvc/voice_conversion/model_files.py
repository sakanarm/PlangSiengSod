"""Validation of RVC model files before they reach the real-time engine.

File kinds this module distinguishes (see MODEL_TRAINING.md):

* reference audio      - .wav/.flac/.mp3 recordings. NOT a model.
* training checkpoint  - G_*.pth / D_*.pth written during training ("model" + "optimizer").
                         Too large / wrong format for inference.
* inference model      - the small .pth exported by RVC ("weight" + "config" + "version").
                         This is what the real-time engine loads.
* feature index        - added_*.index (faiss). Optional, improves timbre similarity.

Security: .pth files are Python pickles. They are always opened with
``torch.load(weights_only=True)``, which refuses to execute arbitrary code, so a
malicious model downloaded from the internet cannot run commands on this machine.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

AUDIO_MAGIC = {
    b"RIFF": "WAV audio",
    b"fLaC": "FLAC audio",
    b"OggS": "OGG audio",
    b"ID3": "MP3 audio",
}
SUPPORTED_SR = (32000, 40000, 48000)
FEATURE_DIM = {"v1": 256, "v2": 768}
TEST_MODEL_MARKER = "RTVC-TEST-RANDOM-WEIGHTS"
MAX_INFERENCE_MODEL_MB = 400


class ModelValidationError(Exception):
    """Raised with a user-facing explanation of why a file cannot be used."""


@dataclass
class ModelInfo:
    path: Path
    version: str
    sample_rate: int
    has_f0: bool
    n_speakers: int
    info: str
    sha256: str
    size_mb: float
    is_test_model: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM[self.version]

    def summary(self) -> str:
        kind = "TEST MODEL (random weights - produces noise)" if self.is_test_model else "RVC model"
        return (
            f"{kind}: {self.version}, {self.sample_rate // 1000} kHz, "
            f"{'pitch-guided (f0)' if self.has_f0 else 'no f0'}, "
            f"{self.n_speakers} speaker slot(s), {self.size_mb:.1f} MB"
        )


@dataclass
class IndexInfo:
    path: Path
    dim: int
    ntotal: int
    warnings: list[str] = field(default_factory=list)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _sniff_audio(path: Path) -> str | None:
    with open(path, "rb") as f:
        head = f.read(12)
    for magic, kind in AUDIO_MAGIC.items():
        if head.startswith(magic):
            return kind
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return "MP3 audio"
    return None


def _load_checkpoint(path: Path) -> dict:
    import pickle

    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ModelValidationError(
            f"{path.name} contains Python objects other than plain weights and was refused "
            f"for safety (weights_only loading failed: {str(exc).splitlines()[0]})."
        ) from exc
    except Exception as exc:  # zip errors, truncated downloads, ...
        raise ModelValidationError(
            f"{path.name} is not a readable PyTorch checkpoint ({type(exc).__name__}: {exc}). "
            "The download may be incomplete or the file is not an RVC model."
        ) from exc


def check_checkpoint_dict(cpt: object, name: str = "model") -> tuple[str, int, bool, int, list[str]]:
    """Structural checks on an already-loaded checkpoint dict.

    Returns (version, sample_rate, has_f0, n_speakers, warnings).
    """
    if not isinstance(cpt, dict):
        raise ModelValidationError(f"{name}: checkpoint is a {type(cpt).__name__}, expected a dict.")
    keys = set(cpt.keys())
    if "weight" not in keys:
        if {"model", "optimizer"} & keys or "iteration" in keys or keys == {"model"}:
            raise ModelValidationError(
                f"{name} is a TRAINING checkpoint (G_*.pth / D_*.pth or a pretrained base), not an "
                "inference model. Use the small model RVC writes to assets/weights/<name>.pth "
                "after training (or export it with RVC's 'ckpt processing' tab)."
            )
        raise ModelValidationError(
            f"{name}: missing 'weight' entry. Keys found: {sorted(map(str, keys))[:10]}"
        )
    weight = cpt["weight"]
    config = cpt.get("config")
    if not isinstance(weight, dict) or not weight:
        raise ModelValidationError(f"{name}: 'weight' is empty or not a dict.")
    if not isinstance(config, (list, tuple)) or len(config) != 18:
        raise ModelValidationError(
            f"{name}: 'config' must be a list of 18 hyper-parameters, got "
            f"{type(config).__name__} of length {len(config) if hasattr(config, '__len__') else '?'}."
        )

    warnings: list[str] = []
    version = cpt.get("version", "v1")
    if version not in FEATURE_DIM:
        raise ModelValidationError(f"{name}: unsupported RVC version {version!r} (expected v1 or v2).")

    for k in ("enc_p.emb_phone.weight", "emb_g.weight", "dec.conv_pre.weight", "dec.conv_post.weight"):
        if k not in weight:
            raise ModelValidationError(f"{name}: required tensor {k!r} is missing.")

    in_dim = int(weight["enc_p.emb_phone.weight"].shape[1])
    if in_dim != FEATURE_DIM[version]:
        raise ModelValidationError(
            f"{name}: declared version {version} expects {FEATURE_DIM[version]}-dim features but the "
            f"network takes {in_dim}-dim features. The file is inconsistent."
        )

    has_f0 = int(cpt.get("f0", 1)) == 1
    if has_f0 and "enc_p.emb_pitch.weight" not in weight:
        raise ModelValidationError(f"{name}: f0 model without a pitch embedding (enc_p.emb_pitch.weight).")

    sr = config[-1]
    if isinstance(sr, str):
        sr = {"32k": 32000, "40k": 40000, "48k": 48000}.get(sr, sr)
    if sr not in SUPPORTED_SR:
        raise ModelValidationError(f"{name}: unsupported sample rate {sr!r}; expected one of {SUPPORTED_SR}.")
    upsample = config[12]
    if not isinstance(upsample, (list, tuple)) or math.prod(upsample) * 100 != sr:
        raise ModelValidationError(
            f"{name}: upsample rates {upsample!r} do not produce {sr} Hz at a 10 ms frame hop."
        )
    n_speakers = int(weight["emb_g.weight"].shape[0])
    if not has_f0:
        warnings.append(
            "This model has no pitch (f0) guidance; the pitch-shift setting has no effect and "
            "intonation may sound flat."
        )
    return version, int(sr), has_f0, n_speakers, warnings


def validate_model(path: str | Path, build_network: bool = True) -> ModelInfo:
    """Fully validate an RVC inference model. Raises ModelValidationError on failure."""
    path = Path(path)
    if not path.is_file():
        raise ModelValidationError(f"Model file not found: {path}")
    if path.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}:
        raise ModelValidationError(
            f"{path.name} is a recording (reference audio), not a trained model. Recordings must "
            "first be prepared and used to TRAIN a model - see MODEL_TRAINING.md."
        )
    if path.suffix.lower() == ".index":
        raise ModelValidationError(f"{path.name} is a feature index; select it in the 'Index' field instead.")
    audio_kind = _sniff_audio(path)
    if audio_kind:
        raise ModelValidationError(f"{path.name} is actually {audio_kind}, not a model.")
    if path.suffix.lower() == ".onnx":
        raise ModelValidationError(
            "ONNX files from other tools are not loaded directly; select the original RVC .pth. "
            "This application creates and verifies its own ONNX export."
        )
    size_mb = path.stat().st_size / 1e6
    if size_mb > MAX_INFERENCE_MODEL_MB:
        raise ModelValidationError(
            f"{path.name} is {size_mb:.0f} MB; RVC inference models are typically 50-60 MB. "
            "This is probably a training checkpoint."
        )

    cpt = _load_checkpoint(path)
    version, sr, has_f0, n_spk, warnings = check_checkpoint_dict(cpt, path.name)
    info_str = str(cpt.get("info", ""))

    if build_network:
        from .synth import build_synthesizer

        try:
            build_synthesizer(cpt)
        except ModelValidationError:
            raise
        except Exception as exc:
            raise ModelValidationError(
                f"{path.name}: weights do not fit the RVC network ({type(exc).__name__}: {exc})."
            ) from exc

    return ModelInfo(
        path=path,
        version=version,
        sample_rate=sr,
        has_f0=has_f0,
        n_speakers=n_spk,
        info=info_str,
        sha256=sha256_file(path),
        size_mb=size_mb,
        is_test_model=info_str.startswith(TEST_MODEL_MARKER),
        warnings=warnings,
    )


def validate_index(path: str | Path, expected_dim: int | None = None) -> IndexInfo:
    """Validate a faiss feature index (optional companion of the model)."""
    path = Path(path)
    if not path.is_file():
        raise ModelValidationError(f"Index file not found: {path}")
    if path.suffix.lower() != ".index":
        raise ModelValidationError(f"{path.name}: expected a .index file.")
    import faiss

    try:
        index = faiss.read_index(str(path))
    except Exception as exc:
        raise ModelValidationError(f"{path.name} is not a readable faiss index ({exc}).") from exc
    warnings = []
    if path.name.startswith("trained_"):
        warnings.append(
            "This is a 'trained_*.index' (cluster centroids only). Use the 'added_*.index' file "
            "from the same training run, otherwise retrieval has no effect."
        )
    if index.ntotal == 0:
        raise ModelValidationError(f"{path.name} contains no vectors.")
    if expected_dim is not None and index.d != expected_dim:
        raise ModelValidationError(
            f"{path.name} has {index.d}-dim vectors but the model needs {expected_dim}-dim "
            "(v1 = 256, v2 = 768). The index belongs to a different model."
        )
    return IndexInfo(path=path, dim=int(index.d), ntotal=int(index.ntotal), warnings=warnings)
