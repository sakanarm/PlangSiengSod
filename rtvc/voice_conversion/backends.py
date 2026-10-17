"""Inference backends: ONNX Runtime (DirectML GPU or CPU) and a PyTorch CPU reference.

Every backend exposes the same four operations used by the converter:

    hubert(audio16k)                         -> features [T, C]
    encoder_flow(phone, lengths, pitch, noise, g) -> z  [1, inter, T]
    decoder(z, f0, g, noise)                 -> audio [L * upp]
    rmvpe(mel)                               -> salience [T, 360]

A GPU backend is only reported as usable after ``self_test`` has run real inference
through it and compared the result against the PyTorch CPU reference.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import paths

logger = logging.getLogger(__name__)

EXPORT_FORMAT_VERSION = 1  # bump when export wrappers change, invalidates the cache
OPSET = 17


# ----------------------------------------------------------------------------- HuBERT
def load_hubert_torch(hubert_dir: Path = paths.HUBERT_DIR):
    """Load the official Transformers-format HuBERT/ContentVec model on the CPU."""
    import torch
    from torch import nn
    from transformers import HubertModel

    class HubertModelWithFinalProj(HubertModel):
        def __init__(self, config):
            super().__init__(config)
            self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

    if not (hubert_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"HuBERT base model not found in {hubert_dir}. Run setup.ps1 or use "
            "'Download base models' in the app."
        )
    model = HubertModelWithFinalProj.from_pretrained(
        str(hubert_dir), local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _strip_weight_norm(module) -> None:
    """Fold weight-norm (old hooks or new parametrizations) into plain weights."""
    import torch
    from torch.nn.utils import parametrize

    for m in module.modules():
        if parametrize.is_parametrized(m):
            for name in list(m.parametrizations.keys()):
                parametrize.remove_parametrizations(m, name, leave_parametrized=True)
        for hook in list(getattr(m, "_forward_pre_hooks", {}).values()):
            if hook.__class__.__name__ == "WeightNorm":
                torch.nn.utils.remove_weight_norm(m, hook.name)


class HubertFeatures:
    """Picks the RVC feature layer: v1 = final_proj(layer 9), v2 = layer 12."""

    @staticmethod
    def module(model, version: str):
        from torch import nn

        class _Wrap(nn.Module):
            def __init__(self, m, v):
                super().__init__()
                self.m = m
                self.v = v

            def forward(self, audio):  # audio [1, N] float32 at 16 kHz
                if self.v == "v1":
                    out = self.m(input_values=audio, output_hidden_states=True, return_dict=True)
                    return self.m.final_proj(out.hidden_states[9])
                return self.m(input_values=audio, return_dict=True).last_hidden_state

        return _Wrap(model, version).eval()


# ------------------------------------------------------------------------ ONNX export
def _export(module, args, path: Path, input_names, output_names, dynamic_axes) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.onnx")
    with torch.no_grad():
        torch.onnx.export(
            module, args, str(tmp), input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, opset_version=OPSET, do_constant_folding=True,
            dynamo=False,
        )
    tmp.replace(path)


def export_hubert_onnx(version: str, out: Path) -> Path:
    import torch

    model = load_hubert_torch()
    _strip_weight_norm(model)
    wrap = HubertFeatures.module(model, version)
    dummy = torch.randn(1, 16000) * 0.1
    _export(wrap, (dummy,), out, ["audio"], ["feats"], {"audio": {1: "n"}, "feats": {1: "t"}})
    return out


def export_synth_onnx(net_g, has_f0: bool, out_dir: Path) -> tuple[Path, Path]:
    import torch

    from .synth import Decoder, EncoderFlow, speaker_embedding

    enc = EncoderFlow(net_g, has_f0).eval()
    dec = Decoder(net_g, has_f0).eval()
    t = 200
    c = net_g.enc_p.emb_phone.in_features
    inter = net_g.inter_channels
    g = speaker_embedding(net_g)
    phone = torch.randn(1, t, c)
    lengths = torch.tensor([t], dtype=torch.long)
    pitch = torch.randint(1, 255, (1, t), dtype=torch.long)
    noise = torch.randn(1, inter, t)
    enc_path = out_dir / "encoder_flow.onnx"
    dec_path = out_dir / "decoder.onnx"
    _export(enc, (phone, lengths, pitch, noise, g), enc_path,
            ["phone", "lengths", "pitch", "noise", "g"], ["z"],
            {"phone": {1: "t"}, "pitch": {1: "t"}, "noise": {2: "t"}, "z": {2: "t"}})
    upp = int(net_g.dec.upp) if hasattr(net_g.dec, "upp") else int(np.prod(net_g.upsample_rates))
    lz = 40
    z = torch.randn(1, inter, lz)
    f0 = torch.full((1, lz), 200.0)
    dnoise = torch.randn(1, lz * upp, 1)
    _export(dec, (z, f0, g, dnoise), dec_path, ["z", "f0", "g", "noise"], ["audio"],
            {"z": {2: "l"}, "f0": {1: "l"}, "noise": {1: "s"}, "audio": {2: "s"}})
    return enc_path, dec_path


def onnx_cache_dir(model_sha256: str) -> Path:
    import torch

    tag = f"{model_sha256[:16]}_f{EXPORT_FORMAT_VERSION}_t{torch.__version__.split('+')[0]}"
    return paths.ONNX_CACHE_DIR / tag


# --------------------------------------------------------------------------- backends
@dataclass
class BackendReport:
    requested: str
    active: str = ""
    device: str = ""
    verified: bool = False
    detail: str = ""
    max_rel_error: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)
    fallback_reason: str = ""

    def text(self) -> str:
        parts = [f"Backend: {self.active or 'none'}"]
        if self.device:
            parts.append(f"device: {self.device}")
        parts.append("VERIFIED by test inference" if self.verified else "NOT verified")
        if self.fallback_reason:
            parts.append(f"fallback because: {self.fallback_reason}")
        if self.max_rel_error:
            parts.append("rel.err vs CPU reference: " + ", ".join(
                f"{k}={v:.1e}" for k, v in self.max_rel_error.items()))
        return " | ".join(parts)


class OnnxRuntimeBackend:
    def __init__(self, files: dict[str, Path], use_dml: bool, device_id: int = 0):
        import onnxruntime as ort

        self.use_dml = use_dml
        self.name = "onnx-dml" if use_dml else "onnx-cpu"
        self.sessions = {}
        for key, path in files.items():
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.log_severity_level = 3
            if use_dml:
                # Required by the DirectML EP (see onnxruntime DirectML docs).
                so.enable_mem_pattern = False
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                providers = [("DmlExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
            else:
                so.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
                providers = ["CPUExecutionProvider"]
            sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
            active = sess.get_providers()
            if use_dml and active[0] != "DmlExecutionProvider":
                raise RuntimeError(f"DirectML provider was not activated for {path.name} (got {active})")
            self.sessions[key] = sess
        self.rmvpe_input = self.sessions["rmvpe"].get_inputs()[0].name if "rmvpe" in self.sessions else None

    def hubert(self, audio16k: np.ndarray) -> np.ndarray:
        return self.sessions["hubert"].run(None, {"audio": audio16k[None].astype(np.float32)})[0][0]

    def encoder_flow(self, phone, lengths, pitch, noise, g) -> np.ndarray:
        feed = {"phone": phone, "lengths": lengths, "noise": noise, "g": g}
        if "pitch" in {i.name for i in self.sessions["enc"].get_inputs()}:
            feed["pitch"] = pitch
        return self.sessions["enc"].run(None, feed)[0]

    def decoder(self, z, f0, g, noise) -> np.ndarray:
        sess = self.sessions["dec"]
        names = {i.name for i in sess.get_inputs()}
        feed = {k: v for k, v in {"z": z, "f0": f0, "g": g, "noise": noise}.items() if k in names}
        return sess.run(None, feed)[0].reshape(-1)

    def rmvpe(self, mel: np.ndarray) -> np.ndarray:
        return self.sessions["rmvpe"].run(None, {self.rmvpe_input: mel.astype(np.float32)})[0][0]


class TorchCPUBackend:
    name = "torch-cpu"

    def __init__(self, net_g, has_f0: bool, version: str, rmvpe_session=None):
        import torch

        from .synth import Decoder, EncoderFlow

        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        self.torch = torch
        self.enc = EncoderFlow(net_g, has_f0).eval()
        self.dec = Decoder(net_g, has_f0).eval()
        self.hub = HubertFeatures.module(load_hubert_torch(), version)
        self.rmvpe_session = rmvpe_session

    def hubert(self, audio16k):
        with self.torch.no_grad():
            return self.hub(self.torch.from_numpy(audio16k[None].astype(np.float32)))[0].numpy()

    def encoder_flow(self, phone, lengths, pitch, noise, g):
        t = self.torch.from_numpy
        with self.torch.no_grad():
            return self.enc(t(phone), t(lengths), t(pitch), t(noise), t(g)).numpy()

    def decoder(self, z, f0, g, noise):
        t = self.torch.from_numpy
        with self.torch.no_grad():
            return self.dec(t(z), t(f0), t(g), t(noise)).numpy().reshape(-1)

    def rmvpe(self, mel):
        name = self.rmvpe_session.get_inputs()[0].name
        return self.rmvpe_session.run(None, {name: mel.astype(np.float32)})[0][0]


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


def self_test(backend, reference: TorchCPUBackend | None, inter: int, gin_g: np.ndarray,
              feat_dim: int, upp: int, tolerance: float = 2e-2) -> tuple[dict, dict]:
    """Run real inference through ``backend``; compare with the CPU reference if given.

    Returns (relative_errors, timings_ms). Raises RuntimeError if results disagree.
    """
    rng = np.random.default_rng(1234)
    audio = (0.1 * np.sin(2 * np.pi * 180 * np.arange(16000) / 16000)
             + 0.01 * rng.standard_normal(16000)).astype(np.float32)
    t = 100
    phone = rng.standard_normal((1, t, feat_dim)).astype(np.float32)
    lengths = np.array([t], dtype=np.int64)
    pitch = rng.integers(1, 255, (1, t)).astype(np.int64)
    noise = rng.standard_normal((1, inter, t)).astype(np.float32)
    lz = 30
    f0 = np.full((1, lz), 180.0, dtype=np.float32)
    dnoise = rng.standard_normal((1, lz * upp, 1)).astype(np.float32)

    timings = {}
    outs = {}
    for name, fn in (
        ("hubert", lambda: backend.hubert(audio)),
        ("encoder_flow", lambda: backend.encoder_flow(phone, lengths, pitch, noise, gin_g)),
    ):
        fn()  # warm-up (DirectML compiles kernels on first run)
        t0 = time.perf_counter()
        outs[name] = fn()
        timings[name] = (time.perf_counter() - t0) * 1000
    z = outs["encoder_flow"][:, :, :lz].astype(np.float32)
    backend.decoder(z, f0, gin_g, dnoise)
    t0 = time.perf_counter()
    outs["decoder"] = backend.decoder(z, f0, gin_g, dnoise)
    timings["decoder"] = (time.perf_counter() - t0) * 1000

    for k, v in outs.items():
        if not np.all(np.isfinite(v)):
            raise RuntimeError(f"{backend.name}: {k} produced NaN/Inf")

    errors = {}
    if reference is not None and reference is not backend:
        ref = {
            "hubert": reference.hubert(audio),
            "encoder_flow": reference.encoder_flow(phone, lengths, pitch, noise, gin_g),
        }
        ref["decoder"] = reference.decoder(z, f0, gin_g, dnoise)
        for k in ref:
            errors[k] = _rel_err(outs[k], ref[k])
        bad = {k: v for k, v in errors.items() if v > tolerance}
        if bad:
            raise RuntimeError(f"{backend.name} disagrees with the CPU reference: {bad}")
    return errors, timings


def write_cache_manifest(cache: Path, info: dict) -> None:
    (cache / "manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
