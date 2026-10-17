"""RVC synthesizer construction plus streaming/ONNX-friendly wrappers.

The official ``net_g.infer`` draws random noise internally and slices with Python
integers, which makes it impossible to (a) export one ONNX graph that works for any
block size and (b) check an accelerated backend against the PyTorch reference.

Here inference is split into two networks whose noise is an explicit input:

* ``EncoderFlow``  phone features + coarse pitch + noise -> latent z   (all frames)
* ``Decoder``      z slice + f0 slice + noise          -> waveform     (played frames only)

Given identical inputs, PyTorch and ONNX Runtime must produce matching audio; the
backend self-test relies on that. The math mirrors ``SynthesizerTrnMs*NSFsid.infer``
and ``GeneratorNSF.forward`` of the vendored upstream code.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .rvc_lib import models as rvc_models

NOISE_SCALE = 0.66666  # upstream z_p noise temperature


def build_synthesizer(cpt: dict) -> nn.Module:
    """Instantiate the RVC network for a validated checkpoint and load its weights."""
    from .model_files import ModelValidationError

    config = list(cpt["config"])
    config[-3] = cpt["weight"]["emb_g.weight"].shape[0]  # speaker count, as upstream does
    if isinstance(config[-1], str):
        config[-1] = {"32k": 32000, "40k": 40000, "48k": 48000}[config[-1]]
    version = cpt.get("version", "v1")
    has_f0 = int(cpt.get("f0", 1)) == 1
    cls = {
        ("v1", True): rvc_models.SynthesizerTrnMs256NSFsid,
        ("v1", False): rvc_models.SynthesizerTrnMs256NSFsid_nono,
        ("v2", True): rvc_models.SynthesizerTrnMs768NSFsid,
        ("v2", False): rvc_models.SynthesizerTrnMs768NSFsid_nono,
    }[(version, has_f0)]
    net_g = cls(*config, is_half=False) if has_f0 else cls(*config)
    del net_g.enc_q
    weight = {k: v for k, v in cpt["weight"].items() if not k.startswith("enc_q.")}
    result = net_g.load_state_dict(weight, strict=False)
    if result.missing_keys:
        raise ModelValidationError(
            f"Model is missing {len(result.missing_keys)} tensors, e.g. {result.missing_keys[:3]}"
        )
    if result.unexpected_keys:
        raise ModelValidationError(
            f"Model has {len(result.unexpected_keys)} unexpected tensors, e.g. {result.unexpected_keys[:3]}"
        )
    net_g = net_g.float().eval()
    net_g.remove_weight_norm()
    for p in net_g.parameters():
        p.requires_grad_(False)
    return net_g


class EncoderFlow(nn.Module):
    """phone [1,T,C], phone_lengths [1], (pitch [1,T] int64), noise [1,inter,T], g [1,gin,1] -> z [1,inter,T]."""

    def __init__(self, net_g: nn.Module, has_f0: bool):
        super().__init__()
        self.enc_p = net_g.enc_p
        self.flow = net_g.flow
        self.has_f0 = has_f0

    def forward(self, phone, phone_lengths, pitch, noise, g):
        m_p, logs_p, x_mask = self.enc_p(phone, pitch if self.has_f0 else None, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * noise * NOISE_SCALE) * x_mask
        return self.flow(z_p, x_mask, g=g, reverse=True)


class Decoder(nn.Module):
    """z [1,inter,L], f0 [1,L] (Hz), g [1,gin,1], noise [1,L*upp,1] -> audio [1,1,L*upp].

    For models without f0 the f0 and noise inputs are accepted but unused.
    """

    def __init__(self, net_g: nn.Module, has_f0: bool):
        super().__init__()
        self.dec = net_g.dec
        self.has_f0 = has_f0
        if has_f0:
            src = self.dec.m_source
            sine = src.l_sin_gen
            if sine.harmonic_num != 0:
                raise ValueError("Only harmonic_num=0 NSF sources are supported (RVC default).")
            self.upp = int(self.dec.upp)
            self.sampling_rate = float(sine.sampling_rate)
            self.sine_amp = float(sine.sine_amp)
            self.noise_std = float(sine.noise_std)
            self.voiced_threshold = float(sine.voiced_threshold)

    def _harmonic_source(self, f0, noise):
        # SineGen._f02sine with harmonic_num=0 (rand_ini is identically zero then).
        f0 = f0.unsqueeze(-1)                                              # [1, L, 1]
        steps = torch.arange(1, self.upp + 1, device=f0.device, dtype=f0.dtype)
        rad = f0 / self.sampling_rate * steps                              # [1, L, upp]
        rad2 = torch.fmod(rad[..., -1:] + 0.5, 1.0) - 0.5                  # [1, L, 1]
        rad_acc = torch.fmod(torch.cumsum(rad2, dim=1), 1.0)
        rad_acc = torch.cat([torch.zeros_like(rad_acc[:, :1]), rad_acc[:, :-1]], dim=1)
        rad = (rad + rad_acc).reshape(f0.shape[0], -1, 1)                  # [1, L*upp, 1]
        sine = torch.sin(2 * math.pi * rad) * self.sine_amp
        uv = (f0 > self.voiced_threshold).to(f0.dtype)                     # [1, L, 1]
        uv = uv.expand(-1, -1, self.upp).reshape(f0.shape[0], -1, 1)       # nearest upsample
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        sine = sine * uv + noise_amp * noise
        merged = self.dec.m_source.l_tanh(self.dec.m_source.l_linear(sine))
        return merged.transpose(1, 2)                                      # [1, 1, L*upp]

    def forward(self, z, f0, g, noise):
        dec = self.dec
        if not self.has_f0:
            return dec(z, g=g)
        har_source = self._harmonic_source(f0, noise)
        x = dec.conv_pre(z) + dec.cond(g)
        for i, (up, noise_conv) in enumerate(zip(dec.ups, dec.noise_convs)):
            x = F.leaky_relu(x, dec.lrelu_slope)
            x = up(x)
            x = x + noise_conv(har_source)
            xs = None
            for j in range(dec.num_kernels):
                r = dec.resblocks[i * dec.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / dec.num_kernels
        x = F.leaky_relu(x)
        x = dec.conv_post(x)
        return torch.tanh(x)


def speaker_embedding(net_g: nn.Module, sid: int = 0) -> torch.Tensor:
    """g = emb_g(sid) as [1, gin, 1]."""
    with torch.no_grad():
        return net_g.emb_g(torch.tensor([sid], dtype=torch.long)).unsqueeze(-1)


def make_random_checkpoint(version: str = "v2", sample_rate: int = 40000, has_f0: bool = True,
                           seed: int = 0) -> dict:
    """A structurally valid RVC checkpoint with RANDOM weights.

    Used only for tests and latency benchmarks: compute cost equals a real model of
    the same configuration, but the output is noise, never a voice.
    """
    from .model_files import TEST_MODEL_MARKER

    presets = {
        32000: ([10, 8, 2, 2], [20, 16, 4, 4], 513),
        40000: ([10, 10, 2, 2], [16, 16, 4, 4], 1025),
        48000: ([12, 10, 2, 2], [24, 20, 4, 4], 1025),
    }
    up_rates, up_kernels, spec_ch = presets[sample_rate]
    config = [spec_ch, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11],
              [[1, 3, 5], [1, 3, 5], [1, 3, 5]], up_rates, 512, up_kernels, 109, 256, sample_rate]
    torch.manual_seed(seed)
    cls = {
        ("v1", True): rvc_models.SynthesizerTrnMs256NSFsid,
        ("v1", False): rvc_models.SynthesizerTrnMs256NSFsid_nono,
        ("v2", True): rvc_models.SynthesizerTrnMs768NSFsid,
        ("v2", False): rvc_models.SynthesizerTrnMs768NSFsid_nono,
    }[(version, has_f0)]
    net = cls(*config, is_half=False) if has_f0 else cls(*config)
    weight = {k: v.half() for k, v in net.state_dict().items() if not k.startswith("enc_q.")}
    return {
        "weight": weight,
        "config": config,
        "info": f"{TEST_MODEL_MARKER} - not a voice model",
        "sr": f"{sample_rate // 1000}k",
        "f0": int(has_f0),
        "version": version,
    }

