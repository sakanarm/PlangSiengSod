"""Pitch (F0) estimation for 16 kHz audio at a 10 ms hop. Ported from upstream RVC.

* ``rmvpe`` - official RMVPE network (ONNX), robust to noise. Mel front-end runs on CPU.
* ``pm``    - Praat autocorrelation (parselmouth). Very cheap; less robust.
"""

from __future__ import annotations

import numpy as np

F0_MIN = 50.0
F0_MAX = 1100.0
_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)


def coarse_pitch(f0: np.ndarray) -> np.ndarray:
    """Quantize Hz to the 1..255 mel bins used by the RVC pitch embedding."""
    f0_mel = 1127 * np.log(1 + f0 / 700)
    voiced = f0_mel > 0
    f0_mel[voiced] = (f0_mel[voiced] - _MEL_MIN) * 254 / (_MEL_MAX - _MEL_MIN) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    return np.rint(f0_mel).astype(np.int64)


def fill_unvoiced(f0: np.ndarray) -> np.ndarray:
    """Interpolate across unvoiced frames, as upstream realtime RVC does."""
    uv = f0 == 0
    if np.any(~uv) and np.any(uv):
        f0 = f0.copy()
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return f0


class RMVPE:
    SR = 16000
    N_FFT = 1024
    HOP = 160

    def __init__(self, run_network):
        """``run_network(mel[1,128,T]) -> salience[T,360]`` (ONNX session wrapper)."""
        import librosa
        import torch

        self.torch = torch
        self.run_network = run_network
        basis = librosa.filters.mel(sr=self.SR, n_fft=self.N_FFT, n_mels=128, fmin=30, fmax=8000, htk=True)
        self.mel_basis = torch.from_numpy(basis).float()
        self.window = torch.hann_window(self.N_FFT)
        cents = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents, (4, 4))

    def mel(self, audio: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None]
        spec = torch.stft(x, n_fft=self.N_FFT, hop_length=self.HOP, win_length=self.N_FFT,
                          window=self.window, center=True, return_complex=True)
        mag = spec.abs()
        mel = torch.matmul(self.mel_basis, mag)
        return torch.log(torch.clamp(mel, min=1e-5)).numpy()

    def decode(self, salience: np.ndarray, thred: float = 0.03) -> np.ndarray:
        center = np.argmax(salience, axis=1)
        padded = np.pad(salience, ((0, 0), (4, 4)))
        idx = center[:, None] + np.arange(9)[None, :]          # window of 9 bins in padded coords
        sal = np.take_along_axis(padded, idx, axis=1)
        cents = self.cents_mapping[idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.sum(sal * cents, axis=1) / np.sum(sal, axis=1)
        avg[np.max(padded, axis=1) <= thred] = 0
        avg = np.nan_to_num(avg)
        f0 = 10 * (2 ** (avg / 1200))
        f0[f0 == 10] = 0
        return f0

    def __call__(self, audio16k: np.ndarray, thred: float = 0.03) -> np.ndarray:
        mel = self.mel(audio16k)
        n_frames = mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad:
            mel = np.pad(mel, ((0, 0), (0, 0), (0, n_pad)))
        salience = self.run_network(mel)[:n_frames]
        return self.decode(salience, thred)


def pm_f0(audio16k: np.ndarray) -> np.ndarray:
    import parselmouth

    p_len = audio16k.shape[0] // 160 + 1
    f0_min = 65
    l_pad = int(np.ceil(1.5 / f0_min * 16000))
    snd = parselmouth.Sound(np.pad(audio16k.astype(np.float64), (l_pad, l_pad + 1)), 16000)
    pitch = snd.to_pitch_ac(time_step=0.01, voicing_threshold=0.6, pitch_floor=f0_min, pitch_ceiling=1100)
    f0 = pitch.selected_array["frequency"]
    if len(f0) < p_len:
        f0 = np.pad(f0, (0, p_len - len(f0)))
    return f0[:p_len]
