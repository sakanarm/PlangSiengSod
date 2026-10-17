"""Official base models required for inference, pinned by SHA-256.

Source: the official RVC Hugging Face repository linked from the RVC README
(https://huggingface.co/lj1995/VoiceConversionWebUI). Nothing else is downloaded,
and every file is verified against the hash below before it is used.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .. import paths
from .model_files import sha256_file

logger = logging.getLogger(__name__)

HF_BASE = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"


@dataclass(frozen=True)
class Asset:
    remote: str
    local: Path
    sha256: str
    size: int
    purpose: str
    optional: bool = False   # optional assets are only fetched on explicit request

    @property
    def url(self) -> str:
        return HF_BASE + self.remote


REQUIRED_ASSETS = (
    Asset("hubert_base/config.json", paths.HUBERT_DIR / "config.json",
          "0346950779dfb7f9316fa74ed846e2b8a22a08eedfdc5387b73f327cb1a4a7cf", 1492,
          "HuBERT/ContentVec speech-content encoder (config)"),
    Asset("hubert_base/preprocessor_config.json", paths.HUBERT_DIR / "preprocessor_config.json",
          "7c1976a680fb7acc757cd36fb08eef878fa36c70b4c9d2d595df9c608bbbbf0e", 225,
          "HuBERT/ContentVec feature-extractor settings"),
    Asset("hubert_base/pytorch_model.bin", paths.HUBERT_DIR / "pytorch_model.bin",
          "cc8c20f4b90a520757260197a3ff2505705a7adbd20ad9eeaa4e1a9b38442ef5", 189206711,
          "HuBERT/ContentVec weights"),
    Asset("rmvpe.onnx", paths.RMVPE_ONNX,
          "5370e71ac80af8b4b7c793d27efd51fd8bf962de3a7ede0766dac0befa3660fd", 361688443,
          "RMVPE pitch (f0) estimator, ONNX (optional: the 'pm' f0 method needs no download)",
          optional=True),
)


def asset_status(verify_hash: bool = False) -> list[tuple[Asset, str]]:
    """Return (asset, status) with status in {'ok', 'missing', 'wrong size', 'corrupt'}."""
    out = []
    for a in REQUIRED_ASSETS:
        if not a.local.is_file():
            out.append((a, "missing"))
        elif a.local.stat().st_size != a.size:
            out.append((a, "wrong size"))
        elif verify_hash and sha256_file(a.local) != a.sha256:
            out.append((a, "corrupt"))
        else:
            out.append((a, "ok"))
    return out


def missing_assets(verify_hash: bool = False, include_optional: bool = False) -> list[Asset]:
    return [a for a, s in asset_status(verify_hash) if s != "ok" and (include_optional or not a.optional)]


def download_asset(asset: Asset, progress: Callable[[str, int, int], None] | None = None) -> None:
    """Download one asset to a temp file, verify SHA-256, then move into place."""
    asset.local.parent.mkdir(parents=True, exist_ok=True)
    tmp = asset.local.with_suffix(asset.local.suffix + ".part")
    req = urllib.request.Request(asset.url, headers={"User-Agent": "rtvc-setup"})
    logger.info("Downloading %s", asset.url)
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(asset.remote, done, asset.size)
    digest = sha256_file(tmp)
    if digest != asset.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {asset.remote}: expected {asset.sha256}, got {digest}. "
            "The file was discarded."
        )
    tmp.replace(asset.local)


def download_missing(progress: Callable[[str, int, int], None] | None = None,
                     include_optional: bool = False) -> list[str]:
    done = []
    for a in missing_assets(verify_hash=True, include_optional=include_optional):
        download_asset(a, progress)
        done.append(a.remote)
    return done


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    def _p(name, done, total):
        pct = 100 * done / total if total else 0
        sys.stdout.write(f"\r  {name}: {done / 1e6:7.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)")
        sys.stdout.flush()
        if done >= total:
            sys.stdout.write("\n")

    for asset, status in asset_status(verify_hash=True):
        size = f"{asset.size / 1e6:.0f} MB"
        print(f"[{status:>10}] {asset.remote} {size:>7}  ({asset.purpose})")
    if "--download" in sys.argv:
        got = download_missing(_p, include_optional="--with-rmvpe" in sys.argv)
        print("Downloaded and verified:", ", ".join(got) if got else "nothing (all present)")
