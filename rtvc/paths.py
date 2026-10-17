"""Project-relative locations. Nothing here is an absolute, machine-specific path."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANUAL_FILE = PROJECT_ROOT / "docs" / "MANUAL.html"   # user manual, opened by the Manual button

# Allow relocating user data (models, cache, logs) without editing code.
DATA_ROOT = Path(os.environ.get("RTVC_DATA_DIR", PROJECT_ROOT)).resolve()

ASSETS_DIR = DATA_ROOT / "assets"          # official base models (HuBERT, RMVPE)
MODELS_DIR = DATA_ROOT / "models"          # user's trained RVC .pth / .index files
CACHE_DIR = DATA_ROOT / "cache"            # ONNX exports, test models
LOGS_DIR = DATA_ROOT / "logs"
RECORDINGS_DIR = DATA_ROOT / "recordings"  # default folder for the Record button (local only)
CONFIG_FILE = DATA_ROOT / "config.json"

HUBERT_DIR = ASSETS_DIR / "hubert_base"
RMVPE_ONNX = ASSETS_DIR / "rmvpe" / "rmvpe.onnx"
ONNX_CACHE_DIR = CACHE_DIR / "onnx"
TEST_MODELS_DIR = CACHE_DIR / "test_models"


def ensure_dirs() -> None:
    for d in (ASSETS_DIR, MODELS_DIR, CACHE_DIR, LOGS_DIR, ONNX_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
