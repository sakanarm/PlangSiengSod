"""Hardware / software diagnostics. Reports only what was actually detected or tested.

GPU acceleration is reported as "verified" only after a real inference test passed
(``verify_directml``); merely having a GPU or a provider listed is not enough.

CLI:  python -m rtvc.hardware.diagnostics [--verify-gpu]
"""

from __future__ import annotations

import importlib.util
import json
import logging
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GpuInfo:
    name: str
    driver_version: str = ""
    driver_date: str = ""
    vram_mb: int | None = None
    vendor: str = ""


@dataclass
class Report:
    windows: str = ""
    python: str = ""
    cpu: str = ""
    cores_physical: int | None = None
    cores_logical: int | None = None
    ram_total_gb: float | None = None
    ram_available_gb: float | None = None
    gpus: list[GpuInfo] = field(default_factory=list)
    ai_addon_installed: bool = False
    onnxruntime_version: str = ""
    ort_providers: list[str] = field(default_factory=list)
    torch_version: str = ""
    directml_verified: str = "not tested"
    audio_inputs: list[str] = field(default_factory=list)
    audio_outputs: list[str] = field(default_factory=list)
    vb_cable: bool = False
    virtual_devices: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def text(self) -> str:
        L = [f"Windows        : {self.windows}", f"Python         : {self.python}",
             f"CPU            : {self.cpu} ({self.cores_physical} cores / {self.cores_logical} threads)",
             f"RAM            : {self.ram_total_gb} GB total, {self.ram_available_gb} GB free"]
        for g in self.gpus:
            vram = f", {g.vram_mb} MB" if g.vram_mb else ""
            L.append(f"GPU            : {g.name} (driver {g.driver_version} {g.driver_date}{vram})")
        L.append(f"AI add-on      : {'installed' if self.ai_addon_installed else 'not installed (lite mode: robot voice only)'}")
        if self.ai_addon_installed:
            L.append(f"ONNX Runtime   : {self.onnxruntime_version or '-'}  providers: {', '.join(self.ort_providers) or '-'}")
            L.append(f"PyTorch (CPU)  : {self.torch_version or '-'}")
        L.append(f"DirectML GPU   : {self.directml_verified}")
        L.append(f"VB-CABLE       : {'installed' if self.vb_cable else 'NOT FOUND'}")
        L.append(f"Virtual devices: {', '.join(self.virtual_devices) or '-'}")
        L.append("Inputs         : " + "; ".join(self.audio_inputs))
        L.append("Outputs        : " + "; ".join(self.audio_outputs))
        L += [f"Note           : {n}" for n in self.notes]
        return "\n".join(L)


def _windows_version() -> str:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as k:
            get = lambda n: winreg.QueryValueEx(k, n)[0]  # noqa: E731
            build = int(get("CurrentBuildNumber"))
            name = get("ProductName")
            if build >= 22000:  # ProductName still says "Windows 10" on Windows 11
                name = name.replace("Windows 10", "Windows 11")
            try:
                disp = get("DisplayVersion")
            except OSError:
                disp = ""
            return f"{name} {disp} (build {build}.{get('UBR')})".replace("  ", " ")
    except Exception:
        return platform.platform()


def _cpu_name() -> str:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor()


def _gpus() -> list[GpuInfo]:
    cmd = ("Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate,"
           "AdapterRAM,AdapterCompatibility | ConvertTo-Json -Compress")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                             capture_output=True, text=True, timeout=20,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        data = json.loads(out) if out.strip() else []
    except Exception as e:
        logger.warning("GPU query failed: %s", e)
        return []
    if isinstance(data, dict):
        data = [data]
    gpus = []
    for d in data:
        date = str(d.get("DriverDate") or "")
        if date.startswith("/Date("):          # PowerShell 5 JSON date
            import datetime

            ms = int(date[6:date.index(")")].split("+")[0].split("-")[0] or 0)
            date = datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc).strftime("%Y-%m-%d")
        ram = d.get("AdapterRAM")
        gpus.append(GpuInfo(d.get("Name") or "?", d.get("DriverVersion") or "", date,
                            int(ram) // (1 << 20) if isinstance(ram, int) and ram > 0 else None,
                            d.get("AdapterCompatibility") or ""))
    return gpus


def collect(include_audio: bool = True) -> Report:
    r = Report(windows=_windows_version(), python=sys.version.split()[0], cpu=_cpu_name())
    try:
        import psutil

        r.cores_physical = psutil.cpu_count(logical=False)
        r.cores_logical = psutil.cpu_count(logical=True)
        vm = psutil.virtual_memory()
        r.ram_total_gb = round(vm.total / 2**30, 1)
        r.ram_available_gb = round(vm.available / 2**30, 1)
    except Exception as e:
        r.notes.append(f"psutil unavailable: {e}")
    r.gpus = _gpus()
    if any("nvidia" in g.name.lower() for g in r.gpus):
        r.notes.append("An NVIDIA GPU is present but is not required or used; DirectML targets the selected adapter.")
    r.ai_addon_installed = all(importlib.util.find_spec(m) for m in ("onnxruntime", "torch"))
    if importlib.util.find_spec("onnxruntime"):
        try:
            import onnxruntime as ort

            r.onnxruntime_version = ort.__version__
            r.ort_providers = ort.get_available_providers()
        except Exception as e:
            r.notes.append(f"onnxruntime import failed: {e}")
    if importlib.util.find_spec("torch"):
        try:
            import torch

            r.torch_version = torch.__version__
        except Exception as e:
            r.notes.append(f"torch import failed: {e}")
    if not r.ai_addon_installed:
        r.directml_verified = "not tested (AI add-on not installed; not needed for robot voice)"
    if include_audio:
        try:
            from ..audio import devices as dv

            ds = dv.list_devices()
            api = "Windows WASAPI"
            r.audio_inputs = [d.name for d in dv.inputs(ds, api)]
            r.audio_outputs = [d.name for d in dv.outputs(ds, api)]
            r.vb_cable = dv.cable_installed(ds)
            r.virtual_devices = sorted({d.name for d in ds if d.is_virtual and d.hostapi == api})
        except Exception as e:
            r.notes.append(f"Audio enumeration failed: {e}")
    return r


def verify_directml(device_id: int = 0) -> tuple[bool, str]:
    """Run a small real model on DirectML and compare against CPU. Only a pass counts as verified."""
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper
    except ImportError:
        return False, "not tested (AI add-on not installed)"
    if "DmlExecutionProvider" not in ort.get_available_providers():
        return False, "DirectML provider not available in this onnxruntime build"
    rng = np.random.default_rng(0)
    w1 = rng.standard_normal((256, 512)).astype(np.float32) * 0.05
    w2 = rng.standard_normal((512, 256)).astype(np.float32) * 0.05
    g = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w1"], ["h"]), helper.make_node("Tanh", ["h"], ["t"]),
         helper.make_node("MatMul", ["t", "w2"], ["y"])], "dmltest",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 256])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 256])],
        [helper.make_tensor("w1", TensorProto.FLOAT, w1.shape, w1.ravel()),
         helper.make_tensor("w2", TensorProto.FLOAT, w2.shape, w2.ravel())])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    data = model.SerializeToString()
    x = rng.standard_normal((64, 256)).astype(np.float32)
    try:
        so = ort.SessionOptions()
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        dml = ort.InferenceSession(data, so, providers=[("DmlExecutionProvider", {"device_id": device_id})])
        if dml.get_providers()[0] != "DmlExecutionProvider":
            return False, f"DirectML session fell back to {dml.get_providers()}"
        cpu = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
        y_d = dml.run(None, {"x": x})[0]
        y_c = cpu.run(None, {"x": x})[0]
    except Exception as e:
        return False, f"DirectML test failed: {e}"
    err = float(np.max(np.abs(y_d - y_c)) / (np.max(np.abs(y_c)) + 1e-9))
    if not np.isfinite(err) or err > 1e-3:
        return False, f"DirectML output mismatch vs CPU (rel. error {err:.2e})"
    return True, f"verified on adapter {device_id} (small test network, rel. error {err:.1e})"


if __name__ == "__main__":
    rep = collect()
    if "--verify-gpu" in sys.argv:
        ok, msg = verify_directml()
        rep.directml_verified = msg
    print(rep.text())
    if "--json" in sys.argv:
        print(json.dumps(asdict(rep), indent=2, ensure_ascii=False))
