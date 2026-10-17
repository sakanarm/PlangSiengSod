"""Audio device enumeration, VB-CABLE detection and routing validation.

Pure helpers (``validate_routing``, ``find_device``) take plain dicts so they can be
unit-tested without any sound hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

# VB-Audio Virtual Cable: apps PLAY into "CABLE Input", apps RECORD from "CABLE Output".
CABLE_PLAYBACK_HINT = "cable input"
CABLE_RECORD_HINT = "cable output"
VIRTUAL_HINTS = ("cable", "vb-audio", "voicemeeter", "virtual")


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    hostapi: str
    max_in: int
    max_out: int
    default_sr: float

    @property
    def label(self) -> str:
        return f"{self.name} [{self.hostapi}]"

    @property
    def is_virtual(self) -> bool:
        n = self.name.lower()
        return any(h in n for h in VIRTUAL_HINTS)

    @property
    def is_cable_playback(self) -> bool:
        return CABLE_PLAYBACK_HINT in self.name.lower() and self.max_out > 0

    @property
    def is_cable_record(self) -> bool:
        return CABLE_RECORD_HINT in self.name.lower() and self.max_in > 0


def list_devices(rescan: bool = False) -> list[Device]:
    import sounddevice as sd

    if rescan:  # PortAudio caches the device list; re-init picks up plug/unplug.
        sd._terminate()
        sd._initialize()
    apis = sd.query_hostapis()
    out = []
    for i, d in enumerate(sd.query_devices()):
        out.append(Device(i, d["name"], apis[d["hostapi"]]["name"], d["max_input_channels"],
                          d["max_output_channels"], d["default_samplerate"]))
    return out


def host_apis() -> list[str]:
    import sounddevice as sd

    return [a["name"] for a in sd.query_hostapis()]


def inputs(devices: list[Device], hostapi: str) -> list[Device]:
    return [d for d in devices if d.hostapi == hostapi and d.max_in > 0]


def outputs(devices: list[Device], hostapi: str) -> list[Device]:
    return [d for d in devices if d.hostapi == hostapi and d.max_out > 0]


def find_device(devices: list[Device], name: str, hostapi: str, kind: str) -> Device | None:
    """Exact name match first, then case-insensitive prefix (WASAPI/MME truncate names)."""
    if not name:
        return None
    pool = inputs(devices, hostapi) if kind == "input" else outputs(devices, hostapi)
    for d in pool:
        if d.name == name:
            return d
    low = name.lower()
    for d in pool:
        if d.name.lower().startswith(low[:31]) or low.startswith(d.name.lower()):
            return d
    return None


def default_device_name(kind: str) -> str:
    import sounddevice as sd

    try:
        idx = sd.default.device[0 if kind == "input" else 1]
        return sd.query_devices(idx)["name"] if idx is not None and idx >= 0 else ""
    except Exception:
        return ""


def suggest_cable_output(devices: list[Device], hostapi: str) -> Device | None:
    return next((d for d in outputs(devices, hostapi) if d.is_cable_playback), None)


def validate_routing(input_name: str, output_name: str, monitor_name: str = "",
                     devices: list[Device] | None = None, hostapi: str = "") -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for a chosen routing. Errors block Start."""
    errors: list[str] = []
    warnings: list[str] = []
    if not input_name:
        errors.append("Choose an input (microphone).")
    if not output_name:
        errors.append("Choose an output (normally 'CABLE Input (VB-Audio Virtual Cable)').")
    in_low, out_low, mon_low = input_name.lower(), output_name.lower(), monitor_name.lower()

    if input_name and CABLE_RECORD_HINT in in_low and CABLE_PLAYBACK_HINT in out_low:
        errors.append("Feedback loop: input is 'CABLE Output' and output is 'CABLE Input' - "
                      "the converted voice would be fed back into itself. Select your real microphone as input.")
    elif input_name and any(h in in_low for h in VIRTUAL_HINTS):
        warnings.append(f"Input '{input_name}' looks like a virtual device, not a microphone.")
    if input_name and ("stereo mix" in in_low or "what u hear" in in_low):
        errors.append("Input is a loopback/Stereo Mix device - this records your speakers and causes feedback.")
    if output_name and CABLE_PLAYBACK_HINT not in out_low:
        warnings.append("Output is not 'CABLE Input'. Teams will only receive the voice if it records "
                        "from the device you are playing into.")
    if monitor_name:
        if monitor_name == output_name:
            errors.append("Monitor device must differ from the main output.")
        elif any(h in mon_low for h in VIRTUAL_HINTS):
            errors.append("Monitor must be headphones/speakers, not a virtual device.")
        else:
            warnings.append("Monitoring through speakers can leak into your microphone (echo). Use headphones.")

    if devices is not None and hostapi:
        if input_name and find_device(devices, input_name, hostapi, "input") is None:
            errors.append(f"Input device '{input_name}' is not connected (or not available in {hostapi}).")
        if output_name and find_device(devices, output_name, hostapi, "output") is None:
            errors.append(f"Output device '{output_name}' is not connected (or not available in {hostapi}).")
        if monitor_name and find_device(devices, monitor_name, hostapi, "output") is None:
            errors.append(f"Monitor device '{monitor_name}' is not connected.")
    return errors, warnings


def cable_installed(devices: list[Device]) -> bool:
    return any(d.is_cable_playback for d in devices) and any(d.is_cable_record for d in devices)
