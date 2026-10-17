# Vendored RVC model definitions

These files are copied unmodified (except for package-relative imports) from the
official RVC project:

- Repository: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- Commit: `81eed5e8f68b6bed1789f682fe78cdd324495afc` (2026-08-04)
- Source directory: `infer/module/`
- License: MIT (see `LICENSE` in this directory)

They define the RVC synthesizer network so that trained `.pth` voice models can be
loaded. Do not edit them; streaming/export wrappers live in `../synth.py`.
