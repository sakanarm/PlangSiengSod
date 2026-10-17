# AI voice models (RVC) – concepts and plan

> Status: the AI voice path is an **optional add-on still in development**. The robot voice
> (v1) does not use any model. This page explains what will be needed.

## Terms – these are different things
| Item | What it is | Can it be used directly for conversion? |
|---|---|---|
| **Reference audio** | Your raw recordings (wav/mp3) | **No.** A recording is not a model. |
| **Dataset** | Recordings cleaned, cut and resampled for training | No – input for training |
| **Checkpoints** (`G_*.pth`, `D_*.pth`, 300–900 MB) | Intermediate training state | No – the app rejects them |
| **Inference model** (`<name>.pth`, ~55 MB) | Final small model RVC writes after training | **Yes** |
| **Index** (`added_*.index`) | Optional feature index improving timbre | Yes, together with the model |

## Requirements for a good voice model
* 10–30 minutes of clean speech of **one** speaker you are authorized to use
* no music, no background noise, no reverb, no other voices
* consistent microphone; WAV/FLAC, ideally 40–48 kHz

## Training
Training is not practical on this laptop (no CUDA GPU). It can be done with the official
RVC project (https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) on a PC
with an NVIDIA GPU or on a cloud GPU (e.g. Google Colab). Uploading recordings to a cloud
service is **your decision**; the app never uploads anything.

## Base models used by the AI add-on
Downloaded only by `setup.ps1 -WithAI`, from the official RVC Hugging Face repository,
each verified by SHA-256:
* `hubert_base` (181 MB, required) – speech-content encoder, **not a voice**
* `rmvpe.onnx` (345 MB, optional, `-WithRMVPE`) – pitch tracker, **not a voice**

Model files (.pth) are loaded with safe weights-only loading; files containing code are refused.
