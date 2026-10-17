"""Model-file validation. Runs in the lite install (no torch): tensors are faked with
numpy arrays, which expose the same ``.shape`` the checks use. The one test that
really loads/builds a network is skipped unless the AI add-on is installed."""

import numpy as np
import pytest

from rtvc.voice_conversion.model_files import ModelValidationError, check_checkpoint_dict, validate_model

V2_40K_CONFIG = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5]] * 3,
                 [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000]


def _cpt(version="v2", f0=1, dim=768, config=None, drop=()):
    weight = {
        "enc_p.emb_phone.weight": np.zeros((192, dim)),
        "enc_p.emb_pitch.weight": np.zeros((256, 192)),
        "emb_g.weight": np.zeros((1, 256)),
        "dec.conv_pre.weight": np.zeros((512, 192, 7)),
        "dec.conv_post.weight": np.zeros((1, 32, 7)),
    }
    for k in drop:
        weight.pop(k)
    return {"weight": weight, "config": list(config or V2_40K_CONFIG), "version": version, "f0": f0,
            "sr": "40k", "info": "test"}


def test_valid_v2_checkpoint():
    version, sr, has_f0, n_spk, warnings = check_checkpoint_dict(_cpt())
    assert (version, sr, has_f0, n_spk, warnings) == ("v2", 40000, True, 1, [])


def test_v1_dims_and_string_sample_rate():
    cfg = list(V2_40K_CONFIG)
    cfg[-1] = "40k"
    assert check_checkpoint_dict(_cpt("v1", dim=256, config=cfg))[:2] == ("v1", 40000)


def test_version_feature_dim_mismatch():
    with pytest.raises(ModelValidationError, match="768-dim"):
        check_checkpoint_dict(_cpt("v2", dim=256))


def test_training_checkpoint_rejected():
    with pytest.raises(ModelValidationError, match="TRAINING checkpoint"):
        check_checkpoint_dict({"model": {}, "optimizer": {}, "iteration": 100})


def test_missing_tensor_and_pitch_embedding():
    with pytest.raises(ModelValidationError, match="dec.conv_post"):
        check_checkpoint_dict(_cpt(drop=["dec.conv_post.weight"]))
    with pytest.raises(ModelValidationError, match="pitch embedding"):
        check_checkpoint_dict(_cpt(drop=["enc_p.emb_pitch.weight"]))


def test_no_f0_model_warns():
    assert check_checkpoint_dict(_cpt(f0=0))[4]


def test_bad_config():
    with pytest.raises(ModelValidationError, match="18 hyper-parameters"):
        check_checkpoint_dict(_cpt(config=V2_40K_CONFIG[:10]))
    cfg = list(V2_40K_CONFIG)
    cfg[12] = [10, 8, 2, 2]               # 32k upsampling with a 40k header
    with pytest.raises(ModelValidationError, match="upsample"):
        check_checkpoint_dict(_cpt(config=cfg))
    cfg = list(V2_40K_CONFIG)
    cfg[-1] = 22050
    with pytest.raises(ModelValidationError, match="sample rate"):
        check_checkpoint_dict(_cpt(config=cfg))


def test_unsupported_version_and_non_dict():
    with pytest.raises(ModelValidationError, match="version"):
        check_checkpoint_dict(_cpt(version="v3"))
    with pytest.raises(ModelValidationError, match="expected a dict"):
        check_checkpoint_dict([1, 2, 3])


def test_recording_is_not_a_model(tmp_path):
    wav = tmp_path / "my_voice.wav"
    wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    with pytest.raises(ModelValidationError, match="recording"):
        validate_model(wav)
    disguised = tmp_path / "voice.pth"     # audio renamed to .pth
    disguised.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    with pytest.raises(ModelValidationError, match="WAV"):
        validate_model(disguised)


def test_index_onnx_and_missing_rejected(tmp_path):
    for name, msg in (("added_x.index", "feature index"), ("model.onnx", "ONNX")):
        p = tmp_path / name
        p.write_bytes(b"\x00" * 16)
        with pytest.raises(ModelValidationError, match=msg):
            validate_model(p)
    with pytest.raises(ModelValidationError, match="not found"):
        validate_model(tmp_path / "nope.pth")


def test_real_checkpoint_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from rtvc.voice_conversion.synth import make_random_checkpoint

    p = tmp_path / "test_model.pth"
    torch.save(make_random_checkpoint("v2", 40000), p)
    info = validate_model(p)
    assert info.is_test_model and info.version == "v2" and info.sample_rate == 40000
